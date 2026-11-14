from __future__ import annotations

import gc
import io
import logging
import subprocess
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hermeneia import seamless_guard

logger = logging.getLogger(__name__)

KNOWN_MODELS = ("whisper", "seamless-m4t-v2")
_LANGUAGES = ("ko", "en")
_SEAMLESS_LANG = {"ko": "kor", "en": "eng"}
_SAMPLE_RATE = 16000
# Input rates outside this band are refused before resampling, so a tiny body
# with an absurd rate cannot ask for a huge output buffer.
_MIN_RATE = 8000
_MAX_RATE = 48000
# Utterances are about 12 s (README). Leave headroom, but bound the work.
_MAX_SECONDS = 30.0
# One forward pass, plus two callers waiting on it. A third waiter is refused.
_ADMIT = 3


class TranslateError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass(frozen=True, slots=True)
class Translation:
    text: str
    model: str
    source_language: str
    target_language: str
    duration_s: float
    inference_ms: int
    load_ms: int
    prompt_applied: bool


@dataclass(frozen=True, slots=True)
class ModelStatus:
    id: str
    loaded: bool


@dataclass(frozen=True, slots=True)
class Status:
    ready: bool
    device: str
    vram_gb: float | None
    resident_model: str | None
    models: tuple[ModelStatus, ...]


class _Engine:
    # The device the weights actually live on, when it can differ from the
    # configured one (seamless falls back to CPU without CUDA).
    device: str | None = None

    def run(
        self,
        samples: np.ndarray,
        *,
        source: str,
        target: str,
        initial_prompt: str,
        condition_on_previous_text: bool,
    ) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return None


Loader = Callable[[], _Engine]


class Translator:
    """One utterance in, text out. One model resident at a time."""

    def __init__(
        self,
        loaders: dict[str, Loader],
        *,
        startup_model: str = "whisper",
        device: str = "cpu",
        vram_gb: float | None = None,
    ) -> None:
        unknown = [name for name in loaders if name not in KNOWN_MODELS]
        if unknown or set(loaders) != set(KNOWN_MODELS):
            raise ValueError(f"loaders must be {', '.join(KNOWN_MODELS)}")
        if startup_model not in KNOWN_MODELS:
            raise ValueError(f"unknown startup model {startup_model}")
        self._loaders = loaders
        self._startup_model = startup_model
        self._device = device
        self._vram_gb = vram_gb
        self._engine: _Engine | None = None
        # Serializes load, unload and inference: one GPU user at a time.
        self._gpu = threading.Lock()
        # Guards what is resident. Held only for reads and writes of these two
        # fields, never across a load or a forward pass, so status() does not
        # wait on the GPU.
        self._state = threading.Lock()
        self._resident: str | None = None
        self._resident_device: str | None = None
        self._admit = threading.Lock()
        self._inflight = 0

    def warmup(self) -> None:
        # The startup load holds the GPU like any request, so it takes a slot.
        try:
            self._enter()
        except TranslateError:
            logger.warning("skipping warmup: requests already queued")
            return
        try:
            with self._gpu:
                self._ensure(self._startup_model)
        except TranslateError:
            logger.exception("startup model failed to load")
        finally:
            self._leave()

    def status(self) -> Status:
        with self._state:
            resident = self._resident
            device = self._resident_device
        return Status(
            ready=resident is not None,
            device=device or self._device,
            vram_gb=self._vram_gb,
            resident_model=resident,
            models=tuple(ModelStatus(id=name, loaded=name == resident) for name in KNOWN_MODELS),
        )

    def translate(
        self,
        audio: bytes,
        *,
        kind: str,
        sample_rate: int | None,
        model: str,
        source_language: str,
        target_language: str,
        initial_prompt: str = "",
        condition_on_previous_text: bool = False,
    ) -> Translation:
        self._check_request(model, source_language, target_language)
        samples = _decode(audio, kind=kind, sample_rate=sample_rate)
        # Only whisper takes a prompt, and only a non-empty one is passed on.
        prompt = initial_prompt if model == "whisper" else ""
        self._enter()
        try:
            with self._gpu:
                load_ms = self._ensure(model)
                started = time.perf_counter()
                assert self._engine is not None
                text = self._engine.run(
                    samples,
                    source=source_language,
                    target=target_language,
                    initial_prompt=prompt,
                    condition_on_previous_text=condition_on_previous_text,
                )
                inference_ms = int((time.perf_counter() - started) * 1000)
        finally:
            self._leave()
        return Translation(
            text=(text or "").strip(),
            model=model,
            source_language=source_language,
            target_language=target_language,
            duration_s=len(samples) / _SAMPLE_RATE,
            inference_ms=inference_ms,
            load_ms=load_ms,
            prompt_applied=bool(prompt),
        )

    def _check_request(self, model: str, source: str, target: str) -> None:
        if model not in KNOWN_MODELS:
            known = ", ".join(KNOWN_MODELS)
            raise TranslateError(404, f"unknown model; known: {known}")
        if source not in _LANGUAGES or target not in _LANGUAGES:
            raise TranslateError(400, "language must be ko or en")
        if model == "whisper" and target != "en":
            raise TranslateError(400, "whisper only translates into English")

    def _enter(self) -> None:
        with self._admit:
            if self._inflight >= _ADMIT:
                raise TranslateError(503, "busy")
            self._inflight += 1

    def _leave(self) -> None:
        with self._admit:
            self._inflight -= 1

    def _ensure(self, model: str) -> int:
        """Caller holds self._gpu."""
        if self._resident == model and self._engine is not None:
            return 0
        started = time.perf_counter()
        if self._engine is not None:
            previous = self._engine
            self._engine = None
            self._set_resident(None, None)
            previous.close()
        try:
            engine = self._loaders[model]()
        except Exception as exc:
            logger.exception("failed to load %s", model)
            _log_nvidia_smi()
            raise TranslateError(503, f"failed to load {model}") from exc
        load_ms = int((time.perf_counter() - started) * 1000)
        self._engine = engine
        self._set_resident(model, getattr(engine, "device", None))
        return load_ms

    def _set_resident(self, model: str | None, device: str | None) -> None:
        with self._state:
            self._resident = model
            self._resident_device = device


def _decode(audio: bytes, *, kind: str, sample_rate: int | None) -> np.ndarray:
    if not audio:
        raise TranslateError(400, "empty audio")
    try:
        if kind == "wav" or audio.startswith(b"RIFF"):
            samples, rate = _decode_wav(audio)
        elif kind == "s16le":
            if sample_rate is None:
                raise TranslateError(400, "sample_rate is required for raw pcm")
            if len(audio) % 2:
                raise TranslateError(400, "pcm length is not a whole number of samples")
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            rate = sample_rate
        else:
            raise TranslateError(400, "audio must be wav or s16le")
    except TranslateError:
        raise
    except Exception as exc:
        raise TranslateError(400, "undecodable audio") from exc
    if samples.size == 0:
        raise TranslateError(400, "empty audio")
    if not _MIN_RATE <= rate <= _MAX_RATE:
        raise TranslateError(400, f"sample rate must be {_MIN_RATE} to {_MAX_RATE} Hz")
    if samples.size / rate > _MAX_SECONDS:
        raise TranslateError(400, f"audio is longer than {_MAX_SECONDS:g} s")
    resampled = _resample(samples, rate)
    if resampled.size == 0:
        raise TranslateError(400, "empty audio")
    return resampled


def _decode_wav(audio: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(audio), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2 or channels <= 0:
        raise TranslateError(400, "wav must be 16-bit")
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, rate


def _resample(samples: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == _SAMPLE_RATE:
        return np.ascontiguousarray(samples, dtype=np.float32)
    count = int(round(len(samples) * _SAMPLE_RATE / source_rate))
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=count, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def _log_nvidia_smi() -> None:
    try:
        report = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            timeout=5,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        logger.info("nvidia-smi unavailable: %s", exc)
        return
    logger.info("nvidia-smi\n%s", report)


def load_whisper(model_name: str, device: str, compute_type: str) -> _Engine:
    _add_cuda_libs()
    from faster_whisper import WhisperModel

    whisper = WhisperModel(model_name, device=device, compute_type=compute_type)

    class _Whisper(_Engine):
        def run(
            self,
            samples: np.ndarray,
            *,
            source: str,
            target: str,
            initial_prompt: str,
            condition_on_previous_text: bool,
        ) -> str:
            task = "transcribe" if source == target else "translate"
            segments, _info = whisper.transcribe(
                samples,
                language=source,
                task=task,
                beam_size=1,
                vad_filter=False,
                without_timestamps=True,
                initial_prompt=initial_prompt or None,
                condition_on_previous_text=condition_on_previous_text,
            )
            return "".join(segment.text for segment in segments).strip()

        def close(self) -> None:
            nonlocal whisper
            whisper = None
            gc.collect()

    engine = _Whisper()
    engine.device = device
    return engine


def _seamless_features(processor: object, samples: np.ndarray, *, return_tensors: str = "pt"):
    # transformers 5 takes `audio`; the old `audios` keyword raises ValueError.
    return processor(audio=samples, sampling_rate=_SAMPLE_RATE, return_tensors=return_tensors)  # type: ignore[operator]


def load_seamless(device: str, num_beams: int = 5) -> _Engine:
    """Greedy decoding (num_beams=1) often stops after the first sentence of
    Korean output; beam search keeps the rest for a few hundred ms."""
    if num_beams < 1:
        raise ValueError("num_beams must be at least 1")
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    use_cuda = device == "cuda" and torch.cuda.is_available()
    torch_device = "cuda" if use_cuda else "cpu"
    if device == "cuda" and not use_cuda:
        logger.warning("CUDA unavailable to torch; seamless is loading on CPU")
    dtype = torch.float16 if use_cuda else torch.float32
    checkpoint = "facebook/seamless-m4t-v2-large"
    processor = AutoProcessor.from_pretrained(checkpoint)
    weights = SeamlessM4Tv2ForSpeechToText.from_pretrained(checkpoint, torch_dtype=dtype)
    weights.to(torch_device)
    weights.eval()

    class _Seamless(_Engine):
        def run(
            self,
            samples: np.ndarray,
            *,
            source: str,
            target: str,
            initial_prompt: str,
            condition_on_previous_text: bool,
        ) -> str:
            del source, initial_prompt, condition_on_previous_text
            seconds = samples.size / _SAMPLE_RATE
            speech = seamless_guard.speech_seconds(samples)
            if speech < seamless_guard.MIN_SPEECH_SECONDS:
                logger.info("seamless skipped: %.2f s of speech in %.2f s", speech, seconds)
                return ""
            features = _seamless_features(processor, samples)
            features = {key: value.to(torch_device) for key, value in features.items()}
            with torch.inference_mode():
                tokens = weights.generate(
                    **features,
                    tgt_lang=_SEAMLESS_LANG[target],
                    num_beams=num_beams,
                    **seamless_guard.generation_limits(seconds),
                )
            text = processor.decode(tokens[0], skip_special_tokens=True)
            return seamless_guard.finish(text, target=target)

        def close(self) -> None:
            nonlocal weights
            try:
                weights.to("cpu")
            except Exception:
                logger.exception("seamless move to cpu failed")
            weights = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    engine = _Seamless()
    engine.device = torch_device
    return engine


def _add_cuda_libs() -> None:
    """Windows: let CTranslate2 find the pip CUDA DLLs.

    On Linux the dynamic linker reads LD_LIBRARY_PATH only at process start, so
    setting it from here would do nothing. The image sets it (deploy/Dockerfile).
    """
    import os
    import sys
    from pathlib import Path

    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in nvidia.__path__:
        for name in ("cublas", "cudnn", "cuda_runtime"):
            bin_dir = Path(root) / name / "bin"
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
