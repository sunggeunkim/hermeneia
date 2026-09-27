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

logger = logging.getLogger(__name__)

KNOWN_MODELS = ("whisper", "seamless-m4t-v2")
_LANGUAGES = ("ko", "en")
_SEAMLESS_LANG = {"ko": "kor", "en": "eng"}
_SAMPLE_RATE = 16000
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
        self._resident: str | None = None
        self._gpu = threading.Lock()
        self._admit = threading.Lock()
        self._inflight = 0

    def warmup(self) -> None:
        try:
            with self._gpu:
                self._ensure(self._startup_model)
        except TranslateError:
            logger.exception("startup model failed to load")

    def status(self) -> Status:
        with self._gpu:
            resident = self._resident
        return Status(
            ready=resident is not None,
            device=self._device,
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
                    initial_prompt=initial_prompt,
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
            prompt_applied=model == "whisper",
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
        if self._resident == model and self._engine is not None:
            return 0
        started = time.perf_counter()
        if self._engine is not None:
            previous = self._engine
            self._engine = None
            self._resident = None
            previous.close()
        try:
            self._engine = self._loaders[model]()
        except Exception as exc:
            self._engine = None
            self._resident = None
            logger.exception("failed to load %s", model)
            _log_nvidia_smi()
            raise TranslateError(503, f"failed to load {model}") from exc
        self._resident = model
        return int((time.perf_counter() - started) * 1000)


def _decode(audio: bytes, *, kind: str, sample_rate: int | None) -> np.ndarray:
    if not audio:
        raise TranslateError(400, "empty audio")
    try:
        if kind == "wav" or audio.startswith(b"RIFF"):
            samples, rate = _decode_wav(audio)
        elif kind == "s16le":
            if sample_rate is None or sample_rate <= 0:
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
    if width != 2 or rate <= 0 or channels <= 0:
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

    return _Whisper()


def load_seamless(device: str) -> _Engine:
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    use_cuda = device == "cuda" and torch.cuda.is_available()
    torch_device = "cuda" if use_cuda else "cpu"
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
            features = processor(audios=samples, sampling_rate=_SAMPLE_RATE, return_tensors="pt")
            features = {key: value.to(torch_device) for key, value in features.items()}
            with torch.inference_mode():
                tokens = weights.generate(**features, tgt_lang=_SEAMLESS_LANG[target])
            return processor.decode(tokens[0], skip_special_tokens=True).strip()

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

    return _Seamless()


def _add_cuda_libs() -> None:
    """Linux containers find CTranslate2's CUDA libs via LD_LIBRARY_PATH."""
    import os
    import sys
    from pathlib import Path

    roots: list[Path] = []
    try:
        import nvidia

        roots.append(Path(nvidia.__path__[0]))
    except Exception:
        return
    lib_dirs = [path for root in roots for path in root.glob("*/lib") if path.is_dir()]
    if not lib_dirs or not hasattr(os, "environ"):
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    extra = os.pathsep.join(str(path) for path in lib_dirs)
    if extra not in current:
        os.environ["LD_LIBRARY_PATH"] = extra + (os.pathsep + current if current else "")
    # Windows loader path, same idea as Metagrafo.
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        for root in roots:
            for bin_dir in root.glob("*/bin"):
                if bin_dir.is_dir():
                    os.add_dll_directory(str(bin_dir))
