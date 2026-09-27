from __future__ import annotations

import io
import threading
import wave

import numpy as np
import pytest

from hermeneia.translate import KNOWN_MODELS, TranslateError, Translator


class _Engine:
    def __init__(self, name: str, log: list[str]) -> None:
        self._name = name
        self._log = log
        log.append(f"open {name}")

    def run(self, samples: np.ndarray, **kwargs: object) -> str:
        self._log.append(f"run {self._name}")
        return self._name

    def close(self) -> None:
        self._log.append(f"close {self._name}")


def _loaders(log: list[str] | None = None) -> dict:
    record = log if log is not None else []

    def build(name: str):
        def loader() -> _Engine:
            return _Engine(name, record)

        return loader

    return {name: build(name) for name in KNOWN_MODELS}


def _translator(**kwargs: object) -> Translator:
    loaders = kwargs.pop("loaders", None)
    return Translator(loaders or _loaders(), device="cpu", vram_gb=12.0, **kwargs)  # type: ignore[arg-type]


def _pcm(n: int = 1600) -> bytes:
    return b"\x00\x00" * n


def _wav(*, rate: int = 16000, channels: int = 1, frames: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x01" * (frames * channels))
    return buf.getvalue()


def _translate(translator: Translator, **overrides: object):
    payload = {
        "audio": _pcm(),
        "kind": "s16le",
        "sample_rate": 16000,
        "model": "whisper",
        "source_language": "ko",
        "target_language": "en",
    }
    payload.update(overrides)
    return translator.translate(**payload)  # type: ignore[arg-type]


def test_startup_is_not_ready_until_warmup() -> None:
    translator = _translator()
    assert translator.status().ready is False
    assert translator.status().resident_model is None
    translator.warmup()
    status = translator.status()
    assert status.ready is True
    assert status.resident_model == "whisper"
    assert [item.loaded for item in status.models] == [True, False]


def test_whisper_uses_prompt_and_seamless_does_not() -> None:
    translator = _translator()
    korean = _translate(translator, initial_prompt="Jesus")
    assert korean.text == "whisper"
    assert korean.prompt_applied is True
    assert korean.load_ms >= 0
    english = _translate(
        translator,
        model="seamless-m4t-v2",
        source_language="en",
        target_language="ko",
        initial_prompt="Jesus",
    )
    assert english.text == "seamless-m4t-v2"
    assert english.prompt_applied is False
    assert english.target_language == "ko"


def test_switch_unloads_the_previous_engine_before_the_next_opens() -> None:
    log: list[str] = []
    translator = _translator(loaders=_loaders(log))
    first = _translate(translator)
    second = _translate(translator, model="seamless-m4t-v2", source_language="en", target_language="en")
    third = _translate(translator, model="seamless-m4t-v2", source_language="en", target_language="en")
    assert first.load_ms >= 0
    assert second.load_ms >= 0
    assert third.load_ms == 0
    assert log.index("close whisper") < log.index("open seamless-m4t-v2")
    assert log.count("open whisper") == 1
    assert log.count("open seamless-m4t-v2") == 1


def test_whisper_refuses_korean_target() -> None:
    translator = _translator()
    with pytest.raises(TranslateError) as caught:
        _translate(translator, target_language="ko")
    assert caught.value.status == 400


def test_unknown_model_names_the_known_ids() -> None:
    translator = _translator()
    with pytest.raises(TranslateError) as caught:
        _translate(translator, model="nllb")
    assert caught.value.status == 404
    assert "whisper" in caught.value.detail
    assert "seamless-m4t-v2" in caught.value.detail


def test_empty_and_odd_pcm_are_rejected() -> None:
    translator = _translator()
    with pytest.raises(TranslateError) as empty:
        _translate(translator, audio=b"")
    assert empty.value.status == 400
    with pytest.raises(TranslateError) as odd:
        _translate(translator, audio=b"\x00")
    assert odd.value.status == 400


def test_wav_duration_and_resample() -> None:
    translator = _translator()
    mono = _translate(translator, audio=_wav(), kind="wav", sample_rate=None)
    assert mono.duration_s == pytest.approx(1.0)
    stereo = _translate(translator, audio=_wav(channels=2, frames=8000), kind="wav")
    assert stereo.text == "whisper"
    slow = _translate(translator, audio=_wav(rate=8000, frames=8000), kind="wav")
    assert slow.duration_s == pytest.approx(1.0)


def test_failed_load_leaves_no_resident_model() -> None:
    loaders = _loaders()

    def boom() -> _Engine:
        raise RuntimeError("out of memory")

    loaders["seamless-m4t-v2"] = boom
    translator = _translator(loaders=loaders)
    _translate(translator)
    assert translator.status().resident_model == "whisper"
    with pytest.raises(TranslateError) as caught:
        _translate(
            translator,
            model="seamless-m4t-v2",
            source_language="en",
            target_language="en",
        )
    assert caught.value.status == 503
    assert translator.status().resident_model is None
    assert translator.status().ready is False


def test_third_waiter_is_refused() -> None:
    release = threading.Event()
    entered = threading.Event()

    class _Blocked(_Engine):
        def __init__(self) -> None:
            pass

        def run(self, samples: np.ndarray, **kwargs: object) -> str:
            entered.set()
            assert release.wait(timeout=5)
            return "ok"

        def close(self) -> None:
            return None

    def blocked() -> _Blocked:
        return _Blocked()

    loaders = _loaders()
    loaders["whisper"] = blocked
    translator = _translator(loaders=loaders)
    codes: list[int] = []
    lock = threading.Lock()

    def call() -> None:
        try:
            _translate(translator)
            code = 200
        except TranslateError as exc:
            code = exc.status
        with lock:
            codes.append(code)

    first = threading.Thread(target=call)
    first.start()
    assert entered.wait(timeout=2)
    others = [threading.Thread(target=call) for _ in range(3)]
    for thread in others:
        thread.start()
    deadline = threading.Event()
    for _ in range(200):
        with lock:
            if 503 in codes:
                break
        deadline.wait(0.01)
    with lock:
        assert 503 in codes
    release.set()
    first.join(timeout=2)
    for thread in others:
        thread.join(timeout=2)
    with lock:
        assert codes.count(503) == 1
        assert codes.count(200) == 3
