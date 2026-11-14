from __future__ import annotations

import numpy as np
import pytest

from hermeneia import seamless_guard as guard


def _tone(seconds: float, dbfs: float) -> np.ndarray:
    t = np.arange(int(seconds * 16000)) / 16000
    return (np.sin(2 * np.pi * 220 * t) * 10 ** (dbfs / 20) * np.sqrt(2)).astype(np.float32)


def test_digital_silence_has_no_speech() -> None:
    assert guard.speech_seconds(np.zeros(16000, dtype=np.float32)) == 0.0
    assert guard.speech_seconds(np.zeros(0, dtype=np.float32)) == 0.0


def test_loudness_fallback_counts_loud_frames_only() -> None:
    quiet = _tone(1.0, -60)
    loud = _tone(0.5, -20)
    assert guard._loud_seconds(quiet) == 0.0
    assert guard._loud_seconds(np.concatenate([quiet, loud])) == pytest.approx(0.5, abs=0.03)
    # Shorter than one frame still works.
    assert guard._loud_seconds(_tone(0.01, -20)) == pytest.approx(0.01)


def test_generation_limits_scale_with_audio() -> None:
    limits = guard.generation_limits(2.0)
    assert limits["no_repeat_ngram_size"] == 4
    assert limits["repetition_penalty"] == 1.2
    assert limits["max_new_tokens"] == 44
    assert guard.generation_limits(0.25)["max_new_tokens"] == 23


def test_runaway_repeat_is_dropped() -> None:
    looped = "-(인터뷰) " + " ".join(["장히"] * 48)
    assert guard.compression_ratio(looped) > guard.MAX_COMPRESSION_RATIO
    assert guard.finish(looped, target="ko") == ""
    assert guard.finish(", ".join(["리 리"] * 30), target="ko") == ""
    assert guard.finish("hello " * 40, target="en") == ""


def test_normal_output_is_kept() -> None:
    korean = "안하세요, 여러분. 오늘은 요한복음 3장부터 읽겠습니다."
    assert guard.compression_ratio(korean) < guard.MAX_COMPRESSION_RATIO
    assert guard.finish(korean, target="ko") == korean
    english = "good morning everyone today we will read from the gospel of john chapter 3"
    assert guard.finish(english, target="en") == english


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("-(인터뷰) 안하세요, 여러분.", "안하세요, 여러분."),
        ("-(커) 여러분, 안하십니까?", "여러분, 안하십니까?"),
        ("[음악] 오늘은 요한복음을 읽겠습니다.", "오늘은 요한복음을 읽겠습니다."),
        ("- 안하세요. -(기자) 여러분.", "안하세요. 여러분."),
        ("오늘은 (박수) 요한복음을 읽겠습니다.", "오늘은 요한복음을 읽겠습니다."),
    ],
)
def test_korean_tags_and_dashes_are_removed(raw: str, clean: str) -> None:
    assert guard.finish(raw, target="ko") == clean


@pytest.mark.parametrize(
    "stock",
    [
        "-오늘 말씀 여기까지 듣겠습니다.",
        "시청해주셔서 감사합니다.",
        "시청해 주셔서 감사합니다!",
        "MBC 뉴스 박윤수입니다.",
        "KBS 뉴스 김철수 기자입니다.",
    ],
)
def test_stock_sign_offs_are_dropped(stock: str) -> None:
    assert guard.finish(stock, target="ko") == ""
    assert guard.finish(f"안하세요, 여러분. {stock}", target="ko") == "안하세요, 여러분."


def test_english_output_is_not_rewritten() -> None:
    assert guard.finish("-(laughs) thank you for watching.", target="en") == "-(laughs) thank you for watching."
