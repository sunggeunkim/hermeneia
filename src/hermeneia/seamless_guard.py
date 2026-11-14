"""Guards for SeamlessM4T v2 output.

Seamless has none of faster-whisper's defenses (no-speech threshold,
compression-ratio check, temperature fallback). On silence or a very short
piece it makes up subtitle-style lines ("MBC 뉴스 …입니다"), loops on a token
("장히 장히 …"), and tags Korean lines with "-(인터뷰) ". Each function here
covers one of those, and none of them needs torch.
"""

from __future__ import annotations

import logging
import re
import zlib

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000

# Below this the request is treated as silent without running the model.
MIN_SPEECH_SECONDS = 0.1
# Fallback gate when Silero is unavailable: frames quieter than this are not speech.
_QUIET_DBFS = -45.0
_FRAME = 480  # 30 ms at 16 kHz

# Same threshold as faster-whisper's default.
MAX_COMPRESSION_RATIO = 2.4

_TOKENS_PER_SECOND = 12
_TOKEN_HEADROOM = 20


def speech_seconds(samples: np.ndarray) -> float:
    """Seconds of speech, from Silero VAD when available (it ships with
    faster-whisper), otherwise from a loudness gate."""
    if samples.size == 0:
        return 0.0
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError:
        return _loud_seconds(samples)
    options = VadOptions(min_silence_duration_ms=100, speech_pad_ms=0)
    spans = get_speech_timestamps(samples, vad_options=options, sampling_rate=_SAMPLE_RATE)
    return sum(span["end"] - span["start"] for span in spans) / _SAMPLE_RATE


def _loud_seconds(samples: np.ndarray) -> float:
    count = samples.size // _FRAME
    if count == 0:
        frames = samples[np.newaxis, :]
    else:
        frames = samples[: count * _FRAME].reshape(count, _FRAME)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    loud = rms > 10 ** (_QUIET_DBFS / 20)
    return float(loud.sum()) * frames.shape[1] / _SAMPLE_RATE


def generation_limits(seconds: float) -> dict[str, float | int]:
    """Keyword arguments for generate() that stop runaway repeats."""
    return {
        "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.2,
        "max_new_tokens": int(seconds * _TOKENS_PER_SECOND) + _TOKEN_HEADROOM,
    }


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


# "(인터뷰)", "[음악]" and the like: a short all-Hangul tag in brackets.
_TAG = re.compile(r"\s*[(\[（［]\s*[가-힣 ]{1,10}\s*[)\]）］]\s*")
_LEADING = re.compile(r"^[\s\-–—~·•]+")
_SENTENCE_END = re.compile(r"(?<=[.!?。])\s+")
_SPACES = re.compile(r"\s{2,}")
_BROADCASTERS = "mbc|kbs|sbs|ytn|jtbc|obs|mbn|tvchosun|채널a"
_STOCK = (
    re.compile(r"^오늘말씀여기까지듣겠습니다$"),
    re.compile(r"^시청해주셔서감사합니다$"),
    re.compile(rf"^({_BROADCASTERS})?뉴스[가-힣]{{2,4}}(기자)?입니다$"),
)
_NOISE = re.compile(r"[\s.,!?。·\-–—~\"'“”‘’]")


def _is_stock(sentence: str) -> bool:
    key = _NOISE.sub("", sentence).lower()
    return any(pattern.match(key) for pattern in _STOCK)


def clean_korean(text: str) -> str:
    """Drop subtitle tags, leading dashes and stock sign-off lines."""
    kept = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = _LEADING.sub("", _SPACES.sub(" ", _TAG.sub(" ", sentence)).strip())
        if sentence and not _is_stock(sentence):
            kept.append(sentence)
    return " ".join(kept)


def finish(text: str, *, target: str) -> str:
    """Post-process one decoded output; "" means nothing trustworthy was said."""
    text = text.strip()
    if compression_ratio(text) > MAX_COMPRESSION_RATIO:
        logger.info("seamless output dropped: repetitive (%r)", text[:80])
        return ""
    if target == "ko":
        cleaned = clean_korean(text)
        if cleaned != text:
            logger.info("seamless output cleaned: %r -> %r", text[:80], cleaned[:80])
        text = cleaned
    return text
