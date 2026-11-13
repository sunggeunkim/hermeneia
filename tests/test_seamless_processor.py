from __future__ import annotations

import io
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import sentencepiece as spm
from transformers import AutoTokenizer, SeamlessM4TFeatureExtractor, SeamlessM4TProcessor

from hermeneia.translate import _seamless_features

_ROOT = Path(__file__).resolve().parents[1]
_KOREAN = "안녕하세요 여러분"


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny checkpoint laid out like facebook/seamless-m4t-v2-large's
    tokenizer: SentencePiece models and no tokenizer.json, so loading it goes
    through the same conversion as production."""
    text = ["hello everyone today we read", _KOREAN + " 오늘은 요한복음을 읽겠습니다"] * 50
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(text),
        model_writer=model,
        vocab_size=60,
        character_coverage=1.0,
        model_type="bpe",
        minloglevel=2,
    )
    root = tmp_path_factory.mktemp("seamless-tokenizer")
    (root / "tokenizer.model").write_bytes(model.getvalue())
    (root / "sentencepiece.bpe.model").write_bytes(model.getvalue())
    (root / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "SeamlessM4TTokenizer"}))
    return root


def _processor(tokenizer_dir: Path) -> SeamlessM4TProcessor:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return SeamlessM4TProcessor(feature_extractor=SeamlessM4TFeatureExtractor(), tokenizer=tokenizer)


def test_real_tokenizer_loads_from_a_sentencepiece_checkpoint(tokenizer_dir: Path) -> None:
    # Without sentencepiece (or protobuf) this raises "`tiktoken` is required",
    # which is how the first Seamless load failed in production.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    assert type(tokenizer).__name__ == "SeamlessM4TTokenizer"
    ids = tokenizer(_KOREAN, add_special_tokens=False)["input_ids"]
    assert tokenizer.unk_token_id not in ids
    assert tokenizer.decode(ids) == _KOREAN


def test_seamless_extra_ships_what_the_tokenizer_needs() -> None:
    # CI installs `dev`, the image installs `seamless`. The test above only
    # protects the image if these stay in both.
    extras = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]

    def names(extra: str) -> set[str]:
        return {spec.split(">")[0].split("=")[0].split("<")[0].strip() for spec in extras[extra]}

    needed = {"transformers", "sentencepiece", "protobuf"}
    assert needed <= names("seamless")
    assert needed <= names("dev")


def test_real_processor_accepts_our_audio_keyword(tokenizer_dir: Path) -> None:
    samples = np.zeros(16000, dtype=np.float32)
    features = _seamless_features(_processor(tokenizer_dir), samples, return_tensors="np")
    assert features["input_features"].shape[0] == 1
    assert features["input_features"].shape[-1] == 160


def test_old_audios_keyword_is_refused(tokenizer_dir: Path) -> None:
    # Guards the reason for the helper: if this starts passing, transformers
    # changed the keyword again.
    with pytest.raises(ValueError, match="audios"):
        _processor(tokenizer_dir)(
            audios=np.zeros(16000, dtype=np.float32), sampling_rate=16000, return_tensors="np"
        )
