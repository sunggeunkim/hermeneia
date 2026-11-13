from __future__ import annotations

import numpy as np
import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast, SeamlessM4TFeatureExtractor, SeamlessM4TProcessor

from hermeneia.translate import _seamless_features


def _processor() -> SeamlessM4TProcessor:
    # The real processor and feature extractor. The tokenizer is a stand-in:
    # it is not used for audio input, and the real one needs a download.
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
    )
    return SeamlessM4TProcessor(feature_extractor=SeamlessM4TFeatureExtractor(), tokenizer=tokenizer)


def test_real_processor_accepts_our_audio_keyword() -> None:
    samples = np.zeros(16000, dtype=np.float32)
    features = _seamless_features(_processor(), samples, return_tensors="np")
    assert features["input_features"].shape[0] == 1
    assert features["input_features"].shape[-1] == 160


def test_old_audios_keyword_is_refused() -> None:
    # Guards the reason for the helper: if this starts passing, transformers
    # changed the keyword again.
    with pytest.raises(ValueError, match="audios"):
        _processor()(audios=np.zeros(16000, dtype=np.float32), sampling_rate=16000, return_tensors="np")
