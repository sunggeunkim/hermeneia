from __future__ import annotations

import pytest

from hermeneia.main import _positive_int


def test_positive_int_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEAMLESS_NUM_BEAMS", raising=False)
    assert _positive_int("SEAMLESS_NUM_BEAMS", 5) == 5
    monkeypatch.setenv("SEAMLESS_NUM_BEAMS", " ")
    assert _positive_int("SEAMLESS_NUM_BEAMS", 5) == 5


def test_positive_int_reads_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEAMLESS_NUM_BEAMS", "1")
    assert _positive_int("SEAMLESS_NUM_BEAMS", 5) == 1


@pytest.mark.parametrize("raw", ["0", "-2", "five", "2.5"])
def test_positive_int_refuses_bad_values(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("SEAMLESS_NUM_BEAMS", raw)
    with pytest.raises(ValueError, match="SEAMLESS_NUM_BEAMS"):
        _positive_int("SEAMLESS_NUM_BEAMS", 5)
