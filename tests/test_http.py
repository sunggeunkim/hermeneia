from __future__ import annotations

from fastapi.testclient import TestClient

from hermeneia.http import create_app
from hermeneia.translate import KNOWN_MODELS, Translator


class _Engine:
    def run(self, samples, **kwargs):  # noqa: ANN001
        return "hello"

    def close(self) -> None:
        return None


def _app(token: str | None = "secret"):
    loaders = {name: _Engine for name in KNOWN_MODELS}
    translator = Translator(loaders, device="cpu", vram_gb=None)
    return create_app(translator, token=token)


def test_missing_token_is_rejected() -> None:
    with TestClient(_app()) as client:
        response = client.post(
            "/v1/translations",
            data={"model": "whisper", "source_language": "ko", "target_language": "en", "sample_rate": "16000"},
            files={"audio": ("chunk.pcm", b"\x00\x00" * 160, "application/octet-stream")},
        )
    assert response.status_code == 401
    assert "token" in response.json()["detail"]


def test_oversized_body_is_rejected() -> None:
    blob = b"x" * (2 * 1024 * 1024 + 1)
    with TestClient(_app()) as client:
        response = client.post(
            "/v1/translations",
            headers={"Authorization": "Bearer secret"},
            data={"model": "whisper", "source_language": "ko", "target_language": "en"},
            files={"audio": ("chunk.pcm", blob, "application/octet-stream")},
        )
    assert response.status_code == 413


def test_health_stays_open() -> None:
    with TestClient(_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "token" not in body
    assert "text" not in body
