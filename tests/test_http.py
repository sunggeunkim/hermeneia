from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from hermeneia.http import MAX_BODY, _Guard, create_app
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


def _form(blob: bytes) -> tuple[bytes, str]:
    boundary = "hermeneia-test"
    fields = {"model": "whisper", "source_language": "ko", "target_language": "en", "sample_rate": "16000"}
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        for key, value in fields.items()
    ]
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="chunk.pcm"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n".encode()
        + blob
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _chunked(body: bytes):
    step = 64 * 1024
    for start in range(0, len(body), step):
        yield body[start : start + step]


def test_chunked_body_without_length_is_capped() -> None:
    body, content_type = _form(b"\x00" * (2 * 1024 * 1024 + 2))
    with TestClient(_app()) as client:
        response = client.post(
            "/v1/translations",
            headers={"Authorization": "Bearer secret", "Content-Type": content_type},
            content=_chunked(body),
        )
    assert response.status_code == 413


def test_guard_stops_reading_a_chunked_body_at_the_cap() -> None:
    step = 64 * 1024
    pulled = 0
    sent: list[dict] = []
    reached_app = False

    async def receive() -> dict:
        nonlocal pulled
        pulled += step
        return {"type": "http.request", "body": b"\x00" * step, "more_body": True}

    async def send(message: dict) -> None:
        sent.append(message)

    async def app(scope, receive, send) -> None:  # noqa: ANN001
        nonlocal reached_app
        reached_app = True

    scope = {
        "type": "http",
        "path": "/v1/translations",
        "headers": [(b"authorization", b"Bearer secret")],
    }
    asyncio.run(_Guard(app, token="secret")(scope, receive, send))
    assert reached_app is False
    assert sent[0]["status"] == 413
    assert pulled <= MAX_BODY + step


def test_chunked_body_under_the_cap_is_served() -> None:
    body, content_type = _form(b"\x00\x00" * 1600)
    with TestClient(_app()) as client:
        response = client.post(
            "/v1/translations",
            headers={"Authorization": "Bearer secret", "Content-Type": content_type},
            content=_chunked(body),
        )
    assert response.status_code == 200
    assert response.json()["text"] == "hello"


def test_health_stays_open() -> None:
    with TestClient(_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "token" not in body
    assert "text" not in body
