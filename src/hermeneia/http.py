from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from hermeneia.translate import TranslateError, Translator

MAX_BODY = 2 * 1024 * 1024


class _Guard:
    """Bearer auth and the body cap for /v1/.

    The cap counts the bytes actually received, not the Content-Length header,
    so a chunked upload cannot get past it. A capped body is at most 2 MB, so
    it is buffered here and replayed to the app before any form parsing.
    """

    def __init__(self, app: ASGIApp, *, token: str | None) -> None:
        self._app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            await self._app(scope, receive, send)
            return
        headers = {key.decode("latin-1"): value.decode("latin-1") for key, value in scope["headers"]}
        length = headers.get("content-length")
        if length is not None:
            try:
                if int(length) > MAX_BODY:
                    await _reply(send, 413, "body too large")
                    return
            except ValueError:
                await _reply(send, 400, "bad content-length")
                return
        if self._token is not None and headers.get("authorization") != f"Bearer {self._token}":
            await _reply(send, 401, "missing or invalid token")
            return

        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            size += len(body)
            if size > MAX_BODY:
                await _reply(send, 413, "body too large")
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self._app(scope, replay, send)


async def _reply(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def create_app(translator: Translator, *, token: str | None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        import threading

        threading.Thread(target=translator.warmup, name="hermeneia-warmup", daemon=True).start()
        yield

    app = FastAPI(title="hermeneia", lifespan=lifespan)
    app.add_middleware(_Guard, token=token)

    @app.get("/health")
    def health() -> dict[str, object]:
        status = translator.status()
        return {
            "ok": True,
            "ready": status.ready,
            "device": status.device,
            "vram_gb": status.vram_gb,
            "model": status.resident_model,
        }

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        status = translator.status()
        return {
            "models": [{"id": item.id, "loaded": item.loaded} for item in status.models],
        }

    @app.post("/v1/translations")
    def translations(
        audio: UploadFile,
        model: str = Form(),
        source_language: str = Form(),
        target_language: str = Form(),
        sample_rate: int | None = Form(default=None),
        encoding: str = Form(default="s16le"),
        initial_prompt: str = Form(default=""),
        condition_on_previous_text: bool = Form(default=False),
    ) -> dict[str, object]:
        # _Guard already capped the whole request at MAX_BODY.
        payload = audio.file.read()
        kind = "wav" if encoding == "wav" else encoding
        try:
            result = translator.translate(
                payload,
                kind=kind,
                sample_rate=sample_rate,
                model=model,
                source_language=source_language,
                target_language=target_language,
                initial_prompt=initial_prompt,
                condition_on_previous_text=condition_on_previous_text,
            )
        except TranslateError as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status)
        return asdict(result)

    return app
