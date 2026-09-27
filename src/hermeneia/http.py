from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from hermeneia.translate import TranslateError, Translator

MAX_BODY = 2 * 1024 * 1024


def create_app(translator: Translator, *, token: str | None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        import threading

        threading.Thread(target=translator.warmup, name="hermeneia-warmup", daemon=True).start()
        yield

    app = FastAPI(title="hermeneia", lifespan=lifespan)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path.startswith("/v1/"):
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    if int(length) > MAX_BODY:
                        return JSONResponse({"detail": "body too large"}, status_code=413)
                except ValueError:
                    return JSONResponse({"detail": "bad content-length"}, status_code=400)
            if token is not None and request.headers.get("authorization") != f"Bearer {token}":
                return JSONResponse({"detail": "missing or invalid token"}, status_code=401)
        return await call_next(request)

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
        payload = audio.file.read(MAX_BODY + 1)
        if len(payload) > MAX_BODY:
            return JSONResponse({"detail": "body too large"}, status_code=413)
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
