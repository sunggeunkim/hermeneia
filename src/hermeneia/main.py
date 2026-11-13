from __future__ import annotations

import logging
import os
import subprocess

import uvicorn

from hermeneia.http import create_app
from hermeneia.translate import Translator, load_seamless, load_whisper


def detect_vram_gb() -> float | None:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return float(raw.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def build_translator() -> Translator:
    whisper_model = os.environ.get("WHISPER_MODEL", "large-v3")
    compute = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
    device = os.environ.get("HERMENEIA_DEVICE", "cuda")
    startup = os.environ.get("HERMENEIA_MODEL", "whisper")
    seamless_beams = _positive_int("SEAMLESS_NUM_BEAMS", 5)
    return Translator(
        {
            "whisper": lambda: load_whisper(whisper_model, device, compute),
            "seamless-m4t-v2": lambda: load_seamless(device, num_beams=seamless_beams),
        },
        startup_model=startup,
        device=device,
        vram_gb=detect_vram_gb(),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    token = os.environ.get("HERMENEIA_TOKEN") or None
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(create_app(build_translator(), token=token), host=host, port=port)


if __name__ == "__main__":
    main()
