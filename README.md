# Hermeneia

Hermeneia turns one finished utterance of audio into text. Metagrafo stays the caption app (capture, silence detection, booth, OBS). This process only runs the model.

Two models, one resident at a time:

| `model` | Direction |
|---|---|
| `whisper` | Korean or English speech into English text. It cannot write Korean. |
| `seamless-m4t-v2` | Korean or English speech into English or Korean text. |

A request names the model. If it is not the one already on the GPU, Hermeneia unloads that one and loads the requested one. The first request after a switch waits on the load (tens of seconds). Later requests for the same model do not. Keep one model for a whole service.

`POST /v1/translations` takes a WAV or 16 kHz mono `s16le` chunk (Metagrafo's chunks are the second kind, at most about 12 seconds). The request body is capped at 2 MB, the input rate must be 8–48 kHz, and audio longer than 30 seconds is refused. `GET /health` reports the device the resident model actually loaded on, which is `cpu` if seamless could not reach CUDA. `GET /health` needs no token. `GET /v1/models` and `POST /v1/translations` need `Authorization: Bearer` when `HERMENEIA_TOKEN` is set.

SeamlessM4T v2 weights are CC-BY-NC 4.0.

## Local

[uv](https://docs.astral.sh/uv/) and Python 3.12.

```powershell
uv sync --extra dev
uv run pytest
uv run python -m hermeneia.main
```

The dev process listens on `127.0.0.1:8080` and does not require a token until `HERMENEIA_TOKEN` is set. Loading a real model needs the `whisper` or `seamless` extra and a GPU. Tests use stand-in models and do not download weights.

## Production

Push to `main` builds `ghcr.io/sunggeunkim/hermeneia` and deploys it. The guest, the Cloudflare hostname `hermeneia.stugen.net`, and the GitHub secrets are a one-time setup: `docs/deploy.md`.
