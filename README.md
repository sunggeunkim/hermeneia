# Hermeneia

Hermeneia turns one finished utterance of audio into text. Metagrafo stays the caption app (capture, silence detection, booth, OBS). This process only runs the model.

Two models, one resident at a time:

| `model` | Direction |
|---|---|
| `whisper` | Korean or English speech into English text. It cannot write Korean. |
| `seamless-m4t-v2` | Korean or English speech into English or Korean text. |

A request names the model. If it is not the one already on the GPU, Hermeneia unloads that one and loads the requested one. The first request after a switch waits on the load (tens of seconds). Later requests for the same model do not. Keep one model for a whole service.

`POST /v1/translations` takes a WAV or 16 kHz mono `s16le` chunk (Metagrafo's chunks are the second kind, at most about 12 seconds). The request body is capped at 2 MB, the input rate must be 8–48 kHz, and audio longer than 30 seconds is refused. `GET /health` reports the device the resident model actually loaded on, which is `cpu` if seamless could not reach CUDA. `GET /health` needs no token. `GET /v1/models` and `POST /v1/translations` need `Authorization: Bearer` when `HERMENEIA_TOKEN` is set.

Seamless decodes with beam search, 5 beams by default (`SEAMLESS_NUM_BEAMS`). With greedy decoding (`1`) it often stops after the first sentence of Korean output. Its text vocabulary also lacks some Korean syllables, for example 녕, so "안녕하세요" comes out as "안하세요"; no setting fixes that.

Seamless has no equivalent of Whisper's no-speech or compression-ratio checks, so on silence or a very short piece it makes up text ("MBC 뉴스 …입니다"), repeats one token until the length limit, or prefixes Korean lines with "-(인터뷰) ". The server guards against this ([seamless_guard.py](src/hermeneia/seamless_guard.py)):

- Audio with under 0.1 s of speech returns `""` without running the model. Speech is measured with the Silero VAD that ships with faster-whisper.
- Generation uses `no_repeat_ngram_size=4`, `repetition_penalty=1.2`, and at most 12 tokens per second of audio plus 20.
- Output whose gzip compression ratio is above 2.4 is dropped, the same threshold Whisper uses.
- Korean output loses short bracketed tags such as "(인터뷰)" and leading dashes. Stock subtitle sign-offs are removed: "오늘 말씀 여기까지 듣겠습니다", "시청해주셔서 감사합니다", "<broadcaster> 뉴스 <name>입니다". A speaker who actually says one of these is removed too.

Pieces under about 2 s still translate less reliably, so callers should join short pieces before sending them.

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
