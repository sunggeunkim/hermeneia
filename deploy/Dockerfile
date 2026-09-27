FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --extra whisper --extra seamless --no-dev

ENV HOST=0.0.0.0 \
    PORT=8080 \
    HF_HOME=/models/hf \
    HUGGINGFACE_HUB_CACHE=/models/hf
EXPOSE 8080
CMD ["uv", "run", "python", "-m", "hermeneia.main"]
