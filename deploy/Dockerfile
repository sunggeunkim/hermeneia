FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --extra whisper --extra seamless --no-dev \
    && .venv/bin/python -c "import hermeneia.main"

# CTranslate2 dlopens cuBLAS and cuDNN through the dynamic linker, which reads
# LD_LIBRARY_PATH only when the process starts. Point it at the CUDA 12 wheels.
ENV HOST=0.0.0.0 \
    PORT=8080 \
    HF_HOME=/models/hf \
    HUGGINGFACE_HUB_CACHE=/models/hf \
    LD_LIBRARY_PATH=/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
EXPOSE 8080
CMD ["uv", "run", "--frozen", "python", "-m", "hermeneia.main"]
