# Multi-Stage Production Dockerfile for Insurance LLM Mining, Hybrid RAG & Agentic API
# Supports both GPU (CUDA 12.x / vLLM / Unsloth QLoRA) and lightweight CPU inference profiles

FROM python:3.12-slim AS builder

WORKDIR /build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn pydantic numpy pandas scikit-learn httpx

FROM python:3.12-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Create non-root execution user for Azure AKS security compliance
RUN groupadd -r llmuser && useradd -r -g llmuser llmuser

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/
COPY deploy/ ./deploy/

RUN chown -R llmuser:llmuser /app
USER llmuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
