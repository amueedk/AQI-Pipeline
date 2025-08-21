# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KERAS_BACKEND=tensorflow

WORKDIR /app

# OS deps for TF/sklearn/kafka; curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libstdc++6 libssl3 libsasl2-2 tzdata curl \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Writable cache for models
RUN mkdir -p /app/temp/production_models /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


