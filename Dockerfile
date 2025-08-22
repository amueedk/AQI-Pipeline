# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KERAS_BACKEND=tensorflow

WORKDIR /app

# Minimal OS deps incl. build tools for native wheels (e.g., twofish)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# Python deps from repo requirements.txt
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    PIP_DEFAULT_TIMEOUT=180 pip install --no-cache-dir --prefer-binary -r requirements.txt

# App code
COPY . .

# Ensure log directory exists for FileHandler
RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]



