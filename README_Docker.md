## Containerizing the AQI FastAPI app

### Files used
- `Dockerfile`: Builds Python 3.10-slim image, installs `requirements.txt`, sets `KERAS_BACKEND=tensorflow`, creates `/app/temp/production_models` and `/app/logs`, and runs Uvicorn on port 8000.
- `docker-compose.yml`: Defines service `api`, exposes `8000:8000`, loads `.env`, sets `TZ=Asia/Karachi`, and mounts a persistent model cache volume.
- `.dockerignore`: Excludes caches, logs, notebooks, etc., from the build context.
- `.env` (you create): Holds API keys and model names.

### Prerequisites
- Docker Desktop (or Docker Engine with Compose v2)

### Environment variables (.env)
Create a `.env` file next to `docker-compose.yml`:
```dotenv
HOPSWORKS_API_KEY=YOUR_HOPSWORKS_KEY
OPENWEATHER_API_KEY=YOUR_OPENWEATHER_KEY
MODEL_SHORT_NAME=direct_lstm_short
MODEL_MIDLONG_NAME=direct_lstm_midlong
```

### Build and run
First time (build + start):
```bash
docker compose up -d --build
```

Check logs/health:
```bash
docker compose logs -f api
curl http://localhost:8000/healthz
```

Open in browser:
- Dashboard: `http://localhost:8000/`
- Health: `http://localhost:8000/healthz`
- Current AQI/PM JSON: `http://localhost:8000/current`
- Forecast JSON: `http://localhost:8000/predict`

Subsequent runs:
```bash
docker compose up -d
```

Rebuild after code/dependency changes:
```bash
docker compose up -d --build
```

Stop and clean up:
```bash
docker compose down
```

### Notes & troubleshooting
- **Port**: Container listens on 8000. If you prefer `http://localhost:8080`, change compose mapping to `8080:8000`.
- **API keys**: Missing `HOPSWORKS_API_KEY` or `OPENWEATHER_API_KEY` can cause startup errors or 500s. Ensure `.env` is set.
- **Model cache**: Artifacts are cached in a named volume (`model_cache`) to speed subsequent boots. To clear, stop the stack and remove the volume via Docker Desktop or `docker volume rm <volume-name>`.
- **Logs directory**: The image creates `/app/logs` so the app can write its log files. No extra setup needed.
- **GPU warnings**: TensorFlow messages about CUDA/cuDNN/TensorRT are informational when running CPU-only; safe to ignore.


