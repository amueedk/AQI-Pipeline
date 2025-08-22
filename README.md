## AQI Forecasting – Project Guide (no Docker)

### What this repo does
- **Direct Multi‑Horizon LSTM (banded)** to forecast PM2.5 and PM10 for the next 72 hours.
- **Data pipelines** to keep historic features and 72‑hour forecasts in Hopsworks Feature Store.
- **Online inference** via Hopsworks Feature Views and a **FastAPI** dashboard.

---

## Key scripts

### Training – `lstm_direct_multi_horizon_v1.py`
- **Model**: Direct multi‑horizon LSTM (no autoregression). Predicts all horizons in one forward pass.
  - Encoder: historical window → context vector
  - Decoder auxiliary per horizon: scaled weather + pollutants + wind direction + positional encoding + repeated `pm(t)`
  - Output: `TimeDistributed(Dense(2))` → `[pm2_5, pm10]` for horizons 1..72
  - Loss: horizon‑weighted (emphasizes 1–24h by default)
- **Banded training (default)**:
  - Short band: 1–12h, `sequence_length=72`, delta targets, stronger short‑horizon weights → saves to `temp/direct_lstm_short/*`
  - Mid/Long band: 1–72h (used for 13–72), `sequence_length=96`, absolute targets → saves to `temp/direct_lstm_midlong/*`
- **Data source**: Tries Hopsworks FG (`HOPSWORKS_CONFIG.feature_group_name`) if `HOPSWORKS_API_KEY` is set; otherwise falls back to a CSV in repo if present.
- **Artifacts saved** (per model name):
  - `<name>.keras`, `<name>_scalers.pkl`, `<name>_config.json`, `<name>_features.json`

Run (local example):
```bash
pip install -r requirements.txt
# optional: export HOPSWORKS_API_KEY=... (to load training data from Feature Store)
python lstm_direct_multi_horizon_v1.py
```

Environment switches:
- `WALK_CV=1` – run walk‑forward CV instead of final training
- `SINGLE_RUN=1` – single unbanded run using default `CONFIG`
- `REGISTER_TO_HOPSWORKS=1` – after training, register/upload artifacts to Hopsworks (uses `model_registry_utils.py`)

Artifacts land under `temp/<model_name>/`.

---

### Online inference – `infer_online_lstm.py`
- Loads Production artifacts for two models by default:
  - `MODEL_SHORT_NAME` (default `direct_lstm_short`)
  - `MODEL_MIDLONG_NAME` (default `direct_lstm_midlong`)
  - Strict banded: both must exist.
- Pulls inputs from Feature Views:
  - `historic_fv` (encoder window, version 1)
  - `forecasts_fv` (72‑hour exogenous, version 1)
- Rebuilds tensors with saved scalers and produces 72× `[pm2_5, pm10]` plus AQI per step.

Run:
```bash
export HOPSWORKS_API_KEY=...  # required
export MODEL_SHORT_NAME=direct_lstm_short
export MODEL_MIDLONG_NAME=direct_lstm_midlong
python infer_online_lstm.py
```

Outputs JSON with arrays: `pm25`, `pm10`, `aqi_pm25`, `aqi_pm10`, `aqi` (length 72).

---

### FastAPI app – `fastapi_app.py`
- Caches the two models at startup and exposes:
  - `GET /` – dashboard (ECharts)
  - `GET /healthz` – health check
  - `GET /current` – current PM and AQI (OpenWeather)
  - `GET /predict` – 72‑hour forecast JSON
- UI shows labels in Pakistan time (PKT), AQI band shading, and 24h min/avg/max cards.
- Local run (no Docker):
```bash
export HOPSWORKS_API_KEY=...
export OPENWEATHER_API_KEY=...
export MODEL_SHORT_NAME=direct_lstm_short
export MODEL_MIDLONG_NAME=direct_lstm_midlong
python fastapi_app.py
# Opens on http://localhost:8080
```

Ports:
- Local `python fastapi_app.py` → port `8080` (see `if __name__ == "__main__"` block).

---

## Data pipelines (overview)
- `automated_hourly_run_updated.py` – populates historic Feature Group consumed by `historic_fv`.
- `automated_forecast_collector.py` – collects 72‑hour OpenWeather weather + pollution and upserts by `time_str` into forecast FG consumed by `forecasts_fv`.
- `build_feature_views.py` – defines/initializes Feature Views and their serving schemas.
- `model_registry_utils.py` – register/download artifacts from Hopsworks Model Registry or Datasets.

### Environment & .env
- **.env is not committed**. Create your own `.env` in the repo root for local runs, e.g.:
  ```dotenv
  HOPSWORKS_API_KEY=YOUR_HOPSWORKS_KEY
  OPENWEATHER_API_KEY=YOUR_OPENWEATHER_KEY
  MODEL_SHORT_NAME=direct_lstm_short
  MODEL_MIDLONG_NAME=direct_lstm_midlong
  ```
- **Required keys**:
  - `HOPSWORKS_API_KEY` – Feature Store access, FV reads, and model artifact download/registration.
  - `OPENWEATHER_API_KEY` – `/current` endpoint and forecast collector.
- Ensure `config.py` has your real Hopsworks project name in `HOPSWORKS_CONFIG['project_name']`.

---

## GitHub Actions workflows
- `.github/workflows/aqi-pipeline.yml`
  - Purpose: update historic features (backfill/hourly style) used by `historic_fv`.
  - Trigger: manual (`workflow_dispatch`) or external (`repository_dispatch`).
  - Secrets required: `HOPSWORKS_API_KEY`, optionally `OPENWEATHER_API_KEY` if fetching external data.

- `.github/workflows/forecast-collector.yml`
  - Purpose: collect 72‑hour weather + pollution forecasts into the forecast Feature Group.
  - Trigger: manual (`workflow_dispatch`) or external (`repository_dispatch`).
  - Secrets required: `HOPSWORKS_API_KEY`, `OPENWEATHER_API_KEY`.

- `.github/workflows/daily_retrain.yml`
  - Purpose: retrain the Direct LSTM (banded) and publish artifacts to Hopsworks.
  - Trigger: manual (`workflow_dispatch`) or external scheduler. Runs `lstm_direct_multi_horizon_v1.py` (can set `REGISTER_TO_HOPSWORKS=1`).
  - Secrets required: `HOPSWORKS_API_KEY`.

All workflows install from `requirements.txt` and run the corresponding script(s). Make sure secrets are configured in your repository settings.

---

## Tips
- If forecasts in app error with length != 72, rerun `automated_forecast_collector.py` to refresh the 72 rows.
- Keep `tensorflow`, `keras`, and `scikit-learn` in sync with `requirements.txt` to avoid deserialization issues.
- Feature Views expected: `historic_fv` and `forecasts_fv` both at version 1 with the columns referenced in code.


