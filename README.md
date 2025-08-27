# AQI Forecasting System

A real-time Air Quality Index (AQI) forecasting system that predicts PM2.5 and PM10 concentrations up to 72 hours ahead, with automatic AQI calculation and web-based visualization.

## 🌟 Features

- **72-hour AQI forecasts** with hourly granularity
- **Real-time predictions** via FastAPI
- **Interactive web dashboard** with ECharts visualization
- **Two-band forecasting architecture** (short-term + mid/long-term)
- **Automated data collection** from OpenWeather API
- **Production-ready deployment** with Docker
- **Model versioning** via Hopsworks Model Registry
- **Feature store integration** for scalable ML serving

## 🏗️ Architecture

### Forecasting Approach
The system uses a **two-band forecasting strategy**:

1. **Short-term (1-12 hours)**: Direct LSTM model for immediate predictions
2. **Mid/long-term (13-72 hours)**: Separate LSTM model for extended forecasts

### AQI Calculation
AQI is **calculated** using EPA standards:
- **PM2.5 AQI**: Based on PM2.5 concentration
- **PM10 AQI**: Based on PM10 concentration
- **Final AQI**: Maximum of PM2.5 and PM10 AQI values

### Data Flow
```
OpenWeather API → Feature Engineering → Feature Store → ML Models → Predictions → AQI Calculation → Web Dashboard
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Hopsworks account with API key
- OpenWeather API key

### Environment Setup
1. Clone the repository:
```bash
git clone <https://github.com/amueedk/AQI-Pipeline>
cd AQI-Pipeline
```

2. Create environment file:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the dashboard at http://localhost:8000
```

## 📊 FastAPI Endpoints

### `/predict`
Returns 72-hour forecasts for PM2.5, PM10, and calculated AQI values.

**Response:**
```json
{
  "pm25": [12.3, 15.7, ...],
  "pm10": [45.2, 52.1, ...],
  "aqi_pm25": [51, 65, ...],
  "aqi_pm10": [45, 52, ...],
  "aqi": [51, 65, ...]
}
```

### `/current`
Returns current air quality data from OpenWeather API.

**Response:**
```json
{
  "pm2_5": 12.3,
  "pm10": 45.2,
  "aqi": 51,
  "source": "openweather"
}
```

### `/healthz`
Health check endpoint.

## 🧠 Machine Learning Models

### Model Architecture
- **Type**: LSTM (Long Short-Term Memory) neural networks
- **Input**: Historical PM values + weather forecasts + engineered features
- **Output**: Multi-horizon PM2.5 and PM10 predictions
- **Training**: TensorFlow/Keras with custom loss functions

### Feature Engineering
- **Temporal features**: Hour, day, month, day-of-week (sin/cos encoding)
- **Weather features**: Temperature, humidity, pressure, wind speed/direction
- **Pollutant features**: CO, O₃, SO₂, NH₃ concentrations
- **Interaction features**: Cross-feature combinations
- **Change rate features**: PM concentration gradients

### Model Training
```bash
# Train both models (default - 72-hour predictions)
python lstm_direct_multi_horizon_v1.py

```

## 🔧 Configuration

### Environment Variables
```bash
# Required
HOPSWORKS_API_KEY=your_hopsworks_api_key
OPENWEATHER_API_KEY=your_openweather_api_key

# Optional
MODEL_SHORT_NAME=direct_lstm_short
MODEL_MIDLONG_NAME=direct_lstm_midlong
PORT=8000
```

### Hopsworks Configuration
The system integrates with Hopsworks for:
- **Feature Store**: Historical data and forecast features
- **Model Registry**: Model versioning and deployment
- **Feature Views**: Real-time feature serving

## 📈 Data Pipeline

### Automated Collection
```bash
# Run hourly data collection
python automated_hourly_run_updated.py

# Manual historical data collection for data back-fill
python manual_historic_run.py
```

### Feature Engineering
```bash
# Create engineered features
python feature_engineering.py

# Build feature views
python build_feature_views.py
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

The project includes automated CI/CD pipelines for data collection and forecast updates:

#### `aqi-pipeline.yml` - Hourly Data Collection
- **Trigger**: Manual or external API (`repository_dispatch`)
- **Purpose**: Collects hourly air quality and weather data
- **Features**:
  - Runs `automated_hourly_run_updated.py`
  - Updates feature group in Hopsworks
  - 15-minute timeout for comprehensive data processing
  - Automatic log upload on failure
  - Uses Python 3.9 with dependency caching

#### `forecast-collector.yml` - Weather Forecast Collection
- **Trigger**: Manual or external API (`repository_dispatch`)
- **Purpose**: Collects weather forecasts for ML inference
- **Features**:
  - Runs `automated_forecast_collector.py`
  - 10-minute timeout for forecast processing
  - Updates forecast feature groups
  - Automatic artifact upload on failure

### Workflow Usage
```bash
# Manual trigger via GitHub CLI
gh workflow run aqi-pipeline.yml
gh workflow run forecast-collector.yml

# External API trigger
curl -X POST https://api.github.com/repos/{owner}/{repo}/dispatches \
  -H "Authorization: token {token}" \
  -H "Accept: application/vnd.github.v3+json" \
  -d '{"event_type": "hourly-trigger"}'
```

## 🐳 Docker Setup

### Production Deployment
```bash
# Build production image
docker build -t aqi-forecast .

# Run with environment variables
docker run -p 8000:8000 --env-file .env aqi-forecast
```

### Development
```bash
# Development with volume mounting
docker-compose -f docker-compose.dev.yml up
```

## 📊 Dashboard Features

The web dashboard provides:
- **Real-time AQI display** with color-coded bands
- **72-hour forecast charts** for PM2.5, PM10, and AQI
- **24-hour statistics** (min, max, average)
- **Pakistan timezone** display (Asia/Karachi)
- **Responsive design** for mobile and desktop

## 🔍 Model Analysis

### SHAP Analysis
SHAP (SHapley Additive exPlanations) visualizations are available for model interpretability:
- Feature importance analysis
- Individual prediction explanations
- Located in `shap_lstm/` directory

### Model Performance
- **Short-term**: Optimized for 1-12 hour predictions
- **Mid/long-term**: Optimized for 13-72 hour predictions
- **Ensemble approach**: Combines both models for full forecast

## 🛠️ Development

### Project Structure
```
├── fastapi_app.py              # Main web application
├── lstm_direct_multi_horizon_v1.py  # LSTM model training
├── data_collector.py           # OpenWeather data collection
├── feature_engineering.py      # Feature creation
├── hopsworks_integration.py    # Feature store integration
├── automated_hourly_run_updated.py  # Hourly data pipeline
├── automated_forecast_collector.py  # Forecast collection
├── .github/workflows/          # CI/CD pipelines
│   ├── aqi-pipeline.yml        # Hourly data collection
│   └── forecast-collector.yml  # Forecast collection
├── Models (alternate)/         # Alternative ML models
├── data/                       # Historical datasets
├── shap_lstm/                  # Model interpretability
├── static/                     # Web assets
└── logs/                       # Application logs
```

### Adding New Models
1. Create model training script in `Models/`
2. Update `model_registry_utils.py` for deployment
3. Modify `fastapi_app.py` to load new models
4. Update feature engineering if needed

---

**Note**: This system is designed for Multan's air quality monitoring but can be adapted for other regions by updating the OpenWeather coordinates and timezone settings.
