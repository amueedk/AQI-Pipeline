# 🌤️ Multan AQI Data Collection & Feature Engineering

**US AQI Data Collection and Feature Engineering System for Multan**

A serverless AQI prediction system that collects air quality and weather data from Open-Meteo API, engineers features, and can be automated with GitHub Actions.

## 📋 System Overview

This system focuses on:
- ✅ **Data Collection**: Fetch US AQI and weather data from Open-Meteo API
- ✅ **Feature Engineering**: Create comprehensive features for ML models
- ✅ **GitHub Actions Automation**: Hourly data collection and feature engineering
- ✅ **Local Storage**: CSV storage with master datasets
- ✅ **Hopsworks Integration**: Feature store integration (optional)

## 🇺🇸 US AQI Scale

| Category | Range | Color | Description |
|----------|-------|-------|-------------|
| Good | 0-50 | #009966 | Air quality is good |
| Moderate | 51-100 | #ffde33 | Air quality is moderate |
| Unhealthy for Sensitive Groups | 101-150 | #ff9933 | Sensitive groups should limit outdoor activities |
| Unhealthy | 151-200 | #cc0033 | Everyone should limit outdoor activities |
| Very Unhealthy | 201-300 | #660099 | Health warnings of emergency conditions |
| Hazardous | 301-500 | #7e0023 | Health alert: everyone may experience serious effects |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/amueedk/AQI-Pipeline.git
   cd AQI-Pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run data collection and feature engineering**
   ```bash
   python data_collector.py
   python feature_engineering.py
   ```

## 📁 Project Structure

```
AQI-Pipeline/
├── config.py                 # Configuration settings
├── data_collector.py         # Data collection from Open-Meteo API
├── feature_engineering.py    # Feature engineering pipeline
├── hopsworks_integration.py  # Hopsworks feature store integration
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore rules
├── .github/                  # GitHub Actions workflows
│   └── workflows/
│       └── aqi-pipeline.yml  # Automated pipeline workflow
├── data/                     # Data storage (generated)
├── logs/                     # Application logs (generated)
└── temp/                     # Temporary files (generated)
```

## 🔧 Configuration

### Multan City Settings
- **Latitude**: 30.1575° N
- **Longitude**: 71.5249° E
- **Timezone**: Asia/Karachi

### Data Collection
- **Source**: Open-Meteo API (free, no API key required)
- **Data Frequency**: Hourly
- **Historical Data**: 14 days
- **Current Data**: Last 6 hours
- **Target Variable**: US AQI

## 📊 Data Collection

### Air Quality Data
- **US AQI**: Primary target variable
- **PM2.5 & PM10**: Particulate matter
- **NO2**: Nitrogen dioxide
- **O3**: Ozone
- **CO**: Carbon monoxide
- **SO2**: Sulphur dioxide

### Weather Data
- **Temperature**: 2m temperature
- **Humidity**: Relative humidity
- **Wind**: Speed and direction
- **Pressure**: Atmospheric pressure
- **Precipitation**: Rain and snow
- **Cloud Cover**: Cloud coverage
- **UV Index**: Ultraviolet radiation

## 🔧 Feature Engineering

### Time Features
- **Basic**: Hour, day, month, year, day of week
- **Cyclical**: Sin/cos encoding for periodic features
- **Seasonal**: Spring, summer, autumn, winter indicators
- **Temporal**: Day/night, rush hour indicators

### Lag Features
- **Historical AQI**: 1h, 6h, 12h, 24h, 48h, 72h lags
- **Rolling Statistics**: Mean, std, min, max over 3h, 6h, 12h, 24h windows
- **Change Rates**: 1h, 6h, 24h percentage changes

### Weather Features
- **Temperature**: Squared, cubed, change rates, extremes
- **Humidity**: Squared, high/low indicators
- **Wind**: Squared, high/calm indicators
- **Pressure**: Change rates, high/low indicators
- **Precipitation**: Rain indicators, intensity levels

### Pollutant Features
- **PM2.5/PM10**: Squared, change rates, high/low indicators
- **NO2/O3/CO/SO2**: Squared, change rates, high indicators
- **Ratios**: PM2.5/PM10, NO2/O3 interactions

### Interaction Features
- **Temperature-Humidity**: Interaction terms
- **Temperature-Wind**: Interaction terms
- **Weather-Pollutants**: Cross-feature interactions

## 🎯 Usage

### Manual Execution

**Collect data and engineer features:**
```bash
python data_collector.py
python feature_engineering.py
```

**Push features to Hopsworks (optional):**
```bash
python hopsworks_integration.py
```

### GitHub Actions Automation

The repository includes a GitHub Actions workflow that:
- Runs every hour automatically
- Collects new data
- Engineers features
- Pushes to Hopsworks (if configured)
- Saves artifacts

**To enable automation:**
1. Push your code to GitHub
2. Go to Actions tab in your repository
3. The workflow will run automatically every hour

## 📈 Expected Output

After running the system, you'll get:

1. **Master Dataset**: `data/master_dataset.csv`
   - Combined raw data from all collections
   - ~500+ records (historical + current)
   - ~50+ columns (weather + air quality)

2. **Master Features**: `data/master_features.csv`
   - Engineered features ready for ML
   - Same number of records
   - ~150+ columns (original + engineered features)

## 🤖 Automation Setup

### GitHub Actions (Recommended)
The repository includes a workflow that runs every hour:
- Fetches latest data
- Engineers features
- Pushes to Hopsworks (optional)
- Saves artifacts

### Manual Scheduling
For local automation, you can use:
- **Windows Task Scheduler**
- **Linux/Mac Cron jobs**
- **Cloud functions** (AWS Lambda, Google Cloud Functions)

## 🔒 Privacy & Security

- Repository can be kept **private**
- No sensitive data in the code
- API keys (if needed) stored as GitHub Secrets
- Data files excluded from repository via `.gitignore`

## 📝 License

This project is open source and available under the MIT License. 