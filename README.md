# 🌤️ Multan AQI Data Collection & Feature Engineering

**Serverless AQI Prediction System for Multan with Automated GitHub Actions Pipeline**

A complete serverless system that collects air quality and weather data from Open-Meteo API, engineers features, and stores them in Hopsworks for ML training - all automated with GitHub Actions.

## 📋 System Overview

This system provides:
- ✅ **Automated Data Collection**: Hourly AQI and weather data from Open-Meteo API
- ✅ **Feature Engineering**: 150+ ML-ready features with time-series patterns
- ✅ **GitHub Actions Automation**: Runs every hour automatically
- ✅ **Hopsworks Integration**: Professional feature store for ML training
- ✅ **Local Storage**: CSV backup with master datasets
- ✅ **Real-time Monitoring**: Continuous data collection and feature updates

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
- GitHub account (for automation)
- Hopsworks account (for feature store)

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

3. **Configure Hopsworks** (optional)
   - Create account at [Hopsworks.ai](https://www.hopsworks.ai/)
   - Create project named "AQIMultan"
   - Get API key from Settings → API Keys
   - Add to GitHub Secrets as `HOPSWORKS_API_KEY`

4. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

5. **Enable GitHub Actions**
   - Go to Actions tab in your repository
   - Workflow will run automatically every hour

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

### Hopsworks Settings
- **Project Name**: AQIMultan
- **Feature Group**: multan_aqi_features
- **Version**: 1

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

### Time Features (20+ features)
- **Basic**: Hour, day, month, year, day of week
- **Cyclical**: Sin/cos encoding for periodic features
- **Seasonal**: Spring, summer, autumn, winter indicators
- **Temporal**: Day/night, rush hour indicators

### Lag Features (30+ features)
- **Historical AQI**: 1h, 6h, 12h, 24h, 48h, 72h lags
- **Rolling Statistics**: Mean, std, min, max over 3h, 6h, 12h, 24h windows
- **Change Rates**: 1h, 6h, 24h percentage changes

### Weather Features (25+ features)
- **Temperature**: Squared, cubed, change rates, extremes
- **Humidity**: Squared, high/low indicators
- **Wind**: Squared, high/calm indicators
- **Pressure**: Change rates, high/low indicators
- **Precipitation**: Rain indicators, intensity levels

### Pollutant Features (20+ features)
- **PM2.5/PM10**: Squared, change rates, high/low indicators
- **NO2/O3/CO/SO2**: Squared, change rates, high indicators
- **Ratios**: PM2.5/PM10, NO2/O3 interactions

### Interaction Features (10+ features)
- **Temperature-Humidity**: Interaction terms
- **Temperature-Wind**: Interaction terms
- **Weather-Pollutants**: Cross-feature interactions

## 🤖 GitHub Actions Automation

### Workflow Schedule
- **Runs every hour** at minute 0 (1:00, 2:00, 3:00, etc.)
- **Manual trigger** available via "Run workflow" button

### Workflow Steps
1. ✅ **Checkout code** - Downloads latest code
2. ✅ **Set up Python 3.9** - Installs Python environment
3. ✅ **Install dependencies** - Installs all required packages
4. ✅ **Create directories** - Sets up data/logs folders
5. ✅ **Run data collection** - Fetches latest AQI/weather data
6. ✅ **Run feature engineering** - Creates 150+ ML features
7. ✅ **Push to Hopsworks** - Stores features in feature store
8. ✅ **Upload artifacts** - Saves data and logs for download

### Artifacts
- **Data artifacts**: CSV files with raw and engineered data
- **Log artifacts**: Detailed execution logs
- **Retention**: 7 days (configurable)

## 🎯 Usage

### Manual Execution

**Collect data and engineer features:**
```bash
python data_collector.py
python feature_engineering.py
```

**Push features to Hopsworks:**
```bash
python hopsworks_integration.py
```

### Automated Execution

**GitHub Actions runs automatically every hour:**
- No manual intervention required
- Data collection and feature engineering
- Hopsworks integration
- Artifact upload

## 📈 Expected Output

### Local Files
1. **Master Dataset**: `data/master_dataset.csv`
   - Combined raw data from all collections
   - ~500+ records (historical + current)
   - ~50+ columns (weather + air quality)

2. **Master Features**: `data/master_features.csv`
   - Engineered features ready for ML
   - Same number of records
   - ~150+ columns (original + engineered features)

### Hopsworks Feature Store
- **Feature Group**: `multan_aqi_features`
- **Version**: 1
- **Primary Key**: `time_key` (string format: YYYYMMDDHH)
- **Event Time**: `time` (timestamp)
- **Online Enabled**: Yes (for real-time serving)

## 🔒 Privacy & Security

- Repository can be kept **private**
- No sensitive data in the code
- API keys stored as GitHub Secrets
- Data files excluded from repository via `.gitignore`

## 📊 Monitoring & Logs

### GitHub Actions Logs
- View in Actions tab of your repository
- Download logs as artifacts
- Monitor workflow success/failure

### Local Logs
- `logs/data_collector.log` - Data collection details
- `logs/feature_engineering.log` - Feature engineering process

## 🚀 Next Steps

### Ready for ML Training
Your features are now stored in Hopsworks and ready for:
- **Model Training**: Use Hopsworks ML pipeline
- **Feature Serving**: Real-time feature serving
- **Model Deployment**: Deploy trained models
- **Monitoring**: Track model performance

### Potential Enhancements
- **Real-time Alerts**: AQI threshold notifications
- **Dashboard**: Interactive visualization
- **Model Training**: Automated ML pipeline
- **Predictions**: AQI forecasting models

## 📝 License

This project is open source and available under the MIT License.

---

**🎉 Your AQI pipeline is now fully automated and ready for ML!** 