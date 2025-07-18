"""
Configuration file for AQI Data Collection & Feature Engineering
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Multan City Configuration
MULTAN_CONFIG = {
    "latitude": 30.1575,
    "longitude": 71.5249,
    "timezone": "auto",
    "city_name": "Multan"
}

# Data Collection Configuration
DATA_CONFIG = {
    "historical_days": 14,  # Days of historical data to collect initially
    "update_frequency_hours": 1,  # How often to collect new data
    "batch_size": 1000
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    "target_columns": ["pm2_5", "pm10"],  # Target variables for future ML models (raw concentrations)
    "primary_target": "pm2_5",  # Primary target for feature engineering
    "lag_hours": [1, 2, 3, 6, 12, 24, 48, 72],  # Lag features to create
    "rolling_windows": [3, 6, 12, 24]  # Rolling statistics windows
}

# US AQI Categories and Thresholds
AQI_CATEGORIES = {
    "Good": {"min": 0, "max": 50, "color": "#009966"},
    "Moderate": {"min": 51, "max": 100, "color": "#ffde33"},
    "Unhealthy for Sensitive Groups": {"min": 101, "max": 150, "color": "#ff9933"},
    "Unhealthy": {"min": 151, "max": 200, "color": "#cc0033"},
    "Very Unhealthy": {"min": 201, "max": 300, "color": "#660099"},
    "Hazardous": {"min": 301, "max": 500, "color": "#7e0023"}
}

# File Paths
PATHS = {
    "data_dir": "data",
    "logs_dir": "logs",
    "temp_dir": "temp"
}

# Hopsworks Feature Store Configuration
# To set up Hopsworks:
# 1. Go to https://www.hopsworks.ai/ and create account
# 2. Create a project (e.g., "AQI-Pipeline")
# 3. Go to Settings -> API Keys -> Create API Key
# 4. Add the API key to GitHub Secrets as HOPSWORKS_API_KEY
# 5. IMPORTANT: Update project_name below to match your actual Hopsworks project name
HOPSWORKS_CONFIG = {
    "api_key": os.getenv("HOPSWORKS_API_KEY", ""),  # Will be empty if not set
    "project_name": "AQIMultan",  # Your actual Hopsworks project name
    "feature_group_name": "multan_aqi_features_clean"  # Feature group name
}

# IQAir API Configuration
IQAIR_CONFIG = {
    "api_key": os.getenv("IQAIR_API_KEY", ""),
    "base_url": "https://api.airvisual.com/v2",
    "city": "Multan",
    "state": "Punjab",
    "country": "Pakistan"
}

# OpenWeather API Configuration
OPENWEATHER_CONFIG = {
    "api_key": os.getenv("OPENWEATHER_API_KEY", ""),
    "base_url": "https://pro.openweathermap.org/data/2.5",  # Pro endpoint for current data
    "lat": 30.1575,
    "lon": 71.5249,
    "city": "Multan",
    "state": "Punjab",
    "country": "Pakistan"
}



# Add OpenWeather historic weather endpoint 
OPENWEATHER_HISTORY_WEATHER_URL = "http://history.openweathermap.org/data/2.5/history/city" 