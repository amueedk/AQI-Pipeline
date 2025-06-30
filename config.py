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

# Open-Meteo API Configuration
OPEN_METEO_CONFIG = {
    "base_url": "https://api.open-meteo.com/v1",
    "air_quality_url": "https://api.open-meteo.com/v1/air-quality",
    "weather_url": "https://api.open-meteo.com/v1/forecast"
}

# Data Collection Configuration
DATA_CONFIG = {
    "historical_days": 14,  # Days of historical data to collect initially
    "update_frequency_hours": 1,  # How often to collect new data
    "batch_size": 1000
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    "target_column": "us_aqi",  # Target variable for future ML models
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
HOPSWORKS_CONFIG = {
    "api_key": os.getenv("HOPSWORKS_API_KEY"),
    "project_name": "AQIMultan",
    "feature_group_name": "AQI_Mul"
} 