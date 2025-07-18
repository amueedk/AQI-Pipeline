"""
Data Collector for AQI Data Collection & Feature Engineering
Fetches air quality and weather data from OpenWeather API for Multan
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from functools import wraps
import random
import json
import os
from config import DATA_CONFIG, PATHS, IQAIR_CONFIG, OPENWEATHER_CONFIG, OPENWEATHER_HISTORY_WEATHER_URL
import datetime

logger = None

# Ensure logger is always initialized
if logger is None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(PATHS['logs_dir'], 'data_collector.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

COMMON_POLLUTANTS = [
    "pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "us_aqi"
]
COMMON_WEATHER = [
    "temperature", "humidity", "pressure", "wind_speed", "wind_direction"
]

# US EPA AQI breakpoints for each pollutant
AQI_BREAKPOINTS = {
    "pm2_5": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ],
    "pm10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    "o3_8h": [
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300),
    ],
    "o3_1h": [
        (125, 164, 101, 150),
        (165, 204, 151, 200),
        (205, 404, 201, 300),
        (405, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    "co": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500),
    ],
    "so2": [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805, 1004, 401, 500),
    ],
    "no2": [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ],
}

def calc_aqi(conc, breakpoints):
    """Calculate AQI for a given concentration using US EPA breakpoints"""
    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= conc <= C_high:
            return round((I_high - I_low) / (C_high - C_low) * (conc - C_low) + I_low)
    return None

def compute_pm2_5_aqi(row):
    """Compute PM2.5 AQI only (since PM2.5 is typically the dominant pollutant)"""
    if not pd.isna(row.get("pm2_5")):
        return calc_aqi(row["pm2_5"], AQI_BREAKPOINTS["pm2_5"])
    return None

def retry_on_network_error(max_retries=3, delay=300, backoff_factor=1.5, jitter=True):
    """
    Retry decorator for network operations with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Initial delay in seconds (default: 300 = 5 minutes)
        backoff_factor: Multiplier for delay on each retry (default: 1.5)
        jitter: Add random jitter to prevent thundering herd (default: True)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, 
                       requests.exceptions.Timeout,
                       requests.exceptions.ConnectionError,
                       requests.exceptions.HTTPError) as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"❌ {func.__name__} failed after {max_retries} retries. Final error: {e}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    current_delay = delay * (backoff_factor ** attempt)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        current_delay *= (0.5 + random.random())
                    
                    logger.warning(f"⚠️ {func.__name__} attempt {attempt + 1} failed: {e}")
                    logger.info(f"🔄 Retrying in {current_delay:.1f} seconds...")
                    time.sleep(current_delay)
                except Exception as e:
                    # Non-network errors should not be retried
                    logger.error(f"❌ {func.__name__} failed with non-network error: {e}")
                    raise e
            
            # This should never be reached, but just in case
            raise last_exception
        return wrapper
    return decorator

class IQAirDataCollector:
    def __init__(self):
        self.api_key = IQAIR_CONFIG["api_key"]
        self.base_url = IQAIR_CONFIG["base_url"]
        self.city = IQAIR_CONFIG["city"]
        self.state = IQAIR_CONFIG["state"]
        self.country = IQAIR_CONFIG["country"]

    @retry_on_network_error()
    def fetch_current_aqi(self) -> pd.DataFrame:
        """
        Fetches the latest AQI (us_aqi) for the configured city from IQAir.
        Returns a DataFrame with a single row: index is time, column is 'iqair_aqi'.
        """
        url = f"{self.base_url}/city"
        params = {
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "key": self.api_key
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data["status"] != "success":
                logger.warning(f"IQAir API returned non-success: {data.get('status')}")
                return pd.DataFrame()
            pollution = data["data"]["current"]["pollution"]
            row = {
                "time": pd.to_datetime(pollution["ts"]),
                "iqair_aqi": pollution.get("aqius")
            }
            df = pd.DataFrame([row])
            df.set_index("time", inplace=True)
            return df
        except Exception as e:
            logger.warning(f"IQAir API error: {e}")
            return pd.DataFrame()

@retry_on_network_error()
def fetch_historic_weather(start_unix, end_unix):
    """
    Fetch hourly weather data from OpenWeather historic endpoint for Multan.
    Returns a DataFrame indexed by UTC timestamp.
    """
    url = f"{OPENWEATHER_HISTORY_WEATHER_URL}?lat={OPENWEATHER_CONFIG['lat']}&lon={OPENWEATHER_CONFIG['lon']}&type=hour&start={start_unix}&end={end_unix}&appid={OPENWEATHER_CONFIG['api_key']}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for entry in data.get("list", []):
        dt = pd.to_datetime(entry["dt"], unit="s", utc=True)
        main = entry.get("main", {})
        wind = entry.get("wind", {})
        row = {
            "time": dt,
            "temperature": main.get("temp"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("deg"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.set_index("time", inplace=True)
    return df

class OpenWeatherDataCollector:
    def __init__(self):
        self.api_key = OPENWEATHER_CONFIG["api_key"]
        self.base_url = OPENWEATHER_CONFIG["base_url"]
        self.lat = OPENWEATHER_CONFIG["lat"]
        self.lon = OPENWEATHER_CONFIG["lon"]

    @retry_on_network_error()
    def fetch_air_pollution(self, start_unix=None, end_unix=None):
        if start_unix and end_unix:
            url = f"{self.base_url}/air_pollution/history?lat={self.lat}&lon={self.lon}&start={start_unix}&end={end_unix}&appid={self.api_key}"
        else:
            url = f"{self.base_url}/air_pollution?lat={self.lat}&lon={self.lon}&appid={self.api_key}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for entry in data["list"]:
            row = {"time": pd.to_datetime(entry["dt"], unit="s", utc=True)}
            for k, v in entry["components"].items():
                row[k] = v
            # Extract OpenWeather AQI (1-5 scale)
            if "main" in entry and "aqi" in entry["main"]:
                row["openweather_aqi"] = entry["main"]["aqi"]
            else:
                row["openweather_aqi"] = None
            rows.append(row)
        df = pd.DataFrame(rows)
        df.set_index("time", inplace=True)
        # Calculate PM2.5 AQI
        df["pm2_5_aqi"] = df.apply(compute_pm2_5_aqi, axis=1)
        # Calculate PM10 AQI
        df["pm10_aqi"] = df.apply(lambda row: calc_aqi(row.get('pm10'), AQI_BREAKPOINTS['pm10']) if pd.notna(row.get('pm10')) else None, axis=1)
        # Calculate US AQI as maximum of PM2.5 and PM10 AQI
        df["us_aqi"] = df.apply(lambda row: max(row['pm2_5_aqi'], row['pm10_aqi']) if pd.notna(row.get('pm2_5_aqi')) and pd.notna(row.get('pm10_aqi')) else None, axis=1)
        return df

    @retry_on_network_error()
    def fetch_weather(self, dt_unix=None):
        url = f"{self.base_url}/weather?lat={self.lat}&lon={self.lon}&appid={self.api_key}&units=metric"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        row = {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"]["deg"],
        }
        return row

    def collect_historical_data(self, days_back=14):
        end_time = int(time.time())
        start_time = end_time - days_back * 24 * 3600
        df_pollution = self.fetch_air_pollution(start_unix=start_time, end_unix=end_time)
        df_weather = fetch_historic_weather(start_unix=start_time, end_unix=end_time)
        # Merge on timestamp (outer join to keep all pollution records)
        df_merged = df_pollution.join(df_weather, how="outer")
        df_merged["city"] = OPENWEATHER_CONFIG["city"]
        df_merged["latitude"] = self.lat
        df_merged["longitude"] = self.lon
        return df_merged

    def collect_current_data(self):
        """
        Collect current air pollution and weather data (most recent hour).
        Returns a DataFrame with current data from OpenWeather APIs.
        """
        df_pollution = self.fetch_air_pollution()
        weather = self.fetch_weather()
        for k, v in weather.items():
            df_pollution[k] = v
        df_pollution["city"] = OPENWEATHER_CONFIG["city"]
        df_pollution["latitude"] = self.lat
        df_pollution["longitude"] = self.lon
        # Note: CSV saving is handled in collect_current_data_with_iqair()
        return df_pollution

@retry_on_network_error()
def fetch_iqair_aqi():
    url = f"https://api.airvisual.com/v2/nearest_city?lat={OPENWEATHER_CONFIG['lat']}&lon={OPENWEATHER_CONFIG['lon']}&key={IQAIR_CONFIG['api_key']}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data["status"] == "success":
            return data["data"]["current"]["pollution"]["aqius"]
    except Exception as e:
        logger.warning(f"IQAir API error: {e}")
    return None

@retry_on_network_error(max_retries=2, delay=180)  # 2 retries, start with 3 minutes
def collect_current_data_with_iqair():
    df = OpenWeatherDataCollector().collect_current_data()
    iqair_aqi = fetch_iqair_aqi()
    if iqair_aqi is not None:
        df["iqair_aqi"] = iqair_aqi
        # Compute absolute deviation
        if "us_aqi" in df.columns:
            df["abs_deviation"] = abs(df["us_aqi"] - iqair_aqi)
    else:
        df["iqair_aqi"] = None
        df["abs_deviation"] = None
    
    # Save only validation data (much smaller files)
    # Reset index to make 'time' a column again
    df_reset = df.reset_index()
    validation_cols = ['time', 'us_aqi', 'iqair_aqi', 'abs_deviation']
    # Only include columns that exist
    available_cols = [col for col in validation_cols if col in df_reset.columns]
    logger.info(f"DEBUG: Available columns in DataFrame: {list(df_reset.columns)}")
    logger.info(f"DEBUG: Validation columns requested: {validation_cols}")
    logger.info(f"DEBUG: Validation columns available: {available_cols}")
    validation_df = df_reset[available_cols].copy()
    
    # Save to monthly validation file
    monthly = datetime.datetime.utcnow().strftime('%Y%m')
    validation_path = f"data/pm2_5_validation_monthly_{monthly}.csv"
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(validation_path):
        # Append to existing monthly validation file
        try:
            existing_df = pd.read_csv(validation_path, index_col=0, parse_dates=True)
            combined_df = pd.concat([existing_df, validation_df])
            # Remove duplicates based on timestamp (keep the latest)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            # Sort by timestamp
            combined_df = combined_df.sort_index()
        except Exception as e:
            logger.warning(f"Error reading existing validation file: {e}. Creating new file.")
            combined_df = validation_df
    else:
        combined_df = validation_df
    
    combined_df.to_csv(validation_path)
    logger.info(f"Validation data appended to monthly file: {validation_path} (total {len(combined_df)} records)")
    return df

def main():
    """
    Main function to test data collection.
    Can be used for a one-off historical data collection.
    """
    collector = OpenWeatherDataCollector()
    
    # --- For Historical Data Collection ---
    logger.info("--- Starting Historical Data Collection ---")
    historical_df = collector.collect_historical_data(days_back=14)
    if not historical_df.empty:
        logger.info(f"Successfully collected {len(historical_df)} historical records.")
    else:
        logger.error("Historical data collection failed.")

    # --- For Current Data Collection ---
    logger.info("\n--- Starting Current Data Collection ---")
    current_df = collector.collect_current_data()
    if not current_df.empty:
        logger.info(f"Successfully collected {len(current_df)} current records.")
    else:
        logger.warning("No new current data collected.")


if __name__ == "__main__":
    main() 