"""
Automated Forecast Collector for Seq2Seq LSTM
--------------------------------------------
This script collects 72-hour weather forecasts from OpenWeather API
and stores them in a dedicated feature group for Seq2Seq inference.

Runs every hour to ensure fresh forecasts are always available.
"""
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from typing import Dict, List, Optional, Tuple
from functools import wraps
import random

# Import existing helpers
from config import HOPSWORKS_CONFIG, OPENWEATHER_CONFIG, PATHS
from hopsworks_integration import HopsworksUploader, retry_on_hopsworks_error
from data_collector import retry_on_network_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/forecast_collector.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Forecast collection configuration
FORECAST_CONFIG = {
    "forecast_hours": 72,  # Collect 72-hour forecasts
    "update_frequency_hours": 1,  # Update every hour
    "feature_group_name": "aqi_weather_forecasts",  # Dedicated forecast feature group
    "description": "72-hour weather + pollution forecasts for LSTM direct/banded inference"
}

@retry_on_network_error(max_retries=3, delay=300)
def fetch_openweather_forecast(hours: int = 72) -> pd.DataFrame:
    """
    Fetch weather forecast from OpenWeather API for next N hours
    
    Args:
        hours: Number of hours to forecast (default: 72)
    
    Returns:
        DataFrame with hourly forecast data
    """
    logger.info(f"🌤️ Fetching {hours}-hour weather forecast from OpenWeather...")
    
    # OpenWeather Pro API hourly forecast endpoint
    base_url = f"{OPENWEATHER_CONFIG['base_url']}/forecast/hourly"
    
    # Parameters for Multan (Pro API hourly forecast)
    params = {
        'lat': OPENWEATHER_CONFIG['lat'],
        'lon': OPENWEATHER_CONFIG['lon'],
        'appid': OPENWEATHER_CONFIG['api_key'],
        'units': 'metric',  # Celsius, m/s, hPa
        'cnt': hours  # Pro API hourly: up to 96 hours (4 days), we need 72
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"✅ Successfully fetched hourly forecast data (Pro API)")
        logger.info(f"   Location: {data['city']['name']}, {data['city']['country']}")
        logger.info(f"   Forecast periods: {len(data['list'])} (hourly intervals)")
        logger.info(f"   Forecast duration: {len(data['list'])} hours")
        
        # Process forecast data
        forecast_records = []
        current_time = datetime.utcnow()
        
        for i, forecast in enumerate(data['list']):
            # Calculate step hour (1, 2, 3, ..., 72)
            step_hour = i + 1
            
            # Forecast timestamp
            forecast_time = pd.to_datetime(forecast['dt'], unit='s', utc=True)
            
            # Weather data
            main = forecast.get('main', {})
            wind = forecast.get('wind', {})
            
            # Create record
            record = {
                'step_hour': step_hour,  # Forecast horizon (1, 2, 3, ..., 72)
                'target_time': forecast_time,  # API's dt (forecasted hour)
                'temperature': main.get('temp'),
                'humidity': main.get('humidity'),
                'pressure': main.get('pressure'),
                'wind_speed': wind.get('speed'),
                'wind_direction': wind.get('deg'),
                'city': OPENWEATHER_CONFIG['city'],
                'latitude': OPENWEATHER_CONFIG['lat'],
                'longitude': OPENWEATHER_CONFIG['lon']
            }
            
            forecast_records.append(record)
        
        # Create DataFrame
        forecast_df = pd.DataFrame(forecast_records)
        
        # Add time features for target_time (exactly as decoder expects)
        forecast_df = add_time_features(forecast_df)
        
        # Add wind direction engineering (exactly as decoder expects)
        forecast_df = add_wind_direction_features(forecast_df)
        
        logger.info(f"✅ Processed {len(forecast_df)} forecast records")
        logger.info(f"   Date range: {forecast_df['target_time'].min()} to {forecast_df['target_time'].max()}")
        logger.info(f"   Temperature range: {forecast_df['temperature'].min():.1f}°C to {forecast_df['temperature'].max():.1f}°C")
        
        return forecast_df
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch forecast data: {e}")
        raise e

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time features for target_time (exactly as decoder expects)
    """
    df = df.copy()
    
    # Convert target_time to datetime if needed
    if not isinstance(df['target_time'], pd.DatetimeIndex):
        df['target_time'] = pd.to_datetime(df['target_time'])
    
    # Hour features (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df['target_time'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['target_time'].dt.hour / 24)
    
    # Day of year features (1-365)
    df['day_sin'] = np.sin(2 * np.pi * df['target_time'].dt.dayofyear / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['target_time'].dt.dayofyear / 365)
    
    # Month features (1-12)
    df['month_sin'] = np.sin(2 * np.pi * df['target_time'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['target_time'].dt.month / 12)
    
    # Day of week features (0-6, Monday=0)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['target_time'].dt.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['target_time'].dt.dayofweek / 7)
    
    logger.info("✅ Added cyclical time features for target_time")
    return df

def add_wind_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add wind direction engineering features (exactly as decoder expects)
    """
    df = df.copy()
    
    if 'wind_direction' in df.columns:
        # Convert wind direction to radians and calculate sin/cos
        wind_dir_rad = np.radians(df['wind_direction'])
        df['wind_direction_sin'] = np.sin(wind_dir_rad)
        df['wind_direction_cos'] = np.cos(wind_dir_rad)
        
        # Fix floating-point precision issues
        df['wind_direction_sin'] = df['wind_direction_sin'].round(10)
        df['wind_direction_cos'] = df['wind_direction_cos'].round(10)
        
        logger.info("✅ Added wind direction engineering features")
    else:
        logger.warning("⚠️ No wind_direction column found, skipping wind direction features")
    
    return df

@retry_on_network_error(max_retries=3, delay=300)
def fetch_openweather_pollution_forecast(hours: int = 72) -> pd.DataFrame:
    """
    Fetch air pollution forecast (hourly) for the next N hours using OpenWeather Air Pollution API.
    We keep only the components needed by the LSTM: CO, O3, SO2, NH3.
    """
    logger.info(f"🧪 Fetching {hours}-hour pollution forecast from OpenWeather...")

    base_url = f"{OPENWEATHER_CONFIG['base_url']}/air_pollution/forecast"
    params = {
        'lat': OPENWEATHER_CONFIG['lat'],
        'lon': OPENWEATHER_CONFIG['lon'],
        'appid': OPENWEATHER_CONFIG['api_key'],
    }
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        lst = data.get('list', [])
        if not lst:
            raise ValueError('Empty pollution forecast list')

        records = []
        current_time = datetime.utcnow()
        for i, item in enumerate(lst[:hours]):
            comp = item.get('components', {})
            rec = {
                'step_hour': i + 1,
                'target_time': pd.to_datetime(item.get('dt'), unit='s', utc=True),
                'carbon_monoxide': comp.get('co'),
                'ozone': comp.get('o3'),
                'sulphur_dioxide': comp.get('so2'),
                'nh3': comp.get('nh3'),
            }
            records.append(rec)
        pol_df = pd.DataFrame(records)
        logger.info(f"✅ Processed {len(pol_df)} pollution forecast records")
        return pol_df
    except Exception as e:
        logger.error(f"❌ Failed to fetch pollution forecast: {e}")
        raise

def format_forecast_for_decoder(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format forecast data for inference using raw feature names only.
    Suitable for direct LSTM aux and for seq2seq decoders that map columns in code.
    """
    logger.info("🔄 Formatting forecast data (raw feature names)...")
    df = forecast_df.copy()

    raw_keep = [
        'temperature', 'humidity', 'pressure', 'wind_speed',
        'wind_direction_sin', 'wind_direction_cos',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
        'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3',
    ]
    base_columns = ['step_hour', 'target_time']
    existing = [c for c in raw_keep if c in df.columns]
    cols = base_columns + existing
    result_df = df[cols].copy()

    logger.info("✅ Formatted forecast data (raw)")
    logger.info(f"   Kept features: {len(existing)}")
    logger.info(f"   Records: {len(result_df)}")
    return result_df

@retry_on_hopsworks_error()
def create_forecast_feature_group(uploader: HopsworksUploader) -> bool:
    """
    Create the forecast feature group if it doesn't exist
    """
    try:
        logger.info(f"🏗️ Creating forecast feature group: {FORECAST_CONFIG['feature_group_name']}")
        
        fg = uploader.fs.get_or_create_feature_group(
            name=FORECAST_CONFIG['feature_group_name'],
            version=1,
            description=FORECAST_CONFIG['description'],
            primary_key=['time_str'],  # Use time_str as the sole primary key
            event_time='time',
            online_enabled=True  # Enable online storage for fast inference
        )
        
        logger.info(f"✅ Forecast feature group created/verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create forecast feature group: {e}")
        return False

@retry_on_hopsworks_error()
def push_forecast_data(uploader: HopsworksUploader, forecast_df: pd.DataFrame) -> bool:
    """
    Push forecast data to Hopsworks feature group
    """
    try:
        logger.info(f"📤 Pushing {len(forecast_df)} forecast records to Hopsworks...")
        
        # Ensure proper data types
        forecast_df['target_time'] = pd.to_datetime(forecast_df['target_time'])
        forecast_df['step_hour'] = forecast_df['step_hour'].astype(int)
        
        # Set time to API's dt (the forecasted hour)
        forecast_df['time'] = forecast_df['target_time']
        
        # Primary key is time_str -> floor to hour and format
        forecast_df['time_str'] = forecast_df['time'].dt.floor('H').dt.strftime('%Y-%m-%d %H:%M:%S')

        # Drop forecast_time if present
        if 'forecast_time' in forecast_df.columns:
            forecast_df = forecast_df.drop(columns=['forecast_time'])
        
        # Push to feature group
        success = uploader.push_features(
            df=forecast_df,
            group_name=FORECAST_CONFIG['feature_group_name'],
            description=FORECAST_CONFIG['description']
        )
        
        if success:
            logger.info(f"✅ Successfully pushed forecast data to Hopsworks")
            logger.info(f"   Records: {len(forecast_df)}")
            logger.info(f"   Steps: {forecast_df['step_hour'].min()} to {forecast_df['step_hour'].max()}")
            return True
        else:
            logger.error(f"❌ Failed to push forecast data")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error pushing forecast data: {e}")
        return False

def run_forecast_collection():
    """
    Main function to collect and store weather forecasts
    """
    logger.info("🚀 Starting forecast collection...")
    logger.info("=" * 60)
    
    # Check API keys
    openweather_key = os.getenv("OPENWEATHER_API_KEY")
    hopsworks_key = os.getenv("HOPSWORKS_API_KEY")
    
    if not openweather_key:
        logger.error("❌ OPENWEATHER_API_KEY not set")
        return False
    
    if not hopsworks_key:
        logger.error("❌ HOPSWORKS_API_KEY not set")
        return False
    
    # 1. Connect to Hopsworks
    logger.info("STEP 1: Connecting to Hopsworks...")
    uploader = HopsworksUploader(
        api_key=hopsworks_key,
        project_name=HOPSWORKS_CONFIG['project_name']
    )
    
    if not uploader.connect():
        logger.error("❌ Failed to connect to Hopsworks")
        return False
    
    # 2. Create forecast feature group
    logger.info("STEP 2: Creating forecast feature group...")
    if not create_forecast_feature_group(uploader):
        logger.error("❌ Failed to create forecast feature group")
        return False
    
    # 3. Fetch weather forecasts
    logger.info("STEP 3: Fetching weather forecasts...")
    try:
        weather_df = fetch_openweather_forecast(hours=FORECAST_CONFIG['forecast_hours'])
    except Exception as e:
        logger.error(f"❌ Failed to fetch weather forecasts: {e}")
        return False

    # 3b. Fetch pollution forecasts
    logger.info("STEP 3b: Fetching pollution forecasts...")
    try:
        pollution_df = fetch_openweather_pollution_forecast(hours=FORECAST_CONFIG['forecast_hours'])
    except Exception as e:
        logger.error(f"❌ Failed to fetch pollution forecasts: {e}")
        return False

    # 3c. Merge on target_time/step_hour
    logger.info("STEP 3c: Merging weather + pollution forecasts...")
    # Use weather as anchor; align pollution times to the same hourly timestamps
    wx = weather_df.copy()
    pl = pollution_df.copy()

    # Ensure proper dtypes
    wx['target_time'] = pd.to_datetime(wx['target_time'], utc=True)
    pl['target_time'] = pd.to_datetime(pl['target_time'], utc=True)
    wx['step_hour'] = wx['step_hour'].astype(int)
    pl['step_hour'] = pl['step_hour'].astype(int)

    # Floor to the hour to remove minute/second drift
    wx['tt_hr'] = wx['target_time'].dt.floor('H')
    pl['tt_hr'] = pl['target_time'].dt.floor('H')

    # Auto-detect consistent hour offset between pollution and weather (e.g., -1h)
    try:
        # Compare overlapping steps to find typical offset in hours
        join_steps = sorted(set(wx['step_hour']).intersection(set(pl['step_hour'])))
        if join_steps:
            wx_sample = wx[wx['step_hour'].isin(join_steps)][['step_hour', 'tt_hr']].sort_values('step_hour')
            pl_sample = pl[pl['step_hour'].isin(join_steps)][['step_hour', 'tt_hr']].sort_values('step_hour')
            merged_steps = wx_sample.merge(pl_sample, on='step_hour', suffixes=('_wx', '_pl'))
            diffs_hrs = (merged_steps['tt_hr_wx'] - merged_steps['tt_hr_pl']).dt.total_seconds() / 3600.0
            # Use median rounded to nearest int for robustness
            offset_hours = int(round(float(diffs_hrs.median()))) if len(diffs_hrs) > 0 else 0
        else:
            offset_hours = 0
    except Exception:
        offset_hours = 0

    # Apply detected offset to pollution hours so they line up with weather
    pl['tt_hr_adj'] = pl['tt_hr'] + pd.Timedelta(hours=offset_hours)

    # Merge exactly on step and adjusted hour to avoid NaNs
    merged = wx.merge(
        pl[['step_hour', 'tt_hr_adj', 'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']],
        left_on=['step_hour', 'tt_hr'], right_on=['step_hour', 'tt_hr_adj'], how='left'
    )

    # Cleanup temp columns and keep only the first N steps
    drop_cols = [c for c in ['tt_hr', 'tt_hr_adj'] if c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    merged = merged.sort_values('step_hour').head(FORECAST_CONFIG['forecast_hours']).reset_index(drop=True)

    # 4. Format for decoder and direct LSTM (raw names included)
    logger.info("STEP 4: Formatting forecasts for decoder/direct LSTM...")
    decoder_df = format_forecast_for_decoder(merged)

    # 5. Push to Hopsworks
    logger.info("STEP 5: Pushing forecasts to Hopsworks...")
    if not push_forecast_data(uploader, decoder_df):
        logger.error("❌ Failed to push forecast data")
        return False
    
    logger.info("🎉 Forecast collection completed successfully!")
    logger.info(f"📊 Next forecast update in {FORECAST_CONFIG['update_frequency_hours']} hour(s)")
    
    return True

def main():
    """
    Main function for forecast collection
    """
    print("🌤️ Weather Forecast Collector for Seq2Seq LSTM")
    print("=" * 60)
    print(f"📊 Collecting {FORECAST_CONFIG['forecast_hours']}-hour forecasts")
    print(f"🔄 Update frequency: {FORECAST_CONFIG['update_frequency_hours']} hour(s)")
    print(f"📁 Feature group: {FORECAST_CONFIG['feature_group_name']}")
    print("=" * 60)
    
    success = run_forecast_collection()
    
    if success:
        print("✅ Forecast collection completed successfully!")
        print("📊 Forecasts are now available for Seq2Seq inference")
    else:
        print("❌ Forecast collection failed!")
    
    return success

if __name__ == "__main__":
    main() 