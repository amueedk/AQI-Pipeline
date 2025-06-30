"""
Data Collector for AQI Data Collection & Feature Engineering
Fetches air quality and weather data from Open-Meteo API for Multan
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import json
import os
from config import MULTAN_CONFIG, OPEN_METEO_CONFIG, DATA_CONFIG, PATHS

# Import Open-Meteo official client
try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    OPENMETEO_AVAILABLE = True
except ImportError:
    OPENMETEO_AVAILABLE = False
    openmeteo = None

logger = None

class MultanDataCollector:
    def __init__(self):
        self.latitude = MULTAN_CONFIG["latitude"]
        self.longitude = MULTAN_CONFIG["longitude"]
        self.timezone = MULTAN_CONFIG["timezone"]
        self.city_name = MULTAN_CONFIG["city_name"]
        
        # Create directories if they don't exist
        for path in PATHS.values():
            os.makedirs(path, exist_ok=True)
        
        # Configure logging after directories are created
        global logger
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
    
    def fetch_air_quality_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch air quality data from Open-Meteo API using official client
        """
        if not OPENMETEO_AVAILABLE:
            logger.error("Open-Meteo client not available. Please install: pip install openmeteo-requests requests-cache retry-requests")
            return pd.DataFrame()
        
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": [
                "pm10", "us_aqi", "pm2_5", "carbon_monoxide", "carbon_dioxide", 
                "nitrogen_dioxide", "sulphur_dioxide", "ozone", "dust", "uv_index", 
                "us_aqi_pm2_5", "us_aqi_pm10", "us_aqi_nitrogen_dioxide", 
                "us_aqi_carbon_monoxide", "us_aqi_ozone", "us_aqi_sulphur_dioxide"
            ],
            "timezone": self.timezone,
            "past_days": 14
        }
        
        try:
            logger.info(f"Fetching air quality data for past 14 days")
            responses = openmeteo.weather_api(url, params=params)
            
            # Get the first (and only) response
            response = responses[0]
            
            # Process hourly data
            hourly = response.Hourly()
            hourly_data = {
                "time": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s"),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }
            
            # Add all hourly variables
            for i, variable in enumerate(params["hourly"]):
                hourly_data[variable] = hourly.Variables(i).ValuesAsNumpy()
            
            # Convert to DataFrame
            df = pd.DataFrame(hourly_data)
            df.set_index('time', inplace=True)
            
            # Add metadata
            df['city'] = self.city_name
            df['latitude'] = self.latitude
            df['longitude'] = self.longitude
            
            logger.info(f"Successfully fetched {len(df)} air quality records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching air quality data: {e}")
            return pd.DataFrame()
    
    def fetch_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch comprehensive weather data from Open-Meteo API
        """
        url = OPEN_METEO_CONFIG["weather_url"]
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": [
                "temperature_2m", "relative_humidity_2m", "dew_point_2m", 
                "apparent_temperature", "pressure_msl", "surface_pressure", 
                "precipitation", "rain", "snowfall", "cloud_cover", 
                "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", 
                "visibility", "wind_speed_10m", "wind_speed_100m", 
                "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m", 
                "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm", 
                "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", 
                "vapour_pressure_deficit", "evapotranspiration", 
                "weather_code", "is_day"
            ],
            "timezone": self.timezone,
            "past_days": 14
        }
        
        try:
            logger.info(f"Fetching weather data for past 14 days")
            responses = openmeteo.weather_api(url, params=params)
            
            # Get the first (and only) response
            response = responses[0]
            
            # Process hourly data
            hourly = response.Hourly()
            hourly_data = {
                "time": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s"),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }
            
            # Add all hourly variables
            for i, variable in enumerate(params["hourly"]):
                hourly_data[variable] = hourly.Variables(i).ValuesAsNumpy()
            
            # Convert to DataFrame
            df = pd.DataFrame(hourly_data)
            df.set_index('time', inplace=True)
            
            # Add metadata
            df['city'] = self.city_name
            df['latitude'] = self.latitude
            df['longitude'] = self.longitude
            
            logger.info(f"Successfully fetched {len(df)} weather records")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error in weather data fetch: {e}")
            return pd.DataFrame()
    
    def collect_historical_data(self, days_back: int = None) -> pd.DataFrame:
        """
        Collect historical data for the past N days (manual one-time collection)
        """
        if days_back is None:
            days_back = DATA_CONFIG["historical_days"]
            
        # Use yesterday as end date to ensure we don't request future dates
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back + 1)).strftime('%Y-%m-%d')
        
        logger.info(f"Collecting historical data from {start_date} to {end_date}")
        
        # Fetch both air quality and weather data
        aq_data = self.fetch_air_quality_data(start_date, end_date)
        weather_data = self.fetch_weather_data(start_date, end_date)
        
        # Merge the dataframes
        if not aq_data.empty and not weather_data.empty:
            combined_data = pd.concat([aq_data, weather_data], axis=1)
            # Remove duplicate columns if any
            combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
            
            # Save to file
            filename = f"historical_data_{start_date}_to_{end_date}.csv"
            filepath = os.path.join(PATHS['data_dir'], filename)
            combined_data.to_csv(filepath)
            logger.info(f"Historical data saved to {filepath}")
            
            return combined_data
        else:
            logger.error("Failed to fetch data from one or both APIs")
            return pd.DataFrame()
    
    def collect_current_data(self) -> pd.DataFrame:
        """
        Collect current data (last few hours) - for automated hourly collection
        """
        # Calculate time range for current data (last 6 hours)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=6)
        
        # Format dates for API
        end_date = end_time.strftime('%Y-%m-%d')
        start_date = start_time.strftime('%Y-%m-%d')
        
        logger.info(f"Collecting current data from {start_time} to {end_time}")
        
        # Use different API parameters for current data
        url_aq = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params_aq = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": [
                "pm10", "us_aqi", "pm2_5", "carbon_monoxide", "carbon_dioxide", 
                "nitrogen_dioxide", "sulphur_dioxide", "ozone", "dust", "uv_index", 
                "us_aqi_pm2_5", "us_aqi_pm10", "us_aqi_nitrogen_dioxide", 
                "us_aqi_carbon_monoxide", "us_aqi_ozone", "us_aqi_sulphur_dioxide"
            ],
            "timezone": self.timezone,
            "start_date": start_date,
            "end_date": end_date
        }
        
        url_weather = OPEN_METEO_CONFIG["weather_url"]
        params_weather = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": [
                "temperature_2m", "relative_humidity_2m", "dew_point_2m", 
                "apparent_temperature", "pressure_msl", "surface_pressure", 
                "precipitation", "rain", "snowfall", "cloud_cover", 
                "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", 
                "visibility", "wind_speed_10m", "wind_speed_100m", 
                "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m", 
                "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm", 
                "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", 
                "vapour_pressure_deficit", "evapotranspiration", 
                "weather_code", "is_day"
            ],
            "timezone": self.timezone,
            "start_date": start_date,
            "end_date": end_date
        }
        
        try:
            # Fetch air quality data
            logger.info("Fetching current air quality data")
            responses_aq = openmeteo.weather_api(url_aq, params=params_aq)
            response_aq = responses_aq[0]
            
            # Fetch weather data
            logger.info("Fetching current weather data")
            responses_weather = openmeteo.weather_api(url_weather, params=params_weather)
            response_weather = responses_weather[0]
            
            # Process air quality data
            hourly_aq = response_aq.Hourly()
            aq_data = {
                "time": pd.date_range(
                    start=pd.to_datetime(hourly_aq.Time(), unit="s"),
                    end=pd.to_datetime(hourly_aq.TimeEnd(), unit="s"),
                    freq=pd.Timedelta(seconds=hourly_aq.Interval()),
                    inclusive="left"
                )
            }
            
            for i, variable in enumerate(params_aq["hourly"]):
                aq_data[variable] = hourly_aq.Variables(i).ValuesAsNumpy()
            
            df_aq = pd.DataFrame(aq_data)
            df_aq.set_index('time', inplace=True)
            df_aq['city'] = self.city_name
            df_aq['latitude'] = self.latitude
            df_aq['longitude'] = self.longitude
            
            # Process weather data
            hourly_weather = response_weather.Hourly()
            weather_data = {
                "time": pd.date_range(
                    start=pd.to_datetime(hourly_weather.Time(), unit="s"),
                    end=pd.to_datetime(hourly_weather.TimeEnd(), unit="s"),
                    freq=pd.Timedelta(seconds=hourly_weather.Interval()),
                    inclusive="left"
                )
            }
            
            for i, variable in enumerate(params_weather["hourly"]):
                weather_data[variable] = hourly_weather.Variables(i).ValuesAsNumpy()
            
            df_weather = pd.DataFrame(weather_data)
            df_weather.set_index('time', inplace=True)
            df_weather['city'] = self.city_name
            df_weather['latitude'] = self.latitude
            df_weather['longitude'] = self.longitude
            
            # Combine data
            combined_data = pd.concat([df_aq, df_weather], axis=1)
            combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
            
            # Save current data with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"current_data_{timestamp}.csv"
            filepath = os.path.join(PATHS['data_dir'], filename)
            combined_data.to_csv(filepath)
            
            logger.info(f"Current data saved to {filepath}")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error fetching current data: {e}")
            return pd.DataFrame()
    
    def append_to_master_dataset(self, new_data: pd.DataFrame) -> bool:
        """
        Append new data to the master dataset
        """
        try:
            master_file = os.path.join(PATHS['data_dir'], 'master_dataset.csv')
            
            if os.path.exists(master_file):
                # Load existing master dataset
                master_df = pd.read_csv(master_file, index_col=0, parse_dates=True)
                
                # Combine with new data
                combined_df = pd.concat([master_df, new_data])
                
                # Remove duplicates based on timestamp
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                
                # Sort by timestamp
                combined_df = combined_df.sort_index()
                
            else:
                combined_df = new_data
            
            # Save updated master dataset
            combined_df.to_csv(master_file)
            logger.info(f"Master dataset updated with {len(new_data)} new records")
            return True
            
        except Exception as e:
            logger.error(f"Error updating master dataset: {e}")
            return False

def main():
    """
    Main function to run data collection
    """
    collector = MultanDataCollector()
    
    # Collect historical data (manual one-time)
    print("Collecting historical data...")
    historical_data = collector.collect_historical_data()
    
    if not historical_data.empty:
        print(f"Collected {len(historical_data)} historical records")
        print(f"Columns: {list(historical_data.columns)}")
        print(f"Date range: {historical_data.index.min()} to {historical_data.index.max()}")
        
        # Show US AQI statistics
        if 'us_aqi' in historical_data.columns:
            aqi_stats = historical_data['us_aqi'].describe()
            print(f"\nUS AQI Statistics:")
            print(f"Mean: {aqi_stats['mean']:.2f}")
            print(f"Min: {aqi_stats['min']:.2f}")
            print(f"Max: {aqi_stats['max']:.2f}")
            print(f"Std: {aqi_stats['std']:.2f}")
    
    # Collect current data (for ongoing collection)
    print("\nCollecting current data...")
    current_data = collector.collect_current_data()
    
    if not current_data.empty:
        print(f"Collected {len(current_data)} current records")
        
        # Append to master dataset
        collector.append_to_master_dataset(current_data)

if __name__ == "__main__":
    main() 