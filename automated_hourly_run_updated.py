import requests
import pandas as pd
import numpy as np
import hopsworks
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AQIDataCollector:
    """
    Updated AQI data collector that writes to both old and new feature groups
    """
    
    def __init__(self):
        # API keys
        self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY')
        self.hopsworks_api_key = os.getenv('HOPSWORKS_API_KEY')
        
        # Multan coordinates
        self.lat = 30.1575
        self.lon = 71.5249
        
        # Connect to Hopsworks
        self.connection = hopsworks.connection(api_key_value=self.hopsworks_api_key)
        self.fs = self.connection.get_feature_store()
        
    def collect_weather_data(self):
        """
        Collect weather data from OpenWeather API
        """
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            weather_data = {
                'timestamp': datetime.now(),
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind']['deg']
            }
            
            print(f"✅ Collected weather data: {weather_data['temperature']}°C, {weather_data['humidity']}%")
            return weather_data
            
        except Exception as e:
            print(f"❌ Error collecting weather data: {e}")
            return None
    
    def collect_aqi_data(self):
        """
        Collect AQI data from OpenWeather API
        """
        try:
            url = "https://api.openweathermap.org/data/2.5/air_pollution"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.openweather_api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            aqi_data = {
                'pm2_5': data['list'][0]['components']['pm2_5'],
                'pm10': data['list'][0]['components']['pm10'],
                'carbon_monoxide': data['list'][0]['components']['co'],
                'nitrogen_dioxide': data['list'][0]['components']['no2'],
                'ozone': data['list'][0]['components']['o3'],
                'sulphur_dioxide': data['list'][0]['components']['so2'],
                'nh3': data['list'][0]['components']['nh3'],
                'no': data['list'][0]['components']['no']
            }
            
            print(f"✅ Collected AQI data: PM2.5={aqi_data['pm2_5']}, PM10={aqi_data['pm10']}")
            return aqi_data
            
        except Exception as e:
            print(f"❌ Error collecting AQI data: {e}")
            return None
    
    def create_old_features(self, weather_data, aqi_data):
        """
        Create old 127 messy features (existing logic)
        """
        # Combine data
        combined_data = {**weather_data, **aqi_data}
        df = pd.DataFrame([combined_data])
        
        # Add timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create old feature engineering (your existing logic)
        # This should match your current automated_hourly_run.py logic
        
        # Example old features (you'll need to add your actual logic here)
        old_features = [
            'timestamp', 'pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 
            'ozone', 'sulphur_dioxide', 'nh3', 'no', 'temperature', 'humidity', 
            'pressure', 'wind_speed', 'wind_direction'
        ]
        
        # Add your existing feature engineering here
        # (lag features, rolling features, binary features, etc.)
        
        return df[old_features]
    
    def create_clean_features(self, weather_data, aqi_data):
        """
        Create new 58 clean features using feature_engineering.py for consistency
        """
        # Combine data
        combined_data = {**weather_data, **aqi_data}
        df = pd.DataFrame([combined_data])
        
        # Set timestamp as index for feature engineering
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Use feature_engineering.py for consistency
        from feature_engineering import AQIFeatureEngineer
        
        # Run full feature engineering pipeline
        engineer = AQIFeatureEngineer()
        engineered_df = engineer.engineer_features(df)
        
        if engineered_df.empty:
            print("❌ Feature engineering failed")
            return None
        
        # Add wind direction engineering (not in feature_engineering.py)
        if 'wind_direction' in df.columns:
            wind_dir = df['wind_direction']
            engineered_df['wind_direction_sin'] = np.sin(np.radians(wind_dir))
            engineered_df['wind_direction_cos'] = np.cos(np.radians(wind_dir))
            
            # Pollution source indicators
            engineered_df['is_wind_from_high_pm'] = (
                (engineered_df['wind_direction_sin'] > 0.5) |
                (engineered_df['wind_direction_sin'] > 0.3) & (engineered_df['wind_direction_cos'] > 0.3)
            ).astype(int)
            
            engineered_df['is_wind_from_low_pm'] = (
                (engineered_df['wind_direction_sin'] < -0.5) |
                (engineered_df['wind_direction_cos'] < -0.5)
            ).astype(int)
        
        # Add pollutant lags (not in feature_engineering.py)
        engineered_df['co_lag_1h'] = df['carbon_monoxide'].shift(1)
        engineered_df['o3_lag_1h'] = df['ozone'].shift(1)
        engineered_df['so2_lag_1h'] = df['sulphur_dioxide'].shift(1)
        
        # Add new interactions (not in feature_engineering.py)
        engineered_df['wind_direction_temp_interaction'] = engineered_df['wind_direction_sin'] * df['temperature']
        engineered_df['wind_direction_humidity_interaction'] = engineered_df['wind_direction_sin'] * df['humidity']
        engineered_df['pressure_humidity_interaction'] = df['pressure'] * df['humidity']
        engineered_df['co_pressure_interaction'] = df['carbon_monoxide'] * df['pressure']
        engineered_df['o3_temp_interaction'] = df['ozone'] * df['temperature']
        engineered_df['so2_humidity_interaction'] = df['sulphur_dioxide'] * df['humidity']
        
        # Select only the 58 clean features we want
        clean_features = [
            # Time columns
            'time', 'time_str',
            
            # Current pollutants
            'pm2_5', 'pm10', 'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3',
            
            # AQI
            'us_aqi',
            
            # Weather
            'temperature', 'humidity', 'pressure', 'wind_speed',
            
            # Wind direction engineering
            'wind_direction_sin', 'wind_direction_cos', 'is_wind_from_high_pm', 'is_wind_from_low_pm',
            
            # Lag features (1h, 2h, 3h only)
            'pm2_5_lag_1h', 'pm2_5_lag_2h', 'pm2_5_lag_3h',
            'pm10_lag_1h', 'pm10_lag_2h', 'pm10_lag_3h',
            
            # Pollutant lags
            'co_lag_1h', 'o3_lag_1h', 'so2_lag_1h',
            
            # Rolling features (optimized)
            'pm2_5_rolling_min_3h', 'pm2_5_rolling_mean_3h', 'pm2_5_rolling_min_12h', 'pm2_5_rolling_mean_12h',
            'pm10_rolling_min_3h', 'pm10_rolling_mean_3h', 'pm10_rolling_mean_24h',
            
            # Change rates
            'pm2_5_change_rate_1h', 'pm2_5_change_rate_6h', 'pm2_5_change_rate_24h',
            'pm10_change_rate_1h', 'pm10_change_rate_6h', 'pm10_change_rate_24h',
            
            # Cyclical time features
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            
            # Binary features (optimized)
            'is_hot', 'is_night', 'is_morning_rush', 'is_evening_rush', 'is_high_pm2_5', 'is_high_o3',
            
            # Interaction features (optimized)
            'temp_humidity_interaction', 'temp_wind_interaction', 'wind_direction_temp_interaction', 
            'wind_direction_humidity_interaction', 'pressure_humidity_interaction',
            
            # Pollutant-weather interactions
            'co_pressure_interaction', 'o3_temp_interaction', 'so2_humidity_interaction'
        ]
        
        # Select available features
        available_features = [f for f in clean_features if f in engineered_df.columns]
        clean_df = engineered_df[available_features].copy()
        
        # Create time_str for Hopsworks
        clean_df['time_str'] = clean_df['time'].dt.floor('H').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return clean_df
    
    def store_to_feature_groups(self, old_features, clean_features):
        """
        Store data to both old and new feature groups
        """
        try:
            # Store to OLD feature group (consistent with config.py)
            old_fg = self.fs.get_feature_group("multan_aqi_features_clean", version=1)
            old_fg.insert(old_features)
            print(f"📊 Stored in OLD group: {len(old_features.columns)} features")
            
            # Store to NEW feature group
            new_fg = self.fs.get_feature_group("aqi_clean_features_v2", version=1)
            new_fg.insert(clean_features)
            print(f"✨ Stored in NEW group: {len(clean_features.columns)} features")
            
            return True
            
        except Exception as e:
            print(f"❌ Error storing data: {e}")
            return False
    
    def run_collection(self):
        """
        Main collection function
        """
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting data collection...")
        
        # 1. Collect raw data
        weather_data = self.collect_weather_data()
        aqi_data = self.collect_aqi_data()
        
        if weather_data is None or aqi_data is None:
            print("❌ Failed to collect data")
            return False
        
        # 2. Create old features (127 messy features)
        old_features = self.create_old_features(weather_data, aqi_data)
        
        # 3. Create clean features (58 clean features)
        clean_features = self.create_clean_features(weather_data, aqi_data)
        
        # 4. Store to both feature groups
        success = self.store_to_feature_groups(old_features, clean_features)
        
        if success:
            print("✅ Data collection and storage completed successfully!")
            print(f"📊 Old features: {len(old_features.columns)} columns")
            print(f"✨ Clean features: {len(clean_features.columns)} columns")
        else:
            print("❌ Data storage failed!")
        
        return success

def main():
    """
    Main function to run hourly data collection
    """
    print("🚀 AQI Data Collection - Dual Feature Group Mode")
    print("=" * 60)
    print("📊 Writing to BOTH old and new feature groups")
    print("🔄 This is the transition period - monitoring both groups")
    print("=" * 60)
    
    collector = AQIDataCollector()
    
    # Run collection
    success = collector.run_collection()
    
    if success:
        print("🎉 Collection cycle completed!")
        print("📊 Next steps:")
        print("   1. Monitor both feature groups for data quality")
        print("   2. Validate clean features are working correctly")
        print("   3. Test model performance with new features")
        print("   4. Switch to new group only when validated")
    else:
        print("❌ Collection cycle failed!")

if __name__ == "__main__":
    main() 