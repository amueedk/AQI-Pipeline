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
        
        # Connect to Hopsworks (same as feature_migration.py)
        self.project_name = 'AQIMultan'  # From config.py
        
        if not self.hopsworks_api_key:
            print("❌ HOPSWORKS_API_KEY environment variable is not set!")
            return
            
        try:
            print(f"🔗 Connecting to Hopsworks project: {self.project_name}...")
            self.project = hopsworks.login(
                api_key_value=self.hopsworks_api_key,
                project=self.project_name
            )
            self.fs = self.project.get_feature_store()
            print("✅ Successfully connected to Hopsworks Feature Store.")
        except Exception as e:
            print(f"❌ Failed to connect to Hopsworks: {e}")
            self.fs = None
        
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
        Collect AQI data from OpenWeather API (same as original script)
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
            
            # Use same column names as original script
            aqi_data = {
                'pm2_5': data['list'][0]['components']['pm2_5'],
                'pm10': data['list'][0]['components']['pm10'],
                'co': data['list'][0]['components']['co'],  # Keep as 'co' for consistency
                'no2': data['list'][0]['components']['no2'],  # Keep as 'no2' for consistency
                'o3': data['list'][0]['components']['o3'],  # Keep as 'o3' for consistency
                'so2': data['list'][0]['components']['so2'],  # Keep as 'so2' for consistency
                'nh3': data['list'][0]['components']['nh3'],
                'no': data['list'][0]['components']['no']
            }
            
            print(f"✅ Collected AQI data: PM2.5={aqi_data['pm2_5']}, PM10={aqi_data['pm10']}")
            return aqi_data
            
        except Exception as e:
            print(f"❌ Error collecting AQI data: {e}")
            return None
    
    def create_old_features_with_context(self, new_df, group_name):
        """
        Create old 127 messy features with historical context from OLD group
        """
        print(f"🔧 Creating old features with historical context from {group_name}...")
        
        # Fetch existing data from OLD group
        existing_df = self.fetch_existing_data(group_name)
        
        # Sort existing data by timestamp to ensure proper lag feature calculation
        if not existing_df.empty:
            existing_df = existing_df.sort_index()
            print("✅ Sorted existing data by timestamp for proper lag feature calculation")
        
        # Combine existing and new data for proper feature engineering
        if not existing_df.empty:
            print("🔄 Combining existing and new data for proper feature engineering...")
            
            # Extract raw features from existing data for combination (same as original)
            raw_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
                           'carbon_monoxide', 'no', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide', 
                           'pm2_5', 'pm10', 'nh3', 'openweather_aqi', 'pm2_5_aqi', 'pm10_aqi', 'us_aqi',
                           'city', 'latitude', 'longitude']
            
            # Get only raw features from existing data
            available_raw_features = [col for col in raw_features if col in existing_df.columns]
            existing_raw_df = existing_df[available_raw_features].copy()
            
            print(f"📊 Extracted {len(available_raw_features)} raw features from existing data")
            
            # Combine raw data (existing + new)
            combined_raw_df = pd.concat([existing_raw_df, new_df], axis=0)
            
            # Sort by timestamp
            combined_raw_df = combined_raw_df.sort_index()
            
            print(f"📊 Combined dataset: {len(existing_raw_df)} existing + {len(new_df)} new = {len(combined_raw_df)} total records")
            print(f"📊 Date range: {combined_raw_df.index.min()} to {combined_raw_df.index.max()}")
            
        else:
            print("📊 No existing data found. Using only new data...")
            combined_raw_df = new_df
        
        # Use the SAME feature engineering as original script
        from feature_engineering import AQIFeatureEngineer
        
        # Run full feature engineering pipeline (creates 127 features)
        engineer = AQIFeatureEngineer()
        engineered_df = engineer.engineer_features(combined_raw_df)
        
        if engineered_df.empty:
            print("❌ Old feature engineering failed")
            return None
        
        # Create time_str for Hopsworks (same as original)
        engineered_df['time_str'] = engineered_df['time'].dt.floor('H').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"✅ Old features created: {len(engineered_df.columns)} columns")
        return engineered_df
    
    def create_clean_features_with_context(self, new_df, group_name):
        """
        Create new 58 clean features with historical context from NEW group
        """
        print(f"🔧 Creating clean features with historical context from {group_name}...")
        
        # Fetch existing data from NEW group
        existing_df = self.fetch_existing_data(group_name)
        
        # Sort existing data by timestamp to ensure proper lag feature calculation
        if not existing_df.empty:
            existing_df = existing_df.sort_index()
            print("✅ Sorted existing data by timestamp for proper lag feature calculation")
        
        # Combine existing and new data for proper feature engineering
        if not existing_df.empty:
            print("🔄 Combining existing and new data for proper feature engineering...")
            
            # Extract raw features from existing data for combination (same as original)
            raw_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
                           'carbon_monoxide', 'no', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide', 
                           'pm2_5', 'pm10', 'nh3', 'openweather_aqi', 'pm2_5_aqi', 'pm10_aqi', 'us_aqi',
                           'city', 'latitude', 'longitude']
            
            # Get only raw features from existing data
            available_raw_features = [col for col in raw_features if col in existing_df.columns]
            existing_raw_df = existing_df[available_raw_features].copy()
            
            print(f"📊 Extracted {len(available_raw_features)} raw features from existing data")
            
            # Combine raw data (existing + new)
            combined_raw_df = pd.concat([existing_raw_df, new_df], axis=0)
            
            # Sort by timestamp
            combined_raw_df = combined_raw_df.sort_index()
            
            print(f"📊 Combined dataset: {len(existing_raw_df)} existing + {len(new_df)} new = {len(combined_raw_df)} total records")
            print(f"📊 Date range: {combined_raw_df.index.min()} to {combined_raw_df.index.max()}")
            
        else:
            print("📊 No existing data found. Using only new data...")
            combined_raw_df = new_df
        
        # Use feature_engineering.py for consistency
        from feature_engineering import AQIFeatureEngineer
        
        # Run full feature engineering pipeline
        engineer = AQIFeatureEngineer()
        engineered_df = engineer.engineer_features(combined_raw_df)
        
        if engineered_df.empty:
            print("❌ Feature engineering failed")
            return None
        
        # Add wind direction engineering (not in feature_engineering.py) - only if not already present
        if 'wind_direction' in combined_raw_df.columns:
            if 'wind_direction_sin' not in engineered_df.columns:
                wind_dir = combined_raw_df['wind_direction']
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
                print("✅ Added wind direction engineering features")
            else:
                print("✅ Wind direction features already present in existing data - preserving them")
        
        # Add pollutant lags (not in feature_engineering.py)
        # Handle both column name formats
        co_col = 'carbon_monoxide' if 'carbon_monoxide' in combined_raw_df.columns else 'co'
        o3_col = 'ozone' if 'ozone' in combined_raw_df.columns else 'o3'
        so2_col = 'sulphur_dioxide' if 'sulphur_dioxide' in combined_raw_df.columns else 'so2'
        
        engineered_df['co_lag_1h'] = combined_raw_df[co_col].shift(1)
        engineered_df['o3_lag_1h'] = combined_raw_df[o3_col].shift(1)
        engineered_df['so2_lag_1h'] = combined_raw_df[so2_col].shift(1)
        
        # Add new interactions (not in feature_engineering.py) - only if not already present
        if 'wind_direction_temp_interaction' not in engineered_df.columns:
            engineered_df['wind_direction_temp_interaction'] = engineered_df['wind_direction_sin'] * combined_raw_df['temperature']
            engineered_df['wind_direction_humidity_interaction'] = engineered_df['wind_direction_sin'] * combined_raw_df['humidity']
            engineered_df['pressure_humidity_interaction'] = combined_raw_df['pressure'] * combined_raw_df['humidity']
            engineered_df['co_pressure_interaction'] = combined_raw_df[co_col] * combined_raw_df['pressure']
            engineered_df['o3_temp_interaction'] = combined_raw_df[o3_col] * combined_raw_df['temperature']
            engineered_df['so2_humidity_interaction'] = combined_raw_df[so2_col] * combined_raw_df['humidity']
            print("✅ Added new interaction features")
        else:
            print("✅ New interaction features already present in existing data - preserving them")
        
        # Select only the 59 clean features we want (including raw wind_direction)
        clean_features = [
            # Time columns
            'time', 'time_str',
            
            # Current pollutants
            'pm2_5', 'pm10', 'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3',
            
            # AQI
            'us_aqi',
            
            # Weather (including raw wind_direction)
            'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
            
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
        
        print(f"✅ Clean features created: {len(clean_df.columns)} columns")
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
    
    def fetch_existing_data(self, group_name):
        """
        Fetch existing data from Hopsworks feature group (same as original script)
        """
        try:
            print(f"📥 Fetching existing data from feature group: {group_name}")
            
            # Get the feature group
            fg = self.fs.get_feature_group(name=group_name, version=1)
            
            # Read all data from the feature group
            df = fg.read()
            
            if df.empty:
                print("No existing data found in Hopsworks feature group.")
                return pd.DataFrame()
            
            # Fix timestamp handling - ensure proper datetime index (same as original)
            print("Fixing timestamp handling for existing data...")
            if 'time' in df.columns:
                # Convert time column to proper datetime
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                # Set as index
                df.set_index('time', inplace=True)
                # Remove time_str column if it exists (it's just for Hopsworks primary key)
                if 'time_str' in df.columns:
                    df = df.drop(columns=['time_str'])
            elif isinstance(df.index, pd.DatetimeIndex):
                # Index is already datetime, but let's ensure it's proper
                df.index = pd.to_datetime(df.index, errors='coerce')
            else:
                print("No 'time' column found and index is not datetime. Cannot proceed.")
                return pd.DataFrame()
            
            # Ensure timezone consistency - if timezone-naive, localize to UTC
            if df.index.tz is None:
                print("Converting timezone-naive timestamps to UTC...")
                df.index = df.index.tz_localize('UTC')
            else:
                print("Timestamps are already timezone-aware")
            
            # Remove any rows with invalid timestamps
            invalid_timestamps = df.index.isna()
            if invalid_timestamps.any():
                print(f"Found {invalid_timestamps.sum()} rows with invalid timestamps. Removing them.")
                df = df[~invalid_timestamps]
            
            if df.empty:
                print("No valid data remaining after timestamp cleanup.")
                return pd.DataFrame()
            
            print(f"✅ Successfully fetched {len(df)} existing records from Hopsworks.")
            print(f"📊 Existing data date range: {df.index.min()} to {df.index.max()}")
            print(f"📊 Existing data columns: {len(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"⚠️ Could not fetch existing data from Hopsworks: {e}")
            print("Will proceed with only new data (lag features will be NaN for first few records).")
            return pd.DataFrame()

    def run_collection(self):
        """
        Main collection function (updated to use separate historical context for each group)
        """
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting data collection...")
        
        # 2. Collect new raw data
        print("📡 STEP 1: Collecting new raw data...")
        weather_data = self.collect_weather_data()
        aqi_data = self.collect_aqi_data()
        
        if weather_data is None or aqi_data is None:
            print("❌ Failed to collect data")
            return False
        
        # Create new data DataFrame
        combined_data = {**weather_data, **aqi_data}
        new_df = pd.DataFrame([combined_data])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
        new_df = new_df.set_index('timestamp')
        
        # Ensure timezone consistency - convert to UTC and make timezone-aware
        new_df.index = new_df.index.tz_localize('UTC')
        
        # Add location metadata (same as original)
        new_df["city"] = "Multan"
        new_df["latitude"] = 30.1575
        new_df["longitude"] = 71.5249
        
        # Rename columns to match feature engineering expectations (same as original)
        rename_map = {
            'co': 'carbon_monoxide',
            'no2': 'nitrogen_dioxide',
            'o3': 'ozone',
            'so2': 'sulphur_dioxide'
        }
        new_df = new_df.rename(columns=rename_map)
        
        # Ensure 'no' column exists (nitric oxide from OpenWeather API)
        if 'no' not in new_df.columns:
            print("⚠️ 'no' column missing from API response. Adding with NaN values.")
            new_df['no'] = None
        
        # 3. Create OLD features with historical context from OLD group
        print("📥 STEP 2: Creating OLD features with historical context from OLD group...")
        old_features = self.create_old_features_with_context(new_df, "multan_aqi_features_clean")
        
        # 4. Create CLEAN features with historical context from NEW group
        print("📥 STEP 3: Creating CLEAN features with historical context from NEW group...")
        clean_features = self.create_clean_features_with_context(new_df, "aqi_clean_features_v2")
        
        if old_features is None or clean_features is None:
            print("❌ Feature engineering failed")
            return False
        
        # 5. Store to both feature groups
        print("💾 STEP 4: Storing to both feature groups...")
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
    
    # Check if connection was successful
    if collector.fs is None:
        print("❌ Failed to connect to Hopsworks. Cannot proceed.")
        return False
    
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
    
    return success

if __name__ == "__main__":
    main() 