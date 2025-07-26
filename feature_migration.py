import pandas as pd
import numpy as np
import hopsworks
from datetime import datetime
import warnings
import os
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class HopsworksFeatureMigration:
    """
    Migrate from old 127 messy features to new 58 clean features
    Based on comprehensive EDA findings
    """
    
    def __init__(self):
        # Connect to Hopsworks with proper configuration (consistent with existing scripts)
        self.api_key = os.getenv('HOPSWORKS_API_KEY')
        self.project_name = 'AQIMultan'  # From config.py
        
        if not self.api_key:
            print("❌ HOPSWORKS_API_KEY environment variable is not set!")
            return
            
        try:
            print(f"🔗 Connecting to Hopsworks project: {self.project_name}...")
            self.project = hopsworks.login(
                api_key_value=self.api_key,
                project=self.project_name
            )
            self.fs = self.project.get_feature_store()
            print("✅ Successfully connected to Hopsworks Feature Store.")
        except Exception as e:
            print(f"❌ Failed to connect to Hopsworks: {e}")
            self.fs = None
        
    def create_new_feature_group(self):
        """
        Create new clean feature group in Hopsworks (consistent with existing setup)
        """
        try:
            fg = self.fs.get_or_create_feature_group(
                name="aqi_clean_features_v2",  # New version to avoid conflicts
                version=1,
                description="Clean, optimized features for AQI prediction (58 features based on EDA)",
                primary_key=["time_str"],  # Fixed: primary_key (singular), not primary_keys
                event_time="time",  # Consistent with existing: time as event time
                online_enabled=True  # Enable both offline and online storage (like manual_historic_run.py)
            )
            print("✅ Created new feature group: aqi_clean_features_v2 (offline + online)")
            return fg
        except Exception as e:
            print(f"❌ Error creating feature group: {e}")
            return None
    
    def read_old_feature_group(self):
        """
        Read old messy data from Hopsworks (consistent with existing feature group name)
        """
        try:
            # Use the feature group name from config.py
            old_fg = self.fs.get_feature_group("multan_aqi_features_clean", version=1)
            old_data = old_fg.read()
            print(f"✅ Read {len(old_data)} rows from old feature group")
            print(f"📊 Old features: {len(old_data.columns)} columns")
            return old_data
        except Exception as e:
            print(f"❌ Error reading old feature group: {e}")
            return None
    
    def transform_to_clean_features(self, df):
        """
        Transform 127 messy features → 58 clean features
        Uses feature_engineering.py for consistency
        """
        print("🔄 Starting feature transformation using feature_engineering.py...")
        
        # Import feature engineering
        from feature_engineering import AQIFeatureEngineer
        
        # Ensure proper time index
        if 'time' in df.columns:
            df = df.set_index('time')
        elif 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            print("❌ No time column found in data")
            return None
        
        # Use feature_engineering.py for consistent feature creation
        engineer = AQIFeatureEngineer()
        
        # Run full feature engineering pipeline
        engineered_df = engineer.engineer_features(df)
        
        if engineered_df.empty:
            print("❌ Feature engineering failed")
            return None
        
        # Now select only the 58 clean features we want
        print("🔍 Selecting 58 clean features from engineered data...")
        
        # Define the 59 clean features we want (including raw wind_direction)
        clean_features = [
            # Time columns (required for Hopsworks)
            'time', 'time_str',
            
            # 1. CURRENT POLLUTANTS (6 features)
            'pm2_5', 'pm10', 'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3',
            
            # 2. AQI (1 feature)
            'us_aqi',
            
            # 3. CURRENT WEATHER (5 features - including raw wind_direction)
            'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
            
            # 4. WIND DIRECTION ENGINEERING (4 features) - NEW
            'wind_direction_sin', 'wind_direction_cos', 'is_wind_from_high_pm', 'is_wind_from_low_pm',
            
            # 5. LAG FEATURES (6 features) - 1h, 2h, 3h only
            'pm2_5_lag_1h', 'pm2_5_lag_2h', 'pm2_5_lag_3h',
            'pm10_lag_1h', 'pm10_lag_2h', 'pm10_lag_3h',
            
            # 6. POLLUTANT LAGS (3 features) - NEW
            'co_lag_1h', 'o3_lag_1h', 'so2_lag_1h',
            
            # 7. ROLLING FEATURES (7 features) - OPTIMIZED
            'pm2_5_rolling_min_3h', 'pm2_5_rolling_mean_3h', 'pm2_5_rolling_min_12h', 'pm2_5_rolling_mean_12h',
            'pm10_rolling_min_3h', 'pm10_rolling_mean_3h', 'pm10_rolling_mean_24h',
            
            # 8. CHANGE RATES (6 features)
            'pm2_5_change_rate_1h', 'pm2_5_change_rate_6h', 'pm2_5_change_rate_24h',
            'pm10_change_rate_1h', 'pm10_change_rate_6h', 'pm10_change_rate_24h',
            
            # 9. CYCLICAL TIME FEATURES (8 features)
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            
            # 10. BINARY FEATURES (6 features) - OPTIMIZED
            'is_hot', 'is_night', 'is_morning_rush', 'is_evening_rush', 'is_high_pm2_5', 'is_high_o3',
            
            # 11. INTERACTION FEATURES (5 features) - OPTIMIZED
            'temp_humidity_interaction', 'temp_wind_interaction', 'wind_direction_temp_interaction', 
            'wind_direction_humidity_interaction', 'pressure_humidity_interaction',
            
            # 12. POLLUTANT-WEATHER INTERACTIONS (3 features) - NEW
            'co_pressure_interaction', 'o3_temp_interaction', 'so2_humidity_interaction'
        ]
        
        # Add wind direction engineering (not in feature_engineering.py)
        print("💨 Adding wind direction engineering...")
        if 'wind_direction' in df.columns:
            wind_dir = df['wind_direction']
            engineered_df['wind_direction_sin'] = np.sin(np.radians(wind_dir))
            engineered_df['wind_direction_cos'] = np.cos(np.radians(wind_dir))
            
            # Fix floating-point precision issues (round tiny values to 0)
            engineered_df['wind_direction_sin'] = engineered_df['wind_direction_sin'].round(10)
            engineered_df['wind_direction_cos'] = engineered_df['wind_direction_cos'].round(10)
            
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
        print("🧪 Adding pollutant lags...")
        engineered_df['co_lag_1h'] = df['carbon_monoxide'].shift(1)
        engineered_df['o3_lag_1h'] = df['ozone'].shift(1)
        engineered_df['so2_lag_1h'] = df['sulphur_dioxide'].shift(1)
        
        # Add new interactions (not in feature_engineering.py)
        print("🔗 Adding new interaction features...")
        engineered_df['wind_direction_temp_interaction'] = engineered_df['wind_direction_sin'] * df['temperature']
        engineered_df['wind_direction_humidity_interaction'] = engineered_df['wind_direction_sin'] * df['humidity']
        engineered_df['pressure_humidity_interaction'] = df['pressure'] * df['humidity']
        engineered_df['co_pressure_interaction'] = df['carbon_monoxide'] * df['pressure']
        engineered_df['o3_temp_interaction'] = df['ozone'] * df['temperature']
        engineered_df['so2_humidity_interaction'] = df['sulphur_dioxide'] * df['humidity']
        
        # Select only the clean features
        available_features = [f for f in clean_features if f in engineered_df.columns]
        missing_features = [f for f in clean_features if f not in engineered_df.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
        
        clean_df = engineered_df[available_features].copy()
        
        # Create time_str for Hopsworks
        clean_df['time_str'] = clean_df['time'].dt.floor('H').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"✅ Feature transformation complete!")
        print(f"📊 Clean features: {len(clean_df.columns)} columns")
        print(f"📊 Clean rows: {len(clean_df)} rows")
        print(f"✅ Available features: {len(available_features)}")
        
        return clean_df
    
    def migrate_historical_data(self):
        """
        Complete migration process
        """
        print("🚀 Starting historical data migration...")
        
        # Check if connection was successful
        if self.fs is None:
            print("❌ Not connected to Hopsworks. Cannot proceed.")
            return None
        
        # 1. Read old data
        old_data = self.read_old_feature_group()
        if old_data is None:
            return None
        
        # 2. Transform to clean features
        clean_data = self.transform_to_clean_features(old_data)
        
        # 3. Create new feature group
        new_fg = self.create_new_feature_group()
        if new_fg is None:
            return None
        
        # 4. Store in new feature group
        try:
            new_fg.insert(clean_data, write_options={"wait_for_job": True})
            print(f"✅ Successfully migrated {len(clean_data)} rows to new feature group")
            print(f"📊 Old features: {len(old_data.columns)} → New features: {len(clean_data.columns)}")
            print("✅ Feature group is online-enabled and data is committed!")
            
            return clean_data
        except Exception as e:
            print(f"❌ Error storing data: {e}")
            return None
    
    def validate_migration(self, clean_data):
        """
        Validate the migration results
        """
        print("🔍 Validating migration...")
        
        # Check required features
        required_features = [
            'wind_direction_sin', 'wind_direction_cos',
            'is_wind_from_high_pm', 'is_wind_from_low_pm',
            'co_lag_1h', 'o3_lag_1h', 'so2_lag_1h',
            'pm2_5_rolling_min_3h', 'pm10_rolling_min_3h'
        ]
        
        missing_features = [f for f in required_features if f not in clean_data.columns]
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
            return False
        else:
            print("✅ All required features present")
        
        # Check data quality
        print(f"📊 Data quality check:")
        print(f"   Missing values: {clean_data.isnull().sum().sum()}")
        print(f"   Zero values: {(clean_data == 0).sum().sum()}")
        print(f"   Infinite values: {np.isinf(clean_data.select_dtypes(include=[np.number])).sum().sum()}")
        
        # Check feature counts
        expected_features = 59  # Based on our plan (including raw wind_direction)
        actual_features = len(clean_data.columns)
        print(f"📊 Feature count: {actual_features} (expected: {expected_features})")
        
        if actual_features == expected_features:
            print("✅ Feature count matches expected")
            return True
        else:
            print(f"❌ Feature count mismatch: {actual_features} vs {expected_features}")
            return False

def main():
    """
    Main migration function
    """
    print("🚀 AQI Feature Migration Tool")
    print("=" * 50)
    
    # Initialize migration
    migrator = HopsworksFeatureMigration()
    
    # Perform migration
    clean_data = migrator.migrate_historical_data()
    
    if clean_data is not None:
        # Validate migration
        success = migrator.validate_migration(clean_data)
        
        if success:
            print("🎉 Migration completed successfully!")
            print("📊 Next steps:")
            print("   1. Update automated_hourly_run.py to write to both groups")
            print("   2. Monitor both groups for 1 week")
            print("   3. Validate data quality and model performance")
            print("   4. Switch to new group only when validated")
            
            # Create success file for GitHub Actions
            from datetime import datetime
            with open("migration_success.txt", "w") as f:
                f.write(f"Migration completed successfully!\n")
                f.write(f"New feature group: aqi_clean_features_v2\n")
                f.write(f"Features migrated: {len(clean_data.columns)}\n")
                f.write(f"Rows migrated: {len(clean_data)}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            print("❌ Migration validation failed!")
    else:
        print("❌ Migration failed!")

if __name__ == "__main__":
    main() 