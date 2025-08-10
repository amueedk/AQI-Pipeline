import pandas as pd
import numpy as np
import hopsworks
from datetime import datetime, timedelta
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
            # Read from the updated source feature group
            old_fg = self.fs.get_feature_group("multan_aqi_features_clean_2", version=1)
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
            
            # 5. LAG FEATURES (8 features) - 1h, 2h, 3h, 24h
            'pm2_5_lag_1h', 'pm2_5_lag_2h', 'pm2_5_lag_3h', 'pm2_5_lag_24h',
            'pm10_lag_1h', 'pm10_lag_2h', 'pm10_lag_3h', 'pm10_lag_24h',
            
                    # 6. POLLUTANT LAGS (3 features) - NEW
        'co_lag_1h', 'o3_lag_1h', 'so2_lag_1h',
        
        # 6a. WEATHER LAGS (4 features) - NEW
        'temp_lag_1h', 'wind_speed_lag_1h', 'humidity_lag_1h', 'pressure_lag_1h',
        
        # 6b. OZONE LAG 3H (1 feature) - NEW
        'ozone_lag_3h',
            
                    # 7. ROLLING FEATURES (12 features) - OPTIMIZED
        'pm2_5_rolling_min_3h', 'pm2_5_rolling_mean_3h', 'pm2_5_rolling_max_3h', 'pm2_5_rolling_min_12h', 'pm2_5_rolling_mean_12h', 'pm2_5_rolling_max_12h', 'pm2_5_rolling_mean_24h', 'pm2_5_rolling_max_24h',
        'pm10_rolling_min_3h', 'pm10_rolling_mean_3h', 'pm10_rolling_mean_12h', 'pm10_rolling_mean_24h',
        
        # 7a. WEATHER ROLLING FEATURES (8 features) - NEW
        'temp_rolling_mean_3h', 'humidity_rolling_mean_3h', 'wind_speed_rolling_mean_3h', 'pressure_rolling_mean_3h',
        'temp_rolling_mean_12h', 'humidity_rolling_mean_12h', 'wind_speed_rolling_mean_12h', 'pressure_rolling_mean_12h',
        
        # 7b. OZONE ROLLING FEATURES (2 features) - NEW
        'ozone_rolling_mean_3h', 'ozone_rolling_mean_12h',
            
            # 8. CHANGE RATES (6 features)
            'pm2_5_change_rate_1h', 'pm2_5_change_rate_6h', 'pm2_5_change_rate_24h',
            'pm10_change_rate_1h', 'pm10_change_rate_6h', 'pm10_change_rate_24h',
            
            # 9. CYCLICAL TIME FEATURES (8 features)
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            
            # 10. BINARY FEATURES (6 features) - OPTIMIZED
            'is_hot', 'is_night', 'is_morning_rush', 'is_evening_rush', 'is_high_pm2_5', 'is_high_o3',
            
            # 11. INTERACTION FEATURES (11 features) - OPTIMIZED
            'temp_humidity_interaction', 'temp_wind_interaction', 'wind_direction_temp_interaction', 
            'wind_direction_humidity_interaction', 'pressure_humidity_interaction',
            
            # 12. PM × WEATHER INTERACTIONS (5 features) - HIGHLY PREDICTIVE
            'pm2_5_temp_interaction', 'pm2_5_humidity_interaction', 'pm2_5_pressure_interaction',
            'pm10_temperature_interaction', 'pm10_pressure_interaction',
            
            # 13. POLLUTANT-WEATHER INTERACTIONS (3 features) - NEW
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
        
        # Add pollutant lags (not in feature_engineering.py) - using exact time-based formulas
        print("🧪 Adding pollutant lags using time-based formulas...")
        engineered_df['co_lag_1h'] = self._create_time_based_lag(df, 'carbon_monoxide', 1)
        engineered_df['o3_lag_1h'] = self._create_time_based_lag(df, 'ozone', 1)
        engineered_df['so2_lag_1h'] = self._create_time_based_lag(df, 'sulphur_dioxide', 1)
        
        # Add weather lags (NEW FEATURES) - using exact time-based formulas
        print("🌤️ Adding weather lags using time-based formulas...")
        engineered_df['temp_lag_1h'] = self._create_time_based_lag(df, 'temperature', 1)
        engineered_df['wind_speed_lag_1h'] = self._create_time_based_lag(df, 'wind_speed', 1)
        engineered_df['humidity_lag_1h'] = self._create_time_based_lag(df, 'humidity', 1)
        engineered_df['pressure_lag_1h'] = self._create_time_based_lag(df, 'pressure', 1)
        
        # Add ozone lag 3h (NEW FEATURE) - using exact time-based formula
        engineered_df['ozone_lag_3h'] = self._create_time_based_lag(df, 'ozone', 3)
        
        # Add new interactions (not in feature_engineering.py)
        print("🔗 Adding new interaction features...")
        engineered_df['wind_direction_temp_interaction'] = engineered_df['wind_direction_sin'] * df['temperature']
        engineered_df['wind_direction_humidity_interaction'] = engineered_df['wind_direction_sin'] * df['humidity']
        engineered_df['pressure_humidity_interaction'] = df['pressure'] * df['humidity']
        engineered_df['co_pressure_interaction'] = df['carbon_monoxide'] * df['pressure']
        engineered_df['o3_temp_interaction'] = df['ozone'] * df['temperature']
        engineered_df['so2_humidity_interaction'] = df['sulphur_dioxide'] * df['humidity']
        
        # Add weather rolling features (NEW FEATURES) - using exact time-based formulas
        print("🌤️ Adding weather rolling features using time-based formulas...")
        # 3h rolling features
        engineered_df['temp_rolling_mean_3h'] = self._create_time_based_rolling(df, 'temperature', 3, 'mean')
        engineered_df['humidity_rolling_mean_3h'] = self._create_time_based_rolling(df, 'humidity', 3, 'mean')
        engineered_df['wind_speed_rolling_mean_3h'] = self._create_time_based_rolling(df, 'wind_speed', 3, 'mean')
        engineered_df['pressure_rolling_mean_3h'] = self._create_time_based_rolling(df, 'pressure', 3, 'mean')
        
        # 12h rolling features
        engineered_df['temp_rolling_mean_12h'] = self._create_time_based_rolling(df, 'temperature', 12, 'mean')
        engineered_df['humidity_rolling_mean_12h'] = self._create_time_based_rolling(df, 'humidity', 12, 'mean')
        engineered_df['wind_speed_rolling_mean_12h'] = self._create_time_based_rolling(df, 'wind_speed', 12, 'mean')
        engineered_df['pressure_rolling_mean_12h'] = self._create_time_based_rolling(df, 'pressure', 12, 'mean')
        
        # Add ozone rolling features (NEW FEATURES) - using exact time-based formulas
        engineered_df['ozone_rolling_mean_3h'] = self._create_time_based_rolling(df, 'ozone', 3, 'mean')
        engineered_df['ozone_rolling_mean_12h'] = self._create_time_based_rolling(df, 'ozone', 12, 'mean')
        
        # Add PM × weather interactions (from feature_engineering.py + new ones)
        print("🔥 Adding PM × weather interaction features...")
        # These 2 come from feature_engineering.py (already created by engineer.engineer_features())
        # pm2_5_temp_interaction and pm2_5_humidity_interaction are already in engineered_df
        
        # Add the 3 new PM × weather interactions
        engineered_df['pm2_5_pressure_interaction'] = df['pm2_5'] * df['pressure']
        engineered_df['pm10_temperature_interaction'] = df['pm10'] * df['temperature']
        engineered_df['pm10_pressure_interaction'] = df['pm10'] * df['pressure']
        
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
        print(f"📊 Expected: 87 features (72 + 15 new weather/ozone features)")
        
        return clean_df
    
    def _get_tolerance(self, period: int, feature_type: str = 'lag') -> timedelta:
        """
        Get tolerance based on period and feature type with custom specifications
        """
        if feature_type == 'lag' or feature_type == 'change_rate':
            # Custom tolerance specifications
            if period == 1:
                tolerance_hours = 0.5  # ±30min for 1h
            elif period == 2:
                tolerance_hours = 0.75  # ±45min for 2h
            elif period == 3:
                tolerance_hours = 50/60  # ±50min for 3h
            else:
                # All other periods (6h, 12h, 24h, 48h, 72h) get 1h tolerance
                tolerance_hours = 1.0
            return timedelta(hours=tolerance_hours)
        elif feature_type == 'rolling':
            # Rolling features don't use tolerance anyway (uses ALL data in window)
            tolerance_hours = period * 0.25
            return timedelta(hours=tolerance_hours)
    
    def _create_time_based_lag(self, df: pd.DataFrame, target: str, lag_hours: int) -> pd.Series:
        """
        Create time-based lag feature that finds data approximately lag_hours ago
        """
        import numpy as np
        from datetime import timedelta
        
        lag_series = pd.Series(index=df.index, dtype=float)
        tolerance = self._get_tolerance(lag_hours, 'lag')
        
        for i, current_time in enumerate(df.index):
            target_time = current_time - timedelta(hours=lag_hours)
            
            # Find data within acceptable range
            acceptable_range_start = target_time - tolerance
            acceptable_range_end = target_time + tolerance
            
            # Use pandas boolean indexing for efficiency
            mask = (df.index >= acceptable_range_start) & (df.index <= acceptable_range_end)
            matching_data = df[mask][target]
            
            if len(matching_data) > 0:
                # Use the closest data point to target_time
                matching_indices = df.index[mask]
                time_diffs = [(idx - target_time).total_seconds() for idx in matching_indices]
                min_diff_idx = np.argmin(np.abs(time_diffs))
                closest_idx = matching_indices[min_diff_idx]
                
                # Handle potential duplicate timestamps by ensuring scalar value
                value = df.loc[closest_idx, target]
                lag_series.iloc[i] = value if np.isscalar(value) else value.iloc[0]
            else:
                lag_series.iloc[i] = np.nan
        
        return lag_series
    
    def _create_time_based_rolling(self, df: pd.DataFrame, target: str, window_hours: int, stat_type: str) -> pd.Series:
        """
        Create time-based rolling statistics using ALL data in the time window
        """
        import numpy as np
        from datetime import timedelta
        
        rolling_series = pd.Series(index=df.index, dtype=float)
        
        for i, current_time in enumerate(df.index):
            window_start = current_time - timedelta(hours=window_hours)
            
            # Get ALL data in the window (no tolerance checks)
            window_mask = (df.index >= window_start) & (df.index <= current_time)
            window_data = df[window_mask][target]
            
            if len(window_data) < 2:
                rolling_series.iloc[i] = np.nan
                continue
            
            # Calculate rolling statistic with all available data
            if stat_type == 'mean':
                rolling_series.iloc[i] = window_data.mean()
            elif stat_type == 'std':
                rolling_series.iloc[i] = window_data.std()
            elif stat_type == 'min':
                rolling_series.iloc[i] = window_data.min()
            elif stat_type == 'max':
                rolling_series.iloc[i] = window_data.max()
        
        return rolling_series
    
    def read_new_feature_group(self):
        """
        Read existing data from new feature group to check for duplicates
        """
        try:
            new_fg = self.fs.get_feature_group("aqi_clean_features_v2", version=1)
            new_data = new_fg.read()
            print(f"✅ Read {len(new_data)} rows from new feature group")
            return new_data
        except Exception as e:
            print(f"⚠️ Could not read new feature group (may not exist yet): {e}")
            return pd.DataFrame()

    def filter_new_data_only(self, old_data, new_data):
        """
        Filter old data to only include rows that don't exist in new data
        """
        if new_data.empty:
            print("📊 No existing data in new group - will migrate all old data")
            return old_data
        
        # Ensure both have time_str for comparison
        if 'time_str' not in old_data.columns:
            old_data['time_str'] = old_data.index.dt.floor('H').dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'time_str' not in new_data.columns:
            new_data['time_str'] = new_data.index.dt.floor('H').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Find timestamps that exist in new data
        existing_timestamps = set(new_data['time_str'].dropna())
        
        # Filter old data to only include new timestamps
        old_data_with_time = old_data.copy()
        if 'time_str' not in old_data_with_time.columns:
            old_data_with_time['time_str'] = old_data_with_time.index.dt.floor('H').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        new_only_data = old_data_with_time[~old_data_with_time['time_str'].isin(existing_timestamps)]
        
        print(f"📊 Migration analysis:")
        print(f"   Old data rows: {len(old_data)}")
        print(f"   New data rows: {len(new_data)}")
        print(f"   Existing timestamps in new: {len(existing_timestamps)}")
        print(f"   New rows to migrate: {len(new_only_data)}")
        
        return new_only_data

    def migrate_historical_data(self):
        """
        Complete migration process - only migrates NEW data
        """
        print("🚀 Starting historical data migration (NEW DATA ONLY)...")
        
        # Check if connection was successful
        if self.fs is None:
            print("❌ Not connected to Hopsworks. Cannot proceed.")
            return None
        
        # 1. Read old data
        old_data = self.read_old_feature_group()
        if old_data is None:
            return None
        
        # 2. Read existing new data to check for duplicates
        new_data = self.read_new_feature_group()
        
        # 3. Filter to only new data
        new_only_data = self.filter_new_data_only(old_data, new_data)
        
        if new_only_data.empty:
            print("✅ No new data to migrate - all data already exists in new group")
            return None
        
        # 4. Transform to clean features
        clean_data = self.transform_to_clean_features(new_only_data)
        
        # 5. Create new feature group (if doesn't exist)
        new_fg = self.create_new_feature_group()
        if new_fg is None:
            return None
        
        # 6. Store only new data in new feature group
        try:
            new_fg.insert(clean_data, write_options={"wait_for_job": True})
            print(f"✅ Successfully migrated {len(clean_data)} NEW rows to new feature group")
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
        expected_features = 87  # Updated: 72 + 15 new weather/ozone features
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