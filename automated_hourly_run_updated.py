"""
Automated Hourly Data Run - Clean Version (NEW GROUP ONLY)
------------------------------------------------------------
This script is designed to be run automatically (e.g., by a GitHub Action)
on an hourly schedule.

It will:
1. Fetch the most recent air quality and weather data (current hour).
2. Engineer features for this new data using the NEW group logic.
3. Push the resulting features to the NEW feature group only.
"""
import logging
import os
from config import HOPSWORKS_CONFIG, OPENWEATHER_CONFIG
from data_collector import collect_current_data_with_iqair, retry_on_network_error
from feature_engineering import AQIFeatureEngineer
from hopsworks_integration import HopsworksUploader
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/automated_run_clean.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_tolerance(period: int, feature_type: str = 'lag'):
    """
    Get tolerance based on period and feature type with custom specifications
    """
    from datetime import timedelta
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

def create_time_based_lag(df: pd.DataFrame, target: str, lag_hours: int) -> pd.Series:
    """
    Create time-based lag feature that finds data approximately lag_hours ago
    """
    import numpy as np
    from datetime import timedelta
    
    lag_series = pd.Series(index=df.index, dtype=float)
    tolerance = get_tolerance(lag_hours, 'lag')
    
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

def create_time_based_rolling(df: pd.DataFrame, target: str, window_hours: int, stat_type: str) -> pd.Series:
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

def fetch_existing_hopsworks_data(uploader, group_name):
    """
    Fetch existing data from Hopsworks feature group.
    Returns DataFrame with existing data or empty DataFrame if none exists.
    """
    try:
        logger.info(f"Fetching existing data from feature group: {group_name}")
        
        # Get the feature group
        fg = uploader.fs.get_feature_group(
            name=group_name,
            version=1  # Use version 1 as specified in hopsworks_integration.py
        )
        
        # Read all data from the feature group
        df = fg.read()
        
        if df.empty:
            logger.info("No existing data found in Hopsworks feature group.")
            return pd.DataFrame()
        
        # Fix timestamp handling - ensure proper datetime index
        logger.info("Fixing timestamp handling for existing data...")
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
            logger.error("No 'time' column found and index is not datetime. Cannot proceed.")
            return pd.DataFrame()
        
        # Remove any rows with invalid timestamps
        invalid_timestamps = df.index.isna()
        if invalid_timestamps.any():
            logger.warning(f"Found {invalid_timestamps.sum()} rows with invalid timestamps. Removing them.")
            df = df[~invalid_timestamps]
        
        if df.empty:
            logger.warning("No valid data remaining after timestamp cleanup.")
            return pd.DataFrame()
        
        logger.info(f"Successfully fetched {len(df)} existing records from Hopsworks.")
        logger.info(f"Existing data date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Existing data columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.warning(f"Could not fetch existing data from Hopsworks: {e}")
        logger.info("Will proceed with only new data (lag features will be NaN for first few records).")
        return pd.DataFrame()

def create_clean_features_with_context(raw_df, existing_df):
    """
    Create clean features with historical context from NEW group
    """
    logger.info("Creating clean features with historical context from NEW group...")
    
    # Sort existing data by timestamp to ensure proper lag feature calculation
    if not existing_df.empty:
        existing_df = existing_df.sort_index()
        logger.info("Sorted existing data by timestamp for proper lag feature calculation")
    
    # Combine existing and new data for proper feature engineering
    if not existing_df.empty:
        logger.info("Combining existing and new data for proper feature engineering...")
        
        # Extract raw features from existing data for combination
        raw_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
                       'carbon_monoxide', 'no', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide', 
                       'pm2_5', 'pm10', 'nh3', 'openweather_aqi', 'pm2_5_aqi', 'pm10_aqi', 'us_aqi',
                       'city', 'latitude', 'longitude']
        
        # Get only raw features from existing data
        available_raw_features = [col for col in raw_features if col in existing_df.columns]
        existing_raw_df = existing_df[available_raw_features].copy()
        
        logger.info(f"Extracted {len(available_raw_features)} raw features from existing data")
        
        # Combine raw data (existing + new)
        combined_raw_df = pd.concat([existing_raw_df, raw_df], axis=0)
        
        # Sort by timestamp
        combined_raw_df = combined_raw_df.sort_index()
        
        logger.info(f"Combined dataset: {len(existing_raw_df)} existing + {len(raw_df)} new = {len(combined_raw_df)} total records")
        logger.info(f"Date range: {combined_raw_df.index.min()} to {combined_raw_df.index.max()}")
        
    else:
        logger.info("No existing data found. Using only new data...")
        combined_raw_df = raw_df
    
    # Use feature_engineering.py for consistency
    engineer = AQIFeatureEngineer()
    engineered_df = engineer.engineer_features(combined_raw_df)
    
    if engineered_df.empty:
        logger.error("Feature engineering failed")
        return None
    
    # Add wind direction engineering (not in feature_engineering.py) - only if not already present
    if 'wind_direction' in combined_raw_df.columns:
        if 'wind_direction_sin' not in engineered_df.columns:
            import numpy as np
            wind_dir = combined_raw_df['wind_direction']
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
            logger.info("Added wind direction engineering features")
        else:
            logger.info("Wind direction features already present in existing data - preserving them")
    
    # Add pollutant lags (not in feature_engineering.py)
    # Handle both column name formats
    co_col = 'carbon_monoxide' if 'carbon_monoxide' in combined_raw_df.columns else 'co'
    o3_col = 'ozone' if 'ozone' in combined_raw_df.columns else 'o3'
    so2_col = 'sulphur_dioxide' if 'sulphur_dioxide' in combined_raw_df.columns else 'so2'
    
    # Add pollutant lags using exact time-based formulas
    engineered_df['co_lag_1h'] = create_time_based_lag(combined_raw_df, co_col, 1)
    engineered_df['o3_lag_1h'] = create_time_based_lag(combined_raw_df, o3_col, 1)
    engineered_df['so2_lag_1h'] = create_time_based_lag(combined_raw_df, so2_col, 1)
    
    # Add weather lags (NEW FEATURES) using exact time-based formulas
    engineered_df['temp_lag_1h'] = create_time_based_lag(combined_raw_df, 'temperature', 1)
    engineered_df['wind_speed_lag_1h'] = create_time_based_lag(combined_raw_df, 'wind_speed', 1)
    engineered_df['humidity_lag_1h'] = create_time_based_lag(combined_raw_df, 'humidity', 1)
    engineered_df['pressure_lag_1h'] = create_time_based_lag(combined_raw_df, 'pressure', 1)
    
    # Add ozone lag 3h (NEW FEATURE) using exact time-based formula
    engineered_df['ozone_lag_3h'] = create_time_based_lag(combined_raw_df, o3_col, 3)
    
    # Add weather rolling features (NEW FEATURES) using exact time-based formulas
    # 3h rolling features
    engineered_df['temp_rolling_mean_3h'] = create_time_based_rolling(combined_raw_df, 'temperature', 3, 'mean')
    engineered_df['humidity_rolling_mean_3h'] = create_time_based_rolling(combined_raw_df, 'humidity', 3, 'mean')
    engineered_df['wind_speed_rolling_mean_3h'] = create_time_based_rolling(combined_raw_df, 'wind_speed', 3, 'mean')
    engineered_df['pressure_rolling_mean_3h'] = create_time_based_rolling(combined_raw_df, 'pressure', 3, 'mean')
    
    # 12h rolling features
    engineered_df['temp_rolling_mean_12h'] = create_time_based_rolling(combined_raw_df, 'temperature', 12, 'mean')
    engineered_df['humidity_rolling_mean_12h'] = create_time_based_rolling(combined_raw_df, 'humidity', 12, 'mean')
    engineered_df['wind_speed_rolling_mean_12h'] = create_time_based_rolling(combined_raw_df, 'wind_speed', 12, 'mean')
    engineered_df['pressure_rolling_mean_12h'] = create_time_based_rolling(combined_raw_df, 'pressure', 12, 'mean')
    
    # Add ozone rolling features (NEW FEATURES) using exact time-based formulas
    engineered_df['ozone_rolling_mean_3h'] = create_time_based_rolling(combined_raw_df, o3_col, 3, 'mean')
    engineered_df['ozone_rolling_mean_12h'] = create_time_based_rolling(combined_raw_df, o3_col, 12, 'mean')
    
    # Add new interactions (not in feature_engineering.py) - only if not already present
    if 'wind_direction_temp_interaction' not in engineered_df.columns:
        engineered_df['wind_direction_temp_interaction'] = engineered_df['wind_direction_sin'] * combined_raw_df['temperature']
        engineered_df['wind_direction_humidity_interaction'] = engineered_df['wind_direction_sin'] * combined_raw_df['humidity']
        engineered_df['pressure_humidity_interaction'] = combined_raw_df['pressure'] * combined_raw_df['humidity']
        engineered_df['co_pressure_interaction'] = combined_raw_df[co_col] * combined_raw_df['pressure']
        engineered_df['o3_temp_interaction'] = combined_raw_df[o3_col] * combined_raw_df['temperature']
        engineered_df['so2_humidity_interaction'] = combined_raw_df[so2_col] * combined_raw_df['humidity']
        logger.info("Added new interaction features")
    else:
        logger.info("New interaction features already present in existing data - preserving them")
    
    # Add PM × weather interactions (from feature_engineering.py + new ones) - only if not already present
    if 'pm2_5_pressure_interaction' not in engineered_df.columns:
        # These 2 come from feature_engineering.py (already created by engineer.engineer_features())
        # pm2_5_temp_interaction and pm2_5_humidity_interaction are already in engineered_df
        
        # Add the 3 new PM × weather interactions
        engineered_df['pm2_5_pressure_interaction'] = combined_raw_df['pm2_5'] * combined_raw_df['pressure']
        engineered_df['pm10_temperature_interaction'] = combined_raw_df['pm10'] * combined_raw_df['temperature']
        engineered_df['pm10_pressure_interaction'] = combined_raw_df['pm10'] * combined_raw_df['pressure']
        logger.info("Added PM × weather interaction features")
    else:
        logger.info("PM × weather interaction features already present in existing data - preserving them")
    
    # Select only the 87 clean features we want (including raw wind_direction)
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
        
        # Lag features (1h, 2h, 3h, 24h)
        'pm2_5_lag_1h', 'pm2_5_lag_2h', 'pm2_5_lag_3h', 'pm2_5_lag_24h',
        'pm10_lag_1h', 'pm10_lag_2h', 'pm10_lag_3h', 'pm10_lag_24h',
        
        # Pollutant lags
        'co_lag_1h', 'o3_lag_1h', 'so2_lag_1h',
        
        # Weather lags (NEW FEATURES)
        'temp_lag_1h', 'wind_speed_lag_1h', 'humidity_lag_1h', 'pressure_lag_1h',
        
        # Ozone lag 3h (NEW FEATURE)
        'ozone_lag_3h',
        
        # Rolling features (optimized)
        'pm2_5_rolling_min_3h', 'pm2_5_rolling_mean_3h', 'pm2_5_rolling_max_3h', 'pm2_5_rolling_min_12h', 'pm2_5_rolling_mean_12h', 'pm2_5_rolling_max_12h', 'pm2_5_rolling_mean_24h', 'pm2_5_rolling_max_24h',
        'pm10_rolling_min_3h', 'pm10_rolling_mean_3h', 'pm10_rolling_mean_12h', 'pm10_rolling_mean_24h',
        
        # Weather rolling features (NEW FEATURES)
        'temp_rolling_mean_3h', 'humidity_rolling_mean_3h', 'wind_speed_rolling_mean_3h', 'pressure_rolling_mean_3h',
        'temp_rolling_mean_12h', 'humidity_rolling_mean_12h', 'wind_speed_rolling_mean_12h', 'pressure_rolling_mean_12h',
        
        # Ozone rolling features (NEW FEATURES)
        'ozone_rolling_mean_3h', 'ozone_rolling_mean_12h',
        
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
        
        # PM × weather interactions (5 features) - HIGHLY PREDICTIVE
        'pm2_5_temp_interaction', 'pm2_5_humidity_interaction', 'pm2_5_pressure_interaction',
        'pm10_temperature_interaction', 'pm10_pressure_interaction',
        
        # Pollutant-weather interactions
        'co_pressure_interaction', 'o3_temp_interaction', 'so2_humidity_interaction'
    ]
    
    # Select available features
    available_features = [f for f in clean_features if f in engineered_df.columns]
    clean_df = engineered_df[available_features].copy()
    
    # Create time_str for Hopsworks
    clean_df['time_str'] = clean_df['time'].dt.floor('H').dt.strftime('%Y-%m-%d %H:%M:%S')
    
    logger.info(f"Clean features created: {len(clean_df.columns)} columns")
    logger.info(f"📊 Expected: 87 features (72 + 15 new weather/ozone features)")
    return clean_df

@retry_on_network_error(max_retries=1, delay=300)  # 1 retry, 5 minute delay
def run_hourly_update():
    """
    Executes the hourly data update pipeline for NEW feature group only.
    """
    logger.info("=======================================")
    logger.info("=== STARTING AUTOMATED HOURLY UPDATE (NEW GROUP ONLY) ===")
    logger.info("=======================================")
    
    # Debug: Check API keys are loaded
    logger.info("DEBUG: Checking API keys...")
    openweather_key = os.getenv("OPENWEATHER_API_KEY")
    iqair_key = os.getenv("IQAIR_API_KEY")
    hopsworks_key = os.getenv("HOPSWORKS_API_KEY")
    
    logger.info(f"DEBUG: OPENWEATHER_API_KEY present: {'Yes' if openweather_key else 'No'}")
    logger.info(f"DEBUG: IQAIR_API_KEY present: {'Yes' if iqair_key else 'No'}")
    logger.info(f"DEBUG: HOPSWORKS_API_KEY present: {'Yes' if hopsworks_key else 'No'}")
    
    if not openweather_key:
        logger.error("ERROR: OPENWEATHER_API_KEY environment variable is not set!")
    if not iqair_key:
        logger.warning("WARNING: IQAIR_API_KEY environment variable is not set!")
    if not hopsworks_key:
        logger.error("ERROR: HOPSWORKS_API_KEY environment variable is not set!")

    # 1. Initialize Hopsworks connection first
    logger.info("STEP 1: Connecting to Hopsworks...")
    uploader = HopsworksUploader(
        api_key=os.getenv("HOPSWORKS_API_KEY", HOPSWORKS_CONFIG.get('api_key')),
        project_name=HOPSWORKS_CONFIG['project_name']
    )
    if not uploader.connect():
        logger.error("Could not connect to Hopsworks. Aborting.")
        return False

    # 2. Fetch existing data from NEW feature group for lag/rolling features
    logger.info("STEP 2: Fetching existing data from NEW feature group...")
    existing_df = fetch_existing_hopsworks_data(uploader, HOPSWORKS_CONFIG['feature_group_name'])
    
    # 3. Collect Current Data (OpenWeather + IQAir AQI for comparison) - USING EXISTING INFRASTRUCTURE
    logger.info("STEP 3: Collecting current data (OpenWeather + IQAir AQI for comparison)...")
    raw_df_with_iqair = collect_current_data_with_iqair()  # USING EXISTING FUNCTION!
    if raw_df_with_iqair.empty:
        logger.warning("No new data collected in the last hour. Exiting gracefully.")
        return True
    logger.info(f"Successfully collected {len(raw_df_with_iqair)} new records.")
    
    # Create copy for Hopsworks (without IQAir columns to match historic schema)
    raw_df = raw_df_with_iqair.copy()
    columns_to_remove = ['iqair_aqi', 'abs_deviation']
    for col in columns_to_remove:
        if col in raw_df.columns:
            raw_df = raw_df.drop(columns=[col])
            logger.info(f"Removed column '{col}' for Hopsworks to match historic schema")

    # Explicitly cast all numerics to float64
    numeric_cols = raw_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        raw_df[col] = raw_df[col].astype('float64')
    logger.info(f"Casted {len(numeric_cols)} numeric columns to float64 for Hopsworks consistency")

    # Rename columns to match feature engineering expectations
    rename_map = {
        'co': 'carbon_monoxide',
        'no2': 'nitrogen_dioxide',
        'o3': 'ozone',
        'so2': 'sulphur_dioxide'
    }
    raw_df = raw_df.rename(columns=rename_map)
    
    # Ensure 'no' column exists (nitric oxide from OpenWeather API)
    if 'no' not in raw_df.columns:
        logger.warning("'no' column missing from API response. Adding with NaN values.")
        raw_df['no'] = None

    # Save AQI validation data as CSV (for your reference only)
    import datetime
    now = datetime.datetime.utcnow()
    validation_path = f"data/aqi_validation_current_{now.strftime('%Y%m%d_%H')}.csv"
    os.makedirs("data", exist_ok=True)
    # Extract only relevant validation columns (use DataFrame with IQAir data)
    raw_df_reset = raw_df_with_iqair.reset_index()
    validation_cols = ['time', 'openweather_aqi', 'us_aqi', 'iqair_aqi', 'abs_deviation']
    available_cols = [col for col in validation_cols if col in raw_df_reset.columns]
    validation_df = raw_df_reset[available_cols].copy()
    validation_df.to_csv(validation_path, index=False)
    logger.info(f"Created new hourly AQI validation file: {validation_path}")

    # 4. Create CLEAN features with historical context from NEW group
    logger.info("\nSTEP 4: Creating CLEAN features with historical context from NEW group...")
    clean_features = create_clean_features_with_context(raw_df, existing_df)
    
    if clean_features is None:
        logger.error("Feature engineering failed")
        return False

    # 5. Push to NEW feature group only
    logger.info("\nSTEP 5: Pushing to NEW feature group only...")
    
    # Push to NEW feature group
    logger.info(f"Pushing to NEW feature group ({HOPSWORKS_CONFIG['feature_group_name']})...")
    success = uploader.push_features(
        df=clean_features,
        group_name=HOPSWORKS_CONFIG['feature_group_name'],
        description="Clean, optimized features for AQI prediction (NEW GROUP - 87 features). Target: pm2_5 (raw concentration), AQI: us_aqi (calculated from PM2.5)."
    )
    
    if success:
        logger.info("✅ Successfully pushed data to NEW feature group!")
        logger.info(f"✨ Clean features: {len(clean_features.columns)} columns (87 expected)")
        return True
    else:
        logger.error("❌ Failed to push data to NEW feature group!")
        return False

def main():
    """
    Main function to run hourly data collection
    """
    print("🚀 AQI Data Collection - NEW GROUP ONLY")
    print("=" * 60)
    print("📊 Writing to NEW feature group only")
    print("🔄 Old group deprecated - clean and optimized")
    print("=" * 60)
    
    # Run collection
    success = run_hourly_update()
    
    if success:
        print("🎉 Collection cycle completed!")
        print("📊 Next steps:")
        print("   1. Monitor NEW feature group for data quality")
        print("   2. Validate clean features are working correctly")
        print("   3. Test model performance with optimized features")
        print("   4. Old group is deprecated - no longer used")
    else:
        print("❌ Collection cycle failed!")
    
    return success

if __name__ == "__main__":
    main() 