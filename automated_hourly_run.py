"""
Automated Hourly Data Run
--------------------------
This script is designed to be run automatically (e.g., by a GitHub Action)
on an hourly schedule.

It will:
1. Fetch the most recent air quality and weather data (current hour).
2. Engineer features for this new data.
3. Push the resulting features to the Hopsworks feature group to keep it
   up-to-date.
"""
import logging
import os
from config import HOPSWORKS_CONFIG
from data_collector import collect_current_data_with_iqair, retry_on_network_error
from feature_engineering import AQIFeatureEngineer
from hopsworks_integration import HopsworksUploader
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/automated_run.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

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
        logger.info(f"Sample of existing data lag features:")
        lag_columns = [col for col in df.columns if 'lag' in col]
        if lag_columns:
            sample_lags = df[lag_columns].head(3)
            logger.info(f"First 3 rows of lag features:\n{sample_lags}")
        
        # Debug: Check timestamp format
        logger.info(f"Timestamp sample (first 3): {list(df.index[:3])}")
        logger.info(f"Timestamp type: {type(df.index)}")
        
        return df
        
    except Exception as e:
        logger.warning(f"Could not fetch existing data from Hopsworks: {e}")
        logger.info("Will proceed with only new data (lag features will be NaN for first few records).")
        return pd.DataFrame()

@retry_on_network_error(max_retries=1, delay=300)  # 1 retry, 5 minute delay
def run_hourly_update():
    """
    Executes the hourly data update pipeline.
    """
    logger.info("=======================================")
    logger.info("=== STARTING AUTOMATED HOURLY UPDATE ===")
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

    # 2. Fetch existing data from Hopsworks for lag/rolling features
    logger.info("STEP 2: Fetching existing data from Hopsworks...")
    existing_df = fetch_existing_hopsworks_data(uploader, HOPSWORKS_CONFIG['feature_group_name'])
    
    # Sort existing data by timestamp to ensure proper lag feature calculation
    if not existing_df.empty:
        existing_df = existing_df.sort_index()
        logger.info("Sorted existing data by timestamp for proper lag feature calculation")
    
    # 3. Collect Current Data (OpenWeather + IQAir AQI for comparison)
    logger.info("STEP 3: Collecting current data (OpenWeather + IQAir AQI for comparison)...")
    raw_df_with_iqair = collect_current_data_with_iqair()
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

    # 4. Combine existing and new data for proper feature engineering
    logger.info("\nSTEP 4: Combining existing and new data...")
    if not existing_df.empty:
        # The existing data already has all engineered features from the manual run
        # We need to engineer features for new data and combine properly
        
        logger.info("Processing new data with existing historical context...")
        
        # Ensure both DataFrames have the same index type
        logger.info("Ensuring consistent index types...")
        if isinstance(existing_df.index, pd.DatetimeIndex):
            raw_df.index = pd.to_datetime(raw_df.index)
        elif isinstance(raw_df.index, pd.DatetimeIndex):
            existing_df.index = pd.to_datetime(existing_df.index)
        
        # Remove any new data that already exists in Hopsworks
        existing_timestamps = set(existing_df.index)
        new_timestamps = set(raw_df.index)
        duplicate_timestamps = new_timestamps.intersection(existing_timestamps)
        
        if duplicate_timestamps:
            logger.warning(f"Found {len(duplicate_timestamps)} duplicate timestamps. Removing duplicates.")
            raw_df = raw_df[~raw_df.index.isin(duplicate_timestamps)]
        
        if raw_df.empty:
            logger.warning("No new unique data to add after removing duplicates.")
            return True
        
        # Add location metadata to new data
        raw_df["city"] = raw_df["city"].iloc[0] if "city" in raw_df.columns else "Multan"
        raw_df["latitude"] = raw_df["latitude"].iloc[0] if "latitude" in raw_df.columns else 30.1575
        raw_df["longitude"] = raw_df["longitude"].iloc[0] if "longitude" in raw_df.columns else 71.5249
        
        # Combine existing and new data BEFORE feature engineering
        logger.info(f"Combining existing data ({len(existing_df)} records) with new raw data ({len(raw_df)} records)")
        logger.info(f"Existing data columns: {list(existing_df.columns)}")
        logger.info(f"New raw data columns: {list(raw_df.columns)}")
        
        # Extract raw features from existing data for combination
        logger.info("Extracting raw features from existing data...")
        raw_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
                       'carbon_monoxide', 'no', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide', 
                       'pm2_5', 'pm10', 'nh3', 'openweather_aqi', 'pm2_5_aqi', 'pm10_aqi', 'us_aqi',
                       'city', 'latitude', 'longitude']
        
        # Get only raw features from existing data
        available_raw_features = [col for col in raw_features if col in existing_df.columns]
        existing_raw_df = existing_df[available_raw_features].copy()
        
        logger.info(f"Extracted {len(available_raw_features)} raw features from existing data")
        logger.info(f"Existing raw data shape: {existing_raw_df.shape}")
        
        # Combine raw data (existing + new)
        combined_raw_df = pd.concat([existing_raw_df, raw_df], axis=0)
        
        # Debug: Check timestamps before processing
        logger.info("DEBUG: Checking timestamps before processing...")
        logger.info(f"Existing data timestamp sample: {list(existing_raw_df.index[:3])}")
        logger.info(f"New data timestamp sample: {list(raw_df.index[:3])}")
        logger.info(f"Combined data timestamp sample: {list(combined_raw_df.index[:5])}")
        
        # Robust timezone fix: always convert to UTC, then to naive
        logger.info("Fixing timezone consistency for sorting...")
        combined_raw_df.index = pd.to_datetime(combined_raw_df.index, utc=True, errors='coerce')
        if isinstance(combined_raw_df.index, pd.DatetimeIndex) and combined_raw_df.index.tz is not None:
            logger.info("Converting timezone-aware timestamps to timezone-naive...")
            combined_raw_df.index = combined_raw_df.index.tz_localize(None)
        combined_raw_df = combined_raw_df.sort_index()  # Sort by time
        
        # Debug: Check timestamps after processing
        logger.info("DEBUG: Checking timestamps after processing...")
        logger.info(f"Final combined data timestamp sample: {list(combined_raw_df.index[:5])}")
        logger.info(f"Timestamp type: {type(combined_raw_df.index)}")
        logger.info(f"Any NaT timestamps: {combined_raw_df.index.isna().any()}")
        
        logger.info(f"Combined raw dataset: {len(existing_raw_df)} existing + {len(raw_df)} new = {len(combined_raw_df)} total records")
        logger.info(f"Combined raw data shape: {combined_raw_df.shape}")
        logger.info(f"Date range: {combined_raw_df.index.min()} to {combined_raw_df.index.max()}")
        logger.info(f"Sample timestamps: {list(combined_raw_df.index[-5:])}")  # Last 5 timestamps
        
        # Engineer features on the complete combined raw dataset
        logger.info("Engineering features on complete combined raw dataset...")
        engineer = AQIFeatureEngineer()
        engineered_df = engineer.engineer_features(combined_raw_df)
        
        if engineered_df.empty:
            logger.error("Feature engineering on combined dataset resulted in an empty DataFrame. Aborting.")
            return False
        
        logger.info(f"Successfully re-engineered features on combined dataset: {engineered_df.shape[1]} features.")
        
    else:
        # No existing data, just engineer features for the new data
        logger.info("No existing data found. Engineering features for new data...")
        engineer = AQIFeatureEngineer()
        engineered_df = engineer.engineer_features(raw_df)
        
        if engineered_df.empty:
            logger.error("Feature engineering resulted in an empty DataFrame. Aborting.")
            return False
        
        logger.info(f"Successfully engineered {engineered_df.shape[1]} features.")

    # 5. Push to Hopsworks
    logger.info("\nSTEP 5: Pushing updated features to Hopsworks...")
    
    success = uploader.push_features(
        df=engineered_df,
        group_name=HOPSWORKS_CONFIG['feature_group_name'],
        description="Hourly update of PM2.5 and PM10 prediction features for Multan. Targets: pm2_5, pm10 (raw concentrations), Reference: us_aqi (final AQI)."
    )
    
    if not success:
        logger.error("Failed to push features to Hopsworks.")
        return False

    logger.info("\n=============================================")
    logger.info("=== AUTOMATED HOURLY RUN COMPLETED SUCCESSFULLY ===")
    logger.info("=============================================")
    return True

if __name__ == "__main__":
    if not run_hourly_update():
        logger.error("The automated hourly update process failed. Check logs for details.")
        exit(1) 