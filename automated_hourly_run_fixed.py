"""
Automated Hourly Data Run - FIXED VERSION
-----------------------------------------
This script is designed to be run automatically (e.g., by a GitHub Action)
on an hourly schedule.

It will:
1. Fetch the most recent air quality and weather data (current hour).
2. Fetch existing data from Hopsworks to get historical context.
3. Append new data to existing data.
4. Re-engineer features for the combined dataset (ensuring proper lag/rolling features).
5. Push the updated features back to Hopsworks.
"""
import logging
import os
from config import HOPSWORKS_CONFIG
from data_collector import collect_current_data_with_iqair
from feature_engineering import AQIFeatureEngineer
from hopsworks_integration import HopsworksUploader
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("automated_run_fixed.log"),
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
        fg = uploader.fs.get_or_create_feature_group(
            name=group_name,
            version=None  # Get the latest version
        )
        
        # Read all data from the feature group
        df = fg.read()
        
        if df.empty:
            logger.info("No existing data found in Hopsworks feature group.")
            return pd.DataFrame()
        
        logger.info(f"Successfully fetched {len(df)} existing records from Hopsworks.")
        return df
        
    except Exception as e:
        logger.warning(f"Could not fetch existing data from Hopsworks: {e}")
        logger.info("Will proceed with only new data (lag features will be NaN for first few records).")
        return pd.DataFrame()

def run_hourly_update_fixed():
    """
    Executes the hourly data update pipeline with proper lag/rolling feature handling.
    """
    logger.info("=======================================")
    logger.info("=== STARTING FIXED HOURLY UPDATE ===")
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

    # Initialize Hopsworks connection first
    logger.info("STEP 1: Connecting to Hopsworks...")
    uploader = HopsworksUploader(
        api_key=os.getenv("HOPSWORKS_API_KEY", HOPSWORKS_CONFIG.get('api_key')),
        project_name=HOPSWORKS_CONFIG['project_name']
    )
    if not uploader.connect():
        logger.error("Could not connect to Hopsworks. Aborting.")
        return False

    # 2. Fetch existing data from Hopsworks
    logger.info("STEP 2: Fetching existing data from Hopsworks...")
    existing_df = fetch_existing_hopsworks_data(uploader, HOPSWORKS_CONFIG['feature_group_name'])
    
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

    # Rename columns to match feature engineering expectations
    rename_map = {
        'co': 'carbon_monoxide',
        'no2': 'nitrogen_dioxide',
        'o3': 'ozone',
        'so2': 'sulphur_dioxide'
    }
    raw_df = raw_df.rename(columns=rename_map)

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

    # 4. Combine existing and new data
    logger.info("STEP 4: Combining existing and new data...")
    if not existing_df.empty:
        # If we have existing data, we need to handle the schema properly
        # The existing data will have all engineered features, but new data only has raw features
        
        # For the new data, we'll engineer features separately and then combine
        logger.info("Engineering features for new data only...")
        engineer = AQIFeatureEngineer()
        new_engineered_df = engineer.engineer_features(raw_df)
        
        if new_engineered_df.empty:
            logger.error("Feature engineering for new data resulted in an empty DataFrame. Aborting.")
            return False
        
        logger.info(f"Successfully engineered {new_engineered_df.shape[1]} features for new data.")
        
        # Now combine the existing engineered data with new engineered data
        # We need to ensure the new data doesn't duplicate existing timestamps
        existing_timestamps = set(existing_df.index)
        new_timestamps = set(new_engineered_df.index)
        
        # Remove any new data that already exists in Hopsworks
        duplicate_timestamps = new_timestamps.intersection(existing_timestamps)
        if duplicate_timestamps:
            logger.warning(f"Found {len(duplicate_timestamps)} duplicate timestamps. Removing duplicates.")
            new_engineered_df = new_engineered_df[~new_engineered_df.index.isin(duplicate_timestamps)]
        
        if new_engineered_df.empty:
            logger.warning("No new unique data to add after removing duplicates.")
            return True
        
        # Combine existing and new data
        combined_df = pd.concat([existing_df, new_engineered_df], axis=0)
        combined_df = combined_df.sort_index()  # Sort by time
        
        logger.info(f"Combined dataset: {len(existing_df)} existing + {len(new_engineered_df)} new = {len(combined_df)} total records")
        
    else:
        # No existing data, just use the new data
        logger.info("No existing data found. Using only new data.")
        combined_df = raw_df

    # 5. Re-engineer features on the combined dataset (if we had existing data)
    if not existing_df.empty:
        logger.info("STEP 5: Re-engineering features on combined dataset...")
        engineer = AQIFeatureEngineer()
        combined_df = engineer.engineer_features(combined_df)
        
        if combined_df.empty:
            logger.error("Feature engineering on combined dataset resulted in an empty DataFrame. Aborting.")
            return False
        
        logger.info(f"Successfully re-engineered features on combined dataset: {combined_df.shape[1]} features.")
    else:
        # If no existing data, the new data was already engineered above
        logger.info("STEP 5: Using already engineered new data (no existing data to combine).")

    # 6. Push updated data to Hopsworks
    logger.info("STEP 6: Pushing updated features to Hopsworks...")
    
    success = uploader.push_features(
        df=combined_df,
        group_name=HOPSWORKS_CONFIG['feature_group_name'],
        description="Hourly update of PM2.5 and PM10 prediction features for Multan. Targets: pm2_5, pm10 (raw concentrations), Reference: us_aqi (final AQI)."
    )
    
    if not success:
        logger.error("Failed to push features to Hopsworks.")
        return False

    logger.info("\n=============================================")
    logger.info("=== FIXED HOURLY RUN COMPLETED SUCCESSFULLY ===")
    logger.info("=============================================")
    return True

if __name__ == "__main__":
    if not run_hourly_update_fixed():
        logger.error("The fixed hourly update process failed. Check logs for details.")
        exit(1) 