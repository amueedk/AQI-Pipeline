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
from data_collector import collect_current_data_with_iqair
from feature_engineering import AQIFeatureEngineer
from hopsworks_integration import HopsworksUploader
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("automated_run.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

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

    # 1. Collect Current Data (OpenWeather + IQAir AQI for comparison)
    logger.info("STEP 1: Collecting current data (OpenWeather + IQAir AQI for comparison)...")
    raw_df_with_iqair = collect_current_data_with_iqair()
    if raw_df_with_iqair.empty:
        logger.warning("No new data collected in the last hour. Exiting gracefully.")
        return True
    logger.info(f"Successfully collected {len(raw_df_with_iqair)} new records.")
    
    # Create copy for Hopsworks (without IQAir columns to match historic schema)
    raw_df = raw_df_with_iqair.copy()
    columns_to_remove = ['iqair_aqi', 'abs_deviation', 'openweather_aqi']
    for col in columns_to_remove:
        if col in raw_df.columns:
            raw_df = raw_df.drop(columns=[col])
            logger.info(f"Removed column '{col}' for Hopsworks to match historic schema")

    # Explicitly cast all numerics to float64
    numeric_cols = raw_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        raw_df[col] = raw_df[col].astype('float64')
    logger.info(f"Casted {len(numeric_cols)} numeric columns to float64 for Hopsworks consistency")

    # Save AQI validation data as CSV (only AQI data for comparison)
    import datetime
    daily_date = datetime.datetime.utcnow().strftime('%Y%m%d')
    validation_path = f"data/aqi_validation_current_{daily_date}.csv"
    os.makedirs("data", exist_ok=True)
    
    # Extract only AQI validation columns (use DataFrame with IQAir data)
    raw_df_reset = raw_df_with_iqair.reset_index()
    validation_cols = ['time', 'openweather_aqi', 'us_aqi', 'iqair_aqi', 'abs_deviation']
    # Only include columns that exist
    available_cols = [col for col in validation_cols if col in raw_df_reset.columns]
    logger.info(f"DEBUG: Available columns in DataFrame: {list(raw_df_reset.columns)}")
    logger.info(f"DEBUG: Validation columns requested: {validation_cols}")
    logger.info(f"DEBUG: Validation columns available: {available_cols}")
    validation_df = raw_df_reset[available_cols].copy()
    
    # Check if file exists and append, or create new file
    if os.path.exists(validation_path):
        try:
            existing_df = pd.read_csv(validation_path, index_col=0, parse_dates=True)
            # Append new data
            combined_df = pd.concat([existing_df, validation_df])
            # Remove duplicates based on timestamp (keep the latest)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            # Sort by timestamp
            combined_df = combined_df.sort_index()
            combined_df.to_csv(validation_path)
            logger.info(f"Appended to existing daily AQI validation file: {validation_path}")
        except Exception as e:
            logger.warning(f"Error reading existing daily AQI validation file: {e}. Creating new file.")
            validation_df.to_csv(validation_path)
            logger.info(f"Created new daily AQI validation file: {validation_path}")
    else:
        validation_df.to_csv(validation_path)
        logger.info(f"Created new daily AQI validation file: {validation_path}")

    # 2. Engineer Features
    logger.info("\nSTEP 2: Engineering features for new data...")
    logger.info(f"DEBUG: Columns being sent to feature engineering: {list(raw_df.columns)}")
    engineer = AQIFeatureEngineer()
    engineered_df = engineer.engineer_features(raw_df)
    if engineered_df.empty:
        logger.error("Feature engineering resulted in an empty DataFrame. Aborting.")
        return False
    logger.info(f"Successfully engineered {engineered_df.shape[1]} features.")

    # 3. Push to Hopsworks
    logger.info("\nSTEP 3: Pushing new features to Hopsworks...")
    uploader = HopsworksUploader(
        api_key=os.getenv("HOPSWORKS_API_KEY", HOPSWORKS_CONFIG.get('api_key')),
        project_name=HOPSWORKS_CONFIG['project_name']
    )
    if not uploader.connect():
        logger.error("Could not connect to Hopsworks. Aborting.")
        return False
    
    success = uploader.push_features(
        df=engineered_df,
        group_name=HOPSWORKS_CONFIG['feature_group_name'],
        description="Hourly update of AQI and weather features for Multan."
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