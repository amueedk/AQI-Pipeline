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