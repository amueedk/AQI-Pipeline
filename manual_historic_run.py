"""
Manual Historic Data Run
--------------------------
This script is intended to be run ONCE to populate the Hopsworks feature store
with historical data.

It will:
1. Fetch the last 14 days of air quality and weather data.
2. Engineer a comprehensive set of features from this data.
3. Push the resulting feature DataFrame to the Hopsworks feature group.
"""
import logging
import os
from config import HOPSWORKS_CONFIG
from data_collector import OpenWeatherDataCollector
from feature_engineering import AQIFeatureEngineer
from hopsworks_integration import HopsworksUploader
import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("manual_run.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_manual_backfill():
    """
    Executes the full historical data backfill pipeline.
    """
    logger.info("=================================================")
    logger.info("=== STARTING MANUAL HISTORICAL DATA BACKFILL ===")
    logger.info("=================================================")
    
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

    # 1. Collect Historical Data
    logger.info("STEP 1: Collecting historical data for the last 14 days (OpenWeather only)...")
    collector = OpenWeatherDataCollector()
    raw_df = collector.collect_historical_data(days_back=14)
    if raw_df.empty:
        logger.error("Data collection failed. No data to process. Aborting.")
        return False
    logger.info(f"Successfully collected {len(raw_df)} records.")

    # Save AQI validation data as CSV (only AQI data for comparison)
    start = (datetime.datetime.utcnow() - datetime.timedelta(days=14)).strftime('%Y%m%d')
    end = datetime.datetime.utcnow().strftime('%Y%m%d')
    
    # Extract only AQI validation columns
    raw_df_reset = raw_df.reset_index()
    validation_cols = ['time', 'openweather_aqi', 'us_aqi']
    validation_df = raw_df_reset[validation_cols].copy()
    
    validation_path = f"data/aqi_validation_historic_{start}_to_{end}.csv"
    os.makedirs("data", exist_ok=True)
    validation_df.to_csv(validation_path)
    logger.info(f"AQI validation data saved to {validation_path}")

    # 2. Engineer Features
    logger.info("\nSTEP 2: Engineering features...")
    engineer = AQIFeatureEngineer()
    engineered_df = engineer.engineer_features(raw_df)
    if engineered_df.empty:
        logger.error("Feature engineering resulted in an empty DataFrame. Aborting.")
        return False
    logger.info(f"Successfully engineered {engineered_df.shape[1]} features.")

    # 3. Push to Hopsworks
    logger.info("\nSTEP 3: Pushing features to Hopsworks...")
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
        description="Historical backfill of AQI and weather features for Multan."
    )

    if not success:
        logger.error("Failed to push features to Hopsworks.")
        return False

    logger.info("\n=============================================")
    logger.info("=== MANUAL HISTORICAL RUN COMPLETED SUCCESSFULLY ===")
    logger.info("=============================================")
    return True

if __name__ == "__main__":
    if not run_manual_backfill():
        logger.error("The manual backfill process failed. Check logs for details.")
        exit(1) 