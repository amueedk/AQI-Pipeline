import os
import pandas as pd
import hopsworks
import logging
from config import HOPSWORKS_CONFIG
from typing import Optional
from functools import wraps
import time
import random

# Configure logging
logger = logging.getLogger(__name__)

# This version will be incremented if we change the feature group schema
FEATURE_GROUP_VERSION = 1

def retry_on_hopsworks_error(max_retries=2, delay=120, backoff_factor=2.0):
    """
    Retry decorator specifically for Hopsworks operations
    
    Args:
        max_retries: Maximum number of retry attempts (default: 2)
        delay: Initial delay in seconds (default: 120 = 2 minutes)
        backoff_factor: Multiplier for delay on each retry (default: 2.0)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"❌ {func.__name__} failed after {max_retries} retries. Final error: {e}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    current_delay = delay * (backoff_factor ** attempt)
                    
                    logger.warning(f"⚠️ {func.__name__} attempt {attempt + 1} failed: {e}")
                    logger.info(f"🔄 Retrying Hopsworks operation in {current_delay:.1f} seconds...")
                    time.sleep(current_delay)
            
            # This should never be reached, but just in case
            raise last_exception
        return wrapper
    return decorator

class HopsworksUploader:
    def __init__(self, api_key: str, project_name: str):
        self.api_key = api_key
        self.project_name = project_name
        self.project = None
        self.fs = None

    @retry_on_hopsworks_error()
    def connect(self) -> bool:
        """Connects to the Hopsworks project."""
        if not self.api_key:
            logger.error("Hopsworks API key is not set. Please set the HOPSWORKS_API_KEY environment variable.")
            return False
        try:
            logger.info(f"Logging into Hopsworks project: {self.project_name}...")
            self.project = hopsworks.login(
                api_key_value=self.api_key, 
                project=self.project_name
            )
            self.fs = self.project.get_feature_store()
            logger.info("Successfully connected to Hopsworks Feature Store.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}")
            return False

    @retry_on_hopsworks_error()
    def push_features(self, df: pd.DataFrame, group_name: str, description: str) -> bool:
        """
        Pushes an engineered features DataFrame to a feature group in Hopsworks.
        """
        if self.fs is None:
            logger.error("Not connected to Hopsworks. Please call connect() first.")
            return False
            
        if df.empty:
            logger.warning("Input DataFrame is empty. Nothing to push to Hopsworks.")
            return True # Not a failure, but nothing to do

        # Prepare the DataFrame for Hopsworks
        df_to_insert = df.copy()
        if 'time' not in df_to_insert.columns:
             if isinstance(df_to_insert.index, pd.DatetimeIndex):
                 df_to_insert.reset_index(inplace=True)
             else:
                 logger.error("DataFrame must have a 'time' column or a DatetimeIndex.")
                 return False

        # Ensure event_time column is correct format
        df_to_insert['time'] = pd.to_datetime(df_to_insert['time'])
        
        # Round time to nearest hour for primary key (e.g., 22/07/2025 9:00:45 PM -> 22/07/2025 9:00:00 PM)
        df_to_insert['time_rounded'] = df_to_insert['time'].dt.floor('H')
        
        # Primary key must be unique and is required for online feature store
        df_to_insert['time_str'] = df_to_insert['time_rounded'].dt.strftime('%Y-%m-%d %H:%M:%S')

        try:
            # Get or create the feature group
            fg = self.fs.get_or_create_feature_group(
                name=group_name,
                version=FEATURE_GROUP_VERSION,
                description=description,
                primary_key=['time_str'],
                event_time='time',
                online_enabled=True  # Enable online storage for real-time app predictions
            )

            logger.info(f"Inserting {len(df_to_insert)} rows into feature group '{group_name}'...")
            fg.insert(df_to_insert, write_options={"wait_for_job": True})
            logger.info("Successfully inserted data into Hopsworks.")
            return True
        except Exception as e:
            logger.error(f"Failed to insert data into Hopsworks: {e}")
            return False



def main():
    """
    Main function to test Hopsworks integration.
    This will:
    1. Fetch a small amount of historical data.
    2. Engineer features.
    3. Push those features to Hopsworks.
    """
    from data_collector import OpenWeatherDataCollector
    from feature_engineering import AQIFeatureEngineer

    logging.basicConfig(level=logging.INFO)

    # 1. Fetch data
    logger.info("--- Hopsworks Test: Fetching data ---")
    collector = OpenWeatherDataCollector()
    raw_df = collector.collect_historical_data(days_back=2)
    if raw_df.empty:
        logger.error("--- Hopsworks Test: Failed to collect data. Aborting. ---")
        return

    # 2. Engineer features
    logger.info("--- Hopsworks Test: Engineering features ---")
    engineer = AQIFeatureEngineer()
    engineered_df = engineer.engineer_features(raw_df)
    if engineered_df.empty:
        logger.error("--- Hopsworks Test: Failed to engineer features. Aborting. ---")
        return

    # 3. Push to Hopsworks
    logger.info("--- Hopsworks Test: Pushing features to Hopsworks ---")
    uploader = HopsworksUploader(
        api_key=HOPSWORKS_CONFIG['api_key'],
        project_name=HOPSWORKS_CONFIG['project_name']
    )
    if uploader.connect():
        success = uploader.push_features(
            df=engineered_df,
            group_name=HOPSWORKS_CONFIG['feature_group_name'],
            description="PM2.5 prediction features for Multan. Target: pm2_5 (raw concentration), AQI: us_aqi (calculated from PM2.5)."
        )
        if success:
            logger.info("--- Hopsworks Test: Successfully completed. ---")
        else:
            logger.error("--- Hopsworks Test: Failed to push features. ---")

if __name__ == "__main__":
    main() 