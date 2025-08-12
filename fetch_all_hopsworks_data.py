"""
Fetch All Data from Hopsworks
-----------------------------
This script fetches ALL data from the Hopsworks feature store exactly as it is stored.
No filtering, no modifications - just raw data export.
"""
import os
import pandas as pd
from datetime import datetime
import logging
from hopsworks_integration import HopsworksUploader
from config import HOPSWORKS_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fetch_all_data.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def fetch_all_hopsworks_data():
    """
    Fetch ALL data from Hopsworks feature store exactly as it is stored.
    """
    logger.info("=========================================")
    logger.info("=== FETCHING ALL HOPSWORKS DATA ===")
    logger.info("=========================================")
    
    # Check API key
    hopsworks_key = os.getenv("HOPSWORKS_API_KEY")
    if not hopsworks_key:
        logger.error("ERROR: HOPSWORKS_API_KEY environment variable is not set!")
        return None
    
    logger.info("✓ Hopsworks API key found")
    
    # Connect to Hopsworks
    logger.info("Connecting to Hopsworks...")
    uploader = HopsworksUploader(
        api_key=hopsworks_key,
        project_name=HOPSWORKS_CONFIG['project_name']
    )
    
    if not uploader.connect():
        logger.error("Failed to connect to Hopsworks!")
        return None
    
    logger.info("✓ Connected to Hopsworks successfully")
    
    # Get the feature store
    try:
        fs = uploader.project.get_feature_store()
        logger.info("✓ Feature store accessed")
    except Exception as e:
        logger.error(f"Failed to get feature store: {e}")
        return None
    
    # Get the feature group
    try:
        fg = fs.get_or_create_feature_group(
            name='multan_aqi_features_clean_2',
            version=1
        )
        logger.info("✓ Feature group accessed")
    except Exception as e:
        logger.error(f"Failed to get feature group: {e}")
        return None
    
    # Fetch ALL data
    logger.info("Fetching ALL data from feature group...")
    try:
        # Get all data without any filters
        df = fg.read()
        logger.info(f"✓ Successfully fetched {len(df)} records")
        logger.info(f"✓ Data shape: {df.shape}")
        
        # Show basic info
        logger.info(f"✓ Columns: {list(df.columns)}")
        logger.info(f"✓ Date range: {df.index.min()} to {df.index.max()}")
        print(df['time_str'].head())
        
    except Exception as e:
        logger.error(f"Failed to read data from feature group: {e}")
        return None
    
    # Save to CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_hopsworks_data_{timestamp}.csv"
    
    try:
        # Reset index to make timestamp a column
        df_reset = df.reset_index()
        df_reset.to_csv(filename, index=False)
        logger.info(f"✓ Data saved to: {filename}")
        
        # Also save a summary
        summary_filename = f"data_summary_{timestamp}.txt"
        with open(summary_filename, 'w') as f:
            f.write("=== HOPSWORKS DATA SUMMARY ===\n\n")
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Date Range: {df.index.min()} to {df.index.max()}\n")
            f.write(f"Columns: {len(df.columns)}\n")
            f.write(f"Column Names: {list(df.columns)}\n\n")
            
            f.write("=== DATA TYPES ===\n")
            f.write(str(df.dtypes) + "\n\n")
            
            f.write("=== BASIC STATISTICS ===\n")
            f.write(str(df.describe()) + "\n\n")
            
            f.write("=== FIRST 10 ROWS ===\n")
            f.write(str(df.head(10)) + "\n\n")
            
            f.write("=== LAST 10 ROWS ===\n")
            f.write(str(df.tail(10)) + "\n")
            

        
        logger.info(f"✓ Summary saved to: {summary_filename}")
        
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        return None
    
    logger.info("\n=========================================")
    logger.info("=== DATA FETCH COMPLETED SUCCESSFULLY ===")
    logger.info("=========================================")
    
    return df

if __name__ == "__main__":
    df = fetch_all_hopsworks_data()
    if df is not None:
        print(f"\nData fetched successfully!")
        print(f"Records: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
    else:
        print("\nFailed to fetch data. Check logs for details.")
        exit(1) 