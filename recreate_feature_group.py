"""
Recreate Feature Group
---------------------
This script completely deletes the corrupted feature group and recreates it
with the proper schema and clean data.
"""

import os
import pandas as pd
import logging
from datetime import datetime
from hopsworks_integration import HopsworksUploader
from config import HOPSWORKS_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recreate_feature_group.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def recreate_feature_group():
    """Delete and recreate the feature group with clean data"""
    
    print("Recreate Feature Group")
    print("=" * 40)
    print("This will completely delete and recreate the feature group")
    print("with the proper schema and clean data.")
    
    # Find the good data CSV
    csv_file = "good_hopsworks_data_20250716_204528.csv"
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return False
    
    try:
        # Load the data
        logger.info(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Validate temperature data
        if 'temperature' in df.columns:
            temp_range = df['temperature'].agg(['min', 'max', 'mean'])
            logger.info(f"Temperature range: {temp_range['min']:.2f} to {temp_range['max']:.2f}°C (mean: {temp_range['mean']:.2f}°C)")
            
            if temp_range['max'] > 100:
                logger.warning("⚠️  Found temperatures > 100°C - this might be Kelvin data!")
                return False
        
        # Initialize Hopsworks connection
        logger.info("Connecting to Hopsworks...")
        uploader = HopsworksUploader(
            api_key=HOPSWORKS_CONFIG['api_key'],
            project_name=HOPSWORKS_CONFIG['project_name']
        )
        
        if not uploader.connect():
            logger.error("Failed to connect to Hopsworks")
            return False
        
        # Delete the existing feature group
        logger.info("Deleting existing feature group...")
        try:
            fg = uploader.fs.get_or_create_feature_group(
                name=HOPSWORKS_CONFIG['feature_group_name'],
                version=1
            )
            fg.delete()
            logger.info("✅ Feature group deleted successfully")
        except Exception as e:
            logger.warning(f"Could not delete feature group: {e}")
            logger.info("Feature group may already be deleted or corrupted")
        
        # Wait a moment for deletion to complete
        import time
        time.sleep(2)
        
        # Create new feature group with proper schema
        logger.info("Creating new feature group with proper schema...")
        
        # Create feature group with automatic schema detection
        logger.info("Creating new feature group with automatic schema...")
        
        # Create feature group - Hopsworks will auto-detect schema from data
        fg = uploader.fs.create_feature_group(
            name=HOPSWORKS_CONFIG['feature_group_name'],
            version=1,
            description="Multan AQI features with clean data (Celsius temperatures only)",
            primary_keys=[],
            partition_keys=[],
            online_enabled=True,
            statistics_config=None
        )
        
        logger.info("✅ Feature group created successfully")
        
        # Prepare data for insertion
        logger.info("Preparing data for insertion...")
        
        # Convert time column to proper format
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        # Insert data to create the schema
        logger.info(f"Inserting {len(df)} records to create schema...")
        fg.insert(df)
        
        logger.info("✅ Data inserted successfully!")
        
        # Verify the insertion
        logger.info("Verifying data insertion...")
        try:
            read_df = fg.read()
            logger.info(f"✅ Verification successful: {len(read_df)} records read back")
            
            if 'temperature' in read_df.columns:
                temp_stats = read_df['temperature'].agg(['min', 'max', 'mean'])
                logger.info(f"Temperature in Hopsworks: {temp_stats['min']:.2f} to {temp_stats['max']:.2f}°C")
            
        except Exception as e:
            logger.warning(f"Could not verify data: {e}")
        
        logger.info("🎉 Feature group recreation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error recreating feature group: {e}")
        return False
    finally:
        try:
            uploader.close()
        except:
            pass

if __name__ == "__main__":
    success = recreate_feature_group()
    if success:
        print("\n✅ Feature group recreated successfully!")
        print("Your GitHub Actions should continue working normally.")
    else:
        print("\n❌ Failed to recreate feature group. Check logs for details.") 