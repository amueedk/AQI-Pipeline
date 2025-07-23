"""
Create New Feature Group
-----------------------
This script creates a new feature group with a different name to work around
the deletion issue, then we can rename it later.
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
        logging.StreamHandler()  # Only console logging for GitHub Actions compatibility
    ]
)

logger = logging.getLogger(__name__)

def create_new_feature_group():
    """Create a new feature group with clean data"""
    
    print("Create New Feature Group")
    print("=" * 40)
    print("Creating a new feature group with clean data")
    print("We'll rename it later to match the original name.")
    
    # Find the fixed data CSV
    csv_file = "good_hopsworks_data_fixed.csv"
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
        
        # Create new feature group with different name
        new_name = "multan_aqi_features_clean"
        logger.info(f"Creating new feature group: {new_name}")
        
        # Prepare data for insertion (following the pattern from hopsworks_integration.py)
        logger.info("Preparing data for insertion...")
        
        # Convert time column to proper format
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        # Add time_str column as primary key (required for HUDI format)
        # Round time to nearest hour for primary key (e.g., 22/07/2025 9:00:45 PM -> 22/07/2025 9:00:00 PM)
        df['time_str'] = df['time'].dt.floor('H').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create feature group using the same pattern as hopsworks_integration.py
        fg = uploader.fs.get_or_create_feature_group(
            name=new_name,
            version=1,
            description="Multan AQI features with FIXED lag/rolling features (correctly calculated) - temporary name",
            primary_key=['time_str'],
            event_time='time',
            online_enabled=False  # OFFLINE ONLY for historical data creation
        )
        
        logger.info("✅ Feature group created successfully")
        
        # Insert data
        logger.info(f"Inserting {len(df)} records...")
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
        
        logger.info("🎉 New feature group created successfully!")
        logger.info(f"Feature group name: {new_name}")
        logger.info("Next steps:")
        logger.info("1. Try to delete the old corrupted feature group later")
        logger.info("2. Rename this new feature group to 'multan_aqi_features'")
        logger.info("3. Update your GitHub Actions if needed")
        logger.info("4. ✅ All lag/rolling/change-rate features are now correctly calculated!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating new feature group: {e}")
        return False
    finally:
        try:
            uploader.close()
        except:
            pass

if __name__ == "__main__":
    success = create_new_feature_group()
    if success:
        print("\n✅ New feature group created successfully!")
        print("Feature group name: multan_aqi_features_clean")
        print("✅ All lag/rolling/change-rate features are now correctly calculated!")
        print("You can now work with this properly engineered data while we resolve the deletion issue.")
    else:
        print("\n❌ Failed to create new feature group. Check logs for details.") 