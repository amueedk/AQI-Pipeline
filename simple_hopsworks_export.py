"""
Simple Hopsworks Export
-----------------------
This script tries different methods to read data from a potentially corrupted feature group.
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
        logging.FileHandler("simple_export.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def simple_export():
    """Try different methods to export data from Hopsworks"""
    
    print("Simple Hopsworks Export")
    print("=" * 40)
    print("Trying different methods to read data...")
    
    try:
        # Initialize connection
        logger.info("Connecting to Hopsworks...")
        uploader = HopsworksUploader(
            api_key=HOPSWORKS_CONFIG['api_key'],
            project_name=HOPSWORKS_CONFIG['project_name']
        )
        
        if not uploader.connect():
            logger.error("Failed to connect to Hopsworks")
            return
        
        # Get feature group
        logger.info("Getting feature group...")
        fg = uploader.fs.get_or_create_feature_group(
            name=HOPSWORKS_CONFIG['feature_group_name'],
            version=1
        )
        
        # Try different reading methods
        methods_to_try = [
            ("read()", lambda: fg.read()),
            ("read(online=False)", lambda: fg.read(online=False)),
            ("read(online=True)", lambda: fg.read(online=True)),
            ("read_all()", lambda: fg.read_all()),
            ("read_all(online=False)", lambda: fg.read_all(online=False)),
            ("read_all(online=True)", lambda: fg.read_all(online=True))
        ]
        
        for method_name, method_func in methods_to_try:
            try:
                logger.info(f"Trying method: {method_name}")
                df = method_func()
                
                if df is not None and not df.empty:
                    logger.info(f"✅ Success with {method_name}")
                    logger.info(f"Data shape: {df.shape}")
                    
                    # Save the data
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"hopsworks_data_{method_name.replace('()', '').replace('(', '_').replace(')', '').replace(', ', '_')}_{timestamp}.csv"
                    
                    df.to_csv(filename, index=False)
                    logger.info(f"✅ Data saved to: {filename}")
                    
                    # Show temperature stats
                    if 'temp' in df.columns:
                        logger.info(f"Temperature range: {df['temp'].min()} to {df['temp'].max()}")
                        logger.info(f"Temperature mean: {df['temp'].mean():.2f}")
                    
                    return filename
                    
            except Exception as e:
                logger.warning(f"❌ Method {method_name} failed: {str(e)[:100]}...")
                continue
        
        logger.error("❌ All methods failed to read data")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return None
    finally:
        try:
            uploader.close()
        except:
            pass

if __name__ == "__main__":
    result = simple_export()
    if result:
        print(f"\n✅ Success! Data exported to: {result}")
    else:
        print("\n❌ Failed to export data. Feature group may be corrupted.")
        print("Recommendation: Delete the feature group and recreate it with the good data CSV.") 