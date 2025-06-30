import os
import pandas as pd
import hopsworks
from config import PATHS, HOPSWORKS_CONFIG

def save_features_to_hopsworks():
    api_key = HOPSWORKS_CONFIG["api_key"]
    project_name = HOPSWORKS_CONFIG["project_name"]
    feature_group_name = HOPSWORKS_CONFIG["feature_group_name"]

    # Connect to Hopsworks - login now returns Project directly
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()

    # Load engineered features
    features_file = os.path.join(PATHS['data_dir'], 'master_features.csv')
    if not os.path.exists(features_file):
        print(f"❌ Features file not found: {features_file}")
        print("Please run feature_engineering.py first to generate features.")
        return False
    
    df = pd.read_csv(features_file, index_col=0, parse_dates=True)
    if 'time' not in df.columns and df.index.name == 'time':
        df = df.reset_index()
    
    print(f"Loaded {len(df)} records with {len(df.columns)} features")

    # Try to get existing feature group first
    try:
        fg = fs.get_feature_group(name=feature_group_name, version=1)
        print(f"✅ Found existing feature group: {feature_group_name}")
    except:
        print(f"Creating new feature group: {feature_group_name}")
        
        # Create feature group with proper parameters
        fg = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=1,
            description="AQI engineered features for Multan",
            primary_keys=["time"],
            event_time="time",
            online_enabled=True
        )
        
        # Verify feature group was created successfully
        if fg is None:
            print("❌ Failed to create feature group. This might be due to:")
            print("   - Missing primary key or event time")
            print("   - Schema mismatch")
            print("   - Insufficient permissions")
            return False
        
        print(f"✅ Created new feature group: {feature_group_name}")

    # Verify feature group exists before inserting
    if fg is None:
        print("❌ Feature group is None. Cannot insert data.")
        return False
    
    print(f"Feature group ready: {fg.name} (version {fg.version})")
    
    try:
        # Insert data with proper error handling
        fg.insert(df, write_options={"wait_for_job": True})
        print(f"✅ Successfully saved {len(df)} records to Hopsworks feature group '{feature_group_name}'.")
        return True
    except Exception as e:
        print(f"❌ Error inserting data: {e}")
        return False

if __name__ == "__main__":
    success = save_features_to_hopsworks()
    if not success:
        exit(1) 