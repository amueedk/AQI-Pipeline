import os
import pandas as pd
import hopsworks
from config import PATHS, HOPSWORKS_CONFIG

VERSION = 1

def save_features_to_hopsworks():
    api_key = HOPSWORKS_CONFIG["api_key"]
    project_name = HOPSWORKS_CONFIG["project_name"]
    feature_group_name = HOPSWORKS_CONFIG["feature_group_name"]

    # Connect to Hopsworks
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()

    # Load features
    features_file = os.path.join(PATHS['data_dir'], 'master_features.csv')
    if not os.path.exists(features_file):
        print(f"❌ Features file not found: {features_file}")
        print("Please run feature_engineering.py first to generate features.")
        return False
    
    df = pd.read_csv(features_file, index_col=0, parse_dates=True)
    if 'time' not in df.columns and df.index.name == 'time':
        df = df.reset_index()

    print(f"📄 Loaded {len(df)} records with {len(df.columns)} features")
    print("📊 Preview:")
    print(df.head())

    # Always use get_or_create_feature_group (never try to get first)
    print(f"⚙️ Getting or creating feature group: {feature_group_name}")
    fg = fs.get_or_create_feature_group(
        name=feature_group_name,
        version=VERSION,
        description="AQI engineered features for Multan",
        primary_key=["time"],
        event_time="time",
        online_enabled=True
    )
    
    if fg is None:
        print("❌ Feature group creation failed. Check your schema, keys, or project.")
        return False
    
    print(f"✅ Feature group ready: {fg.name} (v{fg.version})")

    # Insert the features
    try:
        fg.insert(df, write_options={"wait_for_job": True})
        print(f"✅ Inserted {len(df)} records into '{fg.name}' (v{fg.version})")
        return True
    except Exception as e:
        print(f"❌ Insert error: {e}")
        return False

if __name__ == "__main__":
    if not save_features_to_hopsworks():
        exit(1) 