import os
import pandas as pd
import hopsworks
from config import PATHS, HOPSWORKS_CONFIG

def save_features_to_hopsworks():
    api_key = HOPSWORKS_CONFIG["api_key"]
    project_name = HOPSWORKS_CONFIG["project_name"]
    feature_group_name = HOPSWORKS_CONFIG["feature_group_name"]

    # Connect to Hopsworks
    project = hopsworks.login(api_key_value=api_key).get_project(project_name)
    fs = project.get_feature_store()

    # Load engineered features
    features_file = os.path.join(PATHS['data_dir'], 'master_features.csv')
    df = pd.read_csv(features_file, index_col=0, parse_dates=True)
    if 'time' not in df.columns and df.index.name == 'time':
        df = df.reset_index()

    # Create or get feature group
    try:
        fg = fs.get_feature_group(name=feature_group_name, version=1)
    except:
        fg = fs.create_feature_group(
            name=feature_group_name,
            version=1,
            description="AQI engineered features for Multan",
            primary_keys=["time"],
            online_enabled=True
        )

    # Insert data
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"✅ Successfully saved {len(df)} records to Hopsworks feature group '{feature_group_name}'.")

if __name__ == "__main__":
    save_features_to_hopsworks() 