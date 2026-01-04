import os
import sys
import pandas as pd

# Add project root to path to find modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from hopsworks_integration import HopsworksUploader
from config import HOPSWORKS_CONFIG

u = HopsworksUploader(api_key=os.getenv("HOPSWORKS_API_KEY"), project_name=HOPSWORKS_CONFIG["project_name"])
assert u.connect()
fs = u.project.get_feature_store()
fg = fs.get_feature_group(name="aqi_weather_forecasts", version=1)
df = fg.read()
df = df.toPandas() if hasattr(df, "toPandas") else df
df.to_csv("aqi_weather_forecasts_offline_full.csv", index=False)