import pandas as pd
import numpy as np
from feature_engineering import AQIFeatureEngineer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the data
df = pd.read_csv('good_hopsworks_data_20250716_204528.csv')

# Convert time column to datetime and set as index
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Test the feature engineering pipeline step by step
engineer = AQIFeatureEngineer()

print("\n=== Testing lag features creation ===")
# Test lag features before any missing value handling
lag_df = engineer.create_lag_features(df.copy())
print(f"PM2.5 lag 1h (first 10): {lag_df['pm2_5_lag_1h'].head(10).values}")
print(f"PM2_5 lag 1h (last 10): {lag_df['pm2_5_lag_1h'].tail(10).values}")

print("\n=== Testing missing value handling ===")
# Test what happens during missing value handling
missing_df = engineer.handle_missing_values(lag_df.copy())
print(f"After missing value handling - PM2.5 lag 1h (first 10): {missing_df['pm2_5_lag_1h'].head(10).values}")

print("\n=== Testing full pipeline ===")
# Test the full pipeline
full_df = engineer.engineer_features(df.copy())
print(f"Full pipeline - PM2.5 lag 1h (first 10): {full_df['pm2_5_lag_1h'].head(10).values}")

# Check how many lag features are 0 vs NaN
lag_features = [col for col in full_df.columns if 'lag_' in col or 'rolling_' in col or 'change_rate' in col]
print(f"\n=== Lag feature analysis ===")
for feature in lag_features[:5]:  # Show first 5
    zero_count = (full_df[feature] == 0).sum()
    nan_count = full_df[feature].isna().sum()
    total_count = len(full_df)
    print(f"{feature}: {zero_count} zeros, {nan_count} NaNs, {total_count} total") 