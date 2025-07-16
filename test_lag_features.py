import pandas as pd
import numpy as np
from datetime import datetime

# Load the data
df = pd.read_csv('good_hopsworks_data_20250716_204528.csv')

# Convert time column to datetime and set as index
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"First few rows:")
print(df.head())

# Test lag features manually
print("\nTesting lag features for PM2.5:")
print(f"PM2.5 values (first 10): {df['pm2_5'].head(10).values}")

# Create lag features manually
df['pm2_5_lag_1h'] = df['pm2_5'].shift(1)
df['pm2_5_lag_2h'] = df['pm2_5'].shift(2)
df['pm2_5_lag_3h'] = df['pm2_5'].shift(3)

print(f"\nPM2.5 lag 1h (first 10): {df['pm2_5_lag_1h'].head(10).values}")
print(f"PM2.5 lag 2h (first 10): {df['pm2_5_lag_2h'].head(10).values}")
print(f"PM2.5 lag 3h (first 10): {df['pm2_5_lag_3h'].head(10).values}")

# Check if any lag values are non-zero
non_zero_lags = df[['pm2_5_lag_1h', 'pm2_5_lag_2h', 'pm2_5_lag_3h']].sum().sum()
print(f"\nSum of all lag values: {non_zero_lags}")

# Test rolling features
df['pm2_5_rolling_mean_3h'] = df['pm2_5'].rolling(window=3).mean()
print(f"\nPM2.5 rolling mean 3h (first 10): {df['pm2_5_rolling_mean_3h'].head(10).values}")

# Check if any rolling values are non-zero
non_zero_rolling = df['pm2_5_rolling_mean_3h'].sum()
print(f"Sum of rolling mean values: {non_zero_rolling}")

# Check the original CSV for lag features
print(f"\nColumns in original CSV that contain 'lag':")
lag_cols = [col for col in df.columns if 'lag' in col]
print(f"Found {len(lag_cols)} lag columns: {lag_cols[:5]}...")

# Check values in original lag columns
if lag_cols:
    print(f"\nFirst few values in original lag columns:")
    for col in lag_cols[:3]:
        print(f"{col}: {df[col].head(5).values}") 