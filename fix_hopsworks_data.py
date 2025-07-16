import pandas as pd
from feature_engineering import AQIFeatureEngineer

# Define the raw columns to keep
RAW_COLUMNS = [
    'time', 'carbon_monoxide', 'no', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide',
    'pm2_5', 'pm10', 'nh3', 'openweather_aqi', 'pm2_5_aqi', 'pm10_aqi', 'us_aqi',
    'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction'
]

# Load the original (broken) engineered data
input_csv = 'good_hopsworks_data_20250716_204528.csv'
df = pd.read_csv(input_csv)

# Keep only the raw columns
raw_df = df[RAW_COLUMNS].copy()

# Convert time to datetime and set as index
raw_df['time'] = pd.to_datetime(raw_df['time'])
raw_df.set_index('time', inplace=True)

# Run the fixed feature engineering pipeline
engineer = AQIFeatureEngineer()
fixed_df = engineer.engineer_features(raw_df)

# Reset index for saving
fixed_df.reset_index(inplace=True)

# Save to new CSV
output_csv = 'good_hopsworks_data_fixed.csv'
fixed_df.to_csv(output_csv, index=False)

print(f"Fixed data exported to {output_csv} with shape {fixed_df.shape}") 