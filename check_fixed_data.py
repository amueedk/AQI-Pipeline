import pandas as pd

# Load the fixed data
df = pd.read_csv('good_hopsworks_data_fixed.csv')

print(f"Data shape: {df.shape}")

# Check lag features
lag_cols = [col for col in df.columns if 'lag_' in col]
print(f"\nNumber of lag features: {len(lag_cols)}")

# Show first few lag features
print("\nFirst 5 lag features:")
for col in lag_cols[:5]:
    print(f"{col}: {df[col].head(5).values}")

# Check for zeros vs NaNs
print("\nChecking for zeros in lag features:")
for col in lag_cols[:3]:
    zero_count = (df[col] == 0).sum()
    nan_count = df[col].isna().sum()
    print(f"{col}: {zero_count} zeros, {nan_count} NaNs")

print("\nData is ready for use!") 