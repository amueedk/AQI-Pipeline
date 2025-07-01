"""
Feature Engineering for AQI Data Collection & Feature Engineering
Computes time-based features and derived features from raw weather and AQI data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Optional, Tuple
from config import PATHS, FEATURE_CONFIG

logger = None

# US EPA AQI breakpoints for each pollutant
AQI_BREAKPOINTS = {
    "pm2_5": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ],
    "pm10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    "o3_8h": [
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300),
    ],
    "o3_1h": [
        (125, 164, 101, 150),
        (165, 204, 151, 200),
        (205, 404, 201, 300),
        (405, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    "co": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500),
    ],
    "so2": [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805, 1004, 401, 500),
    ],
    "no2": [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ],
}

def calc_aqi(conc, breakpoints):
    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= conc <= C_high:
            return round((I_high - I_low) / (C_high - C_low) * (conc - C_low) + I_low)
    return None

def compute_all_aqi(row):
    aqi_values = {}
    if not pd.isna(row.get("pm2_5")):
        aqi_values["pm2_5_aqi"] = calc_aqi(row["pm2_5"], AQI_BREAKPOINTS["pm2_5"])
    if not pd.isna(row.get("pm10")):
        aqi_values["pm10_aqi"] = calc_aqi(row["pm10"], AQI_BREAKPOINTS["pm10"])
    if not pd.isna(row.get("ozone")):
        aqi_values["o3_aqi"] = calc_aqi(row["ozone"], AQI_BREAKPOINTS["o3_8h"])
    if not pd.isna(row.get("carbon_monoxide")):
        aqi_values["co_aqi"] = calc_aqi(row["carbon_monoxide"], AQI_BREAKPOINTS["co"])
    if not pd.isna(row.get("sulphur_dioxide")):
        aqi_values["so2_aqi"] = calc_aqi(row["sulphur_dioxide"], AQI_BREAKPOINTS["so2"])
    if not pd.isna(row.get("nitrogen_dioxide")):
        aqi_values["no2_aqi"] = calc_aqi(row["nitrogen_dioxide"], AQI_BREAKPOINTS["no2"])
    return aqi_values

def compute_overall_aqi(row):
    aqi_values = compute_all_aqi(row)
    if aqi_values:
        return max([v for v in aqi_values.values() if v is not None])
    return None

class AQIFeatureEngineer:
    def __init__(self):
        self.target_column = FEATURE_CONFIG["target_column"]
        self.lag_hours = FEATURE_CONFIG["lag_hours"]
        self.rolling_windows = FEATURE_CONFIG["rolling_windows"]
        
        # Create directories if they don't exist
        for path in PATHS.values():
            os.makedirs(path, exist_ok=True)
        
        # Configure logging after directories are created
        global logger
        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(os.path.join(PATHS['logs_dir'], 'feature_engineering.log')),
                    logging.StreamHandler()
                ]
            )
            logger = logging.getLogger(__name__)
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from datetime index
        """
        df = df.copy()
        
        # Basic time features
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Season features
        df['is_spring'] = ((df['month'] >= 3) & (df['month'] <= 5)).astype(int)
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        df['is_autumn'] = ((df['month'] >= 9) & (df['month'] <= 11)).astype(int)
        df['is_winter'] = ((df['month'] == 12) | (df['month'] <= 2)).astype(int)
        
        # Day/Night feature
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Rush hour features
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        logger.info("Created time-based features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Create lag features for time series prediction
        """
        if target_col is None:
            target_col = self.target_column
            
        df = df.copy()
        
        # Create lag features for target variable
        for lag in self.lag_hours:
            df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
        
        # Create rolling statistics
        for window in self.rolling_windows:
            df[f'{target_col}_rolling_mean_{window}h'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}h'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}h'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}h'] = df[target_col].rolling(window=window).max()
        
        # Create change rate features
        df[f'{target_col}_change_rate_1h'] = df[target_col].pct_change(1)
        df[f'{target_col}_change_rate_6h'] = df[target_col].pct_change(6)
        df[f'{target_col}_change_rate_24h'] = df[target_col].pct_change(24)
        
        logger.info("Created lag features")
        return df
    
    def create_weather_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from weather data (OpenWeather variable names only)
        """
        df = df.copy()
        # Temperature features
        if 'temperature' in df.columns:
            df['temp_squared'] = df['temperature'] ** 2
            df['temp_cubed'] = df['temperature'] ** 3
            df['temp_change_rate'] = df['temperature'].pct_change(1)
            df['is_hot'] = (df['temperature'] > 35).astype(int)
            df['is_cold'] = (df['temperature'] < 10).astype(int)
        # Humidity features
        if 'humidity' in df.columns:
            df['humidity_squared'] = df['humidity'] ** 2
            df['is_high_humidity'] = (df['humidity'] > 80).astype(int)
            df['is_low_humidity'] = (df['humidity'] < 30).astype(int)
        # Wind features
        if 'wind_speed' in df.columns:
            df['wind_speed_squared'] = df['wind_speed'] ** 2
            df['is_high_wind'] = (df['wind_speed'] > 20).astype(int)
            df['is_calm'] = (df['wind_speed'] < 5).astype(int)
        # Pressure features
        if 'pressure' in df.columns:
            df['pressure_change_rate'] = df['pressure'].pct_change(1)
            df['is_low_pressure'] = (df['pressure'] < 1010).astype(int)
            df['is_high_pressure'] = (df['pressure'] > 1020).astype(int)
        return df
    
    def create_pollutant_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from pollutant data
        """
        df = df.copy()
        
        # PM2.5 features
        if 'pm2_5' in df.columns:
            df['pm2_5_squared'] = df['pm2_5'] ** 2
            df['pm2_5_change_rate'] = df['pm2_5'].pct_change(1)
            df['is_high_pm2_5'] = (df['pm2_5'] > 35).astype(int)
            df['is_low_pm2_5'] = (df['pm2_5'] < 12).astype(int)
        
        # PM10 features
        if 'pm10' in df.columns:
            df['pm10_squared'] = df['pm10'] ** 2
            df['pm10_change_rate'] = df['pm10'].pct_change(1)
            df['is_high_pm10'] = (df['pm10'] > 50).astype(int)
            df['is_low_pm10'] = (df['pm10'] < 20).astype(int)
        
        # NO2 features
        if 'nitrogen_dioxide' in df.columns:
            df['no2_squared'] = df['nitrogen_dioxide'] ** 2
            df['no2_change_rate'] = df['nitrogen_dioxide'].pct_change(1)
            df['is_high_no2'] = (df['nitrogen_dioxide'] > 200).astype(int)
        
        # O3 features
        if 'ozone' in df.columns:
            df['o3_squared'] = df['ozone'] ** 2
            df['o3_change_rate'] = df['ozone'].pct_change(1)
            df['is_high_o3'] = (df['ozone'] > 100).astype(int)
        
        # CO features
        if 'carbon_monoxide' in df.columns:
            df['co_squared'] = df['carbon_monoxide'] ** 2
            df['co_change_rate'] = df['carbon_monoxide'].pct_change(1)
            df['is_high_co'] = (df['carbon_monoxide'] > 5000).astype(int)
        
        # SO2 features
        if 'sulphur_dioxide' in df.columns:
            df['so2_squared'] = df['sulphur_dioxide'] ** 2
            df['so2_change_rate'] = df['sulphur_dioxide'].pct_change(1)
            df['is_high_so2'] = (df['sulphur_dioxide'] > 500).astype(int)
        
        # Ratio features
        if 'pm2_5' in df.columns and 'pm10' in df.columns:
            df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-8)
        
        if 'nitrogen_dioxide' in df.columns and 'ozone' in df.columns:
            df['no2_o3_ratio'] = df['nitrogen_dioxide'] / (df['ozone'] + 1e-8)
        
        logger.info("Created pollutant-derived features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables
        """
        df = df.copy()
        
        # Temperature-Humidity interaction
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        # Temperature-Wind interaction
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            df['temp_wind_interaction'] = df['temperature'] * df['wind_speed']
        
        # PM2.5-Temperature interaction
        if 'pm2_5' in df.columns and 'temperature' in df.columns:
            df['pm2_5_temp_interaction'] = df['pm2_5'] * df['temperature']
        
        # PM2.5-Humidity interaction
        if 'pm2_5' in df.columns and 'humidity' in df.columns:
            df['pm2_5_humidity_interaction'] = df['pm2_5'] * df['humidity']
        
        # Wind-PM2.5 interaction
        if 'wind_speed' in df.columns and 'pm2_5' in df.columns:
            df['wind_pm2_5_interaction'] = df['wind_speed'] * df['pm2_5']
        
        logger.info("Created interaction features")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset using a forward-fill strategy,
        which is suitable for time-series data. Also handles infinite values.
        """
        df = df.copy()
        
        # Replace infinite values with NaN first
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        nan_count_before = df.isnull().sum().sum()
        if nan_count_before > 0:
            logger.info(f"Found {nan_count_before} missing or infinite values to handle.")

        # Step 1: Forward-fill
        df.ffill(inplace=True)
        
        # Step 2: Backward-fill for any NaNs remaining at the start of the series
        df.bfill(inplace=True)

        # Step 3: Fill any remaining NaNs (if a whole column was NaN) with 0
        df.fillna(0, inplace=True)
        
        nan_count_after = df.isnull().sum().sum()
        if nan_count_after > 0:
            logger.warning(f"There are still {nan_count_after} missing values after handling.")
        else:
            logger.info("Successfully handled all missing values.")
            
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline on a dataframe
        Computes AQI for each pollutant and overall AQI using OpenWeather data.
        Excludes IQAir and abs_deviation fields from features.
        """
        logger.info(f"Starting feature engineering on dataframe with {len(df)} rows.")
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Compute AQI for each pollutant and overall AQI
        aqi_cols = ["pm2_5", "pm10", "ozone", "carbon_monoxide", "sulphur_dioxide", "nitrogen_dioxide"]
        for col in aqi_cols:
            aqi_name = f"{col}_aqi"
            df[aqi_name] = df.apply(lambda row: calc_aqi(row[col], AQI_BREAKPOINTS[col if col not in ["ozone", "carbon_monoxide", "sulphur_dioxide", "nitrogen_dioxide"] else {"ozone": "o3_8h", "carbon_monoxide": "co", "sulphur_dioxide": "so2", "nitrogen_dioxide": "no2"}[col]]) if not pd.isna(row.get(col)) else None, axis=1)
        df["overall_aqi"] = df.apply(compute_overall_aqi, axis=1)

        # The rest of the pipeline (time features, lags, etc.)
        engineered_df = self.create_time_features(df)
        engineered_df = self.create_lag_features(engineered_df)
        engineered_df = self.create_weather_derived_features(engineered_df)
        engineered_df = self.create_pollutant_derived_features(engineered_df)
        engineered_df = self.create_interaction_features(engineered_df)
        engineered_df = self.handle_missing_values(engineered_df)
        
        logger.info(f"Feature engineering complete. Final dataframe shape: {engineered_df.shape}")
        return engineered_df
    
    def get_feature_columns(self, df: pd.DataFrame, exclude_target: bool = True) -> List[str]:
        """
        Get the list of feature column names from the dataframe.
        Excludes non-feature columns and optionally the target column.
        Excludes IQAir and abs_deviation fields.
        """
        cols_to_exclude = ['city', 'latitude', 'longitude', 'iqair_aqi', 'abs_deviation']
        if exclude_target:
            cols_to_exclude.append(self.target_column)
            
        feature_cols = [col for col in df.columns if col not in cols_to_exclude]
        return feature_cols

def main():
    """
    Main function to test feature engineering.
    This function will:
    1. Collect fresh historical data using the DataCollector.
    2. Run feature engineering on that data.
    3. Save the engineered features to a CSV file for inspection.
    """
    from data_collector import OpenWeatherDataCollector

    # 1. Collect data
    logger.info("--- Main Test: Running Data Collector ---")
    collector = OpenWeatherDataCollector()
    # Fetch a smaller range for a quick test
    raw_data_df = collector.collect_historical_data(days_back=3) 
    
    if raw_data_df.empty:
        logger.error("--- Main Test: Data collection failed. Aborting feature engineering test. ---")
        return

    # 2. Engineer features
    logger.info("--- Main Test: Running Feature Engineer ---")
    engineer = AQIFeatureEngineer()
    engineered_df = engineer.engineer_features(raw_data_df)

    if not engineered_df.empty:
        # 3. Save for inspection
        output_path = os.path.join(PATHS['temp_dir'], 'test_engineered_features.csv')
        engineered_df.to_csv(output_path)
        logger.info(f"--- Main Test: Successfully engineered features and saved to {output_path} ---")
        logger.info(f"Engineered DataFrame shape: {engineered_df.shape}")
        logger.info(f"First 5 rows:\n{engineered_df.head()}")
    else:
        logger.error("--- Main Test: Feature engineering failed. ---")


if __name__ == "__main__":
    main()