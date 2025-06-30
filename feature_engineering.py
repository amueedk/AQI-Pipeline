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
        Create derived features from weather data
        """
        df = df.copy()
        
        # Temperature features
        if 'temperature_2m' in df.columns:
            df['temp_squared'] = df['temperature_2m'] ** 2
            df['temp_cubed'] = df['temperature_2m'] ** 3
            
            # Temperature change rate
            df['temp_change_rate'] = df['temperature_2m'].pct_change(1)
            
            # Temperature extremes
            df['is_hot'] = (df['temperature_2m'] > 35).astype(int)
            df['is_cold'] = (df['temperature_2m'] < 10).astype(int)
        
        # Humidity features
        if 'relative_humidity_2m' in df.columns:
            df['humidity_squared'] = df['relative_humidity_2m'] ** 2
            df['is_high_humidity'] = (df['relative_humidity_2m'] > 80).astype(int)
            df['is_low_humidity'] = (df['relative_humidity_2m'] < 30).astype(int)
        
        # Wind features
        if 'wind_speed_10m' in df.columns:
            df['wind_speed_squared'] = df['wind_speed_10m'] ** 2
            df['is_high_wind'] = (df['wind_speed_10m'] > 20).astype(int)
            df['is_calm'] = (df['wind_speed_10m'] < 5).astype(int)
        
        # Pressure features
        if 'pressure_msl' in df.columns:
            df['pressure_change_rate'] = df['pressure_msl'].pct_change(1)
            df['is_low_pressure'] = (df['pressure_msl'] < 1010).astype(int)
            df['is_high_pressure'] = (df['pressure_msl'] > 1020).astype(int)
        
        # Precipitation features
        if 'precipitation' in df.columns:
            df['is_raining'] = (df['precipitation'] > 0).astype(int)
            df['rain_intensity'] = pd.cut(df['precipitation'], 
                                        bins=[0, 0.1, 2.5, 7.5, 50, 1000], 
                                        labels=[0, 1, 2, 3, 4], 
                                        include_lowest=True).astype(int)
        
        # Cloud cover features
        if 'cloud_cover' in df.columns:
            df['is_cloudy'] = (df['cloud_cover'] > 80).astype(int)
            df['is_clear'] = (df['cloud_cover'] < 20).astype(int)
        
        # UV index features
        if 'uv_index' in df.columns:
            df['is_high_uv'] = (df['uv_index'] > 8).astype(int)
            df['is_low_uv'] = (df['uv_index'] < 3).astype(int)
        
        logger.info("Created weather-derived features")
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
        if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
            df['temp_humidity_interaction'] = df['temperature_2m'] * df['relative_humidity_2m']
        
        # Temperature-Wind interaction
        if 'temperature_2m' in df.columns and 'wind_speed_10m' in df.columns:
            df['temp_wind_interaction'] = df['temperature_2m'] * df['wind_speed_10m']
        
        # PM2.5-Temperature interaction
        if 'pm2_5' in df.columns and 'temperature_2m' in df.columns:
            df['pm2_5_temp_interaction'] = df['pm2_5'] * df['temperature_2m']
        
        # PM2.5-Humidity interaction
        if 'pm2_5' in df.columns and 'relative_humidity_2m' in df.columns:
            df['pm2_5_humidity_interaction'] = df['pm2_5'] * df['relative_humidity_2m']
        
        # Wind-PM2.5 interaction
        if 'wind_speed_10m' in df.columns and 'pm2_5' in df.columns:
            df['wind_pm2_5_interaction'] = df['wind_speed_10m'] * df['pm2_5']
        
        logger.info("Created interaction features")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        """
        df = df.copy()
        
        # Forward fill for time series data
        df = df.fillna(method='ffill')
        
        # Backward fill for remaining NaNs
        df = df.fillna(method='bfill')
        
        # For remaining NaNs, use median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        logger.info(f"Handled missing values. Remaining NaNs: {df.isnull().sum().sum()}")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        """
        logger.info("Starting feature engineering pipeline")
        
        # Remove non-numeric columns that shouldn't be processed
        exclude_cols = ['city', 'latitude', 'longitude']
        df_clean = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        
        # Create all feature types
        df_clean = self.create_time_features(df_clean)
        df_clean = self.create_lag_features(df_clean)
        df_clean = self.create_weather_derived_features(df_clean)
        df_clean = self.create_pollutant_derived_features(df_clean)
        df_clean = self.create_interaction_features(df_clean)
        
        # Handle missing values
        df_clean = self.handle_missing_values(df_clean)
        
        # Remove infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(df_clean.median())
        
        # Add back the original metadata columns
        for col in exclude_cols:
            if col in df.columns:
                df_clean[col] = df[col]
        
        logger.info(f"Feature engineering completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def get_feature_columns(self, df: pd.DataFrame, exclude_target: bool = True) -> List[str]:
        """
        Get list of feature columns (excluding target and metadata)
        """
        exclude_cols = [
            self.target_column,
            'city', 'latitude', 'longitude',
            'weather_icon', 'weather_description'
        ]
        
        if exclude_target:
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        else:
            feature_cols = [col for col in df.columns if col not in ['city', 'latitude', 'longitude', 'weather_icon', 'weather_description']]
        
        return feature_cols
    
    def update_master_features(self, master_data_file: str = None) -> bool:
        """
        Update master features dataset with latest data
        """
        try:
            if master_data_file is None:
                master_data_file = os.path.join(PATHS['data_dir'], 'master_dataset.csv')
            
            if not os.path.exists(master_data_file):
                logger.error(f"Master dataset not found: {master_data_file}")
                return False
            
            # Load master dataset
            df = pd.read_csv(master_data_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded master dataset with {len(df)} records")
            
            # Engineer features
            df_engineered = self.engineer_features(df)
            
            # Save engineered features
            output_file = os.path.join(PATHS['data_dir'], 'master_features.csv')
            df_engineered.to_csv(output_file)
            
            logger.info(f"Master features updated and saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating master features: {e}")
            return False

def main():
    """
    Main function to test feature engineering
    """
    # Check for master dataset
    master_file = os.path.join(PATHS['data_dir'], 'master_dataset.csv')
    
    if os.path.exists(master_file):
        print(f"Found master dataset: {master_file}")
        
        # Initialize feature engineer
        engineer = AQIFeatureEngineer()
        
        # Update master features
        success = engineer.update_master_features(master_file)
        
        if success:
            # Load and show statistics
            features_file = os.path.join(PATHS['data_dir'], 'master_features.csv')
            df_engineered = pd.read_csv(features_file, index_col=0, parse_dates=True)
            
            print(f"\nFeature engineering completed!")
            print(f"Master features shape: {df_engineered.shape}")
            print(f"Number of features: {len(engineer.get_feature_columns(df_engineered))}")
            
            # Show US AQI statistics if available
            if 'us_aqi' in df_engineered.columns:
                aqi_stats = df_engineered['us_aqi'].describe()
                print(f"\nUS AQI Statistics:")
                print(f"Mean: {aqi_stats['mean']:.2f}")
                print(f"Min: {aqi_stats['min']:.2f}")
                print(f"Max: {aqi_stats['max']:.2f}")
                print(f"Std: {aqi_stats['std']:.2f}")
        else:
            print("Feature engineering failed")
    else:
        print("No master dataset found. Please run data_collector.py first to collect data.")

if __name__ == "__main__":
    main() 