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
            aqi = round((I_high - I_low) / (C_high - C_low) * (conc - C_low) + I_low)
            print(f"calc_aqi: conc={conc}, range=({C_low},{C_high}), AQI={aqi}")
            return aqi
    print(f"calc_aqi: conc={conc} did not match any breakpoint, returning None")
    return None

def compute_pm2_5_aqi(row):
    """
    Compute PM2.5 AQI only (since PM2.5 is typically the dominant pollutant)
    """
    if not pd.isna(row.get("pm2_5")):
        return calc_aqi(row["pm2_5"], AQI_BREAKPOINTS["pm2_5"])
    return None

class AQIFeatureEngineer:
    def __init__(self):
        self.target_columns = FEATURE_CONFIG["target_columns"]
        self.primary_target = FEATURE_CONFIG["primary_target"]
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
        Using only cyclic encoding for better ML performance
        """
        df = df.copy()
        
        # Extract time components for cyclic encoding
        hour = df.index.hour
        day = df.index.day
        month = df.index.month
        day_of_week = df.index.dayofweek
        
        # Cyclical encoding for periodic features (more effective for ML)
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * day / 31)
        df['day_cos'] = np.cos(2 * np.pi * day / 31)
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Season features
        df['is_spring'] = ((month >= 3) & (month <= 5)).astype(int)
        df['is_summer'] = ((month >= 6) & (month <= 8)).astype(int)
        df['is_autumn'] = ((month >= 9) & (month <= 11)).astype(int)
        df['is_winter'] = ((month == 12) | (month <= 2)).astype(int)
        
        # Day/Night feature
        df['is_night'] = ((hour >= 22) | (hour <= 6)).astype(int)
        
        # Rush hour features
        df['is_morning_rush'] = ((hour >= 7) & (hour <= 9)).astype(int)
        df['is_evening_rush'] = ((hour >= 17) & (hour <= 19)).astype(int)
        
        logger.info("Created cyclic time-based features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Create true time-based lag features for time series prediction
        Only create lags for PM2.5 and PM10 (target variables)
        Other pollutants are used as current features only
        """
        if target_col is None:
            target_col = self.primary_target
            
        df = df.copy()
        
        # Ensure data is sorted by timestamp first
        df = df.sort_index()
        
        # Debug: Log some info about the dataset being processed
        logger.info(f"DEBUG: Creating lag features for dataset with {len(df)} records")
        logger.info(f"DEBUG: Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"DEBUG: Sample timestamps: {list(df.index[:5])}")
        logger.info(f"DEBUG: Last 5 timestamps: {list(df.index[-5:])}")
        logger.info(f"DEBUG: Target columns: {self.target_columns}")
        logger.info(f"DEBUG: Available columns: {list(df.columns)}")
        
        # Create lag features for both target variables (PM2.5 and PM10)
        for target in self.target_columns:
            if target in df.columns:
                logger.info(f"DEBUG: Creating lag features for {target}")
                # Create time-based lag features for target variable
                for lag in self.lag_hours:
                    logger.info(f"DEBUG: Creating {lag}h lag for {target}")
                    df[f'{target}_lag_{lag}h'] = self._create_time_based_lag(df, target, lag)
                
                # Create time-based rolling statistics
                for window in self.rolling_windows:
                    df[f'{target}_rolling_mean_{window}h'] = self._create_time_based_rolling(df, target, window, 'mean')
                    df[f'{target}_rolling_std_{window}h'] = self._create_time_based_rolling(df, target, window, 'std')
                    df[f'{target}_rolling_min_{window}h'] = self._create_time_based_rolling(df, target, window, 'min')
                    df[f'{target}_rolling_max_{window}h'] = self._create_time_based_rolling(df, target, window, 'max')
                
                # Create time-based change rate features
                df[f'{target}_change_rate_1h'] = self._create_time_based_change_rate(df, target, 1)
                df[f'{target}_change_rate_6h'] = self._create_time_based_change_rate(df, target, 6)
                df[f'{target}_change_rate_24h'] = self._create_time_based_change_rate(df, target, 24)
        
        logger.info(f"Created true time-based lag features for target variables: {self.target_columns}")
        return df
    
    def _get_tolerance(self, period: int, feature_type: str = 'lag') -> timedelta:
        """
        Get tolerance based on period and feature type with custom specifications
        """
        if feature_type == 'lag' or feature_type == 'change_rate':
            # Custom tolerance specifications
            if period == 1:
                tolerance_hours = 0.5  # ±30min for 1h
            elif period == 2:
                tolerance_hours = 0.75  # ±45min for 2h
            elif period == 3:
                tolerance_hours = 50/60  # ±50min for 3h
            else:
                # All other periods (6h, 12h, 24h, 48h, 72h) get 1h tolerance
                tolerance_hours = 1.0
            return timedelta(hours=tolerance_hours)
        elif feature_type == 'rolling':
            # Rolling features don't use tolerance anyway (uses ALL data in window)
            tolerance_hours = period * 0.25
            return timedelta(hours=tolerance_hours)
    
    def _create_time_based_lag(self, df: pd.DataFrame, target: str, lag_hours: int) -> pd.Series:
        """
        Create time-based lag feature that finds data approximately lag_hours ago
        """
        import numpy as np
        from datetime import timedelta
        
        lag_series = pd.Series(index=df.index, dtype=float)
        tolerance = self._get_tolerance(lag_hours, 'lag')
        
        # Debug: Log the chronologically latest few calculations to see what's happening
        # Since df is sorted by timestamp, the last few records are the most recent
        total_records = len(df.index)
        debug_start = max(0, total_records - 5)  # Last 5 records (most recent timestamps)
        
        for i, current_time in enumerate(df.index):
            target_time = current_time - timedelta(hours=lag_hours)
            
            # Find data within acceptable range
            acceptable_range_start = target_time - tolerance
            acceptable_range_end = target_time + tolerance
            
            # Use pandas boolean indexing for efficiency
            mask = (df.index >= acceptable_range_start) & (df.index <= acceptable_range_end)
            matching_data = df[mask][target]
            
            # Debug logging for latest few records
            if i >= debug_start:
                logger.info(f"DEBUG LAG {lag_hours}h - Record {i}:")
                logger.info(f"  Current time: {current_time}")
                logger.info(f"  Target time: {target_time}")
                logger.info(f"  Tolerance: ±{tolerance}")
                logger.info(f"  Acceptable range: {acceptable_range_start} to {acceptable_range_end}")
                logger.info(f"  Found {len(matching_data)} matching data points")
                if len(matching_data) > 0:
                    logger.info(f"  Matching timestamps: {list(df.index[mask])}")
                    logger.info(f"  Matching values: {list(matching_data.values)}")
                else:
                    logger.info(f"  No matching data found")
            
            if len(matching_data) > 0:
                # Use the closest data point to target_time
                matching_indices = df.index[mask]
                time_diffs = [(idx - target_time).total_seconds() for idx in matching_indices]
                min_diff_idx = np.argmin(np.abs(time_diffs))
                closest_idx = matching_indices[min_diff_idx]
                
                # Handle potential duplicate timestamps by ensuring scalar value
                value = df.loc[closest_idx, target]
                lag_series.iloc[i] = value if np.isscalar(value) else value.iloc[0]
                
                if i >= debug_start:
                    logger.info(f"  Selected closest: {closest_idx} (diff: {time_diffs[min_diff_idx]:.0f}s)")
                    logger.info(f"  Final lag value: {lag_series.iloc[i]}")
            else:
                lag_series.iloc[i] = np.nan
                if i >= debug_start:
                    logger.info(f"  Result: NaN (no data found)")
            
            if i >= debug_start:
                logger.info("")
        
        return lag_series
    
    def _create_time_based_rolling(self, df: pd.DataFrame, target: str, window_hours: int, stat_type: str) -> pd.Series:
        """
        Create time-based rolling statistics using ALL data in the time window
        """
        import numpy as np
        from datetime import timedelta
        
        rolling_series = pd.Series(index=df.index, dtype=float)
        
        for i, current_time in enumerate(df.index):
            window_start = current_time - timedelta(hours=window_hours)
            
            # Get ALL data in the window (no tolerance checks)
            window_mask = (df.index >= window_start) & (df.index <= current_time)
            window_data = df[window_mask][target]
            
            if len(window_data) < 2:
                rolling_series.iloc[i] = np.nan
                continue
            
            # Calculate rolling statistic with all available data
            if stat_type == 'mean':
                rolling_series.iloc[i] = window_data.mean()
            elif stat_type == 'std':
                rolling_series.iloc[i] = window_data.std()
            elif stat_type == 'min':
                rolling_series.iloc[i] = window_data.min()
            elif stat_type == 'max':
                rolling_series.iloc[i] = window_data.max()
        
        return rolling_series
    
    def _create_time_based_change_rate(self, df: pd.DataFrame, target: str, period_hours: int) -> pd.Series:
        """
        Create time-based change rate that finds data approximately period_hours ago
        """
        import numpy as np
        from datetime import timedelta
        
        change_rate_series = pd.Series(index=df.index, dtype=float)
        tolerance = self._get_tolerance(period_hours, 'change_rate')
        
        for i, current_time in enumerate(df.index):
            target_time = current_time - timedelta(hours=period_hours)
            current_value = df.iloc[i][target]
            
            # Find data within acceptable range
            acceptable_range_start = target_time - tolerance
            acceptable_range_end = target_time + tolerance
            
            # Use pandas boolean indexing for efficiency
            mask = (df.index >= acceptable_range_start) & (df.index <= acceptable_range_end)
            matching_data = df[mask][target]
            
            if len(matching_data) > 0 and not pd.isna(current_value):
                # Use the closest data point to target_time
                matching_indices = df.index[mask]
                time_diffs = [(idx - target_time).total_seconds() for idx in matching_indices]
                min_diff_idx = np.argmin(np.abs(time_diffs))
                closest_idx = matching_indices[min_diff_idx]
                
                # Handle potential duplicate timestamps by ensuring scalar value
                previous_value = df.loc[closest_idx, target]
                previous_value = previous_value if np.isscalar(previous_value) else previous_value.iloc[0]
                
                if not pd.isna(previous_value) and previous_value != 0:
                    change_rate_series.iloc[i] = (current_value - previous_value) / previous_value
                else:
                    change_rate_series.iloc[i] = np.nan
            else:
                change_rate_series.iloc[i] = np.nan
        
        return change_rate_series
    
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
        Create derived features from pollutant data, focusing on PM2.5 prediction
        """
        df = df.copy()
        
        # PM2.5 features (target variable)
        if 'pm2_5' in df.columns:
            df['pm2_5_squared'] = df['pm2_5'] ** 2
            df['pm2_5_change_rate'] = df['pm2_5'].pct_change(1)
            df['is_high_pm2_5'] = (df['pm2_5'] > 35).astype(int)
            df['is_low_pm2_5'] = (df['pm2_5'] < 12).astype(int)
        
        # PM10 features (important for PM2.5 prediction)
        if 'pm10' in df.columns:
            df['pm10_squared'] = df['pm10'] ** 2
            df['pm10_change_rate'] = df['pm10'].pct_change(1)
            df['is_high_pm10'] = (df['pm10'] > 50).astype(int)
            df['is_low_pm10'] = (df['pm10'] < 20).astype(int)
        
        # Keep raw pollutant concentrations as features (they influence PM2.5)
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
        
        # Important ratio features for PM2.5 prediction
        # Note: We don't create pm2_5_pm10_ratio as it would cause data leakage
        # since pm2_5 is our target variable
        
        logger.info("Created PM2.5/PM10-focused pollutant-derived features")
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
        IMPORTANT: Preserves legitimate NaN values in lag features.
        """
        df = df.copy()
        
        # Replace infinite values with NaN first
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        nan_count_before = df.isnull().sum().sum()
        if nan_count_before > 0:
            logger.info(f"Found {nan_count_before} missing or infinite values to handle.")

        # Identify lag, rolling, and change rate features that should preserve NaN values
        lag_features = [col for col in df.columns if any(x in col for x in ['lag_', 'rolling_', 'change_rate'])]
        non_lag_features = [col for col in df.columns if col not in lag_features]
        
        # Handle missing values for non-lag features only
        if non_lag_features:
            non_lag_df = df[non_lag_features].copy()
            
            # Step 1: Forward-fill for non-lag features
            non_lag_df.ffill(inplace=True)
            
            # Step 2: Backward-fill for any NaNs remaining at the start
            non_lag_df.bfill(inplace=True)
            
            # Step 3: Fill any remaining NaNs with 0
            non_lag_df.fillna(0, inplace=True)
            
            # Update the original dataframe with cleaned non-lag features
            df[non_lag_features] = non_lag_df
        
        # For lag features, preserve ALL NaN values as they indicate legitimate gaps
        # Only handle infinite values, but keep NaN for time-based features
        if lag_features:
            lag_df = df[lag_features].copy()
            
            # Only replace infinite values with NaN, but preserve all other NaN values
            lag_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Update the original dataframe with cleaned lag features
            df[lag_features] = lag_df
        
        nan_count_after = df.isnull().sum().sum()
        if nan_count_after > 0:
            logger.info(f"Preserved {nan_count_after} legitimate NaN values in lag features.")
        else:
            logger.info("Successfully handled all missing values.")
            
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline on a dataframe
        Focuses on PM2.5 and PM10 prediction while keeping all raw pollutant concentrations as features.
        Computes PM2.5 AQI, PM10 AQI, and US AQI (max of both).
        Uses cyclic time encoding instead of raw time features for better ML performance.
        Removes static location data (city, lat, lon) as they don't add predictive value.
        """
        logger.info(f"Starting PM2.5/PM10-focused feature engineering on dataframe with {len(df)} rows.")
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        # --- AQIs ARE ALREADY CALCULATED IN DATA COLLECTOR ---
        # The data collector already calculates pm2_5_aqi, pm10_aqi, and us_aqi
        # We keep all three in Hopsworks for analysis and debugging
        if 'us_aqi' not in df.columns:
            logger.warning("us_aqi not found in DataFrame, calculating from pm2_5_aqi and pm10_aqi")
            if 'pm2_5_aqi' in df.columns and 'pm10_aqi' in df.columns:
                df['us_aqi'] = df.apply(lambda row: max(row['pm2_5_aqi'], row['pm10_aqi']) if pd.notna(row.get('pm2_5_aqi')) and pd.notna(row.get('pm10_aqi')) else None, axis=1)
            else:
                logger.error("Cannot calculate us_aqi: pm2_5_aqi or pm10_aqi not found")
                df['us_aqi'] = None

        # The rest of the pipeline (time features, lags, etc.)
        engineered_df = self.create_time_features(df)
        engineered_df = self.create_lag_features(engineered_df)
        engineered_df = self.create_weather_derived_features(engineered_df)
        engineered_df = self.create_pollutant_derived_features(engineered_df)
        engineered_df = self.create_interaction_features(engineered_df)
        engineered_df = self.handle_missing_values(engineered_df)
        
        # Convert numeric columns to appropriate types for Hopsworks schema
        # Boolean features should be bigint (0/1), others should be float64
        boolean_features = [
            'is_spring', 'is_summer', 'is_autumn', 'is_winter', 'is_night',
            'is_morning_rush', 'is_evening_rush', 'is_hot', 'is_cold',
            'is_high_humidity', 'is_low_humidity', 'is_high_wind', 'is_calm',
            'is_low_pressure', 'is_high_pressure', 'is_high_pm2_5', 'is_low_pm2_5',
            'is_high_pm10', 'is_low_pm10', 'is_high_no2', 'is_high_o3',
            'is_high_co', 'is_high_so2'
        ]
        
        # Convert boolean features to int64 (bigint in Hopsworks)
        for col in boolean_features:
            if col in engineered_df.columns:
                engineered_df[col] = engineered_df[col].astype('int64')
        
        # Convert other numeric columns to float64
        other_numeric_cols = engineered_df.select_dtypes(include=['int64', 'float64']).columns
        for col in other_numeric_cols:
            if col not in boolean_features:
                engineered_df[col] = engineered_df[col].astype('float64')
        
        # Drop location and raw time columns (using cyclic encoding instead)
        columns_to_drop = [
            'city', 'latitude', 'longitude',  # Static location data
            'hour', 'day', 'month', 'year', 'day_of_week', 'day_of_year', 'week_of_year'  # Raw time features (replaced by cyclic)
        ]
        
        for col in columns_to_drop:
            if col in engineered_df.columns:
                engineered_df = engineered_df.drop(columns=[col])
                logger.info(f"Dropped column '{col}' (using cyclic encoding or static data)")

        # Add time column back (required for Hopsworks)
        engineered_df['time'] = engineered_df.index
        
        logger.info(f"PM2.5/PM10-focused feature engineering complete. Final dataframe shape: {engineered_df.shape}")
        logger.info(f"Converted {len(other_numeric_cols)} numeric columns to float64 for Hopsworks compatibility")
        return engineered_df
    
    def get_feature_columns(self, df: pd.DataFrame, exclude_targets: bool = True) -> List[str]:
        """
        Get the list of feature column names from the dataframe.
        Excludes non-feature columns and optionally the target columns.
        Excludes location data, raw time features, and validation fields.
        """
        cols_to_exclude = [
            'city', 'latitude', 'longitude',  # Static location data
            'hour', 'day', 'month', 'year', 'day_of_week', 'day_of_year', 'week_of_year',  # Raw time features
            'iqair_aqi', 'abs_deviation'  # Validation fields
        ]
        if exclude_targets:
            cols_to_exclude.extend(self.target_columns)
            
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