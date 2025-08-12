''' better '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple
import joblib
import json
from datetime import datetime
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ Imports completed successfully!")

# ----------------------
# SHAP helper utilities
# ----------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _save_current_fig(filepath: str) -> None:
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def compute_shap_for_estimator(estimator, X_test: pd.DataFrame, feature_names: List[str], title: str, out_prefix: str) -> None:
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP not installed; skipping SHAP plots.")
        return
    # Sample for speed
    X_use = X_test[feature_names]
    if len(X_use) > 400:
        X_use = X_use.sample(n=400, random_state=42)
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_use)
    except Exception:
        explainer = shap.Explainer(estimator)
        shap_values = explainer(X_use).values

    # Beeswarm
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_use, feature_names=feature_names, show=False)
        plt.title(f"SHAP Beeswarm - {title}")
        _save_current_fig(f"{out_prefix}_beeswarm.png")
    except Exception as e:
        print(f"⚠️ Failed SHAP beeswarm for {title}: {e}")

    # Bar
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_use, feature_names=feature_names, show=False, plot_type='bar')
        plt.title(f"SHAP Bar - {title}")
        _save_current_fig(f"{out_prefix}_bar.png")
    except Exception as e:
        print(f"⚠️ Failed SHAP bar for {title}: {e}")

# Import config if available, otherwise set defaults
try:
    from config import PATHS, HOPSWORKS_CONFIG
    print("✅ Loaded config from config.py")
except ImportError:
    # Default paths
    PATHS = {
        'logs_dir': 'logs',
        'temp_dir': 'temp',
        'data_dir': 'data'
    }
    print("⚠️ Using default paths (config.py not found)")

# Create directories if they don't exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    # Tuned for distant-horizon stability
    'n_estimators': 1800,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': 10,
    'min_data_in_leaf': 140,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'random_state': 42,
    'n_jobs': -1,  # Use all available cores
    'test_size': 0.2
}

# Horizon bands
SHORT_HORIZONS = list(range(1, 13))        # 1..12h
MIDLONG_HORIZONS = list(range(12, 73))     # 12..72h (overlap at 12 by design)

print(f"📋 Configuration:")
print(f"   Short horizons: 1..12 (total {len(SHORT_HORIZONS)})")
print(f"   Mid/Long horizons: 12..72 (total {len(MIDLONG_HORIZONS)})")
print(f"   Model: Multi-Output LightGBM (n_estimators={MODEL_CONFIG['n_estimators']})")
print(f"   Test size: {MODEL_CONFIG['test_size']}")

# Option 2: Load from CSV file (backup)
csv_files = [
     'new_feature_group_data_20250805_050932.csv'
 ]

for csv_file in csv_files:
     if os.path.exists(csv_file):
         df_raw = pd.read_csv(csv_file)
         df_raw['time'] = pd.to_datetime(df_raw['time'], utc=True, errors='coerce')
         df_raw = df_raw.sort_values('time').reset_index(drop=True)
         print("✅ Data sorted by time")

         print(f"✅ Loaded {len(df_raw)} records from {csv_file}")
         print(f"📊 Data shape: {df_raw.shape}")
         break
else:
     print("❌ No CSV files found")
     df_raw = None

# Apply universal preprocessing directly in this script
print("🔄 Applying universal data preprocessing...")

# Basic data cleaning
df_preprocessed = df_raw.copy()

# Remove duplicates
df_preprocessed = df_preprocessed.drop_duplicates()

# Sort by time
df_preprocessed = df_preprocessed.sort_values('time').reset_index(drop=True)

# Handle missing values
df_preprocessed = df_preprocessed.fillna(method='ffill').fillna(method='bfill')

# Remove outliers using IQR method for key pollutants
for col in ['pm2_5', 'pm10', 'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']:
    if col in df_preprocessed.columns:
        Q1 = df_preprocessed[col].quantile(0.25)
        Q3 = df_preprocessed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_preprocessed = df_preprocessed[(df_preprocessed[col] >= lower_bound) & (df_preprocessed[col] <= upper_bound)]

print(f"✅ Preprocessing complete!")
print(f"   Original shape: {df_raw.shape}")
print(f"   Processed shape: {df_preprocessed.shape}")

# Show basic statistics
print("\n📊 Key statistics after preprocessing:")
key_cols = ['pm2_5', 'pm10']
available_key_cols = [col for col in key_cols if col in df_preprocessed.columns]
print(df_preprocessed[available_key_cols].describe().round(3))

class LightGBMMultiOutputTrainer:
    """
    Multi-Output LightGBM training pipeline for AQI forecasting
    Single model predicts all horizons simultaneously
    """

    def __init__(self, model_config=None):
        self.model_config = model_config or MODEL_CONFIG
        self.models = {}
        self.feature_columns = {}
        self.results = {}

        # Initialize scalers (same as original approach)
        self.pm_scaler = StandardScaler()
        self.weather_scaler = StandardScaler()
        self.pollutant_scaler = StandardScaler()
        self.interaction_scaler = StandardScaler()

    def apply_log_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log transformation ONLY to PM2.5 (same as original approach)
        """
        df = df.copy()

        if 'pm2_5' in df.columns:
            df['pm2_5_log'] = np.log1p(df['pm2_5'])
            print(f"✅ Applied log transformation to PM2.5")

        print(f"✅ Kept PM10 raw")
        return df

    def scale_features_consistently(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale features consistently (same as original approach)
        """
        df = df.copy()

        # PM features scaling (group all PM features together)
        pm_features = ['pm2_5_log', 'pm10', 'pm2_5_lag_1h', 'pm2_5_lag_2h', 'pm2_5_lag_3h', 'pm2_5_lag_24h',
                      'pm10_lag_1h', 'pm10_lag_2h', 'pm10_lag_3h', 'pm10_lag_24h',
                      'pm2_5_rolling_min_3h', 'pm2_5_rolling_mean_3h', 'pm2_5_rolling_max_3h', 'pm2_5_rolling_min_12h', 'pm2_5_rolling_mean_12h', 'pm2_5_rolling_max_12h', 'pm2_5_rolling_mean_24h', 'pm2_5_rolling_max_24h',
                      'pm10_rolling_min_3h', 'pm10_rolling_mean_3h', 'pm10_rolling_mean_12h', 'pm10_rolling_mean_24h',
                      'pm2_5_change_rate_1h', 'pm2_5_change_rate_6h', 'pm2_5_change_rate_24h',
                      'pm10_change_rate_1h', 'pm10_change_rate_6h', 'pm10_change_rate_24h']
        available_pm = [f for f in pm_features if f in df.columns]

        if available_pm:
            pm_values = df[available_pm].values
            pm_scaled = self.pm_scaler.fit_transform(pm_values)
            for i, col in enumerate(available_pm):
                df[f'{col}_scaled'] = pm_scaled[:, i]
            print(f"✅ Scaled {len(available_pm)} PM features consistently")

        # Weather features scaling (group all weather features together)
        weather_features = ['temperature', 'humidity', 'pressure', 'wind_speed']
        available_weather = [f for f in weather_features if f in df.columns]

        if available_weather:
            weather_values = df[available_weather].values
            weather_scaled = self.weather_scaler.fit_transform(weather_values)
            for i, col in enumerate(available_weather):
                df[f'{col}_scaled'] = weather_scaled[:, i]
            print(f"✅ Scaled {len(available_weather)} weather features consistently")

        # Pollutant features scaling (group all pollutants together)
        pollutant_features = ['carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']
        available_pollutants = [f for f in pollutant_features if f in df.columns]

        if available_pollutants:
            pollutant_values = df[available_pollutants].values
            pollutant_scaled = self.pollutant_scaler.fit_transform(pollutant_values)
            for i, col in enumerate(available_pollutants):
                df[f'{col}_scaled'] = pollutant_scaled[:, i]
            print(f"✅ Scaled {len(available_pollutants)} pollutant features consistently")

        # Weather interaction features scaling (group together)
        interaction_features = ['pm2_5_temp_interaction', 'pm2_5_humidity_interaction', 'pm2_5_pressure_interaction',
                              'pm10_temperature_interaction', 'pm10_pressure_interaction',
                              'temp_humidity_interaction', 'temp_wind_interaction',
                              'wind_direction_temp_interaction', 'wind_direction_humidity_interaction',
                              'pressure_humidity_interaction',
                              'co_pressure_interaction', 'o3_temp_interaction', 'so2_humidity_interaction']
        available_interactions = [f for f in interaction_features if f in df.columns]

        if available_interactions:
            interaction_values = df[available_interactions].values
            interaction_scaled = self.interaction_scaler.fit_transform(interaction_values)
            for i, col in enumerate(available_interactions):
                df[f'{col}_scaled'] = interaction_scaled[:, i]
            print(f"✅ Scaled {len(available_interactions)} interaction features consistently")

        # Time features: NO SCALING (already 0-1)
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        available_time = [f for f in time_features if f in df.columns]
        print(f"✅ Time features: NO SCALING (already 0-1): {available_time}")

        return df

    def create_target_variables(self, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        """
        Create target variables for different prediction horizons
        """
        df = df.copy()

        print(f"🎯 Creating target variables for horizons: {horizons}")

        # Use original PM2.5 and PM10 (not log-transformed) for targets
        for horizon in horizons:
            # PM2.5 targets
            df[f'pm2_5_target_{horizon}h'] = df['pm2_5'].shift(-horizon)

            # PM10 targets
            df[f'pm10_target_{horizon}h'] = df['pm10'].shift(-horizon)

        return df

    def create_target_horizon_weather_features(self, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        """
        Create weather features at the target horizon (t+horizon)
        This uses actual historical weather data at the target time
        """
        df = df.copy()

        print(f"🌤️ Creating target horizon weather features for all horizons...")

        for horizon in horizons:
            # Weather features at target horizon (t+horizon)
            weather_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction']

            for feature in weather_features:
                if feature in df.columns:
                    # Shift weather data to target horizon
                    df[f'{feature}_target_{horizon}h'] = df[feature].shift(-horizon)

                    # Create scaled versions for target horizon weather
                    if f'{feature}_scaled' in df.columns:
                        df[f'{feature}_scaled_target_{horizon}h'] = df[f'{feature}_scaled'].shift(-horizon)

            # Pollutant features at target horizon (CO, O3, SO2, NH3)
            pollutant_features = ['carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']
            for pollutant in pollutant_features:
                if pollutant in df.columns:
                    df[f'{pollutant}_target_{horizon}h'] = df[pollutant].shift(-horizon)
                scaled_name = f'{pollutant}_scaled'
                if scaled_name in df.columns:
                    df[f'{scaled_name}_target_{horizon}h'] = df[scaled_name].shift(-horizon)

            # Create wind direction features at target horizon
            if 'wind_direction_sin' in df.columns and 'wind_direction_cos' in df.columns:
                df[f'wind_direction_sin_target_{horizon}h'] = df['wind_direction_sin'].shift(-horizon)
                df[f'wind_direction_cos_target_{horizon}h'] = df['wind_direction_cos'].shift(-horizon)

            # Create binary features at target horizon
            binary_features = ['is_hot', 'is_night', 'is_morning_rush', 'is_evening_rush']
            for feature in binary_features:
                if feature in df.columns:
                    df[f'{feature}_target_{horizon}h'] = df[feature].shift(-horizon)

            # Create time features at target horizon (t+horizon)
            time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                            'day_of_week_sin', 'day_of_week_cos']

            for feature in time_features:
                if feature in df.columns:
                    df[f'{feature}_target_{horizon}h'] = df[feature].shift(-horizon)

            # Create weather interaction features at target horizon
            weather_interaction_features = [
                'temp_humidity_interaction', 'temp_wind_interaction',
                'wind_direction_temp_interaction', 'wind_direction_humidity_interaction',
                'pressure_humidity_interaction'
            ]

            for feature in weather_interaction_features:
                if feature in df.columns:
                    df[f'{feature}_target_{horizon}h'] = df[feature].shift(-horizon)
                    # Also create scaled versions if they exist
                    if f'{feature}_scaled' in df.columns:
                        df[f'{feature}_scaled_target_{horizon}h'] = df[f'{feature}_scaled'].shift(-horizon)

        logger.info(f"Created target horizon weather, time, and interaction features for all horizons")

        return df

    def get_features_for_multi_output(self, df: pd.DataFrame, horizons: List[int], use_target_horizon: bool = False) -> List[str]:
        """Get features for multi-output model.
        If use_target_horizon=True, include target-horizon features for provided horizons.
        """

        multi_output_features = [
            # Current pollutants (SCALED)
            'pm2_5_log_scaled', 'pm10_scaled', 'carbon_monoxide_scaled', 'ozone_scaled', 
            'sulphur_dioxide_scaled', 'nh3_scaled',
            
            # Current weather features (SCALED)
            'temperature_scaled', 'humidity_scaled', 'pressure_scaled', 'wind_speed_scaled',
            'wind_direction_sin', 'wind_direction_cos',
            
            # Current time features (NO SCALING)
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'day_of_week_sin', 'day_of_week_cos',
            
            # Current binary features (NO SCALING)
            'is_hot', 'is_night', 'is_morning_rush', 'is_evening_rush',
            'is_high_pm2_5', 'is_high_o3',
            
            # Short-term lag features (SCALED)
            'pm2_5_lag_1h_scaled', 'pm2_5_lag_2h_scaled', 'pm2_5_lag_3h_scaled', 'pm2_5_lag_24h_scaled',
            'pm10_lag_1h_scaled', 'pm10_lag_2h_scaled', 'pm10_lag_3h_scaled', 'pm10_lag_24h_scaled',
            'co_lag_1h_scaled', 'o3_lag_1h_scaled', 'so2_lag_1h_scaled',
            
            # Rolling features (SCALED)
            'pm2_5_rolling_min_3h_scaled', 'pm2_5_rolling_mean_3h_scaled', 'pm2_5_rolling_max_3h_scaled', 'pm2_5_rolling_min_12h_scaled', 'pm2_5_rolling_mean_12h_scaled', 'pm2_5_rolling_max_12h_scaled', 'pm2_5_rolling_mean_24h_scaled', 'pm2_5_rolling_max_24h_scaled',
            'pm10_rolling_min_3h_scaled', 'pm10_rolling_mean_3h_scaled', 'pm10_rolling_mean_12h_scaled', 'pm10_rolling_mean_24h_scaled',
            
            # Change rates (SCALED)
            'pm2_5_change_rate_1h_scaled', 'pm2_5_change_rate_6h_scaled', 'pm2_5_change_rate_24h_scaled',
            'pm10_change_rate_1h_scaled', 'pm10_change_rate_6h_scaled', 'pm10_change_rate_24h_scaled',
            
            # PM × weather interactions (SCALED)
            'pm2_5_temp_interaction_scaled', 'pm2_5_humidity_interaction_scaled', 'pm2_5_pressure_interaction_scaled',
            'pm10_temperature_interaction_scaled', 'pm10_pressure_interaction_scaled',
            
            # Weather interactions (SCALED)
            'temp_humidity_interaction_scaled', 'temp_wind_interaction_scaled',
            'wind_direction_temp_interaction_scaled', 'wind_direction_humidity_interaction_scaled',
            'pressure_humidity_interaction_scaled',
            
            # Pollutant-weather interactions (SCALED)
            'co_pressure_interaction_scaled', 'o3_temp_interaction_scaled', 'so2_humidity_interaction_scaled'
        ]
        features = [f for f in multi_output_features if f in df.columns]

        if use_target_horizon:
            # Add target-horizon exogenous for requested horizons
            add_cols = []
            for h in horizons:
                add_cols += [
                    f'temperature_scaled_target_{h}h', f'humidity_scaled_target_{h}h', f'pressure_scaled_target_{h}h', f'wind_speed_scaled_target_{h}h',
                    f'wind_direction_sin_target_{h}h', f'wind_direction_cos_target_{h}h',
                    f'carbon_monoxide_scaled_target_{h}h', f'ozone_scaled_target_{h}h', f'sulphur_dioxide_scaled_target_{h}h', f'nh3_scaled_target_{h}h'
                ]
            features += [c for c in add_cols if c in df.columns]

        return features

    def prepare_training_data(self, df: pd.DataFrame, horizons: List[int], use_target_horizon: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data for multi-output model
        """
        print(f"📋 Preparing training data for multi-output model...")

        # Step 1: Apply transformations (same as original approach)
        df = self.apply_log_transformation(df)
        df = self.scale_features_consistently(df)

        # Step 2: Create target variables for all horizons
        df = self.create_target_variables(df, horizons)

        # Step 3: Optionally add target horizon features
        if use_target_horizon:
            df = self.create_target_horizon_weather_features(df, horizons)
            print(f"✅ Included target-horizon exogenous features for horizons: {horizons[:3]}... +")
        else:
            print(f"⏭️ Skipping target horizon features - using only current features")

        # Step 4: Get features for multi-output model
        feature_columns = self.get_features_for_multi_output(df, horizons, use_target_horizon=use_target_horizon)
        self.feature_columns['multi_output'] = feature_columns

        print(f"   Selected {len(feature_columns)} features for multi-output model")

        # Step 5: Remove rows with NaN targets (end of dataset)
        target_columns = []
        for horizon in horizons:
            target_columns.extend([f'pm2_5_target_{horizon}h', f'pm10_target_{horizon}h'])

        valid_indices = ~df[target_columns].isna().any(axis=1)
        df_clean = df[valid_indices].copy()

        print(f"   Cleaned dataset: {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows with NaN targets)")

        # Step 6: Split into features and targets
        X = df_clean[feature_columns]

        # Create target matrix for all horizons
        y_pm2_5 = df_clean[[f'pm2_5_target_{h}h' for h in horizons]]
        y_pm10 = df_clean[[f'pm10_target_{h}h' for h in horizons]]

        # Step 7: Train-test split
        X_train, X_test, y_pm2_5_train, y_pm2_5_test = train_test_split(
            X, y_pm2_5, test_size=self.model_config['test_size'], random_state=42, shuffle=False
        )
        _, _, y_pm10_train, y_pm10_test = train_test_split(
            X, y_pm10, test_size=self.model_config['test_size'], random_state=42, shuffle=False
        )

        print(f"   Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        print(f"   Target shape: PM2.5 {y_pm2_5_train.shape}, PM10 {y_pm10_train.shape}")

        # Validate data alignment
        if not self.validate_data_alignment(X_train, X_test, y_pm2_5_train, y_pm2_5_test, y_pm10_train, y_pm10_test):
            raise ValueError(f"❌ Critical data alignment issues detected for multi-output model!")

        return X_train, X_test, y_pm2_5_train, y_pm2_5_test, y_pm10_train, y_pm10_test

    def train_multi_output_models(self, df: pd.DataFrame, horizons: List[int]) -> Dict:
        """
        Train multi-output LightGBM models
        """
        print(f"\n🌳 Training Multi-Output LightGBM models for all horizons: {horizons}")

        # Prepare training data
        X_train, X_test, y_pm2_5_train, y_pm2_5_test, y_pm10_train, y_pm10_test = self.prepare_training_data(df, horizons)

        # Store test features for downstream explainability
        self.X_test = X_test

        # Base LightGBM with tuned config
        base_lgbm = LGBMRegressor(
            n_estimators=self.model_config.get('n_estimators', 1800),
            learning_rate=self.model_config.get('learning_rate', 0.03),
            num_leaves=self.model_config.get('num_leaves', 63),
            max_depth=self.model_config.get('max_depth', 10),
            min_data_in_leaf=self.model_config.get('min_data_in_leaf', 140),
            feature_fraction=self.model_config.get('feature_fraction', 0.7),
            bagging_fraction=self.model_config.get('bagging_fraction', 0.7),
            bagging_freq=self.model_config.get('bagging_freq', 1),
            lambda_l1=self.model_config.get('lambda_l1', 0.5),
            lambda_l2=self.model_config.get('lambda_l2', 0.5),
            random_state=self.model_config['random_state'],
            n_jobs=self.model_config['n_jobs'],
            verbose=-1,
        )


        lgbm_pm2_5 = MultiOutputRegressor(base_lgbm)
        lgbm_pm10 = MultiOutputRegressor(base_lgbm)

        # Train PM2.5 model
        print(f"   Training Multi-Output PM2.5 model...")
        lgbm_pm2_5.fit(X_train, y_pm2_5_train)

        # Train PM10 model
        print(f"   Training Multi-Output PM10 model...")
        lgbm_pm10.fit(X_train, y_pm10_train)

        # Evaluate models
        pm2_5_pred = lgbm_pm2_5.predict(X_test)
        pm10_pred = lgbm_pm10.predict(X_test)

        # Calculate metrics for each horizon
        results = {}

        for i, horizon in enumerate(horizons):
            # Get predictions for this horizon
            pm2_5_pred_horizon = pm2_5_pred[:, i]
            pm10_pred_horizon = pm10_pred[:, i]

            # Get true values for this horizon
            y_pm2_5_test_horizon = y_pm2_5_test.iloc[:, i]
            y_pm10_test_horizon = y_pm10_test.iloc[:, i]

            # Calculate metrics
            pm2_5_rmse = np.sqrt(mean_squared_error(y_pm2_5_test_horizon, pm2_5_pred_horizon))
            pm10_rmse = np.sqrt(mean_squared_error(y_pm10_test_horizon, pm10_pred_horizon))

            pm2_5_mae = mean_absolute_error(y_pm2_5_test_horizon, pm2_5_pred_horizon)
            pm10_mae = mean_absolute_error(y_pm10_test_horizon, pm10_pred_horizon)

            pm2_5_r2 = r2_score(y_pm2_5_test_horizon, pm2_5_pred_horizon)
            pm10_r2 = r2_score(y_pm10_test_horizon, pm10_pred_horizon)

            # Calculate train metrics for comparison
            pm2_5_train_pred = lgbm_pm2_5.predict(X_train)
            pm10_train_pred = lgbm_pm10.predict(X_train)

            pm2_5_train_pred_horizon = pm2_5_train_pred[:, i]
            pm10_train_pred_horizon = pm10_train_pred[:, i]

            y_pm2_5_train_horizon = y_pm2_5_train.iloc[:, i]
            y_pm10_train_horizon = y_pm10_train.iloc[:, i]

            pm2_5_train_r2 = r2_score(y_pm2_5_train_horizon, pm2_5_train_pred_horizon)
            pm10_train_r2 = r2_score(y_pm10_train_horizon, pm10_train_pred_horizon)

            pm2_5_train_rmse = np.sqrt(mean_squared_error(y_pm2_5_train_horizon, pm2_5_train_pred_horizon))
            pm10_train_rmse = np.sqrt(mean_squared_error(y_pm10_train_horizon, pm10_train_pred_horizon))

            print(f"\n   📊 {horizon}h horizon results:")
            print(f"      PM2.5 - Train R²: {pm2_5_train_r2:.3f}, Test R²: {pm2_5_r2:.3f}")
            print(f"      PM2.5 - Train RMSE: {pm2_5_train_rmse:.2f}, Test RMSE: {pm2_5_rmse:.2f}")
            print(f"      PM10 - Train R²: {pm10_train_r2:.3f}, Test R²: {pm10_r2:.3f}")
            print(f"      PM10 - Train RMSE: {pm10_train_rmse:.2f}, Test RMSE: {pm10_rmse:.2f}")

            # Check for overfitting
            pm2_5_overfit = pm2_5_train_r2 - pm2_5_r2
            pm10_overfit = pm10_train_r2 - pm10_r2

            if pm2_5_overfit > 0.1:
                print(f"      ⚠️ PM2.5 overfitting detected: Train-Test R² diff = {pm2_5_overfit:.3f}")
            if pm10_overfit > 0.1:
                print(f"      ⚠️ PM10 overfitting detected: Train-Test R² diff = {pm10_overfit:.3f}")

            # Store results for this horizon
            results[f'{horizon}h'] = {
                'pm2_5_rmse': pm2_5_rmse,
                'pm10_rmse': pm10_rmse,
                'pm2_5_mae': pm2_5_mae,
                'pm10_mae': pm10_mae,
                'pm2_5_r2': pm2_5_r2,
                'pm10_r2': pm10_r2,
                'pm2_5_train_r2': pm2_5_train_r2,
                'pm10_train_r2': pm10_train_r2,
                'pm2_5_train_rmse': pm2_5_train_rmse,
                'pm10_train_rmse': pm10_train_rmse,
                'y_test_pm2_5': y_pm2_5_test_horizon,
                'y_pred_pm2_5': pm2_5_pred_horizon,
                'y_test_pm10': y_pm10_test_horizon,
                'y_pred_pm10': pm10_pred_horizon
            }

        # Store models
        self.models['pm2_5_multi_output'] = lgbm_pm2_5
        self.models['pm10_multi_output'] = lgbm_pm10

        # Store overall results
        self.results = results

        return results

    def validate_data_alignment(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_pm2_5_train: pd.DataFrame, y_pm2_5_test: pd.DataFrame,
                               y_pm10_train: pd.DataFrame, y_pm10_test: pd.DataFrame) -> bool:
        """
        Validate critical data alignment issues for Multi-Output LightGBM
        """
        print(f"🔍 Validating data alignment for Multi-Output LightGBM...")

        # Check 1: Shapes consistency
        if len(X_train) != len(y_pm2_5_train) or len(X_train) != len(y_pm10_train):
            print(f"❌ Training shape mismatch: X_train={len(X_train)}, y_pm2_5_train={len(y_pm2_5_train)}, y_pm10_train={len(y_pm10_train)}")
            return False

        if len(X_test) != len(y_pm2_5_test) or len(X_test) != len(y_pm10_test):
            print(f"❌ Test shape mismatch: X_test={len(X_test)}, y_pm2_5_test={len(y_pm2_5_test)}, y_pm10_test={len(y_pm10_test)}")
            return False

        # Check 2: Data ranges (should be in original scale for LGBM)
        if X_train.min().min() < -10 or X_train.max().max() > 10:
            print(f"⚠️ Training features range suspicious: [{X_train.min().min():.3f}, {X_train.max().max():.3f}]")

        if X_test.min().min() < -10 or X_test.max().max() > 10:
            print(f"⚠️ Test features range suspicious: [{X_test.min().min():.3f}, {X_test.max().max():.3f}]")

        # Check 3: Target scaling validation (should be in original scale)
        if abs(y_pm2_5_train.mean().mean()) > 100 or abs(y_pm10_train.mean().mean()) > 200:
            print(f"⚠️ Training targets not in expected range: PM2.5 mean={y_pm2_5_train.mean().mean():.3f}, PM10 mean={y_pm10_train.mean().mean():.3f}")

        if abs(y_pm2_5_test.mean().mean()) > 100 or abs(y_pm10_test.mean().mean()) > 200:
            print(f"⚠️ Test targets not in expected range: PM2.5 mean={y_pm2_5_test.mean().mean():.3f}, PM10 mean={y_pm10_test.mean().mean():.3f}")

        # Check 4: No NaN values
        if X_train.isna().any().any() or X_test.isna().any().any():
            print(f"❌ NaN values found in features")
            return False

        if y_pm2_5_train.isna().any().any() or y_pm2_5_test.isna().any().any():
            print(f"❌ NaN values found in PM2.5 targets")
            return False

        if y_pm10_train.isna().any().any() or y_pm10_test.isna().any().any():
            print(f"❌ NaN values found in PM10 targets")
            return False

        print(f"✅ Data alignment validation passed!")
        return True

print("✅ LightGBMMultiOutputTrainer class defined!")

# Initialize and train the model
print(f"🌳 Initializing Multi-Output LightGBM Trainer")
trainer = LightGBMMultiOutputTrainer(MODEL_CONFIG)

print(f"\n🏋️ Starting multi-output model training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Training for horizons: short={SHORT_HORIZONS}, mid/long={MIDLONG_HORIZONS}")

# Train short-band (current features)
results_short = trainer.train_multi_output_models(df_preprocessed, SHORT_HORIZONS)

# Train mid/long-band (include target horizon exogenous)
trainer_midlong = LightGBMMultiOutputTrainer(MODEL_CONFIG)
X_train_ml, X_test_ml, y25_tr_ml, y25_te_ml, y10_tr_ml, y10_te_ml = trainer_midlong.prepare_training_data(
    df_preprocessed, MIDLONG_HORIZONS, use_target_horizon=True
)
base_lgbm_ml = LGBMRegressor(
    n_estimators=MODEL_CONFIG.get('n_estimators', 1800),
    learning_rate=MODEL_CONFIG.get('learning_rate', 0.03),
    num_leaves=MODEL_CONFIG.get('num_leaves', 63),
    max_depth=MODEL_CONFIG.get('max_depth', 10),
    min_data_in_leaf=MODEL_CONFIG.get('min_data_in_leaf', 140),
    feature_fraction=MODEL_CONFIG.get('feature_fraction', 0.7),
    bagging_fraction=MODEL_CONFIG.get('bagging_fraction', 0.7),
    bagging_freq=MODEL_CONFIG.get('bagging_freq', 1),
    lambda_l1=MODEL_CONFIG.get('lambda_l1', 0.5),
    lambda_l2=MODEL_CONFIG.get('lambda_l2', 0.5),
    random_state=MODEL_CONFIG['random_state'],
    n_jobs=MODEL_CONFIG['n_jobs'],
    verbose=-1,
)
model25_ml = MultiOutputRegressor(base_lgbm_ml)
model10_ml = MultiOutputRegressor(base_lgbm_ml)
print("\n🌳 Training Mid/Long Multi-Output models with horizon exogenous...")
model25_ml.fit(X_train_ml, y25_tr_ml)
model10_ml.fit(X_train_ml, y10_tr_ml)

# Evaluate mid/long
pm25_ml_pred = model25_ml.predict(X_test_ml)
pm10_ml_pred = model10_ml.predict(X_test_ml)
results_midlong = {}
for i, h in enumerate(MIDLONG_HORIZONS):
    y25_true_i = y25_te_ml.iloc[:, i]
    y10_true_i = y10_te_ml.iloc[:, i]
    y25_pred_i = pm25_ml_pred[:, i]
    y10_pred_i = pm10_ml_pred[:, i]
    r = {
        'pm2_5_rmse': float(np.sqrt(mean_squared_error(y25_true_i, y25_pred_i))),
        'pm10_rmse': float(np.sqrt(mean_squared_error(y10_true_i, y10_pred_i))),
        'pm2_5_mae': float(mean_absolute_error(y25_true_i, y25_pred_i)),
        'pm10_mae': float(mean_absolute_error(y10_true_i, y10_pred_i)),
        'pm2_5_r2': float(r2_score(y25_true_i, y25_pred_i)),
        'pm10_r2': float(r2_score(y10_true_i, y10_pred_i)),
        'y_test_pm2_5': y25_true_i,
        'y_pred_pm2_5': y25_pred_i,
        'y_test_pm10': y10_true_i,
        'y_pred_pm10': y10_pred_i,
    }
    results_midlong[f'{h}h'] = r

# Merge results for reporting
results = {**results_short, **results_midlong}

print(f"\n🎉 Multi-output training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}!")

# ----------------------
# SHAP explainability
# ----------------------
try:
    shap_output_dir = os.path.join(PATHS['logs_dir'], 'shap')
    _ensure_dir(shap_output_dir)

    def compute_shap_for_multioutput(multi_model: MultiOutputRegressor, X_test: pd.DataFrame, feature_names: List[str], 
                                     band_label: str, target_label: str, horizons: List[int], plot_horizons: List[int]):
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not installed; skipping SHAP plots.")
            return
        # Map requested horizons to estimator indices
        horizon_to_idx = {h: i for i, h in enumerate(horizons)}
        for h in plot_horizons:
            if h not in horizon_to_idx:
                continue
            idx = horizon_to_idx[h]
            if idx >= len(multi_model.estimators_):
                continue
            est = multi_model.estimators_[idx]
            title = f"{target_label.upper()} | {band_label} | h={h}"
            prefix = os.path.join(shap_output_dir, f"{band_label}_{target_label}_h{h}")
            compute_shap_for_estimator(est, X_test, feature_names, title, prefix)

    # Short band SHAP for selected horizons
    short_plot_h = [1, 6, 12]
    short_feature_names = list(trainer.X_test.columns) if hasattr(trainer, 'X_test') else []
    if short_feature_names:
        compute_shap_for_multioutput(trainer.models['pm2_5_multi_output'], trainer.X_test, short_feature_names, 'short', 'pm25', SHORT_HORIZONS, short_plot_h)
        compute_shap_for_multioutput(trainer.models['pm10_multi_output'], trainer.X_test, short_feature_names, 'short', 'pm10', SHORT_HORIZONS, short_plot_h)

    # Mid/Long band SHAP for selected horizons
    midlong_plot_h = [12, 24, 48, 72]
    ml_feature_names = list(X_test_ml.columns)
    compute_shap_for_multioutput(model25_ml, X_test_ml, ml_feature_names, 'midlong', 'pm25', MIDLONG_HORIZONS, midlong_plot_h)
    compute_shap_for_multioutput(model10_ml, X_test_ml, ml_feature_names, 'midlong', 'pm10', MIDLONG_HORIZONS, midlong_plot_h)
    print(f"✅ SHAP plots saved to {shap_output_dir}")
except Exception as e:
    print(f"⚠️ SHAP explainability step skipped due to error: {e}")

# Results Analysis and Visualization
print("\n📊 Analyzing Multi-Output Results...")

# Display summary results
for horizon, horizon_results in results.items():
    print(f"\n🔍 {horizon} Results:")
    print(f"   PM2.5: RMSE={horizon_results['pm2_5_rmse']:.2f}, R²={horizon_results['pm2_5_r2']:.3f}")
    print(f"   PM10:  RMSE={horizon_results['pm10_rmse']:.2f}, R²={horizon_results['pm10_r2']:.3f}")

# Plot model performance comparison
import matplotlib.pyplot as plt

horizons_int = [int(h.replace('h', '')) for h in results.keys()]
pm25_rmse = [results[f'{h}h']['pm2_5_rmse'] for h in horizons_int]
pm25_r2 = [results[f'{h}h']['pm2_5_r2'] for h in horizons_int]
pm10_rmse = [results[f'{h}h']['pm10_rmse'] for h in horizons_int]
pm10_r2 = [results[f'{h}h']['pm10_r2'] for h in horizons_int]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

# PM2.5 RMSE
ax1.plot(horizons_int, pm25_rmse, 'o-', linewidth=2, markersize=8, color='blue')
ax1.set_title('PM2.5 RMSE by Horizon (Multi-Output LightGBM)')
ax1.set_xlabel('Forecast Horizon (hours)')
ax1.set_ylabel('RMSE')
ax1.grid(True, alpha=0.3)

# PM2.5 R²
ax2.plot(horizons_int, pm25_r2, 'o-', linewidth=2, markersize=8, color='orange')
ax2.set_title('PM2.5 R² by Horizon (Multi-Output LightGBM)')
ax2.set_xlabel('Forecast Horizon (hours)')
ax2.set_ylabel('R² Score')
ax2.grid(True, alpha=0.3)

# PM10 RMSE
ax3.plot(horizons_int, pm10_rmse, 'o-', linewidth=2, markersize=8, color='green')
ax3.set_title('PM10 RMSE by Horizon (Multi-Output LightGBM)')
ax3.set_xlabel('Forecast Horizon (hours)')
ax3.set_ylabel('RMSE')
ax3.grid(True, alpha=0.3)

# PM10 R²
ax4.plot(horizons_int, pm10_r2, 'o-', linewidth=2, markersize=8, color='red')
ax4.set_title('PM10 R² by Horizon (Multi-Output LightGBM)')
ax4.set_xlabel('Forecast Horizon (hours)')
ax4.set_ylabel('R² Score')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Sample validation: Show actual predicted vs true values for a small subset
print("\n🔍 Sample Validation: Predicted vs True Values (First 10 samples)")
print("=" * 80)

for horizon, horizon_results in results.items():
    print(f"\n📊 {horizon} Horizon - Sample Comparison:")
    # Guard: skip horizons without stored sample arrays
    if not all(k in horizon_results for k in ['y_test_pm2_5', 'y_pred_pm2_5', 'y_test_pm10', 'y_pred_pm10']):
        print("   (Skipping sample print: no stored y_true/y_pred for this horizon)")
        continue

    # Get first 10 samples
    y_true_pm25 = horizon_results['y_test_pm2_5'].iloc[:10]
    y_pred_pm25 = horizon_results['y_pred_pm2_5'][:10]
    y_true_pm10 = horizon_results['y_test_pm10'].iloc[:10]
    y_pred_pm10 = horizon_results['y_pred_pm10'][:10]
    
    print(f"   PM2.5 (μg/m³):")
    print(f"   {'Index':<6} {'True':<8} {'Predicted':<10} {'Diff':<8} {'% Error':<8}")
    print(f"   {'-'*50}")
    for i in range(len(y_true_pm25)):
        true_val = y_true_pm25.iloc[i]
        pred_val = y_pred_pm25[i]
        diff = pred_val - true_val
        pct_error = (abs(diff) / true_val) * 100 if true_val != 0 else 0
        print(f"   {i:<6} {true_val:<8.2f} {pred_val:<10.2f} {diff:<8.2f} {pct_error:<8.1f}%")
    
    print(f"\n   PM10 (μg/m³):")
    print(f"   {'Index':<6} {'True':<8} {'Predicted':<10} {'Diff':<8} {'% Error':<8}")
    print(f"   {'-'*50}")
    for i in range(len(y_true_pm10)):
        true_val = y_true_pm10.iloc[i]
        pred_val = y_pred_pm10[i]
        diff = pred_val - true_val
        pct_error = (abs(diff) / true_val) * 100 if true_val != 0 else 0
        print(f"   {i:<6} {true_val:<8.2f} {pred_val:<10.2f} {diff:<8.2f} {pct_error:<8.1f}%")

print("\n" + "=" * 80)

# Feature Importance Analysis (for first horizon as representative)
print("\n🔍 Feature Importance Analysis (Multi-Output Model)")
print("=" * 80)

# Get feature importance from the first estimator (representative of all)
pm25_model = trainer.models['pm2_5_multi_output']
pm10_model = trainer.models['pm10_multi_output']

# Get feature names
feature_names = trainer.feature_columns['multi_output']

# Get feature importance from first estimator (all estimators have same importance in LGBM)
pm25_importance = pm25_model.estimators_[0].feature_importances_
pm10_importance = pm10_model.estimators_[0].feature_importances_

# Create DataFrame for easier plotting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'PM2.5_Importance': pm25_importance,
    'PM10_Importance': pm10_importance
})

# Sort by PM2.5 importance (descending)
importance_df = importance_df.sort_values('PM2.5_Importance', ascending=False)

# Plot top 15 features for both PM2.5 and PM10
top_features = importance_df.head(15)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Create horizontal bar plot
y_pos = np.arange(len(top_features))
width = 0.35

ax.barh(y_pos - width/2, top_features['PM2.5_Importance'], width, 
        label='PM2.5', color='blue', alpha=0.7)
ax.barh(y_pos + width/2, top_features['PM10_Importance'], width, 
        label='PM10', color='orange', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_features['Feature'], fontsize=8)
ax.set_xlabel('Feature Importance')
ax.set_title('Multi-Output LightGBM - Top 15 Features')
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels on bars
for i, (pm25_val, pm10_val) in enumerate(zip(top_features['PM2.5_Importance'], top_features['PM10_Importance'])):
    ax.text(pm25_val + 0.001, i - width/2, f'{pm25_val:.3f}', 
            va='center', ha='left', fontsize=6)
    ax.text(pm10_val + 0.001, i + width/2, f'{pm10_val:.3f}', 
            va='center', ha='left', fontsize=6)

plt.tight_layout()
plt.show()

# Print top 10 most important features
print("\n📊 Top 10 Most Important Features (Multi-Output Model):")
print("=" * 80)

# Sort by combined importance
importance_df['Combined_Importance'] = importance_df['PM2.5_Importance'] + importance_df['PM10_Importance']
importance_df = importance_df.sort_values('Combined_Importance', ascending=False)

print(f"{'Rank':<4} {'Feature':<35} {'PM2.5_Imp':<10} {'PM10_Imp':<10} {'Combined':<10}")
print("-" * 75)

for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
    print(f"{i:<4} {row['Feature']:<35} {row['PM2.5_Importance']:<10.4f} {row['PM10_Importance']:<10.4f} {row['Combined_Importance']:<10.4f}")

print("\n" + "=" * 80)
print("✅ Multi-Output LightGBM analysis complete!") 