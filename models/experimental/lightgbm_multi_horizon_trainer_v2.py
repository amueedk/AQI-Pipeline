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
from typing import Dict, List, Tuple
import joblib
import json
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ Imports completed successfully!")

try:
    from config import PATHS, HOPSWORKS_CONFIG
    print("✅ Loaded config from config.py")
except ImportError:
    PATHS = {'logs_dir': 'logs', 'temp_dir': 'temp', 'data_dir': 'data'}
    print("⚠️ Using default paths (config.py not found)")

for p in PATHS.values():
    os.makedirs(p, exist_ok=True)

MODEL_CONFIG: Dict = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 30,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'test_size': 0.2,
    'num_steps': 72,
}

HORIZONS_ALL = list(range(1, MODEL_CONFIG['num_steps'] + 1))

print(f"📋 v3 Configuration:")
print(f"   Steps (autoregressive): {HORIZONS_ALL[-1]}h")
print(f"   Base model: LightGBM 1-step (n_estimators={MODEL_CONFIG['n_estimators']})")


class LightGBMAutoregressiveV3:
    def __init__(self, model_config: Dict | None = None) -> None:
        self.model_config = model_config or MODEL_CONFIG
        self.model: MultiOutputRegressor | None = None
        self.feature_columns: List[str] = []
        self.history = None

        # Scalers
        self.pm_scaler = StandardScaler()
        self.weather_scaler = StandardScaler()
        self.pollutant_scaler = StandardScaler()
        self.interaction_scaler = StandardScaler()

        # Keep feature order used to fit pm_scaler so we can scale preds for lag updates
        self.pm_scaler_feature_list: List[str] = []

    def apply_log_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'pm2_5' in df.columns:
            df['pm2_5_log'] = np.log1p(df['pm2_5'])
        return df

    def scale_features_consistently(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # PM groups
        pm_features = [
            'pm2_5_log', 'pm10',
            'pm2_5_lag_1h', 'pm2_5_lag_2h', 'pm2_5_lag_3h', 'pm2_5_lag_24h',
            'pm10_lag_1h', 'pm10_lag_2h', 'pm10_lag_3h', 'pm10_lag_24h',
            'pm2_5_rolling_min_3h', 'pm2_5_rolling_mean_3h', 'pm2_5_rolling_max_3h',
            'pm2_5_rolling_min_12h', 'pm2_5_rolling_mean_12h', 'pm2_5_rolling_max_12h',
            'pm2_5_rolling_mean_24h', 'pm2_5_rolling_max_24h',
            'pm10_rolling_min_3h', 'pm10_rolling_mean_3h', 'pm10_rolling_mean_12h', 'pm10_rolling_mean_24h',
            'pm2_5_change_rate_1h', 'pm2_5_change_rate_6h', 'pm2_5_change_rate_24h',
            'pm10_change_rate_1h', 'pm10_change_rate_6h', 'pm10_change_rate_24h',
        ]
        avail_pm = [c for c in pm_features if c in df.columns]
        if avail_pm:
            v = df[avail_pm].values
            s = self.pm_scaler.fit_transform(v)
            # Record order for later scaling of lag updates
            self.pm_scaler_feature_list = list(avail_pm)
            for i, c in enumerate(avail_pm):
                df[f'{c}_scaled'] = s[:, i]

        weather_features = ['temperature', 'humidity', 'pressure', 'wind_speed']
        avail_w = [c for c in weather_features if c in df.columns]
        if avail_w:
            v = df[avail_w].values
            s = self.weather_scaler.fit_transform(v)
            for i, c in enumerate(avail_w):
                df[f'{c}_scaled'] = s[:, i]

        pollutant_features = ['carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']
        avail_p = [c for c in pollutant_features if c in df.columns]
        if avail_p:
            v = df[avail_p].values
            s = self.pollutant_scaler.fit_transform(v)
            for i, c in enumerate(avail_p):
                df[f'{c}_scaled'] = s[:, i]

        interaction_features = [
            'pm2_5_temp_interaction', 'pm2_5_humidity_interaction', 'pm2_5_pressure_interaction',
            'pm10_temperature_interaction', 'pm10_pressure_interaction',
            'temp_humidity_interaction', 'temp_wind_interaction',
            'wind_direction_temp_interaction', 'wind_direction_humidity_interaction',
            'pressure_humidity_interaction',
            'co_pressure_interaction', 'o3_temp_interaction', 'so2_humidity_interaction',
        ]
        avail_i = [c for c in interaction_features if c in df.columns]
        if avail_i:
            v = df[avail_i].values
            s = self.interaction_scaler.fit_transform(v)
            for i, c in enumerate(avail_i):
                df[f'{c}_scaled'] = s[:, i]

        return df

    @staticmethod
    def _decoder_step_candidates(df: pd.DataFrame) -> List[str]:
        candidates = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction_sin', 'wind_direction_cos',
            'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3'
        ]
        return [c for c in candidates if c in df.columns]

    def add_horizon_exogenous_scaled(self, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        df = df.copy()
        decoder_feats = self._decoder_step_candidates(df)
        # For features that have scaled counterparts, shift their scaled columns. For wind dirs (sin/cos) keep raw.
        for h in horizons:
            for feat in decoder_feats:
                scaled_col = f'{feat}_scaled'
                if scaled_col in df.columns:
                    df[f'{feat}_scaled_target_{h}h'] = df[scaled_col].shift(-h)
                else:
                    # wind_direction_sin/cos or others without scaler
                    df[f'{feat}_target_{h}h'] = df[feat].shift(-h)
        return df

    def get_feature_list(self, df: pd.DataFrame) -> List[str]:
        base = [
            # Current pollutants (SCALED)
            'pm2_5_log_scaled', 'pm10_scaled', 'carbon_monoxide_scaled', 'ozone_scaled', 'sulphur_dioxide_scaled', 'nh3_scaled',
            # Current weather (SCALED) + wind dir raw
            'temperature_scaled', 'humidity_scaled', 'pressure_scaled', 'wind_speed_scaled',
            'wind_direction_sin', 'wind_direction_cos',
            # Time
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            # Binary
            'is_hot', 'is_night', 'is_morning_rush', 'is_evening_rush', 'is_high_pm2_5', 'is_high_o3',
            # Lags (SCALED)
            'pm2_5_lag_1h_scaled', 'pm2_5_lag_2h_scaled', 'pm2_5_lag_3h_scaled', 'pm2_5_lag_24h_scaled',
            'pm10_lag_1h_scaled', 'pm10_lag_2h_scaled', 'pm10_lag_3h_scaled', 'pm10_lag_24h_scaled',
            # Rolling (SCALED)
            'pm2_5_rolling_min_3h_scaled', 'pm2_5_rolling_mean_3h_scaled', 'pm2_5_rolling_max_3h_scaled',
            'pm2_5_rolling_min_12h_scaled', 'pm2_5_rolling_mean_12h_scaled', 'pm2_5_rolling_max_12h_scaled',
            'pm2_5_rolling_mean_24h_scaled', 'pm2_5_rolling_max_24h_scaled',
            'pm10_rolling_min_3h_scaled', 'pm10_rolling_mean_3h_scaled', 'pm10_rolling_mean_12h_scaled', 'pm10_rolling_mean_24h_scaled',
            # Change rates (SCALED)
            'pm2_5_change_rate_1h_scaled', 'pm2_5_change_rate_6h_scaled', 'pm2_5_change_rate_24h_scaled',
            'pm10_change_rate_1h_scaled', 'pm10_change_rate_6h_scaled', 'pm10_change_rate_24h_scaled',
            # Interactions (SCALED)
            'pm2_5_temp_interaction_scaled', 'pm2_5_humidity_interaction_scaled', 'pm2_5_pressure_interaction_scaled',
            'pm10_temperature_interaction_scaled', 'pm10_pressure_interaction_scaled',
            'temp_humidity_interaction_scaled', 'temp_wind_interaction_scaled',
            'wind_direction_temp_interaction_scaled', 'wind_direction_humidity_interaction_scaled',
            'pressure_humidity_interaction_scaled',
            'co_pressure_interaction_scaled', 'o3_temp_interaction_scaled', 'so2_humidity_interaction_scaled',
            # Decoder step-based exogenous for 1h (training)
            'temperature_scaled_target_1h', 'humidity_scaled_target_1h', 'pressure_scaled_target_1h', 'wind_speed_scaled_target_1h',
            'wind_direction_sin_target_1h', 'wind_direction_cos_target_1h',
            'carbon_monoxide_scaled_target_1h', 'ozone_scaled_target_1h', 'sulphur_dioxide_scaled_target_1h', 'nh3_scaled_target_1h',
        ]
        return [c for c in base if c in df.columns]

    def prepare_training_data(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        df = df_raw.copy()
        df['__row_id__'] = np.arange(len(df))
        df = df.sort_values('time').reset_index(drop=True)

        df = self.apply_log_transformation(df)
        df = self.scale_features_consistently(df)
        # Build step-based exogenous features for 1..72 so we can use them in rollout
        df = self.add_horizon_exogenous_scaled(df, HORIZONS_ALL)

        # Targets: 1-step ahead only
        df['pm2_5_target_1h'] = df['pm2_5'].shift(-1)
        df['pm10_target_1h'] = df['pm10'].shift(-1)

        # Remove rows with NaN targets or NaN in features
        feature_cols = self.get_feature_list(df)
        req_cols = feature_cols + ['pm2_5_target_1h', 'pm10_target_1h', '__row_id__']
        df_clean = df[req_cols].dropna().reset_index(drop=True)

        X = df_clean[feature_cols]
        y_pm25 = df_clean['pm2_5_target_1h']
        y_pm10 = df_clean['pm10_target_1h']
        row_ids = df_clean[['__row_id__']]

        X_train, X_test, y25_tr, y25_te, y10_tr, y10_te, row_tr, row_te = train_test_split(
            X, y_pm25, y_pm10, row_ids, test_size=self.model_config['test_size'], random_state=42, shuffle=False
        )

        return X_train, X_test, y25_tr, y10_tr, y25_te.to_frame(), y10_te.to_frame(), row_tr, row_te

    def train(self, df_preprocessed: pd.DataFrame) -> Dict:
        X_train, X_test, y25_tr, y10_tr, y25_te_df, y10_te_df, row_tr, row_te = self.prepare_training_data(df_preprocessed)

        self.feature_columns = list(X_train.columns)
        base_lgbm = LGBMRegressor(
            n_estimators=self.model_config['n_estimators'],
            learning_rate=self.model_config['learning_rate'],
            num_leaves=self.model_config['num_leaves'],
            max_depth=self.model_config['max_depth'],
            min_data_in_leaf=self.model_config['min_data_in_leaf'],
            feature_fraction=self.model_config['feature_fraction'],
            bagging_fraction=self.model_config['bagging_fraction'],
            bagging_freq=self.model_config['bagging_freq'],
            lambda_l1=self.model_config['lambda_l1'],
            lambda_l2=self.model_config['lambda_l2'],
            random_state=self.model_config['random_state'],
            n_jobs=self.model_config['n_jobs'],
            verbose=-1,
        )
        self.model = MultiOutputRegressor(base_lgbm)
        self.model.fit(X_train, np.stack([y25_tr.values, y10_tr.values], axis=1))

        # 1-step test metrics
        preds_1step = self.model.predict(X_test)
        pm25_pred_1 = preds_1step[:, 0]
        pm10_pred_1 = preds_1step[:, 1]
        pm25_true_1 = y25_te_df.values.ravel()
        pm10_true_1 = y10_te_df.values.ravel()

        print("\n📊 1-step Test (t+1) Results:")
        print(f"   PM2.5: RMSE={np.sqrt(mean_squared_error(pm25_true_1, pm25_pred_1)):.2f}, R²={r2_score(pm25_true_1, pm25_pred_1):.3f}")
        print(f"   PM10 : RMSE={np.sqrt(mean_squared_error(pm10_true_1, pm10_pred_1)):.2f}, R²={r2_score(pm10_true_1, pm10_pred_1):.3f}")

        # Autoregressive rollout on test set
        results = self.rollout_autoregressive(df_preprocessed, X_test, row_te['__row_id__'].values)
        return results

    def _collect_step_features(self, df_feat: pd.DataFrame, base_row_idx: int, base_x_row: pd.Series, step_h: int) -> np.ndarray:
        x = base_x_row.copy()
        # Replace decoder step-based exogenous for this step
        repl_map = {
            'temperature_scaled_target_1h': f'temperature_scaled_target_{step_h}h',
            'humidity_scaled_target_1h': f'humidity_scaled_target_{step_h}h',
            'pressure_scaled_target_1h': f'pressure_scaled_target_{step_h}h',
            'wind_speed_scaled_target_1h': f'wind_speed_scaled_target_{step_h}h',
            'wind_direction_sin_target_1h': f'wind_direction_sin_target_{step_h}h',
            'wind_direction_cos_target_1h': f'wind_direction_cos_target_{step_h}h',
            'carbon_monoxide_scaled_target_1h': f'carbon_monoxide_scaled_target_{step_h}h',
            'ozone_scaled_target_1h': f'ozone_scaled_target_{step_h}h',
            'sulphur_dioxide_scaled_target_1h': f'sulphur_dioxide_scaled_target_{step_h}h',
            'nh3_scaled_target_1h': f'nh3_scaled_target_{step_h}h',
        }
        for k, v in repl_map.items():
            if k in x.index and v in df_feat.columns:
                x[k] = df_feat.loc[base_row_idx, v]
        return x.values.reshape(1, -1)

    def _scale_value_for_pm_feature(self, feature_name: str, raw_value: float) -> float:
        """
        Scale a raw value using the pm_scaler stats for a specific pm feature column.
        Returns scaled value; if feature not found, returns raw_value (fallback).
        """
        if not self.pm_scaler_feature_list:
            return raw_value
        try:
            idx = self.pm_scaler_feature_list.index(feature_name)
        except ValueError:
            return raw_value
        mean = float(self.pm_scaler.mean_[idx])
        scale = float(self.pm_scaler.scale_[idx]) if hasattr(self.pm_scaler, 'scale_') else 1.0
        if scale == 0:
            return 0.0
        return (raw_value - mean) / scale

    def rollout_autoregressive(self, df_raw: pd.DataFrame, X_test: pd.DataFrame, test_row_ids: np.ndarray) -> Dict:
        df = df_raw.copy()
        df = df.sort_values('time').reset_index(drop=True)
        df = self.apply_log_transformation(df)
        df = self.scale_features_consistently(df)
        df = self.add_horizon_exogenous_scaled(df, HORIZONS_ALL)

        horizons = HORIZONS_ALL
        out: Dict[str, Dict] = {}

        preds_pm25_by_h = {h: [] for h in horizons}
        preds_pm10_by_h = {h: [] for h in horizons}
        true_pm25_by_h = {h: [] for h in horizons}
        true_pm10_by_h = {h: [] for h in horizons}

        # Names for lag features we will update each step
        lag25_names = ['pm2_5_lag_1h_scaled', 'pm2_5_lag_2h_scaled', 'pm2_5_lag_3h_scaled']
        lag10_names = ['pm10_lag_1h_scaled', 'pm10_lag_2h_scaled', 'pm10_lag_3h_scaled']

        for xi, base_row_idx in enumerate(test_row_ids):
            # Mutable feature vector for this sample
            feat_vec = X_test.iloc[xi].copy()

            step_preds_pm25: List[float] = []
            step_preds_pm10: List[float] = []
            for h in horizons:
                # Update step-based exogenous features to t+h for this step
                repl_map = {
                    'temperature_scaled_target_1h': f'temperature_scaled_target_{h}h',
                    'humidity_scaled_target_1h': f'humidity_scaled_target_{h}h',
                    'pressure_scaled_target_1h': f'pressure_scaled_target_{h}h',
                    'wind_speed_scaled_target_1h': f'wind_speed_scaled_target_{h}h',
                    'wind_direction_sin_target_1h': f'wind_direction_sin_target_{h}h',
                    'wind_direction_cos_target_1h': f'wind_direction_cos_target_{h}h',
                    'carbon_monoxide_scaled_target_1h': f'carbon_monoxide_scaled_target_{h}h',
                    'ozone_scaled_target_1h': f'ozone_scaled_target_{h}h',
                    'sulphur_dioxide_scaled_target_1h': f'sulphur_dioxide_scaled_target_{h}h',
                    'nh3_scaled_target_1h': f'nh3_scaled_target_{h}h',
                }
                for k, v in repl_map.items():
                    if k in feat_vec.index and v in df.columns:
                        feat_vec[k] = df.loc[base_row_idx, v]

                # Predict next step
                y_hat = self.model.predict(feat_vec.values.reshape(1, -1))[0]
                pm25_hat, pm10_hat = float(y_hat[0]), float(y_hat[1])
                step_preds_pm25.append(pm25_hat)
                step_preds_pm10.append(pm10_hat)

                # Record metrics targets and preds if valid
                tgt_idx = base_row_idx + h
                if 0 <= tgt_idx < len(df):
                    true_pm25_by_h[h].append(float(df.loc[tgt_idx, 'pm2_5']))
                    true_pm10_by_h[h].append(float(df.loc[tgt_idx, 'pm10']))
                    preds_pm25_by_h[h].append(pm25_hat)
                    preds_pm10_by_h[h].append(pm10_hat)

                # Autoregressive update: shift lag scaled features and insert predicted as new lag_1h
                # Scale predicted raw PM to corresponding lag feature scales
                pm25_lag1_scaled = self._scale_value_for_pm_feature('pm2_5_lag_1h', pm25_hat)
                pm10_lag1_scaled = self._scale_value_for_pm_feature('pm10_lag_1h', pm10_hat)

                # Shift lag_1h -> lag_2h, lag_2h -> lag_3h
                if all(name in feat_vec.index for name in lag25_names):
                    feat_vec[lag25_names[2]] = feat_vec[lag25_names[1]]
                    feat_vec[lag25_names[1]] = feat_vec[lag25_names[0]]
                    feat_vec[lag25_names[0]] = pm25_lag1_scaled
                if all(name in feat_vec.index for name in lag10_names):
                    feat_vec[lag10_names[2]] = feat_vec[lag10_names[1]]
                    feat_vec[lag10_names[1]] = feat_vec[lag10_names[0]]
                    feat_vec[lag10_names[0]] = pm10_lag1_scaled

        # Metrics per horizon
        for h in horizons:
            if len(true_pm25_by_h[h]) == 0:
                continue
            t25 = np.array(true_pm25_by_h[h]); p25 = np.array(preds_pm25_by_h[h])
            t10 = np.array(true_pm10_by_h[h]); p10 = np.array(preds_pm10_by_h[h])
            res = {
                'pm2_5_rmse': float(np.sqrt(mean_squared_error(t25, p25))),
                'pm10_rmse': float(np.sqrt(mean_squared_error(t10, p10))),
                'pm2_5_mae': float(mean_absolute_error(t25, p25)),
                'pm10_mae': float(mean_absolute_error(t10, p10)),
                'pm2_5_r2': float(r2_score(t25, p25)),
                'pm10_r2': float(r2_score(t10, p10)),
            }
            out[f'{h}h'] = res

        # Print summary
        print("\n📊 Autoregressive (step-based exogenous + lag updates) Results:")
        for h in horizons:
            if f'{h}h' in out:
                r = out[f'{h}h']
                print(f"   {h}h → PM2.5 RMSE={r['pm2_5_rmse']:.2f}, R²={r['pm2_5_r2']:.3f} | PM10 RMSE={r['pm10_rmse']:.2f}, R²={r['pm10_r2']:.3f}")

        return out


print("✅ LightGBMAutoregressiveV3 class defined!")

if __name__ == '__main__':
    print("🌳 Initializing LightGBM Autoregressive v3 Trainer")

    # Backup CSV path(s); adjust if needed
    csv_files = [
        'new_feature_group_data_20250810_224420.csv'
    ]

    df_raw = None
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df_raw = pd.read_csv(csv_file)
            df_raw['time'] = pd.to_datetime(df_raw['time'], utc=True, errors='coerce')
            df_raw = df_raw.sort_values('time').reset_index(drop=True)
            print(f"✅ Loaded {len(df_raw)} rows from {csv_file}")
            break
    if df_raw is None:
        raise FileNotFoundError("No CSV files found for training")

    # Basic preprocessing similar to v2 script
    df_pre = df_raw.copy()
    df_pre = df_pre.drop_duplicates()
    df_pre = df_pre.sort_values('time').reset_index(drop=True)
    df_pre = df_pre.fillna(method='ffill').fillna(method='bfill')

    for col in ['pm2_5', 'pm10', 'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']:
        if col in df_pre.columns:
            Q1, Q3 = df_pre[col].quantile(0.25), df_pre[col].quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df_pre = df_pre[(df_pre[col] >= lb) & (df_pre[col] <= ub)]

    trainer = LightGBMAutoregressiveV3(MODEL_CONFIG)
    print(f"\n🏋️ Training 1-step model and running 72-step rollout at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = trainer.train(df_pre)

    # Simple plot of PM2.5 metrics by horizon
    horizons_int = [int(h.replace('h', '')) for h in results.keys()]
    pm25_rmse = [results[f'{h}h']['pm2_5_rmse'] for h in horizons_int]
    pm10_rmse = [results[f'{h}h']['pm10_rmse'] for h in horizons_int]
    pm25_r2 = [results[f'{h}h']['pm2_5_r2'] for h in horizons_int]
    pm10_r2 = [results[f'{h}h']['pm10_r2'] for h in horizons_int]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(horizons_int, pm25_rmse, 'o-', label='PM2.5 RMSE')
    ax1.plot(horizons_int, pm10_rmse, 'o-', label='PM10 RMSE')
    ax1.set_title('RMSE by Horizon (AR v3)'); ax1.set_xlabel('Horizon (h)'); ax1.set_ylabel('RMSE'); ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.plot(horizons_int, pm25_r2, 'o-', label='PM2.5 R2')
    ax2.plot(horizons_int, pm10_r2, 'o-', label='PM10 R2')
    ax2.set_title('R² by Horizon (AR v3)'); ax2.set_xlabel('Horizon (h)'); ax2.set_ylabel('R²'); ax2.grid(True, alpha=0.3); ax2.legend()

    plt.tight_layout(); plt.show()