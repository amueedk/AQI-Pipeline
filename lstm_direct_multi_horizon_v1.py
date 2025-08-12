"""
Direct Multi-Horizon LSTM (no autoregression): predicts all 72 horizons in one pass

Design:
- Encoder-only historical window → context vector
- Decoder stream = RepeatVector(72)(context) concatenated with:
  - sinusoidal positional embeddings per step [sin(2πh/72), cos(2πh/72)]
  - target-horizon exogenous features at t+h (weather + pollutants + wind dir), scaled
- LSTM(return_sequences=True) + TimeDistributed(Dense(2)) → [PM2.5, PM10] for h=1..72

Notes:
- Training == Inference (no teacher forcing, no feedback), so no exposure bias
- Horizon-weighted loss emphasizes 6–24h
"""

import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy

import joblib
import json
from hopsworks_integration import HopsworksUploader
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, TimeDistributed, RepeatVector, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

try:
    from config import PATHS, HOPSWORKS_CONFIG
except Exception:
    PATHS = {'logs_dir': 'logs', 'temp_dir': 'temp', 'data_dir': 'data'}
    HOPSWORKS_CONFIG = {'project_name': '', 'feature_group_name': ''}

for p in PATHS.values():
    os.makedirs(p, exist_ok=True)

# Optional Hopsworks integration
try:
    from hopsworks_integration import HopsworksUploader
except Exception:
    HopsworksUploader = None  # type: ignore


CONFIG: Dict = {
    'sequence_length': 72,
    'num_steps': 72,
    'encoder_units': [128, 64],
    'decoder_units': 192,
    'dropout_rate': 0.3,
    'learning_rate': 3e-4,
    'batch_size': 32,
    'epochs': 80,
    'test_size': 0.2,
    'early_stopping_patience': 12,
    'reduce_lr_patience': 6,
    'reduce_lr_factor': 0.6,
    'min_lr': 1e-6,
    # Loss weighting (overridden by banded runner)
    'weights_1_3': 5.0,
    'weights_4_6': 3.0,
    'weights_7_24': 3.0,
    'weights_25_72': 1.0,
    # Training target mode
    'use_delta_targets': True,
}


def preprocess_aqi_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
        df = df.sort_values('time')
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    # Drop remaining NaNs
    df = df.dropna().reset_index(drop=True)
    # Remove non-positive PM rows
    for col in ['pm2_5', 'pm10']:
        if col in df.columns:
            df = df[df[col] > 0].reset_index(drop=True)
    # IQR outliers for key pollutants
    for col in ['pm2_5', 'pm10', 'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']:
        if col in df.columns and df[col].notna().any():
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df = df[(df[col] >= lb) & (df[col] <= ub)].reset_index(drop=True)
    print(f"✅ Preprocessing applied. Rows after cleaning: {len(df)}")
    return df


class DirectLSTMMultiHorizon:
    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or CONFIG
        # Scalers for groups
        self.pm_scaler = StandardScaler()
        self.target_pm_scaler = StandardScaler()
        self.weather_scaler = StandardScaler()
        self.pollutant_scaler = StandardScaler()
        self.interaction_scaler = StandardScaler()
        self.model: Model | None = None
        self.history = None
        self.encoder_feature_list: List[str] = []
        self.horizon_exogenous_list: List[str] = []
        # Cache for SHAP
        self._val_X_enc: np.ndarray | None = None
        self._val_X_aux: np.ndarray | None = None
        self._val_mask: np.ndarray | None = None

    # ---------- Feature Engineering ----------
    def get_encoder_features(self, df: pd.DataFrame) -> List[str]:
        pm_block = [
            'pm2_5', 'pm10',
            'pm2_5_rolling_mean_3h', 'pm2_5_rolling_max_3h',
            'pm2_5_rolling_mean_12h', 'pm2_5_rolling_max_12h',
            'pm2_5_rolling_mean_24h', 'pm2_5_rolling_max_24h',
            'pm10_rolling_mean_3h', 'pm10_rolling_mean_12h', 'pm10_rolling_mean_24h',
            'pm2_5_change_rate_1h', 'pm2_5_change_rate_6h', 'pm2_5_change_rate_24h',
            'pm10_change_rate_1h', 'pm10_change_rate_6h', 'pm10_change_rate_24h',
        ]
        interactions = [
            'pm2_5_temp_interaction', 'pm2_5_humidity_interaction', 'pm2_5_pressure_interaction',
            'pm10_temperature_interaction', 'pm10_pressure_interaction',
        ]
        weather = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction_sin', 'wind_direction_cos']
        time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
        pollutants = ['carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']

        enc = [c for c in pm_block + interactions + weather + time_cols + pollutants if c in df.columns]
        return enc

    def scale_encoder_features(self, X_raw: np.ndarray, feature_names: List[str], train_mask: np.ndarray) -> np.ndarray:
        # Split by groups based on name substrings (simple, robust)
        nfeat = len(feature_names)
        X_train = X_raw[train_mask]
        X_val = X_raw[~train_mask]

        def idx_where(pred):
            return [i for i, c in enumerate(feature_names) if pred(c)]

        pm_idx = idx_where(lambda c: c.startswith('pm') or 'change_rate' in c)
        w_idx = idx_where(lambda c: c in ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction_sin', 'wind_direction_cos'])
        pol_idx = idx_where(lambda c: c in ['carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3'])
        inter_idx = idx_where(lambda c: 'interaction' in c)
        # time left unscaled

        X_tr_s, X_va_s = X_train.copy(), X_val.copy()

        def fit_apply(scaler: StandardScaler, idxs: List[int]):
            nonlocal X_tr_s, X_va_s
            if not idxs:
                return
            tr = X_train[:, :, idxs].reshape(-1, len(idxs))
            va = X_val[:, :, idxs].reshape(-1, len(idxs))
            tr_s = scaler.fit_transform(tr)
            va_s = scaler.transform(va)
            X_tr_s[:, :, idxs] = tr_s.reshape(X_train.shape[0], X_train.shape[1], len(idxs))
            X_va_s[:, :, idxs] = va_s.reshape(X_val.shape[0], X_val.shape[1], len(idxs))

        fit_apply(self.pm_scaler, pm_idx)
        fit_apply(self.weather_scaler, w_idx)
        fit_apply(self.pollutant_scaler, pol_idx)
        fit_apply(self.interaction_scaler, inter_idx)

        return np.concatenate([X_tr_s, X_va_s], axis=0)

    def build_target_horizon_exogenous(self, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        df = df.copy()
        # Ensure scaled base columns exist for weather and pollutants
        def ensure_scaled(cols: List[str], scaler: StandardScaler):
            avail = [c for c in cols if c in df.columns]
            if not avail:
                return
            s = scaler.fit_transform(df[avail].values)
            for i, c in enumerate(avail):
                df[f'{c}_scaled'] = s[:, i]

        ensure_scaled(['temperature', 'humidity', 'pressure', 'wind_speed'], self.weather_scaler)
        ensure_scaled(['carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3'], self.pollutant_scaler)

        for h in horizons:
            # Weather
            for feat in ['temperature', 'humidity', 'pressure', 'wind_speed']:
                if f'{feat}_scaled' in df.columns:
                    df[f'{feat}_scaled_target_{h}h'] = df[f'{feat}_scaled'].shift(-h)
            # Wind dir raw
            for feat in ['wind_direction_sin', 'wind_direction_cos']:
                if feat in df.columns:
                    df[f'{feat}_target_{h}h'] = df[feat].shift(-h)
            # Pollutants
            for feat in ['carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']:
                if f'{feat}_scaled' in df.columns:
                    df[f'{feat}_scaled_target_{h}h'] = df[f'{feat}_scaled'].shift(-h)
            # Time-at-horizon (raw cyclical, not scaled)
            for feat in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                         'day_of_week_sin', 'day_of_week_cos']:
                if feat in df.columns:
                    df[f'{feat}_target_{h}h'] = df[feat].shift(-h)
        return df

    @staticmethod
    def build_positional_encoding(steps: int) -> np.ndarray:
        h = np.arange(1, steps + 1, dtype=np.float32)
        ang = 2.0 * np.pi * (h / float(steps))
        return np.stack([np.sin(ang), np.cos(ang)], axis=1)  # [steps,2]

    def create_windows(self, df: pd.DataFrame, encoder_features: List[str]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        seq_len = self.config['sequence_length']
        steps = self.config['num_steps']
        enc_mat = df[encoder_features].values.astype(np.float32)
        pm_mat = df[['pm2_5', 'pm10']].values.astype(np.float32)
        first = seq_len
        last = len(df) - steps
        base_idx = list(range(first, last)) if last > first else []
        if not base_idx:
            return np.zeros((0, seq_len, len(encoder_features)), dtype=np.float32), np.zeros((0, steps, 2), dtype=np.float32), []
        X_enc = np.stack([enc_mat[i - seq_len:i] for i in base_idx], axis=0)
        Y = np.stack([pm_mat[i + 1:i + steps + 1] for i in base_idx], axis=0)
        return X_enc, Y, base_idx

    def collect_horizon_exogenous_tensor(self, df: pd.DataFrame, base_indices: List[int], horizons: List[int]) -> Tuple[np.ndarray, List[str]]:
        # Features to include per horizon
        feats = [
            'temperature_scaled_target_{h}h', 'humidity_scaled_target_{h}h', 'pressure_scaled_target_{h}h', 'wind_speed_scaled_target_{h}h',
            'wind_direction_sin_target_{h}h', 'wind_direction_cos_target_{h}h',
            'carbon_monoxide_scaled_target_{h}h', 'ozone_scaled_target_{h}h', 'sulphur_dioxide_scaled_target_{h}h', 'nh3_scaled_target_{h}h',
            # Time-at-horizon (raw cyclical)
            'hour_sin_target_{h}h', 'hour_cos_target_{h}h', 'day_sin_target_{h}h', 'day_cos_target_{h}h',
            'month_sin_target_{h}h', 'month_cos_target_{h}h', 'day_of_week_sin_target_{h}h', 'day_of_week_cos_target_{h}h',
        ]
        # Determine available feature templates
        # Use h=1 to probe existence
        available_templates: List[str] = []
        for t in feats:
            sample_col = t.format(h=1)
            if sample_col in df.columns:
                available_templates.append(t)
        if not available_templates:
            return np.zeros((len(base_indices), len(horizons), 0), dtype=np.float32), []

        B = len(base_indices)
        T = len(horizons)
        F = len(available_templates)
        out = np.zeros((B, T, F), dtype=np.float32)
        for bi, i in enumerate(base_indices):
            for ti, h in enumerate(horizons):
                vals = []
                for t in available_templates:
                    col = t.format(h=h)
                    v = df.at[i, col] if col in df.columns else np.nan
                    vals.append(0.0 if pd.isna(v) else float(v))
                out[bi, ti, :] = np.array(vals, dtype=np.float32)
        # Save the realized feature names
        realized = [t.format(h='h') for t in available_templates]  # template marker for logging
        return out, realized

    # ---------- Model ----------
    def build_model(self, encoder_shape: Tuple[int, int], dec_aux_dim: int) -> None:
        steps = self.config['num_steps']
        enc_in = Input(shape=encoder_shape, name='enc_in')
        x = enc_in
        for i, units in enumerate(self.config['encoder_units']):
            return_sequences = i < len(self.config['encoder_units']) - 1
            x = LSTM(units, return_sequences=return_sequences, dropout=self.config['dropout_rate'], name=f'enc_lstm_{i+1}')(x)
            x = BatchNormalization(name=f'enc_bn_{i+1}')(x)
        context = x  # [B, enc_units_last]

        # Decoder inputs
        aux_in = Input(shape=(steps, dec_aux_dim), name='dec_aux_in')
        rep = RepeatVector(steps, name='repeat_ctx')(context)  # [B,steps,enc_dim]
        dec_stream = Concatenate(axis=-1, name='dec_concat')([rep, aux_in])
        dec = LSTM(self.config['decoder_units'], return_sequences=True, name='dec_lstm')(dec_stream)
        dec = BatchNormalization(name='dec_bn')(dec)
        dec = Dropout(self.config['dropout_rate'])(dec)
        out = TimeDistributed(Dense(2, activation='linear', kernel_regularizer=l2(1e-4)), name='dec_head')(dec)

        model = Model([enc_in, aux_in], out, name='direct_lstm_mh')

        # Horizon-weighted loss (configurable)
        steps_arr = self.config['num_steps']
        weights = np.ones(steps_arr, dtype=np.float32)
        w13 = float(self.config.get('weights_1_3', 5.0))
        w46 = float(self.config.get('weights_4_6', 3.0))
        w724 = float(self.config.get('weights_7_24', 3.0))
        w2572 = float(self.config.get('weights_25_72', 1.0))
        e1 = min(3, steps_arr); e2 = min(6, steps_arr); e3 = min(24, steps_arr)
        if e1 > 0:
            weights[0:e1] = w13
        if e2 > e1:
            weights[e1:e2] = w46
        if e3 > e2:
            weights[e2:e3] = w724
        if steps_arr > e3:
            weights[e3:] = w2572

        def weighted_loss(y_true, y_pred):
            err = y_true - y_pred
            mse = tf.reduce_mean(tf.square(err), axis=2)   # [B,steps]
            mae = tf.reduce_mean(tf.abs(err), axis=2)
            w = tf.constant(weights, dtype=tf.float32)
            return tf.reduce_mean((mse + 0.2 * mae) * w)

        model.compile(optimizer=Adam(learning_rate=self.config['learning_rate'], clipnorm=1.0), loss=weighted_loss, metrics=['mae'])
        self.model = model

    # ---------- Training / Evaluation ----------
    def train(self, df: pd.DataFrame) -> Dict:
        steps = self.config['num_steps']
        horizons = list(range(1, steps + 1))

        # Build exogenous at t+h (scaled)
        df_ex = self.build_target_horizon_exogenous(df, horizons)

        # Encoder features
        enc_features = self.get_encoder_features(df_ex)
        if not enc_features:
            raise ValueError('No encoder features available')
        self.encoder_feature_list = enc_features

        # Windows
        X_enc_raw, Y, base_idx = self.create_windows(df_ex, enc_features)
        if len(base_idx) == 0:
            raise ValueError('Insufficient rows for sequences')

        # Train/val split (chronological)
        n = len(base_idx)
        n_train = int((1.0 - self.config['test_size']) * n)
        mask_tr = np.zeros(n, dtype=bool); mask_tr[:n_train] = True
        mask_va = ~mask_tr

        # Scale encoder groups on train only
        X_enc_scaled = self.scale_encoder_features(X_enc_raw, enc_features, mask_tr)

        # Scale targets (PM) on train only
        Y_tr_flat = Y[mask_tr].reshape(-1, 2)
        self.target_pm_scaler.fit(Y_tr_flat)
        Y_scaled = self.target_pm_scaler.transform(Y.reshape(-1, 2)).reshape(Y.shape)

        # Horizon exogenous tensor [B, steps, Fh]
        X_hexo, realized_templates = self.collect_horizon_exogenous_tensor(df_ex, base_idx, horizons)
        self.horizon_exogenous_list = realized_templates

        # Positional enc [steps,2] → broadcast to batch
        pos = self.build_positional_encoding(steps)  # [steps,2]
        pos_b = np.repeat(pos[None, :, :], repeats=len(base_idx), axis=0)

        # Decoder aux = concat(horizon exogenous, positional, repeated pm(t))
        pm0 = df_ex.loc[base_idx, ['pm2_5', 'pm10']].values.astype(np.float32)
        pm0_scaled = self.target_pm_scaler.transform(pm0)
        pm0_rep = np.repeat(pm0_scaled[:, None, :], repeats=steps, axis=1)
        X_aux = np.concatenate([X_hexo, pos_b.astype(np.float32), pm0_rep], axis=2)

        # Cache validation tensors for SHAP
        self._val_X_enc = X_enc_scaled[mask_va]
        self._val_X_aux = X_aux[mask_va]
        self._val_mask = mask_va

        # Build model
        self.build_model(encoder_shape=(X_enc_scaled.shape[1], X_enc_scaled.shape[2]), dec_aux_dim=X_aux.shape[2])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.config['early_stopping_patience'], restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=self.config['reduce_lr_factor'], patience=self.config['reduce_lr_patience'], min_lr=self.config['min_lr'], verbose=1),
            ModelCheckpoint(filepath=os.path.join(PATHS['temp_dir'], 'direct_lstm_mh_best.keras'), monitor='val_loss', save_best_only=True, verbose=1),
        ]

        # Train either on absolute scaled targets or deltas based on config
        use_delta = bool(self.config.get('use_delta_targets', True))
        if use_delta:
            Y_train_scaled = (Y_scaled - pm0_rep)
            val_targets_scaled = Y_scaled  # for reporting
        else:
            Y_train_scaled = Y_scaled
            val_targets_scaled = Y_scaled

        history = self.model.fit(
            [X_enc_scaled[mask_tr], X_aux[mask_tr]], Y_train_scaled[mask_tr],
            validation_data=([X_enc_scaled[mask_va], X_aux[mask_va]], val_targets_scaled[mask_va]),
            epochs=self.config['epochs'], batch_size=self.config['batch_size'], verbose=1, callbacks=callbacks,
        )
        self.history = history

        # Evaluation per horizon on val
        Y_val_true = Y[mask_va]
        Y_val_pred_scaled_model = self.model.predict([X_enc_scaled[mask_va], X_aux[mask_va]], verbose=0)
        if use_delta:
            # Reconstruct absolute scaled preds: scale(pm(t+h)) ≈ scale(pm(t)) + Δ̂_scaled
            pm0_rep_val = pm0_rep[mask_va]
            Y_val_pred_scaled = pm0_rep_val + Y_val_pred_scaled_model
        else:
            Y_val_pred_scaled = Y_val_pred_scaled_model
        Y_val_pred = self.target_pm_scaler.inverse_transform(Y_val_pred_scaled.reshape(-1, 2)).reshape(Y_val_pred_scaled.shape)
        results: Dict[str, Dict[str, float]] = {}
        for h in horizons:
            t25 = Y_val_true[:, h - 1, 0]
            p25 = Y_val_pred[:, h - 1, 0]
            t10 = Y_val_true[:, h - 1, 1]
            p10 = Y_val_pred[:, h - 1, 1]
            results[f'{h}h'] = {
                'pm2_5_rmse': float(np.sqrt(mean_squared_error(t25, p25))),
                'pm10_rmse': float(np.sqrt(mean_squared_error(t10, p10))),
                'pm2_5_mae': float(mean_absolute_error(t25, p25)),
                'pm10_mae': float(mean_absolute_error(t10, p10)),
                'pm2_5_r2': float(r2_score(t25, p25)),
                'pm10_r2': float(r2_score(t10, p10)),
            }

        # Print summary
        print("\n📊 Direct LSTM Multi-Horizon Validation (key steps):")
        for h in [1, 6, 12, 24, 48, 72]:
            r = results.get(f'{h}h')
            if r:
                print(f"  {h}h → PM2.5 RMSE={r['pm2_5_rmse']:.2f}, R²={r['pm2_5_r2']:.3f} | PM10 RMSE={r['pm10_rmse']:.2f}, R²={r['pm10_r2']:.3f}")

        return results

    # ---------- Persistence ----------
    def save_artifacts(self, out_dir_name: str = 'direct_lstm_multi_horizon') -> str:
        if self.model is None:
            return ''
        out_dir = os.path.join(PATHS['temp_dir'], out_dir_name)
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, f'{out_dir_name}.keras')
        self.model.save(model_path)
        joblib.dump({
            'pm': self.pm_scaler,
            'weather': self.weather_scaler,
            'pollutant': self.pollutant_scaler,
            'interaction': self.interaction_scaler,
            'target_pm': self.target_pm_scaler,
        }, os.path.join(out_dir, f'{out_dir_name}_scalers.pkl'))
        with open(os.path.join(out_dir, f'{out_dir_name}_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        with open(os.path.join(out_dir, f'{out_dir_name}_features.json'), 'w') as f:
            json.dump({'encoder_features': self.encoder_feature_list, 'horizon_exogenous_templates': self.horizon_exogenous_list}, f, indent=2)
        print(f"✅ Saved model and artifacts to {out_dir}")
        # Optional: auto-register/upload to Hopsworks when env flag is set
        if os.getenv('REGISTER_TO_HOPSWORKS', '0') == '1':
            try:
                from model_registry_utils import register_or_upload_model
                _ = register_or_upload_model(
                    artifacts_dir=out_dir,
                    model_name=out_dir_name,
                    metrics={},
                    description='Direct LSTM multi-horizon model artifacts',
                )
            except Exception as e:
                print(f"⚠️ Model registration/upload skipped/failed: {e}")
        return out_dir


def main():
    print("🚀 Training Direct Multi-Horizon LSTM (no AR)")
    # Prefer Hopsworks if available; fallback to CSV
    df = None
    if HopsworksUploader is not None and os.getenv('HOPSWORKS_API_KEY') and HOPSWORKS_CONFIG.get('project_name'):
        try:
            print("📊 Loading dataset from Hopsworks...")
            uploader = HopsworksUploader(api_key=os.getenv('HOPSWORKS_API_KEY'), project_name=HOPSWORKS_CONFIG['project_name'])
            if uploader.connect():
                fs = uploader.project.get_feature_store()
                fg = fs.get_or_create_feature_group(name=HOPSWORKS_CONFIG.get('feature_group_name', ''), version=1)
                df = fg.read()
                print(f"✅ Loaded {len(df)} rows from Hopsworks")
        except Exception as e:
            print(f"⚠️ Hopsworks load failed: {e}")
            df = None
    if df is None:
        csv_files = ['new_feature_group_data_20250805_050932.csv']
        for pth in csv_files:
            if os.path.exists(pth):
                df = pd.read_csv(pth)
                print(f"✅ Loaded {len(df)} rows from {pth}")
                break
        if df is None:
            raise FileNotFoundError('No CSV files found and Hopsworks unavailable')

    df = preprocess_aqi_df(df)

    trainer = DirectLSTMMultiHorizon(CONFIG)
    results = trainer.train(df)
    trainer.save_artifacts()
    return results


if __name__ == '__main__':
    for p in PATHS.values():
        os.makedirs(p, exist_ok=True)
    # Optional walk-forward CV runner (banded), triggered by WALK_CV=1
    if os.environ.get('WALK_CV', '0') == '1':
        # Load data (prefer Hopsworks)
        df = None
        if HopsworksUploader is not None and os.getenv('HOPSWORKS_API_KEY') and HOPSWORKS_CONFIG.get('project_name'):
            try:
                print("📊 Loading dataset from Hopsworks for CV...")
                uploader = HopsworksUploader(api_key=os.getenv('HOPSWORKS_API_KEY'), project_name=HOPSWORKS_CONFIG['project_name'])
                if uploader.connect():
                    fs = uploader.project.get_feature_store()
                    fg = fs.get_or_create_feature_group(name=HOPSWORKS_CONFIG.get('feature_group_name', ''), version=1)
                    df = fg.read()
                    print(f"✅ Loaded {len(df)} rows from Hopsworks")
            except Exception as e:
                print(f"⚠️ Hopsworks load failed: {e}")
                df = None
        if df is None:
            csv_files = ['new_feature_group_data_20250805_050932.csv']
            for pth in csv_files:
                if os.path.exists(pth):
                    df = pd.read_csv(pth)
                    print(f"✅ Loaded {len(df)} rows from {pth}")
                    break
            if df is None:
                raise FileNotFoundError('No CSV files found for CV and Hopsworks unavailable')
        df = preprocess_aqi_df(df)

        # Define folds as growing time windows (even fractions of the dataset)
        n_folds = int(os.environ.get('WALK_CV_FOLDS', '4'))
        N = len(df)
        fold_fracs = np.linspace(0.55, 0.95, n_folds)  # end fractions; ensure enough data
        merged_list = []
        for fi, frac in enumerate(fold_fracs, 1):
            end_idx = int(max(1, min(N - 1, round(frac * N))))
            df_fold = df.iloc[:end_idx].copy()
            print(f"\n🔁 Fold {fi}/{n_folds} — using first {end_idx}/{N} rows (~{frac:.2f})")

            # Short band config
            short_cfg = deepcopy(CONFIG)
            short_cfg['num_steps'] = 12
            short_cfg['use_delta_targets'] = True
            # Keep current weights
            short_trainer = DirectLSTMMultiHorizon(short_cfg)
            res_short = short_trainer.train(df_fold)

            # Mid/Long band config
            mid_cfg = deepcopy(CONFIG)
            mid_cfg['sequence_length'] = 96
            mid_cfg['num_steps'] = 72
            mid_cfg['use_delta_targets'] = False
            mid_cfg['weights_1_3'] = 1.0
            mid_cfg['weights_4_6'] = 1.5
            mid_cfg['weights_7_24'] = 3.0
            mid_cfg['weights_25_72'] = 1.5
            mid_trainer = DirectLSTMMultiHorizon(mid_cfg)
            res_mid = mid_trainer.train(df_fold)

            # Merge
            merged = {}
            for h in range(1, 73):
                key = f'{h}h'
                if h <= 12 and key in res_short:
                    merged[key] = res_short[key]
                elif key in res_mid:
                    merged[key] = res_mid[key]
            merged_list.append(merged)

            # Print fold key metrics
            print("\n📋 Fold metrics (key steps):")
            for h in [1, 6, 12, 24, 48, 72]:
                key = f'{h}h'
                if key in merged:
                    r = merged[key]
                    print(f"  {key} → PM2.5 RMSE={r['pm2_5_rmse']:.2f}, R²={r['pm2_5_r2']:.3f} | PM10 RMSE={r['pm10_rmse']:.2f}, R²={r['pm10_r2']:.3f}")

        # Aggregate across folds
        agg = {}
        for h in range(1, 73):
            key = f'{h}h'
            vals = [m[key] for m in merged_list if key in m]
            if not vals:
                continue
            agg[key] = {
                'pm2_5_rmse': float(np.mean([v['pm2_5_rmse'] for v in vals])),
                'pm10_rmse': float(np.mean([v['pm10_rmse'] for v in vals])),
                'pm2_5_mae': float(np.mean([v['pm2_5_mae'] for v in vals])),
                'pm10_mae': float(np.mean([v['pm10_mae'] for v in vals])),
                'pm2_5_r2': float(np.mean([v['pm2_5_r2'] for v in vals])),
                'pm10_r2': float(np.mean([v['pm10_r2'] for v in vals])),
            }

        print("\n📊 Walk-Forward CV (average across folds) – key steps:")
        for h in [1, 6, 12, 24, 48, 72]:
            key = f'{h}h'
            if key in agg:
                r = agg[key]
                print(f"  {key} → PM2.5 RMSE={r['pm2_5_rmse']:.2f}, R²={r['pm2_5_r2']:.3f} | PM10 RMSE={r['pm10_rmse']:.2f}, R²={r['pm10_r2']:.3f}")
        # Exit after CV
        raise SystemExit(0)
    # Default to banded; allow single-run with SINGLE_RUN=1
    if os.environ.get('SINGLE_RUN', '0') != '1':
        # Load data once (prefer Hopsworks)
        df = None
        if HopsworksUploader is not None and os.getenv('HOPSWORKS_API_KEY') and HOPSWORKS_CONFIG.get('project_name'):
            try:
                print("📊 Loading dataset from Hopsworks...")
                uploader = HopsworksUploader(api_key=os.getenv('HOPSWORKS_API_KEY'), project_name=HOPSWORKS_CONFIG['project_name'])
                if uploader.connect():
                    fs = uploader.project.get_feature_store()
                    fg = fs.get_or_create_feature_group(name=HOPSWORKS_CONFIG.get('feature_group_name', ''), version=1)
                    df = fg.read()
                    print(f"✅ Loaded {len(df)} rows from Hopsworks")
            except Exception as e:
                print(f"⚠️ Hopsworks load failed: {e}")
                df = None
        if df is None:
            csv_files = ['new_feature_group_data_20250805_050932.csv']
            for pth in csv_files:
                if os.path.exists(pth):
                    df = pd.read_csv(pth)
                    print(f"✅ Loaded {len(df)} rows from {pth}")
                    break
            if df is None:
                raise FileNotFoundError('No CSV files found')
        df = preprocess_aqi_df(df)

        # Short (1-12h): delta targets, strong short weights
        short_cfg = deepcopy(CONFIG)
        short_cfg['num_steps'] = 12
        short_cfg['use_delta_targets'] = True
        short_cfg['weights_1_3'] = 8.0
        short_cfg['weights_4_6'] = 4.0
        short_cfg['weights_7_24'] = 3.0
        short_cfg['weights_25_72'] = 1.0

        short_trainer = DirectLSTMMultiHorizon(short_cfg)
        print("\n🏃 Training SHORT (1–12h) direct LSTM...")
        res_short = short_trainer.train(df)
        short_trainer.save_artifacts('direct_lstm_short')

        # Mid/Long (12-72h): absolute targets, milder weights, optional longer context
        mid_cfg = deepcopy(CONFIG)
        mid_cfg['sequence_length'] = 96
        mid_cfg['num_steps'] = 72
        mid_cfg['use_delta_targets'] = False
        mid_cfg['weights_1_3'] = 1.0
        mid_cfg['weights_4_6'] = 1.5
        mid_cfg['weights_7_24'] = 3.0
        mid_cfg['weights_25_72'] = 1.5

        mid_trainer = DirectLSTMMultiHorizon(mid_cfg)
        print("\n🏃 Training MID/LONG (12–72h) direct LSTM...")
        res_mid = mid_trainer.train(df)
        mid_trainer.save_artifacts('direct_lstm_midlong')

        # Merge results for reporting
        merged = {}
        for h in range(1, 73):
            key = f'{h}h'
            if h <= 12 and key in res_short:
                merged[key] = res_short[key]
            elif key in res_mid:
                merged[key] = res_mid[key]

        print("\n🎛️ Banded Direct LSTM: Merged Validation (key steps)")
        for h in [1, 6, 12, 24, 48, 72]:
            key = f'{h}h'
            if key in merged:
                r = merged[key]
                print(f"  {key} → PM2.5 RMSE={r['pm2_5_rmse']:.2f}, R²={r['pm2_5_r2']:.3f} | PM10 RMSE={r['pm10_rmse']:.2f}, R²={r['pm10_r2']:.3f}")

        # Auto-run SHAP for both bands
        try:
            short_trainer.run_shap([1, 6, 12], tag='short')
            mid_trainer.run_shap([12, 24, 48, 72], tag='midlong')
            print("✅ SHAP plots saved to logs/shap_lstm/")
        except Exception as e:
            print(f"⚠️ SHAP step skipped: {e}")
    else:
        _ = main()


