"""
Seq2Seq LSTM v7 (simplified): Full-sequence Teacher Forcing (training) + Autoregressive Rollout (val/test/inference)

Changes per request:
- Remove all horizon decoder features (no weather/pollutants fed to decoder)
- Decoder only consumes previous PM values (2-d: [pm2_5, pm10])
- Training uses full teacher forcing (no scheduled sampling probability)
- Inference/validation uses pure autoregressive rollout (feed last prediction)
- Encoder remains unchanged
"""

import os
import warnings
from typing import Dict, List, Tuple

import joblib
import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from hopsworks_integration import HopsworksUploader

warnings.filterwarnings("ignore")

# Paths and config
try:
    from config import PATHS, HOPSWORKS_CONFIG
except ImportError:
    PATHS = {'logs_dir': 'logs', 'temp_dir': 'temp', 'data_dir': 'data'}
    HOPSWORKS_CONFIG = {'project_name': '', 'feature_group_name': ''}

for p in PATHS.values():
    os.makedirs(p, exist_ok=True)


def preprocess_aqi_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standard preprocessing: sort, dedupe, drop NaNs, remove zero/neg PM, IQR outlier filter."""
    df = df.copy()
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
        df = df.sort_values('time')
    df = df.drop_duplicates().reset_index(drop=True)
    # Drop NaNs
    df = df.dropna().reset_index(drop=True)
    # Remove non-positive PM rows
    for col in ['pm2_5', 'pm10']:
        if col in df.columns:
            df = df[df[col] > 0].reset_index(drop=True)
    # IQR outlier filter for key cols
    for col in ['pm2_5', 'pm10', 'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lb = Q1 - 1.5 * IQR
            ub = Q3 + 1.5 * IQR
            df = df[(df[col] >= lb) & (df[col] <= ub)].reset_index(drop=True)
    print(f"✅ Preprocessing applied. Rows after cleaning: {len(df)}")
    return df


SEQ2SEQ_V7_CONFIG: Dict = {
    'sequence_length': 72,
    'num_steps': 72,
    'encoder_lstm_units': [128, 64],
    #'encoder_lstm_units': [256, 128],
    'decoder_lstm_units': 64,
    #'decoder_lstm_units': 128,
    'dropout_rate': 0.2,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 80,
    'random_state': 42,
    # Training noise on teacher-forced decoder PM inputs (std in scaled units)
    'decoder_input_noise_std': 0.05,
    # Horizon-weighted loss focusing mid-range
    'midrange_start': 6,   # inclusive (1-indexed)
    'midrange_end': 24,    # inclusive (1-indexed)
    'midrange_weight': 2.0,
    # Early stopping and LR scheduling
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.0,
    'reduce_lr_patience': 6,
    'reduce_lr_factor': 0.7,
    'min_lr': 1e-6,
}


class Seq2SeqFullSeqTFV7:
    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or SEQ2SEQ_V7_CONFIG
        self.model: Model | None = None  # training model
        self.encoder_infer: Model | None = None
        self.decoder_step_infer: Model | None = None

        # Scalers
        self.encoder_pm_scaler = StandardScaler()
        self.weather_scaler = StandardScaler()
        self.interaction_scaler = StandardScaler()
        self.pm_scaler = StandardScaler()  # for decoder PM inputs/targets and rollout
        self.extra_pollutant_scaler = StandardScaler()  # encoder extra pollutants

        self.history = None

        # Cached indices
        self.pm_feature_indices: List[int] | None = None
        self.interaction_feature_indices: List[int] | None = None
        self.weather_feature_indices: List[int] | None = None
        self.time_feature_indices: List[int] | None = None

    # -----------------------------
    # Features
    # -----------------------------
    def get_encoder_features(self, df: pd.DataFrame) -> List[str]:
        # PM block (will be scaled together): base + rolling (unscaled names from dataset)
        pm_block_all = [
            'pm2_5', 'pm10',
            'pm2_5_rolling_mean_3h', 'pm2_5_rolling_max_3h',
            'pm2_5_rolling_mean_12h', 'pm2_5_rolling_max_12h',
            'pm2_5_rolling_mean_24h', 'pm2_5_rolling_max_24h',
            'pm10_rolling_mean_3h', 'pm10_rolling_mean_12h', 'pm10_rolling_mean_24h',
        ]
        pm_block = [f for f in pm_block_all if f in df.columns]

        weather_pm_interactions_all = [
            'pm2_5_temp_interaction', 'pm2_5_humidity_interaction', 'pm2_5_pressure_interaction',
            'pm10_temperature_interaction', 'pm10_pressure_interaction'
        ]
        weather_pm_interactions = [f for f in weather_pm_interactions_all if f in df.columns]

        weather_features_all = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction_sin', 'wind_direction_cos']
        weather_features = [f for f in weather_features_all if f in df.columns]

        time_features_all = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos'
        ]
        time_features = [f for f in time_features_all if f in df.columns]

        extra_pollutants_all = ['carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']
        extra_pollutants = [f for f in extra_pollutants_all if f in df.columns]

        # Preserve grouping order: PM block → interactions → weather → time → extra pollutants
        all_features = pm_block + weather_pm_interactions + weather_features + time_features + extra_pollutants

        # Store block sizes for correct scaling group indices later
        self.pm_block_size = len(pm_block)
        self.interaction_block_size = len(weather_pm_interactions)
        self.weather_block_size = len(weather_features)
        self.time_block_size = len(time_features)
        self.extra_pollutants_block_size = len(extra_pollutants)

        return all_features

    # -----------------------------
    # Sequences
    # -----------------------------
    def create_base(self, df: pd.DataFrame, encoder_features: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        seq_len = self.config['sequence_length']
        steps = self.config['num_steps']

        df_enc = df[encoder_features].values.astype(np.float32)
        pm_np = df[['pm2_5', 'pm10']].values.astype(np.float32)

        first = seq_len
        last = len(df) - steps  # exclusive
        base_indices = list(range(first, last)) if last > first else []

        X_list = [df_enc[i - seq_len:i] for i in base_indices]
        X_encoder = np.stack(X_list, axis=0) if X_list else np.zeros((0, seq_len, len(encoder_features)), dtype=np.float32)
        return X_encoder, pm_np[:, 0], pm_np[:, 1], base_indices

    # No decoder horizon features in the simplified version

    def build_feature_indices(self, num_features: int) -> None:
        total = num_features
        # Use stored block sizes; fall back to safe defaults if not set
        pm_size = getattr(self, 'pm_block_size', 2)
        inter_size = getattr(self, 'interaction_block_size', 5)
        weather_size = getattr(self, 'weather_block_size', 6)
        time_size = getattr(self, 'time_block_size', 8)
        extra_size = getattr(self, 'extra_pollutants_block_size', 4)

        pm_end = min(pm_size, total)
        inter_end = min(pm_end + inter_size, total)
        weather_end = min(inter_end + weather_size, total)
        time_end = min(weather_end + time_size, total)
        extra_end = min(time_end + extra_size, total)

        self.pm_feature_indices = list(range(0, pm_end))
        self.interaction_feature_indices = list(range(pm_end, inter_end))
        self.weather_feature_indices = list(range(inter_end, weather_end))
        self.time_feature_indices = list(range(weather_end, time_end))
        self.extra_pollutants_feature_indices = list(range(time_end, extra_end))

    def scale_encoder(self, X_encoder_raw: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
        seq_len = self.config['sequence_length']
        nfeat = X_encoder_raw.shape[2]
        self.build_feature_indices(nfeat)

        X_tr = X_encoder_raw[train_mask]
        X_vt = X_encoder_raw[~train_mask]

        X_tr_s = X_tr.copy()
        X_vt_s = X_vt.copy()

        # PM block (includes rolling PM features) — scale together
        if self.pm_feature_indices and len(self.pm_feature_indices) > 0:
            pm_tr = X_tr[:, :, self.pm_feature_indices].reshape(-1, len(self.pm_feature_indices))
            pm_vt = X_vt[:, :, self.pm_feature_indices].reshape(-1, len(self.pm_feature_indices))
            pm_tr_s = self.encoder_pm_scaler.fit_transform(pm_tr)
            pm_vt_s = self.encoder_pm_scaler.transform(pm_vt)
            X_tr_s[:, :, self.pm_feature_indices] = pm_tr_s.reshape(X_tr.shape[0], seq_len, len(self.pm_feature_indices))
            X_vt_s[:, :, self.pm_feature_indices] = pm_vt_s.reshape(X_vt.shape[0], seq_len, len(self.pm_feature_indices))

        # Weather
        if self.weather_feature_indices and len(self.weather_feature_indices) > 0 and max(self.weather_feature_indices) < nfeat:
            w_tr = X_tr[:, :, self.weather_feature_indices].reshape(-1, len(self.weather_feature_indices))
            w_vt = X_vt[:, :, self.weather_feature_indices].reshape(-1, len(self.weather_feature_indices))
            w_tr_s = self.weather_scaler.fit_transform(w_tr)
            w_vt_s = self.weather_scaler.transform(w_vt)
            X_tr_s[:, :, self.weather_feature_indices] = w_tr_s.reshape(X_tr.shape[0], seq_len, len(self.weather_feature_indices))
            X_vt_s[:, :, self.weather_feature_indices] = w_vt_s.reshape(X_vt.shape[0], seq_len, len(self.weather_feature_indices))

        # Interactions
        if self.interaction_feature_indices and len(self.interaction_feature_indices) > 0 and max(self.interaction_feature_indices) < nfeat:
            it_tr = X_tr[:, :, self.interaction_feature_indices].reshape(-1, len(self.interaction_feature_indices))
            it_vt = X_vt[:, :, self.interaction_feature_indices].reshape(-1, len(self.interaction_feature_indices))
            it_tr_s = self.interaction_scaler.fit_transform(it_tr)
            it_vt_s = self.interaction_scaler.transform(it_vt)
            X_tr_s[:, :, self.interaction_feature_indices] = it_tr_s.reshape(X_tr.shape[0], seq_len, len(self.interaction_feature_indices))
            X_vt_s[:, :, self.interaction_feature_indices] = it_vt_s.reshape(X_vt.shape[0], seq_len, len(self.interaction_feature_indices))

        # Extra pollutants group
        if hasattr(self, 'extra_pollutants_feature_indices') and self.extra_pollutants_feature_indices and len(self.extra_pollutants_feature_indices) > 0 and max(self.extra_pollutants_feature_indices) < nfeat:
            ep_tr = X_tr[:, :, self.extra_pollutants_feature_indices].reshape(-1, len(self.extra_pollutants_feature_indices))
            ep_vt = X_vt[:, :, self.extra_pollutants_feature_indices].reshape(-1, len(self.extra_pollutants_feature_indices))
            ep_tr_s = self.extra_pollutant_scaler.fit_transform(ep_tr)
            ep_vt_s = self.extra_pollutant_scaler.transform(ep_vt)
            X_tr_s[:, :, self.extra_pollutants_feature_indices] = ep_tr_s.reshape(X_tr.shape[0], seq_len, len(self.extra_pollutants_feature_indices))
            X_vt_s[:, :, self.extra_pollutants_feature_indices] = ep_vt_s.reshape(X_vt.shape[0], seq_len, len(self.extra_pollutants_feature_indices))

        # Time: left unscaled
        return np.concatenate([X_tr_s, X_vt_s], axis=0)

    @staticmethod
    def build_step_positional_encoding(steps: int) -> np.ndarray:
        """
        Build 2-d sinusoidal positional encoding per step h in [1..steps]: [sin(2πh/steps), cos(2πh/steps)].
        Returns array of shape [steps, 2].
        """
        h = np.arange(1, steps + 1, dtype=np.float32)
        ang = 2.0 * np.pi * (h / float(steps))
        pe = np.stack([np.sin(ang), np.cos(ang)], axis=1)
        return pe.astype(np.float32)

    def make_decoder_sequences_subset(
        self,
        df: pd.DataFrame,
        base_indices: List[int],
        subset_mask: np.ndarray,
        pm2_5_array: np.ndarray,
        pm10_array: np.ndarray,
        fit_pm_scaler: bool = False,
        fit_weather_scaler: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build teacher-forced decoder sequences for a subset (train/val) using ONLY PM sequences.
        Returns X_dec_in: [batch, steps, 2] (previous PM), y_dec: [batch, steps, 2] (targets), both scaled by pm_scaler.
        """
        steps = self.config['num_steps']
        base_arr = np.array(base_indices)[subset_mask]

        dec_in_list: List[np.ndarray] = []
        dec_tgt_list: List[np.ndarray] = []

        for i in base_arr:
            # PM part (unscaled yet)
            pm_prev_seq = np.stack([
                np.array([pm2_5_array[i + h - 1], pm10_array[i + h - 1]], dtype=np.float32)
                for h in range(1, steps + 1)
            ], axis=0)  # [steps,2]

            # Targets PM(t+h)
            tgt_seq = np.stack([
                np.array([pm2_5_array[i + h], pm10_array[i + h]], dtype=np.float32)
                for h in range(1, steps + 1)
            ], axis=0)  # [steps,2]

            dec_in_list.append(pm_prev_seq)
            dec_tgt_list.append(tgt_seq)

        if not dec_in_list:
            return np.zeros((0, steps, 2), dtype=np.float32), np.zeros((0, steps, 2), dtype=np.float32)

        # Stack
        pm_prev_all = np.stack(dec_in_list, axis=0)   # [B,steps,2]
        y_all = np.stack(dec_tgt_list, axis=0)        # [B,steps,2]

        # Fit/transform PM scaler on combined prev and targets if requested
        if fit_pm_scaler:
            pm_fit = np.vstack([pm_prev_all.reshape(-1, 2), y_all.reshape(-1, 2)])
            self.pm_scaler.fit(pm_fit)
        pm_prev_scaled = self.pm_scaler.transform(pm_prev_all.reshape(-1, 2)).reshape(pm_prev_all.shape)
        y_scaled = self.pm_scaler.transform(y_all.reshape(-1, 2)).reshape(y_all.shape)

        # Add step positional encodings (constant across batch)
        step_pe = self.build_step_positional_encoding(steps)  # [steps,2]
        batch_size = pm_prev_scaled.shape[0]
        step_pe_b = np.repeat(step_pe[np.newaxis, :, :], batch_size, axis=0)  # [B,steps,2]

        # Optional noise on teacher-forced PM inputs (training only)
        if fit_pm_scaler and float(self.config.get('decoder_input_noise_std', 0.0)) > 0.0:
            noise_std = float(self.config.get('decoder_input_noise_std', 0.0))
            noise = np.random.normal(loc=0.0, scale=noise_std, size=pm_prev_scaled.shape).astype(np.float32)
            pm_prev_scaled = pm_prev_scaled + noise

        # Decoder input is previous PM + step positional embedding → 4 features
        X_dec_in = np.concatenate([pm_prev_scaled, step_pe_b], axis=2)
        return X_dec_in.astype(np.float32), y_scaled.astype(np.float32)

    # -----------------------------
    # Model (train + inference graphs)
    # -----------------------------
    def build_models(self, encoder_shape: Tuple[int, int], decoder_input_dim: int) -> None:
        steps = self.config['num_steps']
        enc_units = self.config['encoder_lstm_units']
        dec_units = self.config['decoder_lstm_units']

        # Encoder
        enc_in = Input(shape=encoder_shape, name='encoder_input')
        x = enc_in
        for i, units in enumerate(enc_units):
            return_sequences = i < len(enc_units) - 1
            if return_sequences:
                x = LSTM(units, return_sequences=True, dropout=self.config['dropout_rate'],
                         recurrent_dropout=self.config['dropout_rate'], name=f'enc_lstm_{i+1}')(x)
                x = BatchNormalization(name=f'enc_bn_{i+1}')(x)
            else:
                x, h, c = LSTM(units, return_sequences=False, return_state=True,
                               dropout=self.config['dropout_rate'],
                               recurrent_dropout=self.config['dropout_rate'], name=f'enc_lstm_{i+1}')(x)
                x = BatchNormalization(name=f'enc_bn_{i+1}')(x)
        # x is last output (ignored); h, c are encoder final states

        # Decoder (training) — input is previous PM (2) + step positional enc (2) → 4 features
        dec_in_seq = Input(shape=(steps, 4), name='decoder_inputs')
        dec_lstm = LSTM(dec_units, return_sequences=True, return_state=True, name='dec_lstm')
        dec_seq, _, _ = dec_lstm(dec_in_seq, initial_state=[h, c])
        td_dense = TimeDistributed(Dense(2, activation='linear'), name='td_output')
        dec_out = td_dense(dec_seq)

        # Horizon-weighted MSE
        mid_s = int(self.config.get('midrange_start', 6))
        mid_e = int(self.config.get('midrange_end', 24))
        mid_w = float(self.config.get('midrange_weight', 2.0))
        horizon_weights = np.ones(steps, dtype=np.float32)
        # convert to 0-indexed slices
        s0 = max(0, mid_s - 1)
        e0 = min(steps, mid_e)
        if e0 > s0:
            horizon_weights[s0:e0] = mid_w
        self.horizon_weights = horizon_weights

        def weighted_mse(y_true, y_pred):
            # y_* shape: [B,steps,2]
            err = tf.reduce_mean(tf.square(y_true - y_pred), axis=2)  # [B,steps]
            w = tf.constant(self.horizon_weights, dtype=tf.float32)   # [steps]
            weighted = err * w
            return tf.reduce_mean(weighted)

        # Compile training model (teacher-forced input path)
        train_model = Model([enc_in, dec_in_seq], dec_out, name='v7_train_model')
        train_model.compile(optimizer=Adam(learning_rate=self.config['learning_rate'], clipnorm=0.5),
                           loss=weighted_mse, metrics=['mae'])
        self.model = train_model

        # Inference models
        self.encoder_infer = Model(enc_in, [h, c], name='v7_encoder_infer')

    # -----------------------------
    # Rollout and metrics
    # -----------------------------
    def rollout_autoregressive(self, X_encoder_scaled_all: np.ndarray, base_indices: List[int], subset_mask: np.ndarray,
                               pm2_5_array: np.ndarray, pm10_array: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """
        Batched rollout: single encoder pass for all samples, and an in-graph TF loop for 72-step decoding.
        Decoder input uses previous PM + step positional enc (no exogenous weather/pollutants).
        Returns predictions in original scale: [batch, steps, 2].
        """
        steps = self.config['num_steps']

        # Batch select encoder sequences for this subset
        X_subset = X_encoder_scaled_all[subset_mask]
        if X_subset.shape[0] == 0:
            return np.zeros((0, steps, 2), dtype=np.float32)

        # Batch encoder states
        state_h, state_c = self.encoder_infer.predict(X_subset, verbose=0)

        # Initial PM at t (original scale) → scale as decoder input
        base_arr = np.array(base_indices)[subset_mask]
        pm0 = np.stack([np.array([pm2_5_array[i], pm10_array[i]], dtype=np.float32) for i in base_arr], axis=0)  # [batch,2]
        pm0_scaled = self.pm_scaler.transform(pm0).astype(np.float32)

        # Prepare TF tensors
        pm0_tf = tf.convert_to_tensor(pm0_scaled, dtype=tf.float32)       # [batch,2]
        h_tf = tf.convert_to_tensor(state_h, dtype=tf.float32)            # [batch,dec_units]
        c_tf = tf.convert_to_tensor(state_c, dtype=tf.float32)            # [batch,dec_units]

        # Grab decoder cell and dense head
        dec_layer = self.model.get_layer('dec_lstm')
        dec_cell = dec_layer.cell
        head_dense = self.model.get_layer('td_output').layer  # Dense(2)

        # Build step positional encoding (batch-aligned)
        step_pe_np = self.build_step_positional_encoding(steps)  # [steps,2]
        step_pe_b = np.repeat(step_pe_np[np.newaxis, :, :], pm0_scaled.shape[0], axis=0)  # [batch,steps,2]
        step_pe_tf = tf.convert_to_tensor(step_pe_b, dtype=tf.float32)

        @tf.function(reduce_retracing=True)
        def tf_rollout(pm_init: tf.Tensor, h0: tf.Tensor, c0: tf.Tensor, step_pe: tf.Tensor) -> tf.Tensor:
            ta = tf.TensorArray(tf.float32, size=steps)
            x = pm_init  # [batch,2]
            state_h = h0
            state_c = c0
            t = tf.constant(0)
            for _ in tf.range(steps):
                # concat previous PM with step positional enc for t
                pe_t = step_pe[:, t, :]                   # [batch,2]
                step_in = tf.concat([x, pe_t], axis=1)    # [batch,4]
                out, [state_h, state_c] = dec_cell(step_in, states=[state_h, state_c], training=False)
                y = head_dense(out, training=False)  # [batch,2]
                ta = ta.write(t, y)
                x = y  # next pm_prev
                t += 1
            stacked = ta.stack()                 # [steps,batch,2]
            preds = tf.transpose(stacked, [1, 0, 2])  # [batch,steps,2]
            return preds

        preds_scaled = tf_rollout(pm0_tf, h_tf, c_tf, step_pe_tf).numpy()  # [batch,steps,2]
        preds_orig = self.pm_scaler.inverse_transform(preds_scaled.reshape(-1, 2)).reshape(-1, steps, 2)
        return preds_orig.astype(np.float32)

    @staticmethod
    def per_horizon_metrics(predictions_orig: np.ndarray, base_indices_subset: np.ndarray,
                            pm2_5_array: np.ndarray, pm10_array: np.ndarray, steps: int) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict] = {}
        for s in range(1, steps + 1):
            y_true_25, y_pred_25, y_true_10, y_pred_10 = [], [], [], []
            for j, base_i in enumerate(base_indices_subset):
                tgt_idx = base_i + s
                if tgt_idx < len(pm2_5_array):
                    y_true_25.append(pm2_5_array[tgt_idx])
                    y_true_10.append(pm10_array[tgt_idx])
                    y_pred_25.append(predictions_orig[j, s - 1, 0])
                    y_pred_10.append(predictions_orig[j, s - 1, 1])
            if y_true_25:
                y_true_25 = np.array(y_true_25)
                y_true_10 = np.array(y_true_10)
                y_pred_25 = np.array(y_pred_25)
                y_pred_10 = np.array(y_pred_10)
                res = {
                    'pm2_5': {
                        'rmse': float(np.sqrt(mean_squared_error(y_true_25, y_pred_25))),
                        'mae': float(mean_absolute_error(y_true_25, y_pred_25)),
                        'r2': float(r2_score(y_true_25, y_pred_25)),
                    },
                    'pm10': {
                        'rmse': float(np.sqrt(mean_squared_error(y_true_10, y_pred_10))),
                        'mae': float(mean_absolute_error(y_true_10, y_pred_10)),
                        'r2': float(r2_score(y_true_10, y_pred_10)),
                    },
                }
                results[f'{s}h'] = res
        return results

    # -----------------------------
    # Training
    # -----------------------------
    def train(self, df: pd.DataFrame) -> Dict:
        df = preprocess_aqi_df(df)
        enc_features = self.get_encoder_features(df)
        if not enc_features:
            raise ValueError('No encoder features available in dataframe')

        X_encoder_raw, pm25, pm10, base_indices = self.create_base(df, enc_features)
        if len(base_indices) == 0:
            raise ValueError('Insufficient rows for sequences')

        # 80/20 split over base indices (train/test)
        n = len(base_indices)
        n_train = int(0.8 * n)
        train_mask = np.zeros(n, dtype=bool); train_mask[:n_train] = True
        test_mask = np.zeros(n, dtype=bool); test_mask[n_train:] = True

        # Scale encoder using train-only stats
        X_encoder_scaled_all = self.scale_encoder(X_encoder_raw, train_mask)
        X_encoder_train_base = X_encoder_scaled_all[train_mask]
        X_encoder_test_base = X_encoder_scaled_all[test_mask]

        # Teacher-forced decoder sequences for train (PM-only)
        X_dec_train, y_dec_train = self.make_decoder_sequences_subset(
            df, base_indices, train_mask, pm25, pm10, fit_pm_scaler=True, fit_weather_scaler=False
        )

        # Build models
        encoder_shape = (X_encoder_train_base.shape[1], X_encoder_train_base.shape[2])
        self.build_models(encoder_shape, 2)

        # Validation (teacher-forced) uses the 20% test split for monitoring
        X_dec_val_tf, y_dec_val_tf = self.make_decoder_sequences_subset(
            df, base_indices, test_mask, pm25, pm10, fit_pm_scaler=False, fit_weather_scaler=False
        )

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=int(self.config.get('early_stopping_patience', 10)),
                min_delta=float(self.config.get('early_stopping_min_delta', 0.0)),
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=float(self.config.get('reduce_lr_factor', 0.7)),
                patience=int(self.config.get('reduce_lr_patience', 6)),
                min_lr=float(self.config.get('min_lr', 1e-6)),
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(PATHS['temp_dir'], 'best_v7.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
        ]

        # Full teacher forcing training on standard model
        history = self.model.fit(
            [X_encoder_train_base, X_dec_train], y_dec_train,
            validation_data=([X_encoder_test_base, X_dec_val_tf], y_dec_val_tf) if len(X_dec_val_tf) > 0 else None,
            epochs=int(self.config['epochs']),
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1,
        )
        self.history = history

        # Autoregressive rollout (test/validation 20% split)
        base_arr = np.array(base_indices)
        test_preds_orig = self.rollout_autoregressive(X_encoder_scaled_all, base_indices, test_mask, pm25, pm10, df)
        test_results = self.per_horizon_metrics(test_preds_orig, base_arr[test_mask], pm25, pm10, self.config['num_steps'])

        def print_results(title: str, res: Dict[str, Dict]):
            print(f"\n{title}")
            print(f"{'Horizon':<6} | {'PM2.5 R2':>8} {'PM2.5 MAE':>10} {'PM2.5 RMSE':>11} | {'PM10 R2':>8} {'PM10 MAE':>10} {'PM10 RMSE':>11}")
            print("-" * 73)
            for h in range(1, self.config['num_steps'] + 1):
                key = f'{h}h'
                if key in res:
                    r = res[key]
                    print(f"{key:<6} | {r['pm2_5']['r2']:8.3f} {r['pm2_5']['mae']:10.3f} {r['pm2_5']['rmse']:11.3f} | {r['pm10']['r2']:8.3f} {r['pm10']['mae']:10.3f} {r['pm10']['rmse']:11.3f}")

        print_results("Test/Validation (Autoregressive Rollout)", test_results)

        return {
            'test_results': test_results,
            'history': self.history.history if self.history else None,
        }

    # -----------------------------
    # Persistence
    # -----------------------------
    def save_artifacts(self, model_dir_name: str = 'seq2seq_fullseq_tf_v7') -> str:
        if self.model is None:
            print("❌ No model to save")
            return ''
        model_dir = os.path.join(PATHS['temp_dir'], model_dir_name)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f'{model_dir_name}.keras')
        self.model.save(model_path)
        print(f"✅ Model saved to {model_path}")

        scalers = {
            'encoder_pm_scaler': self.encoder_pm_scaler,
            'weather_scaler': self.weather_scaler,
            'interaction_scaler': self.interaction_scaler,
            'pm_scaler': self.pm_scaler,
            'extra_pollutant_scaler': self.extra_pollutant_scaler,
        }
        joblib.dump(scalers, os.path.join(model_dir, f'{model_dir_name}_scalers.pkl'))
        print(f"✅ Scalers saved to {os.path.join(model_dir, f'{model_dir_name}_scalers.pkl')}")

        with open(os.path.join(model_dir, f'{model_dir_name}_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        if self.history is not None:
            with open(os.path.join(model_dir, f'{model_dir_name}_history.json'), 'w') as f:
                json.dump(self.history.history, f, indent=2)
        return model_dir


def load_data_from_hopsworks() -> pd.DataFrame:
    print("📊 Loading dataset from Hopsworks...")
    if HopsworksUploader is None:
        raise RuntimeError("Hopsworks integration not available. Please install and configure hopsworks.")

    hopsworks_key = os.getenv("HOPSWORKS_API_KEY")
    if not hopsworks_key:
        raise RuntimeError("HOPSWORKS_API_KEY environment variable is not set!")

    uploader = HopsworksUploader(api_key=hopsworks_key, project_name=HOPSWORKS_CONFIG['project_name'])
    if not uploader.connect():
        raise RuntimeError("Failed to connect to Hopsworks")

    fs = uploader.project.get_feature_store()
    fg = fs.get_or_create_feature_group(name=HOPSWORKS_CONFIG['feature_group_name'], version=1)
    df = fg.read()

    # Time sort and cleaning + standard preprocessing
    df = preprocess_aqi_df(df)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns. Time-sorted and NA-dropped.")
    return df


if __name__ == "__main__":
    print("🚀 v7: Full-sequence Teacher Forcing (train) + Autoregressive Rollout (val/test)")
    df = load_data_from_hopsworks()

    predictor = Seq2SeqFullSeqTFV7(SEQ2SEQ_V7_CONFIG)
    results = predictor.train(df)

    print("\n💾 Saving artifacts...")
    out_dir = predictor.save_artifacts()
    print(f"✅ Saved to {out_dir}")
    print("\n🎯 Done. Per-horizon metrics printed above for validation and test.")

