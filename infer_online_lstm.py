"""
Online inference for Direct LSTM using Hopsworks Feature Views.

- Loads Production artifacts for one or two models (banded optional)
- Pulls encoder window from `historic_fv` and 72h aux from `forecasts_fv`
- Builds tensors to match training preprocessing (group scalers, aux layout)
- Returns 72× [PM2.5, PM10] plus AQI per step

Usage (CLI example):
  HOPSWORKS_API_KEY=... python infer_online_lstm.py

Environment options:
  MODEL_SHORT_NAME=direct_lstm_short      # optional band 1–12
  MODEL_MIDLONG_NAME=direct_lstm_midlong  # optional band 12–72
  SINGLE_MODEL_NAME=direct_lstm_multi_horizon  # used if banded names missing
"""

import os
import json
import math
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import hopsworks

from config import HOPSWORKS_CONFIG
from model_registry_utils import download_production_artifacts, download_registry_artifacts


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_online_lstm")


def login_project():
    api_key = os.getenv("HOPSWORKS_API_KEY", "")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY not set")
    project_name = HOPSWORKS_CONFIG.get("project_name", "")
    if not project_name:
        raise RuntimeError("HOPSWORKS_CONFIG['project_name'] missing")
    logger.info(f"Logging into Hopsworks project: {project_name}...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()
    logger.info("Connected to Hopsworks Feature Store")
    return project, fs


def aqi_piecewise(pm: float, breakpoints: List[Tuple[float, float, int, int]]) -> int:
    pm = max(0.0, float(pm))
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if pm <= c_hi:
            # EPA rounding specifics: PM2.5 to 1 decimal, PM10 to integer handled before passing in
            return int(round((i_hi - i_lo) / (c_hi - c_lo) * (pm - c_lo) + i_lo))
    return 500


def pm25_to_aqi(pm25: float) -> int:
    # Breakpoints in µg/m³ for PM2.5 (rounded to 0.1 before mapping)
    c = round(pm25 * 10) / 10.0
    bps = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    return aqi_piecewise(c, bps)


def pm10_to_aqi(pm10: float) -> int:
    # Breakpoints for PM10 (rounded to nearest integer before mapping)
    c = float(int(round(pm10)))
    bps = [
        (0.0, 54.0, 0, 50),
        (55.0, 154.0, 51, 100),
        (155.0, 254.0, 101, 150),
        (255.0, 354.0, 151, 200),
        (355.0, 424.0, 201, 300),
        (425.0, 504.0, 301, 400),
        (505.0, 604.0, 401, 500),
    ]
    return aqi_piecewise(c, bps)


def build_time_str_list(start_ts: pd.Timestamp, count: int, step_hours: int, forward: bool) -> List[str]:
    series = [start_ts + pd.Timedelta(hours=i * step_hours * (1 if forward else -1)) for i in range(count)]
    return [t.floor('H').strftime('%Y-%m-%d %H:%M:%S') for t in series]


def _wrap_serving_keys(time_strs: List[str]) -> List[Dict[str, str]]:
    """Hopsworks get_feature_vectors expects a list of dicts {entity: value}."""
    return [{"time_str": ts} for ts in time_strs]


def fetch_encoder_window(fs, seq_len: int, enc_features: List[str]) -> pd.DataFrame:
    hist_fv = fs.get_feature_view("historic_fv", version=1)
    now_utc = pd.Timestamp.utcnow().floor('H')
    keys = build_time_str_list(now_utc, seq_len, 1, forward=False)
    keys = list(reversed(keys))  # oldest→newest
    vecs = hist_fv.get_feature_vectors(_wrap_serving_keys(keys))
    # Normalize return type
    if isinstance(vecs, list) and len(vecs) > 0 and isinstance(vecs[0], dict):
        df = pd.DataFrame(vecs)
    else:
        # Fallback: assume list-of-lists aligned to fv.features order
        feat_names = [getattr(f, 'name', str(f)) for f in getattr(hist_fv, 'features', [])]
        try:
            df = pd.DataFrame(vecs, columns=feat_names)
        except Exception:
            df = pd.DataFrame(vecs)
    # Ensure order and presence
    missing = [c for c in enc_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing encoder columns in historic_fv: {missing[:6]} ...")
    df = df.sort_values("time_str")
    df = df[enc_features].reset_index(drop=True)
    if len(df) != seq_len:
        raise ValueError(f"Encoder window length={len(df)} != seq_len={seq_len}")
    return df


def fetch_forecasts(fs, steps: int) -> pd.DataFrame:
    fc_fv = fs.get_feature_view("forecasts_fv", version=1)
    now_utc = pd.Timestamp.utcnow().floor('H')
    future_keys = build_time_str_list(now_utc + pd.Timedelta(hours=1), steps, 1, forward=True)
    vecs = fc_fv.get_feature_vectors(_wrap_serving_keys(future_keys))
    # Normalize return type
    if isinstance(vecs, list) and len(vecs) > 0 and isinstance(vecs[0], dict):
        df = pd.DataFrame(vecs)
    else:
        feat_names = [getattr(f, 'name', str(f)) for f in getattr(fc_fv, 'features', [])]
        try:
            df = pd.DataFrame(vecs, columns=feat_names)
        except Exception:
            df = pd.DataFrame(vecs)
    need = [
        'temperature', 'humidity', 'pressure', 'wind_speed',
        'wind_direction_sin', 'wind_direction_cos',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
        'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3'
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing forecast columns in forecasts_fv: {miss[:6]} ...")
    df = df.sort_values("time_str").reset_index(drop=True)
    if len(df) != steps:
        raise ValueError(f"Forecast rows={len(df)} != steps={steps}")
    return df


def scale_encoder_frame(enc_df: pd.DataFrame, enc_features: List[str], scalers: Dict[str, object]) -> np.ndarray:
    X = enc_df[enc_features].values.astype(np.float32)
    # group indices
    def idx_where(pred):
        return [i for i, c in enumerate(enc_features) if pred(c)]
    pm_idx = idx_where(lambda c: c.startswith('pm') or 'change_rate' in c)
    w_idx = idx_where(lambda c: c in ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction_sin', 'wind_direction_cos'])
    pol_idx = idx_where(lambda c: c in ['carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3'])
    inter_idx = idx_where(lambda c: 'interaction' in c)
    # apply
    X_s = X.copy()
    def apply_group(idx_list, scaler_key):
        if not idx_list:
            return
        group = X[:, idx_list]
        X_s[:, idx_list] = scalers[scaler_key].transform(group)
    apply_group(pm_idx, 'pm')
    apply_group(w_idx, 'weather')
    apply_group(pol_idx, 'pollutant')
    apply_group(inter_idx, 'interaction')
    return X_s[None, :, :]  # [1,seq,feat]


def build_aux_from_forecasts(fc_df: pd.DataFrame, steps: int, templates: List[str], scalers: Dict[str, object], pm0_scaled: np.ndarray) -> np.ndarray:
    # Precompute scaled weather and pollutants
    w_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
    p_cols = ['carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']
    # Weather scaler in training was fit on 6 dims (temp, humidity, pressure, wind_speed, wind_dir_sin, wind_dir_cos)
    # but horizon exogenous only uses the first 4 scaled. Use manual scaling from the saved params to avoid dim mismatch.
    w_vals = fc_df[w_cols].values.astype(np.float32)
    ws = scalers['weather']
    try:
        means = np.asarray(getattr(ws, 'mean_', None))
        scales = np.asarray(getattr(ws, 'scale_', None))
        if means is not None and scales is not None and means.shape[0] >= 4 and scales.shape[0] >= 4:
            w_scaled = (w_vals - means[:4]) / np.where(scales[:4] == 0.0, 1.0, scales[:4])
        else:
            w_scaled = ws.transform(w_vals)
    except Exception:
        w_scaled = ws.transform(w_vals)
    p_scaled = scalers['pollutant'].transform(fc_df[p_cols].values.astype(np.float32))
    # Build map for quick lookup
    by_name = {
        'temperature_scaled': w_scaled[:, 0],
        'humidity_scaled': w_scaled[:, 1],
        'pressure_scaled': w_scaled[:, 2],
        'wind_speed_scaled': w_scaled[:, 3],
        'wind_direction_sin': fc_df['wind_direction_sin'].values.astype(np.float32),
        'wind_direction_cos': fc_df['wind_direction_cos'].values.astype(np.float32),
        'carbon_monoxide_scaled': p_scaled[:, 0],
        'ozone_scaled': p_scaled[:, 1],
        'sulphur_dioxide_scaled': p_scaled[:, 2],
        'nh3_scaled': p_scaled[:, 3],
        'hour_sin': fc_df['hour_sin'].values.astype(np.float32),
        'hour_cos': fc_df['hour_cos'].values.astype(np.float32),
        'day_sin': fc_df['day_sin'].values.astype(np.float32),
        'day_cos': fc_df['day_cos'].values.astype(np.float32),
        'month_sin': fc_df['month_sin'].values.astype(np.float32),
        'month_cos': fc_df['month_cos'].values.astype(np.float32),
        'day_of_week_sin': fc_df['day_of_week_sin'].values.astype(np.float32),
        'day_of_week_cos': fc_df['day_of_week_cos'].values.astype(np.float32),
    }
    # Helper to normalize template names stored as either '*_target_{h}h' or '*_target_hh'
    def base_from_template(t: str) -> str:
        if '_target_{h}h' in t:
            return t.replace('_target_{h}h', '')
        if t.endswith('_target_hh'):
            return t[: -len('_target_hh')]
        if '_target_' in t and t.endswith('h'):
            # generic fallback: strip suffix after '_target_'
            return t[: t.rfind('_target_')]
        return t

    # Assemble exogenous in the exact training order
    F = 0
    cols_for_step: List[str] = []
    for t in templates:
        # t like 'temperature_scaled_target_{h}h' or realized '..._target_hh'
        name = base_from_template(t)
        cols_for_step.append(name)
        F += 1
    hexo = np.zeros((steps, F), dtype=np.float32)
    for h in range(steps):
        row = []
        for name in cols_for_step:
            if name not in by_name:
                raise KeyError(f"Missing decoder feature '{name}' in forecasts dataframe. Available: {list(by_name.keys())[:8]} ...")
            row.append(by_name[name][h])
        hexo[h, :] = np.array(row, dtype=np.float32)
    # Positional encoding
    h = np.arange(1, steps + 1, dtype=np.float32)
    ang = 2.0 * np.pi * (h / float(steps))
    pos = np.stack([np.sin(ang), np.cos(ang)], axis=1)
    # Repeat pm0 across steps
    pm0_rep = np.repeat(pm0_scaled[None, :], repeats=steps, axis=0)  # [steps,2]
    # Concatenate: [hexo | pos | pm0]
    aux = np.concatenate([hexo, pos.astype(np.float32), pm0_rep.astype(np.float32)], axis=1)
    return aux[None, :, :]  # [1,steps,F+2+2]


def load_production(model_name: str, local_root: str) -> Dict:
    local_dir = os.path.join(local_root, model_name)
    os.makedirs(local_dir, exist_ok=True)
    # Prefer Datasets Production path; if missing, try Model Registry latest
    ok = download_production_artifacts(model_name, local_dir)
    if not ok:
        logger.info(f"Datasets Production not found for {model_name}, trying Model Registry…")
        ok = download_registry_artifacts(model_name, local_dir, version=None)
    if not ok:
        raise RuntimeError(f"Failed to download artifacts for {model_name}")
    with open(os.path.join(local_dir, f"{model_name}_config.json"), 'r') as f:
        cfg = json.load(f)
    with open(os.path.join(local_dir, f"{model_name}_features.json"), 'r') as f:
        feats = json.load(f)
    scalers = joblib.load(os.path.join(local_dir, f"{model_name}_scalers.pkl"))
    model = tf.keras.models.load_model(os.path.join(local_dir, f"{model_name}.keras"), compile=False)
    return {"config": cfg, "features": feats, "scalers": scalers, "model": model}


def predict_with_model(fs, bundle: Dict) -> Dict:
    steps = int(bundle['config'].get('num_steps', 72))
    seq_len = int(bundle['config'].get('sequence_length', 72))
    enc_feats: List[str] = bundle['features']['encoder_features']
    templates: List[str] = bundle['features']['horizon_exogenous_templates']
    # Fetch data
    enc_df = fetch_encoder_window(fs, seq_len, enc_feats)
    fc_df = fetch_forecasts(fs, steps)
    # Build tensors
    X_enc = scale_encoder_frame(enc_df, enc_feats, bundle['scalers'])
    pm0 = enc_df.iloc[-1][['pm2_5', 'pm10']].values.astype(np.float32)
    pm0_scaled = bundle['scalers']['target_pm'].transform(pm0.reshape(1, -1)).reshape(-1)
    X_aux = build_aux_from_forecasts(fc_df, steps, templates, bundle['scalers'], pm0_scaled)
    # Predict
    y_pred_scaled = bundle['model'].predict([X_enc, X_aux], verbose=0)[0]  # [steps,2]
    if bool(bundle['config'].get('use_delta_targets', True)):
        y_pred_scaled = y_pred_scaled + np.repeat(pm0_scaled[None, :], repeats=steps, axis=0)
    y_pred = bundle['scalers']['target_pm'].inverse_transform(y_pred_scaled)
    # AQI
    aqi_pm25 = [pm25_to_aqi(v) for v in y_pred[:, 0].tolist()]
    aqi_pm10 = [pm10_to_aqi(v) for v in y_pred[:, 1].tolist()]
    aqi = [int(max(a, b)) for a, b in zip(aqi_pm25, aqi_pm10)]
    return {
        'pm25': y_pred[:, 0].tolist(),
        'pm10': y_pred[:, 1].tolist(),
        'aqi_pm25': aqi_pm25,
        'aqi_pm10': aqi_pm10,
        'aqi': aqi,
    }


def infer() -> Dict:
    project, fs = login_project()
    # Choose banded if available, else single
    short_name = os.getenv('MODEL_SHORT_NAME', 'direct_lstm_short')
    mid_name = os.getenv('MODEL_MIDLONG_NAME', 'direct_lstm_midlong')
    single_name = os.getenv('SINGLE_MODEL_NAME', 'direct_lstm_multi_horizon')
    local_root = os.path.join('temp', 'production_models')
    os.makedirs(local_root, exist_ok=True)
    # Strict banded-only: both must exist
    try:
        short_bundle = load_production(short_name, local_root)
    except Exception as e:
        raise RuntimeError(f"Missing Production artifacts for short model '{short_name}' (/Models/{short_name}/Production): {e}")
    try:
        mid_bundle = load_production(mid_name, local_root)
    except Exception as e:
        raise RuntimeError(f"Missing Production artifacts for mid/long model '{mid_name}' (/Models/{mid_name}/Production): {e}")

    res_short = predict_with_model(fs, short_bundle)
    res_mid = predict_with_model(fs, mid_bundle)
    pm25 = res_short['pm25'][:12] + res_mid['pm25'][12:]
    pm10 = res_short['pm10'][:12] + res_mid['pm10'][12:]
    aqi_pm25 = res_short['aqi_pm25'][:12] + res_mid['aqi_pm25'][12:]
    aqi_pm10 = res_short['aqi_pm10'][:12] + res_mid['aqi_pm10'][12:]
    aqi = [int(max(a, b)) for a, b in zip(aqi_pm25, aqi_pm10)]
    return {'pm25': pm25, 'pm10': pm10, 'aqi_pm25': aqi_pm25, 'aqi_pm10': aqi_pm10, 'aqi': aqi}


if __name__ == '__main__':
    out = infer()
    print(json.dumps(out, indent=2))


