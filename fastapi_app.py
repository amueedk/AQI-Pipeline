"""
FastAPI app for on-demand AQI forecasts (Path B).

Features:
- Loads banded models (short, mid/long) once at startup (from Datasets Production or Model Registry)
- /predict: fetches features from Feature Views, runs inference, returns 72× PM and AQI
- /current: returns current PMs and AQI from chosen source (feature_view | openweather)
- Serves a modern single-page dashboard from / (ECharts)

Env:
- HOPSWORKS_API_KEY (required)
- MODEL_SHORT_NAME=direct_lstm_short (optional)
- MODEL_MIDLONG_NAME=direct_lstm_midlong (optional)
- CURRENT_SOURCE=feature_view | openweather (default: feature_view)
- PORT=8080
"""

import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import keras
import hopsworks
import requests

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import HOPSWORKS_CONFIG, OPENWEATHER_CONFIG
from data_collector import OpenWeatherDataCollector
from model_registry_utils import (
    download_production_artifacts,
    download_registry_artifacts,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fastapi_app")

app = FastAPI(title="AQI Forecasts", version="1.0")
# Ensure static dir exists before mounting
os.makedirs("static", exist_ok=True)


# ---------- Shared helpers (imported/trimmed from infer_online_lstm) ----------
def login_project():
    api_key = os.getenv("HOPSWORKS_API_KEY", "")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY not set")
    project_name = HOPSWORKS_CONFIG.get("project_name", "")
    if not project_name:
        raise RuntimeError("HOPSWORKS_CONFIG['project_name'] missing")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()
    return project, fs


def aqi_piecewise(pm: float, breakpoints: List[List[float]]) -> int:
    pm = max(0.0, float(pm))
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if pm <= c_hi:
            return int(round((i_hi - i_lo) / (c_hi - c_lo) * (pm - c_lo) + i_lo))
    return 500


def pm25_to_aqi(pm25: float) -> int:
    if not np.isfinite(pm25):
        return 0
    c = round(pm25 * 10) / 10.0
    bps = [
        [0.0, 12.0, 0, 50],
        [12.1, 35.4, 51, 100],
        [35.5, 55.4, 101, 150],
        [55.5, 150.4, 151, 200],
        [150.5, 250.4, 201, 300],
        [250.5, 350.4, 301, 400],
        [350.5, 500.4, 401, 500],
    ]
    return aqi_piecewise(c, bps)


def pm10_to_aqi(pm10: float) -> int:
    if not np.isfinite(pm10):
        return 0
    c = float(int(round(pm10)))
    bps = [
        [0.0, 54.0, 0, 50],
        [55.0, 154.0, 51, 100],
        [155.0, 254.0, 101, 150],
        [255.0, 354.0, 151, 200],
        [355.0, 424.0, 201, 300],
        [425.0, 504.0, 301, 400],
        [505.0, 604.0, 401, 500],
    ]
    return aqi_piecewise(c, bps)


def _wrap_serving_keys(time_strs: List[str]) -> List[Dict[str, str]]:
    return [{"time_str": ts} for ts in time_strs]


def build_time_str_list(start_ts: pd.Timestamp, count: int, step_hours: int, forward: bool) -> List[str]:
    series = [start_ts + pd.Timedelta(hours=i * step_hours * (1 if forward else -1)) for i in range(count)]
    return [t.floor('H').strftime('%Y-%m-%d %H:%M:%S') for t in series]


def fetch_encoder_window(fs, seq_len: int, enc_features: List[str]) -> pd.DataFrame:
    hist_fv = fs.get_feature_view("historic_fv", version=1)
    now_utc = pd.Timestamp.utcnow().floor('H')
    keys = build_time_str_list(now_utc, seq_len, 1, forward=False)
    keys = list(reversed(keys))
    vecs = hist_fv.get_feature_vectors(_wrap_serving_keys(keys))
    if isinstance(vecs, list) and len(vecs) > 0 and isinstance(vecs[0], dict):
        df = pd.DataFrame(vecs)
    else:
        feat_names = [getattr(f, 'name', str(f)) for f in getattr(hist_fv, 'features', [])]
        try:
            df = pd.DataFrame(vecs, columns=feat_names)
        except Exception:
            df = pd.DataFrame(vecs)
    miss = [c for c in enc_features if c not in df.columns]
    if miss:
        raise RuntimeError(f"historic_fv missing columns: {miss[:6]} ...")
    df = df.sort_values("time_str")[enc_features].reset_index(drop=True)
    if len(df) != seq_len:
        raise RuntimeError(f"Encoder window length={len(df)} != seq_len={seq_len}")
    return df


def fetch_forecasts(fs, steps: int) -> pd.DataFrame:
    fc_fv = fs.get_feature_view("forecasts_fv", version=1)
    now_utc = pd.Timestamp.utcnow().floor('H')
    future_keys = build_time_str_list(now_utc + pd.Timedelta(hours=1), steps, 1, forward=True)
    vecs = fc_fv.get_feature_vectors(_wrap_serving_keys(future_keys))
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
        raise RuntimeError(f"forecasts_fv missing columns: {miss[:6]} ...")
    df = df.sort_values("time_str").reset_index(drop=True)
    if len(df) != steps:
        raise RuntimeError(f"Forecast rows={len(df)} != steps={steps}. Re-run collector.")
    return df


def scale_encoder(enc_df: pd.DataFrame, enc_features: List[str], scalers: Dict[str, object]) -> np.ndarray:
    X = enc_df[enc_features].values.astype(np.float32)
    def idx_where(pred):
        return [i for i, c in enumerate(enc_features) if pred(c)]
    pm_idx = idx_where(lambda c: c.startswith('pm') or 'change_rate' in c)
    w_idx = idx_where(lambda c: c in ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction_sin', 'wind_direction_cos'])
    pol_idx = idx_where(lambda c: c in ['carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3'])
    inter_idx = idx_where(lambda c: 'interaction' in c)
    Xs = X.copy()
    def apply(idx, key):
        if not idx:
            return
        grp = X[:, idx]
        Xs[:, idx] = scalers[key].transform(grp)
    apply(pm_idx, 'pm'); apply(w_idx, 'weather'); apply(pol_idx, 'pollutant'); apply(inter_idx, 'interaction')
    # Safety: replace any non-finite values that may arise from scaling edge cases
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    return Xs[None, :, :]


def build_aux(fc_df: pd.DataFrame, steps: int, templates: List[str], scalers: Dict[str, object], pm0_scaled: np.ndarray) -> np.ndarray:
    w_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
    p_cols = ['carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']
    w_vals = fc_df[w_cols].values.astype(np.float32)
    ws = scalers['weather']
    try:
        means = np.asarray(getattr(ws, 'mean_', None)); scales = np.asarray(getattr(ws, 'scale_', None))
        if means is not None and scales is not None and means.shape[0] >= 4 and scales.shape[0] >= 4:
            w_scaled = (w_vals - means[:4]) / np.where(scales[:4] == 0.0, 1.0, scales[:4])
        else:
            w_scaled = ws.transform(w_vals)
    except Exception:
        w_scaled = ws.transform(w_vals)
    # Safety: replace any non-finite values
    w_scaled = np.nan_to_num(w_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    p_scaled = scalers['pollutant'].transform(fc_df[p_cols].values.astype(np.float32))
    p_scaled = np.nan_to_num(p_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    by_name = {
        'temperature_scaled': w_scaled[:, 0], 'humidity_scaled': w_scaled[:, 1], 'pressure_scaled': w_scaled[:, 2], 'wind_speed_scaled': w_scaled[:, 3],
        'wind_direction_sin': fc_df['wind_direction_sin'].values.astype(np.float32), 'wind_direction_cos': fc_df['wind_direction_cos'].values.astype(np.float32),
        'carbon_monoxide_scaled': p_scaled[:, 0], 'ozone_scaled': p_scaled[:, 1], 'sulphur_dioxide_scaled': p_scaled[:, 2], 'nh3_scaled': p_scaled[:, 3],
        'hour_sin': fc_df['hour_sin'].values.astype(np.float32), 'hour_cos': fc_df['hour_cos'].values.astype(np.float32),
        'day_sin': fc_df['day_sin'].values.astype(np.float32), 'day_cos': fc_df['day_cos'].values.astype(np.float32),
        'month_sin': fc_df['month_sin'].values.astype(np.float32), 'month_cos': fc_df['month_cos'].values.astype(np.float32),
        'day_of_week_sin': fc_df['day_of_week_sin'].values.astype(np.float32), 'day_of_week_cos': fc_df['day_of_week_cos'].values.astype(np.float32),
    }
    def base_from_template(t: str) -> str:
        if '_target_{h}h' in t: return t.replace('_target_{h}h', '')
        if t.endswith('_target_hh'): return t[:-len('_target_hh')]
        if '_target_' in t and t.endswith('h'): return t[: t.rfind('_target_')]
        return t
    cols = [base_from_template(t) for t in templates]
    F = len(cols); hexo = np.zeros((steps, F), dtype=np.float32)
    for i in range(steps):
        row = [by_name[c][i] for c in cols]
        hexo[i, :] = np.array(row, dtype=np.float32)
    # Safety: replace any non-finite values in auxiliary tensor
    hexo = np.nan_to_num(hexo, nan=0.0, posinf=0.0, neginf=0.0)
    h = np.arange(1, steps + 1, dtype=np.float32); ang = 2.0 * np.pi * (h / float(steps)); pos = np.stack([np.sin(ang), np.cos(ang)], axis=1)
    pm0_rep = np.repeat(pm0_scaled[None, :], repeats=steps, axis=0)
    return np.concatenate([hexo, pos.astype(np.float32), pm0_rep.astype(np.float32)], axis=1)[None, :, :]


def load_bundle(model_name: str, local_root: str) -> Dict:
    local_dir = os.path.join(local_root, model_name)
    if os.path.isdir(local_dir):
        # already cached; use as-is
        pass
    else:
        # Only try Model Registry - models should be deployed there
        ok = download_registry_artifacts(model_name, local_dir, version=None)
        if not ok:
            raise RuntimeError(f"Cannot download artifacts for {model_name} from Model Registry. Make sure models are deployed to Production in Hopsworks UI.")
    
    with open(os.path.join(local_dir, f"{model_name}_config.json"), 'r') as f:
        cfg = json.load(f)
    with open(os.path.join(local_dir, f"{model_name}_features.json"), 'r') as f:
        feats = json.load(f)
    scalers = joblib.load(os.path.join(local_dir, f"{model_name}_scalers.pkl"))
    model_path = os.path.join(local_dir, f"{model_name}.keras")
    
    # Load using tf.keras to match training save stack
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info(f"Successfully loaded {model_name}")
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        raise RuntimeError(f"Cannot load model {model_name}. Please retrain the models with the current environment.")
    
    return {"config": cfg, "features": feats, "scalers": scalers, "model": model}


# ---------- App state ----------
STATE: Dict[str, Dict] = {}


@app.on_event("startup")
def on_startup():
    short = os.getenv('MODEL_SHORT_NAME', 'direct_lstm_short')
    mid = os.getenv('MODEL_MIDLONG_NAME', 'direct_lstm_midlong')
    local_root = os.path.join('temp', 'production_models')
    os.makedirs(local_root, exist_ok=True)
    STATE['short'] = load_bundle(short, local_root)
    STATE['mid'] = load_bundle(mid, local_root)
    # Warm-up Hopsworks connection
    _ = login_project()
    logger.info("Startup complete: models cached and Hopsworks reachable")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/current")
def current():
    """Always fetch current PMs from OpenWeather and compute AQI."""
    try:
        url = f"{OPENWEATHER_CONFIG['base_url']}/air_pollution"
        params = {"lat": OPENWEATHER_CONFIG['lat'], "lon": OPENWEATHER_CONFIG['lon'], "appid": OPENWEATHER_CONFIG['api_key']}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        # Normalize root to a dict
        root = data[0] if isinstance(data, list) and data else data
        # Preferred: root['list'][0]['components']
        lst = root.get('list') if isinstance(root, dict) else None
        entry = (lst[0] if isinstance(lst, list) and lst else root) if root else {}
        comps = entry.get('components') if isinstance(entry, dict) else None
        if comps is None and isinstance(entry, dict):
            # Some variants may flatten components at the same level
            comps = {k: entry[k] for k in ('pm2_5','pm10') if k in entry}
        if isinstance(comps, list):
            comps = comps[0] if comps else {}
        if not isinstance(comps, dict):
            raise RuntimeError("Unexpected OpenWeather payload shape")
        pm25 = float(comps.get('pm2_5', 0.0))
        pm10 = float(comps.get('pm10', 0.0))
        aqi = int(max(pm25_to_aqi(pm25), pm10_to_aqi(pm10)))
        return {"pm2_5": pm25, "pm10": pm10, "aqi": aqi, "source": "openweather"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict")
def predict():
    try:
        _, fs = login_project()
        short = STATE['short']; mid = STATE['mid']
        # Short band
        steps_s = int(short['config'].get('num_steps', 12)); seq = int(short['config'].get('sequence_length', 72))
        enc_feats = short['features']['encoder_features']; templates_s = short['features']['horizon_exogenous_templates']
        enc_df = fetch_encoder_window(fs, seq, enc_feats)
        X_enc = scale_encoder(enc_df, enc_feats, short['scalers'])
        pm0 = enc_df.iloc[-1][['pm2_5', 'pm10']].values.astype(np.float32)
        pm0_scaled = short['scalers']['target_pm'].transform(pm0.reshape(1, -1)).reshape(-1)
        fc_df = fetch_forecasts(fs, 72)  # fetch full; aux builder uses first N
        X_aux_s = build_aux(fc_df.iloc[:steps_s], steps_s, templates_s, short['scalers'], pm0_scaled)
        y_s = short['model'].predict([X_enc, X_aux_s], verbose=0)[0]
        if bool(short['config'].get('use_delta_targets', True)):
            y_s = y_s + np.repeat(pm0_scaled[None, :], repeats=steps_s, axis=0)
        y_s = short['scalers']['target_pm'].inverse_transform(y_s)
        # Mid/long band (use remaining horizons)
        steps_m = 72
        templates_m = mid['features']['horizon_exogenous_templates']
        # Build encoder stream for mid/long with its own sequence length
        seq_m = int(mid['config'].get('sequence_length', 96))
        enc_feats_m = mid['features']['encoder_features']
        enc_df_m = fetch_encoder_window(fs, seq_m, enc_feats_m)
        X_enc_m = scale_encoder(enc_df_m, enc_feats_m, mid['scalers'])
        pm0_m = enc_df_m.iloc[-1][['pm2_5', 'pm10']].values.astype(np.float32)
        pm0_scaled_m = mid['scalers']['target_pm'].transform(pm0_m.reshape(1, -1)).reshape(-1)
        X_aux_m = build_aux(fc_df, steps_m, templates_m, mid['scalers'], pm0_scaled_m)
        y_m = mid['model'].predict([X_enc_m, X_aux_m], verbose=0)[0]
        if bool(mid['config'].get('use_delta_targets', False)):
            y_m = y_m + np.repeat(pm0_scaled_m[None, :], repeats=steps_m, axis=0)
        y_m = mid['scalers']['target_pm'].inverse_transform(y_m)
        # Merge outputs: first steps_s from short, rest from mid
        pm25 = y_s[:, 0].tolist() + y_m[steps_s:, 0].tolist()
        pm10 = y_s[:, 1].tolist() + y_m[steps_s:, 1].tolist()
        # Safety: sanitize any non-finite predictions before AQI computation
        pm25 = [float(v) if np.isfinite(v) else 0.0 for v in pm25]
        pm10 = [float(v) if np.isfinite(v) else 0.0 for v in pm10]
        aqi_pm25 = [pm25_to_aqi(v) for v in pm25]
        aqi_pm10 = [pm10_to_aqi(v) for v in pm10]
        aqi = [int(max(a, b)) for a, b in zip(aqi_pm25, aqi_pm10)]
        return {"pm25": pm25, "pm10": pm10, "aqi_pm25": aqi_pm25, "aqi_pm10": aqi_pm10, "aqi": aqi}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Static UI ----------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    # Minimal SPA served inline; assets loaded from CDN for speed
    return """
<!doctype html>
<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\"/>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
<title>AQI – 72h Forecast</title>\n<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\"/><link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap\" rel=\"stylesheet\"/>
<script src=\"https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js\"></script>
<style>body{margin:0;font-family:Inter,system-ui,Arial;background:#0b1020;color:#e7eaf6} .wrap{max-width:1100px;margin:0 auto;padding:16px} .hdr{display:flex;align-items:center;gap:12px} .badge{background:#1b2440;color:#7aa2f7;padding:4px 8px;border-radius:6px;font-size:12px} .grid{display:grid;gap:16px} .card{background:#0f1630;border:1px solid #1e2a4a;border-radius:12px;padding:12px} .row{display:flex;gap:12px;flex-wrap:wrap} .metric{flex:1;min-width:180px;background:#0f1630;border:1px solid #1e2a4a;border-radius:12px;padding:12px} .val{font-size:28px;font-weight:600}</style>
</head>\n<body>\n<div class=\"wrap\">\n  <div class=\"hdr\"><div style=\"font-size:20px;font-weight:600\">AQI – 72h Forecast</div><div class=\"badge\" id=\"updated\">Loading…</div></div>
  <div class=\"row\">\n    <div class=\"metric\"><div>Current AQI</div><div class=\"val\" id=\"aqi\">–</div></div>
    <div class=\"metric\"><div>PM2.5 (µg/m³)</div><div class=\"val\" id=\"pm25\">–</div></div>
    <div class=\"metric\"><div>PM10 (µg/m³)</div><div class=\"val\" id=\"pm10\">–</div></div>
  </div>
  <div class=\"card\"><div id=\"chart\" style=\"height:480px\"></div></div>
</div>
<script>
async function fetchJSON(u){const r=await fetch(u); if(!r.ok) throw new Error(await r.text()); return r.json();}
function bandColor(a){if(a<=50) return '#00a65a'; if(a<=100) return '#f7e463'; if(a<=150) return '#f39c32'; if(a<=200) return '#dd4b39'; if(a<=300) return '#8e44ad'; return '#7e0023';}
async function load(){
  const cur = await fetchJSON('/current');
  document.getElementById('aqi').textContent = cur.aqi; document.getElementById('pm25').textContent = cur.pm2_5.toFixed(1); document.getElementById('pm10').textContent = cur.pm10.toFixed(0);
  const pred = await fetchJSON('/predict');
  // Show last update in Pakistan time
  const fmtNow = new Intl.DateTimeFormat('en-PK',{ timeZone:'Asia/Karachi', hour:'2-digit', minute:'2-digit', hour12:false });
  document.getElementById('updated').textContent = 'Updated ' + fmtNow.format(new Date()) + ' PKT';
  // Build x-axis labels as actual local times (PKT) for next 72 hours
  const startUtc = new Date();
  startUtc.setUTCMinutes(0,0,0);
  startUtc.setUTCHours(startUtc.getUTCHours()+1);
  const fmt = new Intl.DateTimeFormat('en-PK', { timeZone:'Asia/Karachi', month:'short', day:'2-digit', hour:'2-digit', hour12:false });
  const x = Array.from({length:72}, (_,i)=>{
    const d = new Date(startUtc.getTime() + i*3600*1000);
    return fmt.format(d) + ' PKT';
  });
  // 24h stats cards
  function stats(a){const s=a.slice(0,24); const min=Math.min(...s), max=Math.max(...s), avg=s.reduce((p,c)=>p+c,0)/s.length; return {min, max, avg};}
  const aqiS=stats(pred.aqi), p25S=stats(pred.pm25), p10S=stats(pred.pm10);
  const ins=(id,v)=>document.getElementById(id).textContent=v;
  const keep1=x=>Number.isFinite(x)?x.toFixed(1):'–'; const keep0=x=>Number.isFinite(x)?Math.round(x):'–';
  // Inject cards if not present
  const wrap=document.querySelector('.wrap');
  const statsRow=`<div class=\"row\">\n    <div class=\"metric\"><div>Next 24h AQI</div><div class=\"val\" id=\"aqi24avg\">–</div><div style=\"font-size:12px;color:#9aa5c9\">min <span id=\"aqi24min\">–</span> · max <span id=\"aqi24max\">–</span></div></div>\n    <div class=\"metric\"><div>Next 24h PM2.5</div><div class=\"val\" id=\"pm25_24avg\">–</div><div style=\"font-size:12px;color:#9aa5c9\">min <span id=\"pm25_24min\">–</span> · max <span id=\"pm25_24max\">–</span></div></div>\n    <div class=\"metric\"><div>Next 24h PM10</div><div class=\"val\" id=\"pm10_24avg\">–</div><div style=\"font-size:12px;color:#9aa5c9\">min <span id=\"pm10_24min\">–</span> · max <span id=\"pm10_24max\">–</span></div></div>\n  </div>`;
  if(!document.getElementById('aqi24avg')) wrap.insertAdjacentHTML('beforeend', statsRow);
  ins('aqi24avg', keep0(aqiS.avg)); ins('aqi24min', keep0(aqiS.min)); ins('aqi24max', keep0(aqiS.max));
  ins('pm25_24avg', keep1(p25S.avg)); ins('pm25_24min', keep1(p25S.min)); ins('pm25_24max', keep1(p25S.max));
  ins('pm10_24avg', keep0(p10S.avg)); ins('pm10_24min', keep0(p10S.min)); ins('pm10_24max', keep0(p10S.max));
  const ch = echarts.init(document.getElementById('chart'));
  // AQI band shading
  const minX=x[0], maxX=x[x.length-1];
  const bands=[[0,50,'#00a65a'],[51,100,'#f7e463'],[101,150,'#f39c32'],[151,200,'#dd4b39'],[201,300,'#8e44ad'],[301,500,'#7e0023']];
  const markAreas=bands.map(b=>[{xAxis:minX,yAxis:b[0]},{xAxis:maxX,yAxis:b[1],itemStyle:{color:b[2],opacity:0.06}}]);
  ch.setOption({
    backgroundColor:'transparent',
    tooltip:{trigger:'axis'},
    legend:{textStyle:{color:'#c6cbe0'}},
    xAxis:{type:'category',data:x,axisLine:{lineStyle:{color:'#2b365a'}}},
    yAxis:[{type:'value',name:'AQI',axisLine:{lineStyle:{color:'#2b365a'}},splitLine:{lineStyle:{color:'#1e2a4a'}}},{type:'value',name:'PM',axisLine:{lineStyle:{color:'#2b365a'}},splitLine:{show:false}}],
    series:[
      {name:'AQI',type:'line',yAxisIndex:0,data:pred.aqi,smooth:true,areaStyle:{opacity:0.08},lineStyle:{width:2,color:'#7aa2f7'},markArea:{silent:true,data:markAreas}},
      {name:'PM2.5',type:'line',yAxisIndex:1,data:pred.pm25.map(v=>v.toFixed(1)),smooth:true,lineStyle:{width:1,color:'#4fd1c5'}},
      {name:'PM10',type:'line',yAxisIndex:1,data:pred.pm10.map(v=>Math.round(v)),smooth:true,lineStyle:{width:1,color:'#f6ad55'}}
    ]
  });
}
load().catch(e=>{document.getElementById('updated').textContent='Error'; console.error(e)});
</script>
</body>\n</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))


