# Direct Multi-Horizon LSTM for AQI Forecasting

## Overview

The `lstm_direct_multi_horizon_v1.py` implements a **direct multi-step forecasting** approach for predicting Air Quality Index (AQI) values (PM2.5 and PM10) for the next 72 hours (3 days) in a single forward pass. Unlike autoregressive models that predict one step at a time and feed predictions back, this model predicts all 72 horizons simultaneously.

## Why This Architecture?

### Problem with Autoregressive Models
Traditional autoregressive models suffer from:
- **Exposure Bias**: Training uses ground truth, but inference uses predicted values, causing error accumulation
- **Slow Inference**: Must run 72 sequential predictions for a 72-hour forecast
- **Error Propagation**: Errors in early predictions compound in later steps

### Benefits of Direct Multi-Horizon
- **No Exposure Bias**: Training and inference are identical (no teacher forcing)
- **Fast Inference**: Single forward pass for all 72 predictions
- **Parallel Prediction**: All horizons predicted simultaneously
- **Better Long-Horizon Performance**: No error accumulation

## Architecture Design

### High-Level Structure
```
Input: Historical Window (72h) → Encoder → Context Vector
                                    ↓
Target Horizon Features (72h) → Decoder → 72 PM Predictions
```

### Detailed Architecture

#### 1. Encoder (Historical Context)
- **Input**: 72-hour historical window of engineered features
- **Architecture**: Stacked LSTM layers with BatchNorm
- **Output**: Context vector (128-dimensional) summarizing historical patterns
- **Purpose**: Capture temporal dependencies and patterns in historical data

#### 2. Decoder (Multi-Horizon Prediction)
- **Input**: 
  - Repeated context vector (72 times)
  - Target horizon exogenous features (weather, pollutants at t+h)
  - Positional embeddings (sin/cos encoding of step position)
  - Current PM values (repeated across all steps)
- **Architecture**: Single LSTM layer with TimeDistributed Dense output
- **Output**: 72 × 2 predictions (PM2.5, PM10 for each hour)

## Feature Engineering

### Encoder Features (Historical Context)

#### PM Block (Primary Pollutants)
- **Raw Values**: `pm2_5`, `pm10`
- **Rolling Statistics**: 
  - 3h, 12h, 24h rolling means and maxes
  - Captures short-term trends and peaks
- **Change Rates**: 
  - 1h, 6h, 24h change rates
  - Captures momentum and acceleration

#### Weather Features
- **Basic**: `temperature`, `humidity`, `pressure`, `wind_speed`
- **Wind Direction**: `wind_direction_sin`, `wind_direction_cos` (cyclical encoding)

#### Time Features (Cyclical)
- **Hour**: `hour_sin`, `hour_cos`
- **Day**: `day_sin`, `day_cos`
- **Month**: `month_sin`, `month_cos`
- **Day of Week**: `day_of_week_sin`, `day_of_week_cos`

#### Additional Pollutants
- `carbon_monoxide`, `ozone`, `sulphur_dioxide`, `nh3`

#### Interaction Features
- `pm2_5_temp_interaction`, `pm2_5_humidity_interaction`, `pm2_5_pressure_interaction`
- `pm10_temperature_interaction`, `pm10_pressure_interaction`

### Decoder Features (Target Horizon)

#### Target Horizon Exogenous (t+h)
For each horizon h (1-72), the model receives:
- **Weather at t+h**: `temperature_scaled_target_{h}h`, `humidity_scaled_target_{h}h`, etc.
- **Pollutants at t+h**: `carbon_monoxide_scaled_target_{h}h`, `ozone_scaled_target_{h}h`, etc.
- **Wind Direction at t+h**: `wind_direction_sin_target_{h}h`, `wind_direction_cos_target_{h}h`
- **Time at t+h**: `hour_sin_target_{h}h`, `day_sin_target_{h}h`, etc.

#### Positional Embeddings
- **Purpose**: Help model understand which horizon it's predicting
- **Encoding**: `sin(2πh/72)`, `cos(2πh/72)` for each step h
- **Benefits**: Provides explicit position information to the decoder

#### Current PM Values
- **Purpose**: Provide baseline reference for predictions
- **Implementation**: Current PM2.5 and PM10 values repeated across all 72 steps
- **Benefits**: Helps model predict relative changes from current state

## Loss Function Design

### Horizon-Weighted Loss
The model uses a custom loss function that emphasizes different horizons:

```python
def weighted_loss(y_true, y_pred):
    err = y_true - y_pred
    mse = tf.reduce_mean(tf.square(err), axis=2)   # [B,steps]
    mae = tf.reduce_mean(tf.abs(err), axis=2)
    w = tf.constant(weights, dtype=tf.float32)
    return tf.reduce_mean((mse + 0.2 * mae) * w)
```

### Weight Configuration
- **1-3h**: Weight = 5.0 (highest priority for immediate forecasts)
- **4-6h**: Weight = 3.0 (high priority for short-term)
- **7-24h**: Weight = 3.0 (medium priority for daily forecasts)
- **25-72h**: Weight = 1.0 (lower priority for distant forecasts)

### Loss Components
- **MSE**: Primary error metric
- **MAE**: Secondary error metric (0.2 weight)
- **Horizon Weights**: Emphasize short-term accuracy

## Training Strategy

### Banded Training Approach
The model uses a sophisticated banded training strategy:

#### Short Model (1-12h)
- **Targets**: Delta predictions (change from current PM)
- **Weights**: Heavy emphasis on 1-6h (weights 8.0, 4.0)
- **Context**: 72-hour historical window
- **Purpose**: Optimize for immediate and short-term accuracy

#### Mid/Long Model (12-72h)
- **Targets**: Absolute PM predictions
- **Weights**: Balanced across horizons (weights 1.0-3.0)
- **Context**: 96-hour historical window (longer context)
- **Purpose**: Optimize for medium to long-term forecasts

### Delta vs Absolute Targets
- **Delta Training**: Model predicts `PM(t+h) - PM(t)` in scaled space
- **Benefits**: Easier for model to learn relative changes
- **Reconstruction**: At inference, add current PM to get absolute predictions
- **Use Case**: Short-term forecasts where relative changes are more predictable

## Model Configuration

### Architecture Parameters
```python
CONFIG = {
    'sequence_length': 72,        # Historical context window
    'num_steps': 72,             # Prediction horizons
    'encoder_units': [128, 64],  # LSTM units per encoder layer
    'decoder_units': 192,        # LSTM units in decoder
    'dropout_rate': 0.3,         # Regularization
    'learning_rate': 3e-4,       # Adam optimizer learning rate
    'batch_size': 32,            # Training batch size
    'epochs': 80,                # Maximum training epochs
}
```

### Training Parameters
- **Early Stopping**: Patience = 12 epochs
- **Learning Rate Reduction**: Patience = 6 epochs, factor = 0.6
- **Minimum Learning Rate**: 1e-6
- **Test Split**: 20% for validation

## Data Preprocessing

### Robust Cleaning Pipeline
```python
def preprocess_aqi_df(df):
    # 1. Deduplication
    df = df.drop_duplicates()
    
    # 2. NaN handling
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.dropna()
    
    # 3. Remove non-positive PM values
    df = df[df['pm2_5'] > 0]
    df = df[df['pm10'] > 0]
    
    # 4. IQR outlier removal
    for col in ['pm2_5', 'pm10', 'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3']:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df[(df[col] >= lb) & (df[col] <= ub)]
```

### Feature Scaling
- **PM Features**: Scaled together (includes rolling stats and change rates)
- **Weather Features**: Scaled together
- **Pollutant Features**: Scaled together
- **Interaction Features**: Scaled together
- **Time Features**: Left unscaled (already cyclical)

## Inference Process

### Input Requirements
1. **Historical Data**: 72 hours of historical features (encoder input)
2. **Future Exogenous**: 72 hours of weather/pollutant forecasts (decoder input)
3. **Current PM**: Current PM2.5 and PM10 values

### Output Format
- **Shape**: (72, 2) array
- **Columns**: [PM2.5, PM10] for each hour 1-72
- **Units**: Same as input (typically μg/m³)

### Inference Steps
1. Scale historical features using saved scalers
2. Scale current PM values
3. Build target horizon exogenous tensor (72 × feature_dim)
4. Generate positional embeddings
5. Concatenate all decoder inputs
6. Run model forward pass
7. Inverse scale predictions
8. For delta models: add current PM to get absolute predictions

## Model Persistence

### Saved Artifacts
- **Model**: `direct_lstm_multi_horizon.keras`
- **Scalers**: `direct_lstm_multi_horizon_scalers.pkl`
- **Configuration**: `direct_lstm_multi_horizon_config.json`
- **Feature Lists**: `direct_lstm_multi_horizon_features.json`

### Loading for Inference
```python
# Load model and artifacts
model = tf.keras.models.load_model('model.keras')
scalers = joblib.load('scalers.pkl')
with open('config.json', 'r') as f:
    config = json.load(f)
with open('features.json', 'r') as f:
    features = json.load(f)
```

## SHAP Explainability

### Implementation
- Uses `shap.KernelExplainer` for complex LSTM architecture
- Analyzes feature importance per horizon and target (PM2.5/PM10)
- Generates beeswarm and bar plots for key horizons

### Feature Attribution
- **Encoder Features**: Historical context importance
- **Decoder Features**: Target horizon exogenous importance
- **Positional Features**: Step position importance
- **Current PM**: Baseline reference importance

## Performance Characteristics

### Expected Results
- **1h**: RMSE ~2.9, R² ~0.95 (excellent short-term)
- **6h**: RMSE ~4.5, R² ~0.85 (good short-term)
- **12h**: RMSE ~6.2, R² ~0.70 (acceptable daily)
- **24h**: RMSE ~8.1, R² ~0.50 (moderate daily)
- **48h**: RMSE ~10.5, R² ~0.20 (challenging)
- **72h**: RMSE ~12.5, R² ~0.10 (difficult long-term)

### Strengths
- Excellent short-term accuracy (1-6h)
- Fast inference (single forward pass)
- No exposure bias
- Robust to outliers and missing data

### Limitations
- Declining accuracy with horizon distance
- Requires accurate future exogenous forecasts
- Complex architecture may be harder to interpret
- Memory intensive for long sequences

## Usage Examples

### Training
```bash
# Default banded training
python lstm_direct_multi_horizon_v1.py

# Single model training
SINGLE_RUN=1 python lstm_direct_multi_horizon_v1.py

# Walk-forward cross-validation
WALK_CV=1 python lstm_direct_multi_horizon_v1.py
```

### Environment Variables
- `SINGLE_RUN=1`: Train single model instead of banded
- `WALK_CV=1`: Run walk-forward cross-validation
- `WALK_CV_FOLDS=4`: Number of CV folds (default: 4)
- `HOPSWORKS_API_KEY`: For loading data from Hopsworks

## Integration with Forecast Collector

The model requires future exogenous features from `automated_forecast_collector.py`:
- Weather forecasts (temperature, humidity, pressure, wind)
- Pollution forecasts (CO, O3, SO2, NH3)
- Time features (cyclical encodings)

The collector provides raw feature names that are scaled and assembled into the decoder input tensor during inference.

## Comparison with Other Models

### vs Autoregressive LSTM
- **Direct**: Single forward pass vs 72 sequential passes
- **Direct**: No exposure bias vs significant exposure bias
- **Direct**: Better long-horizon performance vs error accumulation

### vs LightGBM Direct
- **LSTM**: Better temporal modeling vs tabular approach
- **LSTM**: More complex architecture vs simpler tree-based
- **LSTM**: Requires more data vs works with smaller datasets

### vs Ensemble Methods
- **LSTM**: Single model vs multiple models
- **LSTM**: End-to-end training vs separate training
- **LSTM**: Better feature interactions vs independent models

## Future Improvements

### Potential Enhancements
1. **Attention Mechanism**: Add attention to focus on relevant historical patterns
2. **Multi-Scale Encoder**: Different context windows for different horizons
3. **Probabilistic Outputs**: Predict uncertainty intervals
4. **Online Learning**: Update model with new data
5. **Ensemble**: Combine multiple LSTM variants

### Research Directions
- **Transformer Architecture**: Replace LSTM with attention-based models
- **Graph Neural Networks**: Model spatial relationships between monitoring stations
- **Physics-Informed**: Incorporate atmospheric modeling constraints
- **Multi-Task Learning**: Predict multiple air quality metrics simultaneously
