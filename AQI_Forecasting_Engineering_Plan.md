# 🎯 AQI FORECASTING ENGINEERING PLAN
## Based on Comprehensive EDA Analysis (Sections 2-15)

---

## 📊 **EXECUTIVE SUMMARY**

**Target**: Predict hourly PM2.5 and PM10 concentrations for next 72 hours → Calculate AQI  
**Data Quality**: Excellent (99.9% complete, physically consistent)  
**Feasibility**: **HIGHLY FEASIBLE** with phased approach  
**Success Criteria**: RMSE targets and AQI category accuracy defined below

**Key Innovation**: **Wind Direction Engineering** - Critical missing piece for accurate prediction

---

## 🔍 **COMPREHENSIVE EDA FINDINGS SUMMARY**

### **Original EDA (Sections 2-8) - Core Analysis**

#### **Section 2 - Time Series Analysis**
- **PM2.5 Range**: 0.0-100.6 µg/m³ (extreme spikes around July 15th)
- **PM10 Range**: 0.0-222.0 µg/m³ 
- **US AQI Range**: 0.0-174.0
- **Patterns**: High variability, declining trend, daily/weekly cycles
- **Key Insight**: **PM2.5 and PM10 show strong temporal patterns** - good for forecasting

#### **Section 3 - Feature Correlation Analysis**
- **PM2.5 Critical Predictors**: PM10 (0.554), CO (0.503), Ozone (0.429), Pressure (0.394), SO2 (0.386), NO2 (0.266), Wind Speed (0.251)
- **PM10 Critical Predictors**: PM2.5 (0.554), CO (0.503), Ozone (0.429), Pressure (0.394), SO2 (0.386), NO2 (0.266), Wind Speed (0.251)
- **Redundancy Issue**: CO-NO2 correlation (0.69) - **HANDLE THIS**
- **Key Insight**: **Pressure is the dominant weather driver** for both pollutants

#### **Section 4 - AQI Dominance Analysis**
- **PM2.5 Controls AQI**: 94.7% of the time
- **PM10 Controls AQI**: 4.4% of the time  
- **PM-only AQI vs Full EPA AQI**: **99.8% identical**
- **Key Insight**: **PM-only approach is sufficient** - focus 100% on PM prediction accuracy

#### **Section 5 - Lag Features Analysis**
- **PM2.5 Persistence**: 1h (0.977), 24h (0.656), 72h (0.460)
- **PM10 Persistence**: 1h (0.958), 6h (0.500), 12h (0.198), 72h (0.405)
- **Key Insight**: **Strong short-term persistence** - lag features are valuable inputs
- **Scope**: Basic correlation analysis of current PM values vs historical lag features

#### **Section 6 - Data Quality Analysis**
- **Missing Values**: 0.1% (1 zero value each for PM2.5/PM10)
- **Physical Consistency**: 100% (PM2.5 ≤ PM10 always)
- **Weather Ranges**: All within reasonable bounds
- **Key Insight**: **Data quality is excellent** - minimal preprocessing needed

#### **Section 7 - Distribution Analysis**
- **PM2.5**: Moderate right skew (0.702) → **Log(1+x) transformation needed**
- **PM10**: Good distribution (0.442) → **No transformation needed**
- **Outlier Strategy**: 
  - **PM2.5**: Remove outliers (>78.4 µg/m³) - they're sensor errors
  - **PM10**: Keep outliers (>175.8 µg/m³) - they're real pollution events
- **Key Insight**: **Different preprocessing strategies** for each pollutant

#### **Section 8 - Forecasting Performance Analysis**
- **8.1 Temporal Autocorrelation**: 
  - PM2.5: Strong up to 24h (0.625), moderate at 72h (0.403)
  - PM10: Strong at 1h (0.957), erratic pattern, moderate at 72h (0.389)
- **8.2 Lag Feature Effectiveness**:
  - PM2.5: 1h, 2h, 3h lags have 100% utility
  - PM10: 2h lag (100%), 1h (80%), 3h (60%)
  - **6h+ lags are useless** for both
  - **Scope**: Advanced multi-horizon forecasting analysis (past PM → future PM predictions)
- **8.3 Weather Lead-Lag Relationships**:
  - PM2.5: Pressure-driven (0.397-0.405 correlation)
  - PM10: Weak weather correlation, temperature best but still weak
- **8.4 Forecasting Feasibility**:
  - PM2.5: Day 1 (Excellent), Day 2 (Moderate), Day 3 (Challenging)
  - PM10: Day 1 (Excellent), Day 2 (Moderate), Day 3 (Difficult)

### **Comprehensive EDA (Sections 9-15) - Advanced Analysis**

#### **Section 9 - Rolling Features Analysis**
- **32 Rolling Features**: 16 for PM2.5 + 16 for PM10 (4 windows × 4 stats × 2 targets)
- **Windows**: 3h, 6h, 12h, 24h
- **Statistics**: mean, std, min, max
- **Key Findings**:
  - **STD features are USELESS**: All correlations < 0.13, most negative
  - **3h and 6h windows dominate**: Best correlations across all statistics
  - **MIN features slightly better**: More predictive than MEAN for both targets
  - **PM2.5 more predictable**: Higher correlations than PM10
- **Multi-Horizon Insights**:
  - **3h MIN features dominate**: Best for short-term (1-6h) predictions
  - **PM2.5 more predictable**: Maintains 0.45+ correlation even at 72h
  - **PM10 decays rapidly**: Drops to 0.18 at 72h horizon
  - **12h+ windows useful**: For longer-term predictions
- **Engineering Recommendation**: **32 features → 7 features** (78% reduction)

#### **Section 10 - Missing Pollutants Analysis (NO & NH3)**
- **NO (Nitric Oxide)**:
  - **Range**: 0.00-1.39 µg/m³, **Mean**: 0.08 µg/m³
  - **46% zeros**: Sparse data, mostly background levels
  - **Correlations**: PM2.5 (0.128), PM10 (0.207) - **WEAK**
  - **Assessment**: **MARGINAL** - weak correlations, sparse data
- **NH3 (Ammonia)**:
  - **Range**: 2.83-60.86 µg/m³, **Mean**: 14.85 µg/m³
  - **No zeros**: Consistent data quality
  - **Correlations**: PM2.5 (0.369), PM10 (0.132)
  - **Assessment**: **VALUABLE** - 4th best PM2.5 predictor, captures agricultural emissions
- **Key Insight**: **NH3 adds unique agricultural pollution information** that other pollutants don't capture

#### **Section 11 - Wind Direction Analysis**
- **CRITICAL DISCOVERY**: Wind direction is **RAW DATA** from OpenWeather API, not engineered
- **Wind Direction Distribution**:
  - **Dominant directions**: SW (260), N (205), S (180), SE (115), NE (85)
  - **Weak directions**: E, W, and intercardinal directions
  - **Highly skewed**: Not uniform distribution
- **Wind Direction vs PM Correlation**:
  - **PM2.5**: SSE (55+ µg/m³), ENE (50 µg/m³), NE (45 µg/m³), SE (45 µg/m³)
  - **PM10**: ENE (115 µg/m³), SE (110 µg/m³), ESE (110 µg/m³), SSW (110 µg/m³)
  - **Range**: 25-30 µg/m³ difference between highest and lowest directions
- **Key Insights**:
  - **Wind direction is a MASSIVE predictor**: 25-30 µg/m³ difference
  - **Pollution source identification**: High PM from east/southeast directions
  - **Cyclical nature**: Requires sine/cosine transformation
  - **Missing engineering**: Current wind_direction is raw 0-360° data

#### **Section 12 - Interaction Features Analysis**
- **5 Interaction Features**: temp_humidity, temp_wind, pm2_5_temp, pm2_5_humidity, wind_pm2_5
- **❌ INCORRECT ASSESSMENT**: pm2_5_temp_interaction, pm2_5_humidity_interaction were labeled as "data leakage"
- **Performance Analysis**:
  - **4/10 interactions improve** over individual features (40%)
  - **Valuable interactions**: temp_humidity (+0.144), temp_wind (+0.011)
  - **❌ WRONG CONCLUSION**: PM2.5 interactions were incorrectly labeled as problematic
- **Engineering Recommendation**: **❌ INCORRECT - Section 16 proves this wrong**

#### **Section 16 - PM × Weather Interactions → Future PM Analysis (NEW)**
- **CRITICAL DISCOVERY**: Current PM × weather interactions are **HIGHLY PREDICTIVE** for future PM
- **Tested Approach**: Current PM × Current Weather → Future PM (1h, 6h, 12h, 24h, 48h, 72h)
- **Key Findings**:
  - **PM2.5 × Pressure**: 0.977 correlation at 1h, 0.647 at 24h - **🔥 STRONG**
  - **PM2.5 × Temperature**: 0.926 correlation at 1h, 0.603 at 24h - **🔥 STRONG**
  - **PM2.5 × Humidity**: 0.755 correlation at 1h, 0.665 at 12h - **🔥 STRONG**
  - **PM10 × Pressure**: 0.809 correlation at 1h, 0.450 at 48h - **🔥 STRONG**
  - **PM10 × Temperature**: 0.712 correlation at 1h, 0.344 at 72h - **🟡 MODERATE**
  - **PM10 × Humidity**: 0.647 correlation at 1h, 0.320 at 72h - **🟡 MODERATE**
- **Atmospheric Interpretation**: These interactions capture how current pollution levels under specific weather conditions predict future pollution evolution
- **Engineering Recommendation**: **INCLUDE ALL 6 PM × WEATHER INTERACTIONS** - they're the most predictive features discovered

#### **Section 13 - Binary Indicators Analysis**
- **23 Binary Features**: Weather (7), Time (7), Pollution (9)
- **Data Quality Issues**:
  - **Constant features**: is_summer (100%), is_low_pressure (100%), is_high_pm10 (94.9%)
  - **Zero variance**: is_cold, is_high_wind, is_high_pressure, seasonal features (0%)
  - **Threshold problems**: PM10 threshold too low, pressure thresholds wrong
- **Valuable Features**:
  - **is_hot** (40.4%), **is_high_pm2_5** (46.7%), **is_night** (37.5%)
  - **is_morning_rush** (12.5%), **is_evening_rush** (12.5%), **is_high_o3** (14.8%)
- **Engineering Recommendation**: **Keep 6, remove 11** binary features

#### **Section 14 - Comprehensive Feature Ranking**
- **LOGICALLY PROBLEMATIC**: Data leakage dominates rankings
- **Top features use target variable**: pm2_5_aqi, us_aqi, pm2_5_squared, pm2_5_temp_interaction
- **Issues**:
  - **Correlation ≠ predictive power**: Wrong metric for feature selection
  - **No multicollinearity analysis**: Ignores feature redundancy
  - **No stability analysis**: Single-point correlation calculation
- **Engineering Recommendation**: **Skip this section's rankings**, use EDA-based strategy

#### **Section 15 - Secondary Pollutants → Future PM Analysis**
- **CRITICAL GAP ADDRESSED**: How current pollutants predict future PM values
- **Multi-Horizon Analysis**: 1h, 6h, 12h, 24h, 48h, 72h ahead
- **Key Findings**:
  - **Carbon Monoxide**: Best predictor (0.496 at 1h, excellent early warning)
  - **Ozone**: Strong predictor (chemical precursor for PM formation)
  - **Sulphur Dioxide**: Moderate predictor
  - **NH3**: Moderate predictor (agricultural emissions)
  - **NO**: Weak predictor (sparse data, 46% zeros)
- **Pollutant Rankings**:
  - **TIER 1 (Strong)**: carbon_monoxide, ozone
  - **TIER 2 (Moderate)**: sulphur_dioxide, nh3
  - **TIER 3 (Weak)**: nitrogen_dioxide, no
- **Weather vs Pollutants Comparison**:
  - **Pollutants**: Chemical precursor information
  - **Weather**: Atmospheric dispersion information
  - **Combined approach**: Optimal for 72-hour forecasting

---

## 🏗️ **ENGINEERING ARCHITECTURE**

### **1. Data Pipeline Architecture**
```
Raw Data → Preprocessing → Feature Engineering → Model Training → Prediction → AQI Calculation
```

### **2. Model Strategy: Multi-Horizon Approach**
```python
# Train separate models for key horizons
horizons = [1, 6, 12, 24, 48, 72]  # Hours ahead

# For each horizon, train:
# - PM2.5 model
# - PM10 model

# Interpolate intermediate hours (2,3,4,5,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71)
```

---

## 🧹 **DATA PREPROCESSING STRATEGY**

### **1. Outlier Handling**
```python
# PM2.5: Remove outliers (sensor errors)
pm2_5_clean = pm2_5[pm2_5 <= 78.4]

# PM10: Keep outliers (real pollution events)
pm10_clean = pm10  # Keep all data

# Handle zero values (0.1% missing)
pm2_5_clean = pm2_5_clean.replace(0, np.nan).fillna(method='ffill')
pm10_clean = pm10_clean.replace(0, np.nan).fillna(method='ffill')
```

### **2. Feature Transformations**
```python
# PM2.5: Log transformation + Standard scaling
pm2_5_transformed = np.log1p(pm2_5_clean)
pm2_5_scaled = StandardScaler().fit_transform(pm2_5_transformed)

# PM10: Robust scaling (no log transformation)
pm10_scaled = RobustScaler().fit_transform(pm10_clean)

# Weather features: Standard scaling
weather_scaled = StandardScaler().fit_transform(weather_features)

# Cyclical features: Already normalized [-1, 1]
# Binary features: Already 0/1
```

### **3. CO-NO2 Redundancy Handling**
```python
# Option 1: Keep CO, drop NO2 (CO has higher correlation with PM2.5/PM10)
# Option 2: PCA on CO-NO2 pair
# Option 3: Feature selection during model training

# RECOMMENDATION: Drop NO2, keep CO
features_to_drop = ['nitrogen_dioxide']
```

---

## 🔧 **FEATURE ENGINEERING STRATEGY**

### **1. Lag Features (Based on Section 8.2)**
```python
# PM2.5: Use 1h, 2h, 3h lags only (6h+ are useless)
pm2_5_lags = ['pm2_5_lag_1h', 'pm2_5_lag_2h', 'pm2_5_lag_3h']

# PM10: Use 1h, 2h, 3h lags only (6h+ are useless)  
pm10_lags = ['pm10_lag_1h', 'pm10_lag_2h', 'pm10_lag_3h']

# REASONING: Section 8.2 shows 6h+ lags have 0% utility
# Section 5 provided basic persistence analysis, Section 8.2 provided forecasting optimization
```

### **2. Rolling Statistics (Based on Section 9)**
```python
# KEEP ONLY 7 VALUABLE ROLLING FEATURES (78% reduction)
# PM2.5 Rolling Features (4 features)
pm25_rolling_keep = [
    'pm2_5_rolling_min_3h',    # Best overall (0.944 → 0.451)
    'pm2_5_rolling_mean_3h',   # Strong alternative (0.974 → 0.445)
    'pm2_5_rolling_min_12h',   # Long-term (0.699)
    'pm2_5_rolling_mean_12h'   # Long-term alternative
]

# PM10 Rolling Features (3 features)
pm10_rolling_keep = [
    'pm10_rolling_min_3h',     # Best overall (0.879 → 0.488)
    'pm10_rolling_mean_3h',    # Strong alternative
    'pm10_rolling_mean_24h'    # Long-term (0.320)
]

# REMOVE ALL STD FEATURES (useless across all horizons)
# REMOVE 6h+ MAX FEATURES (poor performance)
# REMOVE 12h+ MEAN FEATURES for PM2.5 (3h dominates)
```

### **3. Wind Direction Engineering (CRITICAL - Section 11)**
```python
# MUST-DO FEATURE ENGINEERING (currently missing)
# Cyclical wind direction features
df['wind_direction_sin'] = np.sin(np.radians(df['wind_direction']))
df['wind_direction_cos'] = np.cos(np.radians(df['wind_direction']))

# Wind vector components (more physically meaningful)
df['wind_speed_x'] = df['wind_speed'] * df['wind_direction_sin']
df['wind_speed_y'] = df['wind_speed'] * df['wind_direction_cos']

# Pollution source indicators (HIGH VALUE)
df['is_wind_from_high_pm'] = (
    (df['wind_direction_sin'] > 0.5) |  # SSE, SE, ESE
    (df['wind_direction_sin'] > 0.3) & (df['wind_direction_cos'] > 0.3)  # ENE, NE
).astype(int)

df['is_wind_from_low_pm'] = (
    (df['wind_direction_sin'] < -0.5) |  # SSW, SW
    (df['wind_direction_cos'] < -0.5)    # W, NNW
).astype(int)

# Wind direction categories
def categorize_wind_direction(row):
    sin_val = row['wind_direction_sin']
    cos_val = row['wind_direction_cos']
    
    if sin_val > 0.5:  # SSE, SE, ESE
        return 'high_pm_south_east'
    elif sin_val > 0.3 and cos_val > 0.3:  # ENE, NE
        return 'high_pm_north_east'
    elif sin_val < -0.5:  # SSW, SW
        return 'low_pm_south_west'
    elif cos_val < -0.5:  # W, NNW
        return 'low_pm_west'
    else:
        return 'medium_pm'

df['wind_direction_category'] = df.apply(categorize_wind_direction, axis=1)

# REASONING: 25-30 µg/m³ difference based on wind direction
# This is the MOST IMPORTANT missing feature
```

### **4. Weather Features (Based on Section 8.3)**
```python
# Current weather (t): All weather parameters
current_weather = ['temperature', 'humidity', 'wind_speed', 'pressure']

# Future weather forecasts (t+N): Essential for long horizons
# NEED TO INTEGRATE EXTERNAL WEATHER API
future_weather = [
    'temperature_t_plus_1h', 'temperature_t_plus_6h', 'temperature_t_plus_12h', 
    'temperature_t_plus_24h', 'temperature_t_plus_48h', 'temperature_t_plus_72h',
    'pressure_t_plus_1h', 'pressure_t_plus_6h', 'pressure_t_plus_12h',
    'pressure_t_plus_24h', 'pressure_t_plus_48h', 'pressure_t_plus_72h',
    # ... similar for humidity, wind_speed
]

# REASONING: Section 8.3 shows weather has predictive power for future PM
```

### **5. Time Features (Keep existing)**
```python
# Cyclical encoding already implemented
time_features = [
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
    'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos'
]
```

### **6. Change Rate Features (Keep existing)**
```python
# Keep all change rate features
change_features = [
    'pm2_5_change_rate_1h', 'pm2_5_change_rate_6h',
    'pm10_change_rate_1h', 'pm10_change_rate_6h',
    # ... all other change features
]
```

### **7. Interaction Features (Based on Section 16 - CORRECTED)**
```python
# KEEP ALL 6 PM × WEATHER INTERACTIONS (Section 16 proves they're highly predictive)
pm_weather_interactions = [
    'pm2_5_temperature_interaction',   # 0.926 correlation at 1h, 0.603 at 24h
    'pm2_5_humidity_interaction',      # 0.755 correlation at 1h, 0.665 at 12h
    'pm2_5_pressure_interaction',      # 0.977 correlation at 1h, 0.647 at 24h
    'pm10_temperature_interaction',    # 0.712 correlation at 1h, 0.344 at 72h
    'pm10_humidity_interaction',       # 0.647 correlation at 1h, 0.320 at 72h
    'pm10_pressure_interaction'        # 0.809 correlation at 1h, 0.450 at 48h
]

# KEEP 2 WEATHER-WEATHER INTERACTIONS (from Section 12)
weather_interactions = [
    'temp_humidity_interaction',  # +0.144 improvement for PM2.5
    'temp_wind_interaction'       # +0.011 improvement for PM2.5
]

# NEW HIGH-VALUE INTERACTIONS (based on wind direction findings)
wind_interactions = [
    'wind_direction_temp_interaction',
    'wind_direction_humidity_interaction',
    'pressure_humidity_interaction'
]

# TOTAL: 11 interaction features (6 PM×weather + 2 weather×weather + 3 wind×weather)
# Section 16 proves PM×weather interactions are the MOST predictive features discovered
```

### **8. Binary Indicators (Based on Section 13)**
```python
# KEEP ONLY 6 VALUABLE BINARY FEATURES
useful_binary = [
    'is_hot',              # 40.4% - temperature extremes
    'is_high_pm2_5',       # 46.7% - PM2.5 spikes
    'is_night',            # 37.5% - diurnal patterns
    'is_morning_rush',     # 12.5% - traffic patterns
    'is_evening_rush',     # 12.5% - traffic patterns
    'is_high_o3'           # 14.8% - ozone spikes
]

# REMOVE 11 PROBLEMATIC BINARY FEATURES
remove_binary = [
    'is_summer',           # 100% - no variance
    'is_low_pressure',     # 100% - no variance
    'is_high_pm10',        # 94.9% - threshold too low
    'is_cold',             # 0% - no variance
    'is_high_wind',        # 0% - no variance
    'is_high_pressure',    # 0% - no variance
    'is_spring',           # 0% - no variance
    'is_autumn',           # 0% - no variance
    'is_winter',           # 0% - no variance
    'is_high_no2',         # 0% - no variance
    'is_high_co',          # 0% - no variance
    'is_high_so2'          # 0% - no variance
]

# FIX THRESHOLDS
df['is_high_pm10'] = (df['pm10'] > 154).astype(int)  # EPA "Unhealthy"
df['is_low_pressure'] = (df['pressure'] < 1010).astype(int)  # Atmospheric
df['is_high_pressure'] = (df['pressure'] > 1020).astype(int)  # Atmospheric
df['is_high_wind'] = (df['wind_speed'] > 20).astype(int)  # High wind
```

### **9. Pollutant Features (Based on Sections 3, 10, 15)**
```python
# HIGH-VALUE POLLUTANTS (TIER 1 & 2)
high_value_pollutants = [
    'carbon_monoxide',     # 0.496 correlation at 1h (Section 15)
    'ozone',              # Strong chemical precursor (Section 15)
    'sulphur_dioxide',    # Moderate predictor (Section 15)
    'nh3'                 # Agricultural emissions (Section 10)
]

# REMOVE POLLUTANTS
remove_pollutants = [
    'nitrogen_dioxide'     # High correlation with CO (0.69), lower predictive power
    'no'                   # Weak correlations, 46% zeros, sparse data
]

# POLLUTANT LAG FEATURES (for future prediction)
pollutant_lags = [
    'co_lag_1h',          # Carbon monoxide 1h ago
    'o3_lag_1h',          # Ozone 1h ago
    'so2_lag_1h'          # Sulphur dioxide 1h ago
]

# POLLUTANT-WEATHER INTERACTIONS
pollutant_weather_interactions = [
    'co_pressure_interaction',    # CO × Pressure
    'o3_temp_interaction',        # O3 × Temperature
    'so2_humidity_interaction'    # SO2 × Humidity
]
```

---

## 🤖 **COMPREHENSIVE MODEL STRATEGY**

### **1. Model Selection & Architecture**

#### **Phase 1: Traditional ML Models (Day 1 Forecasting - 1-24h)**
```python
# Simple, interpretable models for short-term forecasting
traditional_models = {
    'RandomForest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    },
    'XGBoost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    'LinearRegression': {
        'fit_intercept': True
    },
    'RidgeRegression': {
        'alpha': 1.0,
        'fit_intercept': True
    },
    'SVR': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    }
}

# Features for traditional models (all engineered features)
traditional_features = [
    # Current state
    'pm2_5', 'pm10', 'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3',
    
    # Lag features (1h, 2h, 3h only)
    'pm2_5_lag_1h', 'pm2_5_lag_2h', 'pm2_5_lag_3h',
    'pm10_lag_1h', 'pm10_lag_2h', 'pm10_lag_3h',
    
    # Pollutant lags
    'co_lag_1h', 'o3_lag_1h', 'so2_lag_1h',
    
    # Current weather
    'temperature', 'humidity', 'pressure', 'wind_speed',
    
    # Wind direction (NEW - cyclical encoding)
    'wind_direction_sin', 'wind_direction_cos',
    'is_wind_from_high_pm', 'is_wind_from_low_pm',
    'wind_direction_category',
    
    # Rolling features (7 features only)
    'pm2_5_rolling_min_3h', 'pm2_5_rolling_mean_3h', 'pm2_5_rolling_min_12h', 'pm2_5_rolling_mean_12h',
    'pm10_rolling_min_3h', 'pm10_rolling_mean_3h', 'pm10_rolling_mean_24h',
    
    # Change rates
    'pm2_5_change_rate_1h', 'pm2_5_change_rate_6h', 'pm2_5_change_rate_24h',
    'pm10_change_rate_1h', 'pm10_change_rate_6h', 'pm10_change_rate_24h',
    
    # Time features (cyclical)
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
    'day_of_week_sin', 'day_of_week_cos',
    
    # Binary features (6 features only)
    'is_hot', 'is_high_pm2_5', 'is_night', 'is_morning_rush', 'is_evening_rush', 'is_high_o3',
    
    # Interaction features (11 total - 6 PM×weather + 2 weather×weather + 3 wind×weather)
    'pm2_5_temperature_interaction', 'pm2_5_humidity_interaction', 'pm2_5_pressure_interaction',
    'pm10_temperature_interaction', 'pm10_humidity_interaction', 'pm10_pressure_interaction',
    'temp_humidity_interaction', 'temp_wind_interaction',
    'wind_direction_temp_interaction', 'wind_direction_humidity_interaction', 'pressure_humidity_interaction',
    
    # Pollutant-weather interactions
    'co_pressure_interaction', 'o3_temp_interaction', 'so2_humidity_interaction'
]

# Training approach: Multi-horizon for short-term
for horizon in [1, 6, 12, 24]:
    target_pm2_5 = df['pm2_5'].shift(-horizon)
    target_pm10 = df['pm10'].shift(-horizon)
    
    # Train separate models for each horizon
    for model_name, model_params in traditional_models.items():
        model_pm2_5 = train_model(traditional_features, target_pm2_5, model_name, model_params)
        model_pm10 = train_model(traditional_features, target_pm10, model_name, model_params)
```

#### **Phase 2: Deep Learning Models (Day 2-3 Forecasting - 48-72h)**
```python
# Advanced models for long-term forecasting with weather forecasts
deep_learning_models = {
    'LSTM': {
        'units': 128,
        'layers': 2,
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping': True
    },
    'GRU': {
        'units': 128,
        'layers': 2,
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping': True
    },
    'Transformer': {
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'dropout': 0.1,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping': True
    },
    'CNN-LSTM_Hybrid': {
        'cnn_filters': 64,
        'cnn_kernel_size': 3,
        'lstm_units': 128,
        'dropout': 0.2,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping': True
    }
}

# Features for deep learning (sequence-based)
deep_learning_features = [
    # Sequence features (last 24 hours)
    'pm2_5_sequence_24h', 'pm10_sequence_24h',
    'temperature_sequence_24h', 'pressure_sequence_24h',
    'wind_speed_sequence_24h', 'wind_direction_sequence_24h',
    
    # Current state (same as traditional)
    'pm2_5', 'pm10', 'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nh3',
    'temperature', 'humidity', 'pressure', 'wind_speed',
    'wind_direction_sin', 'wind_direction_cos',
    
    # Future weather forecasts (CRITICAL for 48-72h)
    'temperature_forecast_48h', 'temperature_forecast_72h',
    'pressure_forecast_48h', 'pressure_forecast_72h',
    'humidity_forecast_48h', 'humidity_forecast_72h',
    'wind_speed_forecast_48h', 'wind_speed_forecast_72h',
    'wind_direction_forecast_48h', 'wind_direction_forecast_72h',
    
    # Time features
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
    'day_of_week_sin', 'day_of_week_cos',
    
    # Binary features
    'is_hot', 'is_high_pm2_5', 'is_night', 'is_morning_rush', 'is_evening_rush', 'is_high_o3'
]

# Training approach: Sequence-based for long-term
for horizon in [48, 72]:
    target_pm2_5 = df['pm2_5'].shift(-horizon)
    target_pm10 = df['pm10'].shift(-horizon)
    
    # Create sequences for deep learning
    sequences = create_sequences(df, sequence_length=24, target_horizon=horizon)
    
    for model_name, model_params in deep_learning_models.items():
        model_pm2_5 = train_sequence_model(sequences, target_pm2_5, model_name, model_params)
        model_pm10 = train_sequence_model(sequences, target_pm10, model_name, model_params)
```

#### **Phase 3: Ensemble Models (Production)**
```python
# Combine best performers from each phase
ensemble_models = {
    'StackingEnsemble': {
        'base_models': ['RandomForest', 'XGBoost', 'LSTM', 'GRU'],
        'meta_model': 'LinearRegression',
        'cv_folds': 5
    },
    'VotingEnsemble': {
        'models': ['RandomForest', 'XGBoost', 'LSTM'],
        'weights': [0.3, 0.3, 0.4],
        'voting': 'soft'
    },
    'WeightedAverage': {
        'models': ['RandomForest_1h', 'XGBoost_6h', 'LSTM_24h', 'GRU_48h', 'Transformer_72h'],
        'weights': [0.25, 0.25, 0.2, 0.15, 0.15]
    }
}
```

### **2. Multi-Horizon Training Strategy**

#### **Key Insight: Weather Forecasts Enable 72-Hour Prediction**
```python
# Section 8.3 proved: Current weather correlates with future PM
# This means: Weather forecasts can predict future PM!

# Training data structure for each horizon:
training_data = {
    'features': {
        'current_state': current_pollutants + current_weather + wind_direction,
        'historical_context': lag_features + rolling_features + change_rates,
        'future_weather': weather_forecasts[horizon],  # CRITICAL for 48h+
        'time_features': cyclical_time_encoding,
        'interactions': pollutant_weather_interactions
    },
    'target': f'pm2_5_future_{horizon}h'  # PM2.5 at t+horizon
}

# Example: Predicting PM2.5 at t+72h
features_72h = [
    # Current state (t)
    'pm2_5', 'pm10', 'co', 'o3', 'so2', 'nh3',
    'temperature', 'pressure', 'wind_speed', 'wind_direction_sin',
    
    # Historical context (t-1h to t-24h)
    'pm2_5_lag_1h', 'pm2_5_lag_2h', 'pm2_5_lag_3h',
    'pm2_5_rolling_mean_3h', 'pm2_5_rolling_min_3h',
    
    # Future weather forecast (t+72h) - CRITICAL!
    'temperature_forecast_72h', 'pressure_forecast_72h',
    'wind_speed_forecast_72h', 'wind_direction_forecast_72h',
    
    # Time features (t+72h)
    'hour_sin_72h', 'day_sin_72h', 'month_sin_72h',
    
    # Interactions
    'co_pressure_interaction', 'o3_temp_interaction'
]

target_72h = 'pm2_5_future_72h'  # PM2.5 at t+72h
```

#### **Horizon-Specific Feature Engineering**
```python
# Short-term horizons (1-24h): Emphasize current state + recent history
short_term_features = [
    'current_pollutants', 'current_weather', 'wind_direction',
    'lag_features_1h_3h', 'rolling_features_3h_6h',
    'change_rates', 'time_features'
]

# Medium-term horizons (24-48h): Balance current + forecast
medium_term_features = [
    'current_pollutants', 'current_weather', 'wind_direction',
    'lag_features_1h_6h', 'rolling_features_6h_12h',
    'weather_forecast_24h', 'time_features_24h'
]

# Long-term horizons (48-72h): Emphasize forecasts + patterns
long_term_features = [
    'current_pollutants', 'current_weather', 'wind_direction',
    'rolling_features_12h_24h', 'change_rates_24h',
    'weather_forecast_48h', 'weather_forecast_72h',
    'time_features_48h', 'time_features_72h',
    'seasonal_patterns', 'weekly_patterns'
]
```

### **3. Weather Forecast Integration Strategy**

#### **External Weather API Integration**
```python
# Integrate with OpenWeather API for future forecasts
def get_weather_forecast(lat, lon, horizon_hours):
    """
    Get weather forecast for specific horizon
    Returns: temperature, pressure, humidity, wind_speed, wind_direction
    """
    url = f"https://api.openweathermap.org/data/2.5/forecast"
    params = {
        'lat': lat, 'lon': lon, 'appid': api_key,
        'units': 'metric', 'cnt': horizon_hours
    }
    response = requests.get(url, params=params)
    return parse_weather_forecast(response.json(), horizon_hours)

# Create weather forecast features
weather_forecast_features = {}
for horizon in [1, 6, 12, 24, 48, 72]:
    forecast = get_weather_forecast(lat, lon, horizon)
    weather_forecast_features[f'temperature_forecast_{horizon}h'] = forecast['temperature']
    weather_forecast_features[f'pressure_forecast_{horizon}h'] = forecast['pressure']
    weather_forecast_features[f'humidity_forecast_{horizon}h'] = forecast['humidity']
    weather_forecast_features[f'wind_speed_forecast_{horizon}h'] = forecast['wind_speed']
    weather_forecast_features[f'wind_direction_forecast_{horizon}h'] = forecast['wind_direction']
```

#### **Wind Direction Forecast Processing**
```python
# Process wind direction forecasts (convert to cyclical features)
for horizon in [1, 6, 12, 24, 48, 72]:
    wind_dir = weather_forecast_features[f'wind_direction_forecast_{horizon}h']
    
    # Convert to cyclical features
    weather_forecast_features[f'wind_direction_sin_forecast_{horizon}h'] = np.sin(np.radians(wind_dir))
    weather_forecast_features[f'wind_direction_cos_forecast_{horizon}h'] = np.cos(np.radians(wind_dir))
    
    # Create pollution source indicators for forecast
    weather_forecast_features[f'is_wind_from_high_pm_forecast_{horizon}h'] = (
        (np.sin(np.radians(wind_dir)) > 0.5) |
        (np.sin(np.radians(wind_dir)) > 0.3) & (np.cos(np.radians(wind_dir)) > 0.3)
    ).astype(int)
```

### **4. Model Training Pipeline**

#### **Data Preparation**
```python
def prepare_training_data(df, horizon):
    """
    Prepare training data for specific horizon
    """
    # Create target
    target_pm2_5 = df['pm2_5'].shift(-horizon)
    target_pm10 = df['pm10'].shift(-horizon)
    
    # Create features based on horizon
    if horizon <= 24:
        features = create_short_term_features(df)
    elif horizon <= 48:
        features = create_medium_term_features(df, horizon)
    else:
        features = create_long_term_features(df, horizon)
    
    # Add weather forecasts for this horizon
    weather_forecast = get_weather_forecast(lat, lon, horizon)
    features.update(weather_forecast)
    
    # Remove rows with NaN targets (end of dataset)
    valid_indices = ~(target_pm2_5.isna() | target_pm10.isna())
    
    return features[valid_indices], target_pm2_5[valid_indices], target_pm10[valid_indices]
```

#### **Model Training Loop**
```python
# Train models for each horizon
trained_models = {}

for horizon in [1, 6, 12, 24, 48, 72]:
    print(f"Training models for {horizon}h horizon...")
    
    # Prepare data
    X, y_pm2_5, y_pm10 = prepare_training_data(df, horizon)
    
    # Split data
    X_train, X_test, y_pm2_5_train, y_pm2_5_test = train_test_split(X, y_pm2_5, test_size=0.2, random_state=42)
    _, _, y_pm10_train, y_pm10_test = train_test_split(X, y_pm10, test_size=0.2, random_state=42)
    
    # Train models based on horizon
    if horizon <= 24:
        # Traditional ML models
        for model_name, model_params in traditional_models.items():
            model_pm2_5 = train_traditional_model(X_train, y_pm2_5_train, model_name, model_params)
            model_pm10 = train_traditional_model(X_train, y_pm10_train, model_name, model_params)
            
            # Evaluate
            score_pm2_5 = model_pm2_5.score(X_test, y_pm2_5_test)
            score_pm10 = model_pm10.score(X_test, y_pm10_test)
            
            trained_models[f'{model_name}_pm2_5_{horizon}h'] = model_pm2_5
            trained_models[f'{model_name}_pm10_{horizon}h'] = model_pm10
            
    else:
        # Deep learning models
        for model_name, model_params in deep_learning_models.items():
            model_pm2_5 = train_deep_learning_model(X_train, y_pm2_5_train, model_name, model_params)
            model_pm10 = train_deep_learning_model(X_train, y_pm10_train, model_name, model_params)
            
            trained_models[f'{model_name}_pm2_5_{horizon}h'] = model_pm2_5
            trained_models[f'{model_name}_pm10_{horizon}h'] = model_pm10
```

### **5. Prediction Pipeline**

#### **72-Hour Prediction Process**
```python
def predict_72_hour_aqi(current_data, weather_forecasts):
    """
    Predict AQI for next 72 hours
    """
    predictions = {}
    
    # Key horizons where we have trained models
    key_horizons = [1, 6, 12, 24, 48, 72]
    
    for horizon in key_horizons:
        # Prepare features for this horizon
        features = prepare_prediction_features(current_data, weather_forecasts, horizon)
        
        # Get best model for this horizon
        best_model_pm2_5 = get_best_model(f'trained_models_pm2_5_{horizon}h')
        best_model_pm10 = get_best_model(f'trained_models_pm10_{horizon}h')
        
        # Make predictions
        pm2_5_pred = best_model_pm2_5.predict(features)
        pm10_pred = best_model_pm10.predict(features)
        
        predictions[f'pm2_5_{horizon}h'] = pm2_5_pred
        predictions[f'pm10_{horizon}h'] = pm10_pred
    
    # Interpolate intermediate hours (2,3,4,5,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71)
    interpolated_predictions = interpolate_hourly_predictions(predictions)
    
    # Calculate AQI for each hour
    aqi_predictions = {}
    for hour in range(1, 73):
        pm2_5 = interpolated_predictions[f'pm2_5_{hour}h']
        pm10 = interpolated_predictions[f'pm10_{hour}h']
        aqi = calculate_aqi_from_pm(pm2_5, pm10)
        aqi_predictions[f'aqi_{hour}h'] = aqi
    
    return aqi_predictions
```

#### **Interpolation Strategy**
```python
def interpolate_hourly_predictions(key_predictions):
    """
    Interpolate between key horizons to get all 72 hours
    """
    interpolated = {}
    
    # Key horizons: [1, 6, 12, 24, 48, 72]
    key_hours = [1, 6, 12, 24, 48, 72]
    
    for hour in range(1, 73):
        if hour in key_hours:
            # Use direct prediction
            interpolated[f'pm2_5_{hour}h'] = key_predictions[f'pm2_5_{hour}h']
            interpolated[f'pm10_{hour}h'] = key_predictions[f'pm10_{hour}h']
        else:
            # Interpolate between nearest key horizons
            lower_hour = max([h for h in key_hours if h < hour])
            upper_hour = min([h for h in key_hours if h > hour])
            
            # Linear interpolation
            weight = (hour - lower_hour) / (upper_hour - lower_hour)
            
            pm2_5_interp = (key_predictions[f'pm2_5_{lower_hour}h'] * (1 - weight) + 
                           key_predictions[f'pm2_5_{upper_hour}h'] * weight)
            pm10_interp = (key_predictions[f'pm10_{lower_hour}h'] * (1 - weight) + 
                          key_predictions[f'pm10_{upper_hour}h'] * weight)
            
            interpolated[f'pm2_5_{hour}h'] = pm2_5_interp
            interpolated[f'pm10_{hour}h'] = pm10_interp
    
    return interpolated
```

### **3. Model Evaluation Metrics**
```python
# Primary Metrics (Based on Section 8.4)
success_criteria = {
    'PM2.5_RMSE': {
        'Day_1': 15,    # µg/m³
        'Day_2': 25,    # µg/m³  
        'Day_3': 35     # µg/m³
    },
    'PM10_RMSE': {
        'Day_1': 25,    # µg/m³
        'Day_2': 40,    # µg/m³
        'Day_3': 55     # µg/m³
    },
    'AQI_Category_Accuracy': {
        'Day_1': 0.85,  # 85%
        'Day_2': 0.70,  # 70%
        'Day_3': 0.60   # 60%
    }
}
```

---

## 📈 **IMPLEMENTATION PHASES**

### **Phase 1: Foundation (Week 1-2)**
1. **Data Preprocessing Pipeline**
   - Implement outlier removal for PM2.5
   - Implement log transformation for PM2.5
   - Handle CO-NO2 redundancy (drop NO2)
   - Fix binary feature thresholds
   - Create train/test split

2. **Critical Feature Engineering**
   - **WIND DIRECTION ENGINEERING** (highest priority)
   - Implement lag features (1h, 2h, 3h only)
   - Reduce rolling features (32 → 7)
   - Keep only 2 useful interactions
   - Keep only 6 useful binary features
   - Add NH3 to pollutant features

3. **Baseline Models**
   - Train Random Forest and XGBoost
   - Focus on Day 1 forecasting (1h, 6h, 12h, 24h)
   - Establish baseline performance

### **Phase 2: Advanced Models (Week 3-4)**
1. **Deep Learning Models**
   - Implement LSTM/GRU models
   - Train on multi-horizon targets
   - Optimize hyperparameters

2. **Weather Integration**
   - Integrate external weather API
   - Add future weather forecast features
   - Test impact on Day 2-3 forecasting

3. **Model Comparison**
   - Compare all models across horizons
   - Select best performers for each horizon

### **Phase 3: Production (Week 5-6)**
1. **Ensemble Development**
   - Create stacking/voting ensembles
   - Optimize ensemble weights
   - Final model selection

2. **Interpolation System**
   - Implement hourly interpolation between key horizons
   - Validate interpolation accuracy
   - Create complete 72-hour prediction pipeline

3. **AQI Calculation Integration**
   - Integrate AQI calculation from PM predictions
   - Validate AQI category accuracy
   - Final system testing

---

## 🎯 **CRITICAL SUCCESS FACTORS**

### **1. Wind Direction Engineering (HIGHEST PRIORITY)**
```python
# MUST HAVE: Proper wind direction feature engineering
# Current wind_direction is raw 0-360° data - needs cyclical encoding
# 25-30 µg/m³ difference based on wind direction = massive signal
# This is the most important missing feature
```

### **2. Feature Selection Discipline**
```python
# MUST NOT: Include 6h+ lag features (Section 8.2 shows 0% utility)
# MUST: Use only 1h, 2h, 3h lags for both PM2.5 and PM10
# MUST: Handle CO-NO2 redundancy (drop NO2)
# MUST: Reduce rolling features (32 → 7)
# MUST: Include PM × weather interactions (Section 16 proves they're highly predictive)
# MUST: Fix binary feature thresholds
```

### **3. Different Preprocessing for Each Pollutant**
```python
# PM2.5: Log transformation + Remove outliers
# PM10: No transformation + Keep outliers
# Weather: Standard scaling
# Time: Already normalized
```

### **4. Multi-Horizon Training**
```python
# Train separate models for each horizon
# Don't try to predict all 72 hours with one model
# Use interpolation for intermediate hours
```

### **5. Weather Forecast Integration**
```python
# MUST HAVE: External weather API for future forecasts
# Current weather alone is insufficient for 48h+ predictions
# Section 8.3 proves weather has predictive power
```

---

## 🚨 **POTENTIAL FAILURE POINTS**

### **1. Ignoring Wind Direction Engineering**
- **CRITICAL**: Wind direction is currently raw data, not engineered
- **Impact**: Missing 25-30 µg/m³ predictive signal
- **Solution**: Implement cyclical encoding + pollution source indicators

### **2. Ignoring Section 8.2 Findings**
- Including 6h+ lag features will add noise
- Will reduce model performance significantly
- **Section 5 vs 8.2**: Section 5 showed basic persistence, Section 8.2 showed forecasting optimization

### **3. Wrong Outlier Strategy**
- Removing PM10 outliers = losing real pollution events
- Keeping PM2.5 outliers = training on sensor errors

### **4. No Weather Forecast Integration**
- Day 2-3 predictions will fail without future weather
- Section 8.3 proves weather is crucial

### **5. Single Model for All Horizons**
- 72-hour prediction with one model = poor performance
- Must use multi-horizon approach

### **6. PM × Weather Interactions (Section 16 Discovery)**
- **CRITICAL**: PM × weather interactions are HIGHLY PREDICTIVE for future PM
- **0.977 correlation** for PM2.5 × pressure at 1h horizon
- **0.926 correlation** for PM2.5 × temperature at 1h horizon
- These are the **MOST PREDICTIVE FEATURES** discovered in the entire analysis

### **7. Broken Binary Feature Thresholds**
- 94.9% "high PM10" = threshold too low
- 100% "low pressure" = threshold wrong
- Will add noise to model

---

## 📋 **IMMEDIATE NEXT STEPS**

1. **Implement wind direction engineering** (highest priority - missing 25-30 µg/m³ signal)
2. **Implement data preprocessing pipeline** (PM2.5 outlier removal, log transformation)
3. **Handle CO-NO2 redundancy** (drop NO2, keep CO)
4. **Reduce rolling features** (32 → 7 features)
5. **Fix binary feature thresholds** (PM10, pressure, wind thresholds)
6. **Include PM × weather interactions** (Section 16 proves they're highly predictive)
7. **Add NH3 to pollutant features** (agricultural emissions)
8. **Create lag features** (1h, 2h, 3h only)
9. **Train baseline models** (Random Forest, XGBoost) for Day 1 forecasting
10. **Integrate weather API** for future forecasts
11. **Implement multi-horizon training** approach

---

## 🎯 **BOTTOM LINE**

**This project is HIGHLY FEASIBLE** based on comprehensive EDA findings. The data quality is excellent, temporal patterns are strong, and we've identified critical missing features (especially wind direction engineering).

**Key Innovations from Comprehensive EDA:**
- **Wind Direction Engineering**: 25-30 µg/m³ predictive signal (currently missing)
- **Rolling Feature Optimization**: 78% reduction (32 → 7 features)
- **Binary Feature Cleanup**: Remove 11 broken features, keep 6 valuable ones
- **Interaction Feature Selection**: Remove data leakage, keep 2 valuable interactions
- **Pollutant Feature Enhancement**: Add NH3, remove NO, handle CO-NO2 redundancy
- **Secondary Pollutants → Future PM**: Carbon monoxide and ozone are strong predictors

**Success depends on disciplined feature selection, proper wind direction engineering, and following the engineering insights from each section.**

**The comprehensive EDA has revealed that wind direction engineering is the most critical missing piece for accurate PM prediction.** 