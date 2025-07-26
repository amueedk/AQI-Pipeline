# AQI Prediction Model - Feature Documentation

## Overview
This document explains all features used in the Air Quality Index (AQI) prediction model for Multan, Pakistan. The model predicts PM2.5 and PM10 concentrations using a combination of weather data, pollutant data, and engineered features.

## Data Sources
- **Weather Data**: OpenWeather API (temperature, humidity, pressure, wind)
- **Pollution Data**: OpenWeather Air Pollution API (PM2.5, PM10, NO2, O3, CO, SO2)
- **Location**: Multan, Pakistan (30.1575°N, 71.5249°E)

---

## 1. Raw Weather Features (Fetched from API)

### Temperature Features
| Feature | Source | Unit | Description |
|---------|--------|------|-------------|
| `temperature` | OpenWeather API | °C | Current air temperature |
| `temp_squared` | Calculated | °C² | Temperature squared (non-linear relationship) |
| `temp_cubed` | Calculated | °C³ | Temperature cubed (higher-order non-linearity) |
| `temp_change_rate` | Calculated | %/hour | Hourly temperature change rate |
| `is_hot` | Calculated | Binary | 1 if temperature > 35°C, 0 otherwise |
| `is_cold` | Calculated | Binary | 1 if temperature < 10°C, 0 otherwise |

### Humidity Features
| Feature | Source | Unit | Description |
|---------|--------|------|-------------|
| `humidity` | OpenWeather API | % | Relative humidity |
| `humidity_squared` | Calculated | %² | Humidity squared (non-linear relationship) |
| `is_high_humidity` | Calculated | Binary | 1 if humidity > 80%, 0 otherwise |
| `is_low_humidity` | Calculated | Binary | 1 if humidity < 30%, 0 otherwise |

### Wind Features
| Feature | Source | Unit | Description |
|---------|--------|------|-------------|
| `wind_speed` | OpenWeather API | m/s | Wind speed |
| `wind_direction` | OpenWeather API | degrees | Wind direction (0-360°) |
| `wind_speed_squared` | Calculated | (m/s)² | Wind speed squared |
| `is_high_wind` | Calculated | Binary | 1 if wind speed > 20 m/s, 0 otherwise |
| `is_calm` | Calculated | Binary | 1 if wind speed < 5 m/s, 0 otherwise |

### Pressure Features
| Feature | Source | Unit | Description |
|---------|--------|------|-------------|
| `pressure` | OpenWeather API | hPa | Atmospheric pressure |
| `pressure_change_rate` | Calculated | %/hour | Hourly pressure change rate |
| `is_low_pressure` | Calculated | Binary | 1 if pressure < 1010 hPa, 0 otherwise |
| `is_high_pressure` | Calculated | Binary | 1 if pressure > 1020 hPa, 0 otherwise |

---

## 2. Raw Pollutant Features (Fetched from API)

### Primary Pollutants (Target Variables)
| Feature | Source | Unit | Description |
|---------|--------|------|-------------|
| `pm2_5` | OpenWeather API | μg/m³ | Fine particulate matter (≤2.5μm) |
| `pm10` | OpenWeather API | μg/m³ | Coarse particulate matter (≤10μm) |

### Secondary Pollutants (Predictors)
| Feature | Source | Unit | Description |
|---------|--------|------|-------------|
| `carbon_monoxide` | OpenWeather API | μg/m³ | Carbon monoxide concentration |
| `nitrogen_dioxide` | OpenWeather API | μg/m³ | Nitrogen dioxide concentration |
| `ozone` | OpenWeather API | μg/m³ | Ozone concentration |
| `sulphur_dioxide` | OpenWeather API | μg/m³ | Sulphur dioxide concentration |
| `no` | OpenWeather API | μg/m³ | Nitric oxide concentration |
| `nh3` | OpenWeather API | μg/m³ | Ammonia concentration |

---

## 3. AQI Features (Calculated from Raw Concentrations)

### US EPA AQI Calculations
The model calculates AQI values using US EPA breakpoints:

#### PM2.5 AQI Breakpoints
| AQI Range | PM2.5 Range (μg/m³) | Category |
|-----------|---------------------|----------|
| 0-50 | 0.0-12.0 | Good |
| 51-100 | 12.1-35.4 | Moderate |
| 101-150 | 35.5-55.4 | Unhealthy for Sensitive Groups |
| 151-200 | 55.5-150.4 | Unhealthy |
| 201-300 | 150.5-250.4 | Very Unhealthy |
| 301-400 | 250.5-350.4 | Hazardous |
| 401-500 | 350.5-500.4 | Hazardous |

#### PM10 AQI Breakpoints
| AQI Range | PM10 Range (μg/m³) | Category |
|-----------|---------------------|----------|
| 0-50 | 0-54 | Good |
| 51-100 | 55-154 | Moderate |
| 101-150 | 155-254 | Unhealthy for Sensitive Groups |
| 151-200 | 255-354 | Unhealthy |
| 201-300 | 355-424 | Very Unhealthy |
| 301-400 | 425-504 | Hazardous |
| 401-500 | 505-604 | Hazardous |

### AQI Features
| Feature | Calculation | Description |
|---------|-------------|-------------|
| `pm2_5_aqi` | US EPA formula | PM2.5 AQI value (0-500) |
| `pm10_aqi` | US EPA formula | PM10 AQI value (0-500) |
| `us_aqi` | max(pm2_5_aqi, pm10_aqi) | Overall US AQI (worst pollutant) |
| `openweather_aqi` | OpenWeather API | OpenWeather's AQI (1-5 scale) |

---

## 4. Time-Based Features (Cyclic Encoding)

### Cyclic Time Features
Instead of raw time values, we use cyclic encoding for better ML performance:

| Feature | Calculation | Range | Description |
|---------|-------------|-------|-------------|
| `hour_sin` | sin(2π × hour / 24) | [-1, 1] | Hour of day (sine) |
| `hour_cos` | cos(2π × hour / 24) | [-1, 1] | Hour of day (cosine) |
| `day_sin` | sin(2π × day / 31) | [-1, 1] | Day of month (sine) |
| `day_cos` | cos(2π × day / 31) | [-1, 1] | Day of month (cosine) |
| `month_sin` | sin(2π × month / 12) | [-1, 1] | Month (sine) |
| `month_cos` | cos(2π × month / 12) | [-1, 1] | Month (cosine) |
| `day_of_week_sin` | sin(2π × day_of_week / 7) | [-1, 1] | Day of week (sine) |
| `day_of_week_cos` | cos(2π × day_of_week / 7) | [-1, 1] | Day of week (cosine) |

### Seasonal Features
| Feature | Calculation | Description |
|---------|-------------|-------------|
| `is_spring` | Binary | 1 if month ∈ [3,4,5], 0 otherwise |
| `is_summer` | Binary | 1 if month ∈ [6,7,8], 0 otherwise |
| `is_autumn` | Binary | 1 if month ∈ [9,10,11], 0 otherwise |
| `is_winter` | Binary | 1 if month ∈ [12,1,2], 0 otherwise |

### Time-of-Day Features
| Feature | Calculation | Description |
|---------|-------------|-------------|
| `is_night` | Binary | 1 if hour ∈ [22,23,0,1,2,3,4,5,6], 0 otherwise |
| `is_morning_rush` | Binary | 1 if hour ∈ [7,8,9], 0 otherwise |
| `is_evening_rush` | Binary | 1 if hour ∈ [17,18,19], 0 otherwise |

---

## 5. Lag Features (Time-Based)

### Lag Hours Used
- 1 hour, 2 hours, 3 hours, 6 hours, 12 hours, 24 hours, 48 hours, 72 hours

### Tolerance System
| Lag Period | Tolerance | Description |
|------------|-----------|-------------|
| 1 hour | ±30 minutes | Flexible 1-hour lag |
| 2 hours | ±45 minutes | Flexible 2-hour lag |
| 3 hours | ±50 minutes | Flexible 3-hour lag |
| 6+ hours | ±1 hour | Flexible 6+ hour lag |

### Lag Features for PM2.5 and PM10

#### Mathematical Formula
For each target variable (PM2.5, PM10) and lag period (h hours):

```
lag_value(t) = value(t - h ± tolerance)
```

Where:
- `t` = current timestamp
- `h` = lag period in hours
- `tolerance` = time window for finding closest data point
- `value(t - h ± tolerance)` = closest value within tolerance window

#### Tolerance Windows
| Lag Period | Tolerance | Search Window |
|------------|-----------|---------------|
| 1 hour | ±30 minutes | t-1.5h to t-0.5h |
| 2 hours | ±45 minutes | t-2.75h to t-1.25h |
| 3 hours | ±50 minutes | t-3.83h to t-2.17h |
| 6+ hours | ±1 hour | t-(h+1) to t-(h-1) |

#### Implementation Algorithm
1. Calculate target time: `target_time = current_time - lag_hours`
2. Define search window: `[target_time - tolerance, target_time + tolerance]`
3. Find all data points within window
4. Select closest data point to target_time
5. If no data found, return NaN

#### Feature List
| Feature | Formula | Description |
|---------|---------|-------------|
| `pm2_5_lag_1h` | PM2.5(t-1h ± 30min) | 1-hour lag of PM2.5 |
| `pm2_5_lag_2h` | PM2.5(t-2h ± 45min) | 2-hour lag of PM2.5 |
| `pm2_5_lag_3h` | PM2.5(t-3h ± 50min) | 3-hour lag of PM2.5 |
| `pm2_5_lag_6h` | PM2.5(t-6h ± 1h) | 6-hour lag of PM2.5 |
| `pm2_5_lag_12h` | PM2.5(t-12h ± 1h) | 12-hour lag of PM2.5 |
| `pm2_5_lag_24h` | PM2.5(t-24h ± 1h) | 24-hour lag of PM2.5 |
| `pm2_5_lag_48h` | PM2.5(t-48h ± 1h) | 48-hour lag of PM2.5 |
| `pm2_5_lag_72h` | PM2.5(t-72h ± 1h) | 72-hour lag of PM2.5 |
| `pm10_lag_1h` | PM10(t-1h ± 30min) | 1-hour lag of PM10 |
| `pm10_lag_2h` | PM10(t-2h ± 45min) | 2-hour lag of PM10 |
| `pm10_lag_3h` | PM10(t-3h ± 50min) | 3-hour lag of PM10 |
| `pm10_lag_6h` | PM10(t-6h ± 1h) | 6-hour lag of PM10 |
| `pm10_lag_12h` | PM10(t-12h ± 1h) | 12-hour lag of PM10 |
| `pm10_lag_24h` | PM10(t-24h ± 1h) | 24-hour lag of PM10 |
| `pm10_lag_48h` | PM10(t-48h ± 1h) | 48-hour lag of PM10 |
| `pm10_lag_72h` | PM10(t-72h ± 1h) | 72-hour lag of PM10 |

---

## 6. Rolling Statistics Features (Time-Based)

### Rolling Windows Used
- 3 hours, 6 hours, 12 hours, 24 hours

### Rolling Features for PM2.5 and PM10

#### Mathematical Formula
For each target variable (PM2.5, PM10), window size (w hours), and statistic type:

```
rolling_stat(t) = statistic(value(t-w) to value(t))
```

Where:
- `t` = current timestamp
- `w` = window size in hours
- `value(t-w) to value(t)` = all data points in the time window
- `statistic` = mean, std, min, or max

#### Implementation Algorithm
1. Define window: `[current_time - window_hours, current_time]`
2. Get ALL data points within window (no tolerance)
3. Calculate statistic on all available data
4. If < 2 data points, return NaN

#### Rolling Statistics Formulas
| Statistic | Formula | Description |
|-----------|---------|-------------|
| **Mean** | `μ = (Σx_i) / n` | Average of all values in window |
| **Standard Deviation** | `σ = √(Σ(x_i - μ)² / (n-1))` | Spread of values in window |
| **Minimum** | `min = min(x_1, x_2, ..., x_n)` | Smallest value in window |
| **Maximum** | `max = max(x_1, x_2, ..., x_n)` | Largest value in window |

#### Feature List
| Feature | Formula | Description |
|---------|---------|-------------|
| `pm2_5_rolling_mean_3h` | mean(PM2.5[t-3h to t]) | 3-hour rolling mean |
| `pm2_5_rolling_std_3h` | std(PM2.5[t-3h to t]) | 3-hour rolling std |
| `pm2_5_rolling_min_3h` | min(PM2.5[t-3h to t]) | 3-hour rolling min |
| `pm2_5_rolling_max_3h` | max(PM2.5[t-3h to t]) | 3-hour rolling max |
| `pm2_5_rolling_mean_6h` | mean(PM2.5[t-6h to t]) | 6-hour rolling mean |
| `pm2_5_rolling_std_6h` | std(PM2.5[t-6h to t]) | 6-hour rolling std |
| `pm2_5_rolling_min_6h` | min(PM2.5[t-6h to t]) | 6-hour rolling min |
| `pm2_5_rolling_max_6h` | max(PM2.5[t-6h to t]) | 6-hour rolling max |
| `pm2_5_rolling_mean_12h` | mean(PM2.5[t-12h to t]) | 12-hour rolling mean |
| `pm2_5_rolling_std_12h` | std(PM2.5[t-12h to t]) | 12-hour rolling std |
| `pm2_5_rolling_min_12h` | min(PM2.5[t-12h to t]) | 12-hour rolling min |
| `pm2_5_rolling_max_12h` | max(PM2.5[t-12h to t]) | 12-hour rolling max |
| `pm2_5_rolling_mean_24h` | mean(PM2.5[t-24h to t]) | 24-hour rolling mean |
| `pm2_5_rolling_std_24h` | std(PM2.5[t-24h to t]) | 24-hour rolling std |
| `pm2_5_rolling_min_24h` | min(PM2.5[t-24h to t]) | 24-hour rolling min |
| `pm2_5_rolling_max_24h` | max(PM2.5[t-24h to t]) | 24-hour rolling max |

*(Same pattern for PM10 with `pm10_` prefix)*

---

## 7. Change Rate Features

### Time-Based Change Rate (Complex)

#### Mathematical Formula
For each target variable and period (h hours):

```
change_rate(t) = (value(t) - value(t-h ± tolerance)) / value(t-h ± tolerance)
```

Where:
- `t` = current timestamp
- `h` = period in hours
- `value(t-h ± tolerance)` = closest value within tolerance window (same as lag features)
- If denominator = 0 or NaN, result = NaN

#### Tolerance Windows (Same as Lag Features)
| Period | Tolerance | Search Window |
|--------|-----------|---------------|
| 1 hour | ±30 minutes | t-1.5h to t-0.5h |
| 6 hours | ±1 hour | t-7h to t-5h |
| 24 hours | ±1 hour | t-25h to t-23h |

#### Implementation Algorithm
1. Get current value: `value(t)`
2. Find historical value using lag algorithm: `value(t-h ± tolerance)`
3. Calculate: `(current - historical) / historical`
4. If historical = 0 or NaN, return NaN

#### Feature List
| Feature | Formula | Description |
|---------|---------|-------------|
| `pm2_5_change_rate_1h` | (PM2.5(t) - PM2.5(t-1h ± 30min)) / PM2.5(t-1h ± 30min) | 1-hour PM2.5 change rate |
| `pm2_5_change_rate_6h` | (PM2.5(t) - PM2.5(t-6h ± 1h)) / PM2.5(t-6h ± 1h) | 6-hour PM2.5 change rate |
| `pm2_5_change_rate_24h` | (PM2.5(t) - PM2.5(t-24h ± 1h)) / PM2.5(t-24h ± 1h) | 24-hour PM2.5 change rate |
| `pm10_change_rate_1h` | (PM10(t) - PM10(t-1h ± 30min)) / PM10(t-1h ± 30min) | 1-hour PM10 change rate |
| `pm10_change_rate_6h` | (PM10(t) - PM10(t-6h ± 1h)) / PM10(t-6h ± 1h) | 6-hour PM10 change rate |
| `pm10_change_rate_24h` | (PM10(t) - PM10(t-24h ± 1h)) / PM10(t-24h ± 1h) | 24-hour PM10 change rate |

### Simple Change Rate (Shift-Based)

#### Mathematical Formula
For each variable:

```
simple_change_rate(t) = (value(t) - value(t-1)) / value(t-1)
```

Where:
- `t` = current timestamp
- `value(t-1)` = value from previous row (assumes consecutive hourly data)
- If denominator = 0 or NaN, result = NaN

#### Implementation
Uses pandas `pct_change(1)` which assumes:
- Data is sorted by timestamp
- Consecutive rows are 1 hour apart
- No missing timestamps

#### Feature List
| Feature | Formula | Description |
|---------|---------|-------------|
| `pm2_5_change_rate` | (PM2.5(t) - PM2.5(t-1)) / PM2.5(t-1) | Simple 1-step PM2.5 change |
| `pm10_change_rate` | (PM10(t) - PM10(t-1)) / PM10(t-1) | Simple 1-step PM10 change |
| `no2_change_rate` | (NO2(t) - NO2(t-1)) / NO2(t-1) | Simple 1-step NO2 change |
| `o3_change_rate` | (O3(t) - O3(t-1)) / O3(t-1) | Simple 1-step O3 change |
| `co_change_rate` | (CO(t) - CO(t-1)) / CO(t-1) | Simple 1-step CO change |
| `so2_change_rate` | (SO2(t) - SO2(t-1)) / SO2(t-1) | Simple 1-step SO2 change |
| `temp_change_rate` | (Temp(t) - Temp(t-1)) / Temp(t-1) | Simple 1-step temperature change |
| `pressure_change_rate` | (Pressure(t) - Pressure(t-1)) / Pressure(t-1) | Simple 1-step pressure change |

---

## 8. Derived Pollutant Features

### PM2.5 Features
| Feature | Calculation | Description |
|---------|-------------|-------------|
| `pm2_5_squared` | pm2_5² | PM2.5 squared (non-linear) |
| `is_high_pm2_5` | Binary | 1 if PM2.5 > 35 μg/m³ |
| `is_low_pm2_5` | Binary | 1 if PM2.5 < 12 μg/m³ |

### PM10 Features
| Feature | Calculation | Description |
|---------|-------------|-------------|
| `pm10_squared` | pm10² | PM10 squared (non-linear) |
| `is_high_pm10` | Binary | 1 if PM10 > 50 μg/m³ |
| `is_low_pm10` | Binary | 1 if PM10 < 20 μg/m³ |

### NO2 Features
| Feature | Calculation | Description |
|---------|-------------|-------------|
| `no2_squared` | nitrogen_dioxide² | NO2 squared (non-linear) |
| `is_high_no2` | Binary | 1 if NO2 > 200 μg/m³ |

### O3 Features
| Feature | Calculation | Description |
|---------|-------------|-------------|
| `o3_squared` | ozone² | O3 squared (non-linear) |
| `is_high_o3` | Binary | 1 if O3 > 100 μg/m³ |

### CO Features
| Feature | Calculation | Description |
|---------|-------------|-------------|
| `co_squared` | carbon_monoxide² | CO squared (non-linear) |
| `is_high_co` | Binary | 1 if CO > 5000 μg/m³ |

### SO2 Features
| Feature | Calculation | Description |
|---------|-------------|-------------|
| `so2_squared` | sulphur_dioxide² | SO2 squared (non-linear) |
| `is_high_so2` | Binary | 1 if SO2 > 500 μg/m³ |

---

## 9. Interaction Features

| Feature | Calculation | Description |
|---------|-------------|-------------|
| `temp_humidity_interaction` | temperature × humidity | Temperature-humidity interaction |
| `temp_wind_interaction` | temperature × wind_speed | Temperature-wind interaction |
| `pm2_5_temp_interaction` | pm2_5 × temperature | PM2.5-temperature interaction |
| `pm2_5_humidity_interaction` | pm2_5 × humidity | PM2.5-humidity interaction |
| `wind_pm2_5_interaction` | wind_speed × pm2_5 | Wind-PM2.5 interaction |

---

## 10. Location Features

| Feature | Source | Description |
|---------|--------|-------------|
| `city` | Static | City name (Multan) |
| `latitude` | Static | Latitude (30.1575°N) |
| `longitude` | Static | Longitude (71.5249°E) |

---

## Feature Summary

### Total Feature Count
- **Raw Features**: ~20 (weather + pollutants)
- **Time Features**: ~16 (cyclic + seasonal + time-of-day)
- **Lag Features**: ~16 (8 lags × 2 targets)
- **Rolling Features**: ~32 (4 windows × 4 stats × 2 targets)
- **Change Rate Features**: ~14 (complex + simple)
- **Derived Features**: ~20 (squared + binary indicators)
- **Interaction Features**: ~5
- **AQI Features**: ~4
- **Total**: ~127 features

### Feature Categories
1. **Raw Data**: Directly from APIs
2. **Time-Based**: Cyclic encoding and temporal patterns
3. **Historical**: Lag and rolling statistics
4. **Derived**: Mathematical transformations
5. **Interaction**: Multi-variable relationships
6. **Categorical**: Binary indicators and thresholds

### Target Variables
- **Primary Target**: `pm2_5` (fine particulate matter)
- **Secondary Target**: `pm10` (coarse particulate matter)
- **Reference**: `us_aqi` (overall air quality index)

---

## Data Flow

1. **Data Collection**: OpenWeather APIs → Raw weather and pollution data
2. **Feature Engineering**: Raw data → 127 engineered features
3. **Model Input**: 127 features → PM2.5/PM10 predictions
4. **Output**: Predicted concentrations + AQI values

---

## Notes

- All features are calculated on hourly data
- Time-based features use tolerance for missing data
- Cyclic encoding improves ML model performance
- Interaction features capture non-linear relationships
- Binary indicators help with threshold-based patterns
- Lag and rolling features capture temporal dependencies 