"""
Manual Historic Data Run (CSV + Hopsworks Version)
--------------------------------------------------
This script collects historical data for June 16-30, 2025 and saves to CSV AND Hopsworks.

Uses EXACT same feature engineering and Hopsworks integration as automated_hourly_run.py:
- Same AQIFeatureEngineer class and logic
- Same column order and naming
- Same Hopsworks upload process
- Same data types (float64 for numerics)
- Same timestamp handling

Note: Uses multiple API calls to work around OpenWeather's record limit per request.
"""
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
from config import OPENWEATHER_CONFIG, FEATURE_CONFIG, PATHS, HOPSWORKS_CONFIG
from feature_engineering import AQIFeatureEngineer
from hopsworks_integration import HopsworksUploader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("manual_historic_csv_run.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

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
}

def calc_aqi(conc, breakpoints):
    """Calculate AQI for a given concentration using US EPA breakpoints"""
    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= conc <= C_high:
            return round((I_high - I_low) / (C_high - C_low) * (conc - C_low) + I_low)
    return None

def compute_pm2_5_aqi(row):
    """Compute PM2.5 AQI only"""
    if not pd.isna(row.get("pm2_5")):
        return calc_aqi(row["pm2_5"], AQI_BREAKPOINTS["pm2_5"])
    return None

def fetch_historic_weather_batch(start_unix, end_unix, max_records_per_call=24):
    """
    Fetch hourly weather data using multiple API calls to work around record limits.
    Returns DataFrame indexed by UTC timestamp.
    """
    all_data = []
    current_start = start_unix
    
    while current_start < end_unix:
        # Calculate end for this batch (max 24 hours per call)
        batch_end = min(current_start + (max_records_per_call * 3600), end_unix)
        
        url = f"https://history.openweathermap.org/data/2.5/history/city"
        params = {
            "lat": OPENWEATHER_CONFIG['lat'],
            "lon": OPENWEATHER_CONFIG['lon'],
            "type": "hour",
            "start": current_start,
            "end": batch_end,
            "appid": OPENWEATHER_CONFIG['api_key'],
            "units": "metric"
        }
        
        try:
            logger.info(f"Fetching weather data: {datetime.fromtimestamp(current_start)} to {datetime.fromtimestamp(batch_end)}")
            
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            records_count = len(data.get("list", []))
            logger.info(f"Weather API response: {records_count} records")
            
            for entry in data.get("list", []):
                dt = pd.to_datetime(entry["dt"], unit="s", utc=True)
                main = entry.get("main", {})
                wind = entry.get("wind", {})
                row = {
                    "time": dt,
                    "temperature": main.get("temp"),
                    "humidity": main.get("humidity"),
                    "pressure": main.get("pressure"),
                    "wind_speed": wind.get("speed"),
                    "wind_direction": wind.get("deg"),
                }
                all_data.append(row)
            
            # Move to next batch
            current_start = batch_end
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error fetching weather data batch: {e}")
            current_start = batch_end  # Move to next batch even if this one failed
    
    df = pd.DataFrame(all_data)
    if not df.empty:
        df.set_index("time", inplace=True)
        df = df.sort_index()  # Ensure chronological order
    return df

def fetch_historic_pollution_batch(start_unix, end_unix, max_records_per_call=24):
    """
    Fetch hourly air pollution data using multiple API calls to work around record limits.
    Returns DataFrame indexed by UTC timestamp.
    """
    all_data = []
    current_start = start_unix
    
    while current_start < end_unix:
        # Calculate end for this batch (max 24 hours per call)
        batch_end = min(current_start + (max_records_per_call * 3600), end_unix)
        
        url = f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        params = {
            "lat": OPENWEATHER_CONFIG['lat'],
            "lon": OPENWEATHER_CONFIG['lon'],
            "start": current_start,
            "end": batch_end,
            "appid": OPENWEATHER_CONFIG['api_key']
        }
        
        try:
            logger.info(f"Fetching pollution data: {datetime.fromtimestamp(current_start)} to {datetime.fromtimestamp(batch_end)}")
            
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            records_count = len(data.get("list", []))
            logger.info(f"Pollution API response: {records_count} records")
            
            for entry in data.get("list", []):
                row = {"time": pd.to_datetime(entry["dt"], unit="s", utc=True)}
                
                # Add pollutant components
                for k, v in entry["components"].items():
                    row[k] = v
                
                # Extract OpenWeather AQI (1-5 scale)
                if "main" in entry and "aqi" in entry["main"]:
                    row["openweather_aqi"] = entry["main"]["aqi"]
                else:
                    row["openweather_aqi"] = None
                    
                all_data.append(row)
            
            # Move to next batch
            current_start = batch_end
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error fetching pollution data batch: {e}")
            current_start = batch_end  # Move to next batch even if this one failed
    
    df = pd.DataFrame(all_data)
    if not df.empty:
        df.set_index("time", inplace=True)
        
        # Calculate PM2.5 AQI
        df["pm2_5_aqi"] = df.apply(compute_pm2_5_aqi, axis=1)
        
        # Calculate PM10 AQI
        df["pm10_aqi"] = df.apply(
            lambda row: calc_aqi(row.get('pm10'), AQI_BREAKPOINTS['pm10']) 
            if pd.notna(row.get('pm10')) else None, axis=1
        )
        
        # Calculate US AQI as maximum of PM2.5 and PM10 AQI
        df["us_aqi"] = df.apply(
            lambda row: max(row['pm2_5_aqi'], row['pm10_aqi']) 
            if pd.notna(row.get('pm2_5_aqi')) and pd.notna(row.get('pm10_aqi')) else None, axis=1
        )
        
        df = df.sort_index()  # Ensure chronological order
    
    return df

def collect_historical_data_march_to_june_2025():
    """
    Collect historical data for March 1, 2025 to June 15, 2025 with timestamp matching.
    Uses multiple API calls to work around record limits.
    Returns DataFrame with matched weather and pollution data.
    """
    logger.info("Collecting historical data for March 1, 2025 to June 15, 2025...")
    
    # Define date range: March 1, 2025 00:00 UTC to June 15, 2025 23:59 UTC
    start_date = datetime(2025, 3, 1, 0, 0, 0, tzinfo=None)   # March 1, 2025 00:00
    end_date = datetime(2025, 6, 15, 23, 59, 59, tzinfo=None)  # June 15, 2025 23:59
    
    # Convert to Unix timestamps
    start_unix = int(start_date.timestamp())
    end_unix = int(end_date.timestamp())
    
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Unix timestamps: {start_unix} to {end_unix}")
    
    # Fetch both weather and pollution data using batch approach
    logger.info("Fetching historic weather data (using multiple API calls)...")
    df_weather = fetch_historic_weather_batch(start_unix, end_unix)
    logger.info(f"Fetched {len(df_weather)} weather records")
    
    logger.info("Fetching historic pollution data (using multiple API calls)...")
    df_pollution = fetch_historic_pollution_batch(start_unix, end_unix)
    logger.info(f"Fetched {len(df_pollution)} pollution records")
    
    if df_pollution.empty:
        logger.error("Failed to fetch pollution data from API")
        return pd.DataFrame()
    
    if df_weather.empty:
        logger.warning("Failed to fetch weather data from API. Proceeding with pollution data only.")
        # Create a minimal weather DataFrame with the same timestamps as pollution data
        weather_data = []
        for timestamp in df_pollution.index:
            weather_data.append({
                "time": timestamp,
                "temperature": None,
                "humidity": None,
                "pressure": None,
                "wind_speed": None,
                "wind_direction": None,
            })
        df_weather = pd.DataFrame(weather_data)
        df_weather.set_index("time", inplace=True)
        logger.info(f"Created placeholder weather data for {len(df_weather)} timestamps")
    
    # Merge on timestamp (inner join to ensure matching timestamps)
    logger.info("Merging weather and pollution data on timestamp...")
    df_merged = df_weather.join(df_pollution, how="inner")
    
    logger.info(f"After merging: {len(df_merged)} records with matching timestamps")
    
    # Add location metadata
    df_merged["city"] = OPENWEATHER_CONFIG["city"]
    df_merged["latitude"] = OPENWEATHER_CONFIG["lat"]
    df_merged["longitude"] = OPENWEATHER_CONFIG["lon"]
    
    # Rename columns to match feature engineering expectations
    rename_map = {
        'co': 'carbon_monoxide',
        'no2': 'nitrogen_dioxide',
        'o3': 'ozone',
        'so2': 'sulphur_dioxide'
    }
    df_merged = df_merged.rename(columns=rename_map)
    
    return df_merged

def run_manual_backfill_csv_and_hopsworks():
    """
    Executes the full historical data backfill pipeline and saves to CSV AND Hopsworks.
    Uses EXACT same logic as automated_hourly_run.py.
    """
    logger.info("=================================================")
    logger.info("=== STARTING MANUAL HISTORICAL DATA BACKFILL (CSV + HOPSWORKS) ===")
    logger.info("=================================================")
    
    # Debug: Check API keys are loaded
    logger.info("DEBUG: Checking API keys...")
    openweather_key = os.getenv("OPENWEATHER_API_KEY")
    hopsworks_key = os.getenv("HOPSWORKS_API_KEY")
    
    logger.info(f"DEBUG: OPENWEATHER_API_KEY present: {'Yes' if openweather_key else 'No'}")
    logger.info(f"DEBUG: HOPSWORKS_API_KEY present: {'Yes' if hopsworks_key else 'No'}")
    
    if not openweather_key:
        logger.error("ERROR: OPENWEATHER_API_KEY environment variable is not set!")
        return False
    if not hopsworks_key:
        logger.error("ERROR: HOPSWORKS_API_KEY environment variable is not set!")
        return False

    # 1. Initialize Hopsworks connection first (same as automated_hourly_run.py)
    logger.info("STEP 1: Connecting to Hopsworks...")
    uploader = HopsworksUploader(
        api_key=os.getenv("HOPSWORKS_API_KEY", HOPSWORKS_CONFIG.get('api_key')),
        project_name=HOPSWORKS_CONFIG['project_name']
    )
    if not uploader.connect():
        logger.error("Could not connect to Hopsworks. Aborting.")
        return False

    # 2. Collect Historical Data
    logger.info("STEP 2: Collecting historical data for March 1, 2025 to June 15, 2025...")
    raw_df = collect_historical_data_march_to_june_2025()
    if raw_df.empty:
        logger.error("Data collection failed. No data to process. Aborting.")
        return False
    logger.info(f"Successfully collected {len(raw_df)} records.")

    # Save raw data as CSV for reference
    raw_data_path = "data/raw_historical_data_march1_june15_2025.csv"
    os.makedirs("data", exist_ok=True)
    raw_df.to_csv(raw_data_path, index=False)
    logger.info(f"Raw data saved to {raw_data_path}")

    # 3. Engineer Features (EXACT same logic as automated_hourly_run.py)
    logger.info("\nSTEP 3: Engineering features...")
    engineer = AQIFeatureEngineer()

    # Explicitly cast all numerics to float64 (same as automated_hourly_run.py)
    numeric_cols = raw_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        raw_df[col] = raw_df[col].astype('float64')
    logger.info(f"Casted {len(numeric_cols)} numeric columns to float64 for Hopsworks consistency")

    engineered_df = engineer.engineer_features(raw_df)
    if engineered_df.empty:
        logger.error("Feature engineering resulted in an empty DataFrame. Aborting.")
        return False
    logger.info(f"Successfully engineered {engineered_df.shape[1]} features.")

    # Debug: Print some statistics
    print("Engineered DataFrame columns:")
    print(engineered_df.columns.tolist())
    print(f"\nDataFrame shape: {engineered_df.shape}")
    print(f"Date range: {engineered_df.index.min()} to {engineered_df.index.max()}")
    print(f"US AQI range: {engineered_df['us_aqi'].min()} - {engineered_df['us_aqi'].max()}")
    if 'pm2_5' in engineered_df.columns:
        print(f"Raw PM2.5 range: {engineered_df['pm2_5'].min():.2f} - {engineered_df['pm2_5'].max():.2f} μg/m³")
    if 'pm10' in engineered_df.columns:
        print(f"Raw PM10 range: {engineered_df['pm10'].min():.2f} - {engineered_df['pm10'].max():.2f} μg/m³")

    # 4. Save to CSV
    logger.info("\nSTEP 4: Saving engineered features to CSV...")
    engineered_data_path = "data/engineered_features_march1_june15_2025.csv"
    engineered_df.to_csv(engineered_data_path, index=False)
    
    logger.info(f"Engineered features saved to {engineered_data_path}")
    logger.info(f"Total records: {len(engineered_df)}")
    logger.info(f"Total features: {len(engineered_df.columns)}")

    # 5. Commit CSV files to Git (optional)
    logger.info("\nSTEP 5: Committing CSV files to Git...")
    try:
        import subprocess
        
        # Configure Git user (if not already configured)
        try:
            subprocess.run(["git", "config", "--local", "user.email", "action@github.com"], check=True)
            subprocess.run(["git", "config", "--local", "user.name", "GitHub Action"], check=True)
            logger.info("✓ Configured Git user")
        except subprocess.CalledProcessError:
            logger.info("Git user already configured or not needed")
        
        # Add the CSV files to git
        subprocess.run(["git", "add", raw_data_path, engineered_data_path], check=True)
        logger.info("✓ Added CSV files to git staging")
        
        # Commit with descriptive message
        commit_message = f"Add historical data for March 1-June 15, 2025 ({len(engineered_df)} records)"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        logger.info(f"✓ Committed CSV files: {commit_message}")
        
        # Push to remote repository
        subprocess.run(["git", "push"], check=True)
        logger.info("✓ Pushed changes to remote repository")
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Git operations failed: {e}")
        logger.info("Continuing with Hopsworks upload...")
    except Exception as e:
        logger.warning(f"Git operations failed: {e}")
        logger.info("Continuing with Hopsworks upload...")

    # 6. Push to Hopsworks (EXACT same logic as automated_hourly_run.py)
    logger.info("\nSTEP 6: Pushing features to Hopsworks...")
    
    # For historical backfill, we need both offline and online storage
    # Create a temporary uploader with both storages enabled
    
    # Override the push_features method for both offline and online storage
    def push_features_both_storages(self, df, group_name, description):
        """Push features to both offline and online storage (for historical backfill)"""
        if self.fs is None:
            logger.error("Not connected to Hopsworks. Please call connect() first.")
            return False
            
        if df.empty:
            logger.warning("Input DataFrame is empty. Nothing to push to Hopsworks.")
            return True

        # Prepare the DataFrame for Hopsworks
        df_to_insert = df.copy()
        if 'time' not in df_to_insert.columns:
             if isinstance(df_to_insert.index, pd.DatetimeIndex):
                 df_to_insert.reset_index(inplace=True)
             else:
                 logger.error("DataFrame must have a 'time' column or a DatetimeIndex.")
                 return False

        # Ensure event_time column is correct format
        df_to_insert['time'] = pd.to_datetime(df_to_insert['time'])
        
        # Primary key must be unique and is required for online feature store
        # Round time to nearest hour for primary key (e.g., 22/07/2025 9:00:45 PM -> 22/07/2025 9:00:00 PM)
        df_to_insert['time_str'] = df_to_insert['time'].dt.floor('H').dt.strftime('%Y-%m-%d %H:%M:%S')

        try:
            # Get or create the feature group with BOTH offline and online storage
            fg = self.fs.get_or_create_feature_group(
                name=group_name,
                version=1,
                description=description,
                primary_key=['time_str'],
                event_time='time',
                online_enabled=True  # Enable both offline and online storage
            )

            logger.info(f"Inserting {len(df_to_insert)} rows into feature group '{group_name}' (offline + online)...")
            fg.insert(df_to_insert, write_options={"wait_for_job": True})
            logger.info("Successfully inserted data into Hopsworks (offline + online storage).")
            return True
        except Exception as e:
            logger.error(f"Failed to insert data into Hopsworks: {e}")
            return False
    
    # Temporarily replace the push_features method
    original_push_features = uploader.push_features
    uploader.push_features = push_features_both_storages.__get__(uploader, HopsworksUploader)
    
    try:
        success = uploader.push_features(
            df=engineered_df,
            group_name=HOPSWORKS_CONFIG['feature_group_name'],
            description="Historical backfill of PM2.5 and PM10 prediction features for Multan (March 1-June 15, 2025). Targets: pm2_5, pm10 (raw concentrations), Reference: us_aqi (final AQI)."
        )
    finally:
        # Restore original method
        uploader.push_features = original_push_features
    
    if not success:
        logger.error("Failed to push features to Hopsworks.")
        return False

    logger.info("\n=============================================")
    logger.info("=== MANUAL HISTORICAL RUN (CSV + HOPSWORKS) COMPLETED SUCCESSFULLY ===")
    logger.info("=============================================")
    return True

if __name__ == "__main__":
    if not run_manual_backfill_csv_and_hopsworks():
        logger.error("The manual backfill process failed. Check logs for details.")
        exit(1) 