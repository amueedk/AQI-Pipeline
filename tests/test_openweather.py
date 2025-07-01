"""
Test OpenWeather API Integration
-------------------------------
This script tests whether the OpenWeather API is working and returns data for Multan.
It prints the resulting DataFrame and a clear success/failure message.
Also fetches and prints the last 48 hours of air pollution history if available.
"""
from data_collector import OpenWeatherDataCollector
import time

print("Testing OpenWeather API integration for Multan...")
collector = OpenWeatherDataCollector()

# Test current air pollution and weather data
try:
    df_current = collector.collect_current_data()
    if df_current is not None and not df_current.empty:
        print("\n✅ OpenWeather API SUCCESS! Current data fetched:")
        print(df_current)
    else:
        print("\n❌ OpenWeather API FAILED or returned no current data.")
        print(df_current)
except Exception as e:
    print(f"\n❌ OpenWeather API ERROR: {e}")

# Test historical air pollution data (last 48 hours)
print("\nFetching last 48 hours of air pollution history from OpenWeather...")
try:
    end_time = int(time.time())
    start_time = end_time - 48 * 3600
    df_hist = collector.fetch_air_pollution(start_unix=start_time, end_unix=end_time)
    if df_hist is not None and not df_hist.empty:
        print("\nLast 48 hours of air pollution history:")
        print(df_hist.head(10))  # Print first 10 rows for brevity
        print("... (truncated) ...")
    else:
        print("No air pollution history available in API response.")
except Exception as e:
    print(f"Error fetching air pollution history: {e}") 