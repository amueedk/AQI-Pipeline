"""
Test IQAir API Integration
-------------------------
This script tests whether the IQAir API is working and returns data for Multan.
It prints the resulting DataFrame and a clear success/failure message.
Also fetches and prints the last 48 hours of pollution history if available.
"""
from data_collector import IQAirDataCollector
import requests
from config import IQAIR_CONFIG
import pandas as pd

print("Testing IQAir API integration for Multan...")
df = IQAirDataCollector().fetch_current_data()

if df is not None and not df.empty:
    print("\n✅ IQAir API SUCCESS! Data fetched:")
    print(df)
else:
    print("\n❌ IQAir API FAILED or returned no data.")
    print(df)

# Fetch and print pollution history (last 48 hours)
print("\nFetching last 48 hours of pollution history from IQAir...")
url = f"{IQAIR_CONFIG['base_url']}/city"
params = {
    "city": IQAIR_CONFIG["city"],
    "state": IQAIR_CONFIG["state"],
    "country": IQAIR_CONFIG["country"],
    "key": IQAIR_CONFIG["api_key"]
}
try:
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data["status"] == "success" and "history" in data["data"] and "pollution" in data["data"]["history"]:
        pollution_hist = data["data"]["history"]["pollution"]
        hist_df = pd.DataFrame(pollution_hist)
        print("\nLast 48 hours of pollution history:")
        print(hist_df.head(10))  # Print first 10 rows for brevity
        print("... (truncated) ...")
    else:
        print("No pollution history available in API response.")
except Exception as e:
    print(f"Error fetching pollution history: {e}") 