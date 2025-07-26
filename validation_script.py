import pandas as pd
import numpy as np
import hopsworks
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureGroupValidator:
    """
    Validate and compare old vs new feature groups during transition
    """
    
    def __init__(self):
        self.connection = hopsworks.connection()
        self.fs = self.connection.get_feature_store()
    
    def read_feature_groups(self):
        """
        Read both old and new feature groups
        """
        try:
            # Read old feature group
            old_fg = self.fs.get_feature_group("aqi_features", version=1)
            old_data = old_fg.read()
            print(f"✅ Read OLD feature group: {len(old_data)} rows, {len(old_data.columns)} columns")
            
            # Read new feature group
            new_fg = self.fs.get_feature_group("aqi_clean_features", version=1)
            new_data = new_fg.read()
            print(f"✅ Read NEW feature group: {len(new_data)} rows, {len(new_data.columns)} columns")
            
            return old_data, new_data
            
        except Exception as e:
            print(f"❌ Error reading feature groups: {e}")
            return None, None
    
    def compare_feature_groups(self, old_data, new_data):
        """
        Compare old vs new feature groups
        """
        print("\n" + "="*60)
        print("📊 FEATURE GROUP COMPARISON")
        print("="*60)
        
        # Basic statistics
        print(f"📊 OLD Feature Group:")
        print(f"   Rows: {len(old_data)}")
        print(f"   Columns: {len(old_data.columns)}")
        print(f"   Date range: {old_data['timestamp'].min()} to {old_data['timestamp'].max()}")
        print(f"   Missing values: {old_data.isnull().sum().sum()}")
        
        print(f"\n✨ NEW Feature Group:")
        print(f"   Rows: {len(new_data)}")
        print(f"   Columns: {len(new_data.columns)}")
        print(f"   Date range: {new_data['timestamp'].min()} to {new_data['timestamp'].max()}")
        print(f"   Missing values: {new_data.isnull().sum().sum()}")
        
        # Feature count comparison
        feature_reduction = len(old_data.columns) - len(new_data.columns)
        reduction_percentage = (feature_reduction / len(old_data.columns)) * 100
        
        print(f"\n📈 Feature Reduction:")
        print(f"   Old features: {len(old_data.columns)}")
        print(f"   New features: {len(new_data.columns)}")
        print(f"   Reduction: {feature_reduction} features ({reduction_percentage:.1f}%)")
        
        # Check for required features in new group
        self.check_required_features(new_data)
        
        # Data quality comparison
        self.compare_data_quality(old_data, new_data)
        
        # Wind direction engineering validation
        self.validate_wind_direction_engineering(new_data)
    
    def check_required_features(self, new_data):
        """
        Check if all required features are present in new group
        """
        print(f"\n🔍 Required Features Check:")
        
        required_features = [
            # Wind direction engineering (CRITICAL)
            'wind_direction_sin', 'wind_direction_cos',
            'is_wind_from_high_pm', 'is_wind_from_low_pm',
            
            # Pollutant lags (NEW)
            'co_lag_1h', 'o3_lag_1h', 'so2_lag_1h',
            
            # Optimized rolling features
            'pm2_5_rolling_min_3h', 'pm2_5_rolling_mean_3h',
            'pm10_rolling_min_3h', 'pm10_rolling_mean_3h',
            
            # Cyclical time features
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            
            # Optimized binary features
            'is_hot', 'is_night', 'is_morning_rush', 'is_evening_rush',
            'is_high_pm2_5', 'is_high_o3',
            
            # Interaction features
            'temp_humidity_interaction', 'temp_wind_interaction',
            'wind_direction_temp_interaction', 'wind_direction_humidity_interaction',
            'pressure_humidity_interaction',
            
            # Pollutant-weather interactions
            'co_pressure_interaction', 'o3_temp_interaction', 'so2_humidity_interaction'
        ]
        
        missing_features = [f for f in required_features if f not in new_data.columns]
        
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
            return False
        else:
            print("✅ All required features present")
            return True
    
    def compare_data_quality(self, old_data, new_data):
        """
        Compare data quality between old and new groups
        """
        print(f"\n🔍 Data Quality Comparison:")
        
        # Missing values comparison
        old_missing = old_data.isnull().sum().sum()
        new_missing = new_data.isnull().sum().sum()
        
        print(f"   Missing values - Old: {old_missing}, New: {new_missing}")
        
        # Zero values comparison
        old_zeros = (old_data == 0).sum().sum()
        new_zeros = (new_data == 0).sum().sum()
        
        print(f"   Zero values - Old: {old_zeros}, New: {new_zeros}")
        
        # Infinite values check
        old_infinite = np.isinf(old_data.select_dtypes(include=[np.number])).sum().sum()
        new_infinite = np.isinf(new_data.select_dtypes(include=[np.number])).sum().sum()
        
        print(f"   Infinite values - Old: {old_infinite}, New: {new_infinite}")
        
        # Range checks for key features
        print(f"\n📊 Key Feature Ranges:")
        
        # PM2.5 range
        old_pm25_range = f"{old_data['pm2_5'].min():.2f} - {old_data['pm2_5'].max():.2f}"
        new_pm25_range = f"{new_data['pm2_5'].min():.2f} - {new_data['pm2_5'].max():.2f}"
        print(f"   PM2.5 range - Old: {old_pm25_range}, New: {new_pm25_range}")
        
        # Temperature range
        old_temp_range = f"{old_data['temperature'].min():.2f} - {old_data['temperature'].max():.2f}"
        new_temp_range = f"{new_data['temperature'].min():.2f} - {new_data['temperature'].max():.2f}"
        print(f"   Temperature range - Old: {old_temp_range}, New: {new_temp_range}")
    
    def validate_wind_direction_engineering(self, new_data):
        """
        Validate wind direction engineering (CRITICAL feature)
        """
        print(f"\n💨 Wind Direction Engineering Validation:")
        
        # Check if wind direction features exist
        wind_features = ['wind_direction_sin', 'wind_direction_cos', 
                        'is_wind_from_high_pm', 'is_wind_from_low_pm']
        
        missing_wind = [f for f in wind_features if f not in new_data.columns]
        if missing_wind:
            print(f"❌ Missing wind direction features: {missing_wind}")
            return False
        
        # Validate cyclical encoding ranges
        sin_range = f"{new_data['wind_direction_sin'].min():.3f} - {new_data['wind_direction_sin'].max():.3f}"
        cos_range = f"{new_data['wind_direction_cos'].min():.3f} - {new_data['wind_direction_cos'].max():.3f}"
        
        print(f"   Wind direction sin range: {sin_range}")
        print(f"   Wind direction cos range: {cos_range}")
        
        # Check if ranges are correct (-1 to 1)
        sin_valid = (-1.001 <= new_data['wind_direction_sin'].min() <= new_data['wind_direction_sin'].max() <= 1.001)
        cos_valid = (-1.001 <= new_data['wind_direction_cos'].min() <= new_data['wind_direction_cos'].max() <= 1.001)
        
        if sin_valid and cos_valid:
            print("✅ Wind direction cyclical encoding is correct")
        else:
            print("❌ Wind direction cyclical encoding has issues")
        
        # Check pollution source indicators
        high_pm_count = new_data['is_wind_from_high_pm'].sum()
        low_pm_count = new_data['is_wind_from_low_pm'].sum()
        total_count = len(new_data)
        
        print(f"   High PM wind direction: {high_pm_count} times ({high_pm_count/total_count*100:.1f}%)")
        print(f"   Low PM wind direction: {low_pm_count} times ({low_pm_count/total_count*100:.1f}%)")
        
        # Check for overlap (should be minimal)
        overlap = ((new_data['is_wind_from_high_pm'] == 1) & (new_data['is_wind_from_low_pm'] == 1)).sum()
        if overlap == 0:
            print("✅ No overlap between high and low PM wind directions")
        else:
            print(f"⚠️ Overlap detected: {overlap} cases")
        
        return True
    
    def check_feature_distributions(self, new_data):
        """
        Check distributions of key features
        """
        print(f"\n📊 Feature Distribution Check:")
        
        # Binary features
        binary_features = ['is_hot', 'is_night', 'is_morning_rush', 'is_evening_rush', 
                          'is_high_pm2_5', 'is_high_o3', 'is_wind_from_high_pm', 'is_wind_from_low_pm']
        
        print("   Binary feature distributions:")
        for feature in binary_features:
            if feature in new_data.columns:
                ones_count = new_data[feature].sum()
                total_count = len(new_data)
                percentage = (ones_count / total_count) * 100
                print(f"     {feature}: {ones_count}/{total_count} ({percentage:.1f}%)")
        
        # Cyclical features
        cyclical_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                           'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
        
        print("   Cyclical feature ranges:")
        for feature in cyclical_features:
            if feature in new_data.columns:
                min_val = new_data[feature].min()
                max_val = new_data[feature].max()
                print(f"     {feature}: {min_val:.3f} to {max_val:.3f}")
    
    def generate_validation_report(self):
        """
        Generate comprehensive validation report
        """
        print("🚀 AQI Feature Group Validation Report")
        print("=" * 60)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Read feature groups
        old_data, new_data = self.read_feature_groups()
        
        if old_data is None or new_data is None:
            print("❌ Failed to read feature groups")
            return False
        
        # Compare feature groups
        self.compare_feature_groups(old_data, new_data)
        
        # Check required features
        features_ok = self.check_required_features(new_data)
        
        # Validate wind direction engineering
        wind_ok = self.validate_wind_direction_engineering(new_data)
        
        # Check feature distributions
        self.check_feature_distributions(new_data)
        
        # Summary
        print(f"\n" + "="*60)
        print("📋 VALIDATION SUMMARY")
        print("="*60)
        
        if features_ok and wind_ok:
            print("✅ VALIDATION PASSED")
            print("📊 Next steps:")
            print("   1. Continue monitoring both groups")
            print("   2. Test model performance with new features")
            print("   3. Switch to new group when validated")
            return True
        else:
            print("❌ VALIDATION FAILED")
            print("📊 Issues found - review and fix before proceeding")
            return False

def main():
    """
    Main validation function
    """
    validator = FeatureGroupValidator()
    success = validator.generate_validation_report()
    
    if success:
        print("\n🎉 Validation completed successfully!")
    else:
        print("\n❌ Validation failed - review issues above")

if __name__ == "__main__":
    main() 