"""
Test script to verify feature engineering works with new temporal features.
"""

import pandas as pd
import numpy as np
import logging
from src.data_cleaning import DataCleaner
from src.feature_engineering import BaselineFeatureEngineer
from src.config import PipelineConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_feature_engineering():
    """Test the feature engineering with temporal features."""
    
    print("## Testing Feature Engineering with Temporal Features")
    print()
    
    # Initialize config and cleaner
    config = PipelineConfig()
    cleaner = DataCleaner(config)
    
    # Clean the data
    df_cleaned = cleaner.clean_data(save_output=False)
    print(f"Cleaned data shape: {df_cleaned.shape}")
    print()
    
    # Engineer features
    engineer = BaselineFeatureEngineer()
    df_engineered = engineer.engineer_features(df_cleaned)
    
    print(f"Engineered data shape: {df_engineered.shape}")
    print()
    
    # Show feature info
    feature_info = engineer.get_feature_info()
    print("Engineered features:")
    for feature, description in feature_info.items():
        print(f"  - {feature}: {description}")
    print()
    
    # Check for new features
    new_features = [col for col in df_engineered.columns if col not in df_cleaned.columns]
    print(f"New features added: {new_features}")
    print()
    
    # Check temporal features are preserved
    temporal_features = [
        'transaction_hour', 'transaction_day_of_week', 'transaction_month',
        'is_weekend', 'is_business_hours', 'is_night_hours',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos', 'is_high_risk_hour'
    ]
    
    preserved_temporal = [col for col in temporal_features if col in df_engineered.columns]
    print(f"Preserved temporal features: {preserved_temporal}")
    print()
    
    # Check interaction features
    interaction_features = [
        'high_risk_hour_weekend', 'high_value_novel_customer', 'night_high_value'
    ]
    
    available_interactions = [col for col in interaction_features if col in df_engineered.columns]
    print(f"Available interaction features: {available_interactions}")
    print()
    
    # Show final columns
    print("Final engineered columns:")
    print(list(df_engineered.columns))
    print()
    
    return df_engineered

if __name__ == "__main__":
    test_feature_engineering() 