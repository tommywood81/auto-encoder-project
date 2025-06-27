"""
Test script to verify datetime features are working properly.
"""

import pandas as pd
import numpy as np
import logging
from src.data_cleaning import DataCleaner
from src.config import PipelineConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_datetime_features():
    """Test the datetime feature processing."""
    
    print("## Testing DateTime Features")
    print()
    
    # Initialize config and cleaner
    config = PipelineConfig()
    cleaner = DataCleaner(config)
    
    # Clean the data
    df = cleaner.clean_data(save_output=False)
    
    print(f"Cleaned data shape: {df.shape}")
    print()
    
    # Check temporal features
    temporal_features = [
        'transaction_hour', 'transaction_day_of_week', 'transaction_month',
        'is_weekend', 'is_business_hours', 'is_night_hours',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos'
    ]
    
    available_temporal = [col for col in temporal_features if col in df.columns]
    print(f"Available temporal features: {available_temporal}")
    print()
    
    # Show sample of temporal data
    if 'transaction_hour' in df.columns:
        print("Hour distribution:")
        print(df['transaction_hour'].value_counts().sort_index())
        print()
    
    if 'is_weekend' in df.columns:
        print("Weekend vs weekday distribution:")
        print(df['is_weekend'].value_counts())
        print()
    
    if 'is_business_hours' in df.columns:
        print("Business hours distribution:")
        print(df['is_business_hours'].value_counts())
        print()
    
    # Check fraud patterns by temporal features
    if 'is_fraudulent' in df.columns:
        print("Fraud patterns by temporal features:")
        
        if 'transaction_hour' in df.columns:
            hour_fraud = df.groupby('transaction_hour')['is_fraudulent'].mean()
            print("Fraud rate by hour:")
            print(hour_fraud)
            print()
        
        if 'is_weekend' in df.columns:
            weekend_fraud = df.groupby('is_weekend')['is_fraudulent'].mean()
            print("Fraud rate by weekend:")
            print(weekend_fraud)
            print()
    
    # Show all columns
    print("All columns:")
    print(list(df.columns))
    print()
    
    return df

if __name__ == "__main__":
    test_datetime_features() 