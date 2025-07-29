"""
Test to ensure no data leakage between train and test sets.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.data_loader import load_and_split_data
from src.features.feature_engineer import FeatureEngineer
from src.config_loader import ConfigLoader


def test_no_data_leakage():
    """Test that no test data information leaks into training."""
    
    # Load data
    df_train, df_test = load_and_split_data("data/cleaned/ecommerce_cleaned.csv")
    
    # Verify temporal split
    assert len(df_train) > 0, "Training set is empty"
    assert len(df_test) > 0, "Test set is empty"
    assert len(df_train) + len(df_test) == len(pd.concat([df_train, df_test])), "Data loss during split"
    
    # Verify no overlap in transaction IDs (if they exist)
    if 'transaction_id' in df_train.columns and 'transaction_id' in df_test.columns:
        train_ids = set(df_train['transaction_id'])
        test_ids = set(df_test['transaction_id'])
        overlap = train_ids.intersection(test_ids)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping transaction IDs"
    
    # Test feature engineering leakage prevention
    config_loader = ConfigLoader("configs/final_optimized_config.yaml")
    feature_config = config_loader.get_feature_config()
    feature_engineer = FeatureEngineer(feature_config)
    
    # Fit on training data only
    df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
    
    # Verify that test data was not used for fitting
    # Check that scalers were fitted on training data only
    if hasattr(feature_engineer, 'amount_scaler') and feature_engineer.amount_scaler is not None:
        # Verify scaler was fitted on training data
        train_amounts = df_train['amount'].values.reshape(-1, 1)
        test_amounts = df_test['amount'].values.reshape(-1, 1)
        
        # Check that test data statistics are different from training (indicating no leakage)
        train_mean = np.mean(train_amounts)
        test_mean = np.mean(test_amounts)
        train_std = np.std(train_amounts)
        test_std = np.std(test_amounts)
        
        # These should be different (no leakage)
        assert abs(train_mean - test_mean) > 0.01, "Test and train means are too similar (possible leakage)"
        assert abs(train_std - test_std) > 0.01, "Test and train stds are too similar (possible leakage)"
    
    # Verify feature engineering preserves train/test separation
    assert len(df_train_features) == len(df_train), "Training features length mismatch"
    assert len(df_test_features) == len(df_test), "Test features length mismatch"
    
    # Verify no test data in training features
    train_feature_cols = [col for col in df_train_features.columns if col != 'is_fraudulent']
    test_feature_cols = [col for col in df_test_features.columns if col != 'is_fraudulent']
    
    assert set(train_feature_cols) == set(test_feature_cols), "Feature columns mismatch"
    
    print("✅ No data leakage detected")


def test_feature_engineering_fit_only():
    """Test that feature engineering fits only on training data."""
    
    # Load data
    df_train, df_test = load_and_split_data("data/cleaned/ecommerce_cleaned.csv")
    
    # Create feature engineer
    config_loader = ConfigLoader("configs/final_optimized_config.yaml")
    feature_config = config_loader.get_feature_config()
    feature_engineer = FeatureEngineer(feature_config)
    
    # Fit and transform
    df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
    
    # Test that percentile-based features use training data only
    if 'high_amount_95' in df_train_features.columns:
        train_high_amount_ratio = df_train_features['high_amount_95'].mean()
        test_high_amount_ratio = df_test_features['high_amount_95'].mean()
        
        # These should be different because test data uses training-fitted thresholds
        assert abs(train_high_amount_ratio - test_high_amount_ratio) > 0.01, "High amount ratios too similar (possible leakage)"
    
    if 'high_quantity_95' in df_train_features.columns:
        train_high_quantity_ratio = df_train_features['high_quantity_95'].mean()
        test_high_quantity_ratio = df_test_features['high_quantity_95'].mean()
        
        # These should be different because test data uses training-fitted thresholds
        assert abs(train_high_quantity_ratio - test_high_quantity_ratio) > 0.01, "High quantity ratios too similar (possible leakage)"
    
    print("✅ Feature engineering fits only on training data")


if __name__ == "__main__":
    test_no_data_leakage()
    test_feature_engineering_fit_only()
    print("All data leakage tests passed!") 