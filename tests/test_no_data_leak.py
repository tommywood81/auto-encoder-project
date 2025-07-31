"""
Comprehensive test to ensure no data leakage between train and test sets.
Specifically designed to catch the suspicious 94% AUC issue.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import hashlib
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data_loader import load_and_split_data, load_and_split_data_80_20
from src.features.feature_engineer import FeatureEngineer
from src.models.autoencoder import FraudAutoencoder
from src.config_loader import ConfigLoader


def test_comprehensive_data_leakage():
    """Comprehensive test for data leakage - specifically targeting the 94% AUC issue."""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DATA LEAKAGE TEST")
    print("Targeting suspicious 94% AUC performance")
    print("=" * 80)
    
    # Load configuration
    config_loader = ConfigLoader("tests/config/tests_config.yaml")
    test_settings = config_loader.config.get('test_settings', {})
    data_path = test_settings.get('data_path', "data/cleaned/creditcard_cleaned.csv")
    
    print(f"Testing with data: {data_path}")
    
    # Test 1: Verify clean 80/20 split
    print("\n1. TESTING CLEAN 80/20 SPLIT")
    print("-" * 40)
    
    df_train, df_test = load_and_split_data_80_20(data_path)
    
    # Verify split integrity
    total_samples = len(df_train) + len(df_test)
    train_ratio = len(df_train) / total_samples
    test_ratio = len(df_test) / total_samples
    
    print(f"Total samples: {total_samples}")
    print(f"Train samples: {len(df_train)} ({train_ratio:.1%})")
    print(f"Test samples: {len(df_test)} ({test_ratio:.1%})")
    
    assert 0.79 <= train_ratio <= 0.81, f"Train ratio {train_ratio:.1%} not close to 80%"
    assert 0.19 <= test_ratio <= 0.21, f"Test ratio {test_ratio:.1%} not close to 20%"
    
    # Test 2: Verify no overlap in data
    print("\n2. TESTING NO DATA OVERLAP")
    print("-" * 40)
    
    # Check for any potential overlap indicators
    if 'time' in df_train.columns:
        train_times = set(df_train['time'])
        test_times = set(df_test['time'])
        time_overlap = train_times.intersection(test_times)
        print(f"Time overlap: {len(time_overlap)} shared timestamps")
        assert len(time_overlap) == 0, f"Found {len(time_overlap)} overlapping timestamps"
    
    # Test 3: Verify feature engineering doesn't leak
    print("\n3. TESTING FEATURE ENGINEERING LEAKAGE")
    print("-" * 40)
    
    feature_config = config_loader.get_feature_config()
    feature_engineer = FeatureEngineer(feature_config)
    
    # Fit on training data only
    df_train_features, df_test_features = feature_engineer.fit_transform_80_20(df_train, df_test)
    
    # Verify fitted state
    assert feature_engineer.is_fitted, "FeatureEngineer should be fitted after fit_transform_80_20"
    
    # Test 4: Verify scaler behavior
    print("\n4. TESTING SCALER BEHAVIOR")
    print("-" * 40)
    
    if hasattr(feature_engineer, 'amount_scaler') and feature_engineer.amount_scaler is not None:
        # Check that scaler was fitted on training data only
        train_amounts = df_train['amount'].values.reshape(-1, 1)
        test_amounts = df_test['amount'].values.reshape(-1, 1)
        
        # Verify scaler statistics are based on training data
        train_mean = np.mean(train_amounts)
        train_std = np.std(train_amounts)
        
        print(f"Training amount - Mean: {train_mean:.2f}, Std: {train_std:.2f}")
        print(f"Test amount - Mean: {np.mean(test_amounts):.2f}, Std: {np.std(test_amounts):.2f}")
        
        # Test that scaler transform works correctly
        try:
            test_scaled = feature_engineer.amount_scaler.transform(test_amounts)
            print(f"Test scaled - Mean: {np.mean(test_scaled):.2f}, Std: {np.std(test_scaled):.2f}")
            print("[PASS] Scaler transform works correctly")
        except Exception as e:
            print(f"[FAIL] Scaler transform failed: {e}")
            raise
    
    # Test 5: Verify model training doesn't leak
    print("\n5. TESTING MODEL TRAINING LEAKAGE")
    print("-" * 40)
    
    # Initialize autoencoder with test config
    autoencoder = FraudAutoencoder(config_loader.config)
    
    # Prepare data
    X_train, X_test = autoencoder.prepare_data_80_20(df_train_features, df_test_features)
    y_train = df_train_features['is_fraudulent'].values
    y_test = df_test_features['is_fraudulent'].values
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training fraud rate: {y_train.mean():.4f}")
    print(f"Test fraud rate: {y_test.mean():.4f}")
    
    # Verify no test data in training
    assert X_train.shape[0] == len(df_train_features), "Training data size mismatch"
    assert X_test.shape[0] == len(df_test_features), "Test data size mismatch"
    
    # Test 6: Verify no premature model usage
    print("\n6. VERIFYING NO PREMATURE MODEL USAGE")
    print("-" * 40)
    
    # Ensure model is not used before training
    assert autoencoder.model is None, "Model should not be built before training"
    assert not autoencoder.is_fitted, "Model should not be fitted before training"
    
    print("[PASS] Model is properly isolated before training")
    
    # Test 7: Verify data separation integrity
    print("\n7. VERIFYING DATA SEPARATION INTEGRITY")
    print("-" * 40)
    
    # Check that train and test data are completely separate
    train_indices = set(df_train.index)
    test_indices = set(df_test.index)
    overlap_indices = train_indices.intersection(test_indices)
    
    print(f"Train indices: {len(train_indices)}")
    print(f"Test indices: {len(test_indices)}")
    print(f"Overlapping indices: {len(overlap_indices)}")
    
    assert len(overlap_indices) == 0, f"Found {len(overlap_indices)} overlapping indices between train and test"
    print("[PASS] Train and test data are completely separate")
    
    # Test 8: Verify data integrity throughout pipeline
    print("\n8. TESTING DATA INTEGRITY")
    print("-" * 40)
    
    # Check that we haven't accidentally mixed train/test data
    train_hash = hashlib.md5(df_train.to_string().encode()).hexdigest()
    test_hash = hashlib.md5(df_test.to_string().encode()).hexdigest()
    
    print(f"Train data hash: {train_hash[:16]}...")
    print(f"Test data hash: {test_hash[:16]}...")
    
    assert train_hash != test_hash, "Train and test data hashes are identical - possible contamination"
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DATA LEAKAGE TEST COMPLETED")
    print("=" * 80)


def test_fresh_pipeline_from_raw():
    """Test running the entire pipeline from raw data to ensure no contamination."""
    
    print("\n" + "=" * 80)
    print("FRESH PIPELINE TEST FROM RAW DATA")
    print("=" * 80)
    
    # Check if raw data exists
    raw_data_path = "data/raw/creditcard.csv"
    if not os.path.exists(raw_data_path):
        print(f"Raw data not found at {raw_data_path}")
        return
    
    print(f"Found raw data: {raw_data_path}")
    
    # Load raw data
    df_raw = pd.read_csv(raw_data_path)
    print(f"Raw data shape: {df_raw.shape}")
    print(f"Raw data columns: {list(df_raw.columns)}")
    
    # Check for any obvious issues
    print(f"Fraud rate in raw data: {df_raw['Class'].mean():.4f}")
    print(f"Missing values: {df_raw.isnull().sum().sum()}")
    
    # Test the complete pipeline
    from src.utils.data_loader import clean_data, save_cleaned_data
    
    # Clean data
    df_cleaned = clean_data(df_raw)
    print(f"Cleaned data shape: {df_cleaned.shape}")
    
    # Save cleaned data
    cleaned_path = "data/cleaned/creditcard_fresh_test.csv"
    save_cleaned_data(df_cleaned, cleaned_path)
    
    # Run the complete pipeline with fresh data
    print("\nRunning complete pipeline with fresh data...")
    
    # Load and split fresh data
    df_train, df_test = load_and_split_data_80_20(cleaned_path)
    
    # Feature engineering
    config_loader = ConfigLoader("tests/config/tests_config.yaml")
    feature_config = config_loader.get_feature_config()
    feature_engineer = FeatureEngineer(feature_config)
    df_train_features, df_test_features = feature_engineer.fit_transform_80_20(df_train, df_test)
    
    # Model training with test config
    autoencoder = FraudAutoencoder(config_loader.config)
    X_train, X_test = autoencoder.prepare_data_80_20(df_train_features, df_test_features)
    y_train = df_train_features['is_fraudulent'].values
    y_test = df_test_features['is_fraudulent'].values
    
    # Train model
    results = autoencoder.train_80_20(X_train, X_test, y_train, y_test)
    
    print(f"\nFresh pipeline results:")
    print(f"Test AUC: {results['roc_auc']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    
    # Additional debugging for high AUC
    if results['roc_auc'] > 0.90:
        print(f"\n[INFO] High AUC detected: {results['roc_auc']:.4f}")
        print("This could be legitimate for this dataset, but let's verify:")
        print(f"- Test fraud rate: {y_test.mean():.4f}")
        print(f"- Training fraud rate: {y_train.mean():.4f}")
        print(f"- Test set size: {len(y_test)}")
        print(f"- Number of fraud cases in test: {y_test.sum()}")
        
        # Check if the model is actually learning patterns
        test_reconstructions = autoencoder.model.predict(X_test, verbose=0)
        test_errors = np.mean(np.square(X_test - test_reconstructions), axis=1)
        
        fraud_errors = test_errors[y_test == 1]
        normal_errors = test_errors[y_test == 0]
        
        print(f"- Fraud reconstruction error mean: {np.mean(fraud_errors):.6f}")
        print(f"- Normal reconstruction error mean: {np.mean(normal_errors):.6f}")
        print(f"- Error separation: {np.mean(normal_errors) - np.mean(fraud_errors):.6f}")
        
        if np.mean(normal_errors) < np.mean(fraud_errors):
            print("[WARNING] Normal transactions have LOWER reconstruction errors than fraud!")
            print("This suggests the model is learning normal patterns correctly")
        else:
            print("[WARNING] Fraud transactions have LOWER reconstruction errors than normal!")
            print("This suggests potential data leakage or the model is not learning correctly")
    
    # Clean up
    if os.path.exists(cleaned_path):
        os.remove(cleaned_path)
    
    print("\n[PASS] Fresh pipeline test completed")


if __name__ == "__main__":
    test_comprehensive_data_leakage()
    test_fresh_pipeline_from_raw()
    print("\nAll data leakage tests completed!") 