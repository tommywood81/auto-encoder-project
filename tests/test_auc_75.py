"""
Test to ensure model achieves AUC ROC >= 0.75.
Config-driven implementation with reproducibility.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config_loader import ConfigLoader
from src.features.feature_engineer import FeatureEngineer
from src.models.autoencoder import FraudAutoencoder
from src.utils.data_loader import load_and_split_data


def run_auc_test():
    """Run the AUC 0.75 test with config-driven approach."""
    
    print("=" * 60)
    print("AUC 0.75 TEST")
    print("=" * 60)
    
    try:
        # Load configuration
        config_loader = ConfigLoader("configs/final_optimized_config.yaml")
        print(f"✅ Configuration loaded from: configs/final_optimized_config.yaml")
        
        # Load and split data
        df_train, df_test = load_and_split_data("data/cleaned/ecommerce_cleaned.csv")
        print(f"✅ Data loaded: {len(df_train)} train, {len(df_test)} test samples")
        
        # Feature engineering
        feature_config = config_loader.get_feature_config()
        feature_engineer = FeatureEngineer(feature_config)
        df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
        print(f"✅ Feature engineering completed: {len(df_train_features.columns)} features")
        
        # Get model and training configuration
        model_config = config_loader.get_model_config()
        training_config = config_loader.get_training_config()
        
        # Combine configurations
        combined_config = {
            **model_config,
            **training_config,
            'threshold_percentile': feature_config.get('threshold_percentile', 95)
        }
        
        print(f"✅ Model config: latent_dim={model_config['latent_dim']}, "
              f"hidden_dims={model_config['hidden_dims']}, "
              f"dropout_rate={model_config['dropout_rate']}")
        print(f"✅ Training config: batch_size={training_config['batch_size']}, "
              f"learning_rate={training_config['learning_rate']}, "
              f"epochs={training_config['epochs']}")
        
        # Initialize and train model
        autoencoder = FraudAutoencoder(combined_config)
        X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
        y_train = df_train_features['is_fraudulent'].values
        y_test = df_test_features['is_fraudulent'].values
        
        print(f"✅ Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        # Train model
        results = autoencoder.train(X_train, X_test, y_train, y_test)
        
        # Test AUC requirement
        test_auc = results['test_auc']
        threshold = results['threshold']
        
        print(f"✅ Training completed")
        print(f"   Test AUC: {test_auc:.4f}")
        print(f"   Threshold: {threshold:.6f}")
        print(f"   Target AUC: 0.7500")
        
        # Assert AUC requirement
        assert test_auc >= 0.75, f"AUC test FAILED: {test_auc:.4f} < 0.75"
        
        print(f"✅ AUC test PASSED: {test_auc:.4f} >= 0.75")
        
        # Additional quality checks
        test_feature_engineering_quality(df_train_features, df_test_features)
        test_model_quality(autoencoder, X_test, y_test)
        
        return True
        
    except Exception as e:
        print(f"❌ AUC test FAILED with error: {e}")
        return False


def test_feature_engineering_quality(df_train_features, df_test_features):
    """Test feature engineering quality."""
    
    # Check feature diversity
    feature_cols = [col for col in df_train_features.columns if col != 'is_fraudulent']
    
    # Check for constant features
    for col in feature_cols:
        train_std = df_train_features[col].std()
        test_std = df_test_features[col].std()
        
        assert train_std > 0, f"Constant feature detected in training: {col}"
        assert test_std > 0, f"Constant feature detected in test: {col}"
    
    # Check for reasonable feature ranges
    for col in feature_cols:
        train_min, train_max = df_train_features[col].min(), df_train_features[col].max()
        test_min, test_max = df_test_features[col].min(), df_test_features[col].max()
        
        # Features should have reasonable ranges (not extreme values)
        assert abs(train_min) < 1e6, f"Extreme minimum value in training feature {col}: {train_min}"
        assert abs(train_max) < 1e6, f"Extreme maximum value in training feature {col}: {train_max}"
        assert abs(test_min) < 1e6, f"Extreme minimum value in test feature {col}: {test_min}"
        assert abs(test_max) < 1e6, f"Extreme maximum value in test feature {col}: {test_max}"
    
    print(f"✅ Feature engineering quality checks passed")


def test_model_quality(autoencoder, X_test, y_test):
    """Test model quality metrics."""
    
    # Test predictions
    predictions = autoencoder.predict(X_test)
    anomaly_scores = autoencoder.predict_anomaly_scores(X_test)
    
    # Check prediction distribution
    fraud_predictions = predictions.sum()
    fraud_actual = y_test.sum()
    
    # Predictions should be reasonable (not all 0 or all 1)
    assert 0 < fraud_predictions < len(predictions), f"Unreasonable prediction distribution: {fraud_predictions}/{len(predictions)}"
    
    # Anomaly scores should have reasonable range
    score_min, score_max = anomaly_scores.min(), anomaly_scores.max()
    score_mean, score_std = anomaly_scores.mean(), anomaly_scores.std()
    
    assert score_std > 0, "Anomaly scores have no variance"
    assert score_min >= 0, "Negative anomaly scores detected"
    
    # Check threshold is reasonable
    threshold = autoencoder.threshold
    assert threshold > 0, "Threshold should be positive"
    assert threshold < score_max, "Threshold should be less than maximum score"
    
    print(f"✅ Model quality checks passed")
    print(f"   Predictions: {fraud_predictions}/{len(predictions)} ({fraud_predictions/len(predictions):.2%})")
    print(f"   Actual fraud: {fraud_actual}/{len(y_test)} ({fraud_actual/len(y_test):.2%})")
    print(f"   Anomaly scores: {score_min:.4f} to {score_max:.4f} (mean: {score_mean:.4f}, std: {score_std:.4f})")
    print(f"   Threshold: {threshold:.6f}")


def test_config_driven_approach():
    """Test that the system works with config-driven approach."""
    
    # Test config loading
    config_loader = ConfigLoader("configs/final_optimized_config.yaml")
    
    # Test config sections
    model_config = config_loader.get_model_config()
    training_config = config_loader.get_training_config()
    feature_config = config_loader.get_feature_config()
    
    assert 'latent_dim' in model_config, "Model config missing latent_dim"
    assert 'hidden_dims' in model_config, "Model config missing hidden_dims"
    assert 'batch_size' in training_config, "Training config missing batch_size"
    assert 'learning_rate' in training_config, "Training config missing learning_rate"
    
    print("✅ Config-driven approach test passed")


def test_reproducibility():
    """Test that results are reproducible with same config."""
    
    # Load data
    df_train, df_test = load_and_split_data("data/cleaned/ecommerce_cleaned.csv")
    
    # Load config
    config_loader = ConfigLoader("configs/final_optimized_config.yaml")
    feature_config = config_loader.get_feature_config()
    model_config = config_loader.get_model_config()
    training_config = config_loader.get_training_config()
    
    # Combine configurations
    combined_config = {
        **model_config,
        **training_config,
        'threshold_percentile': feature_config.get('threshold_percentile', 95)
    }
    
    # Feature engineering
    feature_engineer = FeatureEngineer(feature_config)
    df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
    
    # Train two models with same config
    autoencoder1 = FraudAutoencoder(combined_config)
    autoencoder2 = FraudAutoencoder(combined_config)
    
    X_train, X_test = autoencoder1.prepare_data(df_train_features, df_test_features)
    y_train = df_train_features['is_fraudulent'].values
    y_test = df_test_features['is_fraudulent'].values
    
    results1 = autoencoder1.train(X_train, X_test, y_train, y_test)
    results2 = autoencoder2.train(X_train, X_test, y_train, y_test)
    
    # Results should be very close (allowing for small differences)
    assert abs(results1['test_auc'] - results2['test_auc']) < 1e-3, "Results not reproducible"
    assert abs(results1['threshold'] - results2['threshold']) < 1e-3, "Threshold not reproducible"
    
    print("✅ Reproducibility test passed")


if __name__ == "__main__":
    # Run all tests
    print("Running comprehensive AUC 0.75 test suite...")
    
    test_config_driven_approach()
    test_reproducibility()
    success = run_auc_test()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("✅ Model achieves AUC ROC >= 0.75")
        print("✅ Config-driven approach working")
        print("✅ Results are reproducible")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ TESTS FAILED!")
        print("❌ Model does not achieve AUC ROC >= 0.75")
        print("=" * 60)
        exit(1) 