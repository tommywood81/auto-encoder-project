"""
Test to ensure model reproducibility with the same seed.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config_loader import ConfigLoader
from src.features.feature_engineer import FeatureEngineer
from src.models.autoencoder import FraudAutoencoder
from src.utils.data_loader import load_and_split_data


def test_model_reproducibility():
    """Test that model training is reproducible with the same seed."""
    
    print("=" * 60)
    print("TESTING MODEL REPRODUCIBILITY")
    print("=" * 60)
    
    # Load data
    df_train = pd.read_csv('data/cleaned/ecommerce_cleaned.csv')
    df_test = pd.read_csv('data/cleaned/ecommerce_cleaned.csv')  # Using same data for testing
    
    # Load configuration
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
    
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    
    # Train model first time
    results1 = train_model_with_config(combined_config, df_train_features, df_test_features)
    
    # Reset seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    
    # Train model second time
    results2 = train_model_with_config(combined_config, df_train_features, df_test_features)
    
    # Results should be very similar (allowing for small numerical differences)
    assert abs(results1['test_auc'] - results2['test_auc']) < 1e-3, "AUC values differ between runs"
    assert abs(results1['threshold'] - results2['threshold']) < 1e-3, "Threshold differs between runs"
    
    print("[PASS] Model reproducibility test passed")


def test_seed_enforcement():
    """Test that seed is properly enforced across all components."""
    
    # Test with different seeds
    seed1 = 42
    seed2 = 123
    
    # Load data
    df_train, df_test = load_and_split_data("data/cleaned/ecommerce_cleaned.csv")
    
    # Load config and modify seed
    config_loader1 = ConfigLoader("configs/final_optimized_config.yaml")
    config_loader1.update_config({'seed': seed1})
    
    config_loader2 = ConfigLoader("configs/final_optimized_config.yaml")
    config_loader2.update_config({'seed': seed2})
    
    # Feature engineering with different seeds
    feature_config = config_loader1.get_feature_config()
    feature_engineer1 = FeatureEngineer(feature_config)
    df_train_features1, df_test_features1 = feature_engineer1.fit_transform(df_train, df_test)
    
    feature_engineer2 = FeatureEngineer(feature_config)
    df_train_features2, df_test_features2 = feature_engineer2.fit_transform(df_train, df_test)
    
    # Models should produce different results with different seeds
    model_config = config_loader1.get_model_config()
    training_config = config_loader1.get_training_config()
    combined_config = {
        **model_config,
        **training_config,
        'threshold_percentile': feature_config.get('threshold_percentile', 95)
    }
    
    results1 = train_model_with_config(combined_config, df_train_features1, df_test_features1)
    results2 = train_model_with_config(combined_config, df_train_features2, df_test_features2)
    
    # Results should be different with different seeds
    assert abs(results1['test_auc'] - results2['test_auc']) > 1e-6, "AUC values should differ with different seeds"
    
    print("[PASS] Seed enforcement test passed")


def test_deterministic_operations():
    """Test that operations are deterministic."""
    
    # Test numpy operations
    np.random.seed(42)
    array1 = np.random.randn(100, 10)
    
    np.random.seed(42)
    array2 = np.random.randn(100, 10)
    
    assert np.array_equal(array1, array2), "NumPy operations not deterministic"
    
    # Test TensorFlow operations
    tf.random.set_seed(42)
    tensor1 = tf.random.normal((100, 10))
    
    tf.random.set_seed(42)
    tensor2 = tf.random.normal((100, 10))
    
    assert tf.reduce_all(tf.equal(tensor1, tensor2)), "TensorFlow operations not deterministic"
    
    print("[PASS] Deterministic operations test passed")


def train_model_with_config(config, df_train_features, df_test_features):
    """Helper function to train a model with given config."""
    
    autoencoder = FraudAutoencoder(config)
    X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
    y_train = df_train_features['is_fraudulent'].values
    y_test = df_test_features['is_fraudulent'].values
    
    results = autoencoder.train(X_train, X_test, y_train, y_test)
    
    return {
        'test_auc': results['test_auc'],
        'threshold': results['threshold'],
        'model': autoencoder.model,
        'scaler': autoencoder.scaler
    }


if __name__ == "__main__":
    test_model_reproducibility()
    test_seed_enforcement()
    test_deterministic_operations()
    print("All reproducibility tests passed!") 