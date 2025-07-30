"""
Test to ensure prediction consistency with saved models.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config_loader import ConfigLoader
from src.features.feature_engineer import FeatureEngineer
from src.models.autoencoder import FraudAutoencoder
from src.utils.data_loader import load_and_split_data


def test_model_save_load_consistency():
    """Test that saved and loaded models produce identical predictions."""
    
    # Load configuration
    config_loader = ConfigLoader("tests/config/tests_config.yaml")
    test_settings = config_loader.config.get('test_settings', {})
    data_path = test_settings.get('data_path', "data/cleaned/creditcard_cleaned.csv")
    
    # Load data
    df_train, df_test = load_and_split_data(data_path)
    
    # Get configurations
    feature_config = config_loader.get_feature_config()
    model_config = config_loader.get_model_config()
    training_config = config_loader.get_training_config()
    
    # Combine configurations with test overrides
    combined_config = {
        **model_config,
        **training_config,
        'threshold_percentile': feature_config.get('threshold_percentile', 95),
        'epochs': test_settings.get('epochs', 3),  # Use test epochs
        'patience': test_settings.get('patience', 10)  # Use test patience
    }
    
    # Feature engineering
    feature_engineer = FeatureEngineer(feature_config)
    df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
    
    # Train model
    autoencoder = FraudAutoencoder(combined_config)
    X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
    y_train = df_train_features['is_fraudulent'].values
    y_test = df_test_features['is_fraudulent'].values
    
    results = autoencoder.train(X_train, X_test, y_train, y_test)
    
    # Get predictions from original model
    X_test_numeric = df_test_features.select_dtypes(include=[np.number])
    if 'is_fraudulent' in X_test_numeric.columns:
        X_test_numeric = X_test_numeric.drop(columns=['is_fraudulent'])
    
    original_predictions = autoencoder.predict(X_test_numeric.values)
    original_scores = autoencoder.predict_anomaly_scores(X_test_numeric.values)
    
    # Save model to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "test_model")
        
        # Save model
        autoencoder.save_model(model_path)
        feature_engineer.save_fitted_objects(f"{model_path}_features.pkl")
        
        # Load model
        loaded_autoencoder = FraudAutoencoder({})
        loaded_autoencoder.load_model(model_path)
        
        loaded_feature_engineer = FeatureEngineer({})
        loaded_feature_engineer.load_fitted_objects(f"{model_path}_features.pkl")
        
        # Transform test data with loaded feature engineer
        df_test_features_loaded = loaded_feature_engineer.transform(df_test)
        X_test_numeric_loaded = df_test_features_loaded.select_dtypes(include=[np.number])
        if 'is_fraudulent' in X_test_numeric_loaded.columns:
            X_test_numeric_loaded = X_test_numeric_loaded.drop(columns=['is_fraudulent'])
        
        # Get predictions from loaded model
        loaded_predictions = loaded_autoencoder.predict(X_test_numeric_loaded.values)
        loaded_scores = loaded_autoencoder.predict_anomaly_scores(X_test_numeric_loaded.values)
        
        # Compare predictions
        assert np.array_equal(original_predictions, loaded_predictions), "Predictions differ after save/load"
        
        # Compare anomaly scores (allowing for small floating point differences)
        assert np.allclose(original_scores, loaded_scores, rtol=1e-5, atol=1e-5), "Anomaly scores differ after save/load"
        
        # Compare thresholds
        assert abs(autoencoder.threshold - loaded_autoencoder.threshold) < 1e-6, "Threshold differs after save/load"
        
        print("[PASS] Model save/load consistency test passed")


def test_feature_engineer_save_load_consistency():
    """Test that saved and loaded feature engineers produce identical features."""
    
    # Load data
    df_train, df_test = load_and_split_data("data/cleaned/ecommerce_cleaned.csv")
    
    # Load config
    config_loader = ConfigLoader("tests/config/tests_config.yaml")
    feature_config = config_loader.get_feature_config()
    
    # Create and fit feature engineer
    feature_engineer = FeatureEngineer(feature_config)
    df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
    
    # Save feature engineer to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        feature_path = os.path.join(temp_dir, "test_features.pkl")
        
        # Save feature engineer
        feature_engineer.save_fitted_objects(feature_path)
        
        # Load feature engineer
        loaded_feature_engineer = FeatureEngineer({})
        loaded_feature_engineer.load_fitted_objects(feature_path)
        
        # Transform test data with both feature engineers
        df_test_features_original = feature_engineer.transform(df_test)
        df_test_features_loaded = loaded_feature_engineer.transform(df_test)
        
        # Compare features
        feature_cols = [col for col in df_test_features_original.columns if col != 'is_fraudulent']
        
        for col in feature_cols:
            original_values = df_test_features_original[col].values
            loaded_values = df_test_features_loaded[col].values
            
            # Compare values (allowing for small floating point differences)
            assert np.allclose(original_values, loaded_values, rtol=1e-5, atol=1e-5), f"Feature {col} differs after save/load"
        
        print("[PASS] Feature engineer save/load consistency test passed")


def test_prediction_stability():
    """Test that predictions are stable across multiple calls."""
    
    # Load data
    df_train, df_test = load_and_split_data("data/cleaned/ecommerce_cleaned.csv")
    
    # Load config
    config_loader = ConfigLoader("tests/config/tests_config.yaml")
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
    
    # Train model
    autoencoder = FraudAutoencoder(combined_config)
    X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
    y_train = df_train_features['is_fraudulent'].values
    y_test = df_test_features['is_fraudulent'].values
    
    autoencoder.train(X_train, X_test, y_train, y_test)
    
    # Get test data
    X_test_numeric = df_test_features.select_dtypes(include=[np.number])
    if 'is_fraudulent' in X_test_numeric.columns:
        X_test_numeric = X_test_numeric.drop(columns=['is_fraudulent'])
    
    # Make multiple predictions
    predictions1 = autoencoder.predict(X_test_numeric.values)
    predictions2 = autoencoder.predict(X_test_numeric.values)
    predictions3 = autoencoder.predict(X_test_numeric.values)
    
    scores1 = autoencoder.predict_anomaly_scores(X_test_numeric.values)
    scores2 = autoencoder.predict_anomaly_scores(X_test_numeric.values)
    scores3 = autoencoder.predict_anomaly_scores(X_test_numeric.values)
    
    # All predictions should be identical
    assert np.array_equal(predictions1, predictions2), "Predictions not stable across calls"
    assert np.array_equal(predictions2, predictions3), "Predictions not stable across calls"
    
    # All scores should be identical
    assert np.allclose(scores1, scores2, rtol=1e-5, atol=1e-5), "Scores not stable across calls"
    assert np.allclose(scores2, scores3, rtol=1e-5, atol=1e-5), "Scores not stable across calls"
    
    print("[PASS] Prediction stability test passed")


def test_model_persistence_integrity():
    """Test that model files are created and contain expected data."""
    
    # Load data
    df_train, df_test = load_and_split_data("data/cleaned/ecommerce_cleaned.csv")
    
    # Load config
    config_loader = ConfigLoader("tests/config/tests_config.yaml")
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
    
    # Train model
    autoencoder = FraudAutoencoder(combined_config)
    X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
    y_train = df_train_features['is_fraudulent'].values
    y_test = df_test_features['is_fraudulent'].values
    
    autoencoder.train(X_train, X_test, y_train, y_test)
    
    # Save model to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "test_model")
        
        # Save model
        autoencoder.save_model(model_path)
        feature_engineer.save_fitted_objects(f"{model_path}_features.pkl")
        
        # Check that files exist
        model_file = f"{model_path}.keras"
        scaler_file = f"{model_path}_scaler.pkl"
        threshold_file = f"{model_path}_threshold.pkl"
        features_file = f"{model_path}_features.pkl"
        
        assert os.path.exists(model_file), f"Model file not created: {model_file}"
        assert os.path.exists(scaler_file), f"Scaler file not created: {scaler_file}"
        assert os.path.exists(threshold_file), f"Threshold file not created: {threshold_file}"
        assert os.path.exists(features_file), f"Features file not created: {features_file}"
        
        # Check file sizes (should not be empty)
        assert os.path.getsize(model_file) > 0, "Model file is empty"
        assert os.path.getsize(scaler_file) > 0, "Scaler file is empty"
        assert os.path.getsize(threshold_file) > 0, "Threshold file is empty"
        assert os.path.getsize(features_file) > 0, "Features file is empty"
        
        print("[PASS] Model persistence integrity test passed")


if __name__ == "__main__":
    test_model_save_load_consistency()
    test_feature_engineer_save_load_consistency()
    test_prediction_stability()
    test_model_persistence_integrity()
    print("All prediction consistency tests passed!") 