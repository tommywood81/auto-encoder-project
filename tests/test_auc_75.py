"""
Test for AUC ROC >= 0.75
This test ensures our fraud detection model meets the performance requirement.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.features.feature_engineer import FeatureEngineer
from src.models.autoencoder import FraudAutoencoder
from src.utils.data_loader import load_and_split_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAUC75(unittest.TestCase):
    """Test that the model achieves AUC ROC >= 0.75."""
    
    def setUp(self):
        """Set up test data and model."""
        self.data_path = "data/cleaned/ecommerce_cleaned.csv"
        self.min_auc = 0.75
        
        # Check if data exists
        if not os.path.exists(self.data_path):
            self.skipTest(f"Data file not found: {self.data_path}")
    
    def test_auc_75(self):
        """Test that the model achieves at least 0.75 AUC ROC."""
        
        logger.info("=" * 60)
        logger.info("TESTING AUC ROC >= 0.75")
        logger.info("=" * 60)
        
        # Load and split data
        logger.info("Loading and splitting data...")
        df_train, df_test = load_and_split_data(self.data_path)
        
        # Feature engineering
        logger.info("Performing feature engineering...")
        feature_engineer = FeatureEngineer({})
        df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
        
        # Model configuration (optimized for performance and generalization)
        model_config = {
            'latent_dim': 16,
            'hidden_dims': [128, 64, 32],
            'learning_rate': 0.001,
            'batch_size': 256,
            'epochs': 50,
            'dropout_rate': 0.2,
            'threshold_percentile': 90,
            'early_stopping': True,
            'patience': 10,
            'reduce_lr': True
        }
        
        # Initialize and train model
        logger.info("Training autoencoder...")
        autoencoder = FraudAutoencoder(model_config)
        
        # Prepare data
        X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
        y_train = df_train_features['is_fraudulent'].values
        y_test = df_test_features['is_fraudulent'].values
        
        # Train model
        results = autoencoder.train(X_train, X_test, y_train, y_test)
        
        # Get test AUC
        test_auc = results['test_auc']
        
        logger.info(f"Test AUC achieved: {test_auc:.4f}")
        logger.info(f"Minimum required AUC: {self.min_auc:.4f}")
        
        # Assert AUC requirement
        self.assertGreaterEqual(
            test_auc, 
            self.min_auc, 
            f"Model AUC ({test_auc:.4f}) is below required threshold ({self.min_auc:.4f})"
        )
        
        logger.info("✅ AUC test PASSED!")
        
        # Additional assertions for model quality
        self.assertIsNotNone(autoencoder.model, "Model should be trained")
        self.assertIsNotNone(autoencoder.threshold, "Threshold should be calculated")
        self.assertTrue(autoencoder.is_fitted, "Model should be fitted")
        
        # Test prediction functionality
        logger.info("Testing prediction functionality...")
        anomaly_scores = autoencoder.predict_anomaly_scores(X_test)
        predictions = autoencoder.predict(X_test)
        
        self.assertEqual(len(anomaly_scores), len(y_test), "Anomaly scores should match test set size")
        self.assertEqual(len(predictions), len(y_test), "Predictions should match test set size")
        
        # Test that predictions are binary
        unique_predictions = np.unique(predictions)
        self.assertTrue(all(pred in [0, 1] for pred in unique_predictions), "Predictions should be binary")
        
        logger.info("✅ All tests PASSED!")
    
    def test_feature_engineering_quality(self):
        """Test that feature engineering produces good features."""
        
        logger.info("Testing feature engineering quality...")
        
        # Load data
        df_train, df_test = load_and_split_data(self.data_path)
        
        # Feature engineering
        feature_engineer = FeatureEngineer({})
        df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
        
        # Check feature count
        expected_features = [
            'amount_log', 'amount_per_item', 'amount_scaled',
            'high_amount_95', 'high_amount_99',
            'hour', 'is_late_night', 'is_business_hours',
            'age_group_encoded', 'account_age_days_log',
            'new_account', 'established_account', 'location_freq',
            'payment_method_encoded', 'product_category_encoded', 'device_used_encoded',
            'amount_quantity_interaction', 'age_account_interaction', 'amount_hour_interaction',
            'high_quantity', 'young_customer', 'high_risk_combination', 'new_account_high_amount'
        ]
        
        # Check that all expected features are present
        for feature in expected_features:
            self.assertIn(feature, df_train_features.columns, f"Feature {feature} should be present")
        
        # Check that features are numeric
        numeric_features = df_train_features.select_dtypes(include=[np.number]).columns
        self.assertGreater(len(numeric_features), 20, "Should have at least 20 numeric features")
        
        # Check for no infinite values
        self.assertFalse(np.any(np.isinf(df_train_features.select_dtypes(include=[np.number]))), 
                        "Features should not contain infinite values")
        
        # Check for reasonable number of NaN values (should be minimal)
        nan_count = df_train_features.isnull().sum().sum()
        self.assertLess(nan_count, len(df_train_features) * 0.01, 
                       "Should have less than 1% NaN values")
        
        logger.info("✅ Feature engineering quality test PASSED!")
    
    def test_model_persistence(self):
        """Test that the model can be saved and loaded."""
        
        logger.info("Testing model persistence...")
        
        # Load data
        df_train, df_test = load_and_split_data(self.data_path)
        
        # Feature engineering
        feature_engineer = FeatureEngineer({})
        df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
        
        # Train model
        model_config = {
            'latent_dim': 16,
            'hidden_dims': [64, 32],
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,  # Short training for test
            'dropout_rate': 0.1,
            'threshold_percentile': 95
        }
        
        autoencoder = FraudAutoencoder(model_config)
        X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
        y_train = df_train_features['is_fraudulent'].values
        y_test = df_test_features['is_fraudulent'].values
        
        autoencoder.train(X_train, X_test, y_train, y_test)
        
        # Save model
        test_model_path = "test_model"
        autoencoder.save_model(test_model_path)
        
        # Load model
        new_autoencoder = FraudAutoencoder(model_config)
        new_autoencoder.load_model(test_model_path)
        
        # Test that loaded model produces same predictions
        original_scores = autoencoder.predict_anomaly_scores(X_test)
        loaded_scores = new_autoencoder.predict_anomaly_scores(X_test)
        
        np.testing.assert_array_almost_equal(original_scores, loaded_scores, decimal=6)
        
        # Clean up
        for ext in ['_model.h5', '_scaler.pkl', '_threshold.pkl']:
            if os.path.exists(test_model_path + ext):
                os.remove(test_model_path + ext)
        
        logger.info("✅ Model persistence test PASSED!")


def run_auc_test():
    """Run the AUC test and return results."""
    
    logger.info("Running AUC 0.75 test...")
    
    # Create test instance
    test = TestAUC75()
    test.setUp()
    
    try:
        # Run the main AUC test
        test.test_auc_75()
        logger.info("🎉 AUC 0.75 test PASSED!")
        return True
    except Exception as e:
        logger.error(f"❌ AUC 0.75 test FAILED: {e}")
        return False


if __name__ == "__main__":
    # Run the test
    success = run_auc_test()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Model achieves AUC ROC >= 0.75")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ TESTS FAILED!")
        print("❌ Model does not achieve AUC ROC >= 0.75")
        print("=" * 60)
        sys.exit(1) 