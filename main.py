#!/usr/bin/env python3
"""
Fraud Detection Pipeline - Production Ready
Config-driven autoencoder for fraud detection with optimized performance.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import ConfigLoader
from src.features.feature_engineer import FeatureEngineer
from src.models.autoencoder import FraudAutoencoder
from src.utils.data_loader import load_and_split_data, clean_data, save_cleaned_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment and validate paths."""
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('data/cleaned', exist_ok=True)


def validate_data_paths(data_path, raw_data_path):
    """Validate and prepare data paths."""
    if not os.path.exists(data_path):
        logger.info(f"Cleaned data not found at {data_path}")
        if os.path.exists(raw_data_path):
            logger.info(f"Cleaning raw data from {raw_data_path}")
            import pandas as pd
            df_raw = pd.read_csv(raw_data_path)
            df_cleaned = clean_data(df_raw)
            save_cleaned_data(df_cleaned, data_path)
            logger.info(f"Cleaned data saved to {data_path}")
        else:
            raise FileNotFoundError(f"Neither cleaned data ({data_path}) nor raw data ({raw_data_path}) found")
    
    return data_path


def train_model(config_loader, df_train_features, df_test_features, model_path):
    """Train the fraud detection model."""
    logger.info("Training fraud detection model...")
    
    # Get configurations
    model_config = config_loader.get_model_config()
    training_config = config_loader.get_training_config()
    feature_config = config_loader.get_feature_config()
    
    # Combine configurations
    combined_config = {
        **model_config,
        **training_config,
        'threshold_percentile': feature_config.get('threshold_percentile', 95)
    }
    
    # Initialize and train model
    autoencoder = FraudAutoencoder(combined_config)
    X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
    y_train = df_train_features['is_fraudulent'].values
    y_test = df_test_features['is_fraudulent'].values
    
    # Train model
    results = autoencoder.train(X_train, X_test, y_train, y_test)
    
    # Save model and feature engineer
    autoencoder.save_model(model_path)
    feature_engineer = FeatureEngineer(feature_config)
    feature_engineer.save_fitted_objects(f"{model_path}_features.pkl")
    
    logger.info(f"Model training completed - Test AUC: {results['test_auc']:.4f}")
    return results, autoencoder


def make_predictions(model_path, df_test, df_test_features):
    """Make predictions using trained model."""
    logger.info("Loading model and making predictions...")
    
    # Load model and feature engineer
    autoencoder = FraudAutoencoder({})
    autoencoder.load_model(model_path)
    
    feature_engineer = FeatureEngineer({})
    feature_engineer.load_fitted_objects(f"{model_path}_features.pkl")
    
    # Transform test data if needed
    if df_test_features is None:
        df_test_features = feature_engineer.transform(df_test)
    
    # Prepare data for prediction
    import numpy as np
    X_test_numeric = df_test_features.select_dtypes(include=[np.number])
    if 'is_fraudulent' in X_test_numeric.columns:
        X_test_numeric = X_test_numeric.drop(columns=['is_fraudulent'])
    
    # Make predictions
    anomaly_scores = autoencoder.predict_anomaly_scores(X_test_numeric.values)
    predictions = autoencoder.predict(X_test_numeric.values)
    
    # Save predictions
    results_df = df_test_features.copy()
    results_df['anomaly_score'] = anomaly_scores
    results_df['predicted_fraud'] = predictions
    
    results_df.to_csv('predictions/predictions.csv', index=False)
    
    logger.info(f"Predictions saved - Fraud detected: {predictions.sum()}")
    return results_df


def run_auc_test():
    """Run AUC test to validate model performance."""
    logger.info("Running AUC performance test...")
    
    from tests.test_auc_75 import run_auc_test as run_test
    result = run_test()
    
    if result['success']:
        logger.info(f"[PASS] AUC test PASSED: {result['test_auc']:.4f} >= 0.80")
        return True
    else:
        logger.error(f"[FAIL] AUC test FAILED: {result['test_auc']:.4f} < 0.80")
        return False


def main():
    """Main fraud detection pipeline."""
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/final_optimized_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/cleaned/creditcard_cleaned.csv",
        help="Path to the cleaned data file"
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/raw/creditcard.csv",
        help="Path to the raw data file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "test"],
        default="train",
        help="Pipeline mode"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/fraud_autoencoder",
        help="Path to save/load model"
    )
    
    args = parser.parse_args()
    
    # Setup and validation
    setup_environment()
    
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION PIPELINE")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config_loader = ConfigLoader(args.config)
        logger.info("Configuration loaded successfully")
        
        # Validate and prepare data
        data_path = validate_data_paths(args.data_path, args.raw_data_path)
        
        # Load and split data
        df_train, df_test = load_and_split_data(data_path)
        logger.info(f"Data loaded: {len(df_train)} train, {len(df_test)} test samples")
        
        # Feature engineering
        feature_config = config_loader.get_feature_config()
        feature_engineer = FeatureEngineer(feature_config)
        df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
        logger.info(f"Feature engineering completed: {len(df_train_features.columns)} features")
        
        # Execute mode-specific operations
        if args.mode == "train":
            results, _ = train_model(config_loader, df_train_features, df_test_features, args.model_path)
            logger.info("Training completed successfully")
            
        elif args.mode == "predict":
            make_predictions(args.model_path, df_test, df_test_features)
            logger.info("Predictions completed successfully")
            
        elif args.mode == "test":
            success = run_auc_test()
            if not success:
                sys.exit(1)
            logger.info("Testing completed successfully")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    main() 