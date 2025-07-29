#!/usr/bin/env python3
"""
Main Pipeline for Fraud Detection
Config-driven, production-grade implementation
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the main fraud detection pipeline."""

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
        default="data/cleaned/ecommerce_cleaned.csv",
        help="Path to the cleaned data file"
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/raw/Fraudulent_E-Commerce_Transaction_Data_2.csv",
        help="Path to the raw data file (if cleaned data doesn't exist)"
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

    print("=" * 80)
    print("FRAUD DETECTION PIPELINE")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}")
    print("=" * 80)

    # Load configuration
    try:
        config_loader = ConfigLoader(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Check if cleaned data exists, if not clean raw data
    if not os.path.exists(args.data_path):
        logger.info(f"Cleaned data not found at {args.data_path}")
        if os.path.exists(args.raw_data_path):
            logger.info(f"Cleaning raw data from {args.raw_data_path}")
            df_raw = pd.read_csv(args.raw_data_path)
            df_cleaned = clean_data(df_raw)
            save_cleaned_data(df_cleaned, args.data_path)
            logger.info(f"Cleaned data saved to {args.data_path}")
        else:
            raise FileNotFoundError(f"Neither cleaned nor raw data found")

    # Load and split data
    logger.info("Loading and splitting data...")
    df_train, df_test = load_and_split_data(args.data_path)

    # Feature engineering
    logger.info("Performing feature engineering...")
    feature_config = config_loader.get_feature_config()
    feature_engineer = FeatureEngineer(feature_config)
    df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)

    if args.mode == "train":
        # Train model
        logger.info("Training model...")

        # Get model and training configuration
        model_config = config_loader.get_model_config()
        training_config = config_loader.get_training_config()
        
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

        # Save model
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        autoencoder.save_model(args.model_path)

        # Save feature engineer
        feature_engineer.save_fitted_objects(f"{args.model_path}_features.pkl")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Test AUC: {results['test_auc']:.4f}")
        print(f"Threshold: {results['threshold']:.6f}")
        print(f"Model saved to: {args.model_path}")
        print(f"Config used: {args.config}")

    elif args.mode == "predict":
        # Load model and make predictions
        logger.info("Loading model and making predictions...")

        # Load model
        autoencoder = FraudAutoencoder({})
        autoencoder.load_model(args.model_path)

        # Load feature engineer
        feature_engineer = FeatureEngineer({})
        feature_engineer.load_fitted_objects(f"{args.model_path}_features.pkl")

        # Transform test data
        df_test_features = feature_engineer.transform(df_test)

        # Make predictions
        X_test_numeric = df_test_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in X_test_numeric.columns:
            X_test_numeric = X_test_numeric.drop(columns=['is_fraudulent'])

        anomaly_scores = autoencoder.predict_anomaly_scores(X_test_numeric.values)
        predictions = autoencoder.predict(X_test_numeric.values)

        # Save predictions
        results_df = df_test_features.copy()
        results_df['anomaly_score'] = anomaly_scores
        results_df['predicted_fraud'] = predictions

        os.makedirs('predictions', exist_ok=True)
        results_df.to_csv('predictions/predictions.csv', index=False)

        print("\n" + "=" * 60)
        print("PREDICTIONS COMPLETED")
        print("=" * 60)
        print(f"Predictions saved to: predictions/predictions.csv")
        print(f"Fraud predictions: {predictions.sum()}")
        print(f"Average anomaly score: {anomaly_scores.mean():.4f}")

    elif args.mode == "test":
        # Run the AUC test
        logger.info("Running AUC test...")

        from tests.test_auc_75 import run_auc_test

        success = run_auc_test()

        if success:
            print("\n" + "=" * 60)
            print("✅ AUC TEST PASSED!")
            print("✅ Model achieves AUC ROC >= 0.75")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ AUC TEST FAILED!")
            print("❌ Model does not achieve AUC ROC >= 0.75")
            print("=" * 60)
            sys.exit(1)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    main() 