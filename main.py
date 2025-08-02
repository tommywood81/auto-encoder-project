#!/usr/bin/env python3
"""
Fraud Detection Pipeline - Production Ready
Config-driven autoencoder for fraud detection with optimized performance.
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import ConfigLoader
from src.features.feature_engineer import FeatureEngineer
from src.models.autoencoder import FraudAutoencoder
from src.utils.data_loader import load_and_split_data, load_and_split_data_80_20, clean_data, save_cleaned_data

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
    os.makedirs('configs', exist_ok=True)


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


def load_best_config(config_path):
    """Load the current best configuration and performance metrics."""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract current best performance
            best_metrics = config.get('best_performance', {})
            best_auc = best_metrics.get('roc_auc', 0.0)
            best_f1 = best_metrics.get('f1_score', 0.0)
            best_recall = best_metrics.get('recall', 0.0)
            best_precision = best_metrics.get('precision', 0.0)
            
            logger.info(f"Loaded current best config - AUC: {best_auc:.4f}, F1: {best_f1:.4f}")
            return config, best_auc, best_f1, best_recall, best_precision
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Starting fresh.")
    
    # Return default config if file doesn't exist or is invalid
    default_config = {
        'seed': 42,
        'model': {
            'latent_dim': 16,
            'hidden_dims': [1024, 512, 256, 128, 64],
            'dropout_rate': 0.2
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 0.005,
            'epochs': 5,
            'early_stopping': True,
            'patience': 25,
            'reduce_lr': True,
            'validation_split': 0.2
        },
        'features': {
            'threshold_percentile': 85,
            'use_amount_features': True,
            'use_temporal_features': True,
            'use_customer_features': True,
            'use_risk_flags': True
        },
        'best_performance': {
            'roc_auc': 0.0,
            'f1_score': 0.0,
            'recall': 0.0,
            'precision': 0.0
        }
    }
    
    return default_config, 0.0, 0.0, 0.0, 0.0


def save_best_config(config_path, config, results):
    """Save configuration with updated best performance metrics."""
    # Calculate optimal threshold and performance details
    from sklearn.metrics import confusion_matrix
    
    # Get the threshold that was used for these results
    optimal_threshold = results.get('threshold', 0.0)
    
    # Calculate confusion matrix if we have predictions
    if 'predictions' in results and 'y_true' in results:
        y_true = results['y_true']
        y_pred = results['predictions']
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        performance_details = {
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'true_negatives': int(tn)
        }
    else:
        performance_details = {
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0,
            'true_negatives': 0
        }
    
    # Update best performance metrics
    config['best_performance'] = {
        'roc_auc': results['roc_auc'],
        'f1_score': results['f1_score'],
        'recall': results['recall'],
        'precision': results['precision'],
        'optimal_threshold': optimal_threshold,
        'model_info': {
            'total_params': results.get('total_params', 48496),
            'trainable_params': results.get('trainable_params', 48496)
        },
        'training_history': {
            'final_loss': results.get('final_loss', 0.0),
            'final_mae': results.get('final_mae', 0.0),
            'best_epoch': results.get('best_epoch', 5)
        },
        'performance_details': performance_details
    }
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"*** New best config saved! AUC: {results['roc_auc']:.4f}, F1: {results['f1_score']:.4f}")
    logger.info(f"   Optimal threshold: {optimal_threshold:.6f}")
    logger.info(f"   Performance details: TP={performance_details['true_positives']}, FP={performance_details['false_positives']}, TN={performance_details['true_negatives']}, FN={performance_details['false_negatives']}")


def train_model(config_loader, df_train_features, df_val_features, df_test_features, model_path):
    """Train the autoencoder model."""
    logger.info("Training autoencoder model...")
    
    # Initialize autoencoder
    autoencoder = FraudAutoencoder(config_loader.config)
    
    # Prepare data
    X_train, X_val, X_test = autoencoder.prepare_data(
        df_train_features, df_val_features, df_test_features
    )
    
    # Get target variables
    y_train = df_train_features['is_fraudulent'].values
    y_val = df_val_features['is_fraudulent'].values
    y_test = df_test_features['is_fraudulent'].values
    
    # Train model
    results = autoencoder.train(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Save model and feature engineer
    autoencoder.save_model(model_path)
    feature_engineer = FeatureEngineer(config_loader.config.get('features', {}))
    feature_engineer.save_fitted_objects(f"{model_path}_features.pkl")
    
    logger.info(f"Model training completed - Test AUC: {results['roc_auc']:.4f}")
    return results, autoencoder


def train_model_80_20(config_loader, df_train_features, df_test_features, model_path, y_train, y_test, feature_engineer):
    """Train the autoencoder model with clean 80/20 split."""
    logger.info("Training autoencoder model (80/20 split)...")
    
    # Initialize autoencoder
    autoencoder = FraudAutoencoder(config_loader.config)
    
    # Prepare data
    X_train, X_test = autoencoder.prepare_data_80_20(df_train_features, df_test_features)
    
    # Train model
    results = autoencoder.train_80_20(X_train, X_test, y_train, y_test)
    
    # Save model and feature engineer
    autoencoder.save_model(model_path)
    feature_engineer.save_fitted_objects(f"{model_path}_features.pkl")
    
    logger.info(f"Model training completed (80/20) - Test AUC: {results['roc_auc']:.4f}")
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
        # Load current best configuration and performance
        current_config, best_auc, best_f1, best_recall, best_precision = load_best_config(args.config)
        
        # Create config loader with current best config
        config_loader = ConfigLoader(args.config)
        config_loader.config = current_config
        
        logger.info("Configuration loaded successfully")
        
        # Validate and prepare data
        data_path = validate_data_paths(args.data_path, args.raw_data_path)
        
        # Load and split data (clean 80/20 split)
        df_train, df_test = load_and_split_data_80_20(data_path)
        logger.info(f"Data loaded: {len(df_train)} train, {len(df_test)} test samples")
        
        # Extract target variables before feature engineering
        y_train = df_train['is_fraudulent'].values
        y_test = df_test['is_fraudulent'].values
        
        # Feature engineering
        feature_config = config_loader.get_feature_config()
        feature_engineer = FeatureEngineer(feature_config)
        df_train_features, df_test_features = feature_engineer.fit_transform_80_20(df_train, df_test)
        logger.info(f"Feature engineering completed: {len(df_train_features.columns)} features")
        
        # Save engineered test features for inference
        os.makedirs('data/engineered', exist_ok=True)
        df_test_features.to_csv('data/engineered/test_features.csv', index=False)
        logger.info("Saved engineered test features to data/engineered/test_features.csv")
        
        # Execute mode-specific operations
        if args.mode == "train":
            results, _ = train_model_80_20(config_loader, df_train_features, df_test_features, args.model_path, y_train, y_test, feature_engineer)
            
            # Check if new performance is better than current best
            new_auc = results['roc_auc']
            new_f1 = results['f1_score']
            new_recall = results['recall']
            new_precision = results['precision']
            
            logger.info("=" * 50)
            logger.info("PERFORMANCE COMPARISON")
            logger.info(f"Current Best - AUC: {best_auc:.4f}, F1: {best_f1:.4f}, Recall: {best_recall:.4f}, Precision: {best_precision:.4f}")
            logger.info(f"New Results  - AUC: {new_auc:.4f}, F1: {new_f1:.4f}, Recall: {new_recall:.4f}, Precision: {new_precision:.4f}")
            logger.info("=" * 50)
            
            if new_auc > best_auc:
                logger.info("*** NEW BEST PERFORMANCE! Saving updated configuration...")
                save_best_config(args.config, current_config, results)
                
                # Also save model info
                model_info = {
                    'config_path': args.config,
                    'model_path': args.model_path,
                    'performance': results,
                    'timestamp': str(pd.Timestamp.now())
                }
                
                with open('models/best_model_info.yaml', 'w') as f:
                    yaml.dump(model_info, f, default_flow_style=False, indent=2)
                
                logger.info("*** Best model info saved to models/best_model_info.yaml")
            else:
                logger.info("*** Performance not improved. Keeping current best configuration.")
            
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