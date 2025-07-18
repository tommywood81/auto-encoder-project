#!/usr/bin/env python3
"""
Train Final Model with Best Hyperparameters

This script trains the autoencoder model with the best hyperparameters found
from our hyperparameter sweep and saves it to the models folder.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import pickle
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import ConfigLoader
from src.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder
from src.config import PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_final_model():
    """Train the final model with best hyperparameters and save it."""
    
    print("=" * 80)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("=" * 80)
    
    # Load the best configuration
    logger.info("Loading best configuration...")
    config_loader = ConfigLoader()
    config = config_loader.load_config("final_optimized_config")
    
    # Update with the actual best hyperparameters we found
    config['model']['learning_rate'] = 0.012  # Best from sweep
    config['model']['threshold'] = 97  # Best from sweep
    config['features']['strategy'] = "combined"  # Best feature strategy
    
    logger.info(f"Best hyperparameters:")
    logger.info(f"  Learning Rate: {config['model']['learning_rate']}")
    logger.info(f"  Threshold: {config['model']['threshold']}")
    logger.info(f"  Feature Strategy: {config['features']['strategy']}")
    
    # Load data
    logger.info("Loading cleaned data...")
    data_path = Path("data/cleaned/ecommerce_cleaned.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded. Shape: {df.shape}")
    
    # Generate features
    logger.info("Generating features with combined strategy...")
    feature_engineer = FeatureFactory.create("combined")
    df_features = feature_engineer.generate_features(df)
    logger.info(f"Features generated. Shape: {df_features.shape}")
    
    # Separate features and target
    X = df_features.drop(columns=['is_fraudulent'])
    y = df_features['is_fraudulent']
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Fraud ratio: {y.mean():.4f}")
    
    # Create pipeline config
    pipeline_config = PipelineConfig.get_config("combined")
    pipeline_config.model.learning_rate = config['model']['learning_rate']
    pipeline_config.model.threshold = config['model']['threshold']
    pipeline_config.model.epochs = 50  # Full training
    pipeline_config.model.early_stopping = True
    pipeline_config.model.patience = 10
    
    logger.info("Initializing autoencoder...")
    autoencoder = BaselineAutoencoder(pipeline_config)
    
    # Train the model
    logger.info("Starting model training...")
    results = autoencoder.train()
    history = results['history']
    
    # Get final metrics
    logger.info("Evaluating model...")
    df_numeric = X.select_dtypes(include=[np.number])
    X_numeric = df_numeric.values
    
    # Get predictions
    predictions = autoencoder.model.predict(autoencoder.scaler.transform(X_numeric))
    anomaly_scores = autoencoder.predict_anomaly_scores(X_numeric)
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    
    roc_auc = roc_auc_score(y, anomaly_scores)
    threshold = np.percentile(anomaly_scores, config['model']['threshold'])
    binary_predictions = (anomaly_scores > threshold).astype(int)
    
    precision = precision_score(y, binary_predictions, zero_division=0)
    recall = recall_score(y, binary_predictions, zero_division=0)
    f1 = f1_score(y, binary_predictions, zero_division=0)
    
    logger.info(f"Final Results:")
    logger.info(f"  ROC AUC: {roc_auc:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Threshold: {threshold:.4f}")
    
    # Save model and components
    logger.info("Saving model and components...")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save the trained model
    model_path = models_dir / "final_autoencoder.h5"
    autoencoder.model.save(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save the scaler
    scaler_path = models_dir / "final_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(autoencoder.scaler, f)
    logger.info(f"Scaler saved to: {scaler_path}")
    
    # Save model info
    model_info = {
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'feature_strategy': config['features']['strategy'],
        'learning_rate': config['model']['learning_rate'],
        'threshold_percentile': config['model']['threshold'],
        'threshold_value': float(threshold),
        'roc_auc': float(roc_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'feature_count': X.shape[1],
        'sample_count': X.shape[0],
        'fraud_ratio': float(y.mean()),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_hyperparameters': {
            'learning_rate': config['model']['learning_rate'],
            'threshold': config['model']['threshold'],
            'feature_strategy': config['features']['strategy']
        }
    }
    
    model_info_path = models_dir / "final_model_info.yaml"
    with open(model_info_path, 'w') as f:
        yaml.dump(model_info, f, default_flow_style=False)
    logger.info(f"Model info saved to: {model_info_path}")
    
    # Save the best configuration
    best_config_path = models_dir / "best_config.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Best configuration saved to: {best_config_path}")
    
    print("\n" + "=" * 80)
    print("FINAL MODEL TRAINING COMPLETED!")
    print("=" * 80)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Model info saved to: {model_info_path}")
    print("=" * 80)
    
    return model_info

if __name__ == "__main__":
    try:
        model_info = train_final_model()
        print("\nFinal model training completed successfully!")
        print(f"Best ROC AUC achieved: {model_info['roc_auc']:.4f}")
        print("Model is ready for deployment!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\nTraining failed: {str(e)}")
        sys.exit(1) 