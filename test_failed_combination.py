#!/usr/bin/env python3
"""
Test script to run the failed combination (combination 72) separately.
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.config_loader import ConfigLoader
from src.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder
from src.config import PipelineConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_failed_combination():
    """Test the failed combination (combination 72)."""
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config("best_features")
    
    # Set the failed combination parameters
    config['model']['latent_dim'] = 32
    config['model']['learning_rate'] = 0.001
    config['model']['activation_fn'] = 'leaky_relu'
    config['model']['batch_size'] = 128
    config['model']['threshold'] = 95
    config['model']['epochs'] = 10
    config['features']['strategy'] = 'combined'
    
    logger.info("Testing failed combination:")
    logger.info(f"  latent_dim: {config['model']['latent_dim']}")
    logger.info(f"  learning_rate: {config['model']['learning_rate']}")
    logger.info(f"  activation_fn: {config['model']['activation_fn']}")
    logger.info(f"  batch_size: {config['model']['batch_size']}")
    logger.info(f"  threshold: {config['model']['threshold']}")
    logger.info(f"  epochs: {config['model']['epochs']}")
    
    try:
        # Load data and generate features
        strategy = config['features']['strategy']
        logger.info(f"Using feature strategy: {strategy}")
        pipeline_config = PipelineConfig.get_config(strategy)
        
        # Load cleaned data
        data_path = Path("data/cleaned/ecommerce_cleaned.csv")
        logger.info(f"Loading data from: {data_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Cleaned data not found: {data_path}")
        
        import pandas as pd
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Generate features
        logger.info(f"Creating feature engineer for strategy: {strategy}")
        feature_engineer = FeatureFactory.create(strategy)
        logger.info(f"Generating features...")
        df_features = feature_engineer.generate_features(df)
        logger.info(f"Features generated successfully. Shape: {df_features.shape}")
        
        # Separate features and target
        if 'is_fraudulent' in df_features.columns:
            X = df_features.drop(columns=['is_fraudulent'])
            y = df_features['is_fraudulent']
            logger.info(f"Target column found. X shape: {X.shape}, y shape: {y.shape}")
            logger.info(f"Fraud ratio: {y.mean():.4f}")
        else:
            raise ValueError("Target column 'is_fraudulent' not found in features")
        
        # Initialize and train model
        logger.info(f"Initializing autoencoder with config: {pipeline_config}")
        autoencoder = BaselineAutoencoder(pipeline_config)
        logger.info(f"Autoencoder initialized successfully")
        
        # Train the model
        logger.info(f"Starting model training...")
        results = autoencoder.train()
        history = results['history']
        
        # Access history correctly
        loss_history = history.history['loss']
        final_loss = loss_history[-1] if loss_history else 0.0
        logger.info(f"Training completed. Final loss: {final_loss:.6f}")
        logger.info(f"Training history length: {len(loss_history)} epochs")
        
        # Evaluate model
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        
        logger.info(f"Starting model evaluation...")
        
        # Prepare numeric features for prediction
        logger.info(f"Preparing numeric features for prediction...")
        df_numeric = X.select_dtypes(include=[np.number])
        X_numeric = df_numeric.values
        logger.info(f"Numeric features shape: {X_numeric.shape}")
        
        # Get predictions
        logger.info(f"Generating predictions...")
        anomaly_scores = autoencoder.predict_anomaly_scores(X_numeric)
        logger.info(f"Predictions generated. Anomaly scores shape: {anomaly_scores.shape}")
        
        # Calculate metrics
        logger.info(f"Calculating ROC AUC...")
        roc_auc = roc_auc_score(y, anomaly_scores)
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        
        # Use threshold to get binary predictions
        threshold = np.percentile(anomaly_scores, config['model']['threshold'])
        binary_predictions = (anomaly_scores > threshold).astype(int)
        logger.info(f"Threshold: {threshold:.4f} (percentile: {config['model']['threshold']})")
        
        precision = precision_score(y, binary_predictions, zero_division=0)
        recall = recall_score(y, binary_predictions, zero_division=0)
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Find best epoch
        best_epoch = np.argmin(loss_history)
        logger.info(f"Best epoch: {best_epoch} (loss: {loss_history[best_epoch]:.6f})")
        
        logger.info("=" * 60)
        logger.info("TEST RESULTS:")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  Final Loss: {final_loss:.6f}")
        logger.info(f"  Best Epoch: {best_epoch}")
        logger.info("=" * 60)
        
        return True, roc_auc
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, 0.0

if __name__ == "__main__":
    success, roc_auc = test_failed_combination()
    if success:
        print(f"\n✅ Test passed! ROC AUC: {roc_auc:.4f}")
    else:
        print(f"\n❌ Test failed!") 