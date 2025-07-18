#!/usr/bin/env python3
"""
Test All 9 Strategies - Complete Feature Set
Using the same settings as the 0.7431 AUC run
"""

import wandb
import yaml
import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# NO EXPLICIT SEED SETTING - match original 0.7431 run exactly
# The pipeline config will use random_state=42 for data splitting only

from src.config_loader import ConfigLoader
from src.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder
from src.config import PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_all_9_strategies.log')
    ]
)
logger = logging.getLogger(__name__)

def load_existing_model_info() -> Optional[Dict]:
    """Load existing model info if it exists, converting numpy objects to standard types."""
    model_info_path = Path("models/model_info.yaml")
    if model_info_path.exists():
        try:
            with open(model_info_path, 'r') as f:
                model_info = yaml.safe_load(f)
            
            # Convert any numpy objects to standard Python types
            if model_info and 'best_auc' in model_info:
                if hasattr(model_info['best_auc'], 'item'):
                    model_info['best_auc'] = float(model_info['best_auc'])
                
                if 'threshold' in model_info and hasattr(model_info['threshold'], 'item'):
                    model_info['threshold'] = float(model_info['threshold'])
                
                return model_info
        except Exception as e:
            logger.warning(f"Could not load existing model info: {e}")
            # Remove corrupted file
            model_info_path.unlink()
            logger.info("Removed corrupted model_info.yaml")
    
    return None

def save_model_info(config: Dict, roc_auc: float, feature_count: int, threshold: float):
    """Save model info to YAML file with standard Python types."""
    # Convert numpy types to standard Python types
    if hasattr(threshold, 'item'):
        threshold = float(threshold)
    
    model_info = {
        'best_auc': float(roc_auc),
        'config': config,
        'feature_count': int(feature_count),
        'feature_strategy': str(config['features']['strategy']),
        'model_path': 'models/final_model.h5',
        'threshold': threshold,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    model_info_path = Path("models/model_info.yaml")
    model_info_path.parent.mkdir(exist_ok=True)
    
    with open(model_info_path, 'w') as f:
        yaml.dump(model_info, f, default_flow_style=False)
    
    logger.info(f"Model info saved to {model_info_path}")

def main():
    """Main function to test all 9 strategies."""
    logger.info("=" * 80)
    logger.info("TESTING ALL 9 STRATEGIES")
    logger.info("=" * 80)
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config("best_features")
    
    # Set to use all 9 strategies (combined strategy includes all)
    config['features']['strategy'] = "combined"
    
    # Use same hyperparameters as 0.7431 run
    config['model']['latent_dim'] = 16
    config['model']['batch_size'] = 128
    config['model']['learning_rate'] = 0.01
    config['model']['activation_fn'] = 'relu'
    config['model']['threshold'] = 95
    config['model']['epochs'] = 50
    config['model']['early_stopping'] = True
    
    logger.info(f"Using configuration: {config['model']}")
    logger.info(f"Feature strategy: {config['features']['strategy']}")
    
    # Check existing model
    existing_model_info = load_existing_model_info()
    if existing_model_info:
        logger.info(f"Existing model found with AUC: {existing_model_info['best_auc']:.4f}")
    else:
        logger.info("No existing model found")
    
    # Initialize W&B
    wandb_config = config.copy()
    wandb_config['wandb']['tags'] = ["all_9_strategies", "reproduction_attempt"]
    
    run_name = f"all_9_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        logger.info(f"Initializing W&B run: {run_name}")
        
        with wandb.init(
            project=wandb_config['wandb']['project'],
            entity=wandb_config['wandb'].get('entity'),
            config=wandb_config,
            name=run_name,
            tags=wandb_config['wandb']['tags'],
            force=True
        ) as run:
            
            # Load data and generate features
            strategy = config['features']['strategy']
            logger.info(f"Using feature strategy: {strategy}")
            pipeline_config = PipelineConfig.get_config(strategy)
            logger.info(f"Pipeline config loaded successfully")
            
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
            
            # Log feature information
            logger.info(f"Logging feature information to W&B")
            wandb.log({
                "feature_count": X.shape[1],
                "sample_count": X.shape[0],
                "fraud_ratio": y.mean(),
                "feature_strategy": strategy,
                "all_9_strategies": True
            })
            
            # Initialize and train model
            logger.info(f"Initializing autoencoder with config: {pipeline_config}")
            autoencoder = BaselineAutoencoder(pipeline_config)
            logger.info(f"Autoencoder initialized successfully")
            
            # Train the model
            logger.info(f"Starting model training...")
            results = autoencoder.train()
            history = results['history']
            final_loss = history.history['loss'][-1] if 'loss' in history.history else 0.0
            logger.info(f"Training completed. Final loss: {final_loss:.6f}")
            logger.info(f"Training history length: {len(history.history['loss'])} epochs")
            
            # Evaluate model
            from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
            
            logger.info(f"Starting model evaluation...")
            
            # Get predictions using autoencoder's built-in methods
            logger.info(f"Preparing numeric features for prediction...")
            df_numeric = X.select_dtypes(include=[np.number])
            X_numeric = df_numeric.values
            logger.info(f"Numeric features shape: {X_numeric.shape}")
            
            # Get predictions
            logger.info(f"Generating predictions...")
            predictions = autoencoder.model.predict(autoencoder.scaler.transform(X_numeric))
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
            logger.info(f"Binary predictions shape: {binary_predictions.shape}")
            
            precision = precision_score(y, binary_predictions, zero_division=0)
            recall = recall_score(y, binary_predictions, zero_division=0)
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            # Find best epoch
            loss_history = history.history['loss']
            best_epoch = np.argmin(loss_history)
            logger.info(f"Best epoch: {best_epoch} (loss: {loss_history[best_epoch]:.6f})")
            
            # Log final metrics
            final_metrics = {
                "final_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "threshold": threshold,
                "best_epoch": best_epoch,
                "final_loss": loss_history[-1],
                "best_loss": loss_history[best_epoch],
                "total_epochs": len(loss_history),
                "feature_count": X.shape[1],
                "all_9_strategies": True
            }
            
            logger.info(f"Logging final metrics to W&B: {final_metrics}")
            wandb.log(final_metrics)
            
            # Log confusion matrix
            logger.info(f"Logging confusion matrix to W&B")
            cm = confusion_matrix(y, binary_predictions)
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, 
                    y_true=y, 
                    preds=binary_predictions
                )
            })
            
            logger.info(f"Training completed successfully with ROC AUC: {roc_auc:.4f}")
            logger.info(f"Run URL: {run.url}")
            
            # Check if this model is better than existing
            should_save = False
            if existing_model_info:
                existing_auc = existing_model_info['best_auc']
                if roc_auc > existing_auc:
                    logger.info(f"New model is better! {roc_auc:.4f} > {existing_auc:.4f}")
                    should_save = True
                else:
                    logger.info(f"Existing model is better. {existing_auc:.4f} > {roc_auc:.4f}")
            else:
                logger.info("No existing model to compare against")
                should_save = True
            
            # Save model if it's better
            if should_save:
                logger.info("Saving new model...")
                
                # Save model
                model_path = Path("models/final_model.h5")
                model_path.parent.mkdir(exist_ok=True)
                autoencoder.model.save(model_path)
                logger.info(f"Model saved to {model_path}")
                
                # Save scaler
                scaler_path = Path("models/final_model_scaler.pkl")
                import pickle
                with open(scaler_path, 'wb') as f:
                    pickle.dump(autoencoder.scaler, f)
                logger.info(f"Scaler saved to {scaler_path}")
                
                # Save model info
                save_model_info(config, roc_auc, X.shape[1], threshold)
                logger.info("Model info saved")
            else:
                logger.info("Not saving model - existing model is better")
            
            return True, roc_auc, final_metrics
            
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, 0.0, {}

if __name__ == "__main__":
    success, roc_auc, metrics = main()
    if success:
        print(f"\nTraining completed successfully!")
        print(f"   ROC AUC: {roc_auc:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   Feature Count: {metrics['feature_count']}")
        print(f"   Best Epoch: {metrics['best_epoch']}")
        print(f"   Final Loss: {metrics['final_loss']:.6f}")
        print(f"   Threshold: {metrics['threshold']:.4f}")
    else:
        print(f"\nTraining failed!") 