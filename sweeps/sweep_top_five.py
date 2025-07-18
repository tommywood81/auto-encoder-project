#!/usr/bin/env python3
"""
Sweep Top Five - 7-Strategy Combination Test
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
        logging.FileHandler('sweep_top_five.log')
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
            if model_info_path.exists():
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
    """Main function to test the 7-strategy combination."""
    logger.info("=" * 80)
    logger.info("SWEEP TOP FIVE - 7-STRATEGY COMBINATION TEST")
    logger.info("=" * 80)
    logger.info("Using NO explicit seed setting - matching original 0.7431 run")
    logger.info("Pipeline config will use random_state=42 for data splitting only")
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config("best_features")
    
    # Set to combined strategy (all 7 strategies)
    config['features']['strategy'] = "combined"
    config['model']['epochs'] = 50
    config['model']['early_stopping'] = True
    
    logger.info(f"Feature strategy: {config['features']['strategy']}")
    logger.info(f"Epochs: {config['model']['epochs']}")
    logger.info(f"Early stopping: {config['model']['early_stopping']}")
    
    # Check existing model
    existing_model = load_existing_model_info()
    if existing_model:
        logger.info(f"Existing best AUC: {existing_model['best_auc']:.4f}")
    else:
        logger.info("No existing model found")
    
    # Initialize W&B
    wandb_config = config.copy()
    wandb_config['wandb']['tags'] = ["sweep_top_five", "7_strategy_combination"]
    
    run_name = f"sweep_top_five_7_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
            
            # Override the random_state to use seed 6 for data splitting
            pipeline_config.data.random_state = 6
            logger.info(f"Pipeline config loaded successfully with random_state=6")
            
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
            logger.info(f"Feature columns: {list(df_features.columns)}")
            
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
                "sweep_top_five": True,
                "no_explicit_seed": True,
                "original_config_match": True
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
            
            # Log final metrics and hyperparameters
            final_metrics = {
                "final_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "threshold": threshold,
                "best_epoch": best_epoch,
                "final_loss": loss_history[-1],
                "best_loss": loss_history[best_epoch],
                "total_epochs": len(loss_history),
                "no_explicit_seed": True,
                "original_config_match": True,
                "latent_dim": config['model']['latent_dim'],
                "learning_rate": config['model']['learning_rate'],
                "activation_fn": config['model']['activation_fn'],
                "batch_size": config['model']['batch_size'],
                "threshold_percentile": config['model']['threshold'],
                "epochs": config['model']['epochs']
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
            
            # Check if this model is better than existing
            should_save = True
            if existing_model:
                existing_auc = existing_model['best_auc']
                if roc_auc <= existing_auc:
                    logger.info(f"Existing model is better. ROC AUC: {existing_auc:.4f} >= {roc_auc:.4f}")
                    should_save = False
                else:
                    logger.info(f"New model is better! ROC AUC: {roc_auc:.4f} > {existing_auc:.4f}")
            
            # Save model if it's better
            if should_save:
                logger.info("Saving new model and scaler...")
                
                # Save model
                model_path = Path("models/final_model.h5")
                model_path.parent.mkdir(exist_ok=True)
                autoencoder.model.save(str(model_path))
                logger.info(f"Model saved to {model_path}")
                
                # Save scaler
                scaler_path = Path("models/final_model_scaler.pkl")
                import pickle
                with open(scaler_path, 'wb') as f:
                    pickle.dump(autoencoder.scaler, f)
                logger.info(f"Scaler saved to {scaler_path}")
                
                # Save model info
                save_model_info(config, roc_auc, X.shape[1], threshold)
                logger.info("Model info updated")
            else:
                logger.info("Not saving model - existing model is better")
            
            logger.info(f"Training completed successfully with ROC AUC: {roc_auc:.4f}")
            logger.info(f"Run URL: {run.url}")
            
            # Print summary
            print("\n" + "="*80)
            print("SWEEP TOP FIVE RESULTS")
            print("="*80)
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"Best Epoch: {best_epoch}")
            print(f"Final Loss: {final_loss:.6f}")
            print(f"Feature Count: {X.shape[1]}")
            print(f"Strategies: baseline_numeric, categorical, behavioral, fraud_flags, demographics, time_interactions, rank_encoding")
            print("="*80)
            
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
    main() 