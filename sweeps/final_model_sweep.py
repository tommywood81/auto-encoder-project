#!/usr/bin/env python3
"""
Final Model Sweep - Fine-tuned Hyperparameter Optimization

This script performs a final hyperparameter sweep around our best model's parameters
to find the optimal configuration for production deployment.
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
        logging.FileHandler('final_model_sweep.log')
    ]
)
logger = logging.getLogger(__name__)

# Configure W&B to force logging
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_FORCE"] = "true"

def train_model_with_config(config: Dict, entity: Optional[str] = None, run_name: str = "final_sweep") -> Tuple[bool, float, Dict]:
    """
    Train model with specific hyperparameters and track with W&B.
    
    Args:
        config: Model configuration
        entity: W&B entity (username/team)
        run_name: Name for the W&B run
        
    Returns:
        Tuple of (success, roc_auc, metrics_dict)
    """
    logger.info(f"Starting training with run: {run_name}")
    
    # Initialize W&B
    wandb_config = config.copy()
    wandb_config['wandb']['tags'] = ["final_sweep", "production_tuning"]
    
    if entity:
        wandb_config['wandb']['entity'] = entity
    
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
                "final_sweep": True
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
            
            # Log final metrics ONLY (no per-epoch logging)
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
                "final_sweep": True,
                # Log hyperparameters for easy comparison
                "learning_rate": config['model']['learning_rate'],
                "threshold_percentile": config['model']['threshold']
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
            
            return True, roc_auc, final_metrics
            
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, 0.0, {}

def run_final_sweep(entity: Optional[str] = None):
    """
    Run final hyperparameter sweep around best model parameters.
    """
    logger.info("Final Model Hyperparameter Sweep")
    logger.info("=" * 80)
    
    print("🎯 FINAL MODEL HYPERPARAMETER SWEEP")
    print("=" * 80)
    print("Testing fine-tuned variations around best model parameters")
    print("=" * 80)
    
    # Load best features configuration (our current best)
    logger.info("Loading best features configuration...")
    config_loader = ConfigLoader()
    config = config_loader.load_config("best_features")
    
    # Set to combined strategy (our best performing)
    config['features']['strategy'] = "combined"
    
    # Set our best known parameters (fixed)
    config['model']['latent_dim'] = 16
    config['model']['activation_fn'] = 'relu'
    config['model']['batch_size'] = 128
    config['model']['epochs'] = 50
    config['model']['early_stopping'] = True
    
    # Define fine-tuned hyperparameter variations around our best model
    # Based on our previous results, we'll test variations around these parameters:
    
    # Current best model parameters (estimated from previous runs)
    base_params = {
        'latent_dim': 16,
        'learning_rate': 0.01,
        'activation_fn': 'relu',
        'batch_size': 128,
        'threshold': 95,
        'epochs': 50
    }
    
    # Fine-tuned variations - ONLY learning rate and threshold
    hyperparams = {
        'learning_rate': [0.005, 0.008, 0.01, 0.012, 0.015],  # Fine-tune around 0.01
        'threshold': [91, 93, 95, 97, 99]  # Fine-tune around 95
    }
    
    logger.info(f"Base parameters: {base_params}")
    logger.info(f"Hyperparameter variations: {hyperparams}")
    
    results = []
    total_combinations = len(hyperparams['learning_rate']) * len(hyperparams['threshold'])
    logger.info(f"Total combinations to test: {total_combinations}")
    
    # Generate all combinations
    import itertools
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    
    print(f"Testing {total_combinations} hyperparameter combinations...")
    
    start_time = time.time()
    for i, combination in enumerate(itertools.product(*values)):
        param_dict = dict(zip(keys, combination))
        
        print(f"\n{'='*80}")
        print(f"🎯 COMBINATION {i+1}/{total_combinations}")
        print(f"{'='*80}")
        print(f"📊 Parameters: {param_dict}")
        print(f"⏰ Started: {time.strftime('%H:%M:%S')}")
        print(f"📈 Progress: {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")
        print(f"🔄 Remaining: {total_combinations - i - 1} combinations")
        print(f"{'='*80}")
        
        # Update config with current hyperparameters
        test_config = config.copy()
        for key, value in param_dict.items():
            test_config['model'][key] = value
        
        # Create run name
        run_name = f"final_sweep_{param_dict['learning_rate']}_{param_dict['threshold']}"
        
        # Train model
        success, roc_auc, metrics = train_model_with_config(test_config, entity, run_name)
        
        if success:
            results.append((test_config, roc_auc, param_dict))
            print(f"✅ ROC AUC: {roc_auc:.4f}")
            print(f"📊 Learning Rate: {param_dict['learning_rate']}")
            print(f"🎯 Threshold: {param_dict['threshold']}")
        else:
            print(f"❌ Failed")
        
        # Progress update
        elapsed_time = time.time() - start_time
        avg_time_per_combo = elapsed_time / (i + 1)
        remaining_time = avg_time_per_combo * (total_combinations - i - 1)
        
        print(f"⏱️  Elapsed: {elapsed_time/60:.1f} min")
        print(f"⏱️  Remaining: {remaining_time/60:.1f} min")
        print(f"⏱️  ETA: {time.strftime('%H:%M:%S', time.localtime(time.time() + remaining_time))}")
        
        # Small delay between runs
        time.sleep(1)
    
    # Sort by ROC AUC (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎉 Final sweep completed! Top 10 results:")
    print("=" * 80)
    for i, (config, roc_auc, params) in enumerate(results[:10]):
        print(f"   {i+1}. ROC AUC: {roc_auc:.4f} - {params}")
    
    # Save best configuration
    if results:
        best_config, best_roc_auc, best_params = results[0]
        logger.info(f"Best configuration found: {best_params}")
        logger.info(f"Best ROC AUC: {best_roc_auc:.4f}")
        
        # Save best configuration
        config_loader.update_config("final_optimized_config", {"model": best_config['model']})
        print(f"\n💾 Best configuration saved to configs/final_optimized_config.yaml")
        
        return best_config, best_roc_auc
    else:
        logger.error("No successful configurations found")
        return None, 0.0

def main():
    """Main function to run the final model sweep."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run final model hyperparameter sweep")
    parser.add_argument("--entity", type=str, help="W&B entity (username/team)")
    parser.add_argument("--project", type=str, default="fraud-detection-autoencoder", 
                       help="W&B project name")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("MAIN FUNCTION STARTING")
    logger.info("=" * 80)
    logger.info("Starting final model sweep")
    logger.info(f"Arguments: {args}")
    
    # Set W&B project
    os.environ["WANDB_PROJECT"] = args.project
    logger.info(f"W&B project set to: {args.project}")
    logger.info("=" * 80)
    
    try:
        logger.info("Running final model sweep")
        best_config, best_roc_auc = run_final_sweep(args.entity)
        
        if best_config:
            logger.info("Final model sweep completed successfully")
            print(f"\n🎉 Final model sweep completed!")
            print(f"   Best ROC AUC: {best_roc_auc:.4f}")
            print(f"   Best configuration saved to configs/final_optimized_config.yaml")
            print(f"   Check your W&B dashboard for detailed results")
        else:
            logger.error("Final model sweep failed")
            print(f"\n❌ Final model sweep failed!")
                
    except KeyboardInterrupt:
        logger.warning("Final model sweep interrupted by user")
        print("\nFinal model sweep interrupted by user")
    except Exception as e:
        logger.error(f"Final model sweep failed: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\nFinal model sweep failed: {str(e)}")

if __name__ == "__main__":
    main() 