#!/usr/bin/env python3
"""
Hyperparameter Sweep Script with W&B Integration for Fraud Detection Pipeline

This script performs hyperparameter tuning on the autoencoder model using the best
feature strategy from the feature sweep.
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
        logging.FileHandler('sweep_parameters.log')
    ]
)
logger = logging.getLogger(__name__)

# Configure W&B to force logging
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_FORCE"] = "true"

# Test logging immediately
print("=" * 80)
print("HYPERPARAMETER SWEEP SCRIPT STARTING")
print("=" * 80)
logger.info("=" * 80)
logger.info("HYPERPARAMETER SWEEP SCRIPT STARTING")
logger.info("=" * 80)
logger.info("Logging system initialized successfully")
logger.info("W&B force mode enabled")
logger.info("=" * 80)
print("Logging system initialized successfully")
print("W&B force mode enabled")
print("=" * 80)

def train_model_with_config(config: Dict, entity: Optional[str] = None, stage: str = "broad") -> Tuple[bool, float, Dict]:
    """
    Train model with specific hyperparameters and track with W&B.
    
    Args:
        config: Model configuration
        entity: W&B entity (username/team)
        stage: Sweep stage ("broad", "refined", "final")
        
    Returns:
        Tuple of (success, roc_auc, metrics_dict)
    """
    logger.info(f"Starting training with stage: {stage}")
    logger.info(f"Config keys: {list(config.keys())}")
    logger.info(f"Model config: {config.get('model', {})}")
    logger.info(f"Feature strategy: {config.get('features', {}).get('strategy', 'unknown')}")
    
    # Initialize W&B
    wandb_config = config.copy()
    wandb_config['wandb']['tags'] = ["param_sweep", f"stage_{stage}"]
    
    if entity:
        wandb_config['wandb']['entity'] = entity
    
    run_name = f"param_sweep_{stage}_{wandb_config['model']['latent_dim']}_{wandb_config['model']['learning_rate']}"
    
    try:
        logger.info(f"Initializing W&B run: {run_name}")
        logger.info(f"W&B project: {wandb_config['wandb']['project']}")
        logger.info(f"W&B entity: {wandb_config['wandb'].get('entity', 'None')}")
        logger.info(f"W&B tags: {wandb_config['wandb']['tags']}")
        
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
                "sweep_stage": stage
            })
            
            # Initialize and train model
            logger.info(f"Initializing autoencoder with config: {pipeline_config}")
            autoencoder = BaselineAutoencoder(pipeline_config)
            logger.info(f"Autoencoder initialized successfully")
            
            # Train the model
            logger.info(f"Starting model training...")
            results = autoencoder.train()
            history = results['history']
            logger.info(f"Training completed. Final loss: {history['loss'][-1]:.6f}")
            logger.info(f"Training history length: {len(history['loss'])} epochs")
            
            # Evaluate model
            from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
            
            logger.info(f"Starting model evaluation...")
            
            # Get predictions using autoencoder's built-in methods
            # First, prepare the data properly (get numeric features only)
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
            best_epoch = np.argmin(history['loss'])
            logger.info(f"Best epoch: {best_epoch} (loss: {history['loss'][best_epoch]:.6f})")
            
            # Log metrics
            metrics = {
                "final_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "threshold": threshold,
                "best_epoch": best_epoch,
                "final_loss": history['loss'][-1],
                "best_loss": history['loss'][best_epoch]
            }
            
            logger.info(f"Logging metrics to W&B: {metrics}")
            wandb.log(metrics)
            
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
            
            # Log training history
            logger.info(f"Logging training history to W&B ({len(history['loss'])} epochs)")
            for epoch in range(len(history['loss'])):
                wandb.log({
                    "epoch": epoch,
                    "loss": history['loss'][epoch],
                    "val_loss": history.get('val_loss', [0])[epoch] if epoch < len(history.get('val_loss', [])) else 0
                })
            
            logger.info(f"Training completed successfully with ROC AUC: {roc_auc:.4f}")
            logger.info(f"Run URL: {run.url}")
            
            return True, roc_auc, metrics
            
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, 0.0, {}

def run_broad_sweep(entity: Optional[str] = None) -> List[Tuple[Dict, float]]:
    """
    Run broad hyperparameter sweep (Stage 2.1).
    
    Args:
        entity: W&B entity (username/team)
        
    Returns:
        List of (config, roc_auc) tuples sorted by performance
    """
    logger.info("Stage 2.1: Broad Hyperparameter Sweep")
    logger.info("=" * 60)
    print("Stage 2.1: Broad Hyperparameter Sweep")
    print("=" * 60)
    
    # Load best features configuration
    logger.info("Loading best features configuration...")
    config_loader = ConfigLoader()
    config = config_loader.load_config("best_features")
    logger.info(f"Configuration loaded successfully")
    
    # Define hyperparameter combinations
    hyperparams = {
        'latent_dim': [8, 16, 32],
        'learning_rate': [0.01, 0.005, 0.001],  # Increased learning rates
        'activation_fn': ['relu', 'leaky_relu'],
        'batch_size': [64, 128],
        'threshold': [90, 95]
    }
    logger.info(f"Hyperparameter combinations: {hyperparams}")
    
    # Set epochs for broad sweep
    config['model']['epochs'] = 10
    logger.info(f"Setting epochs to: {config['model']['epochs']}")
    
    results = []
    total_combinations = len(hyperparams['latent_dim']) * len(hyperparams['learning_rate']) * len(hyperparams['activation_fn']) * len(hyperparams['batch_size']) * len(hyperparams['threshold'])
    logger.info(f"Total combinations to test: {total_combinations}")
    
    # Generate all combinations
    import itertools
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    
    total_combinations = len(list(itertools.product(*values)))
    print(f"Testing {total_combinations} hyperparameter combinations...")
    
    for i, combination in enumerate(itertools.product(*values)):
        param_dict = dict(zip(keys, combination))
        
        print(f"\nCombination {i+1}/{total_combinations}: {param_dict}")
        
        # Update config with current hyperparameters
        test_config = config.copy()
        for key, value in param_dict.items():
            test_config['model'][key] = value
        
        # Train model
        success, roc_auc, metrics = train_model_with_config(test_config, entity, "broad")
        
        if success:
            results.append((test_config, roc_auc))
            print(f"   ROC AUC: {roc_auc:.4f}")
        else:
            print(f"   Failed")
        
        # Small delay between runs
        time.sleep(1)
    
    # Sort by ROC AUC (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nBroad sweep completed! Top 10 results:")
    for i, (config, roc_auc) in enumerate(results[:10]):
        print(f"   {i+1}. ROC AUC: {roc_auc:.4f} - {config['model']}")
    
    return results

def run_refined_sweep(broad_results: List[Tuple[Dict, float]], entity: Optional[str] = None) -> List[Tuple[Dict, float]]:
    """
    Run refined hyperparameter sweep (Stage 2.2).
    
    Args:
        broad_results: Results from broad sweep
        entity: W&B entity (username/team)
        
    Returns:
        List of (config, roc_auc) tuples sorted by performance
    """
    print("\nStage 2.2: Refined Hyperparameter Sweep")
    print("=" * 60)
    
    # Take top 10 from broad sweep
    top_10 = broad_results[:10]
    
    # Increase epochs for refined sweep
    config = broad_results[0][0].copy()
    config['model']['epochs'] = 15
    
    print(f"Testing top {len(top_10)} configurations with 15 epochs...")
    
    refined_results = []
    
    for i, (test_config, _) in enumerate(top_10):
        print(f"\nTesting configuration {i+1}/{len(top_10)}")
        
        # Train model with more epochs
        success, roc_auc, metrics = train_model_with_config(test_config, entity, "refined")
        
        if success:
            refined_results.append((test_config, roc_auc))
            print(f"   ROC AUC: {roc_auc:.4f}")
        else:
            print(f"   Failed")
        
        # Small delay between runs
        time.sleep(1)
    
    # Sort by ROC AUC (descending)
    refined_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nRefined sweep completed! Top 3 results:")
    for i, (config, roc_auc) in enumerate(refined_results[:3]):
        print(f"   {i+1}. ROC AUC: {roc_auc:.4f} - {config['model']}")
    
    return refined_results

def run_final_training(top_configs: List[Tuple[Dict, float]], entity: Optional[str] = None) -> Tuple[Dict, float]:
    """
    Run final training with best configuration (Stage 2.3).
    
    Args:
        top_configs: Top configurations from refined sweep
        entity: W&B entity (username/team)
        
    Returns:
        Tuple of (best_config, best_roc_auc)
    """
    print("\nStage 2.3: Final Training")
    print("=" * 60)
    
    # Take top 3 from refined sweep
    top_3 = top_configs[:3]
    
    # Increase epochs for final training
    config = top_configs[0][0].copy()
    config['model']['epochs'] = 50
    config['model']['early_stopping'] = True
    
    print(f"Final training with top {len(top_3)} configurations (50 epochs + early stopping)...")
    
    best_config = None
    best_roc = 0.0
    
    for i, (test_config, _) in enumerate(top_3):
        print(f"\nFinal training {i+1}/{len(top_3)}")
        
        # Train model with full epochs and early stopping
        success, roc_auc, metrics = train_model_with_config(test_config, entity, "final")
        
        if success:
            if roc_auc > best_roc:
                best_roc = roc_auc
                best_config = test_config
                print(f"   New best! ROC AUC: {roc_auc:.4f}")
            else:
                print(f"   ROC AUC: {roc_auc:.4f}")
        else:
            print(f"   Failed")
        
        # Small delay between runs
        time.sleep(1)
    
    if best_config:
        print(f"\nFinal training completed! Best configuration:")
        print(f"   ROC AUC: {best_roc:.4f}")
        print(f"   Configuration: {best_config['model']}")
        
        # Save best configuration
        config_loader = ConfigLoader()
        config_loader.update_config("final_config", {"model": best_config['model']})
        print(f"   Best configuration saved to configs/final_config.yaml")
        
        return best_config, best_roc
    else:
        print(f"\nFinal training failed! No successful configurations.")
        return None, 0.0

def run_combined_feature_sweep(entity: Optional[str] = None):
    """
    Run hyperparameter sweep for the combined features strategy, logging to W&B group 'hyper-parameter-tuning'.
    """
    logger.info("Starting hyperparameter sweep for combined features strategy")
    logger.info("=" * 80)
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SWEEP FOR COMBINED FEATURES")
    print(f"{'='*80}")
    
    # Load config and set features to combined strategy
    logger.info("Loading configuration...")
    config_loader = ConfigLoader()
    config = config_loader.load_config("best_features")
    config['features']['strategy'] = "combined"
    config['wandb']['group'] = "hyper-parameter-tuning"
    config['wandb']['tags'] = ["param_sweep", "combined_features"]
    
    logger.info(f"Updated config with feature strategy: {config['features']['strategy']}")
    logger.info(f"W&B group: {config['wandb']['group']}")
    logger.info(f"W&B tags: {config['wandb']['tags']}")
    
    # Run broad sweep
    logger.info(f"Starting broad sweep for combined features")
    broad_results = run_broad_sweep_for_config(config, entity, "combined_features")
    
    if broad_results:
        logger.info(f"Broad sweep completed. Best AUC: {broad_results[0][1]:.4f}")
    else:
        logger.warning(f"Broad sweep failed")
        return
    
    # Run refined sweep
    logger.info(f"Starting refined sweep")
    refined_results = run_refined_sweep(broad_results, entity)
    
    if refined_results:
        logger.info(f"Refined sweep completed. Best AUC: {refined_results[0][1]:.4f}")
    else:
        logger.warning(f"Refined sweep failed")
        return
    
    # Final training
    logger.info(f"Starting final training")
    run_final_training(refined_results, entity)
    
    logger.info("Completed hyperparameter sweep for combined features")
    logger.info("=" * 80)

def run_broad_sweep_for_config(config: Dict, entity: Optional[str], group_name: str) -> List[Tuple[Dict, float]]:
    """Run broad sweep for a specific config and feature group name."""
    logger.info(f"Starting broad sweep for feature group: {group_name}")
    print(f"\n[Stage 2.1] Broad Hyperparameter Sweep for {group_name}")
    print("=" * 60)
    
    # Define hyperparameter combinations
    hyperparams = {
        'latent_dim': [8, 16, 32],
        'learning_rate': [0.01, 0.005, 0.001],
        'activation_fn': ['relu', 'leaky_relu'],
        'batch_size': [64, 128],
        'threshold': [90, 95]
    }
    logger.info(f"Hyperparameter combinations: {hyperparams}")
    
    config['model']['epochs'] = 10
    logger.info(f"Setting epochs to: {config['model']['epochs']}")
    
    results = []
    import itertools
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    
    total_combinations = len(list(itertools.product(*values)))
    logger.info(f"Total combinations to test: {total_combinations}")
    
    for i, combination in enumerate(itertools.product(*values)):
        logger.info(f"Testing combination {i+1}/{total_combinations}")
        
        sweep_config = yaml.safe_load(yaml.dump(config))  # Deep copy
        for k, v in zip(keys, combination):
            sweep_config['model'][k] = v
        
        # Log the feature group name in W&B
        sweep_config['wandb']['feature_group'] = group_name
        logger.info(f"Updated config with hyperparameters: {dict(zip(keys, combination))}")
        
        success, roc_auc, metrics = train_model_with_config(sweep_config, entity, stage="broad")
        if success:
            results.append((sweep_config, roc_auc))
            logger.info(f"Combination {i+1} successful. ROC AUC: {roc_auc:.4f}")
        else:
            logger.warning(f"Combination {i+1} failed")
    
    # Sort by ROC AUC
    results.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Broad sweep completed for {group_name}. Best AUC: {results[0][1] if results else 0:.4f}")
    
    return results

def main():
    """Main function to run the hyperparameter sweep."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep with W&B integration")
    parser.add_argument("--entity", type=str, help="W&B entity (username/team)")
    parser.add_argument("--project", type=str, default="fraud-detection-autoencoder", 
                       help="W&B project name")
    parser.add_argument("--stage", type=str, choices=["broad", "refined", "final", "all"], 
                       default="all", help="Which stage to run")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("MAIN FUNCTION STARTING")
    logger.info("=" * 80)
    logger.info("Starting hyperparameter sweep script")
    logger.info(f"Arguments: {args}")
    
    # Set W&B project
    os.environ["WANDB_PROJECT"] = args.project
    logger.info(f"W&B project set to: {args.project}")
    logger.info("=" * 80)
    
    try:
        logger.info("Running combined feature sweep")
        run_combined_feature_sweep(args.entity)
        
        logger.info("Hyperparameter sweep completed successfully")
        print(f"\nHyperparameter sweep completed!")
        print(f"   All results logged to W&B group: 'hyper-parameter-tuning'")
        print(f"   Check your W&B dashboard for detailed results")
                
    except KeyboardInterrupt:
        logger.warning("Hyperparameter sweep interrupted by user")
        print("\nHyperparameter sweep interrupted by user")
    except Exception as e:
        logger.error(f"Hyperparameter sweep failed: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\nHyperparameter sweep failed: {str(e)}")

if __name__ == "__main__":
    main() 