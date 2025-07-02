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
    
    # Initialize W&B
    wandb_config = config.copy()
    wandb_config['wandb']['tags'] = ["param_sweep", f"stage_{stage}"]
    
    if entity:
        wandb_config['wandb']['entity'] = entity
    
    run_name = f"param_sweep_{stage}_{wandb_config['model']['latent_dim']}_{wandb_config['model']['learning_rate']}"
    
    try:
        with wandb.init(
            project=wandb_config['wandb']['project'],
            entity=wandb_config['wandb'].get('entity'),
            config=wandb_config,
            name=run_name,
            tags=wandb_config['wandb']['tags']
        ) as run:
            
            # Load data and generate features
            strategy = config['features']['strategy']
            pipeline_config = PipelineConfig.get_config(strategy)
            
            # Load cleaned data
            data_path = Path("data/cleaned/ecommerce_cleaned.csv")
            if not data_path.exists():
                raise FileNotFoundError(f"Cleaned data not found: {data_path}")
            
            import pandas as pd
            df = pd.read_csv(data_path)
            
            # Generate features
            feature_engineer = FeatureFactory.create(strategy)
            df_features = feature_engineer.generate_features(df)
            
            # Separate features and target
            if 'is_fraudulent' in df_features.columns:
                X = df_features.drop(columns=['is_fraudulent'])
                y = df_features['is_fraudulent']
            else:
                raise ValueError("Target column 'is_fraudulent' not found in features")
            
            # Log feature information
            wandb.log({
                "feature_count": X.shape[1],
                "sample_count": X.shape[0],
                "fraud_ratio": y.mean(),
                "feature_strategy": strategy,
                "sweep_stage": stage
            })
            
            # Initialize and train model
            autoencoder = BaselineAutoencoder(pipeline_config)
            
            # Train the model
            results = autoencoder.train()
            history = results['history']
            
            # Evaluate model
            from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
            
            # Get predictions
            predictions = autoencoder.model.predict(X)
            anomaly_scores = autoencoder.predict_anomaly_scores(X)
            
            # Calculate metrics
            roc_auc = roc_auc_score(y, anomaly_scores)
            
            # Use threshold to get binary predictions
            threshold = np.percentile(anomaly_scores, config['model']['threshold'])
            binary_predictions = (anomaly_scores > threshold).astype(int)
            
            precision = precision_score(y, binary_predictions, zero_division=0)
            recall = recall_score(y, binary_predictions, zero_division=0)
            
            # Find best epoch
            best_epoch = np.argmin(history['loss'])
            
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
            
            wandb.log(metrics)
            
            # Log confusion matrix
            cm = confusion_matrix(y, binary_predictions)
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, 
                    y_true=y, 
                    preds=binary_predictions
                )
            })
            
            # Log training history
            for epoch in range(len(history['loss'])):
                wandb.log({
                    "epoch": epoch,
                    "loss": history['loss'][epoch],
                    "val_loss": history.get('val_loss', [0])[epoch] if epoch < len(history.get('val_loss', [])) else 0
                })
            
            logger.info(f"Training completed successfully with ROC AUC: {roc_auc:.4f}")
            
            return True, roc_auc, metrics
            
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        return False, 0.0, {}

def run_broad_sweep(entity: Optional[str] = None) -> List[Tuple[Dict, float]]:
    """
    Run broad hyperparameter sweep (Stage 2.1).
    
    Args:
        entity: W&B entity (username/team)
        
    Returns:
        List of (config, roc_auc) tuples sorted by performance
    """
    print("Stage 2.1: Broad Hyperparameter Sweep")
    print("=" * 60)
    
    # Load best features configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config("best_features")
    
    # Define hyperparameter combinations
    hyperparams = {
        'latent_dim': [8, 16, 32],
        'learning_rate': [0.01, 0.005, 0.001],  # Increased learning rates
        'activation_fn': ['relu', 'leaky_relu'],
        'batch_size': [64, 128],
        'threshold': [90, 95]
    }
    
    # Set epochs for broad sweep
    config['model']['epochs'] = 10
    
    results = []
    
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
    
    # Set W&B project
    os.environ["WANDB_PROJECT"] = args.project
    
    try:
        if args.stage == "all" or args.stage == "broad":
            # Stage 2.1: Broad sweep
            broad_results = run_broad_sweep(args.entity)
            
            if args.stage == "broad":
                return
        
        if args.stage == "all" or args.stage == "refined":
            # Stage 2.2: Refined sweep
            if args.stage == "refined":
                # Load previous results if running refined only
                broad_results = []  # You might want to load from file
            refined_results = run_refined_sweep(broad_results, args.entity)
            
            if args.stage == "refined":
                return
        
        if args.stage == "all" or args.stage == "final":
            # Stage 2.3: Final training
            if args.stage == "final":
                # Load previous results if running final only
                refined_results = []  # You might want to load from file
            
            best_config, best_roc_auc = run_final_training(refined_results, args.entity)
            
            if best_config:
                print(f"\nHyperparameter sweep completed!")
                print(f"   Best ROC AUC: {best_roc_auc:.4f}")
                print(f"   Best configuration saved to configs/final_config.yaml")
                print(f"   Next step: Run final training with 'python train_final_model.py'")
            else:
                print(f"\nHyperparameter sweep failed! No successful configurations.")
                
    except KeyboardInterrupt:
        print("\nHyperparameter sweep interrupted by user")
    except Exception as e:
        logger.error(f"Hyperparameter sweep failed: {str(e)}")
        print(f"\nHyperparameter sweep failed: {str(e)}")

if __name__ == "__main__":
    main() 