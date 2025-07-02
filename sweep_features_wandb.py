#!/usr/bin/env python3
"""
Feature Sweep Script with W&B Integration for Fraud Detection Pipeline

This script runs the fraud detection pipeline with all available feature strategies
and tracks results using Weights & Biases for experiment comparison.
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

# Available strategies
STRATEGIES = [
    "baseline",
    "temporal", 
    "behavioural",
    "demographic_risk",
    "combined"
]

# Strategy descriptions for display
STRATEGY_DESCRIPTIONS = {
    "baseline": "Basic transaction features only (9 features)",
    "temporal": "Basic features + temporal patterns (10 features)",
    "behavioural": "Core features + amount per item (10 features)",
    "demographic_risk": "Core features + customer age risk scores (10 features)",
    "combined": "All unique features from all strategies (no duplicates)"
}

def train_model_with_strategy(strategy: str, config: Dict, entity: Optional[str] = None) -> Tuple[bool, float, Dict]:
    """
    Train model with a specific feature strategy and track with W&B.
    
    Args:
        strategy: Feature strategy to use
        config: Model configuration
        entity: W&B entity (username/team)
        
    Returns:
        Tuple of (success, roc_auc, metrics_dict)
    """
    logger.info(f"Starting training with strategy: {strategy}")
    print(f"   Strategy: {strategy}")
    print(f"   Description: {STRATEGY_DESCRIPTIONS.get(strategy, 'No description')}")
    
    # Initialize W&B
    wandb_config = config.copy()
    wandb_config['features']['strategy'] = strategy
    wandb_config['wandb']['tags'] = ["feature_sweep"]
    
    if entity:
        wandb_config['wandb']['entity'] = entity
    
    run_name = f"feature_sweep_{strategy}"
    
    try:
        with wandb.init(
            project=wandb_config['wandb']['project'],
            entity=wandb_config['wandb'].get('entity'),
            config=wandb_config,
            name=run_name,
            tags=wandb_config['wandb']['tags']
        ) as run:
            
            # Load data and generate features
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
                "feature_strategy": strategy
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
            
            # Log metrics
            metrics = {
                "final_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "threshold": threshold,
                "best_epoch": len(history.history['loss'])
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
            for epoch in range(len(history.history['loss'])):
                wandb.log({
                    "epoch": epoch,
                    "loss": history.history['loss'][epoch],
                    "val_loss": history.history.get('val_loss', [0])[epoch] if epoch < len(history.history.get('val_loss', [])) else 0
                })
            
            logger.info(f"Strategy {strategy} completed successfully with ROC AUC: {roc_auc:.4f}")
            print(f"   SUCCESS: {strategy} - ROC AUC: {roc_auc:.4f}")
            
            return True, roc_auc, metrics
            
    except Exception as e:
        error_msg = f"Training failed for {strategy}: {str(e)}"
        logger.error(error_msg)
        print(f"   FAILED: {strategy} - {str(e)}")
        return False, 0.0, {}

def run_feature_sweep(entity: Optional[str] = None) -> Dict[str, Tuple[bool, float, Dict]]:
    """
    Run feature sweep with all strategies.
    
    Args:
        entity: W&B entity (username/team)
        
    Returns:
        Dictionary of results for each strategy
    """
    print("Starting Feature Sweep with W&B Integration")
    print("=" * 80)
    
    # Load baseline configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config("baseline")
    
    results = {}
    
    for strategy in STRATEGIES:
        print(f"\nTesting Strategy: {strategy}")
        print("-" * 60)
        
        success, roc_auc, metrics = train_model_with_strategy(strategy, config, entity)
        results[strategy] = (success, roc_auc, metrics)
        
        # Small delay between runs
        time.sleep(2)
    
    return results

def save_best_features_config(results: Dict[str, Tuple[bool, float, Dict]]) -> Optional[str]:
    """
    Save the best performing feature strategy to config.
    
    Args:
        results: Results from feature sweep
        
    Returns:
        Best strategy name or None if no successful runs
    """
    # Find best performing strategy
    successful_results = [(s, r[1]) for s, r in results.items() if r[0]]
    
    if not successful_results:
        logger.warning("No strategies completed successfully!")
        return None
    
    # Sort by ROC AUC (descending)
    successful_results.sort(key=lambda x: x[1], reverse=True)
    best_strategy, best_roc = successful_results[0]
    
    # Update best_features.yaml
    config_loader = ConfigLoader()
    updates = {
        "features": {"strategy": best_strategy},
        "wandb": {"tags": ["best_features", f"best_strategy_{best_strategy}"]}
    }
    
    config_loader.update_config("best_features", updates)
    
    logger.info(f"Best strategy '{best_strategy}' saved to config (ROC AUC: {best_roc:.4f})")
    print(f"\nBest strategy '{best_strategy}' saved to config (ROC AUC: {best_roc:.4f})")
    
    return best_strategy

def print_results(results: Dict[str, Tuple[bool, float, Dict]]):
    """Print the results in a formatted table."""
    print("\n" + "="*80)
    print("FEATURE SWEEP RESULTS")
    print("="*80)
    
    # Sort by ROC AUC (descending)
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1][1] if x[1][0] else 0.0, 
        reverse=True
    )
    
    print(f"{'Strategy':<20} {'Status':<10} {'ROC AUC':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 80)
    
    for strategy, (success, roc_auc, metrics) in sorted_results:
        status = "SUCCESS" if success else "FAILED"
        roc_str = f"{roc_auc:.4f}" if success else "N/A"
        precision_str = f"{metrics.get('precision', 0):.4f}" if success else "N/A"
        recall_str = f"{metrics.get('recall', 0):.4f}" if success else "N/A"
        
        print(f"{strategy:<20} {status:<10} {roc_str:<10} {precision_str:<10} {recall_str:<10}")
    
    print("-" * 80)
    
    # Find best performing strategy
    successful_results = [(s, r[1]) for s, r in sorted_results if r[0]]
    
    if successful_results:
        best_strategy, best_roc = successful_results[0]
        print(f"\nBEST PERFORMING STRATEGY: {best_strategy}")
        print(f"   ROC AUC: {best_roc:.4f}")
        
        # Compare with baseline
        baseline_result = results.get("baseline")
        if baseline_result and baseline_result[0]:
            baseline_roc = baseline_result[1]
            if baseline_roc > 0:
                improvement = ((best_roc - baseline_roc) / baseline_roc) * 100
                
                if best_strategy == "baseline":
                    print(f"   Baseline is the best strategy!")
                elif improvement > 0:
                    print(f"   Improvement over baseline: +{improvement:.2f}%")
                else:
                    print(f"   Performance vs baseline: {improvement:.2f}%")
            else:
                print(f"   Baseline ROC AUC is zero - cannot calculate improvement")
        else:
            print(f"   Could not compare with baseline (baseline failed)")
    else:
        print(f"\nNo strategies completed successfully!")

def main():
    """Main function to run the feature sweep."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run feature sweep with W&B integration")
    parser.add_argument("--entity", type=str, help="W&B entity (username/team)")
    parser.add_argument("--project", type=str, default="fraud-detection-autoencoder", 
                       help="W&B project name")
    
    args = parser.parse_args()
    
    # Set W&B project
    os.environ["WANDB_PROJECT"] = args.project
    
    try:
        # Run feature sweep
        results = run_feature_sweep(args.entity)
        
        # Print results
        print_results(results)
        
        # Save best features config
        best_strategy = save_best_features_config(results)
        
        if best_strategy:
            print(f"\nFeature sweep completed! Best strategy '{best_strategy}' saved to config.")
            print(f"   Next step: Run hyperparameter tuning with 'python sweep_parameters_wandb.py'")
        else:
            print(f"\nFeature sweep failed! No successful strategies found.")
            
    except KeyboardInterrupt:
        print("\nFeature sweep interrupted by user")
    except Exception as e:
        logger.error(f"Feature sweep failed: {str(e)}")
        print(f"\nFeature sweep failed: {str(e)}")

if __name__ == "__main__":
    main() 