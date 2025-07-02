#!/usr/bin/env python3
"""
Final Model Training Script with W&B Integration

This script trains the final model using the best configuration from the hyperparameter sweep
and saves it for production deployment.
"""

import wandb
import yaml
import os
import sys
import time
import logging
import numpy as np
from typing import Dict, Optional
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

def train_final_model(entity: Optional[str] = None) -> bool:
    """
    Train the final model with best configuration.
    
    Args:
        entity: W&B entity (username/team)
        
    Returns:
        True if training completed successfully
    """
    print("Training Final Model")
    print("=" * 60)
    
    # Load final configuration
    config_loader = ConfigLoader()
    try:
        config = config_loader.load_config("final_config")
    except FileNotFoundError:
        logger.warning("Final config not found, using best features config")
        config = config_loader.load_config("best_features")
        # Update for final training
        config['model']['epochs'] = 50
        config['model']['early_stopping'] = True
    
    # Initialize W&B
    wandb_config = config.copy()
    wandb_config['wandb']['tags'] = ["final_model", "production"]
    
    if entity:
        wandb_config['wandb']['entity'] = entity
    
    run_name = f"final_model_{wandb_config['model']['latent_dim']}_{wandb_config['model']['learning_rate']}"
    
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
                "final_training": True
            })
            
            # Initialize and train model
            autoencoder = BaselineAutoencoder(pipeline_config)
            
            print(f"Training with configuration:")
            print(f"   Latent dim: {config['model']['latent_dim']}")
            print(f"   Learning rate: {config['model']['learning_rate']}")
            print(f"   Epochs: {config['model']['epochs']}")
            print(f"   Batch size: {config['model']['batch_size']}")
            print(f"   Activation: {config['model']['activation_fn']}")
            print(f"   Threshold: {config['model']['threshold']}")
            print(f"   Early stopping: {config['model'].get('early_stopping', False)}")
            
            # Train the model
            results = autoencoder.train()
            history = results['history']
            
            # Evaluate model
            from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
            
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
            best_epoch = np.argmin(history.history['loss'])
            
            # Log metrics
            metrics = {
                "final_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "threshold": threshold,
                "best_epoch": best_epoch,
                "final_loss": history.history['loss'][-1],
                "best_loss": history.history['loss'][best_epoch]
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
            
            # Log classification report
            report = classification_report(y, binary_predictions, output_dict=True)
            wandb.log({
                "classification_report": wandb.Table(
                    columns=["class", "precision", "recall", "f1-score", "support"],
                    data=[
                        [k, v.get('precision', 0), v.get('recall', 0), v.get('f1-score', 0), v.get('support', 0)]
                        for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']
                    ]
                )
            })
            
            # Log training history
            for epoch in range(len(history.history['loss'])):
                wandb.log({
                    "epoch": epoch,
                    "loss": history.history['loss'][epoch],
                    "val_loss": history.history.get('val_loss', [0])[epoch] if epoch < len(history.history.get('val_loss', [])) else 0
                })
            
            # Save model
            model_path = Path("models/final_model.h5")
            model_path.parent.mkdir(exist_ok=True)
            autoencoder.save_model()
            
            # Move the saved model to the desired location
            import shutil
            default_model_path = Path("models/autoencoder.h5")
            if default_model_path.exists():
                shutil.move(str(default_model_path), str(model_path))
            
            # Create model artifact
            artifact = wandb.Artifact(
                name="final-fraud-detection-model",
                type="model",
                description="Final trained autoencoder for fraud detection"
            )
            artifact.add_file(str(model_path))
            wandb.log_artifact(artifact)
            
            # Save model info
            model_info = {
                "model_path": str(model_path),
                "feature_strategy": strategy,
                "feature_count": X.shape[1],
                "best_auc": roc_auc,
                "threshold": threshold,
                "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": config
            }
            
            info_path = Path("models/model_info.yaml")
            with open(info_path, 'w') as f:
                yaml.dump(model_info, f, default_flow_style=False, indent=2)
            
            print(f"\nFinal model training completed successfully!")
            print(f"   Best ROC AUC: {roc_auc:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   Model saved to: {model_path}")
            print(f"   Model info saved to: {info_path}")
            
            return True
            
    except Exception as e:
        logger.error(f"Final training failed: {str(e)}")
        print(f"Final training failed: {str(e)}")
        return False

def main():
    """Main function to run final model training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train final model with best configuration")
    parser.add_argument("--entity", type=str, help="W&B entity (username/team)")
    parser.add_argument("--project", type=str, default="fraud-detection-autoencoder", 
                       help="W&B project name")
    
    args = parser.parse_args()
    
    # Set W&B project
    os.environ["WANDB_PROJECT"] = args.project
    
    try:
        success = train_final_model(args.entity)
        
        if success:
            print(f"\nFinal model training completed!")
            print(f"   Check W&B dashboard for detailed results")
            print(f"   Model ready for deployment")
        else:
            print(f"\nFinal model training failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nFinal training interrupted by user")
    except Exception as e:
        logger.error(f"Final training failed: {str(e)}")
        print(f"\nFinal training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 