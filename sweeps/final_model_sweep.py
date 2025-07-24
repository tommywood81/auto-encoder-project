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
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from typing import Dict, List, Tuple, Optional, Any
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
            
            # Disable automatic model saving to prevent autoencoder.h5 creation
            autoencoder.config.model.save_model = False
            logger.info("Disabled automatic model saving to prevent autoencoder.h5 creation")
            
            # Train the model
            logger.info(f"Starting model training...")
            results = autoencoder.train()
            history = results['history']
            final_loss = history.history['loss'][-1] if 'loss' in history.history else 0.0
            logger.info(f"Training completed. Final loss: {final_loss:.6f}")
            logger.info(f"Training history length: {len(history.history['loss'])} epochs")
            
            # Evaluate model
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
            
            # Save model if this is the best one so far
            # We'll pass the autoencoder object back to save later
            logger.info(f"Training completed successfully with ROC AUC: {roc_auc:.4f}")
            logger.info(f"Run URL: {run.url}")
            
            return True, roc_auc, final_metrics, autoencoder
            
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, 0.0, {}, None

def train_model_with_config_no_wandb(config: Dict) -> Tuple[bool, float, Dict, Any]:
    """
    Train model with specific hyperparameters WITHOUT W&B logging.
    
    Args:
        config: Model configuration
        
    Returns:
        Tuple of (success, roc_auc, metrics_dict, autoencoder)
    """
    logger.info(f"Starting training without W&B logging")
    
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
        
        # Disable automatic model saving to prevent autoencoder.h5 creation
        autoencoder.config.model.save_model = False
        logger.info("Disabled automatic model saving to prevent autoencoder.h5 creation")
        
        # Train the model
        logger.info(f"Starting model training...")
        results = autoencoder.train()
        history = results['history']
        final_loss = history.history['loss'][-1] if 'loss' in history.history else 0.0
        logger.info(f"Training completed. Final loss: {final_loss:.6f}")
        
        # Evaluate model
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
        
        precision = precision_score(y, binary_predictions, zero_division=0)
        recall = recall_score(y, binary_predictions, zero_division=0)
        f1 = f1_score(y, binary_predictions, zero_division=0)
        accuracy = accuracy_score(y, binary_predictions)
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        # Find best epoch
        loss_history = history.history['loss']
        best_epoch = np.argmin(loss_history)
        logger.info(f"Best epoch: {best_epoch} (loss: {loss_history[best_epoch]:.6f})")
        
        # Final metrics
        final_metrics = {
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "threshold": threshold,
            "best_epoch": best_epoch,
            "final_loss": loss_history[-1],
            "best_loss": loss_history[best_epoch],
            "total_epochs": len(loss_history),
            "feature_count": X.shape[1],
            "learning_rate": config['model']['learning_rate'],
            "threshold_percentile": config['model']['threshold']
        }
        
        logger.info(f"Training completed successfully with ROC AUC: {roc_auc:.4f}")
        
        return True, roc_auc, final_metrics, autoencoder
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, 0.0, {}, None

def run_final_sweep(entity: Optional[str] = None):
    """
    Run final hyperparameter sweep around best model parameters.
    """
    logger.info("Final Model Hyperparameter Sweep")
    logger.info("=" * 80)
    
    print("FINAL MODEL HYPERPARAMETER SWEEP")
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
        'threshold': [95, 96, 97, 98, 99]  # Fine-tune around 95, removed 91/93, added 98
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
        print(f"COMBINATION {i+1}/{total_combinations}")
        print(f"{'='*80}")
        print(f"Parameters: {param_dict}")
        print(f"Started: {time.strftime('%H:%M:%S')}")
        print(f"Progress: {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")
        print(f"Remaining: {total_combinations - i - 1} combinations")
        print(f"{'='*80}")
        
        # Update config with current hyperparameters
        test_config = config.copy()
        for key, value in param_dict.items():
            test_config['model'][key] = value
        
        # Create run name
        run_name = f"final_sweep_{param_dict['learning_rate']}_{param_dict['threshold']}"
        
        # Train model (without W&B logging for individual runs)
        success, roc_auc, metrics, autoencoder = train_model_with_config_no_wandb(test_config)
        
        if success:
            results.append((test_config, roc_auc, param_dict, autoencoder))
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Learning Rate: {param_dict['learning_rate']}")
            print(f"Threshold: {param_dict['threshold']}")
        else:
            print(f"Failed")
        
        # Progress update
        elapsed_time = time.time() - start_time
        avg_time_per_combo = elapsed_time / (i + 1)
        remaining_time = avg_time_per_combo * (total_combinations - i - 1)
        
        print(f"Elapsed: {elapsed_time/60:.1f} min")
        print(f"Remaining: {remaining_time/60:.1f} min")
        print(f"ETA: {time.strftime('%H:%M:%S', time.localtime(time.time() + remaining_time))}")
        
        # Small delay between runs
        time.sleep(1)
    
    # Sort by ROC AUC (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nFinal sweep completed! Top 10 results:")
    print("=" * 80)
    for i, (config, roc_auc, params, autoencoder) in enumerate(results[:10]):
        print(f"   {i+1}. ROC AUC: {roc_auc:.4f} - {params}")
    
    # Log top 3 models to W&B
    logger.info("Logging top 3 models to W&B")
    for i, (config, roc_auc, params, autoencoder) in enumerate(results[:3]):
        try:
            with wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb'].get('entity'),
                group="top_models",
                name=f"rank_{i+1}_roc_{roc_auc:.4f}",
                tags=["top_model", "final_sweep"],
                config={
                    "rank": i+1,
                    "roc_auc": roc_auc,
                    "hyperparameters": params,
                    "model_config": config['model'],
                    "feature_strategy": config['features']['strategy']
                }
            ) as run:
                # Log overall metrics (no per-epoch logging)
                wandb.log({
                    "roc_auc": roc_auc,
                    "learning_rate": params['learning_rate'],
                    "threshold_percentile": params['threshold'],
                    "rank": i+1,
                    "model_architecture": {
                        "latent_dim": config['model']['latent_dim'],
                        "hidden_dims": config['model'].get('hidden_dims', [64, 32]),
                        "activation_fn": config['model']['activation_fn'],
                        "dropout_rate": config['model'].get('dropout_rate', 0.2)
                    },
                    "training_config": {
                        "batch_size": config['model']['batch_size'],
                        "epochs": config['model']['epochs'],
                        "early_stopping": config['model']['early_stopping']
                    }
                })
                
                logger.info(f"Top {i+1} model logged to W&B: {run.url}")
                print(f"Top {i+1} model logged to W&B: {run.url}")
                
        except Exception as e:
            logger.warning(f"Failed to log top {i+1} model to W&B: {e}")
            print(f"Warning: Could not log top {i+1} model to W&B: {e}")
    
    # Save best configuration and model
    if results:
        best_config, best_roc_auc, best_params, best_autoencoder = results[0]
        logger.info(f"Best configuration found: {best_params}")
        logger.info(f"Best ROC AUC: {best_roc_auc:.4f}")
        
        # Save best configuration
        config_loader = ConfigLoader(config_dir="configs")
        config_path = Path("configs/final_optimized_config.yaml")
        model_info_path = Path("models/final_model_info.yaml")
        previous_roc_auc = None
        if model_info_path.exists():
            with open(model_info_path, 'r') as f:
                model_info = yaml.safe_load(f)
                previous_roc_auc = model_info.get('roc_auc', None)
        if previous_roc_auc is None and config_path.exists():
            with open(config_path, 'r') as f:
                config_yaml = yaml.safe_load(f)
                previous_roc_auc = config_yaml.get('roc_auc', None)
        if previous_roc_auc is not None:
            print(f"Previous best ROC AUC: {previous_roc_auc}")
            print(f"New best ROC AUC: {best_roc_auc}")
        if previous_roc_auc is None or best_roc_auc > previous_roc_auc:
            config_loader.save_config(best_config, "final_optimized_config")
            print("\nBest config saved to configs/final_optimized_config.yaml.")
            print("Run run_pipeline.py --config configs/final_optimized_config.yaml to reproduce the production model.")
        else:
            print("\nBest config NOT updated because the new ROC AUC is not better than the previous best.")
        
        # Check if we have a better model than the existing final model
        current_final_roc_auc = 0.0
        try:
            final_model_info_path = Path("models/final_model_info.yaml")
            if final_model_info_path.exists():
                with open(final_model_info_path, 'r') as f:
                    current_final_info = yaml.safe_load(f)
                    current_final_roc_auc = current_final_info.get('roc_auc', 0.0)
                    logger.info(f"Current final model ROC AUC: {current_final_roc_auc:.4f}")
            else:
                logger.info("No existing final_model_info.yaml found - will save first successful model")
        except Exception as e:
            logger.warning(f"Could not read current final model info: {e}")
        
        # Only save if this model is better than the current final model
        if best_roc_auc > current_final_roc_auc:
            logger.info(f"New final model found! ROC AUC: {best_roc_auc:.4f} > {current_final_roc_auc:.4f}")
            
            if best_autoencoder:
                logger.info("Saving new final model...")
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)
                
                # Save the trained model
                model_path = models_dir / "final_model.h5"
                best_autoencoder.model.save(model_path)
                logger.info(f"Final model saved to: {model_path}")
                
                # Save the scaler
                scaler_path = models_dir / "final_model_scaler.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(best_autoencoder.scaler, f)
                logger.info(f"Final scaler saved to: {scaler_path}")
                
                # Save comprehensive model info from sweep
                model_info = {
                    # Model file paths
                    'model_path': str(model_path),
                    'scaler_path': str(scaler_path),
                    
                    # Performance metrics
                    'roc_auc': float(best_roc_auc),
                    'precision': float(best_params.get('precision', 0.0)),
                    'recall': float(best_params.get('recall', 0.0)),
                    'f1_score': float(best_params.get('f1_score', 0.0)),
                    'accuracy': float(best_params.get('accuracy', 0.0)),
                    
                    # Model architecture
                    'latent_dim': best_config['model']['latent_dim'],
                    'hidden_dims': best_config['model'].get('hidden_dims', [64, 32]),
                    'activation_fn': best_config['model']['activation_fn'],
                    'dropout_rate': best_config['model'].get('dropout_rate', 0.2),
                    'l2_reg': best_config['model'].get('l2_reg', 0.001),
                    
                    # Training parameters
                    'learning_rate': best_params['learning_rate'],
                    'batch_size': best_config['model']['batch_size'],
                    'epochs': best_config['model']['epochs'],
                    'threshold_percentile': best_params['threshold'],
                    'early_stopping': best_config['model']['early_stopping'],
                    'patience': best_config['model'].get('patience', 10),
                    
                    # Feature engineering
                    'feature_strategy': best_config['features']['strategy'],
                    'feature_scaling': best_config['features'].get('scaling', 'standard'),
                    'feature_count': best_config.get('feature_count', 'unknown'),
                    
                    # Data configuration
                    'train_split': best_config.get('data', {}).get('train_split', 0.8),
                    'val_split': best_config.get('data', {}).get('val_split', 0.1),
                    'test_split': best_config.get('data', {}).get('test_split', 0.1),
                    'random_state': best_config.get('data', {}).get('random_state', 42),
                    
                    # Sweep information
                    'sweep_total_combinations': len(results),
                    'sweep_top_results': [
                        {
                            'rank': i+1,
                            'roc_auc': float(roc_auc),
                            'learning_rate': params['learning_rate'],
                            'threshold': params['threshold']
                        }
                        for i, (_, roc_auc, params, _) in enumerate(results[:5])  # Top 5 results
                    ],
                    'sweep_hyperparameter_ranges': {
                        'learning_rate': [0.005, 0.008, 0.01, 0.012, 0.015],
                        'threshold': [95, 96, 97, 98, 99]
                    },
                    
                    # Metadata
                    'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'best_hyperparameters': best_params,
                    'model_type': 'autoencoder',
                    'task': 'fraud_detection'
                }
                
                model_info_path = models_dir / "final_model_info.yaml"
                with open(model_info_path, 'w') as f:
                    yaml.dump(model_info, f, default_flow_style=False)
                logger.info(f"Final model info saved to: {model_info_path}")
                
                print(f"NEW FINAL MODEL SAVED!")
                print(f"   ROC AUC: {best_roc_auc:.4f} (previous final: {current_final_roc_auc:.4f})")
                print(f"Final model saved to: {model_path}")
                print(f"Final scaler saved to: {scaler_path}")
                print(f"Final model info saved to: {model_info_path}")
                
                # Log final model details to W&B in a separate group
                logger.info("Logging final model details to W&B group: final-model-details")
                try:
                    with wandb.init(
                        project=best_config['wandb']['project'],
                        entity=best_config['wandb'].get('entity'),
                        group="final-model-details",
                        name=f"final-model-{best_roc_auc:.4f}",
                        tags=["final_model", "production_ready"],
                        config={
                            "model_info": model_info,
                            "sweep_summary": {
                                "total_combinations": len(results),
                                "best_roc_auc": best_roc_auc,
                                "best_hyperparameters": best_params,
                                "top_5_results": [
                                    {
                                        'rank': i+1,
                                        'roc_auc': float(roc_auc),
                                        'learning_rate': params['learning_rate'],
                                        'threshold': params['threshold']
                                    }
                                    for i, (_, roc_auc, params, _) in enumerate(results[:5])
                                ]
                            }
                        }
                    ) as final_run:
                        # Log comprehensive final model information
                        wandb.log({
                            "final_model_roc_auc": best_roc_auc,
                            "final_model_precision": model_info['precision'],
                            "final_model_recall": model_info['recall'],
                            "final_model_f1_score": model_info['f1_score'],
                            "final_model_accuracy": model_info['accuracy'],
                            "final_model_learning_rate": best_params['learning_rate'],
                            "final_model_threshold": best_params['threshold'],
                            "final_model_latent_dim": model_info['latent_dim'],
                            "final_model_feature_count": model_info['feature_count'],
                            "final_model_feature_strategy": model_info['feature_strategy'],
                            "sweep_total_combinations": len(results),
                            "model_architecture": {
                                "latent_dim": model_info['latent_dim'],
                                "hidden_dims": model_info.get('hidden_dims', [64, 32]),
                                "activation_fn": model_info['activation_fn'],
                                "dropout_rate": model_info.get('dropout_rate', 0.2)
                            },
                            "training_config": {
                                "batch_size": model_info['batch_size'],
                                "epochs": model_info['epochs'],
                                "early_stopping": model_info['early_stopping'],
                                "patience": model_info['patience']
                            },
                            "data_config": {
                                "train_split": model_info['train_split'],
                                "val_split": model_info['val_split'],
                                "test_split": model_info['test_split'],
                                "random_state": model_info['random_state']
                            }
                        })
                        
                        # Log the model file as an artifact
                        model_artifact = wandb.Artifact(
                            name=f"final-model-{best_roc_auc:.4f}",
                            type="model",
                            description=f"Final fraud detection autoencoder with ROC AUC {best_roc_auc:.4f}"
                        )
                        model_artifact.add_file(str(model_path), name="final_model.h5")
                        model_artifact.add_file(str(scaler_path), name="final_model_scaler.pkl")
                        model_artifact.add_file(str(model_info_path), name="final_model_info.yaml")
                        final_run.log_artifact(model_artifact)
                        
                        logger.info(f"Final model details logged to W&B: {final_run.url}")
                        print(f"Final model details logged to W&B: {final_run.url}")
                        
                except Exception as e:
                    logger.warning(f"Failed to log final model details to W&B: {e}")
                    print(f"Warning: Could not log final model details to W&B: {e}")
                
            else:
                logger.warning("Final model found but autoencoder object is None - cannot save")
        else:
            logger.info(f"Current final model ({current_final_roc_auc:.4f}) is better than sweep result ({best_roc_auc:.4f})")
            print(f"Sweep completed - no new final model found")
            print(f"   Best sweep result: {best_roc_auc:.4f}")
            print(f"   Current final model: {current_final_roc_auc:.4f}")
        
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
            print(f"\nFinal model sweep completed!")
            print(f"   Best ROC AUC: {best_roc_auc:.4f}")
            print(f"   Best configuration saved to configs/final_optimized_config.yaml")
            print(f"   Check your W&B dashboard for detailed results")
        else:
            logger.error("Final model sweep failed")
            print(f"\nFinal model sweep failed!")
                
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