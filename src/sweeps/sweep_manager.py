"""
Simplified Sweep Manager for Fraud Detection
Three-stage optimization: broad → narrow → final tuning
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import wandb

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from features.feature_engineer import FeatureEngineer
from models.autoencoder import FraudAutoencoder
from utils.data_loader import load_and_split_data

logger = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    """Configuration for sweep stages."""
    name: str
    description: str
    configs: List[Dict[str, Any]]
    max_runs: int
    metric: str = 'test_auc'
    goal: str = 'maximize'


class SweepManager:
    """Manages the three-stage sweep process."""
    
    def __init__(self, project_name: str = "fraud-detection-sweeps"):
        """Initialize sweep manager."""
        self.project_name = project_name
        self.results = []
        
    def run_broad_sweep(self, data_path: str) -> List[Dict[str, Any]]:
        """Stage 1: Broad sweep to find promising architectures."""
        
        logger.info("=" * 80)
        logger.info("STAGE 1: BROAD SWEEP")
        logger.info("=" * 80)
        
        # Load data
        df_train, df_test = load_and_split_data(data_path)
        
        # Define broad configurations
        broad_configs = [
            # Small architecture
            {
                'name': 'small',
                'model': {
                    'latent_dim': 8,
                    'hidden_dims': [32, 16],
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 50,
                    'dropout_rate': 0.1,
                    'threshold_percentile': 95
                }
            },
            # Medium architecture
            {
                'name': 'medium',
                'model': {
                    'latent_dim': 16,
                    'hidden_dims': [64, 32],
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'epochs': 50,
                    'dropout_rate': 0.2,
                    'threshold_percentile': 95
                }
            },
            # Large architecture
            {
                'name': 'large',
                'model': {
                    'latent_dim': 32,
                    'hidden_dims': [128, 64, 32],
                    'learning_rate': 0.0005,
                    'batch_size': 128,
                    'epochs': 50,
                    'dropout_rate': 0.3,
                    'threshold_percentile': 95
                }
            },
            # Deep architecture
            {
                'name': 'deep',
                'model': {
                    'latent_dim': 24,
                    'hidden_dims': [256, 128, 64, 32],
                    'learning_rate': 0.0003,
                    'batch_size': 64,
                    'epochs': 50,
                    'dropout_rate': 0.2,
                    'threshold_percentile': 95
                }
            },
            # Wide architecture
            {
                'name': 'wide',
                'model': {
                    'latent_dim': 20,
                    'hidden_dims': [512, 256],
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 50,
                    'dropout_rate': 0.1,
                    'threshold_percentile': 95
                }
            }
        ]
        
        results = []
        
        for config in broad_configs:
            logger.info(f"Testing configuration: {config['name']}")
            
            try:
                # Feature engineering
                feature_engineer = FeatureEngineer({})
                df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
                
                # Model training
                autoencoder = FraudAutoencoder(config['model'])
                X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
                y_train = df_train_features['is_fraudulent'].values
                y_test = df_test_features['is_fraudulent'].values
                
                # Train model
                result = autoencoder.train(X_train, X_test, y_train, y_test)
                
                # Store results
                config_result = {
                    'name': config['name'],
                    'config': config['model'],
                    'test_auc': result['test_auc'],
                    'threshold': result['threshold']
                }
                results.append(config_result)
                
                logger.info(f"Configuration {config['name']}: AUC = {result['test_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Error with configuration {config['name']}: {e}")
                continue
        
        # Sort by performance
        results.sort(key=lambda x: x['test_auc'], reverse=True)
        
        logger.info("\nBroad sweep results:")
        for i, result in enumerate(results):
            logger.info(f"{i+1}. {result['name']}: AUC = {result['test_auc']:.4f}")
        
        # Save results
        self._save_results(results, "broad_sweep_results.json")
        
        return results
    
    def run_narrow_sweep(self, data_path: str, top_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 2: Narrow sweep on top 3 configurations."""
        
        logger.info("=" * 80)
        logger.info("STAGE 2: NARROW SWEEP")
        logger.info("=" * 80)
        
        # Load data
        df_train, df_test = load_and_split_data(data_path)
        
        # Take top 3 configurations
        top_3 = top_configs[:3]
        
        # Generate variations for each top configuration
        narrow_configs = []
        
        for base_config in top_3:
            base_name = base_config['name']
            base_model = base_config['config']
            
            # Learning rate variations
            for lr in [0.0001, 0.0005, 0.001, 0.005]:
                config = {
                    'name': f"{base_name}_lr_{lr}",
                    'model': {**base_model, 'learning_rate': lr}
                }
                narrow_configs.append(config)
            
            # Batch size variations
            for batch_size in [16, 32, 64, 128]:
                config = {
                    'name': f"{base_name}_batch_{batch_size}",
                    'model': {**base_model, 'batch_size': batch_size}
                }
                narrow_configs.append(config)
            
            # Dropout variations
            for dropout in [0.0, 0.1, 0.2, 0.3]:
                config = {
                    'name': f"{base_name}_dropout_{dropout}",
                    'model': {**base_model, 'dropout_rate': dropout}
                }
                narrow_configs.append(config)
        
        results = []
        
        for config in narrow_configs:
            logger.info(f"Testing configuration: {config['name']}")
            
            try:
                # Feature engineering
                feature_engineer = FeatureEngineer({})
                df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
                
                # Model training
                autoencoder = FraudAutoencoder(config['model'])
                X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
                y_train = df_train_features['is_fraudulent'].values
                y_test = df_test_features['is_fraudulent'].values
                
                # Train model
                result = autoencoder.train(X_train, X_test, y_train, y_test)
                
                # Store results
                config_result = {
                    'name': config['name'],
                    'config': config['model'],
                    'test_auc': result['test_auc'],
                    'threshold': result['threshold']
                }
                results.append(config_result)
                
                logger.info(f"Configuration {config['name']}: AUC = {result['test_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Error with configuration {config['name']}: {e}")
                continue
        
        # Sort by performance
        results.sort(key=lambda x: x['test_auc'], reverse=True)
        
        logger.info("\nNarrow sweep results:")
        for i, result in enumerate(results):
            logger.info(f"{i+1}. {result['name']}: AUC = {result['test_auc']:.4f}")
        
        # Save results
        self._save_results(results, "narrow_sweep_results.json")
        
        return results
    
    def run_final_tuning(self, data_path: str, best_config: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Final tuning of the best configuration."""
        
        logger.info("=" * 80)
        logger.info("STAGE 3: FINAL TUNING")
        logger.info("=" * 80)
        
        # Load data
        df_train, df_test = load_and_split_data(data_path)
        
        # Generate fine-tuned variations
        base_model = best_config['config']
        
        # Fine-tune learning rate
        best_lr = base_model['learning_rate']
        lr_variations = [best_lr * 0.5, best_lr * 0.8, best_lr, best_lr * 1.2, best_lr * 1.5]
        
        # Fine-tune threshold
        threshold_variations = [90, 92, 94, 95, 96, 98]
        
        # Fine-tune epochs
        epoch_variations = [30, 50, 75, 100]
        
        results = []
        
        for lr in lr_variations:
            for threshold in threshold_variations:
                for epochs in epoch_variations:
                    config = {
                        'name': f"final_lr_{lr}_thresh_{threshold}_epochs_{epochs}",
                        'model': {
                            **base_model,
                            'learning_rate': lr,
                            'threshold_percentile': threshold,
                            'epochs': epochs
                        }
                    }
                    
                    logger.info(f"Testing configuration: {config['name']}")
                    
                    try:
                        # Feature engineering
                        feature_engineer = FeatureEngineer({})
                        df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
                        
                        # Model training
                        autoencoder = FraudAutoencoder(config['model'])
                        X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
                        y_train = df_train_features['is_fraudulent'].values
                        y_test = df_test_features['is_fraudulent'].values
                        
                        # Train model
                        result = autoencoder.train(X_train, X_test, y_train, y_test)
                        
                        # Store results
                        config_result = {
                            'name': config['name'],
                            'config': config['model'],
                            'test_auc': result['test_auc'],
                            'threshold': result['threshold']
                        }
                        results.append(config_result)
                        
                        logger.info(f"Configuration {config['name']}: AUC = {result['test_auc']:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error with configuration {config['name']}: {e}")
                        continue
        
        # Sort by performance
        results.sort(key=lambda x: x['test_auc'], reverse=True)
        
        logger.info("\nFinal tuning results:")
        for i, result in enumerate(results):
            logger.info(f"{i+1}. {result['name']}: AUC = {result['test_auc']:.4f}")
        
        # Save results
        self._save_results(results, "final_tuning_results.json")
        
        # Return best configuration
        best_result = results[0] if results else None
        
        if best_result:
            logger.info(f"\nBest configuration found:")
            logger.info(f"Name: {best_result['name']}")
            logger.info(f"AUC: {best_result['test_auc']:.4f}")
            logger.info(f"Threshold: {best_result['threshold']:.6f}")
            
            # Save best configuration
            self._save_best_config(best_result)
        
        return best_result
    
    def run_complete_sweep(self, data_path: str) -> Dict[str, Any]:
        """Run the complete three-stage sweep process."""
        
        logger.info("Starting complete three-stage sweep process...")
        
        # Stage 1: Broad sweep
        broad_results = self.run_broad_sweep(data_path)
        
        if not broad_results:
            raise ValueError("Broad sweep failed to produce results")
        
        # Stage 2: Narrow sweep
        narrow_results = self.run_narrow_sweep(data_path, broad_results)
        
        if not narrow_results:
            raise ValueError("Narrow sweep failed to produce results")
        
        # Stage 3: Final tuning
        final_result = self.run_final_tuning(data_path, narrow_results[0])
        
        if not final_result:
            raise ValueError("Final tuning failed to produce results")
        
        logger.info("=" * 80)
        logger.info("SWEEP PROCESS COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Best AUC achieved: {final_result['test_auc']:.4f}")
        
        return final_result
    
    def _save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save sweep results to file."""
        os.makedirs('results', exist_ok=True)
        filepath = os.path.join('results', filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
    
    def _save_best_config(self, best_result: Dict[str, Any]):
        """Save best configuration to file."""
        os.makedirs('configs', exist_ok=True)
        filepath = os.path.join('configs', 'best_config.yaml')
        
        config_data = {
            'name': best_result['name'],
            'description': 'Best configuration from three-stage sweep',
            'model': best_result['config'],
            'performance': {
                'test_auc': best_result['test_auc'],
                'threshold': best_result['threshold']
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        logger.info(f"Best configuration saved to: {filepath}") 