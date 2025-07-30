"""
Smart sweep manager for three-stage hyperparameter optimization.
Implements auto-promotion logic and clean W&B logging.
"""

import os
import logging
import yaml
import wandb
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config_loader import ConfigLoader
from src.features.feature_engineer import FeatureEngineer
from src.models.autoencoder import FraudAutoencoder
from src.utils.data_loader import load_and_split_data

logger = logging.getLogger(__name__)


class SweepManager:
    """Manages three-stage hyperparameter optimization with auto-promotion."""
    
    def __init__(self, config_path: str):
        """Initialize sweep manager with configuration."""
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        self.stage = self.config.get('sweep.stage', 'broad')
        self.best_auc = 0.0
        self.best_config = None
        self.results = []
        self.best_sweeps_file = "src/sweeps/best_sweeps.yaml"
        self.best_sweeps_data = self._load_best_sweeps()
        self.start_time = None
        self.total_steps = 0
        self.current_step = 0
    
    def run_complete_sweep(self) -> Dict[str, Any]:
        """Run complete three-stage sweep process."""
        overall_start_time = time.time()
        logger.info("Starting complete three-stage sweep process")
        logger.info(f"Overall start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate total steps across all stages
        total_broad_steps = self._calculate_total_steps("broad")
        total_narrow_steps = self._calculate_total_steps("narrow")
        total_final_steps = self._calculate_total_steps("final")
        total_overall_steps = total_broad_steps + total_narrow_steps + total_final_steps
        
        logger.info("=" * 60)
        logger.info("OVERALL SWEEP PROGRESS")
        logger.info("=" * 60)
        logger.info(f"Broad sweep: {total_broad_steps} configurations")
        logger.info(f"Narrow sweep: {total_narrow_steps} configurations")
        logger.info(f"Final sweep: {total_final_steps} configurations")
        logger.info(f"Total configurations: {total_overall_steps}")
        logger.info("=" * 60)
        
        # Stage 1: Broad sweep
        logger.info("=" * 60)
        logger.info("STAGE 1: BROAD SWEEP")
        logger.info("=" * 60)
        broad_results = self.run_broad_sweep()
        
        # Stage 2: Narrow sweep
        logger.info("=" * 60)
        logger.info("STAGE 2: NARROW SWEEP")
        logger.info("=" * 60)
        narrow_results = self.run_narrow_sweep()
        
        # Stage 3: Final sweep
        logger.info("=" * 60)
        logger.info("STAGE 3: FINAL SWEEP")
        logger.info("=" * 60)
        final_results = self.run_final_sweep()
        
        # Overall completion summary
        overall_total_time = time.time() - overall_start_time
        overall_total_time_str = str(timedelta(seconds=int(overall_total_time)))
        
        logger.info("=" * 60)
        logger.info("COMPLETE SWEEP PROCESS FINISHED")
        logger.info("=" * 60)
        logger.info(f"Total time: {overall_total_time_str}")
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Best AUC: {self.best_auc:.4f}")
        logger.info(f"Best config saved to: configs/final_optimized_config.yaml")
        logger.info("=" * 60)
        
        # Print best sweeps summary
        self.print_best_sweeps_summary()
        
        return {
            'broad_results': broad_results,
            'narrow_results': narrow_results,
            'final_results': final_results,
            'best_auc': self.best_auc,
            'best_config': self.best_config
        }
    
    def run_broad_sweep(self) -> List[Dict[str, Any]]:
        """Run broad sweep testing different architectures."""
        # Start progress tracking
        self._start_progress_tracking("broad")
        
        architectures = self.config_loader.get('model.architectures', {})
        results = []
        
        # Load data
        df_train, df_test = load_and_split_data("data/cleaned/ecommerce_cleaned.csv")
        feature_engineer = FeatureEngineer(self.config_loader.get('features', {}))
        df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
        
        for arch_name, arch_config in architectures.items():
            logger.info(f"Testing architecture: {arch_name}")
            self._update_progress(f"Testing {arch_name}")
            
            # Initialize W&B run
            with wandb.init(
                project="fraud-detection-autoencoder",
                group="sweep_broad",
                name=f"broad_{arch_name}",
                config={
                        'architecture': arch_name,
                        'latent_dim': arch_config['latent_dim'],
                        'hidden_dims': arch_config['hidden_dims'],
                        'dropout_rate': arch_config['dropout_rate'],
                        'batch_size': self.config_loader.get('training.batch_size'),
                        'learning_rate': self.config_loader.get('training.learning_rate'),
                        'epochs': self.config_loader.get('training.epochs')
                }
            ) as run:
                
                # Create model config
                model_config = {
                    'latent_dim': arch_config['latent_dim'],
                    'hidden_dims': arch_config['hidden_dims'],
                    'dropout_rate': arch_config['dropout_rate'],
                    'learning_rate': self.config_loader.get('training.learning_rate'),
                    'batch_size': self.config_loader.get('training.batch_size'),
                    'epochs': self.config_loader.get('training.epochs'),
                    'early_stopping': self.config_loader.get('training.early_stopping'),
                    'patience': self.config_loader.get('training.patience'),
                    'reduce_lr': self.config_loader.get('training.reduce_lr'),
                    'threshold_percentile': self.config_loader.get('features.threshold_percentile')
                }
                
                # Train model
                autoencoder = FraudAutoencoder(model_config)
                X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
                y_train = df_train_features['is_fraudulent'].values
                y_test = df_test_features['is_fraudulent'].values
                
                results_dict = autoencoder.train(X_train, X_test, y_train, y_test)
                    
                # Log results
                wandb.log({
                    'auc_roc': results_dict['test_auc'],
                    'threshold': results_dict['threshold'],
                    'train_loss': results_dict.get('train_loss', 0),
                    'val_loss': results_dict.get('val_loss', 0)
                })
                
                # Update best sweep result
                self._update_best_sweep_result(
                    'broad_sweep', 
                    run.id, 
                    arch_name, 
                    results_dict, 
                    model_config
                )
                
                # Store results
                result = {
                    'architecture': arch_name,
                    'config': model_config,
                    'auc': results_dict['test_auc'],
                    'threshold': results_dict['threshold']
                }
                results.append(result)
                
                logger.info(f"Architecture {arch_name}: AUC = {results_dict['test_auc']:.4f}")
        
        # Sort by AUC and select top configurations
        results.sort(key=lambda x: x['auc'], reverse=True)
        top_k = self.config_loader.get('sweep.top_k', 3)
        top_results = results[:top_k]
        
        # Create narrow sweep config
        self._create_narrow_config(top_results)
        
        logger.info(f"Broad sweep complete. Top {top_k} architectures selected.")
        for i, result in enumerate(top_results):
            logger.info(f"{i+1}. {result['architecture']}: AUC = {result['auc']:.4f}")
        
        # Finish progress tracking
        self._finish_progress_tracking("broad")
        
        return results
    
    def run_narrow_sweep(self) -> List[Dict[str, Any]]:
        """Run narrow sweep on top architectures."""
        # Start progress tracking
        self._start_progress_tracking("narrow")
        
        # Load narrow config
        narrow_config_path = self.config.get('sweep.output_config', 'configs/sweep_narrow.yaml')
        if not os.path.exists(narrow_config_path):
            raise FileNotFoundError(f"Narrow config not found. Run broad sweep first: {narrow_config_path}")
        
        narrow_config = ConfigLoader(narrow_config_path)
        architectures = narrow_config.get('model.architectures', {})
        
        if not architectures:
            raise ValueError("No architectures found in narrow config. Run broad sweep first.")
        
        results = []
        
        # Load data
        df_train, df_test = load_and_split_data("data/cleaned/ecommerce_cleaned.csv")
        feature_engineer = FeatureEngineer(narrow_config.get('features', {}))
        df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
        
        # Test hyperparameter combinations
        batch_sizes = narrow_config.get('training.batch_sizes', [64])
        learning_rates = narrow_config.get('training.learning_rates', [0.001])
        dropout_rates = narrow_config.get('training.dropout_rates', [0.2])
        
        for arch_name, arch_config in architectures.items():
            for batch_size in batch_sizes:
                for lr in learning_rates:
                    for dropout in dropout_rates:
                        logger.info(f"Testing: {arch_name}, batch={batch_size}, lr={lr}, dropout={dropout}")
                        self._update_progress(f"{arch_name}_b{batch_size}_lr{lr}_d{dropout}")
                        
                        # Initialize W&B run
                        with wandb.init(
                            project="fraud-detection-autoencoder",
                            group="sweep_narrow",
                            name=f"narrow_{arch_name}_b{batch_size}_lr{lr}_d{dropout}",
                            config={
                                'architecture': arch_name,
                                'batch_size': batch_size,
                                'learning_rate': lr,
                                'dropout_rate': dropout,
                                'latent_dim': arch_config['latent_dim'],
                                'hidden_dims': arch_config['hidden_dims']
                            }
                        ) as run:
                            
                            # Create model config
                            model_config = {
                                'latent_dim': arch_config['latent_dim'],
                                'hidden_dims': arch_config['hidden_dims'],
                                'dropout_rate': dropout,
                                'learning_rate': lr,
                                'batch_size': batch_size,
                                'epochs': narrow_config.get('training.epochs'),
                                'early_stopping': narrow_config.get('training.early_stopping'),
                                'patience': narrow_config.get('training.patience'),
                                'reduce_lr': narrow_config.get('training.reduce_lr'),
                                'threshold_percentile': narrow_config.get('features.threshold_percentile')
                            }
                                
                            # Train model
                            autoencoder = FraudAutoencoder(model_config)
                            X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
                            y_train = df_train_features['is_fraudulent'].values
                            y_test = df_test_features['is_fraudulent'].values
                            
                            results_dict = autoencoder.train(X_train, X_test, y_train, y_test)
                            
                            # Log results
                            wandb.log({
                                'auc_roc': results_dict['test_auc'],
                                'threshold': results_dict['threshold'],
                                'train_loss': results_dict.get('train_loss', 0),
                                'val_loss': results_dict.get('val_loss', 0)
                            })
                            
                            # Update best sweep result
                            self._update_best_sweep_result(
                                'narrow_sweep', 
                                run.id, 
                                arch_name, 
                                results_dict, 
                                model_config
                            )
                            
                            # Store results
                            result = {
                                'architecture': arch_name,
                                'config': model_config,
                                'auc': results_dict['test_auc'],
                                'threshold': results_dict['threshold']
                            }
                            results.append(result)
                            
                            logger.info(f"AUC = {results_dict['test_auc']:.4f}")
        
        # Sort by AUC and select best configuration
        results.sort(key=lambda x: x['auc'], reverse=True)
        best_result = results[0]
        
        # Create final sweep config
        self._create_final_config(best_result)
        
        logger.info(f"Narrow sweep complete. Best configuration:")
        logger.info(f"Architecture: {best_result['architecture']}")
        logger.info(f"AUC: {best_result['auc']:.4f}")
        
        # Finish progress tracking
        self._finish_progress_tracking("narrow")
        
        return results
    
    def run_final_sweep(self) -> List[Dict[str, Any]]:
        """Run final sweep for fine-tuning."""
        # Start progress tracking
        self._start_progress_tracking("final")
        
        # Load final config
        final_config_path = self.config.get('sweep.output_config', 'configs/sweep_final.yaml')
        if not os.path.exists(final_config_path):
            raise FileNotFoundError(f"Final config not found. Run narrow sweep first: {final_config_path}")
        
        final_config = ConfigLoader(final_config_path)
        
        results = []
        
        # Load data
        df_train, df_test = load_and_split_data("data/cleaned/ecommerce_cleaned.csv")
        feature_engineer = FeatureEngineer(final_config.get('features', {}))
        df_train_features, df_test_features = feature_engineer.fit_transform(df_train, df_test)
        
        # Test fine-tuning parameters
        threshold_percentiles = final_config.get('sweep.threshold_percentiles', [95])
        learning_rates = final_config.get('sweep.learning_rates', [0.001])
        epochs_list = final_config.get('sweep.epochs', [100])
        
        base_config = {
            'latent_dim': final_config.get('model.latent_dim'),
            'hidden_dims': final_config.get('model.hidden_dims'),
            'dropout_rate': final_config.get('model.dropout_rate'),
            'batch_size': final_config.get('training.batch_size'),
            'early_stopping': final_config.get('training.early_stopping'),
            'patience': final_config.get('training.patience'),
            'reduce_lr': final_config.get('training.reduce_lr')
        }
        
        for threshold in threshold_percentiles:
            for lr in learning_rates:
                for epochs in epochs_list:
                    logger.info(f"Testing: threshold={threshold}, lr={lr}, epochs={epochs}")
                    self._update_progress(f"th{threshold}_lr{lr}_e{epochs}")
                    
                    # Initialize W&B run
                    with wandb.init(
                        project="fraud-detection-autoencoder",
                        group="sweep_final",
                        name=f"final_th{threshold}_lr{lr}_e{epochs}",
                        config={
                            'threshold_percentile': threshold,
                            'learning_rate': lr,
                            'epochs': epochs,
                            **base_config
                        }
                    ) as run:
                        
                        # Create model config
                        model_config = {
                            **base_config,
                            'learning_rate': lr,
                            'epochs': epochs,
                            'threshold_percentile': threshold
                        }
                        
                        # Train model
                        autoencoder = FraudAutoencoder(model_config)
                        X_train, X_test = autoencoder.prepare_data(df_train_features, df_test_features)
                        y_train = df_train_features['is_fraudulent'].values
                        y_test = df_test_features['is_fraudulent'].values
                        
                        results_dict = autoencoder.train(X_train, X_test, y_train, y_test)
                        
                        # Log results
                        wandb.log({
                            'auc_roc': results_dict['test_auc'],
                            'threshold': results_dict['threshold'],
                            'train_loss': results_dict.get('train_loss', 0),
                            'val_loss': results_dict.get('val_loss', 0)
                        })
                        
                        # Update best sweep result
                        self._update_best_sweep_result(
                            'final_sweep', 
                            run.id, 
                            'final_optimized', 
                            results_dict, 
                            model_config
                        )
                        
                        # Store results
                        result = {
                            'config': model_config,
                            'auc': results_dict['test_auc'],
                            'threshold': results_dict['threshold']
                        }
                        results.append(result)
                        
                        logger.info(f"AUC = {results_dict['test_auc']:.4f}")
                        
                        # Check if this is the best so far
                        if results_dict['test_auc'] > self.best_auc:
                            self.best_auc = results_dict['test_auc']
                            self.best_config = model_config
                            logger.info(f"New best AUC: {self.best_auc:.4f}")
        
        # Sort by AUC
        results.sort(key=lambda x: x['auc'], reverse=True)
        best_result = results[0]
        
        # Auto-promote if AUC improved
        if best_result['auc'] > 0.75:  # Target AUC
            self._promote_final_config(best_result)
            logger.info(f"✅ Target AUC achieved! Config auto-promoted.")
        else:
            logger.info(f"❌ Target AUC not achieved. Best: {best_result['auc']:.4f}")
        
        # Finish progress tracking
        self._finish_progress_tracking("final")
        
        return results
    
    def _create_narrow_config(self, top_results: List[Dict[str, Any]]):
        """Create narrow sweep config with top architectures."""
        narrow_config_path = self.config_loader.get('sweep.output_config', 'configs/sweep_narrow.yaml')
        
        # Load base narrow config
        with open(narrow_config_path, 'r') as f:
            narrow_config = yaml.safe_load(f)
        
        # Add top architectures
        architectures = {}
        for result in top_results:
            arch_name = result['architecture']
            config = result['config']
            architectures[arch_name] = {
                'latent_dim': config['latent_dim'],
                'hidden_dims': config['hidden_dims'],
                'dropout_rate': config['dropout_rate']
            }
        
        narrow_config['model']['architectures'] = architectures
        
        # Save updated config
        with open(narrow_config_path, 'w') as f:
            yaml.dump(narrow_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Narrow config created: {narrow_config_path}")
    
    def _create_final_config(self, best_result: Dict[str, Any]):
        """Create final sweep config with best configuration."""
        final_config_path = self.config.get('sweep.output_config', 'configs/sweep_final.yaml')
        
        # Load base final config
        with open(final_config_path, 'r') as f:
            final_config = yaml.safe_load(f)
        
        # Update with best configuration
        config = best_result['config']
        final_config['model']['latent_dim'] = config['latent_dim']
        final_config['model']['hidden_dims'] = config['hidden_dims']
        final_config['model']['dropout_rate'] = config['dropout_rate']
        final_config['training']['batch_size'] = config['batch_size']
        final_config['training']['learning_rate'] = config['learning_rate']
        
        # Save updated config
        with open(final_config_path, 'w') as f:
            yaml.dump(final_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Final config created: {final_config_path}")
    
    def _promote_final_config(self, best_result: Dict[str, Any]):
        """Promote best configuration to final optimized config."""
        final_optimized_path = "configs/final_optimized_config.yaml"
        
        # Create final optimized config
        final_config = {
            'seed': 42,
            'model': {
                'latent_dim': best_result['config']['latent_dim'],
                'hidden_dims': best_result['config']['hidden_dims'],
                'dropout_rate': best_result['config']['dropout_rate']
            },
            'training': {
                'batch_size': best_result['config']['batch_size'],
                'learning_rate': best_result['config']['learning_rate'],
                'epochs': best_result['config']['epochs'],
                'early_stopping': best_result['config']['early_stopping'],
                'patience': best_result['config']['patience'],
                'reduce_lr': best_result['config']['reduce_lr'],
                'validation_split': 0.2
            },
            'features': {
                'threshold_percentile': best_result['config']['threshold_percentile'],
                'use_amount_features': True,
                'use_temporal_features': True,
                'use_customer_features': True,
                'use_risk_flags': True
            }
        }
        
        # Save final optimized config
        with open(final_optimized_path, 'w') as f:
            yaml.dump(final_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Final optimized config promoted: {final_optimized_path}")
        logger.info(f"Best AUC: {best_result['auc']:.4f}")
    
    def _load_best_sweeps(self) -> Dict[str, Any]:
        """Load existing best sweeps data."""
        try:
            with open(self.best_sweeps_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default structure if file doesn't exist
            return {
                'broad_sweep': {'run_id': None, 'architecture': None, 'auc_roc': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'threshold': 0.0, 'train_loss': 0.0, 'val_loss': 0.0, 'config': {}},
                'narrow_sweep': {'run_id': None, 'architecture': None, 'auc_roc': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'threshold': 0.0, 'train_loss': 0.0, 'val_loss': 0.0, 'config': {}},
                'final_sweep': {'run_id': None, 'architecture': None, 'auc_roc': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'threshold': 0.0, 'train_loss': 0.0, 'val_loss': 0.0, 'config': {}},
                'summary': {'best_overall_auc': 0.0, 'best_overall_stage': None, 'best_overall_run_id': None, 'sweep_completion_date': None}
            }
    
    def _save_best_sweeps(self):
        """Save best sweeps data to file."""
        # Update completion date
        from datetime import datetime
        self.best_sweeps_data['summary']['sweep_completion_date'] = datetime.now().isoformat()
        
        # Update best overall
        best_auc = 0.0
        best_stage = None
        best_run_id = None
        
        for stage in ['broad_sweep', 'narrow_sweep', 'final_sweep']:
            if self.best_sweeps_data[stage]['auc_roc'] > best_auc:
                best_auc = self.best_sweeps_data[stage]['auc_roc']
                best_stage = stage
                best_run_id = self.best_sweeps_data[stage]['run_id']
        
        self.best_sweeps_data['summary']['best_overall_auc'] = best_auc
        self.best_sweeps_data['summary']['best_overall_stage'] = best_stage
        self.best_sweeps_data['summary']['best_overall_run_id'] = best_run_id
        
        # Save to file
        with open(self.best_sweeps_file, 'w') as f:
            yaml.dump(self.best_sweeps_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Best sweeps data saved to: {self.best_sweeps_file}")
    
    def _calculate_total_steps(self, stage: str) -> int:
        """Calculate total number of steps for a sweep stage."""
        if stage == "broad":
            architectures = self.config_loader.get('model.architectures', {})
            return len(architectures)
        elif stage == "narrow":
            narrow_config = ConfigLoader(self.config.get('sweep.output_config', 'configs/sweep_narrow.yaml'))
            architectures = narrow_config.get('model.architectures', {})
            batch_sizes = narrow_config.get('training.batch_sizes', [64])
            learning_rates = narrow_config.get('training.learning_rates', [0.001])
            dropout_rates = narrow_config.get('training.dropout_rates', [0.2])
            return len(architectures) * len(batch_sizes) * len(learning_rates) * len(dropout_rates)
        elif stage == "final":
            final_config = ConfigLoader(self.config.get('sweep.output_config', 'configs/sweep_final.yaml'))
            threshold_percentiles = final_config.get('sweep.threshold_percentiles', [95])
            learning_rates = final_config.get('sweep.learning_rates', [0.001])
            epochs_list = final_config.get('sweep.epochs', [15])
            return len(threshold_percentiles) * len(learning_rates) * len(epochs_list)
        return 0
    
    def _start_progress_tracking(self, stage: str):
        """Start progress tracking for a sweep stage."""
        self.start_time = time.time()
        self.total_steps = self._calculate_total_steps(stage)
        self.current_step = 0
        
        logger.info("=" * 60)
        logger.info(f"STARTING {stage.upper()} SWEEP")
        logger.info("=" * 60)
        logger.info(f"Total configurations to test: {self.total_steps}")
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
    
    def _update_progress(self, step_name: str = ""):
        """Update and log progress."""
        self.current_step += 1
        elapsed_time = time.time() - self.start_time
        
        # Calculate progress percentage
        progress = (self.current_step / self.total_steps) * 100
        
        # Calculate estimated time remaining
        if self.current_step > 0:
            avg_time_per_step = elapsed_time / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = avg_time_per_step * remaining_steps
            eta = datetime.now() + timedelta(seconds=estimated_remaining)
            eta_str = eta.strftime('%H:%M:%S')
        else:
            eta_str = "Calculating..."
        
        # Format elapsed time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        
        # Log progress
        logger.info(f"Progress: {self.current_step}/{self.total_steps} ({progress:.1f}%) | "
                   f"Elapsed: {elapsed_str} | ETA: {eta_str} | {step_name}")
    
    def _finish_progress_tracking(self, stage: str):
        """Finish progress tracking for a sweep stage."""
        total_time = time.time() - self.start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        
        logger.info("=" * 60)
        logger.info(f"{stage.upper()} SWEEP COMPLETED")
        logger.info(f"Total time: {total_time_str}")
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
    
    def print_best_sweeps_summary(self):
        """Print a summary of the best results from each sweep stage."""
        logger.info("=" * 80)
        logger.info("BEST SWEEPS SUMMARY")
        logger.info("=" * 80)
        
        for stage in ['broad_sweep', 'narrow_sweep', 'final_sweep']:
            data = self.best_sweeps_data[stage]
            logger.info(f"\n{stage.upper().replace('_', ' ')}:")
            logger.info(f"  Run ID: {data['run_id']}")
            logger.info(f"  Architecture: {data['architecture']}")
            logger.info(f"  AUC ROC: {data['auc_roc']:.4f}")
            logger.info(f"  F1 Score: {data['f1_score']:.4f}")
            logger.info(f"  Precision: {data['precision']:.4f}")
            logger.info(f"  Recall: {data['recall']:.4f}")
            logger.info(f"  Threshold: {data['threshold']:.6f}")
        
        summary = self.best_sweeps_data['summary']
        logger.info(f"\nOVERALL BEST:")
        logger.info(f"  Best AUC: {summary['best_overall_auc']:.4f}")
        logger.info(f"  Best Stage: {summary['best_overall_stage']}")
        logger.info(f"  Best Run ID: {summary['best_overall_run_id']}")
        logger.info(f"  Completion Date: {summary['sweep_completion_date']}")
        logger.info("=" * 80)
    
    def _update_best_sweep_result(self, stage: str, run_id: str, architecture: str, 
                                 results_dict: Dict[str, Any], model_config: Dict[str, Any]):
        """Update best result for a specific sweep stage."""
        current_auc = self.best_sweeps_data[stage]['auc_roc']
        new_auc = results_dict['test_auc']
        
        if new_auc > current_auc:
            logger.info(f"New best {stage} result: AUC = {new_auc:.4f} (previous: {current_auc:.4f})")
            
            self.best_sweeps_data[stage].update({
                'run_id': run_id,
                'architecture': architecture,
                'auc_roc': new_auc,
                'f1_score': results_dict.get('f1_score', 0.0),
                'precision': results_dict.get('precision', 0.0),
                'recall': results_dict.get('recall', 0.0),
                'threshold': results_dict['threshold'],
                'train_loss': results_dict.get('train_loss', 0.0),
                'val_loss': results_dict.get('val_loss', 0.0),
                'config': {
                    'latent_dim': model_config.get('latent_dim'),
                    'hidden_dims': model_config.get('hidden_dims'),
                    'dropout_rate': model_config.get('dropout_rate'),
                    'learning_rate': model_config.get('learning_rate'),
                    'batch_size': model_config.get('batch_size'),
                    'epochs': model_config.get('epochs'),
                    'threshold_percentile': model_config.get('threshold_percentile')
                }
            })
            
            # Save immediately
            self._save_best_sweeps() 