#!/usr/bin/env python3
"""
Top Feature Combinations Sweep - 50 Epochs
Tests the 15 best combinations from previous sweep results with 50 epochs each.
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
from itertools import combinations
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import ConfigLoader
from src.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder
from src.config import PipelineConfig
from src.data.data_cleaning import DataCleaner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_info(msg):
    """Log info message with timestamp."""
    print(f"{datetime.now().strftime('%H:%M:%S')} - INFO - {msg}")


def log_error(msg):
    """Log error message with timestamp."""
    print(f"{datetime.now().strftime('%H:%M:%S')} - ERROR - {msg}")


class CombinedFeatureStrategy:
    """Custom feature strategy that combines multiple strategies."""
    
    def __init__(self, strategy_names: List[str]):
        self.strategy_names = strategy_names
        self.feature_count = 0
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features by applying multiple strategies sequentially."""
        df_features = df.copy()
        
        for strategy_name in self.strategy_names:
            try:
                feature_engineer = FeatureFactory.create(strategy_name)
                df_features = feature_engineer.generate_features(df_features)
                log_info(f"Applied {strategy_name}: {len(df_features.columns)} total features")
            except Exception as e:
                log_error(f"Failed to apply {strategy_name}: {str(e)}")
                continue
        
        self.feature_count = len(df_features.columns)
        return df_features
    
    def get_feature_info(self) -> Dict[str, any]:
        """Get information about the combined features."""
        return {
            "strategy": "combined",
            "feature_count": self.feature_count,
            "strategies_used": self.strategy_names
        }


def generate_top_combinations() -> List[List[str]]:
    """Generate the 15 top combinations from previous sweep results."""
    
    # Core strategies that will be included in every combination
    core_strategies = ["baseline_numeric", "categorical"]
    
    # The 10 specific combinations of 4 additional strategies (excluding 'combined' to avoid redundancy)
    additional_combinations = [
        ["behavioral", "fraud_flags", "rank_encoding", "demographics"],
        ["behavioral", "fraud_flags", "rank_encoding", "time_interactions"],
        ["behavioral", "fraud_flags", "demographics", "time_interactions"],
        ["behavioral", "rank_encoding", "demographics", "time_interactions"],
        ["fraud_flags", "rank_encoding", "demographics", "time_interactions"],
        ["behavioral", "fraud_flags", "rolling", "demographics"],
        ["behavioral", "fraud_flags", "rolling", "time_interactions"],
        ["behavioral", "rank_encoding", "rolling", "demographics"],
        ["fraud_flags", "rank_encoding", "rolling", "time_interactions"],
        ["behavioral", "fraud_flags", "temporal", "demographics"]
    ]
    
    combinations_list = []
    
    # Add core strategies to each combination
    for additional_combo in additional_combinations:
        strategy_combo = core_strategies + additional_combo
        combinations_list.append(strategy_combo)
    
    return combinations_list


def run_top_combinations_sweep():
    """Run sweep of top 15 combinations with 50 epochs each."""
    
    print("🚀 INITIALIZING TOP COMBINATIONS SWEEP - 50 EPOCHS...")
    print("⏰ Starting at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("📊 Setting up Weights & Biases integration...")
    
    wandb_run = None
    try:
        # Initialize W&B
        wandb.init(
            project="fraud-detection-autoencoder",
            group="top-combinations-sweep",
            name=f"top-combinations-50epochs-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.01,
                "test_size": 0.2,
                "random_state": 42,
                "early_stopping_patience": 10,
                "early_stopping_monitor": "val_loss",
                "early_stopping_min_delta": 0.001
            }
        )
        wandb_run = wandb.run
        print("✅ Weights & Biases initialized successfully!")
        print(f"🔗 W&B Run: {wandb_run.name}")
    except Exception as e:
        print(f"⚠️  WARNING: Could not initialize W&B: {str(e)}")
        print("🔄 Continuing without W&B logging...")
        wandb_run = None
    
    # Generate the 15 top combinations
    strategy_combinations = generate_top_combinations()
    
    print(f"\n📊 SWEEP ANALYSIS:")
    print(f"   Core strategies (always included): baseline_numeric, categorical")
    print(f"   Additional strategies per combination: 4 (excluding 'combined' to avoid redundancy)")
    print(f"   Total combinations to test: {len(strategy_combinations)}")
    print(f"   Epochs per combination: 50 (with early stopping)")
    print(f"   Estimated time: ~{len(strategy_combinations) * 3:.0f} minutes")
    
    log_info(f"Generated {len(strategy_combinations)} top combinations")
    
    # Load and clean data once
    print(f"\n🔄 STEP 1: Loading and cleaning data...")
    log_info("Loading and cleaning data...")
    
    try:
        config = PipelineConfig.get_baseline_numeric_config()
        cleaner = DataCleaner(config)
        df_cleaned = cleaner.clean_data(save_output=False)
        
        log_info(f"✅ Data loaded: {len(df_cleaned)} transactions")
        print(f"✅ Data loaded: {len(df_cleaned):,} transactions")
        print(f"📊 Data shape: {df_cleaned.shape}")
        
        # Log data summary to W&B
        if wandb_run:
            wandb.log({
                "data/transactions_total": len(df_cleaned),
                "features_base": len(df_cleaned.columns),
                "fraud_rate": df_cleaned['is_fraudulent'].mean()
            })
        
    except Exception as e:
        print(f"❌ ERROR loading data: {str(e)}")
        log_error(f"Data loading failed: {str(e)}")
        if wandb_run:
            wandb.finish()
        return {}
    
    # Store results
    results = {}
    start_time = time.time()
    best_auc = 0.0
    best_combination = None
    
    # Test each combination
    for i, strategy_combo in enumerate(strategy_combinations, 1):
        combo_name = "+".join(strategy_combo)
        combo_size = len(strategy_combo)
        
        print(f"\n{'='*100}")
        print(f"🎯 COMBINATION {i}/{len(strategy_combinations)}: {combo_name}")
        print(f"{'='*100}")
        print(f"📊 Size: {combo_size} strategies")
        print(f"⏰ Started: {time.strftime('%H:%M:%S')}")
        print(f"📈 Progress: {i}/{len(strategy_combinations)} ({i/len(strategy_combinations)*100:.1f}%)")
        print(f"🔄 Remaining: {len(strategy_combinations) - i} combinations")
        print(f"{'='*100}")
        
        log_info(f"Testing combination {i}/{len(strategy_combinations)}: {combo_name}")
        
        try:
            strategy_start_time = time.time()
            
            # Create a temporary config for training with 50 epochs
            temp_config = PipelineConfig.get_baseline_numeric_config()
            temp_config.model.epochs = 50  # Set to 50 epochs
            
            # Create custom feature factory for this combination
            custom_feature_factory = CombinedFeatureStrategy(strategy_combo)
            
            # Generate features using the combination
            print(f"\n🔧 Generating features...")
            df_features = custom_feature_factory.generate_features(df_cleaned)
            
            feature_count = len(df_features.columns)
            new_features = feature_count - len(df_cleaned.columns)
            print(f"✅ Features: {feature_count} columns ({new_features} new)")
            
            # Train model
            print(f"\n🤖 Training autoencoder (50 epochs max, early stopping enabled)...")
            autoencoder = BaselineAutoencoder(temp_config)
            
            # Override the feature factory temporarily
            autoencoder.feature_factory = custom_feature_factory
            
            # Train the model
            results_dict = autoencoder.train()
            
            # Extract metrics
            roc_auc = results_dict.get('roc_auc', 0.0)
            threshold = results_dict.get('threshold', 0.0)
            
            # Calculate additional metrics from predictions
            y_test = results_dict.get('y_test', [])
            y_pred = results_dict.get('y_pred', [])
            test_mse = results_dict.get('test_mse', [])
            
            if len(y_test) > 0 and len(y_pred) > 0:
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
            else:
                accuracy = f1 = precision = recall = 0.0
            
            strategy_time = time.time() - strategy_start_time
            
            print(f"\n🎉 COMPLETED: {combo_name}")
            print(f"📊 ROC AUC: {roc_auc:.4f}")
            print(f"🎯 Accuracy: {accuracy:.4f}")
            print(f"📈 F1 Score: {f1:.4f}")
            print(f"🎯 Precision: {precision:.4f}")
            print(f"📋 Recall: {recall:.4f}")
            print(f"⏱️  Time: {strategy_time:.1f}s")
            print(f"📈 Features: {feature_count} ({new_features} new)")
            
            # Update best if improved
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_combination = combo_name
                print(f"🏆 NEW BEST! ROC AUC: {roc_auc:.4f}")
            
            # Store results
            results[combo_name] = {
                'roc_auc': roc_auc,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'threshold': threshold,
                'time_taken': strategy_time,
                'feature_count': feature_count,
                'new_features': new_features,
                'strategies': strategy_combo,
                'combo_size': combo_size,
                'success': True
            }
            
            # Log to W&B
            if wandb_run:
                wandb.log({
                    "roc_auc": roc_auc,
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "threshold": threshold,
                    "training_time_seconds": strategy_time,
                    "feature_count": feature_count,
                    "new_features": new_features,
                    "combo_size": combo_size,
                    "combination_name": combo_name,
                    "progress_completed": i,
                    "progress_total": len(strategy_combinations),
                    "progress_percentage": i / len(strategy_combinations) * 100
                })
            
            # Progress update
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / i * (len(strategy_combinations) - i)
            print(f"\n📊 PROGRESS UPDATE:")
            print(f"   ✅ Completed: {i}/{len(strategy_combinations)} combinations")
            print(f"   ⏱️  Elapsed: {elapsed_time/60:.1f} minutes")
            print(f"   🎯 Est. remaining: {remaining_time/60:.1f} minutes")
            print(f"   📈 Current best ROC AUC: {best_auc:.4f}")
            print(f"   🎯 Current best accuracy: {max([r['accuracy'] for r in results.values()]):.4f}")
            
        except Exception as e:
            print(f"❌ ERROR in combination {combo_name}: {str(e)}")
            log_error(f"Combination {combo_name} failed: {str(e)}")
            
            results[combo_name] = {
                'roc_auc': 0.0,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'threshold': 0.0,
                'time_taken': 0.0,
                'feature_count': 0,
                'new_features': 0,
                'strategies': strategy_combo,
                'combo_size': combo_size,
                'success': False,
                'error': str(e)
            }
            
            if wandb_run:
                wandb.log({
                    "roc_auc": 0.0,
                    "accuracy": 0.0,
                    "f1_score": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "threshold": 0.0,
                    "training_time_seconds": 0.0,
                    "feature_count": 0,
                    "new_features": 0,
                    "combo_size": combo_size,
                    "combination_name": combo_name,
                    "success": False,
                    "error": str(e),
                    "progress_completed": i,
                    "progress_total": len(strategy_combinations),
                    "progress_percentage": i / len(strategy_combinations) * 100
                })
    
    # Final summary
    total_time = time.time() - start_time
    successful_runs = sum(1 for r in results.values() if r['success'])
    failed_runs = len(results) - successful_runs
    
    print(f"\n{'='*100}")
    print(f"🏆 TOP COMBINATIONS SWEEP RESULTS SUMMARY")
    print(f"{'='*100}")
    
    # Sort results by ROC AUC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
    
    print(f"{'Rank':<4} {'Combination':<50} {'Size':<4} {'ROC AUC':<8} {'Accuracy':<8} {'F1':<8} {'Precision':<8} {'Recall':<8} {'Time (s)':<8}")
    print("-"*100)
    
    for rank, (combo_name, result) in enumerate(sorted_results, 1):
        if result['success']:
            print(f"{rank:<4} {combo_name:<50} {result['combo_size']:<4} {result['roc_auc']:<8.4f} {result['accuracy']:<8.4f} {result['f1_score']:<8.4f} {result['precision']:<8.4f} {result['recall']:<8.4f} {result['time_taken']:<8.1f}")
        else:
            print(f"{rank:<4} {combo_name:<50} {result['combo_size']:<4} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8} {result['time_taken']:<8.1f}")
    
    print("-"*100)
    
    if best_combination:
        best_result = results[best_combination]
        print(f"\n🥇 BEST COMBINATION: {best_combination}")
        print(f"   📊 ROC AUC: {best_result['roc_auc']:.4f}")
        print(f"   🎯 Accuracy: {best_result['accuracy']:.4f}")
        print(f"   📈 F1 Score: {best_result['f1_score']:.4f}")
        print(f"   🎯 Precision: {best_result['precision']:.4f}")
        print(f"   📋 Recall: {best_result['recall']:.4f}")
        print(f"   ⏱️  Time taken: {best_result['time_taken']:.2f} seconds")
        print(f"   📈 Features: {best_result['feature_count']} ({best_result['new_features']} new)")
        print(f"   📝 Strategies: {', '.join(best_result['strategies'])}")
        print(f"   📊 Strategy count: {best_result['combo_size']}")
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   ⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"   🎯 Combinations tested: {len(strategy_combinations)}")
    print(f"   ✅ Successful runs: {successful_runs}")
    print(f"   ❌ Failed runs: {failed_runs}")
    print(f"   📈 Average time per combo: {total_time/len(strategy_combinations):.1f} seconds")
    if successful_runs > 0:
        successful_aucs = [r['roc_auc'] for r in results.values() if r['success']]
        print(f"   📊 Average ROC AUC: {np.mean(successful_aucs):.4f}")
        print(f"   🎯 Best ROC AUC: {best_auc:.4f}")
        print(f"   📉 Worst ROC AUC: {min(successful_aucs):.4f}")
    
    # Log final summary to W&B
    if wandb_run:
        if successful_runs > 0:
            successful_aucs = [r['roc_auc'] for r in results.values() if r['success']]
            successful_accuracies = [r['accuracy'] for r in results.values() if r['success']]
            successful_f1s = [r['f1_score'] for r in results.values() if r['success']]
            successful_precisions = [r['precision'] for r in results.values() if r['success']]
            successful_recalls = [r['recall'] for r in results.values() if r['success']]
            
            wandb.log({
                "summary/best_roc_auc": best_auc,
                "summary/best_accuracy": max(successful_accuracies),
                "summary/best_f1_score": max(successful_f1s),
                "summary/best_precision": max(successful_precisions),
                "summary/best_recall": max(successful_recalls),
                "summary/mean_roc_auc": np.mean(successful_aucs),
                "summary/mean_accuracy": np.mean(successful_accuracies),
                "summary/mean_f1_score": np.mean(successful_f1s),
                "summary/mean_precision": np.mean(successful_precisions),
                "summary/mean_recall": np.mean(successful_recalls),
                "summary/std_roc_auc": np.std(successful_aucs),
                "summary/std_accuracy": np.std(successful_accuracies),
                "summary/total_combinations": len(strategy_combinations),
                "summary/successful_combinations": successful_runs,
                "summary/total_time_minutes": total_time / 60
            })
        
        # Log best combination details
        if best_combination:
            best_result = results[best_combination]
            wandb.log({
                "summary/best_combination": best_combination,
                "summary/best_combo_roc_auc": best_result['roc_auc'],
                "summary/best_combo_accuracy": best_result['accuracy'],
                "summary/best_combo_f1_score": best_result['f1_score'],
                "summary/best_combo_precision": best_result['precision'],
                "summary/best_combo_recall": best_result['recall'],
                "summary/best_combo_time_seconds": best_result['time_taken'],
                "summary/best_combo_feature_count": best_result['feature_count'],
                "summary/best_combo_new_features": best_result['new_features']
            })
    
    print(f"\n🎉 Top combinations sweep completed!")
    print(f"{'='*100}")
    
    if wandb_run:
        wandb.finish()
    
    return results


if __name__ == "__main__":
    run_top_combinations_sweep() 