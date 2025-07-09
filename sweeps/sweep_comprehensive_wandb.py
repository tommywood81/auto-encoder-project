"""
Comprehensive feature engineering sweep with Weights & Biases integration.
Tests combinations of 2 and 3 feature strategies with baseline_numeric and categorical.
"""

import pandas as pd
import numpy as np
import logging
import time
import wandb
from typing import Dict, Tuple, Optional, List
import sys
import os
from datetime import datetime
from itertools import combinations

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig
from src.data.data_cleaning import DataCleaner
from src.feature_factory.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder
from src.evaluation.evaluator import FraudEvaluator

# Configure logging with force=True to see all logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

def log_info(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} - INFO - {msg}")

def log_error(msg):
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


def generate_strategy_combinations() -> List[List[str]]:
    """Generate all combinations of 2 and 3 strategies plus baseline_numeric and categorical."""
    
    # Core strategies that will be included in every combination
    core_strategies = ["baseline_numeric", "categorical"]
    
    # Additional strategies to combine
    additional_strategies = [
        "temporal",
        "behavioral", 
        "demographics",
        "fraud_flags",
        "rolling",
        "rank_encoding",
        "time_interactions"
    ]
    
    combinations_list = []
    
    # Generate combinations of 2 additional strategies
    for combo in combinations(additional_strategies, 2):
        strategy_combo = core_strategies + list(combo)
        combinations_list.append(strategy_combo)
    
    # Generate combinations of 3 additional strategies  
    for combo in combinations(additional_strategies, 3):
        strategy_combo = core_strategies + list(combo)
        combinations_list.append(strategy_combo)
    
    return combinations_list


def run_comprehensive_sweep():
    """Run comprehensive feature engineering sweep with W&B integration."""
    
    print("🚀 INITIALIZING COMPREHENSIVE FEATURE COMBINATION SWEEP...")
    print("⏰ Starting at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("📊 Setting up Weights & Biases integration...")
    
    wandb_run = None
    try:
        # Initialize W&B
        wandb.init(
            project="fraud-detection-autoencoder",
            group="feature-combination-sweep",
            name=f"combination-sweep-25epochs-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "epochs": 25,
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
    
    # Generate all strategy combinations
    strategy_combinations = generate_strategy_combinations()
    
    print(f"\n📊 COMBINATION ANALYSIS:")
    print(f"   Core strategies (always included): baseline_numeric, categorical")
    print(f"   Additional strategies: 7 available (excluding 'combined' to avoid redundancy)")
    print(f"   Combinations of 2 additional strategies: C(7,2) = 21")
    print(f"   Combinations of 3 additional strategies: C(7,3) = 35")
    print(f"   Total combinations to test: {len(strategy_combinations)}")
    print(f"   Epochs per combination: 25 (with early stopping)")
    print(f"   Estimated time: ~{len(strategy_combinations) * 1.5:.0f} minutes")
    
    log_info(f"Generated {len(strategy_combinations)} strategy combinations")
    
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
            
            # Create a temporary config for training with 25 epochs
            temp_config = PipelineConfig.get_baseline_numeric_config()
            temp_config.model.epochs = 25  # Set to 25 epochs
            
            # Create custom feature factory for this combination
            custom_feature_factory = CombinedFeatureStrategy(strategy_combo)
            
            # Generate features using the combination
            print(f"\n🔧 Generating features...")
            df_features = custom_feature_factory.generate_features(df_cleaned)
            
            feature_count = len(df_features.columns)
            new_features = feature_count - len(df_cleaned.columns)
            print(f"✅ Features: {feature_count} columns ({new_features} new)")
            
            # Train model
            print(f"\n🤖 Training autoencoder (25 epochs max, early stopping enabled)...")
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
                from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
                f1_score_val = f1_score(y_test, y_pred, zero_division=0)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                accuracy = accuracy_score(y_test, y_pred)
            else:
                f1_score_val = 0.0
                precision = 0.0
                recall = 0.0
                accuracy = 0.0
            
            # Calculate time taken
            strategy_time = time.time() - strategy_start_time
            
            # Store results
            results[combo_name] = {
                'success': True,
                'roc_auc': roc_auc,
                'accuracy': accuracy,
                'f1_score': f1_score_val,
                'precision': precision,
                'recall': recall,
                'time_taken': strategy_time,
                'feature_count': feature_count,
                'new_features': new_features,
                'strategies': strategy_combo,
                'combo_size': combo_size,
                'threshold': threshold
            }
            
            print(f"\n🎉 COMPLETED: {combo_name}")
            print(f"📊 ROC AUC: {roc_auc:.4f}")
            print(f"🎯 Accuracy: {accuracy:.4f}")
            print(f"📈 F1 Score: {f1_score_val:.4f}")
            print(f"🎯 Precision: {precision:.4f}")
            print(f"📋 Recall: {recall:.4f}")
            print(f"⏱️  Time: {strategy_time:.1f}s")
            print(f"📈 Features: {feature_count} ({new_features} new)")
            
            # Log to W&B (only final results, no epoch-by-epoch logging)
            if wandb_run:
                wandb.log({
                    "combination_name": combo_name,
                    "combo_size": combo_size,
                    "roc_auc": roc_auc,
                    "accuracy": accuracy,
                    "f1_score": f1_score_val,
                    "precision": precision,
                    "recall": recall,
                    "training_time_seconds": strategy_time,
                    "feature_count": feature_count,
                    "new_features": new_features,
                    "threshold": threshold,
                    "progress_completed": i,
                    "progress_total": len(strategy_combinations),
                    "progress_percentage": i / len(strategy_combinations) * 100
                })
            
            # Progress update
            elapsed_time = time.time() - start_time
            avg_time_per_combo = elapsed_time / i
            remaining_combos = len(strategy_combinations) - i
            estimated_remaining = remaining_combos * avg_time_per_combo
            
            print(f"\n📊 PROGRESS UPDATE:")
            print(f"   ✅ Completed: {i}/{len(strategy_combinations)} combinations")
            print(f"   ⏱️  Elapsed: {elapsed_time/60:.1f} minutes")
            print(f"   🎯 Est. remaining: {estimated_remaining/60:.1f} minutes")
            print(f"   📈 Current best ROC AUC: {max([r['roc_auc'] for r in results.values() if r['success']], default=0):.4f}")
            print(f"   🎯 Current best accuracy: {max([r['accuracy'] for r in results.values() if r['success']], default=0):.4f}")
            
        except Exception as e:
            error_msg = f"Combination {combo_name} failed: {str(e)}"
            log_error(error_msg)
            print(f"❌ {error_msg}")
            results[combo_name] = {
                'success': False,
                'roc_auc': 0.0,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'time_taken': 0.0,
                'feature_count': 0,
                'new_features': 0,
                'strategies': strategy_combo,
                'combo_size': combo_size,
                'error': str(e)
            }
            continue
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print final summary
    print_summary(results, total_time)
    
    # Log final summary to W&B
    if wandb_run:
        log_final_summary_to_wandb(results, total_time)
        wandb.finish()
    
    return results


def print_summary(results: Dict[str, Dict], total_time: float):
    """Print summary of feature combination sweep results."""
    
    print("\n" + "="*120)
    print("🏆 FEATURE COMBINATION SWEEP RESULTS SUMMARY")
    print("="*120)
    print(f"{'Combination':<60} {'Size':<5} {'ROC AUC':<10} {'Accuracy':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Time (s)':<10}")
    print("-" * 120)
    
    # Sort by ROC AUC (descending)
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v['success']],
        key=lambda x: x[1]['roc_auc'],
        reverse=True
    )
    
    best_combo = None
    best_roc = 0.0
    
    for combo_name, result in sorted_results:
        if result['success'] and result['roc_auc'] > best_roc:
            best_combo = combo_name
            best_roc = result['roc_auc']
        
        print(f"{combo_name:<60} {result['combo_size']:<5} {result['roc_auc']:<10.4f} {result['accuracy']:<10.4f} {result['f1_score']:<8.4f} {result['precision']:<10.4f} {result['recall']:<8.4f} {result['time_taken']:<10.1f}")
    
    print("-" * 120)
    print(f"🏆 BEST COMBINATION: {best_combo}")
    print(f"   ROC AUC: {best_roc:.4f}")
    print(f"   Total combinations tested: {len(results)}")
    print(f"   Successful combinations: {len([r for r in results.values() if r['success']])}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    
    # Show top 10 combinations
    print(f"\n🥇 TOP 10 COMBINATIONS (by ROC AUC):")
    for i, (combo_name, result) in enumerate(sorted_results[:10], 1):
        print(f"   {i:2d}. {combo_name:<50} ROC AUC: {result['roc_auc']:.4f}, Accuracy: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}")


def log_final_summary_to_wandb(results: Dict[str, Dict], total_time: float):
    """Log final summary metrics to Weights & Biases."""
    
    successful_results = [r for r in results.values() if r['success']]
    
    if not successful_results:
        return
    
    # Find best combination
    best_result = max(successful_results, key=lambda x: x['roc_auc'])
    
    # Calculate summary statistics
    roc_aucs = [r['roc_auc'] for r in successful_results]
    accuracies = [r['accuracy'] for r in successful_results]
    f1_scores = [r['f1_score'] for r in successful_results]
    precisions = [r['precision'] for r in successful_results]
    recalls = [r['recall'] for r in successful_results]
    
    wandb.log({
        "summary/best_roc_auc": best_result['roc_auc'],
        "summary/best_accuracy": best_result['accuracy'],
        "summary/best_f1_score": best_result['f1_score'],
        "summary/best_precision": best_result['precision'],
        "summary/best_recall": best_result['recall'],
        "summary/best_combination": best_result['strategies'],
        "summary/mean_roc_auc": np.mean(roc_aucs),
        "summary/mean_accuracy": np.mean(accuracies),
        "summary/mean_f1_score": np.mean(f1_scores),
        "summary/mean_precision": np.mean(precisions),
        "summary/mean_recall": np.mean(recalls),
        "summary/std_roc_auc": np.std(roc_aucs),
        "summary/std_accuracy": np.std(accuracies),
        "summary/total_combinations": len(results),
        "summary/successful_combinations": len(successful_results),
        "summary/total_time_minutes": total_time / 60
    })


if __name__ == "__main__":
    run_comprehensive_sweep() 