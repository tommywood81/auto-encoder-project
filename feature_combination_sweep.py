"""
Comprehensive feature combination sweep - tests all possible combinations of feature strategies.
"""

import pandas as pd
import numpy as np
import logging
import time
import wandb
from typing import Dict, List, Tuple, Optional
import sys
import os
from datetime import datetime
import itertools

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig
from src.data.data_cleaning import DataCleaner
from src.feature_factory.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder
from src.evaluation.evaluator import FraudEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('feature_combination_sweep.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def log_info(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} - INFO - {msg}")
    logger.info(msg)

def log_error(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} - ERROR - {msg}")
    logger.error(msg)


class CombinedFeatureStrategy:
    """Custom feature strategy that combines multiple feature engineering strategies."""
    
    def __init__(self, strategies: List[str]):
        self.strategies = strategies
        self.feature_count = 0
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features by combining multiple strategies."""
        log_info(f"Generating combined features for strategies: {self.strategies}")
        
        df_combined = df.copy()
        
        for strategy_name in self.strategies:
            try:
                feature_engineer = FeatureFactory.create(strategy_name)
                df_combined = feature_engineer.generate_features(df_combined)
                log_info(f"Applied {strategy_name} strategy")
            except Exception as e:
                log_error(f"Failed to apply {strategy_name}: {str(e)}")
                continue
        
        # Remove duplicate columns if any
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
        
        self.feature_count = len(df_combined.columns)
        log_info(f"Combined features generated: {self.feature_count} features")
        return df_combined
    
    def get_feature_info(self) -> Dict[str, any]:
        return {
            "strategy": f"combined_{'_'.join(self.strategies)}",
            "description": f"Combination of: {', '.join(self.strategies)}"
        }


def run_feature_combination_sweep():
    """Run comprehensive feature combination sweep."""
    
    print("🎯 INITIALIZING FEATURE COMBINATION SWEEP...")
    print("⏰ Starting at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("📊 Setting up Weights & Biases integration...")
    
    wandb_run = None
    try:
        # Initialize W&B
        wandb.init(
            project="fraud-detection-autoencoder",
            group="feature-combination-sweep",
            name=f"feature-combination-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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
    
    # Define all available feature strategies
    all_strategies = [
        "baseline_numeric",
        "categorical", 
        "temporal",
        "behavioral",
        "demographics",
        "fraud_flags",
        "rolling",
        "rank_encoding",
        "time_interactions"
    ]
    
    # Strategy descriptions for output
    strategy_descriptions = {
        "baseline_numeric": "Log and ratio features from raw numerics",
        "categorical": "Encoded payment, product, and device columns",
        "temporal": "Late-night and burst transaction flags",
        "behavioral": "Behavioral ratios per age/account age",
        "demographics": "Customer age bucketed into risk bands",
        "fraud_flags": "Optimized rule-based fraud risk indicators (73.05% AUC)",
        "rolling": "Rolling mean and std of amount per customer",
        "rank_encoding": "Rank-based encodings of amount and account age",
        "time_interactions": "Crossed and interaction features using hour"
    }
    
    # Generate all possible combinations (1 to all strategies)
    all_combinations = []
    for r in range(1, len(all_strategies) + 1):
        combinations = list(itertools.combinations(all_strategies, r))
        all_combinations.extend(combinations)
    
    # Convert tuples to lists for easier handling
    all_combinations = [list(combo) for combo in all_combinations]
    
    print(f"\n🎯 FEATURE COMBINATION SWEEP")
    print("="*80)
    print(f"📊 Testing {len(all_combinations)} combinations of {len(all_strategies)} strategies")
    print(f"⏰ Estimated time: ~{len(all_combinations) * 2:.0f} minutes")
    if wandb_run:
        print(f"📈 Logging to Weights & Biases: {wandb_run.name}")
    print("="*80)
    
    log_info(f"Starting feature combination sweep with {len(all_combinations)} combinations")
    log_info(f"Available strategies: {', '.join(all_strategies)}")
    if wandb_run:
        log_info(f"W&B Run initialized: {wandb_run.name}")
    
    log_info("Starting feature combination sweep...")
    log_info(f"Testing {len(all_combinations)} combinations")
    if wandb_run:
        log_info(f"W&B Run: {wandb_run.name}")
    
    print(f"\n📋 AVAILABLE STRATEGIES:")
    for i, strategy in enumerate(all_strategies, 1):
        print(f"   {i:2d}. {strategy:<20} - {strategy_descriptions[strategy]}")
    
    print(f"\n🚀 Starting execution...")
    print("-"*80)
    
    # Load and clean data once
    print(f"\n🔄 STEP 1: Loading and cleaning data...")
    log_info("Loading and cleaning data...")
    
    try:
        config = PipelineConfig.get_baseline_numeric_config()
        cleaner = DataCleaner(config)
        df_cleaned = cleaner.clean_data(save_output=False)
        
        log_info(f"✅ Data loaded: {len(df_cleaned)} transactions")
        print(f"✅ Data loaded: {len(df_cleaned):,} transactions")
        
        # Log data summary to W&B
        if wandb_run:
            wandb.log({
                "data/transactions_total": len(df_cleaned),
                "data/features_base": len(df_cleaned.columns),
                "data/fraud_rate": df_cleaned['is_fraudulent'].mean()
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
    for i, combination in enumerate(all_combinations, 1):
        combination_name = f"combo_{i:03d}_{len(combination)}strategies"
        strategies_str = "_".join(combination)
        
        print(f"\n{'='*80}")
        print(f"🎯 TESTING COMBINATION {i}/{len(all_combinations)}: {combination_name}")
        print(f"{'='*80}")
        print(f"📝 Strategies: {', '.join(combination)}")
        print(f"📊 Strategy count: {len(combination)}")
        print(f"⏰ Starting at: {time.strftime('%H:%M:%S')}")
        print(f"📈 Progress: {i}/{len(all_combinations)} ({i/len(all_combinations)*100:.1f}%)")
        print(f"{'='*80}")
        
        log_info(f"Testing combination {i}/{len(all_combinations)}: {combination_name}")
        
        try:
            combo_start_time = time.time()
            
            # Create custom feature engineer with combination
            feature_engineer = CombinedFeatureStrategy(combination)
            
            # Generate features
            print(f"\n🔧 Generating features for combination...")
            log_info(f"Generating features for combination {i}/{len(all_combinations)}: {combination_name}")
            log_info(f"Strategies in this combination: {', '.join(combination)}")
            
            df_features = feature_engineer.generate_features(df_cleaned)
            
            feature_count = len(df_features.columns)
            new_features = feature_count - len(df_cleaned.columns)
            log_info(f"Features generated: {feature_count} columns ({new_features} new)")
            print(f"✅ Features generated: {feature_count} columns ({new_features} new)")
            
            # Create a temporary config for training
            temp_config = PipelineConfig.get_baseline_numeric_config()
            
            # Train model
            print(f"\n🤖 Training autoencoder with combination...")
            print(f"📈 This will run up to 50 epochs with early stopping...")
            log_info("Training autoencoder...")
            
            autoencoder = BaselineAutoencoder(temp_config)
            
            # Override the feature factory temporarily
            autoencoder.feature_factory = feature_engineer
            
            # Train the model
            train_results = autoencoder.train()
            
            # Extract results
            roc_auc = train_results['roc_auc']
            threshold = train_results['threshold']
            epochs_trained = len(train_results['history'].history['loss'])
            combo_time = time.time() - combo_start_time
            
            # Calculate training efficiency
            training_efficiency = epochs_trained / 50.0
            
            print(f"\n🎉 Combination {combination_name} completed!")
            print(f"📊 ROC AUC: {roc_auc:.4f}")
            print(f"🎯 Threshold: {threshold:.4f}")
            print(f"⏱️  Time taken: {combo_time:.2f} seconds")
            print(f"📈 Feature count: {feature_count} ({new_features} new)")
            print(f"🔄 Epochs trained: {epochs_trained}")
            
            log_info(f"Combination {combination_name} completed successfully!")
            log_info(f"ROC AUC: {roc_auc:.4f}")
            log_info(f"Time taken: {combo_time:.2f} seconds")
            
            # Store results
            results[combination_name] = {
                'roc_auc': roc_auc,
                'threshold': threshold,
                'time_taken': combo_time,
                'feature_count': feature_count,
                'new_features': new_features,
                'epochs_trained': epochs_trained,
                'training_efficiency': training_efficiency,
                'strategies': combination,
                'strategy_count': len(combination),
                'success': True
            }
            
            # Update best if improved
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_combination = combination_name
                print(f"🏆 NEW BEST! ROC AUC: {roc_auc:.4f}")
            
            # Log to W&B
            if wandb_run:
                wandb.log({
                    f"{combination_name}/roc_auc": roc_auc,
                    f"{combination_name}/threshold": threshold,
                    f"{combination_name}/time_taken": combo_time,
                    f"{combination_name}/feature_count": feature_count,
                    f"{combination_name}/new_features": new_features,
                    f"{combination_name}/epochs_trained": epochs_trained,
                    f"{combination_name}/training_efficiency": training_efficiency,
                    f"{combination_name}/strategy_count": len(combination),
                    f"{combination_name}/success": True,
                    f"{combination_name}/strategies": strategies_str,
                    "progress/completed": i,
                    "progress/total": len(all_combinations),
                    "progress/percentage": i / len(all_combinations) * 100,
                    "progress/current_best_roc": best_auc,
                    "progress/elapsed_time": time.time() - start_time,
                    "progress/estimated_remaining": (time.time() - start_time) / i * (len(all_combinations) - i)
                })
                
                # Log individual strategy flags
                for strategy in all_strategies:
                    wandb.log({
                        f"{combination_name}/has_{strategy}": strategy in combination
                    })
            
            # Progress update
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / i * (len(all_combinations) - i)
            print(f"\n📊 PROGRESS UPDATE:")
            print(f"   ✅ Completed: {i}/{len(all_combinations)} combinations ({i/len(all_combinations)*100:.1f}%)")
            print(f"   ⏱️  Elapsed time: {elapsed_time:.1f}s")
            print(f"   🎯 Est. remaining: {remaining_time:.1f}s ({remaining_time/60:.1f} minutes)")
            print(f"   📈 Current best: {best_auc:.4f}")
            
        except Exception as e:
            print(f"❌ ERROR in combination {combination_name}: {str(e)}")
            log_error(f"Combination {combination_name} failed: {str(e)}")
            
            results[combination_name] = {
                'roc_auc': 0.0,
                'threshold': 0.0,
                'time_taken': 0.0,
                'feature_count': 0,
                'new_features': 0,
                'epochs_trained': 0,
                'training_efficiency': 0.0,
                'strategies': combination,
                'strategy_count': len(combination),
                'success': False,
                'error': str(e)
            }
            
            if wandb_run:
                wandb.log({
                    f"{combination_name}/success": False,
                    f"{combination_name}/error": str(e),
                    "progress/completed": i,
                    "progress/total": len(all_combinations),
                    "progress/percentage": i / len(all_combinations) * 100
                })
    
    # Final summary
    total_time = time.time() - start_time
    successful_runs = sum(1 for r in results.values() if r['success'])
    failed_runs = len(results) - successful_runs
    
    print(f"\n{'='*80}")
    print(f"🏆 FEATURE COMBINATION SWEEP RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Sort results by ROC AUC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
    
    print(f"{'Rank':<4} {'Combo':<15} {'Status':<8} {'ROC AUC':<8} {'Time (s)':<8} {'Features':<8} {'Epochs':<8} {'Strategies':<8} {'Top Strategies'}")
    print("-"*100)
    
    for rank, (combo_name, result) in enumerate(sorted_results[:20], 1):  # Show top 20
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        roc_auc = result['roc_auc']
        time_taken = result['time_taken']
        features = result['feature_count']
        epochs = result['epochs_trained']
        strategy_count = result['strategy_count']
        
        # Get top 3 strategies for display
        strategies = result['strategies']
        top_strategies = ", ".join(strategies[:3])
        if len(strategies) > 3:
            top_strategies += f" (+{len(strategies)-3})"
        
        print(f"{rank:<4} {combo_name:<15} {status:<8} {roc_auc:<8.4f} {time_taken:<8.1f} {features:<8} {epochs:<8} {strategy_count:<8} {top_strategies}")
    
    print("-"*100)
    
    if best_combination:
        best_result = results[best_combination]
        print(f"\n🥇 BEST COMBINATION: {best_combination}")
        print(f"   📊 ROC AUC: {best_result['roc_auc']:.4f}")
        print(f"   🎯 Threshold: {best_result['threshold']:.4f}")
        print(f"   ⏱️  Time taken: {best_result['time_taken']:.2f} seconds")
        print(f"   📈 Features: {best_result['feature_count']} ({best_result['new_features']} new)")
        print(f"   🔄 Epochs: {best_result['epochs_trained']}")
        print(f"   📝 Strategies: {', '.join(best_result['strategies'])}")
        print(f"   📊 Strategy count: {best_result['strategy_count']}")
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   ⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   🎯 Combinations tested: {len(all_combinations)}")
    print(f"   ✅ Successful runs: {successful_runs}")
    print(f"   ❌ Failed runs: {failed_runs}")
    print(f"   📈 Average time per combo: {total_time/len(all_combinations):.1f} seconds")
    print(f"   📊 Average ROC AUC: {np.mean([r['roc_auc'] for r in results.values() if r['success']]):.4f}")
    print(f"   🎯 Best ROC AUC: {best_auc:.4f}")
    print(f"   📉 Worst ROC AUC: {min([r['roc_auc'] for r in results.values() if r['success']]):.4f}")
    
    # Analyze strategy performance
    print(f"\n📈 STRATEGY PERFORMANCE ANALYSIS:")
    strategy_performance = {}
    for combo_name, result in results.items():
        if result['success']:
            for strategy in result['strategies']:
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'count': 0, 'total_auc': 0}
                strategy_performance[strategy]['count'] += 1
                strategy_performance[strategy]['total_auc'] += result['roc_auc']
    
    print(f"{'Strategy':<20} {'Count':<8} {'Avg AUC':<8} {'Best AUC':<8}")
    print("-"*50)
    
    for strategy in all_strategies:
        if strategy in strategy_performance:
            count = strategy_performance[strategy]['count']
            avg_auc = strategy_performance[strategy]['total_auc'] / count
            # Find best AUC for this strategy
            best_auc_for_strategy = max([
                r['roc_auc'] for combo_name, r in results.items() 
                if r['success'] and strategy in r['strategies']
            ])
            print(f"{strategy:<20} {count:<8} {avg_auc:<8.4f} {best_auc_for_strategy:<8.4f}")
    
    # Log final summary to W&B
    if wandb_run:
        wandb.log({
            "summary/total_time": total_time,
            "summary/combinations_tested": len(all_combinations),
            "summary/successful_runs": successful_runs,
            "summary/failed_runs": failed_runs,
            "summary/avg_time_per_combo": total_time / len(all_combinations),
            "summary/avg_roc_auc": np.mean([r['roc_auc'] for r in results.values() if r['success']]),
            "summary/best_roc_auc": best_auc,
            "summary/worst_roc_auc": min([r['roc_auc'] for r in results.values() if r['success']]),
            "summary/best_combination": best_combination
        })
        
        # Log best combination details
        if best_combination:
            best_result = results[best_combination]
            wandb.log({
                "best_combo/name": best_combination,
                "best_combo/roc_auc": best_result['roc_auc'],
                "best_combo/threshold": best_result['threshold'],
                "best_combo/time_taken": best_result['time_taken'],
                "best_combo/feature_count": best_result['feature_count'],
                "best_combo/epochs_trained": best_result['epochs_trained'],
                "best_combo/strategy_count": best_result['strategy_count'],
                "best_combo/strategies": ", ".join(best_result['strategies'])
            })
    
    print(f"\n🎉 Feature combination sweep completed!")
    print(f"{'='*80}")
    
    if wandb_run:
        wandb.finish()
    
    return results


def main():
    """Main function to run feature combination sweep."""
    try:
        results = run_feature_combination_sweep()
        
        if results:
            print(f"\n✅ Feature combination sweep completed successfully!")
            print(f"   Check W&B dashboard for detailed results")
            print(f"   Best combination identified")
        else:
            print(f"\n❌ Feature combination sweep failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Feature combination sweep interrupted by user")
    except Exception as e:
        log_error(f"Feature combination sweep failed: {str(e)}")
        print(f"\n❌ Feature combination sweep failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 