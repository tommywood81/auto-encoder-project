"""
Top 4 Feature Combinations Sweep - Tests the best performing feature groups from previous sweeps.
Focuses on baseline_numeric + categorical + 4 of the top performing strategies.
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
        logging.FileHandler('top_4_combinations_sweep.log', mode='w', encoding='utf-8')
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


def generate_4_strategy_combinations() -> List[List[str]]:
    """Generate combinations of exactly 4 strategies (plus baseline_numeric and categorical)."""
    
    # Core strategies that will be included in every combination
    core_strategies = ["baseline_numeric", "categorical"]
    
    # The 5 additional strategies to choose 4 from
    additional_strategies = [
        "behavioral",      # High performance in previous sweeps
        "fraud_flags",     # Optimized rule-based indicators (73.05% AUC)
        "rank_encoding",   # Rank-based encodings performed well
        "demographics",    # Customer age risk bands showed good results
        "time_interactions" # Crossed and interaction features using hour
    ]
    
    # Generate combinations of exactly 4 additional strategies
    combinations_list = []
    for combo in itertools.combinations(additional_strategies, 4):
        strategy_combo = core_strategies + list(combo)
        combinations_list.append(strategy_combo)
    
    return combinations_list


def run_4_strategy_combinations_sweep():
    """Run sweep focusing on combinations of exactly 4 strategies (plus baseline_numeric + categorical)."""
    
    print("🎯 INITIALIZING 4-STRATEGY COMBINATIONS SWEEP...")
    print("⏰ Starting at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("📊 Setting up Weights & Biases integration...")
    
    wandb_run = None
    try:
        # Initialize W&B
        wandb.init(
            project="fraud-detection-autoencoder",
            group="top-4-combinations-sweep",
            name=f"top-4-combinations-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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
    
    # Generate combinations of exactly 4 strategies
    strategy_combinations = generate_4_strategy_combinations()
    
    print(f"\n🎯 4-STRATEGY COMBINATIONS SWEEP")
    print("="*80)
    print(f"📊 Core strategies (always included): baseline_numeric, categorical")
    print(f"📊 Additional strategies: behavioral, fraud_flags, rank_encoding, demographics, time_interactions")
    print(f"📊 Testing combinations of exactly 4 additional strategies")
    print(f"📊 Total combinations to test: {len(strategy_combinations)}")
    print(f"⏰ Estimated time: ~{len(strategy_combinations) * 2:.0f} minutes")
    if wandb_run:
        print(f"📈 Logging to Weights & Biases: {wandb_run.name}")
    print("="*80)
    
    log_info(f"Starting 4-strategy combinations sweep with {len(strategy_combinations)} combinations")
    log_info(f"Core strategies: baseline_numeric, categorical")
    log_info(f"Additional strategies: behavioral, fraud_flags, rank_encoding, demographics, time_interactions")
    if wandb_run:
        log_info(f"W&B Run initialized: {wandb_run.name}")
    
    # Strategy descriptions for output
    strategy_descriptions = {
        "baseline_numeric": "Log and ratio features from raw numerics",
        "categorical": "Encoded payment, product, and device columns",
        "behavioral": "Behavioral ratios per age/account age",
        "fraud_flags": "Optimized rule-based fraud risk indicators (73.05% AUC)",
        "rank_encoding": "Rank-based encodings of amount and account age",
        "demographics": "Customer age bucketed into risk bands",
        "time_interactions": "Crossed and interaction features using hour"
    }
    
    print(f"\n📋 STRATEGIES:")
    print(f"   Core (always included):")
    print(f"     1. baseline_numeric    - {strategy_descriptions['baseline_numeric']}")
    print(f"     2. categorical         - {strategy_descriptions['categorical']}")
    print(f"   Additional (choose 4):")
    print(f"     3. behavioral          - {strategy_descriptions['behavioral']}")
    print(f"     4. fraud_flags         - {strategy_descriptions['fraud_flags']}")
    print(f"     5. rank_encoding       - {strategy_descriptions['rank_encoding']}")
    print(f"     6. demographics        - {strategy_descriptions['demographics']}")
    print(f"     7. time_interactions   - {strategy_descriptions['time_interactions']}")
    
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
    for i, combination in enumerate(strategy_combinations, 1):
        combination_name = f"top4_combo_{i:02d}_{len(combination)}strategies"
        strategies_str = "_".join(combination)
        
        print(f"\n{'='*80}")
        print(f"🎯 TESTING COMBINATION {i}/{len(strategy_combinations)}: {combination_name}")
        print(f"{'='*80}")
        print(f"📝 Strategies: {', '.join(combination)}")
        print(f"📊 Strategy count: {len(combination)}")
        print(f"⏰ Starting at: {time.strftime('%H:%M:%S')}")
        print(f"📈 Progress: {i}/{len(strategy_combinations)} ({i/len(strategy_combinations)*100:.1f}%)")
        print(f"{'='*80}")
        
        log_info(f"Testing combination {i}/{len(strategy_combinations)}: {combination_name}")
        
        try:
            combo_start_time = time.time()
            
            # Create custom feature engineer with combination
            feature_engineer = CombinedFeatureStrategy(combination)
            
            # Generate features
            print(f"\n🔧 Generating features for combination...")
            log_info(f"Generating features for combination {i}/{len(strategy_combinations)}: {combination_name}")
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
                    "progress/total": len(strategy_combinations),
                    "progress/percentage": i / len(strategy_combinations) * 100,
                    "progress/current_best_roc": best_auc,
                    "progress/elapsed_time": time.time() - start_time,
                    "progress/estimated_remaining": (time.time() - start_time) / i * (len(strategy_combinations) - i)
                })
            
            # Progress update
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / i * (len(strategy_combinations) - i)
            print(f"\n📊 PROGRESS UPDATE:")
            print(f"   ✅ Completed: {i}/{len(strategy_combinations)} combinations ({i/len(strategy_combinations)*100:.1f}%)")
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
                    "progress/total": len(strategy_combinations),
                    "progress/percentage": i / len(strategy_combinations) * 100
                })
    
    # Final summary
    total_time = time.time() - start_time
    successful_runs = sum(1 for r in results.values() if r['success'])
    failed_runs = len(results) - successful_runs
    
    print(f"\n{'='*80}")
    print(f"🏆 4-STRATEGY COMBINATIONS SWEEP RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Sort results by ROC AUC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
    
    print(f"{'Rank':<4} {'Combo':<20} {'Status':<8} {'ROC AUC':<8} {'Time (s)':<8} {'Features':<8} {'Epochs':<8} {'Strategies':<8} {'Top Strategies'}")
    print("-"*100)
    
    for rank, (combo_name, result) in enumerate(sorted_results, 1):
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
        
        print(f"{rank:<4} {combo_name:<20} {status:<8} {roc_auc:<8.4f} {time_taken:<8.1f} {features:<8} {epochs:<8} {strategy_count:<8} {top_strategies}")
    
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
    print(f"   🎯 Combinations tested: {len(strategy_combinations)}")
    print(f"   ✅ Successful runs: {successful_runs}")
    print(f"   ❌ Failed runs: {failed_runs}")
    print(f"   📈 Average time per combo: {total_time/len(strategy_combinations):.1f} seconds")
    print(f"   📊 Average ROC AUC: {np.mean([r['roc_auc'] for r in results.values() if r['success']]):.4f}")
    print(f"   🎯 Best ROC AUC: {best_auc:.4f}")
    print(f"   📉 Worst ROC AUC: {min([r['roc_auc'] for r in results.values() if r['success']]):.4f}")
    
    # Log final summary to W&B
    if wandb_run:
        wandb.log({
            "summary/total_time": total_time,
            "summary/combinations_tested": len(strategy_combinations),
            "summary/successful_runs": successful_runs,
            "summary/failed_runs": failed_runs,
            "summary/avg_time_per_combo": total_time / len(strategy_combinations),
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
    
    print(f"\n🎉 4-strategy combinations sweep completed!")
    print(f"{'='*80}")
    
    if wandb_run:
        wandb.finish()
    
    return results


def main():
    """Main function to run 4-strategy combinations sweep."""
    try:
        results = run_4_strategy_combinations_sweep()
        
        if results:
            print(f"\n✅ 4-strategy combinations sweep completed successfully!")
            print(f"   Check W&B dashboard for detailed results")
            print(f"   Best combination identified")
        else:
            print(f"\n❌ 4-strategy combinations sweep failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 4-strategy combinations sweep interrupted by user")
    except Exception as e:
        log_error(f"4-strategy combinations sweep failed: {str(e)}")
        print(f"\n❌ 4-strategy combinations sweep failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 