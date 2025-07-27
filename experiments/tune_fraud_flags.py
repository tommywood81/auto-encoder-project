"""
Hyperparameter tuning for fraud_flags feature strategy.
Tests different combinations of fraud indicators and thresholds.
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import PipelineConfig
from src.data.data_cleaning import DataCleaner
from src.feature_factory.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder
from src.evaluation.evaluator import FraudEvaluator

# Use print statements instead of logging to avoid hanging issues
logger = None
def log_info(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} - INFO - {msg}")

def log_error(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} - ERROR - {msg}")


class TunedFraudFlags:
    """Custom fraud flags strategy with tunable parameters."""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.feature_count = 0
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate tunable fraud flag features."""
        log_info("Generating tuned fraud flag features...")
        
        df = df.copy()
        
        # High amount flag with tunable threshold
        amount_threshold = self.config.get('amount_threshold', 0.9)
        df['high_amount_flag'] = (df['transaction_amount'] > df['transaction_amount'].quantile(amount_threshold)).astype(int)
        
        # New account flag with tunable threshold
        account_age_threshold = self.config.get('account_age_threshold', 30)
        df['new_account_flag'] = (df['account_age_days'] < account_age_threshold).astype(int)
        
        # Young customer flag with tunable threshold
        age_threshold = self.config.get('age_threshold', 25)
        df['young_customer_flag'] = (df['customer_age'] < age_threshold).astype(int)
        
        # Late night flag (if enabled)
        if self.config.get('use_late_night', True):
            df['late_night_flag'] = ((df['transaction_hour'] >= 23) | (df['transaction_hour'] <= 6)).astype(int)
        
        # High quantity flag (if enabled)
        if self.config.get('use_high_quantity', False):
            quantity_threshold = self.config.get('quantity_threshold', 0.95)
            df['high_quantity_flag'] = (df['quantity'] > df['quantity'].quantile(quantity_threshold)).astype(int)
        
        # Unusual location flag (if enabled)
        if self.config.get('use_location_risk', False):
            location_freq = df['customer_location'].value_counts()
            rare_locations = location_freq[location_freq < location_freq.quantile(0.1)].index
            df['unusual_location_flag'] = df['customer_location'].isin(rare_locations).astype(int)
        
        # Calculate fraud risk score with tunable weights
        risk_score = 0
        risk_score += self.config.get('weight_high_amount', 1) * df['high_amount_flag']
        risk_score += self.config.get('weight_new_account', 1) * df['new_account_flag']
        risk_score += self.config.get('weight_young_customer', 1) * df['young_customer_flag']
        
        if self.config.get('use_late_night', True):
            risk_score += self.config.get('weight_late_night', 1) * df['late_night_flag']
        
        if self.config.get('use_high_quantity', False):
            risk_score += self.config.get('weight_high_quantity', 1) * df['high_quantity_flag']
        
        if self.config.get('use_location_risk', False):
            risk_score += self.config.get('weight_location', 1) * df['unusual_location_flag']
        
        df['fraud_risk_score'] = risk_score
        
        # Add interaction features if enabled
        if self.config.get('use_interactions', False):
            df['amount_age_interaction'] = df['high_amount_flag'] * df['young_customer_flag']
            df['account_age_interaction'] = df['new_account_flag'] * df['high_amount_flag']
        
        self.feature_count = len(df.columns)
        log_info(f"Tuned fraud flags generated: {self.feature_count} features")
        return df
    
    def get_feature_info(self) -> Dict[str, any]:
        return {
            "strategy": "tuned_fraud_flags",
            "description": f"Tuned fraud flags with config: {self.config}"
        }


def run_fraud_flags_tuning():
    """Run hyperparameter tuning for fraud flags strategy."""
    
    print("🎯 INITIALIZING FRAUD FLAGS HYPERPARAMETER TUNING...")
    print("⏰ Starting at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("📊 Setting up Weights & Biases integration...")
    
    wandb_run = None
    try:
        # Initialize W&B
        wandb.init(
            project="fraud-detection-autoencoder",
            group="fraud-flags-tuning",
            name=f"fraud-flags-tuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.01,
                "test_size": 0.2,
                "random_state": 42,
                "early_stopping_patience": 10,
                "early_stopping_monitor": "val_loss",
                "early_stopping_min_delta": 0.001,
                "target_auc": 0.7208  # Baseline to beat
            }
        )
        wandb_run = wandb.run
        print("✅ Weights & Biases initialized successfully!")
        print(f"🔗 W&B Run: {wandb_run.name}")
    except Exception as e:
        print(f"⚠️  WARNING: Could not initialize W&B: {str(e)}")
        print("🔄 Continuing without W&B logging...")
        wandb_run = None
    
    # Define hyperparameter combinations to test
    tuning_configs = [
        # Config 1: Baseline (original fraud_flags)
        {
            'amount_threshold': 0.9,
            'account_age_threshold': 30,
            'age_threshold': 25,
            'use_late_night': True,
            'use_high_quantity': False,
            'use_location_risk': False,
            'use_interactions': False,
            'weight_high_amount': 1,
            'weight_new_account': 1,
            'weight_young_customer': 1,
            'weight_late_night': 1
        },
        
        # Config 2: More aggressive thresholds
        {
            'amount_threshold': 0.85,
            'account_age_threshold': 60,
            'age_threshold': 30,
            'use_late_night': True,
            'use_high_quantity': False,
            'use_location_risk': False,
            'use_interactions': False,
            'weight_high_amount': 1,
            'weight_new_account': 1,
            'weight_young_customer': 1,
            'weight_late_night': 1
        },
        
        # Config 3: Add high quantity flag
        {
            'amount_threshold': 0.9,
            'account_age_threshold': 30,
            'age_threshold': 25,
            'use_late_night': True,
            'use_high_quantity': True,
            'quantity_threshold': 0.95,
            'use_location_risk': False,
            'use_interactions': False,
            'weight_high_amount': 1,
            'weight_new_account': 1,
            'weight_young_customer': 1,
            'weight_late_night': 1,
            'weight_high_quantity': 1
        },
        
        # Config 4: Add location risk
        {
            'amount_threshold': 0.9,
            'account_age_threshold': 30,
            'age_threshold': 25,
            'use_late_night': True,
            'use_high_quantity': False,
            'use_location_risk': True,
            'use_interactions': False,
            'weight_high_amount': 1,
            'weight_new_account': 1,
            'weight_young_customer': 1,
            'weight_late_night': 1,
            'weight_location': 1
        },
        
        # Config 5: Add interaction features
        {
            'amount_threshold': 0.9,
            'account_age_threshold': 30,
            'age_threshold': 25,
            'use_late_night': True,
            'use_high_quantity': False,
            'use_location_risk': False,
            'use_interactions': True,
            'weight_high_amount': 1,
            'weight_new_account': 1,
            'weight_young_customer': 1,
            'weight_late_night': 1
        },
        
        # Config 6: All features with balanced weights
        {
            'amount_threshold': 0.9,
            'account_age_threshold': 30,
            'age_threshold': 25,
            'use_late_night': True,
            'use_high_quantity': True,
            'quantity_threshold': 0.95,
            'use_location_risk': True,
            'use_interactions': True,
            'weight_high_amount': 1,
            'weight_new_account': 1,
            'weight_young_customer': 1,
            'weight_late_night': 1,
            'weight_high_quantity': 1,
            'weight_location': 1
        },
        
        # Config 7: Weighted towards high amount
        {
            'amount_threshold': 0.85,
            'account_age_threshold': 30,
            'age_threshold': 25,
            'use_late_night': True,
            'use_high_quantity': True,
            'quantity_threshold': 0.95,
            'use_location_risk': True,
            'use_interactions': True,
            'weight_high_amount': 2,
            'weight_new_account': 1,
            'weight_young_customer': 1,
            'weight_late_night': 1,
            'weight_high_quantity': 1,
            'weight_location': 1
        },
        
        # Config 8: Weighted towards new accounts
        {
            'amount_threshold': 0.9,
            'account_age_threshold': 30,
            'age_threshold': 25,
            'use_late_night': True,
            'use_high_quantity': True,
            'quantity_threshold': 0.95,
            'use_location_risk': True,
            'use_interactions': True,
            'weight_high_amount': 1,
            'weight_new_account': 2,
            'weight_young_customer': 1,
            'weight_late_night': 1,
            'weight_high_quantity': 1,
            'weight_location': 1
        },
        
        # Config 9: Conservative thresholds with all features
        {
            'amount_threshold': 0.95,
            'account_age_threshold': 15,
            'age_threshold': 20,
            'use_late_night': True,
            'use_high_quantity': True,
            'quantity_threshold': 0.98,
            'use_location_risk': True,
            'use_interactions': True,
            'weight_high_amount': 1,
            'weight_new_account': 1,
            'weight_young_customer': 1,
            'weight_late_night': 1,
            'weight_high_quantity': 1,
            'weight_location': 1
        },
        
        # Config 10: Aggressive thresholds with all features
        {
            'amount_threshold': 0.8,
            'account_age_threshold': 90,
            'age_threshold': 35,
            'use_late_night': True,
            'use_high_quantity': True,
            'quantity_threshold': 0.9,
            'use_location_risk': True,
            'use_interactions': True,
            'weight_high_amount': 1,
            'weight_new_account': 1,
            'weight_young_customer': 1,
            'weight_late_night': 1,
            'weight_high_quantity': 1,
            'weight_location': 1
        }
    ]
    
    print(f"\n🎯 FRAUD FLAGS HYPERPARAMETER TUNING")
    print("="*80)
    print(f"📊 Testing {len(tuning_configs)} configurations")
    print(f"⏰ Estimated time: ~15-20 minutes")
    if wandb_run:
        print(f"📈 Logging to Weights & Biases: {wandb_run.name}")
    print("="*80)
    
    log_info("Starting fraud flags hyperparameter tuning...")
    log_info(f"Testing {len(tuning_configs)} configurations")
    if wandb_run:
        log_info(f"W&B Run: {wandb_run.name}")
    
    # Load and clean data once
    print(f"\n🔄 STEP 1: Loading and cleaning data...")
    log_info("Loading and cleaning data...")
    
    try:
        config = PipelineConfig.get_fraud_flags_config()
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
    best_auc = 0.7208  # Baseline to beat
    best_config = None
    
    # Test each configuration
    for i, tuning_config in enumerate(tuning_configs, 1):
        config_name = f"config_{i:02d}"
        
        print(f"\n{'='*80}")
        print(f"🎯 TESTING CONFIGURATION {i}/{len(tuning_configs)}: {config_name}")
        print(f"{'='*80}")
        print(f"📝 Configuration: {tuning_config}")
        print(f"⏰ Starting at: {time.strftime('%H:%M:%S')}")
        print(f"📊 Progress: {i}/{len(tuning_configs)} ({i/len(tuning_configs)*100:.1f}%)")
        print(f"{'='*80}")
        
        log_info(f"Testing configuration {i}/{len(tuning_configs)}: {config_name}")
        
        try:
            config_start_time = time.time()
            
            # Create custom feature engineer with tuning config
            feature_engineer = TunedFraudFlags(tuning_config)
            
            # Generate features
            print(f"\n🔧 Generating tuned fraud flag features...")
            log_info("Generating features...")
            
            df_features = feature_engineer.generate_features(df_cleaned)
            
            feature_count = len(df_features.columns)
            new_features = feature_count - len(df_cleaned.columns)
            log_info(f"Features generated: {feature_count} columns ({new_features} new)")
            print(f"✅ Features generated: {feature_count} columns ({new_features} new)")
            
            # Create a temporary config for training
            temp_config = PipelineConfig.get_fraud_flags_config()
            
            # Train model
            print(f"\n🤖 Training autoencoder with tuned features...")
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
            strategy_time = time.time() - config_start_time
            
            # Calculate training efficiency
            training_efficiency = epochs_trained / 50.0
            
            print(f"\n🎉 Configuration {config_name} completed!")
            print(f"📊 ROC AUC: {roc_auc:.4f}")
            print(f"🎯 Threshold: {threshold:.4f}")
            print(f"⏱️  Time taken: {strategy_time:.2f} seconds")
            print(f"📈 Feature count: {feature_count} ({new_features} new)")
            print(f"🔄 Epochs trained: {epochs_trained}")
            
            log_info(f"Configuration {config_name} completed successfully!")
            log_info(f"ROC AUC: {roc_auc:.4f}")
            log_info(f"Time taken: {strategy_time:.2f} seconds")
            
            # Store results
            results[config_name] = {
                'roc_auc': roc_auc,
                'threshold': threshold,
                'time_taken': strategy_time,
                'feature_count': feature_count,
                'new_features': new_features,
                'epochs_trained': epochs_trained,
                'training_efficiency': training_efficiency,
                'config': tuning_config,
                'success': True
            }
            
            # Update best if improved
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_config = config_name
                print(f"🏆 NEW BEST! ROC AUC: {roc_auc:.4f} (improvement: {roc_auc - 0.7208:.4f})")
            
            # Log to W&B
            if wandb_run:
                wandb.log({
                    f"{config_name}/roc_auc": roc_auc,
                    f"{config_name}/threshold": threshold,
                    f"{config_name}/time_taken": strategy_time,
                    f"{config_name}/feature_count": feature_count,
                    f"{config_name}/new_features": new_features,
                    f"{config_name}/epochs_trained": epochs_trained,
                    f"{config_name}/training_efficiency": training_efficiency,
                    f"{config_name}/success": True,
                    f"{config_name}/amount_threshold": tuning_config.get('amount_threshold', 0.9),
                    f"{config_name}/account_age_threshold": tuning_config.get('account_age_threshold', 30),
                    f"{config_name}/age_threshold": tuning_config.get('age_threshold', 25),
                    f"{config_name}/use_late_night": tuning_config.get('use_late_night', True),
                    f"{config_name}/use_high_quantity": tuning_config.get('use_high_quantity', False),
                    f"{config_name}/use_location_risk": tuning_config.get('use_location_risk', False),
                    f"{config_name}/use_interactions": tuning_config.get('use_interactions', False),
                    f"{config_name}/weight_high_amount": tuning_config.get('weight_high_amount', 1),
                    f"{config_name}/weight_new_account": tuning_config.get('weight_new_account', 1),
                    f"{config_name}/weight_young_customer": tuning_config.get('weight_young_customer', 1),
                    "progress/completed": i,
                    "progress/total": len(tuning_configs),
                    "progress/percentage": i / len(tuning_configs) * 100,
                    "progress/current_best_roc": best_auc,
                    "progress/elapsed_time": time.time() - start_time,
                    "progress/estimated_remaining": (time.time() - start_time) / i * (len(tuning_configs) - i)
                })
            
            # Progress update
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / i * (len(tuning_configs) - i)
            print(f"\n📊 PROGRESS UPDATE:")
            print(f"   ✅ Completed: {i}/{len(tuning_configs)} configurations ({i/len(tuning_configs)*100:.1f}%)")
            print(f"   ⏱️  Elapsed time: {elapsed_time:.1f}s")
            print(f"   🎯 Est. remaining: {remaining_time:.1f}s ({remaining_time/60:.1f} minutes)")
            print(f"   📈 Current best: {best_auc:.4f}")
            
        except Exception as e:
            print(f"❌ ERROR in configuration {config_name}: {str(e)}")
            log_error(f"Configuration {config_name} failed: {str(e)}")
            
            results[config_name] = {
                'roc_auc': 0.0,
                'threshold': 0.0,
                'time_taken': 0.0,
                'feature_count': 0,
                'new_features': 0,
                'epochs_trained': 0,
                'training_efficiency': 0.0,
                'config': tuning_config,
                'success': False,
                'error': str(e)
            }
            
            if wandb_run:
                wandb.log({
                    f"{config_name}/success": False,
                    f"{config_name}/error": str(e),
                    "progress/completed": i,
                    "progress/total": len(tuning_configs),
                    "progress/percentage": i / len(tuning_configs) * 100
                })
    
    # Final summary
    total_time = time.time() - start_time
    successful_runs = sum(1 for r in results.values() if r['success'])
    failed_runs = len(results) - successful_runs
    
    print(f"\n{'='*80}")
    print(f"🏆 FRAUD FLAGS TUNING RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Sort results by ROC AUC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
    
    print(f"{'Config':<12} {'Status':<8} {'ROC AUC':<8} {'Time (s)':<8} {'Features':<8} {'Epochs':<8} {'Notes'}")
    print("-"*80)
    
    for config_name, result in sorted_results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        roc_auc = result['roc_auc']
        time_taken = result['time_taken']
        features = result['feature_count']
        epochs = result['epochs_trained']
        
        # Create notes
        config = result['config']
        notes = []
        if config.get('use_high_quantity', False):
            notes.append("+quantity")
        if config.get('use_location_risk', False):
            notes.append("+location")
        if config.get('use_interactions', False):
            notes.append("+interactions")
        if config.get('weight_high_amount', 1) > 1:
            notes.append("wgt_amt")
        if config.get('weight_new_account', 1) > 1:
            notes.append("wgt_acc")
        
        notes_str = ", ".join(notes) if notes else "baseline"
        
        print(f"{config_name:<12} {status:<8} {roc_auc:<8.4f} {time_taken:<8.1f} {features:<8} {epochs:<8} {notes_str}")
    
    print("-"*80)
    
    if best_config:
        best_result = results[best_config]
        improvement = best_result['roc_auc'] - 0.7208
        print(f"\n🥇 BEST CONFIGURATION: {best_config}")
        print(f"   📊 ROC AUC: {best_result['roc_auc']:.4f}")
        print(f"   📈 Improvement over baseline: {improvement:.4f} ({improvement/0.7208*100:.2f}%)")
        print(f"   🎯 Threshold: {best_result['threshold']:.4f}")
        print(f"   ⏱️  Time taken: {best_result['time_taken']:.2f} seconds")
        print(f"   📈 Features: {best_result['feature_count']} ({best_result['new_features']} new)")
        print(f"   🔄 Epochs: {best_result['epochs_trained']}")
        print(f"   📝 Configuration: {best_result['config']}")
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   ⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   🎯 Configurations tested: {len(tuning_configs)}")
    print(f"   ✅ Successful runs: {successful_runs}")
    print(f"   ❌ Failed runs: {failed_runs}")
    print(f"   📈 Average time per config: {total_time/len(tuning_configs):.1f} seconds")
    print(f"   📊 Average ROC AUC: {np.mean([r['roc_auc'] for r in results.values() if r['success']]):.4f}")
    print(f"   🎯 Best ROC AUC: {best_auc:.4f}")
    print(f"   📉 Worst ROC AUC: {min([r['roc_auc'] for r in results.values() if r['success']]):.4f}")
    
    # Log final summary to W&B
    if wandb_run:
        wandb.log({
            "summary/total_time": total_time,
            "summary/configurations_tested": len(tuning_configs),
            "summary/successful_runs": successful_runs,
            "summary/failed_runs": failed_runs,
            "summary/avg_time_per_config": total_time / len(tuning_configs),
            "summary/avg_roc_auc": np.mean([r['roc_auc'] for r in results.values() if r['success']]),
            "summary/best_roc_auc": best_auc,
            "summary/worst_roc_auc": min([r['roc_auc'] for r in results.values() if r['success']]),
            "summary/best_config": best_config,
            "summary/improvement_over_baseline": best_auc - 0.7208 if best_auc > 0.7208 else 0
        })
        
        # Log best configuration details
        if best_config:
            best_result = results[best_config]
            wandb.log({
                "best_config/name": best_config,
                "best_config/roc_auc": best_result['roc_auc'],
                "best_config/threshold": best_result['threshold'],
                "best_config/time_taken": best_result['time_taken'],
                "best_config/feature_count": best_result['feature_count'],
                "best_config/epochs_trained": best_result['epochs_trained'],
                "best_config/amount_threshold": best_result['config'].get('amount_threshold', 0.9),
                "best_config/account_age_threshold": best_result['config'].get('account_age_threshold', 30),
                "best_config/age_threshold": best_result['config'].get('age_threshold', 25),
                "best_config/use_late_night": best_result['config'].get('use_late_night', True),
                "best_config/use_high_quantity": best_result['config'].get('use_high_quantity', False),
                "best_config/use_location_risk": best_result['config'].get('use_location_risk', False),
                "best_config/use_interactions": best_result['config'].get('use_interactions', False)
            })
    
    print(f"\n🎉 Fraud flags hyperparameter tuning completed!")
    print(f"{'='*80}")
    
    if wandb_run:
        wandb.finish()
    
    return results


def main():
    """Main function to run fraud flags tuning."""
    try:
        results = run_fraud_flags_tuning()
        
        if results:
            print(f"\n✅ Tuning completed successfully!")
            print(f"   Check W&B dashboard for detailed results")
            print(f"   Best configuration identified")
        else:
            print(f"\n❌ Tuning failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Tuning interrupted by user")
    except Exception as e:
        log_error(f"Tuning failed: {str(e)}")
        print(f"\n❌ Tuning failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 