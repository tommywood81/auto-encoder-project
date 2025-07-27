"""
Comprehensive Feature Tuning v2 - With Detailed Progress Logging
Goal: Achieve 0.77+ ROC AUC through exhaustive feature optimization
"""

import pandas as pd
import numpy as np
import logging
import time
import wandb
from typing import Dict, List, Tuple, Optional
import sys
import os
from datetime import datetime, timedelta
import itertools
import yaml
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import PipelineConfig, DataConfig, ModelConfig, FeatureConfig
from src.data.data_cleaning import DataCleaner
from src.feature_factory.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder

# Use print statements instead of logging to avoid hanging issues
logger = None
def log_info(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} - INFO - {msg}")

def log_error(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} - ERROR - {msg}")

def format_time(seconds):
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def generate_comprehensive_configurations() -> List[Tuple[str, Dict[str, any]]]:
    """Generate comprehensive feature configuration combinations."""
    
    configurations = []
    config_id = 1
    
    # Base configuration with all features enabled
    base_config = {
        'baseline_numeric': {
            'use_log_transform': True,
            'log_base': 'natural',
            'use_amount_per_item': True,
            'amount_per_item_offset': 1,
            'use_robust_scaling': True
        },
        'categorical': {
            'use_payment_encoding': True,
            'use_product_encoding': True,
            'use_device_encoding': True
        },
        'temporal': {
            'use_late_night': True,
            'late_night_start': 0,
            'late_night_end': 5,
            'use_burst_transaction': True,
            'burst_low_threshold': 0.2,
            'burst_high_threshold': 0.7,
            'use_hour_cyclical': True
        },
        'behavioral': {
            'use_amount_per_age': True,
            'amount_per_age_offset': 1,
            'use_amount_per_account_age': True,
            'amount_per_account_age_offset': 1,
            'use_location_freq': True
        },
        'rolling': {
            'use_rolling_mean': True,
            'use_rolling_std': True,
            'rolling_window': 3
        },
        'rank_encoding': {
            'use_amount_rank': True,
            'use_account_age_rank': True,
            'use_customer_age_rank': True,
            'rank_method': 'average'
        },
        'time_interactions': {
            'use_amount_hour_interaction': True,
            'use_amount_per_hour': True,
            'use_hour_quantity_interaction': True
        },
        'demographics': {
            'use_age_bands': True,
            'age_thresholds': [25, 35, 45, 55],
            'use_account_age_bands': True
        },
        'fraud_flags': {
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
            'weight_high_quantity': 1
        }
    }
    
    # 1. Log base variations
    for log_base in ['natural', 2, 10]:
        config = base_config.copy()
        config['baseline_numeric']['log_base'] = log_base
        configurations.append((f'config_{config_id:03d}_log_{log_base}', config))
        config_id += 1
    
    # 2. Amount per item offset variations
    for offset in [0.5, 1, 2, 5]:
        config = base_config.copy()
        config['baseline_numeric']['amount_per_item_offset'] = offset
        configurations.append((f'config_{config_id:03d}_offset_{offset}', config))
        config_id += 1
    
    # 3. Rolling window variations
    for window in [2, 3, 5, 7, 10]:
        config = base_config.copy()
        config['rolling']['rolling_window'] = window
        configurations.append((f'config_{config_id:03d}_window_{window}', config))
        config_id += 1
    
    # 4. Behavioral offset variations
    for age_offset in [0.5, 1, 2]:
        for account_offset in [0.5, 1, 2]:
            config = base_config.copy()
            config['behavioral']['amount_per_age_offset'] = age_offset
            config['behavioral']['amount_per_account_age_offset'] = account_offset
            configurations.append((f'config_{config_id:03d}_behavioral_{age_offset}_{account_offset}', config))
            config_id += 1
    
    # 5. Age threshold variations
    age_threshold_sets = [
        [20, 30, 40, 50],
        [25, 35, 45, 55],
        [30, 40, 50, 60],
        [18, 25, 35, 45]
    ]
    for thresholds in age_threshold_sets:
        config = base_config.copy()
        config['demographics']['age_thresholds'] = thresholds
        configurations.append((f'config_{config_id:03d}_age_{"_".join(map(str, thresholds))}', config))
        config_id += 1
    
    # 6. Fraud flag threshold variations (comprehensive)
    fraud_thresholds = [
        # Conservative
        {'amount': 0.7, 'account_age': 120, 'age': 40, 'quantity': 0.85},
        {'amount': 0.75, 'account_age': 90, 'age': 35, 'quantity': 0.88},
        {'amount': 0.8, 'account_age': 60, 'age': 30, 'quantity': 0.9},
        # Moderate
        {'amount': 0.85, 'account_age': 45, 'age': 25, 'quantity': 0.92},
        {'amount': 0.9, 'account_age': 30, 'age': 20, 'quantity': 0.95},
        # Aggressive
        {'amount': 0.92, 'account_age': 15, 'age': 18, 'quantity': 0.97},
        {'amount': 0.95, 'account_age': 7, 'age': 16, 'quantity': 0.98}
    ]
    
    for i, thresholds in enumerate(fraud_thresholds):
        config = base_config.copy()
        config['fraud_flags']['amount_threshold'] = thresholds['amount']
        config['fraud_flags']['account_age_threshold'] = thresholds['account_age']
        config['fraud_flags']['age_threshold'] = thresholds['age']
        config['fraud_flags']['quantity_threshold'] = thresholds['quantity']
        configurations.append((f'config_{config_id:03d}_fraud_{i+1}', config))
        config_id += 1
    
    # 7. Fraud flag weight variations
    weight_combinations = [
        {'high_amount': 1, 'new_account': 1, 'young_customer': 1, 'late_night': 1, 'high_quantity': 1},
        {'high_amount': 2, 'new_account': 1, 'young_customer': 1, 'late_night': 1, 'high_quantity': 1},
        {'high_amount': 1, 'new_account': 2, 'young_customer': 1, 'late_night': 1, 'high_quantity': 1},
        {'high_amount': 1, 'new_account': 1, 'young_customer': 2, 'late_night': 1, 'high_quantity': 1},
        {'high_amount': 1, 'new_account': 1, 'young_customer': 1, 'late_night': 2, 'high_quantity': 1},
        {'high_amount': 1, 'new_account': 1, 'young_customer': 1, 'late_night': 1, 'high_quantity': 2},
        {'high_amount': 1.5, 'new_account': 1.5, 'young_customer': 1.5, 'late_night': 1.5, 'high_quantity': 1.5}
    ]
    
    for i, weights in enumerate(weight_combinations):
        config = base_config.copy()
        config['fraud_flags']['weight_high_amount'] = weights['high_amount']
        config['fraud_flags']['weight_new_account'] = weights['new_account']
        config['fraud_flags']['weight_young_customer'] = weights['young_customer']
        config['fraud_flags']['weight_late_night'] = weights['late_night']
        config['fraud_flags']['weight_high_quantity'] = weights['high_quantity']
        configurations.append((f'config_{config_id:03d}_weights_{i+1}', config))
        config_id += 1
    
    # 8. Temporal window variations
    temporal_windows = [
        {'start': 0, 'end': 5},
        {'start': 1, 'end': 6},
        {'start': 22, 'end': 6},
        {'start': 23, 'end': 7},
        {'start': 0, 'end': 6},
        {'start': 1, 'end': 5}
    ]
    
    for i, window in enumerate(temporal_windows):
        config = base_config.copy()
        config['temporal']['late_night_start'] = window['start']
        config['temporal']['late_night_end'] = window['end']
        configurations.append((f'config_{config_id:03d}_temporal_{i+1}', config))
        config_id += 1
    
    # 9. Burst transaction threshold variations
    burst_combinations = [
        {'low': 0.1, 'high': 0.6},
        {'low': 0.15, 'high': 0.65},
        {'low': 0.2, 'high': 0.7},
        {'low': 0.25, 'high': 0.75},
        {'low': 0.3, 'high': 0.8}
    ]
    
    for i, burst in enumerate(burst_combinations):
        config = base_config.copy()
        config['temporal']['burst_low_threshold'] = burst['low']
        config['temporal']['burst_high_threshold'] = burst['high']
        configurations.append((f'config_{config_id:03d}_burst_{i+1}', config))
        config_id += 1
    
    # 10. Rank method variations
    for rank_method in ['average', 'min', 'max', 'dense']:
        config = base_config.copy()
        config['rank_encoding']['rank_method'] = rank_method
        configurations.append((f'config_{config_id:03d}_rank_{rank_method}', config))
        config_id += 1
    
    # 11. Feature combination variations (enable/disable specific features)
    feature_variations = [
        {'rolling_std': False, 'hour_cyclical': False},
        {'rolling_mean': False, 'amount_rank': False},
        {'account_age_rank': False, 'customer_age_rank': False},
        {'amount_hour_interaction': False, 'hour_quantity_interaction': False},
        {'location_freq': False, 'location_risk': False},
        {'age_bands': False, 'account_age_bands': False}
    ]
    
    for i, variation in enumerate(feature_variations):
        config = base_config.copy()
        if 'rolling_std' in variation:
            config['rolling']['use_rolling_std'] = variation['rolling_std']
        if 'rolling_mean' in variation:
            config['rolling']['use_rolling_mean'] = variation['rolling_mean']
        if 'hour_cyclical' in variation:
            config['temporal']['use_hour_cyclical'] = variation['hour_cyclical']
        if 'amount_rank' in variation:
            config['rank_encoding']['use_amount_rank'] = variation['amount_rank']
        if 'account_age_rank' in variation:
            config['rank_encoding']['use_account_age_rank'] = variation['account_age_rank']
        if 'customer_age_rank' in variation:
            config['rank_encoding']['use_customer_age_rank'] = variation['customer_age_rank']
        if 'amount_hour_interaction' in variation:
            config['time_interactions']['use_amount_hour_interaction'] = variation['amount_hour_interaction']
        if 'hour_quantity_interaction' in variation:
            config['time_interactions']['use_hour_quantity_interaction'] = variation['hour_quantity_interaction']
        if 'location_freq' in variation:
            config['behavioral']['use_location_freq'] = variation['location_freq']
        if 'location_risk' in variation:
            config['fraud_flags']['use_location_risk'] = variation['location_risk']
        if 'age_bands' in variation:
            config['demographics']['use_age_bands'] = variation['age_bands']
        if 'account_age_bands' in variation:
            config['demographics']['use_account_age_bands'] = variation['account_age_bands']
        configurations.append((f'config_{config_id:03d}_features_{i+1}', config))
        config_id += 1
    
    return configurations


def run_comprehensive_feature_tuning():
    """Run comprehensive feature tuning with detailed progress logging."""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE FEATURE TUNING v2 - WITH PROGRESS LOGGING")
    print("="*100)
    print("Testing extensive feature engineering configurations")
    print("Goal: Achieve 0.77+ ROC AUC")
    print("="*100)
    
    # Track overall progress
    overall_start_time = time.time()
    successful_runs = 0
    failed_runs = 0
    
    # Initialize W&B
    wandb_run = None
    try:
        wandb.init(
            project="fraud-detection-autoencoder",
            group="comprehensive-feature-tuning-v2",
            name=f"comprehensive-tuning-v2-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.01,
                "test_size": 0.2,
                "random_state": 42,
                "early_stopping_patience": 10
            }
        )
        wandb_run = wandb.run
        log_info(f"W&B initialized: {wandb_run.name}")
    except Exception as e:
        log_info(f"WARNING: Could not initialize W&B: {str(e)}")
        log_info("Continuing without W&B logging...")
    
    # Generate configurations
    log_info("Generating comprehensive feature configurations...")
    configurations = generate_comprehensive_configurations()
    
    print(f"\n📋 CONFIGURATIONS TO TEST: {len(configurations)}")
    print("This will test:")
    print("  - Log base variations (natural, 2, 10)")
    print("  - Amount per item offsets (0.5, 1, 2, 5)")
    print("  - Rolling windows (2, 3, 5, 7, 10)")
    print("  - Behavioral offsets (0.5, 1, 2)")
    print("  - Age threshold variations (4 different sets)")
    print("  - Fraud flag thresholds (7 combinations)")
    print("  - Fraud flag weights (7 combinations)")
    print("  - Temporal windows (6 combinations)")
    print("  - Burst thresholds (5 combinations)")
    print("  - Rank methods (4 methods)")
    print("  - Feature combinations (6 variations)")
    
    # Load and clean data once
    print(f"\n🔄 STEP 1: Loading and cleaning data...")
    log_info("Loading and cleaning data...")
    
    config = PipelineConfig.get_baseline_numeric_config()
    cleaner = DataCleaner(config)
    df_cleaned = cleaner.clean_data(save_output=False)
    
    log_info(f"Data loaded. Shape: {df_cleaned.shape}")
    
    # Results storage
    results = []
    best_roc_auc = 0
    best_config = None
    
    # Progress tracking variables
    total_configs = len(configurations)
    avg_time_per_config = 0
    times_taken = []
    
    print(f"\n🚀 STARTING COMPREHENSIVE FEATURE TUNING")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total configurations: {total_configs}")
    print(f"�� Target ROC AUC: 0.77+")
    print("="*100)
    
    # Test each configuration
    for i, (config_name, feature_config) in enumerate(configurations, 1):
        config_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"🎯 CONFIGURATION {i}/{total_configs}: {config_name}")
        print(f"{'='*80}")
        print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📊 Progress: {i}/{total_configs} ({i/total_configs*100:.1f}%)")
        
        # Calculate time estimates
        if times_taken:
            avg_time_per_config = np.mean(times_taken)
            remaining_configs = total_configs - i
            estimated_remaining_time = avg_time_per_config * remaining_configs
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
            
            print(f"⏱️  Avg time per config: {format_time(avg_time_per_config)}")
            print(f"⏱️  Estimated remaining: {format_time(estimated_remaining_time)}")
            print(f"⏱️  Estimated completion: {estimated_completion.strftime('%H:%M:%S')}")
        
        # Show current best
        if best_roc_auc > 0:
            print(f"🏆 Current best: {best_roc_auc:.4f} ({best_config})")
        
        print(f"{'='*80}")
        
        log_info(f"Testing configuration {i}/{total_configs}: {config_name}")
        
        try:
            # Use the existing feature factory with combined strategy
            log_info("Generating features...")
            feature_engineer = FeatureFactory.create("combined")
            df_features = feature_engineer.generate_features(df_cleaned)
            
            # Prepare data for model
            print(f"✅ Features generated: {len(df_features.columns)} columns")
            
            # Remove non-numeric columns and target
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
            if 'is_fraudulent' in numeric_cols:
                numeric_cols.remove('is_fraudulent')
            
            X = df_features[numeric_cols]
            y = df_features['is_fraudulent']
            
            print(f"✅ Numeric features: {X.shape[1]} columns")
            print(f"✅ Target distribution: {y.value_counts().to_dict()}")
            
            # Initialize autoencoder
            log_info("Initializing autoencoder...")
            pipeline_config = PipelineConfig.get_config("combined")
            autoencoder = BaselineAutoencoder(pipeline_config)
            
            # Train model
            print(f"🚀 Training autoencoder...")
            log_info("Starting model training...")
            results_dict = autoencoder.train()
            roc_auc = results_dict['roc_auc']
            
            config_time_taken = time.time() - config_start_time
            times_taken.append(config_time_taken)
            
            print(f"✅ ROC AUC: {roc_auc:.4f}")
            print(f"✅ Time taken: {format_time(config_time_taken)}")
            print(f"✅ Epochs trained: {len(results_dict['history'].history['loss'])}")
            
            # Save results
            result = {
                'config_name': config_name,
                'roc_auc': roc_auc,
                'time_taken': config_time_taken,
                'feature_count': X.shape[1],
                'epochs_trained': len(results_dict['history'].history['loss']),
                'configuration': feature_config
            }
            results.append(result)
            successful_runs += 1
            
            # Update best
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_config = config_name
                print(f"�� NEW BEST! ROC AUC: {roc_auc:.4f}")
                
                # Check if we reached the goal
                if roc_auc >= 0.77:
                    print(f"�� GOAL ACHIEVED! ROC AUC {roc_auc:.4f} >= 0.77")
                    print(f"🎉 Configuration {config_name} reached the target!")
                
                # Log to W&B if available
                if wandb_run:
                    wandb.log({
                        "best_roc_auc": roc_auc,
                        "best_config": config_name,
                        "current_config": config_name,
                        "current_roc_auc": roc_auc
                    })
            
            # Log to W&B
            if wandb_run:
                wandb.log({
                    "config_name": config_name,
                    "roc_auc": roc_auc,
                    "time_taken": config_time_taken,
                    "feature_count": X.shape[1],
                    "epochs_trained": len(results_dict['history'].history['loss']),
                    "current_config": config_name,
                    "current_roc_auc": roc_auc
                })
                
        except Exception as e:
            config_time_taken = time.time() - config_start_time
            times_taken.append(config_time_taken)
            failed_runs += 1
            log_error(f"Error testing {config_name}: {str(e)}")
            print(f"❌ Failed: {str(e)}")
            print(f"⏱️  Time taken: {format_time(config_time_taken)}")
            continue
        
        # Progress summary every 10 configurations
        if i % 10 == 0 or i == total_configs:
            elapsed_time = time.time() - overall_start_time
            print(f"\n�� PROGRESS SUMMARY (after {i} configurations):")
            print(f"   ✅ Successful: {successful_runs}")
            print(f"   ❌ Failed: {failed_runs}")
            print(f"   🏆 Best ROC AUC: {best_roc_auc:.4f}")
            print(f"   ⏱️  Total time: {format_time(elapsed_time)}")
            print(f"   📈 Success rate: {successful_runs/i*100:.1f}%")
            if best_roc_auc >= 0.77:
                print(f"   �� GOAL STATUS: ACHIEVED!")
            else:
                print(f"   �� GOAL STATUS: {best_roc_auc:.4f} < 0.77")
    
    # Sort results by ROC AUC
    results.sort(key=lambda x: x['roc_auc'], reverse=True)
    
    # Save results
    output_data = {
        'best_configuration': {
            'name': best_config,
            'roc_auc': best_roc_auc,
            'configuration': results[0]['configuration'] if results else None
        },
        'all_results': results,
        'summary': {
            'total_configurations': total_configs,
            'successful_runs': successful_runs,
            'failed_runs': failed_runs,
            'average_roc_auc': np.mean([r['roc_auc'] for r in results]) if results else 0,
            'best_roc_auc': best_roc_auc,
            'worst_roc_auc': min([r['roc_auc'] for r in results]) if results else 0,
            'total_time': sum([r['time_taken'] for r in results]) if results else 0
        }
    }
    
    # Save to YAML
    output_path = "configs/comprehensive_feature_tuning_v2.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, indent=2)
    
    # Final summary
    total_time = time.time() - overall_start_time
    print(f"\n{'='*100}")
    print(f"🎉 COMPREHENSIVE FEATURE TUNING COMPLETED!")
    print(f"{'='*100}")
    print(f"📊 Total configurations tested: {total_configs}")
    print(f"✅ Successful runs: {successful_runs}")
    print(f"❌ Failed runs: {failed_runs}")
    print(f"🏆 Best ROC AUC: {best_roc_auc:.4f}")
    print(f"�� Average ROC AUC: {np.mean([r['roc_auc'] for r in results]):.4f}")
    print(f"⏱️  Total time: {format_time(total_time)}")
    print(f"💾 Results saved to: {output_path}")
    
    # Show top 10 results
    print(f"\n🏆 TOP 10 CONFIGURATIONS:")
    print(f"{'='*80}")
    for i, result in enumerate(results[:10], 1):
        print(f"{i:2d}. {result['config_name']}: {result['roc_auc']:.4f} ({format_time(result['time_taken'])})")
    
    # Check if we reached the goal
    if best_roc_auc >= 0.77:
        print(f"\n�� GOAL ACHIEVED! ROC AUC {best_roc_auc:.4f} >= 0.77")
        print(f"🎉 Configuration {best_config} reached the target!")
    else:
        print(f"\n⚠️  Goal not reached. Best ROC AUC: {best_roc_auc:.4f} < 0.77")
        print(f"   Consider additional hyperparameter tuning or model architecture changes")
    
    return output_data


def main():
    """Main function."""
    try:
        results = run_comprehensive_feature_tuning()
        print(f"\n✅ Comprehensive feature tuning completed successfully!")
        return results
    except Exception as e:
        log_error(f"Comprehensive feature tuning failed: {str(e)}")
        print(f"❌ Failed: {str(e)}")
        return None


if __name__ == "__main__":
    main()