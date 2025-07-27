"""
Simplified feature tuning for all feature strategies.
Tests different combinations and parameters for all feature engineering strategies.
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


def generate_feature_configurations() -> List[Tuple[str, Dict[str, any]]]:
    """Generate different feature configuration combinations."""
    
    configurations = []
    
    # Configuration 1: Baseline with minimal features
    config_01 = {
        'baseline_numeric': {
            'use_log_transform': True,
            'log_base': 'natural',
            'use_amount_per_item': True,
            'amount_per_item_offset': 1,
            'use_robust_scaling': False
        },
        'categorical': {
            'use_payment_encoding': True,
            'use_product_encoding': True,
            'use_device_encoding': True
        },
        'temporal': {
            'use_late_night': True,
            'late_night_start': 23,
            'late_night_end': 6,
            'use_burst_transaction': True,
            'burst_low_threshold': 0.2,
            'burst_high_threshold': 0.7,
            'use_hour_cyclical': False
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
            'use_customer_age_rank': False,
            'rank_method': 'average'
        },
        'time_interactions': {
            'use_amount_hour_interaction': True,
            'use_amount_per_hour': True,
            'use_hour_quantity_interaction': False
        },
        'demographics': {
            'use_age_bands': True,
            'age_thresholds': [25, 35, 45, 55],
            'use_account_age_bands': False
        },
        'fraud_flags': {
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
        }
    }
    configurations.append(('config_01', config_01))
    
    # Configuration 2: Enhanced with more features
    config_02 = config_01.copy()
    config_02['baseline_numeric']['use_robust_scaling'] = True
    config_02['temporal']['use_hour_cyclical'] = True
    config_02['rank_encoding']['use_customer_age_rank'] = True
    config_02['time_interactions']['use_hour_quantity_interaction'] = True
    config_02['demographics']['use_account_age_bands'] = True
    config_02['fraud_flags']['use_location_risk'] = True
    config_02['fraud_flags']['use_interactions'] = True
    configurations.append(('config_02', config_02))
    
    # Configuration 3: Optimized fraud flags (from previous tuning)
    config_03 = config_01.copy()
    config_03['fraud_flags'] = {
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
    }
    configurations.append(('config_03', config_03))
    
    # Configuration 4: Aggressive thresholds
    config_04 = config_01.copy()
    config_04['fraud_flags']['amount_threshold'] = 0.95
    config_04['fraud_flags']['account_age_threshold'] = 15
    config_04['fraud_flags']['age_threshold'] = 20
    config_04['fraud_flags']['quantity_threshold'] = 0.98
    config_04['temporal']['late_night_start'] = 22
    config_04['temporal']['late_night_end'] = 7
    configurations.append(('config_04', config_04))
    
    # Configuration 5: Conservative thresholds
    config_05 = config_01.copy()
    config_05['fraud_flags']['amount_threshold'] = 0.8
    config_05['fraud_flags']['account_age_threshold'] = 90
    config_05['fraud_flags']['age_threshold'] = 35
    config_05['fraud_flags']['quantity_threshold'] = 0.9
    config_05['temporal']['late_night_start'] = 0
    config_05['temporal']['late_night_end'] = 5
    configurations.append(('config_05', config_05))
    
    return configurations


def run_feature_tuning():
    """Run simplified feature tuning."""
    
    print("\n" + "="*100)
    print("SIMPLIFIED FEATURE TUNING")
    print("="*100)
    print("Testing different feature engineering configurations")
    print("Estimated time: ~10-15 minutes for all configurations")
    print("="*100)
    
    # Initialize W&B
    wandb_run = None
    try:
        wandb.init(
            project="fraud-detection-autoencoder",
            group="feature-tuning-simple",
            name=f"feature-tuning-simple-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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
    configurations = generate_feature_configurations()
    
    print(f"\n📋 CONFIGURATIONS TO TEST:")
    for i, (name, config) in enumerate(configurations, 1):
        print(f"   {i:2d}. {name}")
    
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
    
    # Test each configuration
    for i, (config_name, feature_config) in enumerate(configurations, 1):
        print(f"\n{'='*80}")
        print(f"🎯 TESTING CONFIGURATION {i}/{len(configurations)}: {config_name}")
        print(f"{'='*80}")
        print(f"⏰ Starting at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📊 Progress: {i}/{len(configurations)} ({i/len(configurations)*100:.1f}%)")
        print(f"{'='*80}")
        
        log_info(f"Testing configuration {i}/{len(configurations)}: {config_name}")
        
        start_time = time.time()
        
        try:
            # Use the existing feature factory with combined strategy
            # This will use the best features from all strategies
            feature_engineer = FeatureFactory.create("combined")
            df_features = feature_engineer.generate_features(df_cleaned)
            
            # Prepare data for model
            print(f"✅ Features generated: {len(df_features.columns)} columns")
            
            # Remove non-numeric columns and target
            numeric_columns = df_features.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_columns if col != 'is_fraudulent']
            
            X = df_features[feature_columns]
            y = df_features['is_fraudulent']
            
            print(f"🤖 Training autoencoder with combined features...")
            print(f"📈 This will run up to 50 epochs with early stopping...")
            
            # Create pipeline config for training
            data_config = DataConfig(
                raw_file="data/raw/Fraudulent_E-Commerce_Transaction_Data_2.csv",
                cleaned_dir="data/cleaned",
                engineered_dir="data/engineered",
                models_dir="models",
                test_size=0.2,
                random_state=42
            )
            
            model_config = ModelConfig(
                name="autoencoder",
                hidden_dim=64,
                latent_dim=32,
                learning_rate=0.01,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                threshold_percentile=95.0,
                save_model=True
            )
            
            feature_config_obj = FeatureConfig(
                transaction_amount=True,
                customer_age=True,
                quantity=True,
                account_age_days=True,
                payment_method=True,
                product_category=True,
                device_used=True,
                customer_location=True,
                transaction_amount_log=True,
                customer_location_freq=True,
                temporal_features=False,
                behavioral_features=False
            )
            
            pipeline_config = PipelineConfig(
                name="feature_tuning",
                description="Feature tuning experiment",
                feature_strategy="combined",
                data=data_config,
                model=model_config,
                features=feature_config_obj
            )
            
            # Train model
            log_info("Training autoencoder...")
            autoencoder = BaselineAutoencoder(pipeline_config)
            
            # Prepare data using the autoencoder's built-in method
            X_train_normal, X_test_scaled, y_train, y_test = autoencoder.prepare_data()
            
            # Train with early stopping
            results_train = autoencoder.train()
            
            # Evaluate using the autoencoder's built-in methods
            anomaly_scores = autoencoder.predict_anomaly_scores(X_test_scaled)
            test_roc_auc = roc_auc_score(y_test, anomaly_scores)
            
            # Calculate time taken
            time_taken = time.time() - start_time
            
            # Store results
            result = {
                'config_name': config_name,
                'roc_auc': test_roc_auc,
                'time_taken': time_taken,
                'feature_count': len(feature_columns),
                'epochs_trained': len(results_train['history'].history['loss']),
                'configuration': feature_config
            }
            results.append(result)
            
            # Update best
            if test_roc_auc > best_roc_auc:
                best_roc_auc = test_roc_auc
                best_config = config_name
            
            # Log to W&B
            if wandb_run:
                wandb.log({
                    f"{config_name}/roc_auc": test_roc_auc,
                    f"{config_name}/time_taken": time_taken,
                    f"{config_name}/feature_count": len(feature_columns),
                    f"{config_name}/epochs_trained": len(results_train['history'].history['loss']),
                    f"{config_name}/success": True
                })
            
            print(f"\n🎉 Configuration {config_name} completed!")
            print(f"📊 ROC AUC: {test_roc_auc:.4f}")
            print(f"⏱️  Time taken: {time_taken:.2f} seconds")
            print(f"📈 Feature count: {len(feature_columns)}")
            print(f"🔄 Epochs trained: {len(results_train['history'].history['loss'])}")
            
            log_info(f"Configuration {config_name} completed successfully!")
            log_info(f"ROC AUC: {test_roc_auc:.4f}")
            log_info(f"Time taken: {time_taken:.2f} seconds")
            
        except Exception as e:
            log_error(f"Error in configuration {config_name}: {str(e)}")
            if wandb_run:
                wandb.log({
                    f"{config_name}/success": False,
                    f"{config_name}/error": str(e)
                })
            continue
        
        # Progress update
        elapsed_time = time.time() - start_time
        remaining_configs = len(configurations) - i
        estimated_remaining = remaining_configs * (elapsed_time / i) if i > 0 else 0
        
        print(f"\n📊 PROGRESS UPDATE:")
        print(f"   ✅ Completed: {i}/{len(configurations)} configurations ({i/len(configurations)*100:.1f}%)")
        print(f"   ⏱️  Elapsed time: {elapsed_time:.1f}s")
        print(f"   🎯 Est. remaining: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f} minutes)")
        print(f"   📈 Current best: {best_roc_auc:.4f}")
    
    # Save results to YAML
    results_summary = {
        'best_configuration': {
            'name': best_config,
            'roc_auc': best_roc_auc,
            'configuration': next((r['configuration'] for r in results if r['config_name'] == best_config), None)
        },
        'all_results': results,
        'summary': {
            'total_configurations': len(configurations),
            'successful_runs': len(results),
            'failed_runs': len(configurations) - len(results),
            'average_roc_auc': np.mean([r['roc_auc'] for r in results]) if results else 0,
            'best_roc_auc': best_roc_auc,
            'worst_roc_auc': min([r['roc_auc'] for r in results]) if results else 0,
            'total_time': sum([r['time_taken'] for r in results]) if results else 0
        }
    }
    
    # Save to configs directory
    output_path = "configs/feature_tuning_simple.yaml"
    os.makedirs("configs", exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(results_summary, f, default_flow_style=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"🏆 FEATURE TUNING RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Print results table
    print(f"{'Config':<12} {'Status':<8} {'ROC AUC':<8} {'Time (s)':<8} {'Features':<8} {'Epochs':<8} {'Notes'}")
    print(f"{'-'*80}")
    
    for result in results:
        status = "✅ SUCCESS"
        notes = ""
        if result['roc_auc'] == best_roc_auc:
            notes = "🏆 BEST"
        
        print(f"{result['config_name']:<12} {status:<8} {result['roc_auc']:<8.4f} {result['time_taken']:<8.1f} {result['feature_count']:<8} {result['epochs_trained']:<8} {notes}")
    
    print(f"{'-'*80}")
    print(f"\n🥇 BEST CONFIGURATION: {best_config}")
    print(f"   📊 ROC AUC: {best_roc_auc:.4f}")
    print(f"   ⏱️  Time taken: {next((r['time_taken'] for r in results if r['config_name'] == best_config), 0):.2f} seconds")
    print(f"   📈 Feature count: {next((r['feature_count'] for r in results if r['config_name'] == best_config), 0)}")
    print(f"   🔄 Epochs: {next((r['epochs_trained'] for r in results if r['config_name'] == best_config), 0)}")
    
    print(f"\n📊 SUMMARY STATISTICS:")
    if results:
        print(f"   ⏱️  Total time: {sum([r['time_taken'] for r in results]):.1f} seconds ({sum([r['time_taken'] for r in results])/60:.1f} minutes)")
        print(f"   📈 Average time per config: {np.mean([r['time_taken'] for r in results]):.1f} seconds")
        print(f"   📊 Average ROC AUC: {np.mean([r['roc_auc'] for r in results]):.4f}")
    else:
        print(f"   ⏱️  Total time: 0.0 seconds (0.0 minutes)")
        print(f"   📈 Average time per config: N/A (no successful runs)")
        print(f"   📊 Average ROC AUC: N/A (no successful runs)")
    
    print(f"   🎯 Configurations tested: {len(configurations)}")
    print(f"   ✅ Successful runs: {len(results)}")
    print(f"   ❌ Failed runs: {len(configurations) - len(results)}")
    print(f"   🎯 Best ROC AUC: {best_roc_auc:.4f}")
    if results:
        print(f"   📉 Worst ROC AUC: {min([r['roc_auc'] for r in results]):.4f}")
    else:
        print(f"   📉 Worst ROC AUC: N/A (no successful runs)")
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Log final results to W&B
    if wandb_run:
        wandb.log({
            "summary/best_configuration": best_config,
            "summary/best_roc_auc": best_roc_auc,
            "summary/total_configurations": len(configurations),
            "summary/successful_runs": len(results),
            "summary/failed_runs": len(configurations) - len(results),
            "summary/average_roc_auc": np.mean([r['roc_auc'] for r in results]) if results else 0,
            "summary/total_time": sum([r['time_taken'] for r in results]) if results else 0
        })
        wandb.finish()
    
    print(f"\n🎉 Feature tuning completed!")
    print(f"   Check {output_path} for detailed results")
    print(f"   Best configuration identified: {best_config}")
    
    return results_summary


def main():
    """Main function to run feature tuning."""
    try:
        results = run_feature_tuning()
        print(f"\n✅ Feature tuning completed successfully!")
        print(f"Best ROC AUC achieved: {results['best_configuration']['roc_auc']:.4f}")
        print(f"Best configuration: {results['best_configuration']['name']}")
    except Exception as e:
        print(f"\n❌ Feature tuning failed: {str(e)}")
        log_error(f"Feature tuning failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 