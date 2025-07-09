"""
Comprehensive feature engineering sweep with Weights & Biases integration.
"""

import pandas as pd
import numpy as np
import logging
import time
import wandb
from typing import Dict, Tuple, Optional
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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


def run_comprehensive_sweep():
    """Run comprehensive feature engineering sweep with W&B integration."""
    
    print("🚀 INITIALIZING COMPREHENSIVE FEATURE SWEEP...")
    print("⏰ Starting at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("📊 Setting up Weights & Biases integration...")
    
    wandb_run = None
    try:
        # Initialize W&B
        wandb.init(
            project="fraud-detection-autoencoder",
            group="feature-sweep",
            name=f"comprehensive-sweep-50epochs-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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
    
    # Define all strategies to test
    strategies = [
        "baseline_numeric",
        "categorical", 
        "temporal",
        "behavioral",
        "demographics",
        "fraud_flags",
        "rolling",
        "rank_encoding",
        "time_interactions",
        "combined"
    ]
    
    # Strategy descriptions for output
    strategy_descriptions = {
        "baseline_numeric": "Log and ratio features from raw numerics (1 feature)",
        "categorical": "Encoded payment, product, and device columns (3 features)",
        "temporal": "Late-night and burst transaction flags (2 features)",
        "behavioral": "Behavioral ratios per age/account age (2 features)",
        "demographics": "Customer age bucketed into risk bands (1 feature)",
        "fraud_flags": "Rule-based fraud risk indicators (4 features)",
        "rolling": "Rolling mean and std of amount per customer (2 features)",
        "rank_encoding": "Rank-based encodings of amount and account age (2 features)",
        "time_interactions": "Crossed and interaction features using hour (2 features)",
        "combined": "All feature engineering strategies combined (19 features)"
    }
    
    print("\n🚀 STARTING COMPREHENSIVE FEATURE ENGINEERING SWEEP")
    print("="*100)
    print(f"📊 Testing {len(strategies)} strategies with 50 epochs each (early stopping)")
    print(f"⏰ Estimated time: ~40-50 minutes for all strategies")
    if wandb_run:
        print(f"📈 Logging to Weights & Biases: {wandb_run.name}")
    else:
        print(f"📈 Logging to console and file only")
    print("="*100)
    
    log_info("Starting comprehensive feature engineering sweep...")
    log_info(f"Testing {len(strategies)} strategies")
    if wandb_run:
        log_info(f"W&B Run: {wandb_run.name}")
    else:
        log_info("Running without W&B logging")
    
    print(f"\n📋 STRATEGIES TO TEST:")
    for i, strategy in enumerate(strategies, 1):
        print(f"   {i:2d}. {strategy:<20} - {strategy_descriptions[strategy]}")
    
    print(f"\n🚀 Starting execution...")
    print("-"*100)
    
    # Load and clean data once
    print(f"\n🔄 STEP 1: Loading and cleaning data...")
    print(f"📁 Loading data from: data/raw/Fraudulent_E-Commerce_Transaction_Data_2.csv")
    log_info("Loading and cleaning data...")
    
    try:
        config = PipelineConfig.get_baseline_numeric_config()  # Use any config for data loading
        cleaner = DataCleaner(config)
        df_cleaned = cleaner.clean_data(save_output=False)
        
        log_info(f"✅ Data loaded: {len(df_cleaned)} transactions")
        print(f"✅ Data loaded: {len(df_cleaned):,} transactions")
        print(f"📊 Data shape: {df_cleaned.shape}")
        print(f"🔍 Base columns: {list(df_cleaned.columns)}")
        
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
    
    # Test each strategy
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{'='*100}")
        print(f"🎯 STAGE {i}/{len(strategies)}: TESTING {strategy.upper()}")
        print(f"{'='*100}")
        print(f"📝 Description: {strategy_descriptions[strategy]}")
        print(f"⏰ Starting at: {time.strftime('%H:%M:%S')}")
        print(f"📊 Progress: {i}/{len(strategies)} ({i/len(strategies)*100:.1f}%)")
        print(f"{'='*100}")
        
        log_info(f"Testing strategy {i}/{len(strategies)}: {strategy}")
        
        try:
            strategy_start_time = time.time()
            
            # Get configuration for this strategy
            config = PipelineConfig.get_config(strategy)
            
            # Generate features
            print(f"\n🔧 STEP 2: Getting configuration for {strategy}...")
            print(f"✅ Configuration loaded for {strategy}")
            
            print(f"\n🔧 STEP 3: Generating features for {strategy}...")
            print(f"📝 Strategy description: {strategy_descriptions[strategy]}")
            log_info("Generating features...")
            
            feature_engineer = FeatureFactory.create(strategy)
            print(f"✅ Feature engineer created for {strategy}")
            
            df_features = feature_engineer.generate_features(df_cleaned)
            
            feature_count = len(df_features.columns)
            new_features = feature_count - len(df_cleaned.columns)
            log_info(f"Features generated: {feature_count} columns ({new_features} new)")
            print(f"✅ Features generated: {feature_count} columns ({new_features} new)")
            print(f"📊 Feature columns: {list(df_features.columns)}")
            
            # Train model
            print(f"\n🤖 STEP 4: Training autoencoder for {strategy}...")
            print(f"📈 This will run up to 50 epochs with early stopping - watch for progress updates...")
            print(f"⏰ Starting training at: {time.strftime('%H:%M:%S')}")
            log_info("Training autoencoder...")
            
            autoencoder = BaselineAutoencoder(config)
            print(f"✅ Autoencoder model created")
            
            # Log training start to W&B
            if wandb_run:
                wandb.log({
                    f"{strategy}/training_start": True,
                    f"{strategy}/feature_count": feature_count,
                    f"{strategy}/new_features": new_features,
                    f"{strategy}/epochs_configured": config.model.epochs
                })
            
            results_dict = autoencoder.train()
            print(f"✅ Training completed!")
            
            # Extract metrics from training results
            roc_auc = results_dict.get('roc_auc', 0.0)
            f1_score = results_dict.get('f1_score', 0.0)
            precision = results_dict.get('precision', 0.0)
            recall = results_dict.get('recall', 0.0)
            
            # Extract additional training info
            history = results_dict.get('history', None)
            threshold = results_dict.get('threshold', 0.0)
            epochs_trained = len(history.history['loss']) if history else 0
            
            # Calculate time taken
            strategy_time = time.time() - strategy_start_time
            
            # Store results
            results[strategy] = {
                'success': True,
                'roc_auc': roc_auc,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall,
                'time_taken': strategy_time,
                'feature_count': feature_count,
                'new_features': new_features,
                'threshold': threshold,
                'epochs_trained': epochs_trained
            }
            
            print(f"\n🎉 Strategy {strategy} completed successfully!")
            print(f"📊 ROC AUC: {roc_auc:.4f}")
            print(f"📈 F1 Score: {f1_score:.4f}")
            print(f"🎯 Precision: {precision:.4f}")
            print(f"📋 Recall: {recall:.4f}")
            print(f"⏱️  Time taken: {strategy_time:.2f} seconds")
            print(f"📈 Feature count: {feature_count} ({new_features} new)")
            print(f"🎯 Threshold: {threshold:.4f}")
            print(f"🔄 Epochs trained: {epochs_trained}")
            
            log_info(f"Strategy {strategy} completed successfully!")
            log_info(f"ROC AUC: {roc_auc:.4f}")
            log_info(f"Time taken: {strategy_time:.2f} seconds")
            
            # Log detailed results to W&B
            if wandb_run:
                wandb.log({
                    f"{strategy}/roc_auc": roc_auc,
                    f"{strategy}/f1_score": f1_score,
                    f"{strategy}/precision": precision,
                    f"{strategy}/recall": recall,
                    f"{strategy}/time_taken": strategy_time,
                    f"{strategy}/feature_count": feature_count,
                    f"{strategy}/new_features": new_features,
                    f"{strategy}/threshold": threshold,
                    f"{strategy}/epochs_trained": epochs_trained,
                    f"{strategy}/success": True
                })
                
                # Log training history if available
                if history and hasattr(history, 'history'):
                    # Log final training metrics
                    final_loss = history.history['loss'][-1] if 'loss' in history.history else 0
                    final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 0
                    final_mae = history.history['mae'][-1] if 'mae' in history.history else 0
                    final_val_mae = history.history['val_mae'][-1] if 'val_mae' in history.history else 0
                    
                    wandb.log({
                        f"{strategy}/final_loss": final_loss,
                        f"{strategy}/final_val_loss": final_val_loss,
                        f"{strategy}/final_mae": final_mae,
                        f"{strategy}/final_val_mae": final_val_mae,
                        f"{strategy}/training_efficiency": epochs_trained / config.model.epochs
                    })
            
            # Progress update
            elapsed_time = time.time() - start_time
            avg_time_per_strategy = elapsed_time / i
            remaining_strategies = len(strategies) - i
            estimated_remaining = remaining_strategies * avg_time_per_strategy
            
            print(f"\n📊 PROGRESS UPDATE:")
            print(f"   ✅ Completed: {i}/{len(strategies)} strategies ({i/len(strategies)*100:.1f}%)")
            print(f"   ⏱️  Elapsed time: {elapsed_time:.1f}s")
            print(f"   🎯 Est. remaining: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f} minutes)")
            print(f"   📈 Current best: {max([r['roc_auc'] for r in results.values() if r['success']], default=0):.4f}")
            
            # Log progress to W&B
            if wandb_run:
                wandb.log({
                    "progress/completed": i,
                    "progress/total": len(strategies),
                    "progress/percentage": i/len(strategies)*100,
                    "progress/elapsed_time": elapsed_time,
                    "progress/estimated_remaining": estimated_remaining,
                    "progress/current_best_roc": max([r['roc_auc'] for r in results.values() if r['success']], default=0)
                })
            
        except Exception as e:
            error_msg = f"Strategy {strategy} failed: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            results[strategy] = {
                'success': False,
                'roc_auc': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'time_taken': 0.0,
                'feature_count': 0,
                'new_features': 0,
                'error': str(e)
            }
            
            # Log failure to W&B
            if wandb_run:
                wandb.log({
                    f"{strategy}/success": False,
                    f"{strategy}/error": str(e)
                })
            continue
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print final summary
    print_summary(results, strategy_descriptions, total_time)
    
    # Log final summary to W&B
    if wandb_run:
        log_final_summary_to_wandb(results, total_time)
        # Finish W&B run
        wandb.finish()
    else:
        print("✅ Sweep completed without W&B logging")
    
    return results


def print_summary(results: Dict[str, Dict], strategy_descriptions: Dict[str, str], total_time: float):
    """Print summary of feature sweep results."""
    
    print("\n" + "="*120)
    print("🏆 COMPREHENSIVE FEATURE SWEEP RESULTS SUMMARY")
    print("="*120)
    print(f"{'Strategy':<20} {'Status':<10} {'ROC AUC':<10} {'F1':<8} {'Time (s)':<10} {'Features':<10} {'Epochs':<8} {'Threshold':<10} {'Notes':<30}")
    print("-" * 120)
    
    # Sort by ROC AUC (descending)
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v['success']],  # Only successful runs
        key=lambda x: x[1]['roc_auc'],
        reverse=True
    )
    
    best_strategy = None
    best_roc = 0.0
    
    for strategy, result in sorted_results:
        if result['success'] and result['roc_auc'] > best_roc:
            best_strategy = strategy
            best_roc = result['roc_auc']
        
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        notes = strategy_descriptions.get(strategy, "Unknown strategy")
        epochs_trained = result.get('epochs_trained', 0)
        threshold = result.get('threshold', 0.0)
        print(f"{strategy:<20} {status:<10} {result['roc_auc']:<10.4f} {result['f1_score']:<8.4f} {result['time_taken']:<10.2f} {result['feature_count']:<10} {epochs_trained:<8} {threshold:<10.4f} {notes:<30}")
    
    # Print failed strategies
    failed_strategies = [(k, v) for k, v in results.items() if not v['success']]
    for strategy, result in failed_strategies:
        notes = strategy_descriptions.get(strategy, "Unknown strategy")
        print(f"{strategy:<20} {'❌ FAILED':<10} {'0.0000':<10} {'0.0000':<8} {'0.00':<10} {'0':<10} {'0':<8} {'0.0000':<10} {notes:<30}")
    
    print("-" * 120)
    
    if best_strategy:
        print(f"\n🥇 BEST PERFORMING STRATEGY: {best_strategy}")
        print(f"   📊 ROC AUC: {best_roc:.4f}")
        print(f"   📈 F1 Score: {results[best_strategy]['f1_score']:.4f}")
        print(f"   🎯 Precision: {results[best_strategy]['precision']:.4f}")
        print(f"   📋 Recall: {results[best_strategy]['recall']:.4f}")
        
        # Compare with baseline_numeric
        baseline_result = results.get("baseline_numeric")
        if baseline_result and baseline_result['success']:
            baseline_roc = baseline_result['roc_auc']
            if baseline_roc > 0:
                improvement = ((best_roc - baseline_roc) / baseline_roc) * 100
                if best_strategy == "baseline_numeric":
                    print(f"   🎯 Baseline is the best strategy!")
                else:
                    print(f"   📈 Improvement over baseline_numeric: +{improvement:.2f}%")
    
    # Summary statistics
    successful_runs = sum(1 for r in results.values() if r['success'])
    failed_runs = len(results) - successful_runs
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   ⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   🎯 Strategies tested: {len(results)}")
    print(f"   ✅ Successful runs: {successful_runs}")
    print(f"   ❌ Failed runs: {failed_runs}")
    print(f"   📈 Average time per strategy: {total_time/len(results):.1f} seconds")
    
    if successful_runs > 0:
        avg_roc = sum(r['roc_auc'] for r in results.values() if r['success']) / successful_runs
        avg_f1 = sum(r['f1_score'] for r in results.values() if r['success']) / successful_runs
        print(f"   📊 Average ROC AUC: {avg_roc:.4f}")
        print(f"   📈 Average F1 Score: {avg_f1:.4f}")
        print(f"   🎯 Best ROC AUC: {best_roc:.4f}")
        print(f"   📉 Worst ROC AUC: {min(r['roc_auc'] for r in results.values() if r['success']):.4f}")
    
    print(f"\n🎉 Comprehensive feature sweep completed!")
    print("="*120)


def log_final_summary_to_wandb(results: Dict[str, Dict], total_time: float):
    """Log final summary statistics to Weights & Biases."""
    
    successful_runs = sum(1 for r in results.values() if r['success'])
    failed_runs = len(results) - successful_runs
    
    if successful_runs > 0:
        avg_roc = sum(r['roc_auc'] for r in results.values() if r['success']) / successful_runs
        avg_f1 = sum(r['f1_score'] for r in results.values() if r['success']) / successful_runs
        best_roc = max(r['roc_auc'] for r in results.values() if r['success'])
        worst_roc = min(r['roc_auc'] for r in results.values() if r['success'])
        
        wandb.log({
            "summary/total_time": total_time,
            "summary/strategies_tested": len(results),
            "summary/successful_runs": successful_runs,
            "summary/failed_runs": failed_runs,
            "summary/avg_time_per_strategy": total_time/len(results),
            "summary/avg_roc_auc": avg_roc,
            "summary/avg_f1_score": avg_f1,
            "summary/best_roc_auc": best_roc,
            "summary/worst_roc_auc": worst_roc,
            "summary/success_rate": successful_runs/len(results)*100
        })
        
        # Log individual strategy results
        for strategy, result in results.items():
            if result['success']:
                wandb.log({
                    f"final/{strategy}/roc_auc": result['roc_auc'],
                    f"final/{strategy}/f1_score": result['f1_score'],
                    f"final/{strategy}/precision": result['precision'],
                    f"final/{strategy}/recall": result['recall'],
                    f"final/{strategy}/time_taken": result['time_taken'],
                    f"final/{strategy}/feature_count": result['feature_count'],
                    f"final/{strategy}/threshold": result.get('threshold', 0.0),
                    f"final/{strategy}/epochs_trained": result.get('epochs_trained', 0)
                })


if __name__ == "__main__":
    run_comprehensive_sweep() 