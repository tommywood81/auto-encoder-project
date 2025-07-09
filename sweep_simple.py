"""
Simple comprehensive feature engineering sweep with immediate logging.
"""

import pandas as pd
import numpy as np
import logging
import time
import sys
import os
from datetime import datetime
from typing import Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig
from src.data.data_cleaning import DataCleaner
from src.feature_factory.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sweep_simple.log')
    ],
    force=True
)

logger = logging.getLogger(__name__)


def run_simple_sweep():
    """Run simple comprehensive feature engineering sweep."""
    
    print("🚀 STARTING SIMPLE COMPREHENSIVE FEATURE SWEEP")
    print("="*80)
    print(f"⏰ Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
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
    
    print(f"📊 Testing {len(strategies)} strategies with 25 epochs each")
    print(f"⏰ Estimated time: ~25-30 minutes for all strategies")
    print("="*80)
    
    # Load and clean data once
    print(f"\n🔄 STEP 1: Loading and cleaning data...")
    print(f"📁 Loading data from: data/raw/Fraudulent_E-Commerce_Transaction_Data_2.csv")
    
    try:
        config = PipelineConfig.get_baseline_numeric_config()
        cleaner = DataCleaner(config)
        df_cleaned = cleaner.clean_data(save_output=False)
        
        print(f"✅ Data loaded: {len(df_cleaned):,} transactions")
        print(f"📊 Data shape: {df_cleaned.shape}")
        print(f"🔍 Base columns: {list(df_cleaned.columns)}")
        
    except Exception as e:
        print(f"❌ ERROR loading data: {str(e)}")
        return {}
    
    # Store results
    results = {}
    start_time = time.time()
    
    # Test each strategy
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{'='*80}")
        print(f"🎯 STAGE {i}/{len(strategies)}: TESTING {strategy.upper()}")
        print(f"{'='*80}")
        print(f"⏰ Starting at: {time.strftime('%H:%M:%S')}")
        print(f"📊 Progress: {i}/{len(strategies)} ({i/len(strategies)*100:.1f}%)")
        print(f"{'='*80}")
        
        try:
            strategy_start_time = time.time()
            
            # Get configuration for this strategy
            print(f"🔧 Getting configuration for {strategy}...")
            config = PipelineConfig.get_config(strategy)
            print(f"✅ Configuration loaded")
            
            # Generate features
            print(f"🔧 Generating features for {strategy}...")
            feature_engineer = FeatureFactory.create(strategy)
            df_features = feature_engineer.generate_features(df_cleaned)
            
            feature_count = len(df_features.columns)
            new_features = feature_count - len(df_cleaned.columns)
            print(f"✅ Features generated: {feature_count} columns ({new_features} new)")
            
            # Train model
            print(f"🤖 Training autoencoder for {strategy}...")
            print(f"📈 This will run 25 epochs - watch for progress updates...")
            
            autoencoder = BaselineAutoencoder(config)
            results_dict = autoencoder.train()
            print(f"✅ Training completed!")
            
            # Extract metrics from training results
            roc_auc = results_dict.get('roc_auc', 0.0)
            f1_score = results_dict.get('f1_score', 0.0)
            
            # Calculate time taken
            strategy_time = time.time() - strategy_start_time
            
            # Store results
            results[strategy] = {
                'success': True,
                'roc_auc': roc_auc,
                'f1_score': f1_score,
                'time_taken': strategy_time,
                'feature_count': feature_count,
                'new_features': new_features
            }
            
            print(f"\n🎉 Strategy {strategy} completed successfully!")
            print(f"📊 ROC AUC: {roc_auc:.4f}")
            print(f"📈 F1 Score: {f1_score:.4f}")
            print(f"⏱️  Time taken: {strategy_time:.2f} seconds")
            print(f"📈 Feature count: {feature_count} ({new_features} new)")
            
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
            
        except Exception as e:
            error_msg = f"Strategy {strategy} failed: {str(e)}"
            print(f"❌ {error_msg}")
            results[strategy] = {
                'success': False,
                'roc_auc': 0.0,
                'f1_score': 0.0,
                'time_taken': 0.0,
                'feature_count': 0,
                'new_features': 0,
                'error': str(e)
            }
            continue
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print final summary
    print_summary(results, total_time)
    
    return results


def print_summary(results: Dict[str, Dict], total_time: float):
    """Print summary of feature sweep results."""
    
    print("\n" + "="*80)
    print("🏆 COMPREHENSIVE FEATURE SWEEP RESULTS SUMMARY")
    print("="*80)
    print(f"{'Strategy':<20} {'Status':<10} {'ROC AUC':<10} {'F1':<8} {'Time (s)':<10} {'Features':<10}")
    print("-" * 80)
    
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
        print(f"{strategy:<20} {status:<10} {result['roc_auc']:<10.4f} {result['f1_score']:<8.4f} {result['time_taken']:<10.2f} {result['feature_count']:<10}")
    
    # Print failed strategies
    failed_strategies = [(k, v) for k, v in results.items() if not v['success']]
    for strategy, result in failed_strategies:
        print(f"{strategy:<20} {'❌ FAILED':<10} {'0.0000':<10} {'0.0000':<8} {'0.00':<10} {'0':<10}")
    
    print("-" * 80)
    
    if best_strategy:
        print(f"\n🥇 BEST PERFORMING STRATEGY: {best_strategy}")
        print(f"   📊 ROC AUC: {best_roc:.4f}")
        print(f"   📈 F1 Score: {results[best_strategy]['f1_score']:.4f}")
        
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
    print("="*80)


if __name__ == "__main__":
    run_simple_sweep() 