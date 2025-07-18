#!/usr/bin/env python3
"""
Feature engineering strategy sweep for fraud detection.
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Tuple, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig
from src.data.data_cleaning import DataCleaner
from src.feature_factory.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder
from src.evaluation.evaluator import FraudEvaluator

# Configure logging with more detailed output
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('feature_sweep.log')
    ]
)
logger = logging.getLogger(__name__)


def run_feature_sweep():
    """Run feature engineering strategy sweep."""
    
    print("\n" + "="*100)
    print("FEATURE SWEEP RESULTS SUMMARY")
    print("="*100)
    print("This will test all feature strategies with 25 epochs each")
    print("Estimated time: ~35-45 minutes for all strategies")
    print("="*100)
    
    # Define strategies to test
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
        "baseline_numeric": "Log and ratio features from raw numerics (2 features)",
        "categorical": "Encoded payment, product, and device columns (3 features)",
        "temporal": "Late-night and burst transaction flags (2 features)",
        "behavioral": "Behavioral ratios per age/account age (2 features)",
        "demographics": "Customer age bucketed into risk bands (1 feature)",
        "fraud_flags": "Rule-based fraud risk indicators (4 features)",
        "rolling": "Rolling mean and std of amount per customer (2 features)",
        "rank_encoding": "Rank-based encodings of amount and account age (2 features)",
        "time_interactions": "Crossed and interaction features using hour (2 features)",
        "combined": "All feature engineering strategies combined (20+ features)"
    }
    
    logger.info("Starting feature engineering strategy sweep...")
    logger.info(f"Testing {len(strategies)} strategies")
    print(f"\n📋 STRATEGIES TO TEST:")
    for i, strategy in enumerate(strategies, 1):
        print(f"   {i:2d}. {strategy:<20} - {strategy_descriptions[strategy]}")
    
    # Load and clean data once
    print(f"\n🔄 STEP 1: Loading and cleaning data...")
    logger.info("Loading and cleaning data...")
    config = PipelineConfig.get_baseline_numeric_config()  # Use any config for data loading
    cleaner = DataCleaner(config)
    df_cleaned = cleaner.clean_data(save_output=False)
    
    logger.info(f"Data loaded: {len(df_cleaned)} transactions")
    print(f"Data loaded: {len(df_cleaned):,} transactions")
    
    # Store results
    results = {}
    start_time = time.time()
    
    # Test each strategy
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{'='*100}")
        print(f"STAGE {i}/{len(strategies)}: TESTING {strategy.upper()}")
        print(f"{'='*100}")
        print(f"📝 Description: {strategy_descriptions[strategy]}")
        print(f"⏰ Starting at: {time.strftime('%H:%M:%S')}")
        print(f"{'='*100}")
        
        logger.info(f"Testing strategy {i}/{len(strategies)}: {strategy}")
        
        try:
            strategy_start_time = time.time()
            
            # Get configuration for this strategy
            config = PipelineConfig.get_config(strategy)
            
            # Generate features
            print(f"Generating features for {strategy}...")
            logger.info("Generating features...")
            feature_engineer = FeatureFactory.create(strategy)
            df_features = feature_engineer.generate_features(df_cleaned)
            
            feature_count = len(df_features.columns)
            logger.info(f"Features generated: {feature_count} columns")
            print(f"Features generated: {feature_count} columns")
            print(f"Feature columns: {list(df_features.columns)}")
            
            # Train model
            print(f"🤖 Training autoencoder for {strategy}...")
            print(f"This will run 25 epochs - watch for progress updates...")
            logger.info("Training autoencoder...")
            autoencoder = BaselineAutoencoder(config)
            results_dict = autoencoder.train()
            
            # Extract ROC AUC from training results
            roc_auc = results_dict.get('roc_auc', 0.0)
            
            # Calculate time taken
            strategy_time = time.time() - strategy_start_time
            
            # Store results
            results[strategy] = (True, roc_auc, strategy_time)
            
            print(f"\nStrategy {strategy} completed successfully!")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Time taken: {strategy_time:.2f} seconds")
            print(f"Feature count: {feature_count}")
            
            logger.info(f"Strategy {strategy} completed successfully!")
            logger.info(f"ROC AUC: {roc_auc:.4f}")
            logger.info(f"Time taken: {strategy_time:.2f} seconds")
            
            # Progress update
            elapsed_time = time.time() - start_time
            avg_time_per_strategy = elapsed_time / i
            remaining_strategies = len(strategies) - i
            estimated_remaining = remaining_strategies * avg_time_per_strategy
            
            print(f"\nPROGRESS UPDATE:")
            print(f"   Completed: {i}/{len(strategies)} strategies")
            print(f"   Elapsed time: {elapsed_time:.1f}s")
            print(f"   Est. remaining: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f} minutes)")
            print(f"   Current best: {max([r[1] for r in results.values() if r[0]], default=0):.4f}")
            
        except Exception as e:
            error_msg = f"Strategy {strategy} failed: {str(e)}"
            logger.error(error_msg)
            print(f"{error_msg}")
            results[strategy] = (False, 0.0, 0.0)
            continue
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print final summary
    print_summary(results, strategy_descriptions, total_time)
    
    return results


def print_summary(results: Dict[str, Tuple[bool, float, float]], 
                 strategy_descriptions: Dict[str, str],
                 total_time: float):
    """Print summary of feature sweep results."""
    
    print("\n" + "="*100)
    print("FEATURE SWEEP RESULTS SUMMARY")
    print("="*100)
    print(f"{'Strategy':<20} {'Status':<10} {'ROC AUC':<10} {'Time (s)':<10} {'Features':<10} {'Notes':<30}")
    print("-" * 100)
    
    # Sort by ROC AUC (descending)
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v[0]],  # Only successful runs
        key=lambda x: x[1][1],
        reverse=True
    )
    
    best_strategy = None
    best_roc = 0.0
    
    for strategy, (success, roc_auc, time_taken) in sorted_results:
        if success and roc_auc > best_roc:
            best_strategy = strategy
            best_roc = roc_auc
        
        status = "SUCCESS" if success else "FAILED"
        notes = strategy_descriptions.get(strategy, "Unknown strategy")
        print(f"{strategy:<20} {status:<10} {roc_auc:<10.4f} {time_taken:<10.2f} {'N/A':<10} {notes:<30}")
    
    # Print failed strategies
    failed_strategies = [(k, v) for k, v in results.items() if not v[0]]
    for strategy, (success, roc_auc, time_taken) in failed_strategies:
        notes = strategy_descriptions.get(strategy, "Unknown strategy")
        print(f"{strategy:<20} {'FAILED':<10} {'0.0000':<10} {'0.00':<10} {'N/A':<10} {notes:<30}")
    
    print("-" * 100)
    
    if best_strategy:
        print(f"\nBEST PERFORMING STRATEGY: {best_strategy}")
        print(f"   ROC AUC: {best_roc:.4f}")
        
        # Compare with baseline
        baseline_result = results.get("baseline_numeric")
        if baseline_result and baseline_result[0]:
            baseline_roc = baseline_result[1]
            if baseline_roc > 0:
                improvement = ((best_roc - baseline_roc) / baseline_roc) * 100
                if best_strategy == "baseline_numeric":
                    print(f"   Baseline is the best strategy!")
                else:
                    print(f"   Improvement over baseline: +{improvement:.2f}%")
            else:
                print(f"   Performance vs baseline: {improvement:.2f}%")
        else:
            print(f"   Baseline ROC AUC is zero - cannot calculate improvement")
    else:
        print(f"   Could not compare with baseline (baseline failed)")
    
    # Summary statistics
    successful_runs = sum(1 for r in results.values() if r[0])
    failed_runs = len(results) - successful_runs
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"   Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Strategies tested: {len(results)}")
    print(f"   Successful runs: {successful_runs}")
    print(f"   Failed runs: {failed_runs}")
    print(f"   Average time per strategy: {total_time/len(results):.1f} seconds")
    
    if successful_runs > 0:
        avg_roc = sum(r[1] for r in results.values() if r[0]) / successful_runs
        print(f"   Average ROC AUC: {avg_roc:.4f}")
        print(f"   Best ROC AUC: {best_roc:.4f}")
        print(f"   Worst ROC AUC: {min(r[1] for r in results.values() if r[0]):.4f}")
    
    print(f"\nFeature sweep completed!")
    print("="*100)


if __name__ == "__main__":
    run_feature_sweep() 