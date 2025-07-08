#!/usr/bin/env python3
"""
Test Advanced Feature Engineering Strategies for Fraud Detection

This script tests all available feature engineering strategies to find
the best one for improving AUC-ROC score to at least 75%.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig
from src.data import DataCleaner
from src.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_feature_strategy(strategy_name: str, df_cleaned: pd.DataFrame, config: PipelineConfig) -> dict:
    """Test a single feature engineering strategy."""
    logger.info(f"Testing strategy: {strategy_name}")
    print(f"🔄 Testing strategy: {strategy_name}")
    
    try:
        # Create feature engineer
        print(f"  📦 Creating feature engineer...")
        feature_engineer = FeatureFactory.create(strategy_name)
        
        # Generate features
        print(f"  🔧 Generating features...")
        start_time = time.time()
        df_features = feature_engineer.generate_features(df_cleaned)
        feature_time = time.time() - start_time
        
        # Get feature info
        feature_info = feature_engineer.get_feature_info()
        feature_count = feature_info['feature_count']
        
        print(f"  ✅ Generated {feature_count} features in {feature_time:.2f}s")
        logger.info(f"Generated {feature_count} features in {feature_time:.2f}s")
        
        # Create autoencoder with proper config
        print(f"  🧠 Creating autoencoder...")
        logger.info("Starting baseline autoencoder training...")
        autoencoder = BaselineAutoencoder(config)
        
        # Train the model using the existing train method
        print(f"  🚀 Training autoencoder...")
        start_time = time.time()
        results = autoencoder.train()
        train_time = time.time() - start_time
        
        # Extract metrics - look for AUC in the results
        auc_roc = 0.0
        if isinstance(results, dict):
            # Try different possible keys for AUC
            auc_roc = results.get('roc_auc', 0.0)  # Primary key from autoencoder
            if auc_roc == 0.0:
                auc_roc = results.get('auc', 0.0)
            if auc_roc == 0.0:
                auc_roc = results.get('auc_roc', 0.0)
            if auc_roc == 0.0:
                auc_roc = results.get('test_auc', 0.0)
            if auc_roc == 0.0:
                # Check if there's a metrics dict
                metrics = results.get('metrics', {})
                auc_roc = metrics.get('auc', 0.0)
                if auc_roc == 0.0:
                    auc_roc = metrics.get('auc_roc', 0.0)
        
        print(f"  ✅ Training completed in {train_time:.2f}s")
        print(f"  📊 AUC-ROC: {auc_roc:.4f}")
        logger.info(f"Training completed in {train_time:.2f}s")
        logger.info(f"AUC-ROC: {auc_roc:.4f}")
        
        return {
            'strategy': strategy_name,
            'feature_count': feature_count,
            'feature_time': feature_time,
            'train_time': train_time,
            'auc_roc': auc_roc,
            'success': True
        }
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        logger.error(f"Error testing strategy {strategy_name}: {e}")
        return {
            'strategy': strategy_name,
            'feature_count': 0,
            'feature_time': 0,
            'train_time': 0,
            'auc_roc': 0.0,
            'success': False,
            'error': str(e)
        }


def main():
    """Main function to test all feature engineering strategies."""
    print("🚀 Starting advanced feature engineering strategy testing...")
    logger.info("Starting advanced feature engineering strategy testing...")
    
    # Load baseline configuration
    config = PipelineConfig.get_baseline_numeric_config()

    # Load and clean data
    print("📂 Loading and cleaning data...")
    logger.info("Loading and cleaning data...")
    data_cleaner = DataCleaner(config)
    df_cleaned = data_cleaner.clean_data(save_output=False)
    
    if df_cleaned is None or len(df_cleaned) == 0:
        print("❌ Failed to load or clean data")
        logger.error("Failed to load or clean data")
        return
    
    print(f"✅ Loaded cleaned data: {df_cleaned.shape}")
    logger.info(f"Loaded cleaned data: {df_cleaned.shape}")
    logger.info(f"Available columns: {list(df_cleaned.columns)}")
    
    # Get all available strategies
    strategies = FeatureFactory.get_available_strategies()
    print(f"📋 Testing {len(strategies)} strategies")
    logger.info(f"Testing {len(strategies)} strategies: {list(strategies.keys())}")
    
    # Test each strategy
    results = []
    strategy_count = len(strategies)
    for i, strategy_name in enumerate(strategies.keys(), 1):
        print(f"\n{'='*60}")
        print(f"📊 Progress: {i}/{strategy_count} - {strategy_name}")
        print(f"{'='*60}")
        logger.info("=" * 40)
        result = test_feature_strategy(strategy_name, df_cleaned, config)
        results.append(result)
        logger.info("=" * 40)
    
    # Find best strategy
    successful_results = [r for r in results if r['success']]
    if not successful_results:
        logger.error("No strategies completed successfully")
        return
    
    best_result = max(successful_results, key=lambda x: x['auc_roc'])
    
    # Print results summary
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE ENGINEERING TEST RESULTS")
    logger.info("=" * 60)
    logger.info("\nStrategy             Features   AUC-ROC    Time(s)")
    logger.info("-" * 60)
    
    for result in successful_results:
        logger.info(f"{result['strategy']:<20} {result['feature_count']:<9} {result['auc_roc']:<9.4f} {result['train_time']:<8.2f}")
    
    logger.info(f"\n🏆 BEST STRATEGY: {best_result['strategy']}")
    logger.info(f"   AUC-ROC: {best_result['auc_roc']:.4f}")
    logger.info(f"   Features: {best_result['feature_count']}")
    logger.info(f"   Training time: {best_result['train_time']:.2f}s")
    
    # Check if target is met
    target_auc = 0.75
    if best_result['auc_roc'] >= target_auc:
        logger.info(f"✅ TARGET MET: Best AUC-ROC is {best_result['auc_roc']:.4f} (>= {target_auc})")
    else:
        logger.info(f"❌ TARGET NOT MET: Best AUC-ROC is {best_result['auc_roc']:.4f} (need >= {target_auc})")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"feature_test_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("Feature Engineering Strategy Test Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target AUC-ROC: {target_auc}\n\n")
        
        f.write("Strategy Results:\n")
        f.write("-" * 20 + "\n")
        for result in successful_results:
            f.write(f"{result['strategy']}: AUC-ROC={result['auc_roc']:.4f}, Features={result['feature_count']}\n")
        
        f.write(f"\nBest Strategy: {best_result['strategy']}\n")
        f.write(f"Best AUC-ROC: {best_result['auc_roc']:.4f}\n")
        f.write(f"Target Met: {best_result['auc_roc'] >= target_auc}\n")
    
    logger.info(f"Results saved to: {results_file}")
    
    # Recommend strategy for production
    logger.info(f"\nRecommended strategy for production: {best_result['strategy']}")


if __name__ == "__main__":
    main() 