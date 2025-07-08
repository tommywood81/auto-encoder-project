#!/usr/bin/env python3
"""
Main pipeline for fraud detection using autoencoders.
"""

import argparse
import logging
import sys
import os
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig
from src.data.data_cleaning import DataCleaner
from src.feature_factory.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder
from src.evaluation.evaluator import FraudEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline(strategy: str) -> Dict[str, Any]:
    """Run the complete fraud detection pipeline with given strategy."""
    
    logger.info(f"Starting fraud detection pipeline with strategy: {strategy}")
    
    # Load configuration
    config = PipelineConfig.get_config(strategy)
    logger.info(f"Configuration loaded: {config.name}")
    
    # Data cleaning
    logger.info("Step 1: Data cleaning...")
    cleaner = DataCleaner(config)
    df_cleaned = cleaner.clean_data(save_output=True)
    logger.info(f"Data cleaned: {len(df_cleaned)} transactions")
    
    # Feature engineering
    logger.info("Step 2: Feature engineering...")
    feature_engineer = FeatureFactory.create(config.feature_strategy)
    df_features = feature_engineer.generate_features(df_cleaned)
    logger.info(f"Features generated: {len(df_features.columns)} features")
    
    # Model training
    logger.info("Step 3: Model training...")
    autoencoder = BaselineAutoencoder(config)
    results = autoencoder.train()
    logger.info("Model training completed")
    
    # Model evaluation
    logger.info("Step 4: Model evaluation...")
    # For now, we'll use a simple approach since the evaluator expects different inputs
    # We'll extract ROC AUC from the training results
    roc_auc = results.get('roc_auc', 0.0)
    logger.info(f"Model evaluation completed - ROC AUC: {roc_auc:.4f}")
    
    return {
        'strategy': strategy,
        'roc_auc': roc_auc,
        'config': config,
        'results': results
    }


def list_strategies():
    """List all available feature engineering strategies."""
    strategies = FeatureFactory.get_available_strategies()
    
    print("\nAvailable Feature Engineering Strategies:")
    print("=" * 60)
    
    for strategy, description in strategies.items():
        print(f"  - {strategy}: {description}")
    
    print("\nUsage:")
    print("  python run_pipeline.py --strategy <strategy_name>")
    print("\nExample:")
    print("  python run_pipeline.py --strategy combined")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument(
        "--strategy",
        type=str,
        default="baseline_numeric",
        choices=["baseline_numeric", "categorical", "temporal", "behavioral", "demographics", "fraud_flags", "rolling", "rank_encoding", "time_interactions", "combined"],
        help="Feature engineering strategy to use"
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List all available strategies"
    )
    
    args = parser.parse_args()
    
    if args.list_strategies:
        list_strategies()
        return
    
    try:
        results = run_pipeline(args.strategy)
        print(f"\nPipeline completed successfully!")
        print(f"Strategy: {results['strategy']}")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 