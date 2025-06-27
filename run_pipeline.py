#!/usr/bin/env python3
"""
Fraud Detection Pipeline Runner.
Supports config-driven feature strategies via CLI arguments.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig
from src.data import DataCleaner
from src.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(strategy: str):
    """Run the complete fraud detection pipeline with given strategy."""
    logger.info(f"Starting fraud detection pipeline with strategy: {strategy}")
    
    # Load configuration
    config = PipelineConfig.get_config(strategy)
    logger.info(f"Configuration loaded: {config.name}")
    logger.info(f"Feature strategy: {config.feature_strategy}")
    
    # Step 1: Data Cleaning
    logger.info("Step 1: Data Cleaning")
    cleaner = DataCleaner(config)
    df_cleaned = cleaner.clean_data(save_output=True)
    logger.info(f"Data cleaning completed. Shape: {df_cleaned.shape}")
    
    # Step 2: Feature Engineering
    logger.info("Step 2: Feature Engineering")
    feature_engineer = FeatureFactory.create(config.feature_strategy)
    df_features = feature_engineer.generate_features(df_cleaned)
    
    # Log feature information
    feature_info = feature_engineer.get_feature_info()
    logger.info(f"Feature strategy: {feature_info['strategy']}")
    logger.info(f"Feature count: {feature_info['feature_count']}")
    logger.info(f"Features: {list(feature_info['features'].keys())}")
    
    # Save engineered features
    os.makedirs(config.data.engineered_dir, exist_ok=True)
    output_file = os.path.join(config.data.engineered_dir, f"{config.feature_strategy}_features.csv")
    df_features.to_csv(output_file, index=False)
    logger.info(f"Features saved to: {output_file}")
    
    # Step 3: Model Training
    logger.info("Step 3: Model Training")
    autoencoder = BaselineAutoencoder(config)
    results = autoencoder.train()
    
    # Log results
    logger.info("Pipeline completed successfully!")
    logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
    logger.info(f"Anomaly threshold: {results['threshold']:.6f}")
    
    # Log configuration summary
    config_summary = config.to_dict()
    logger.info("Configuration summary:")
    for key, value in config_summary.items():
        logger.info(f"  {key}: {value}")
    
    return results


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="baseline",
        choices=["baseline", "temporal", "behavioural", "account_risk", "demographic_risk", "combined"],
        help="Feature strategy to use (default: baseline)"
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List available strategies"
    )
    
    args = parser.parse_args()
    
    # List available strategies
    if args.list_strategies:
        print("Available strategies:")
        print("  - baseline: Basic transaction features only (9 features)")
        print("  - temporal: Basic features + temporal patterns (10 features)")
        print("  - behavioural: Core features + amount per item (10 features)")
        print("  - account_risk: Core features + account age risk (newer accounts = higher fraud risk) (10 features)")
        print("  - demographic_risk: Core features + customer age risk scores (10 features)")
        print("  - combined: All unique features from all strategies (no duplicates)")
        return
    
    # Run pipeline
    try:
        results = run_pipeline(args.strategy)
        logger.info("Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 