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

from src.config_loader import ConfigLoader
from src.data.data_cleaning import DataCleaner
from src.feature_factory.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder
from src.evaluation.evaluator import FraudEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete fraud detection pipeline with given config."""
    logger.info(f"Starting fraud detection pipeline with config: {config}")
    # Data cleaning
    logger.info("Step 1: Data cleaning...")
    cleaner = DataCleaner(config)
    df_cleaned = cleaner.clean_data(save_output=True)
    logger.info(f"Data cleaned: {len(df_cleaned)} transactions")
    # Feature engineering
    logger.info("Step 2: Feature engineering...")
    feature_engineer = FeatureFactory.create(config['features']['strategy'])
    df_features = feature_engineer.generate_features(df_cleaned)
    logger.info(f"Features generated: {len(df_features.columns)} features")
    # Model training
    logger.info("Step 3: Model training...")
    autoencoder = BaselineAutoencoder(config)
    results = autoencoder.train()
    logger.info("Model training completed")
    # Model evaluation
    logger.info("Step 4: Model evaluation...")
    roc_auc = results.get('roc_auc', 0.0)
    logger.info(f"Model evaluation completed - ROC AUC: {roc_auc:.4f}")
    return {
        'roc_auc': roc_auc,
        'config': config,
        'results': results
    }

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/final_optimized_config.yaml",
        help="Path to the config YAML file to use for the pipeline"
    )
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print(f"Config file {args.config} does not exist. Please run the sweep first or provide a valid config.")
        sys.exit(1)
    loader = ConfigLoader(config_dir=os.path.dirname(args.config) or '.')
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    config = loader.load_config(config_name)
    try:
        results = run_pipeline(config)
        print(f"\nPipeline completed successfully!")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 