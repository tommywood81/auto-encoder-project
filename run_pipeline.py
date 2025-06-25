#!/usr/bin/env python3
"""
Baseline pipeline runner for fraud detection.
Simplified pipeline focusing on essential components.
"""

import logging
import sys
import os
from typing import List, Optional

from src.config import PipelineConfig
from src.ingest_data import DataIngestion
from src.data_cleaning import DataCleaner
from src.feature_factory import BaselineFeatureFactory
from src.autoencoder import BaselineAutoencoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class BaselinePipelineRunner:
    """Baseline pipeline runner for fraud detection."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.data_ingestion = DataIngestion(self.config)
        self.data_cleaner = DataCleaner(self.config)
        self.feature_factory = BaselineFeatureFactory(self.config)
        self.autoencoder = BaselineAutoencoder(self.config)
    
    def run_stage(self, stage: str):
        """Run a specific pipeline stage."""
        logger.info(f"Running baseline pipeline stage: {stage}")
        
        if stage == "clean":
            self._run_cleaning()
        elif stage == "engineer":
            self._run_feature_engineering()
        elif stage == "train":
            self._run_training()
        elif stage == "all":
            self._run_full_pipeline()
        else:
            logger.error(f"Unknown stage: {stage}")
            return False
        
        logger.info(f"Pipeline stage '{stage}' completed successfully")
        return True
    
    def _run_cleaning(self):
        """Run data cleaning stage."""
        logger.info("Starting data cleaning...")
        self.data_cleaner.clean_data()
        logger.info("Data cleaning completed")
    
    def _run_feature_engineering(self):
        """Run feature engineering stage."""
        logger.info("Starting baseline feature engineering...")
        self.feature_factory.engineer_features()
        logger.info("Feature engineering completed")
    
    def _run_training(self):
        """Run model training and evaluation stage."""
        logger.info("Starting baseline model training...")
        
        # Train the autoencoder
        results = self.autoencoder.train()
        
        # Log results
        logger.info(f"Training completed!")
        logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
        logger.info(f"Anomaly threshold: {results['threshold']:.6f}")
        
        logger.info("Model training and evaluation completed")
    
    def _run_full_pipeline(self):
        """Run the complete baseline pipeline."""
        logger.info("Starting full baseline pipeline...")
        
        self._run_cleaning()
        self._run_feature_engineering()
        self._run_training()
        
        logger.info("Full baseline pipeline completed successfully")


def main():
    """Main entry point for the baseline pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Fraud Detection Pipeline")
    parser.add_argument(
        "stage",
        choices=["clean", "engineer", "train", "all"],
        help="Pipeline stage to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline runner
        config = PipelineConfig()
        if args.config:
            # Load custom config if provided
            config = PipelineConfig.from_file(args.config)
        
        runner = BaselinePipelineRunner(config)
        
        # Run the specified stage
        success = runner.run_stage(args.stage)
        
        if success:
            logger.info("Baseline pipeline completed successfully")
            sys.exit(0)
        else:
            logger.error("Baseline pipeline failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Baseline pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 