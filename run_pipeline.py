#!/usr/bin/env python3
"""
Production pipeline runner for fraud detection.
"""

import logging
import sys
import os
from typing import List, Optional

from src.config import PipelineConfig
from src.data_loader import DataLoader
from src.data_cleaning import DataCleaner
from src.feature_factory import FeatureFactory
from src.data_preprocessor import DataPreprocessor
from src.autoencoder import Autoencoder
from src.evaluator import Evaluator

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


class PipelineRunner:
    """Main pipeline runner for fraud detection."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.data_loader = DataLoader(self.config)
        self.data_cleaner = DataCleaner(self.config)
        self.feature_factory = FeatureFactory(self.config)
        self.data_preprocessor = DataPreprocessor(self.config)
        self.autoencoder = Autoencoder(self.config)
        self.evaluator = Evaluator(self.config)
    
    def run_stage(self, stage: str):
        """Run a specific pipeline stage."""
        logger.info(f"Running pipeline stage: {stage}")
        
        if stage == "clean":
            self._run_cleaning()
        elif stage == "engineer":
            self._run_feature_engineering()
        elif stage == "process":
            self._run_preprocessing()
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
        logger.info("Starting feature engineering...")
        self.feature_factory.engineer_features()
        logger.info("Feature engineering completed")
    
    def _run_preprocessing(self):
        """Run data preprocessing stage."""
        logger.info("Starting data preprocessing...")
        self.data_preprocessor.preprocess_data()
        logger.info("Data preprocessing completed")
    
    def _run_training(self):
        """Run model training and evaluation stage."""
        logger.info("Starting model training...")
        
        # Train the autoencoder
        self.autoencoder.train()
        
        # Evaluate the model
        self.evaluator.evaluate()
        
        logger.info("Model training and evaluation completed")
    
    def _run_full_pipeline(self):
        """Run the complete pipeline."""
        logger.info("Starting full pipeline...")
        
        self._run_cleaning()
        self._run_feature_engineering()
        self._run_preprocessing()
        self._run_training()
        
        logger.info("Full pipeline completed successfully")


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument(
        "stage",
        choices=["clean", "engineer", "process", "train", "all"],
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
        
        runner = PipelineRunner(config)
        
        # Run the specified stage
        success = runner.run_stage(args.stage)
        
        if success:
            logger.info("Pipeline completed successfully")
            sys.exit(0)
        else:
            logger.error("Pipeline failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 