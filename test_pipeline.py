#!/usr/bin/env python3
"""
Test script for the new data pipeline structure.
This script tests each stage of the pipeline and saves data to the appropriate directories.
"""

import os
import sys
import logging
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.data_loader import FraudDataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_pipeline():
    """Test the complete data pipeline."""
    logger.info("🧪 Testing data pipeline...")
    
    try:
        # Step 1: Data Cleaning
        logger.info("=" * 50)
        logger.info("STEP 1: DATA CLEANING")
        logger.info("=" * 50)
        
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_data(save_output=True)
        
        logger.info(f"✅ Data cleaning completed. Shape: {cleaned_df.shape}")
        
        # Step 2: Feature Engineering
        logger.info("=" * 50)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("=" * 50)
        
        engineer = FeatureEngineer()
        engineered_df = engineer.engineer_features(save_output=True)
        
        logger.info(f"✅ Feature engineering completed. Shape: {engineered_df.shape}")
        
        # Step 3: Data Loading and Processing
        logger.info("=" * 50)
        logger.info("STEP 3: DATA LOADING AND PROCESSING")
        logger.info("=" * 50)
        
        data_loader = FraudDataLoader()
        data_dict = data_loader.load_processed_data()
        
        logger.info(f"✅ Data loading completed.")
        logger.info(f"  X_train: {data_dict['X_train'].shape}")
        logger.info(f"  X_test: {data_dict['X_test'].shape}")
        logger.info(f"  X_train_ae: {data_dict['X_train_ae'].shape}")
        logger.info(f"  Features: {len(data_dict['feature_names'])}")
        
        # Step 4: Verify all directories have data
        logger.info("=" * 50)
        logger.info("STEP 4: VERIFYING PIPELINE OUTPUTS")
        logger.info("=" * 50)
        
        verify_pipeline_outputs()
        
        logger.info("🎉 All pipeline tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {str(e)}")
        return False


def verify_pipeline_outputs():
    """Verify that all pipeline directories contain the expected files."""
    from src.config import DATA_CLEANED, DATA_ENGINEERED, DATA_PROCESSED
    
    # Check cleaned directory
    cleaned_files = [
        "train_cleaned.csv",
        "cleaning_stats.json",
        "label_encoders.json"
    ]
    
    for file in cleaned_files:
        path = os.path.join(DATA_CLEANED, file)
        if os.path.exists(path):
            logger.info(f"✅ {path}")
        else:
            logger.warning(f"⚠️ Missing: {path}")
    
    # Check engineered directory
    engineered_files = [
        "train_features.csv",
        "feature_info.json"
    ]
    
    for file in engineered_files:
        path = os.path.join(DATA_ENGINEERED, file)
        if os.path.exists(path):
            logger.info(f"✅ {path}")
        else:
            logger.warning(f"⚠️ Missing: {path}")
    
    # Check processed directory
    processed_files = [
        "X_train.npy",
        "X_test.npy",
        "y_train.npy",
        "y_test.npy",
        "scaler.pkl",
        "label_encoders.pkl",
        "feature_names.txt"
    ]
    
    for file in processed_files:
        path = os.path.join(DATA_PROCESSED, file)
        if os.path.exists(path):
            logger.info(f"✅ {path}")
        else:
            logger.warning(f"⚠️ Missing: {path}")


def list_data_structure():
    """List the current data directory structure."""
    logger.info("📁 Current data directory structure:")
    
    def list_dir_contents(path, indent=0):
        if os.path.exists(path):
            items = os.listdir(path)
            for item in sorted(items):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    logger.info("  " * indent + f"📁 {item}/")
                    list_dir_contents(item_path, indent + 1)
                else:
                    size = os.path.getsize(item_path)
                    size_str = f"({size:,} bytes)" if size < 1024*1024 else f"({size/1024/1024:.1f} MB)"
                    logger.info("  " * indent + f"📄 {item} {size_str}")
        else:
            logger.info("  " * indent + "❌ Directory does not exist")
    
    list_dir_contents("data")


if __name__ == "__main__":
    logger.info("🚀 Starting data pipeline test...")
    
    # List current structure
    list_data_structure()
    
    # Run pipeline test
    success = test_data_pipeline()
    
    # List final structure
    logger.info("\n📁 Final data directory structure:")
    list_data_structure()
    
    if success:
        logger.info("🎉 Pipeline test completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Pipeline test failed!")
        sys.exit(1) 