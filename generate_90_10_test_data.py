#!/usr/bin/env python3
"""
Generate 90/10 Test Data Split
This script creates a new test dataset with 90% training and 10% testing split
for faster dashboard response times.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_split_data_90_10(data_path: str, train_ratio: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data and perform clean 90/10 time-aware split.
    
    Args:
        data_path: Path to the cleaned data file
        train_ratio: Ratio of data to use for training (default: 0.9)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Loading data from: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Ensure data is sorted by transaction date for time-aware split
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df = df.sort_values('transaction_date').reset_index(drop=True)
        logger.info("Data sorted by transaction date for time-aware split")
    
    # Clean 90/10 time-aware split
    total_samples = len(df)
    train_size = int(train_ratio * total_samples)
    
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()
    
    logger.info(f"90/10 time-aware split: train={len(df_train)}, test={len(df_test)}")
    logger.info(f"Train period: transactions 0 to {train_size-1}")
    logger.info(f"Test period: transactions {train_size} to {total_samples-1}")
    
    return df_train, df_test


def generate_90_10_test_data():
    """Generate test data with 90/10 split and save it."""
    
    logger.info("=" * 60)
    logger.info("GENERATING 90/10 TEST DATA SPLIT")
    logger.info("=" * 60)
    
    try:
        # Load cleaned data
        cleaned_data_path = "data/cleaned/creditcard_cleaned.csv"
        if not os.path.exists(cleaned_data_path):
            raise FileNotFoundError(f"Cleaned data not found: {cleaned_data_path}")
        
        # Load and split data (90/10 split)
        df_train, df_test = load_and_split_data_90_10(cleaned_data_path, train_ratio=0.9)
        logger.info(f"Data loaded: {len(df_train)} train, {len(df_test)} test samples")
        
        # Extract target variables
        y_train = df_train['is_fraudulent'].values
        y_test = df_test['is_fraudulent'].values
        
        # Import feature engineering components
        from src.features.feature_engineer import FeatureEngineer
        from src.config_loader import ConfigLoader
        
        # Load configuration
        config_loader = ConfigLoader("configs/final_optimized_config.yaml")
        feature_config = config_loader.get_feature_config()
        
        # Feature engineering
        logger.info("Starting feature engineering...")
        feature_engineer = FeatureEngineer(feature_config)
        df_train_features, df_test_features = feature_engineer.fit_transform_80_20(df_train, df_test)
        logger.info(f"Feature engineering completed: {len(df_test_features.columns)} features")
        
        # Create output directory
        os.makedirs('data/engineered', exist_ok=True)
        
        # Save engineered test features for inference
        test_features_path = 'data/engineered/test_features_90_10.csv'
        df_test_features.to_csv(test_features_path, index=True)
        
        # Save corresponding ground truth labels
        test_labels_path = 'data/engineered/test_labels_90_10.csv'
        df_test_labels = pd.DataFrame({
            'index': df_test.index,
            'is_fraudulent': y_test
        })
        df_test_labels.to_csv(test_labels_path, index=False)
        
        # Print summary statistics
        logger.info("=" * 60)
        logger.info("90/10 SPLIT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Original data size: {len(df_train) + len(df_test)}")
        logger.info(f"Training set: {len(df_train)} samples ({len(df_train)/(len(df_train) + len(df_test))*100:.1f}%)")
        logger.info(f"Test set: {len(df_test)} samples ({len(df_test)/(len(df_train) + len(df_test))*100:.1f}%)")
        logger.info(f"Test features: {len(df_test_features.columns)}")
        logger.info(f"Test fraud rate: {y_test.mean():.4f}")
        
        logger.info(f"✓ Saved engineered test features to: {test_features_path}")
        logger.info(f"✓ Saved test labels to: {test_labels_path}")
        
        # Update inference config to use new test data
        update_inference_config(test_features_path)
        
        logger.info("=" * 60)
        logger.info("90/10 TEST DATA GENERATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error generating 90/10 test data: {e}")
        raise


def update_inference_config(test_features_path: str):
    """Update inference config to use the new 90/10 test data."""
    
    import yaml
    
    config_path = "configs/inference_config.yaml"
    
    # Load current config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Update test data path
    config['inference']['engineered_test_data_path'] = test_features_path
    
    # Save updated config
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    
    logger.info(f"✓ Updated inference config to use: {test_features_path}")


if __name__ == "__main__":
    generate_90_10_test_data() 