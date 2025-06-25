"""
Baseline feature factory for E-commerce Fraud Detection.
Simplified for baseline model with essential features only.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, List, Any

from src.config import PipelineConfig
from src.feature_engineering import BaselineFeatureEngineer, handle_infinite_values

logger = logging.getLogger(__name__)


class BaselineFeatureFactory:
    """Baseline feature factory for essential fraud detection features."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.feature_engineer = BaselineFeatureEngineer()
        self.feature_info = {}
    
    def load_cleaned_data(self):
        """Load cleaned data from the cleaned directory."""
        logger.info("Loading cleaned data...")
        
        cleaned_file = os.path.join(self.config.data.cleaned_dir, "ecommerce_cleaned.csv")
        if not os.path.exists(cleaned_file):
            raise FileNotFoundError(f"Cleaned data not found at {cleaned_file}. Run data cleaning first.")
        
        df = pd.read_csv(cleaned_file)
        logger.info(f"Cleaned data: {df.shape}")
        
        return df
    
    def engineer_features(self, save_output=True):
        """Complete baseline feature engineering pipeline."""
        logger.info("Starting baseline feature engineering...")
        
        # Load cleaned data
        df = self.load_cleaned_data()
        
        # Engineer baseline features
        df = self.feature_engineer.engineer_features(df)
        
        # Handle infinite values
        df = handle_infinite_values(df)
        
        # Get feature information
        self.feature_info = self.feature_engineer.get_feature_info()
        
        logger.info(f"Baseline features engineered. Shape: {df.shape}")
        
        # Save engineered data if requested
        if save_output:
            self.save_engineered_data(df)
        
        return df
    
    def save_engineered_data(self, df, suffix=""):
        """Save engineered data to the engineered directory."""
        logger.info("Saving engineered data...")
        
        os.makedirs(self.config.data.engineered_dir, exist_ok=True)
        
        # Save engineered data
        output_file = os.path.join(self.config.data.engineered_dir, f"baseline_features{suffix}.csv")
        df.to_csv(output_file, index=False)
        
        # Save feature information
        info_file = os.path.join(self.config.data.engineered_dir, f"baseline_feature_info{suffix}.json")
        with open(info_file, 'w') as f:
            json.dump(self.feature_info, f, indent=2)
        
        logger.info(f"Saved baseline features to {output_file}")
        
        return output_file
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of baseline features."""
        return {
            'total_features': len(self.feature_info),
            'feature_info': self.feature_info,
            'feature_categories': {
                'temporal': ['hour', 'day_of_week', 'is_weekend', 'is_night', 'is_high_risk_hour'],
                'entity_frequency': ['customer_frequency', 'customer_novelty'],
                'behavioral': ['amount_log', 'amount_squared', 'amount_category']
            }
        }
    
    def get_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get only numeric features for baseline model."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].copy() 