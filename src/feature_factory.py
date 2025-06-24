"""
Feature factory orchestrator for fraud detection.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Any, Optional
import logging

from src.config import PipelineConfig
from src.feature_engineering import (
    TransactionFeatureBuilder,
    IdentityFeatureBuilder,
    InteractionFeatureBuilder,
    StatisticalFeatureBuilder,
    TemporalFeatureBuilder,
    BehavioralDriftFeatureBuilder,
    EntityNoveltyFeatureBuilder,
    handle_infinite_values
)

logger = logging.getLogger(__name__)


class FeatureFactory:
    """Main feature factory that orchestrates all feature builders."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.builders = {
            'transaction': TransactionFeatureBuilder(self.config.features),
            'identity': IdentityFeatureBuilder(self.config.features),
            'interaction': InteractionFeatureBuilder(self.config.features),
            'statistical': StatisticalFeatureBuilder(self.config.features),
            'temporal': TemporalFeatureBuilder(self.config.features),
            'behavioral_drift': BehavioralDriftFeatureBuilder(self.config.features),
            'entity_novelty': EntityNoveltyFeatureBuilder(self.config.features)
        }
        self.feature_info = {}
    
    def load_cleaned_data(self):
        """Load cleaned data from the cleaned directory."""
        logger.info("Loading cleaned data...")
        
        cleaned_file = os.path.join(self.config.data.cleaned_dir, "train_cleaned.csv")
        if not os.path.exists(cleaned_file):
            raise FileNotFoundError(f"Cleaned data not found at {cleaned_file}. Run data cleaning first.")
        
        df = pd.read_csv(cleaned_file)
        logger.info(f"Cleaned data: {df.shape}")
        
        return df
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all features based on configuration."""
        logger.info("Starting feature factory pipeline...")
        
        # Build features in order
        for builder_name, builder in self.builders.items():
            df = builder.build(df)
            self.feature_info[builder_name] = builder.get_feature_info()
        
        # Handle infinite values
        df = handle_infinite_values(df)
        
        logger.info(f"Feature factory completed. Total features: {df.shape[1]}")
        return df
    
    def engineer_features(self, save_output=True):
        """Complete feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline...")
        
        # Load cleaned data
        df = self.load_cleaned_data()
        
        # Build features
        df = self.build_features(df)
        
        logger.info(f"Final engineered shape: {df.shape}")
        
        # Save engineered data if requested
        if save_output:
            self.save_engineered_data(df)
        
        return df
    
    def save_engineered_data(self, df, suffix=""):
        """Save engineered data to the engineered directory."""
        logger.info("Saving engineered data...")
        
        os.makedirs(self.config.data.engineered_dir, exist_ok=True)
        
        # Save engineered data
        output_file = os.path.join(self.config.data.engineered_dir, f"train_features{suffix}.csv")
        df.to_csv(output_file, index=False)
        
        # Convert NumPy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save feature information
        feature_summary = self.get_feature_summary()
        info_file = os.path.join(self.config.data.engineered_dir, f"feature_info{suffix}.json")
        with open(info_file, 'w') as f:
            json.dump(convert_numpy_types(feature_summary), f, indent=2)
        
        logger.info(f"Saved engineered data to {output_file}")
        
        return output_file
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of all built features."""
        summary = {
            'total_features': len(self.feature_info),
            'feature_groups': {},
            'total_engineered_features': 0
        }
        
        for group_name, group_info in self.feature_info.items():
            summary['feature_groups'][group_name] = {
                'count': len(group_info),
                'features': group_info
            }
            summary['total_engineered_features'] += len(group_info)
        
        return summary
    
    def get_feature_list(self) -> List[str]:
        """Get list of all engineered feature names."""
        feature_list = []
        for group_info in self.feature_info.values():
            feature_list.extend(group_info.keys())
        return feature_list 