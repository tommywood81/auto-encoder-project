"""
Data preprocessing for model training and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from typing import List

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessor for model training and evaluation."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.scaler = None
    
    def preprocess_for_modeling(self, df: pd.DataFrame):
        """Preprocess engineered data for autoencoder training."""
        logger.info("Preprocessing and splitting data...")
        
        # Separate features and target
        X = df.drop(columns=["isFraud"])
        y = df["isFraud"]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.data.test_size, 
            stratify=y, random_state=self.config.data.random_state
        )

        # For autoencoder, we only train on non-fraudulent data
        X_train_ae = X_train[y_train == 0]
        
        logger.info(f"Training samples (non-fraudulent): {X_train_ae.shape[0]}")
        logger.info(f"Test samples: {X_test.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        
        return X_train_ae, X_test, y_train, y_test, self.scaler
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get feature names from dataframe."""
        return df.drop(columns=["isFraud"]).columns.tolist()


# Legacy function for backward compatibility
def preprocess_data(df: pd.DataFrame):
    """Legacy function for backward compatibility."""
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess_for_modeling(df) 