"""
Feature engineering for E-commerce Fraud Detection baseline.
Focus on essential features: temporal patterns and entity frequency.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class BaselineFeatureEngineer:
    """Baseline feature engineer focusing on essential features only."""
    
    def __init__(self):
        self.feature_info = {}
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer baseline features for fraud detection."""
        logger.info("Engineering baseline features...")
        
        df_engineered = df.copy()
        
        # 1. Entity frequency features (customer behavior patterns)
        df_engineered = self._create_entity_frequency_features(df_engineered)
        
        # 2. Advanced behavioral features (rolling averages and patterns)
        df_engineered = self._create_advanced_behavioral_features(df_engineered)
        
        logger.info(f"Engineered {len(self.feature_info)} baseline features")
        return df_engineered
    
    def _create_entity_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create entity frequency features for customer behavior."""
        logger.info("Creating entity frequency features...")
        
        customer_col = None
        if 'customer_location' in df.columns:
            customer_col = 'customer_location'
        elif 'customer_location_freq' in df.columns:
            customer_col = 'customer_location_freq'
            
        if customer_col:
            if customer_col == 'customer_location':
                customer_freq = df['customer_location'].value_counts(normalize=True)
                df['customer_frequency'] = df['customer_location'].map(customer_freq)
                df['customer_novelty'] = 1 - df['customer_frequency']
                customer_counts = df['customer_location'].value_counts()
                df['customer_transaction_count'] = df['customer_location'].map(customer_counts)
                df = df.drop(columns=['customer_location'])
            else:
                df['customer_frequency'] = df['customer_location_freq']
                df['customer_novelty'] = 1 - df['customer_frequency']
                df['customer_transaction_count'] = (1 / df['customer_location_freq']).round().astype(int)
                
            self.feature_info.update({
                'customer_frequency': 'Frequency of customer in dataset',
                'customer_novelty': 'Novelty score (1 - frequency)',
                'customer_transaction_count': 'Total transactions per customer'
            })
        return df
    
    def _create_advanced_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced behavioral features."""
        logger.info("Creating advanced behavioral features...")
        
        if 'transaction_amount' in df.columns:
            if 'transaction_amount_log' in df.columns:
                df['amount_log'] = df['transaction_amount_log']
            else:
                df['amount_log'] = np.log1p(df['transaction_amount'])
                
            df['amount_category'] = pd.cut(
                df['transaction_amount'],
                bins=[0, 25, 100, 500, float('inf')],
                labels=[0, 1, 2, 3]
            ).astype(int)
            df['is_high_value'] = (df['transaction_amount'] > 500).astype(int)
            
            self.feature_info.update({
                'amount_log': 'Log-transformed transaction amount',
                'amount_category': 'Amount category (0-3)',
                'is_high_value': 'High-value transaction flag (>$500)'
            })
        return df
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about engineered features."""
        return self.feature_info


def handle_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle infinite values in the dataset."""
    logger.info("Handling infinite values...")
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled infinite values in {col} with median: {median_val}")
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data for modeling."""
    logger.info("Preprocessing data for modeling...")
    
    # Engineer features
    engineer = BaselineFeatureEngineer()
    df = engineer.engineer_features(df)
    
    # Handle infinite values
    df = handle_infinite_values(df)
    
    # Select only numeric features for baseline
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].copy()
    
    logger.info(f"Final preprocessed shape: {df_numeric.shape}")
    return df_numeric 