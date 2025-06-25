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
        
        # 1. Temporal features (essential for fraud detection)
        df_engineered = self._create_temporal_features(df_engineered)
        
        # 2. Entity frequency features (customer behavior patterns)
        df_engineered = self._create_entity_frequency_features(df_engineered)
        
        # 3. Simple behavioral features (rolling averages)
        df_engineered = self._create_behavioral_features(df_engineered)
        
        logger.info(f"Engineered {len(self.feature_info)} baseline features")
        return df_engineered
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create essential temporal features."""
        logger.info("Creating temporal features...")
        
        if 'transaction_date' in df.columns:
            # Convert to datetime
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            
            # Basic time features
            df['hour'] = df['transaction_date'].dt.hour
            df['day_of_week'] = df['transaction_date'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            
            # High-risk hours (based on fraud patterns)
            df['is_high_risk_hour'] = df['hour'].isin([0, 1, 3, 4, 5]).astype(int)
            
            # Remove original date column
            df = df.drop(columns=['transaction_date'])
            
            self.feature_info.update({
                'hour': 'Hour of day (0-23)',
                'day_of_week': 'Day of week (0-6)',
                'is_weekend': 'Weekend transaction flag',
                'is_night': 'Night transaction flag (10 PM - 6 AM)',
                'is_high_risk_hour': 'High fraud risk hours flag'
            })
        
        return df
    
    def _create_entity_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create entity frequency features for customer behavior."""
        logger.info("Creating entity frequency features...")
        
        if 'customer_location' in df.columns:
            # Customer frequency (how often this customer appears)
            customer_freq = df['customer_location'].value_counts(normalize=True)
            df['customer_frequency'] = df['customer_location'].map(customer_freq)
            
            # Customer novelty (inverse of frequency - rare customers)
            df['customer_novelty'] = 1 - df['customer_frequency']
            
            # Remove original customer_location (keep encoded version)
            df = df.drop(columns=['customer_location'])
            
            self.feature_info.update({
                'customer_frequency': 'Frequency of customer in dataset',
                'customer_novelty': 'Novelty score (1 - frequency)'
            })
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create simple behavioral features."""
        logger.info("Creating behavioral features...")
        
        if 'transaction_amount' in df.columns:
            # Amount-based features
            df['amount_log'] = np.log1p(df['transaction_amount'])
            df['amount_squared'] = df['transaction_amount'] ** 2
            
            # Simple amount categories
            df['amount_category'] = pd.cut(
                df['transaction_amount'],
                bins=[0, 50, 200, 1000, float('inf')],
                labels=[0, 1, 2, 3]
            ).astype(int)
            
            self.feature_info.update({
                'amount_log': 'Log-transformed transaction amount',
                'amount_squared': 'Squared transaction amount',
                'amount_category': 'Amount category (0-3)'
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