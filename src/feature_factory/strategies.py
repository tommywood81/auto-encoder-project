"""
Feature engineering strategies for fraud detection.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer(ABC):
    """Abstract base class for feature engineering strategies."""
    
    @abstractmethod
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for the given dataset."""
        pass
    
    @abstractmethod
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the features generated."""
        pass


class BaselineNumeric(FeatureEngineer):
    """Baseline numeric features - log and ratio transformations."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate baseline numeric features."""
        logger.info("Generating baseline numeric features...")
        
        df = df.copy()
        df['transaction_amount_log'] = np.log1p(df['transaction_amount'])
        df['amount_per_item'] = df['transaction_amount'] / (df['quantity'] + 1)
        
        logger.info(f"Baseline numeric features generated: {len(df.columns)} features")
        return df
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            "strategy": "baseline_numeric", 
            "description": "Log and ratio features from raw numerics"
        }


class CategoricalEncodings(FeatureEngineer):
    """Categorical encoding features."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate categorical encoding features."""
        logger.info("Generating categorical encoding features...")
        
        df = df.copy()
        df['payment_method_encoded'] = pd.Categorical(df['payment_method']).codes
        df['product_category_encoded'] = pd.Categorical(df['product_category']).codes
        df['device_used_encoded'] = pd.Categorical(df['device_used']).codes
        
        logger.info(f"Categorical encoding features generated: {len(df.columns)} features")
        return df
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            "strategy": "categorical", 
            "description": "Encoded payment, product, and device columns"
        }


class TemporalFeatures(FeatureEngineer):
    """Temporal features - time-based patterns."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate temporal features."""
        logger.info("Generating temporal features...")
        
        df = df.copy()
        df['is_late_night'] = ((df['transaction_hour'] >= 23) | (df['transaction_hour'] <= 6)).astype(int)
        df['is_burst_transaction'] = (
            (df['transaction_amount'] < df['transaction_amount'].quantile(0.2)) &
            (df['transaction_amount'].shift(-1) > df['transaction_amount'].quantile(0.7))
        ).astype(int)
        
        logger.info(f"Temporal features generated: {len(df.columns)} features")
        return df
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            "strategy": "temporal", 
            "description": "Late-night and burst transaction flags"
        }


class BehavioralFeatures(FeatureEngineer):
    """Behavioral features - customer behavior patterns."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate behavioral features."""
        logger.info("Generating behavioral features...")
        
        df = df.copy()
        df['amount_per_age'] = df['transaction_amount'] / (df['customer_age'] + 1)
        df['amount_per_account_age'] = df['transaction_amount'] / (df['account_age_days'] + 1)
        
        logger.info(f"Behavioral features generated: {len(df.columns)} features")
        return df
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            "strategy": "behavioral", 
            "description": "Behavioral ratios per age/account age"
        }


class DemographicFeatures(FeatureEngineer):
    """Demographic features - age-based risk bands."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate demographic features."""
        logger.info("Generating demographic features...")
        
        df = df.copy()
        df['customer_age_band'] = pd.cut(
            df['customer_age'], 
            bins=[0, 25, 35, 50, 65, 100], 
            labels=[4, 3, 2, 1, 0], 
            include_lowest=True
        ).astype(int)
        
        logger.info(f"Demographic features generated: {len(df.columns)} features")
        return df
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            "strategy": "demographics", 
            "description": "Customer age bucketed into risk bands"
        }


class FraudFlags(FeatureEngineer):
    """Fraud flag features - rule-based risk indicators."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate fraud flag features."""
        logger.info("Generating fraud flag features...")
        
        df = df.copy()
        df['high_amount_flag'] = (df['transaction_amount'] > df['transaction_amount'].quantile(0.9)).astype(int)
        df['new_account_flag'] = (df['account_age_days'] < 30).astype(int)
        df['young_customer_flag'] = (df['customer_age'] < 25).astype(int)
        df['fraud_risk_score'] = (
            df['high_amount_flag'] +
            df['new_account_flag'] +
            df['young_customer_flag']
        )
        
        logger.info(f"Fraud flag features generated: {len(df.columns)} features")
        return df
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            "strategy": "fraud_flags", 
            "description": "Rule-based fraud risk indicators"
        }


class RollingFeatures(FeatureEngineer):
    """Rolling features - time-series aggregations."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling features."""
        logger.info("Generating rolling features...")
        
        df = df.copy()
        df = df.sort_values(by=["customer_location", "transaction_date", "transaction_hour"]).reset_index(drop=True)
        df['rolling_avg_amount_3'] = (
            df.groupby("customer_location")['transaction_amount']
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )
        df['rolling_std_amount_3'] = (
            df.groupby("customer_location")['transaction_amount']
            .transform(lambda x: x.rolling(window=3, min_periods=1).std().fillna(0))
        )
        
        logger.info(f"Rolling features generated: {len(df.columns)} features")
        return df
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            "strategy": "rolling", 
            "description": "Rolling mean and std of amount per customer"
        }


class RankEncodingFeatures(FeatureEngineer):
    """Rank encoding features - percentile-based encodings."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rank encoding features."""
        logger.info("Generating rank encoding features...")
        
        df = df.copy()
        df['transaction_amount_rank'] = df['transaction_amount'].rank(pct=True)
        df['account_age_rank'] = df['account_age_days'].rank(pct=True)
        
        logger.info(f"Rank encoding features generated: {len(df.columns)} features")
        return df
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            "strategy": "rank_encoding", 
            "description": "Rank-based encodings of amount and account age"
        }


class TimeInteractionFeatures(FeatureEngineer):
    """Time interaction features - crossed features with time."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate time interaction features."""
        logger.info("Generating time interaction features...")
        
        df = df.copy()
        df['amount_x_hour'] = df['transaction_amount'] * df['transaction_hour']
        df['amount_per_hour'] = df['transaction_amount'] / (df['transaction_hour'] + 1)
        
        logger.info(f"Time interaction features generated: {len(df.columns)} features")
        return df
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            "strategy": "time_interactions", 
            "description": "Crossed and interaction features using hour"
        }


class CombinedFeatures(FeatureEngineer):
    """Combined features - all strategies together."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate combined features from all strategies."""
        logger.info("Generating combined features...")
        
        strategies = [
            BaselineNumeric(),
            CategoricalEncodings(),
            TemporalFeatures(),
            RollingFeatures(),
            BehavioralFeatures(),
            RankEncodingFeatures(),
            TimeInteractionFeatures(),
            DemographicFeatures(),
            FraudFlags(),
        ]
        
        df_combined = df.copy()
        
        for strategy in strategies:
            df_combined = strategy.generate_features(df_combined)
            logger.info(f"Applied {strategy.get_feature_info()['strategy']} strategy")
        
        # Remove duplicate columns if any
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
        
        logger.info(f"Combined features generated: {len(df_combined.columns)} features")
        return df_combined
    
    def get_feature_info(self) -> Dict[str, Any]:
        return {
            "strategy": "combined", 
            "description": "All feature engineering strategies combined"
        } 