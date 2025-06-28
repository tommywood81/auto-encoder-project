"""
Feature engineering strategies for fraud detection.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from .feature_factory import FeatureEngineer

logger = logging.getLogger(__name__)


class BaselineFeatures(FeatureEngineer):
    """Baseline feature engineering - core transaction features only."""
    
    def __init__(self):
        self.feature_info = {
            "strategy": "baseline",
            "description": "Core transaction features only",
            "features": {}
        }
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate baseline features."""
        logger.info("Generating baseline features...")
        
        df_features = df.copy()
        
        # Core features are already in the cleaned data
        # Just ensure we have the right columns
        core_features = [
            'transaction_amount', 'customer_age', 'quantity', 'account_age_days',
            'payment_method', 'product_category', 'device_used', 'customer_location_freq',
            'transaction_amount_log'
        ]
        
        available_features = [col for col in core_features if col in df_features.columns]
        logger.info(f"Available baseline features: {available_features}")
        
        # Select only baseline features + target
        if 'is_fraudulent' in df_features.columns:
            final_features = available_features + ['is_fraudulent']
        else:
            final_features = available_features
        
        df_baseline = df_features[final_features].copy()
        
        # Update feature info
        self.feature_info["features"] = {col: "baseline feature" for col in available_features}
        self.feature_info["feature_count"] = len(available_features)
        
        logger.info(f"Baseline features generated: {len(available_features)} features")
        return df_baseline
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get baseline feature information."""
        return self.feature_info


class TemporalFeatures(FeatureEngineer):
    """Temporal feature engineering - includes time-based features."""
    
    def __init__(self):
        self.feature_info = {
            "strategy": "temporal",
            "description": "Core features + temporal patterns",
            "features": {}
        }
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate temporal features."""
        logger.info("Generating temporal features...")
        
        # Start with baseline features
        baseline = BaselineFeatures()
        df_features = baseline.generate_features(df)
        
        # Add temporal features if available
        temporal_cols = ['is_between_11pm_and_6am']
        available_temporal = [col for col in temporal_cols if col in df.columns]
        
        if available_temporal:
            for col in available_temporal:
                df_features[col] = df[col]
            
            logger.info(f"Added temporal features: {available_temporal}")
            
            # Update feature info
            self.feature_info["features"].update({
                col: "temporal feature" for col in available_temporal
            })
            self.feature_info["feature_count"] = len(df_features.columns) - 1  # -1 for target
        
        logger.info(f"Temporal features generated: {len(df_features.columns) - 1} features")
        return df_features
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get temporal feature information."""
        return self.feature_info


class BehaviouralFeatures(FeatureEngineer):
    """Behavioural feature engineering - includes amount per item."""
    
    def __init__(self):
        self.feature_info = {
            "strategy": "behavioural",
            "description": "Core features + amount per item",
            "features": {}
        }
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate behavioural features."""
        logger.info("Generating behavioural features...")
        
        # Start with baseline features
        baseline = BaselineFeatures()
        df_features = baseline.generate_features(df)
        
        # Create behavioural features
        df_behavioural = df.copy()
        
        # Calculate amount per item (transaction amount / quantity)
        df_behavioural['amount_per_item'] = df_behavioural['transaction_amount'] / df_behavioural['quantity'].clip(lower=1)
        
        # Clip extreme values to prevent outliers
        df_behavioural['amount_per_item'] = df_behavioural['amount_per_item'].clip(
            lower=df_behavioural['amount_per_item'].quantile(0.01),
            upper=df_behavioural['amount_per_item'].quantile(0.99)
        )
        
        # Add behavioural features to the baseline features
        behavioural_features = ['amount_per_item']
        
        for feature in behavioural_features:
            if feature in df_behavioural.columns:
                df_features[feature] = df_behavioural[feature]
        
        logger.info(f"Added behavioural features: {behavioural_features}")
        
        # Update feature info
        self.feature_info["features"].update({
            col: "behavioural feature" for col in behavioural_features
        })
        self.feature_info["feature_count"] = len(df_features.columns) - 1  # -1 for target
        
        logger.info(f"Behavioural features generated: {len(df_features.columns) - 1} features")
        return df_features
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get behavioural feature information."""
        return self.feature_info


class DemographicRiskFeatures(FeatureEngineer):
    """Demographic risk feature engineering - includes customer age risk scores."""
    
    def __init__(self):
        self.feature_info = {
            "strategy": "demographic_risk",
            "description": "Core features + customer age risk scores",
            "features": {}
        }
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate demographic risk features."""
        logger.info("Generating demographic risk features...")
        
        # Start with baseline features
        baseline = BaselineFeatures()
        df_features = baseline.generate_features(df)
        
        # Create customer age risk features
        df_risk = df.copy()
        
        # Create risk score based on customer age (younger customers = higher risk)
        # Normalize so that 18 years = 1.0 risk, 65+ years = 0.0 risk
        df_risk['customer_age_risk'] = 1.0 - ((df_risk['customer_age'] - 18) / 47.0).clip(lower=0.0, upper=1.0)
        
        # Add customer age risk features
        risk_features = ['customer_age_risk']
        
        for feature in risk_features:
            if feature in df_risk.columns:
                df_features[feature] = df_risk[feature]
        
        logger.info(f"Added demographic risk features: {risk_features}")
        
        # Update feature info
        self.feature_info["features"].update({
            col: "demographic risk feature" for col in risk_features
        })
        self.feature_info["feature_count"] = len(df_features.columns) - 1  # -1 for target
        
        logger.info(f"Demographic risk features generated: {len(df_features.columns) - 1} features")
        return df_features
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get demographic risk feature information."""
        return self.feature_info


class CombinedFeatures(FeatureEngineer):
    """Combined feature engineering - concatenates all unique features from all strategies."""
    def __init__(self):
        self.feature_info = {
            "strategy": "combined",
            "description": "All unique features from baseline, temporal, behavioural, and demographic risk.",
            "features": {}
        }

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating combined features from all strategies...")
        # Instantiate each strategy
        strategies = [
            BaselineFeatures(),
            TemporalFeatures(),
            BehaviouralFeatures(),
            DemographicRiskFeatures()
        ]
        # Generate features for each
        feature_dfs = [s.generate_features(df) for s in strategies]
        # Concatenate all columns, avoiding duplicates
        combined_df = feature_dfs[0].copy()
        for fdf in feature_dfs[1:]:
            for col in fdf.columns:
                if col not in combined_df.columns:
                    combined_df[col] = fdf[col]
        # Update feature info
        all_features = [col for col in combined_df.columns if col != 'is_fraudulent']
        self.feature_info["features"] = {col: "combined feature" for col in all_features}
        self.feature_info["feature_count"] = len(all_features)
        logger.info(f"Combined features generated: {len(all_features)} features")
        return combined_df

    def get_feature_info(self) -> Dict[str, Any]:
        return self.feature_info 