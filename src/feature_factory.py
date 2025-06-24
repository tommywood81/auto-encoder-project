"""
Feature factory for fraud detection - configurable feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import logging

from src.config import FeatureConfig

logger = logging.getLogger(__name__)


class FeatureBuilder(ABC):
    """Abstract base class for feature builders."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.feature_info = {}
    
    @abstractmethod
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features and return modified dataframe."""
        pass
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about built features."""
        return self.feature_info


class TransactionFeatureBuilder(FeatureBuilder):
    """Build transaction-related features."""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build transaction features based on configuration."""
        logger.info("Building transaction features...")
        
        # Amount transformations
        if self.config.enable_amount_features and 'TransactionAmt' in df.columns:
            if 'log' in self.config.amount_transformations:
                df['amount_log'] = np.log1p(df['TransactionAmt'])
                self.feature_info['amount_log'] = 'Log transformation of transaction amount'
            
            if 'sqrt' in self.config.amount_transformations:
                df['amount_sqrt'] = np.sqrt(df['TransactionAmt'])
                self.feature_info['amount_sqrt'] = 'Square root transformation of transaction amount'
        
        # Card features
        if self.config.enable_card_features:
            card_cols = [col for col in df.columns if 'card' in col.lower()]
            if card_cols:
                df['card_count'] = df[card_cols].notna().sum(axis=1)
                self.feature_info['card_count'] = f'Count of card-related fields filled ({len(card_cols)} columns)'
        
        # Email features
        if self.config.enable_email_features:
            email_cols = [col for col in df.columns if 'email' in col.lower()]
            if email_cols:
                df['email_count'] = df[email_cols].notna().sum(axis=1)
                self.feature_info['email_count'] = f'Count of email-related fields filled ({len(email_cols)} columns)'
        
        # Address features
        if self.config.enable_addr_features:
            addr_cols = [col for col in df.columns if 'addr' in col.lower()]
            if addr_cols:
                df['addr_count'] = df[addr_cols].notna().sum(axis=1)
                self.feature_info['addr_count'] = f'Count of address-related fields filled ({len(addr_cols)} columns)'
        
        logger.info(f"Built {len(self.feature_info)} transaction features")
        return df


class IdentityFeatureBuilder(FeatureBuilder):
    """Build identity-related features."""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build identity features based on configuration."""
        logger.info("Building identity features...")
        
        # V features
        if self.config.enable_v_features:
            v_cols = [col for col in df.columns if col.startswith('V')]
            if v_cols:
                for agg in self.config.v_feature_aggregations:
                    if agg == 'count':
                        df['v_features_count'] = df[v_cols].notna().sum(axis=1)
                        self.feature_info['v_features_count'] = f'Count of V features filled ({len(v_cols)} columns)'
                    elif agg == 'mean':
                        df['v_features_mean'] = df[v_cols].mean(axis=1)
                        self.feature_info['v_features_mean'] = f'Mean of V features ({len(v_cols)} columns)'
                    elif agg == 'std':
                        df['v_features_std'] = df[v_cols].std(axis=1)
                        self.feature_info['v_features_std'] = f'Standard deviation of V features ({len(v_cols)} columns)'
                    elif agg == 'sum':
                        df['v_features_sum'] = df[v_cols].sum(axis=1)
                        self.feature_info['v_features_sum'] = f'Sum of V features ({len(v_cols)} columns)'
        
        # ID features
        if self.config.enable_id_features:
            id_cols = [col for col in df.columns if col.startswith('id_')]
            if id_cols:
                df['id_features_count'] = df[id_cols].notna().sum(axis=1)
                self.feature_info['id_features_count'] = f'Count of ID features filled ({len(id_cols)} columns)'
        
        logger.info(f"Built {len(self.feature_info)} identity features")
        return df


class InteractionFeatureBuilder(FeatureBuilder):
    """Build interaction features between different feature groups."""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build interaction features based on configuration."""
        if not self.config.enable_interaction_features:
            return df
        
        logger.info("Building interaction features...")
        
        # Amount interactions
        if 'TransactionAmt' in df.columns:
            if 'card_count' in df.columns:
                df['amount_card_interaction'] = df['TransactionAmt'] * df['card_count']
                self.feature_info['amount_card_interaction'] = 'Transaction amount * card count'
            
            if 'email_count' in df.columns:
                df['amount_email_interaction'] = df['TransactionAmt'] * df['email_count']
                self.feature_info['amount_email_interaction'] = 'Transaction amount * email count'
        
        # V features interactions
        if 'v_features_count' in df.columns and 'v_features_mean' in df.columns:
            df['v_count_mean_interaction'] = df['v_features_count'] * df['v_features_mean']
            self.feature_info['v_count_mean_interaction'] = 'V features count * V features mean'
        
        logger.info(f"Built {len(self.feature_info)} interaction features")
        return df


class StatisticalFeatureBuilder(FeatureBuilder):
    """Build statistical features."""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build statistical features based on configuration."""
        if not self.config.enable_statistical_features:
            return df
        
        logger.info("Building statistical features...")
        
        # V features statistics
        v_cols = [col for col in df.columns if col.startswith('V')]
        if v_cols:
            for stat in self.config.statistical_features:
                if stat == 'q25':
                    df['v_features_q25'] = df[v_cols].quantile(0.25, axis=1)
                    self.feature_info['v_features_q25'] = '25th percentile of V features'
                elif stat == 'q75':
                    df['v_features_q75'] = df[v_cols].quantile(0.75, axis=1)
                    self.feature_info['v_features_q75'] = '75th percentile of V features'
                elif stat == 'iqr':
                    if 'v_features_q75' in df.columns and 'v_features_q25' in df.columns:
                        df['v_features_iqr'] = df['v_features_q75'] - df['v_features_q25']
                        self.feature_info['v_features_iqr'] = 'Interquartile range of V features'
                elif stat == 'range':
                    df['v_features_range'] = df[v_cols].max(axis=1) - df[v_cols].min(axis=1)
                    self.feature_info['v_features_range'] = 'Range of V features'
        
        logger.info(f"Built {len(self.feature_info)} statistical features")
        return df


class TemporalFeatureBuilder(FeatureBuilder):
    """Build temporal features (placeholder for future implementation)."""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build temporal features (not yet implemented)."""
        if not self.config.enable_temporal_features:
            return df
        
        logger.warning("Temporal features not yet implemented")
        return df


class BehavioralDriftFeatureBuilder(FeatureBuilder):
    """Build behavioral drift features (placeholder for future implementation)."""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build behavioral drift features (not yet implemented)."""
        if not self.config.enable_behavioral_drift:
            return df
        
        logger.warning("Behavioral drift features not yet implemented")
        return df


class EntityNoveltyFeatureBuilder(FeatureBuilder):
    """Build entity novelty features (placeholder for future implementation)."""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build entity novelty features (not yet implemented)."""
        if not self.config.enable_entity_novelty:
            return df
        
        logger.warning("Entity novelty features not yet implemented")
        return df


class FeatureFactory:
    """Main feature factory that orchestrates all feature builders."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.builders = {
            'transaction': TransactionFeatureBuilder(config),
            'identity': IdentityFeatureBuilder(config),
            'interaction': InteractionFeatureBuilder(config),
            'statistical': StatisticalFeatureBuilder(config),
            'temporal': TemporalFeatureBuilder(config),
            'behavioral_drift': BehavioralDriftFeatureBuilder(config),
            'entity_novelty': EntityNoveltyFeatureBuilder(config)
        }
        self.feature_info = {}
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all features based on configuration."""
        logger.info("Starting feature factory pipeline...")
        
        # Build features in order
        for builder_name, builder in self.builders.items():
            df = builder.build(df)
            self.feature_info[builder_name] = builder.get_feature_info()
        
        # Handle infinite values
        df = self._handle_infinite_values(df)
        
        logger.info(f"Feature factory completed. Total features: {df.shape[1]}")
        return df
    
    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite values in the dataset."""
        logger.info("Handling infinite values...")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        logger.info(f"Handled infinite values in {len(numeric_cols)} numeric columns")
        return df
    
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