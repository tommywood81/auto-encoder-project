"""
Feature engineering logic for fraud detection.
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import PipelineConfig, FeatureConfig
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import logging

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


def handle_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle infinite values in the dataset."""
    logger.info("Handling infinite values...")
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    logger.info(f"Handled infinite values in {len(numeric_cols)} numeric columns")
    return df


class FeatureEngineer:
    """Feature engineering pipeline for fraud detection dataset."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.scaler = None
        
    def load_cleaned_data(self):
        """Load cleaned data from the cleaned directory."""
        print("Loading cleaned data...")
        
        cleaned_file = os.path.join(self.config.data.cleaned_dir, "train_cleaned.csv")
        if not os.path.exists(cleaned_file):
            raise FileNotFoundError(f"Cleaned data not found at {cleaned_file}. Run data cleaning first.")
        
        df = pd.read_csv(cleaned_file)
        print(f"Cleaned data: {df.shape}")
        
        return df
    
    def engineer_features(self, save_output=True):
        """Complete feature engineering pipeline using feature factory."""
        print("Starting feature engineering pipeline...")
        
        # Load cleaned data
        df = self.load_cleaned_data()
        
        # Use feature factory to build features
        df = self.feature_factory.build_features(df)
        
        print(f"Final engineered shape: {df.shape}")
        
        # Save engineered data if requested
        if save_output:
            self.save_engineered_data(df)
        
        return df
    
    def save_engineered_data(self, df, suffix=""):
        """Save engineered data to the engineered directory."""
        print("Saving engineered data...")
        
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
        
        # Save feature information from factory
        feature_summary = self.feature_factory.get_feature_summary()
        info_file = os.path.join(self.config.data.engineered_dir, f"feature_info{suffix}.json")
        with open(info_file, 'w') as f:
            json.dump(convert_numpy_types(feature_summary), f, indent=2)
        
        print(f"Saved engineered data to {output_file}")
        
        return output_file
    
    def preprocess_for_modeling(self, df):
        """Preprocess engineered data for autoencoder training."""
        print("Preprocessing and splitting data...")
        
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
        
        print(f"Training samples (non-fraudulent): {X_train_ae.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        
        return X_train_ae, X_test, y_train, y_test, self.scaler


def preprocess_data(df):
    """Legacy function for backward compatibility."""
    engineer = FeatureEngineer()
    return engineer.preprocess_for_modeling(df) 