"""
Feature engineering logic for E-commerce Fraud Detection.
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
        if self.config.enable_amount_features and 'Transaction Amount' in df.columns:
            if 'log' in self.config.amount_transformations:
                df['amount_log'] = np.log1p(df['Transaction Amount'])
                self.feature_info['amount_log'] = 'Log transformation of transaction amount'
            
            if 'sqrt' in self.config.amount_transformations:
                df['amount_sqrt'] = np.sqrt(df['Transaction Amount'])
                self.feature_info['amount_sqrt'] = 'Square root transformation of transaction amount'
        
        # Quantity features
        if 'Quantity' in df.columns:
            df['quantity_squared'] = df['Quantity'] ** 2
            self.feature_info['quantity_squared'] = 'Squared quantity'
            
            # Amount per item
            if 'Transaction Amount' in df.columns:
                df['amount_per_item'] = df['Transaction Amount'] / df['Quantity'].replace(0, 1)
                self.feature_info['amount_per_item'] = 'Transaction amount per item'
        
        logger.info(f"Built {len(self.feature_info)} transaction features")
        return df


class CustomerFeatureBuilder(FeatureBuilder):
    """Build customer-related features."""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build customer features based on configuration."""
        logger.info("Building customer features...")
        
        # Age features
        if 'Customer Age' in df.columns:
            # Age bins
            df['age_bin'] = pd.cut(df['Customer Age'], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3])
            df['age_bin'] = df['age_bin'].astype(int)
            self.feature_info['age_bin'] = 'Age bin (0-25, 26-35, 36-50, 50+)'
            
            # Account age features
            if 'Account Age Days' in df.columns:
                df['account_age_years'] = df['Account Age Days'] / 365.25
                self.feature_info['account_age_years'] = 'Account age in years'
                
                # Age to account age ratio
                df['age_account_ratio'] = df['Customer Age'] / df['account_age_years'].replace(0, 1)
                self.feature_info['age_account_ratio'] = 'Customer age to account age ratio'
        
        # Time-based features
        if 'Transaction Hour' in df.columns:
            # Hour bins
            df['hour_bin'] = pd.cut(df['Transaction Hour'], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3])
            df['hour_bin'] = df['hour_bin'].astype(int)
            self.feature_info['hour_bin'] = 'Hour bin (0-6, 7-12, 13-18, 19-24)'
            
            # Is night transaction
            df['is_night'] = ((df['Transaction Hour'] >= 22) | (df['Transaction Hour'] <= 6)).astype(int)
            self.feature_info['is_night'] = 'Night transaction (10 PM - 6 AM)'
        
        logger.info(f"Built {len(self.feature_info)} customer features")
        return df


class InteractionFeatureBuilder(FeatureBuilder):
    """Build interaction features between different feature groups."""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build interaction features based on configuration."""
        if not self.config.enable_interaction_features:
            return df
        
        logger.info("Building interaction features...")
        
        # Amount interactions
        if 'Transaction Amount' in df.columns:
            if 'Customer Age' in df.columns:
                df['amount_age_interaction'] = df['Transaction Amount'] * df['Customer Age']
                self.feature_info['amount_age_interaction'] = 'Transaction amount * customer age'
            
            if 'Quantity' in df.columns:
                df['amount_quantity_interaction'] = df['Transaction Amount'] * df['Quantity']
                self.feature_info['amount_quantity_interaction'] = 'Transaction amount * quantity'
        
        # Payment method interactions
        if 'Payment Method' in df.columns and 'Transaction Amount' in df.columns:
            # Create interaction between payment method and amount
            payment_methods = df['Payment Method'].unique()
            for method in payment_methods:
                mask = df['Payment Method'] == method
                df[f'amount_{method}_interaction'] = df['Transaction Amount'] * mask.astype(int)
                self.feature_info[f'amount_{method}_interaction'] = f'Amount * {method} payment method'
        
        logger.info(f"Built {len(self.feature_info)} interaction features")
        return df


class StatisticalFeatureBuilder(FeatureBuilder):
    """Build statistical features."""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build statistical features based on configuration."""
        if not self.config.enable_statistical_features:
            return df
        
        logger.info("Building statistical features...")
        
        # Transaction amount statistics
        if 'Transaction Amount' in df.columns:
            for stat in self.config.statistical_features:
                if stat == 'q25':
                    df['amount_q25'] = df['Transaction Amount'].quantile(0.25)
                    self.feature_info['amount_q25'] = '25th percentile of transaction amount'
                elif stat == 'q75':
                    df['amount_q75'] = df['Transaction Amount'].quantile(0.75)
                    self.feature_info['amount_q75'] = '75th percentile of transaction amount'
                elif stat == 'iqr':
                    if 'amount_q75' in df.columns and 'amount_q25' in df.columns:
                        df['amount_iqr'] = df['amount_q75'] - df['amount_q25']
                        self.feature_info['amount_iqr'] = 'Interquartile range of transaction amount'
        
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