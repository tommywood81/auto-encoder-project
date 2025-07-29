"""
Feature engineering for fraud detection.
Config-driven implementation with leakage-free transformations.
"""

import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.preprocessing import RobustScaler, LabelEncoder
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Config-driven feature engineering with leakage-free transformations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature engineer with configuration."""
        self.config = config or {}
        self.is_fitted = False
        
        # Fitted objects (will be set during fit)
        self.amount_scaler = None
        self.label_encoders = {}
        self.percentile_thresholds = {}
        
        # Feature selection flags
        self.use_amount_features = self.config.get('use_amount_features', True)
        self.use_temporal_features = self.config.get('use_temporal_features', True)
        self.use_customer_features = self.config.get('use_customer_features', True)
        self.use_risk_flags = self.config.get('use_risk_flags', True)
    
    def fit_transform(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit on training data and transform both train and test data."""
        
        logger.info("Starting feature engineering fit_transform...")
        
        # Fit on training data only
        df_train_features = self._fit_and_transform(df_train, is_training=True)
        
        # Transform test data using fitted parameters
        df_test_features = self._transform(df_test, is_training=False)
        
        self.is_fitted = True
        
        logger.info(f"Feature engineering completed. Train: {df_train_features.shape}, Test: {df_test_features.shape}")
        
        return df_train_features, df_test_features
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters."""
        
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        return self._transform(df, is_training=False)
    
    def _fit_and_transform(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Fit and transform data (for training data)."""
        
        df_features = df.copy()
        
        # Engineer features based on config
        if self.use_amount_features:
            df_features = self._engineer_amount_features(df_features, is_training)
        
        if self.use_temporal_features:
            df_features = self._engineer_temporal_features(df_features)
        
        if self.use_customer_features:
            df_features = self._engineer_customer_features(df_features, is_training)
        
        if self.use_risk_flags:
            df_features = self._engineer_risk_flags(df_features, is_training)
        
        # Ensure all features are numeric
        df_features = self._ensure_numeric_features(df_features, is_training)
        
        return df_features
    
    def _transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Transform data using fitted parameters (for test data)."""
        
        df_features = df.copy()
        
        # Apply fitted transformations
        if self.use_amount_features:
            df_features = self._apply_amount_features(df_features)
        
        if self.use_temporal_features:
            df_features = self._engineer_temporal_features(df_features)
        
        if self.use_customer_features:
            df_features = self._apply_customer_features(df_features)
        
        if self.use_risk_flags:
            df_features = self._apply_risk_flags(df_features)
        
        # Ensure all features are numeric
        df_features = self._ensure_numeric_features(df_features, is_training)
        
        return df_features
    
    def _engineer_amount_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer amount-related features."""
        
        # Log transform
        df['amount_log'] = np.log1p(df['amount'])
        
        # Amount per item
        df['amount_per_item'] = df['amount'] / (df['quantity'] + 1)
        
        # Robust scaling (fit on training data only)
        if is_training:
            self.amount_scaler = RobustScaler()
            df['amount_scaled'] = self.amount_scaler.fit_transform(df[['amount']]).flatten()
        else:
            df['amount_scaled'] = self.amount_scaler.transform(df[['amount']]).flatten()
        
        # High amount flags (fit percentiles on training data only)
        if is_training:
            self.percentile_thresholds['amount_95'] = np.percentile(df['amount'], 95)
            self.percentile_thresholds['amount_99'] = np.percentile(df['amount'], 99)
        
        df['high_amount_95'] = (df['amount'] > self.percentile_thresholds['amount_95']).astype(int)
        df['high_amount_99'] = (df['amount'] > self.percentile_thresholds['amount_99']).astype(int)
        
        return df
    
    def _apply_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted amount features to test data."""
        
        # Log transform
        df['amount_log'] = np.log1p(df['amount'])
        
        # Amount per item
        df['amount_per_item'] = df['amount'] / (df['quantity'] + 1)
        
        # Robust scaling (using fitted scaler)
        df['amount_scaled'] = self.amount_scaler.transform(df[['amount']]).flatten()
        
        # High amount flags (using fitted thresholds)
        df['high_amount_95'] = (df['amount'] > self.percentile_thresholds['amount_95']).astype(int)
        df['high_amount_99'] = (df['amount'] > self.percentile_thresholds['amount_99']).astype(int)
        
        return df
    
    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer temporal features."""
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract hour
        df['hour'] = df['timestamp'].dt.hour
        
        # Time-based flags
        df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        return df
    
    def _engineer_customer_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer customer-related features."""
        
        # Age group encoding
        df['age_group'] = pd.cut(df['customer_age'], 
                                bins=[0, 18, 25, 35, 50, 100], 
                                labels=['teen', 'young', 'adult', 'middle', 'senior'])
        
        if is_training:
            self.label_encoders['age_group'] = LabelEncoder()
            df['age_group_encoded'] = self.label_encoders['age_group'].fit_transform(df['age_group'])
        else:
            df['age_group_encoded'] = self.label_encoders['age_group'].transform(df['age_group'])
        
        # Account age features
        df['account_age_days_log'] = np.log1p(df['account_age_days'])
        df['new_account'] = (df['account_age_days'] <= 7).astype(int)
        df['established_account'] = (df['account_age_days'] > 30).astype(int)
        
        # Location frequency encoding
        if is_training:
            location_counts = df['customer_location'].value_counts()
            self.percentile_thresholds['location_freq'] = location_counts.to_dict()
        
        df['location_freq'] = df['customer_location'].map(self.percentile_thresholds['location_freq']).fillna(0)
        
        # Categorical encodings
        categorical_cols = ['payment_method', 'product_category', 'device_used']
        
        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _apply_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted customer features to test data."""
        
        # Age group encoding
        df['age_group'] = pd.cut(df['customer_age'], 
                                bins=[0, 18, 25, 35, 50, 100], 
                                labels=['teen', 'young', 'adult', 'middle', 'senior'])
        df['age_group_encoded'] = self.label_encoders['age_group'].transform(df['age_group'])
        
        # Account age features
        df['account_age_days_log'] = np.log1p(df['account_age_days'])
        df['new_account'] = (df['account_age_days'] <= 7).astype(int)
        df['established_account'] = (df['account_age_days'] > 30).astype(int)
        
        # Location frequency encoding
        df['location_freq'] = df['customer_location'].map(self.percentile_thresholds['location_freq']).fillna(0)
        
        # Categorical encodings
        categorical_cols = ['payment_method', 'product_category', 'device_used']
        
        for col in categorical_cols:
            if col in df.columns:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _engineer_risk_flags(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer risk flag features."""
        
        # High quantity flag
        if is_training:
            self.percentile_thresholds['quantity_95'] = np.percentile(df['quantity'], 95)
        
        df['high_quantity'] = (df['quantity'] > self.percentile_thresholds['quantity_95']).astype(int)
        
        # Young customer flag
        df['young_customer'] = (df['customer_age'] <= 18).astype(int)
        
        # High risk combinations
        df['high_risk_combination'] = ((df['amount'] > self.percentile_thresholds['amount_95']) & 
                                      (df['is_late_night'] == 1)).astype(int)
        
        df['new_account_high_amount'] = ((df['new_account'] == 1) & 
                                        (df['amount'] > self.percentile_thresholds['amount_95'])).astype(int)
        
        # Interaction features
        df['amount_quantity_interaction'] = df['amount'] * df['quantity']
        df['age_account_interaction'] = df['customer_age'] * df['account_age_days']
        df['amount_hour_interaction'] = df['amount'] * df['hour']
        
        return df
    
    def _apply_risk_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted risk flags to test data."""
        
        # High quantity flag
        df['high_quantity'] = (df['quantity'] > self.percentile_thresholds['quantity_95']).astype(int)
        
        # Young customer flag
        df['young_customer'] = (df['customer_age'] <= 18).astype(int)
        
        # High risk combinations
        df['high_risk_combination'] = ((df['amount'] > self.percentile_thresholds['amount_95']) & 
                                      (df['is_late_night'] == 1)).astype(int)
        
        df['new_account_high_amount'] = ((df['new_account'] == 1) & 
                                        (df['amount'] > self.percentile_thresholds['amount_95'])).astype(int)
        
        # Interaction features
        df['amount_quantity_interaction'] = df['amount'] * df['quantity']
        df['age_account_interaction'] = df['customer_age'] * df['account_age_days']
        df['amount_hour_interaction'] = df['amount'] * df['hour']
        
        return df
    
    def _ensure_numeric_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Ensure all features are numeric and handle any remaining categorical columns."""
        
        # Drop original categorical columns that have been encoded
        categorical_cols = ['payment_method', 'product_category', 'device_used', 'age_group', 'customer_location']
        for col in categorical_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Drop timestamp column (temporal features extracted)
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        
        # Handle any remaining non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            if col != 'is_fraudulent':  # Keep target variable
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[col] = self.label_encoders[col].transform(df[col])
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names."""
        
        # Core features that are always included
        feature_names = [
            'amount_log', 'amount_per_item', 'amount_scaled',
            'high_amount_95', 'high_amount_99',
            'hour', 'is_late_night', 'is_business_hours',
            'age_group_encoded', 'account_age_days_log',
            'new_account', 'established_account', 'location_freq',
            'payment_method_encoded', 'product_category_encoded', 'device_used_encoded',
            'amount_quantity_interaction', 'age_account_interaction', 'amount_hour_interaction',
            'high_quantity', 'young_customer', 'high_risk_combination', 'new_account_high_amount'
        ]
        
        # Filter based on config
        if not self.use_amount_features:
            feature_names = [f for f in feature_names if not f.startswith('amount')]
        
        if not self.use_temporal_features:
            feature_names = [f for f in feature_names if not f in ['hour', 'is_late_night', 'is_business_hours']]
        
        if not self.use_customer_features:
            feature_names = [f for f in feature_names if not f in ['age_group_encoded', 'account_age_days_log', 
                                                                  'new_account', 'established_account', 'location_freq',
                                                                  'payment_method_encoded', 'product_category_encoded', 
                                                                  'device_used_encoded']]
        
        if not self.use_risk_flags:
            feature_names = [f for f in feature_names if not f in ['high_quantity', 'young_customer', 
                                                                  'high_risk_combination', 'new_account_high_amount']]
        
        return feature_names
    
    def save_fitted_objects(self, filepath: str):
        """Save fitted objects to file."""
        
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before saving")
        
        fitted_objects = {
            'amount_scaler': self.amount_scaler,
            'label_encoders': self.label_encoders,
            'percentile_thresholds': self.percentile_thresholds,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(fitted_objects, f)
        
        logger.info(f"Fitted objects saved to: {filepath}")
    
    def load_fitted_objects(self, filepath: str):
        """Load fitted objects from file."""
        
        with open(filepath, 'rb') as f:
            fitted_objects = pickle.load(f)
        
        self.amount_scaler = fitted_objects['amount_scaler']
        self.label_encoders = fitted_objects['label_encoders']
        self.percentile_thresholds = fitted_objects['percentile_thresholds']
        self.config = fitted_objects['config']
        self.is_fitted = fitted_objects['is_fitted']
        
        # Set feature selection flags
        self.use_amount_features = self.config.get('use_amount_features', True)
        self.use_temporal_features = self.config.get('use_temporal_features', True)
        self.use_customer_features = self.config.get('use_customer_features', True)
        self.use_risk_flags = self.config.get('use_risk_flags', True)
        
        logger.info(f"Fitted objects loaded from: {filepath}") 