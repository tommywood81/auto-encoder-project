"""
Simplified Feature Engineer for Fraud Detection
Based on EDA insights and production best practices
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Production-grade feature engineer for fraud detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.fitted_objects = {}
        self.is_fitted = False
        
    def fit_transform(self, df_train: pd.DataFrame, df_test: Optional[pd.DataFrame] = None) -> tuple:
        """Fit on training data and transform both train and test."""
        logger.info("Fitting feature engineering on training data...")
        
        # Fit on training data
        df_train_features = self._fit_and_transform(df_train, is_training=True)
        
        # Transform test data if provided
        if df_test is not None:
            df_test_features = self._transform(df_test)
            self.is_fitted = True
            return df_train_features, df_test_features
        else:
            self.is_fitted = True
            return df_train_features, None
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted objects."""
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")
        return self._transform(df)
    
    def _fit_and_transform(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Fit and transform data."""
        df = df.copy()
        
        # 1. Transaction Amount Features (most important)
        df = self._engineer_amount_features(df, is_training)
        
        # 2. Temporal Features (fraud patterns)
        df = self._engineer_temporal_features(df)
        
        # 3. Customer Features (behavioral patterns)
        df = self._engineer_customer_features(df, is_training)
        
        # 4. Categorical Features
        df = self._engineer_categorical_features(df, is_training)
        
        # 5. Interaction Features
        df = self._engineer_interaction_features(df)
        
        # 6. Risk Flags (domain knowledge)
        df = self._engineer_risk_flags(df, is_training)
        
        return df
    
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted objects."""
        return self._fit_and_transform(df, is_training=False)
    
    def _engineer_amount_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer transaction amount features."""
        
        # Log transform (handles skewness)
        df['amount_log'] = np.log1p(df['transaction_amount'])
        
        # Square root transform (alternative to log)
        df['amount_sqrt'] = np.sqrt(df['transaction_amount'])
        
        # Amount per item
        df['amount_per_item'] = df['transaction_amount'] / (df['quantity'] + 1)
        
        # Amount percentiles (fit on training data only)
        if is_training:
            amount_90 = df['transaction_amount'].quantile(0.90)
            amount_95 = df['transaction_amount'].quantile(0.95)
            amount_99 = df['transaction_amount'].quantile(0.99)
            self.fitted_objects['amount_90'] = amount_90
            self.fitted_objects['amount_95'] = amount_95
            self.fitted_objects['amount_99'] = amount_99
        else:
            amount_90 = self.fitted_objects.get('amount_90', df['transaction_amount'].quantile(0.90))
            amount_95 = self.fitted_objects.get('amount_95', df['transaction_amount'].quantile(0.95))
            amount_99 = self.fitted_objects.get('amount_99', df['transaction_amount'].quantile(0.99))
        
        # High amount flags
        df['high_amount_90'] = (df['transaction_amount'] > amount_90).astype(int)
        df['high_amount_95'] = (df['transaction_amount'] > amount_95).astype(int)
        df['high_amount_99'] = (df['transaction_amount'] > amount_99).astype(int)
        
        # Amount scaling
        if is_training:
            scaler = RobustScaler()
            df['amount_scaled'] = scaler.fit_transform(df[['transaction_amount']]).flatten()
            self.fitted_objects['amount_scaler'] = scaler
        else:
            scaler = self.fitted_objects.get('amount_scaler')
            if scaler:
                df['amount_scaled'] = scaler.transform(df[['transaction_amount']]).flatten()
        
        return df
    
    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer temporal features."""
        
        # Hour of day (fraud patterns)
        df['hour'] = df['transaction_hour']
        df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Day of week (if available)
        if 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            df['day_of_week'] = df['transaction_date'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def _engineer_customer_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer customer-related features."""
        
        # Age-based features
        df['age_group'] = pd.cut(df['customer_age'], 
                                bins=[0, 18, 25, 35, 50, 100], 
                                labels=['teen', 'young', 'adult', 'middle', 'senior'])
        df['age_group_encoded'] = pd.Categorical(df['age_group']).codes
        
        # Account age features
        df['account_age_days_log'] = np.log1p(df['account_age_days'])
        df['new_account'] = (df['account_age_days'] <= 7).astype(int)
        df['established_account'] = (df['account_age_days'] > 30).astype(int)
        
        # Customer location frequency (fit on training data only)
        if is_training:
            location_freq = df['customer_location'].value_counts(normalize=True)
            self.fitted_objects['location_freq'] = location_freq
        else:
            location_freq = self.fitted_objects.get('location_freq', pd.Series())
        
        df['location_freq'] = df['customer_location'].map(location_freq).fillna(0)
        
        return df
    
    def _engineer_categorical_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer categorical features."""
        
        categorical_cols = ['payment_method', 'product_category', 'device_used']
        
        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col])
                    self.fitted_objects[f'{col}_encoder'] = le
                else:
                    le = self.fitted_objects.get(f'{col}_encoder')
                    if le:
                        # Handle unseen categories
                        df[f'{col}_encoded'] = df[col].map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        return df
    
    def _engineer_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer interaction features."""
        
        # Amount × Quantity interaction
        df['amount_quantity_interaction'] = df['transaction_amount'] * df['quantity']
        
        # Age × Account age interaction
        df['age_account_interaction'] = df['customer_age'] * df['account_age_days']
        
        # Amount × Hour interaction (fraud patterns)
        df['amount_hour_interaction'] = df['transaction_amount'] * df['hour']
        
        return df
    
    def _engineer_risk_flags(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer risk flags based on domain knowledge."""
        
        # High quantity flag
        if is_training:
            quantity_90 = df['quantity'].quantile(0.90)
            quantity_95 = df['quantity'].quantile(0.95)
            self.fitted_objects['quantity_90'] = quantity_90
            self.fitted_objects['quantity_95'] = quantity_95
        else:
            quantity_90 = self.fitted_objects.get('quantity_90', df['quantity'].quantile(0.90))
            quantity_95 = self.fitted_objects.get('quantity_95', df['quantity'].quantile(0.95))
        
        df['high_quantity_90'] = (df['quantity'] > quantity_90).astype(int)
        df['high_quantity_95'] = (df['quantity'] > quantity_95).astype(int)
        
        # Young customer flag
        df['young_customer'] = (df['customer_age'] <= 18).astype(int)
        df['very_young_customer'] = (df['customer_age'] <= 16).astype(int)
        
        # High amount + late night (high risk combination)
        if is_training:
            amount_90 = df['transaction_amount'].quantile(0.9)
            self.fitted_objects['amount_90_threshold'] = amount_90
        else:
            amount_90 = self.fitted_objects.get('amount_90_threshold', df['transaction_amount'].quantile(0.9))
        
        df['high_risk_combination'] = (
            (df['transaction_amount'] > amount_90) & 
            (df['is_late_night'] == 1)
        ).astype(int)
        
        # New account + high amount
        if is_training:
            amount_80 = df['transaction_amount'].quantile(0.8)
            self.fitted_objects['amount_80_threshold'] = amount_80
        else:
            amount_80 = self.fitted_objects.get('amount_80_threshold', df['transaction_amount'].quantile(0.8))
        
        df['new_account_high_amount'] = (
            (df['new_account'] == 1) & 
            (df['transaction_amount'] > amount_80)
        ).astype(int)
        
        # Additional risk combinations
        df['high_amount_young_customer'] = (
            (df['transaction_amount'] > amount_90) & 
            (df['customer_age'] <= 25)
        ).astype(int)
        
        df['new_account_late_night'] = (
            (df['new_account'] == 1) & 
            (df['is_late_night'] == 1)
        ).astype(int)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names."""
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted first")
        
        # Return the most important features based on EDA (focusing on discriminative features)
        return [
            'amount_log', 'amount_per_item', 'amount_scaled',
            'high_amount_90', 'high_amount_95',
            'hour', 'is_late_night',
            'age_group_encoded', 'account_age_days_log',
            'new_account', 'location_freq',
            'payment_method_encoded', 'product_category_encoded',
            'amount_quantity_interaction', 'amount_hour_interaction',
            'high_quantity_90', 'young_customer',
            'high_risk_combination', 'new_account_high_amount'
        ]
    
    def save_fitted_objects(self, filepath: str):
        """Save fitted objects to disk."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.fitted_objects, f)
        logger.info(f"Fitted objects saved to: {filepath}")
    
    def load_fitted_objects(self, filepath: str):
        """Load fitted objects from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            self.fitted_objects = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Fitted objects loaded from: {filepath}") 