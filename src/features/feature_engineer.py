"""
Enhanced Feature Engineering for Fraud Detection
Comprehensive behavioral, temporal, and statistical features.
"""

import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Enhanced feature engineering with comprehensive fraud detection features."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature engineer with configuration."""
        self.config = config
        self.is_fitted = False
        
        # Fitted objects (will be set during fit)
        self.amount_scaler = None
        self.quantity_scaler = None
        self.label_encoders = {}
        self.percentile_thresholds = {}
        self.customer_stats = {}
        self.location_stats = {}
        self.payment_method_stats = {}
        self.product_category_stats = {}
        self.device_stats = {}
        self.temporal_stats = {}
        
        # Feature selection flags
        self.use_amount_features = self.config.get('use_amount_features', True)
        self.use_temporal_features = self.config.get('use_temporal_features', True)
        self.use_customer_features = self.config.get('use_customer_features', True)
        self.use_risk_flags = self.config.get('use_risk_flags', True)
        self.use_behavioral_features = self.config.get('use_behavioral_features', True)
        self.use_velocity_features = self.config.get('use_velocity_features', True)
        self.use_interaction_features = self.config.get('use_interaction_features', True)
    
    def fit_transform(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit on training data and transform both train and test data."""
        
        logger.info("Starting enhanced feature engineering fit_transform...")
        
        # Fit on training data only
        df_train_features = self._fit_and_transform(df_train, is_training=True)
        
        # Transform test data using fitted parameters
        df_test_features = self._transform(df_test, is_training=False)
        
        self.is_fitted = True
        
        logger.info(f"Enhanced feature engineering completed. Train: {df_train_features.shape}, Test: {df_test_features.shape}")
        
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
            df_features = self._engineer_temporal_features(df_features, is_training)
        
        if self.use_customer_features:
            df_features = self._engineer_customer_features(df_features, is_training)
        
        if self.use_behavioral_features:
            df_features = self._engineer_behavioral_features(df_features, is_training)
        
        if self.use_velocity_features:
            df_features = self._engineer_velocity_features(df_features, is_training)
        
        if self.use_interaction_features:
            df_features = self._engineer_interaction_features(df_features, is_training)
        
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
            df_features = self._apply_temporal_features(df_features)
        
        if self.use_customer_features:
            df_features = self._apply_customer_features(df_features)
        
        if self.use_behavioral_features:
            df_features = self._apply_behavioral_features(df_features)
        
        if self.use_velocity_features:
            df_features = self._apply_velocity_features(df_features)
        
        if self.use_interaction_features:
            df_features = self._apply_interaction_features(df_features)
        
        if self.use_risk_flags:
            df_features = self._apply_risk_flags(df_features)
        
        # Ensure all features are numeric
        df_features = self._ensure_numeric_features(df_features, is_training)
        
        return df_features
    
    def _engineer_amount_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer comprehensive amount-related features."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset - use 'amount' column
            amount_col = 'amount'
            
            # Basic amount features
            df['amount_log'] = np.log1p(df[amount_col])
            df['amount_sqrt'] = np.sqrt(df[amount_col])
            
            # Robust scaling (fit on training data only)
            if is_training:
                self.amount_scaler = RobustScaler()
                df['amount_scaled'] = self.amount_scaler.fit_transform(df[[amount_col]]).flatten()
            else:
                df['amount_scaled'] = self.amount_scaler.transform(df[[amount_col]]).flatten()
            
            # Amount percentiles and thresholds (fit on training data only)
            if is_training:
                for percentile in [50, 75, 90, 95, 99]:
                    self.percentile_thresholds[f'amount_{percentile}'] = np.percentile(df[amount_col], percentile)
            
            # Amount threshold flags
            for percentile in [50, 75, 90, 95, 99]:
                df[f'amount_above_{percentile}'] = (df[amount_col] > self.percentile_thresholds[f'amount_{percentile}']).astype(int)
            
            # Amount distribution features
            df['amount_ratio_to_median'] = df[amount_col] / (self.percentile_thresholds['amount_50'] + 1e-8)
            df['amount_ratio_to_95th'] = df[amount_col] / (self.percentile_thresholds['amount_95'] + 1e-8)
            
            # Amount volatility features
            df['amount_volatility'] = np.abs(df[amount_col] - self.percentile_thresholds['amount_50']) / (self.percentile_thresholds['amount_50'] + 1e-8)
            
            # Add V-features statistics
            existing_v_columns = [col for col in v_columns if col in df.columns]
            if existing_v_columns:
                logger.info(f"Found {len(existing_v_columns)} V-features from credit card dataset")
                
                # Add aggregate features for V-features
                df['v_features_mean'] = df[existing_v_columns].mean(axis=1)
                df['v_features_std'] = df[existing_v_columns].std(axis=1)
                df['v_features_max'] = df[existing_v_columns].max(axis=1)
                df['v_features_min'] = df[existing_v_columns].min(axis=1)
                df['v_features_range'] = df['v_features_max'] - df['v_features_min']
                df['v_features_abs_mean'] = df[existing_v_columns].abs().mean(axis=1)
                
                # Add interaction features between amount and V-features
                df['amount_v_mean_interaction'] = df[amount_col] * df['v_features_mean']
                df['amount_v_std_interaction'] = df[amount_col] * df['v_features_std']
                
        else:
            # E-commerce dataset - original logic
            # Basic amount features
            df['amount_log'] = np.log1p(df['transaction_amount'])
            df['amount_sqrt'] = np.sqrt(df['transaction_amount'])
            df['amount_per_item'] = df['transaction_amount'] / (df['quantity'] + 1)
            df['amount_per_day'] = df['transaction_amount'] / (df['account_age_days'] + 1)
            
            # Robust scaling (fit on training data only)
            if is_training:
                self.amount_scaler = RobustScaler()
                df['amount_scaled'] = self.amount_scaler.fit_transform(df[['transaction_amount']]).flatten()
            else:
                df['amount_scaled'] = self.amount_scaler.transform(df[['transaction_amount']]).flatten()
            
            # Amount percentiles and thresholds (fit on training data only)
            if is_training:
                for percentile in [50, 75, 90, 95, 99]:
                    self.percentile_thresholds[f'amount_{percentile}'] = np.percentile(df['transaction_amount'], percentile)
            
            # Amount threshold flags
            for percentile in [50, 75, 90, 95, 99]:
                df[f'amount_above_{percentile}'] = (df['transaction_amount'] > self.percentile_thresholds[f'amount_{percentile}']).astype(int)
            
            # Amount distribution features
            df['amount_ratio_to_median'] = df['transaction_amount'] / (self.percentile_thresholds['amount_50'] + 1e-8)
            df['amount_ratio_to_95th'] = df['transaction_amount'] / (self.percentile_thresholds['amount_95'] + 1e-8)
            
            # Amount volatility features
            df['amount_volatility'] = np.abs(df['transaction_amount'] - self.percentile_thresholds['amount_50']) / (self.percentile_thresholds['amount_50'] + 1e-8)
        
        return df
    
    def _apply_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted amount features to test data."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset - use 'amount' column
            amount_col = 'amount'
            
            # Basic amount features
            df['amount_log'] = np.log1p(df[amount_col])
            df['amount_sqrt'] = np.sqrt(df[amount_col])
            
            # Robust scaling (using fitted scaler)
            df['amount_scaled'] = self.amount_scaler.transform(df[[amount_col]]).flatten()
            
            # Amount threshold flags (using fitted thresholds)
            for percentile in [50, 75, 90, 95, 99]:
                df[f'amount_above_{percentile}'] = (df[amount_col] > self.percentile_thresholds[f'amount_{percentile}']).astype(int)
            
            # Amount distribution features
            df['amount_ratio_to_median'] = df[amount_col] / (self.percentile_thresholds['amount_50'] + 1e-8)
            df['amount_ratio_to_95th'] = df[amount_col] / (self.percentile_thresholds['amount_95'] + 1e-8)
            
            # Amount volatility features
            df['amount_volatility'] = np.abs(df[amount_col] - self.percentile_thresholds['amount_50']) / (self.percentile_thresholds['amount_50'] + 1e-8)
            
            # Add V-features statistics
            existing_v_columns = [col for col in v_columns if col in df.columns]
            if existing_v_columns:
                # Add aggregate features for V-features
                df['v_features_mean'] = df[existing_v_columns].mean(axis=1)
                df['v_features_std'] = df[existing_v_columns].std(axis=1)
                df['v_features_max'] = df[existing_v_columns].max(axis=1)
                df['v_features_min'] = df[existing_v_columns].min(axis=1)
                df['v_features_range'] = df['v_features_max'] - df['v_features_min']
                df['v_features_abs_mean'] = df[existing_v_columns].abs().mean(axis=1)
                
                # Add interaction features between amount and V-features
                df['amount_v_mean_interaction'] = df[amount_col] * df['v_features_mean']
                df['amount_v_std_interaction'] = df[amount_col] * df['v_features_std']
                
        else:
            # E-commerce dataset - original logic
            # Basic amount features
            df['amount_log'] = np.log1p(df['transaction_amount'])
            df['amount_sqrt'] = np.sqrt(df['transaction_amount'])
            df['amount_per_item'] = df['transaction_amount'] / (df['quantity'] + 1)
            df['amount_per_day'] = df['transaction_amount'] / (df['account_age_days'] + 1)
            
            # Robust scaling (using fitted scaler)
            df['amount_scaled'] = self.amount_scaler.transform(df[['transaction_amount']]).flatten()
            
            # Amount threshold flags (using fitted thresholds)
            for percentile in [50, 75, 90, 95, 99]:
                df[f'amount_above_{percentile}'] = (df['transaction_amount'] > self.percentile_thresholds[f'amount_{percentile}']).astype(int)
            
            # Amount distribution features
            df['amount_ratio_to_median'] = df['transaction_amount'] / (self.percentile_thresholds['amount_50'] + 1e-8)
            df['amount_ratio_to_95th'] = df['transaction_amount'] / (self.percentile_thresholds['amount_95'] + 1e-8)
            
            # Amount volatility features
            df['amount_volatility'] = np.abs(df['transaction_amount'] - self.percentile_thresholds['amount_50']) / (self.percentile_thresholds['amount_50'] + 1e-8)
        
        return df
    
    def _engineer_temporal_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer comprehensive temporal features."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset - use 'time' column and derived transaction_date
            if 'transaction_date' not in df.columns and 'time' in df.columns:
                # Create transaction_date from time column if not already done
                df['transaction_date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df['time'], unit='s')
            
            # Basic temporal features (skip if already created by data loader)
            if 'hour' not in df.columns:
                df['hour'] = df['transaction_date'].dt.hour
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['transaction_date'].dt.dayofweek
            if 'day_of_month' not in df.columns:
                df['day_of_month'] = df['transaction_date'].dt.day
            if 'month' not in df.columns:
                df['month'] = df['transaction_date'].dt.month
            if 'quarter' not in df.columns:
                df['quarter'] = df['transaction_date'].dt.quarter
            if 'is_weekend' not in df.columns:
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Time-based flags
            if 'is_late_night' not in df.columns:
                df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
            if 'is_business_hours' not in df.columns:
                df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            if 'is_rush_hour' not in df.columns:
                df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
                df['is_rush_hour'] = df['is_rush_hour'].astype(int)
            
            # Cyclical encoding for hour and day (skip if already created)
            if 'hour_sin' not in df.columns:
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            if 'hour_cos' not in df.columns:
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            if 'day_sin' not in df.columns:
                df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            if 'day_cos' not in df.columns:
                df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            if 'month_sin' not in df.columns and 'month' in df.columns:
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            if 'month_cos' not in df.columns and 'month' in df.columns:
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Time-based risk periods (skip if already created)
            if 'is_high_risk_hour' not in df.columns:
                df['is_high_risk_hour'] = ((df['hour'] >= 0) & (df['hour'] <= 4)).astype(int)
            if 'is_holiday_period' not in df.columns and 'month' in df.columns:
                df['is_holiday_period'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int)
            
            # Temporal statistics (fit on training data only)
            if is_training:
                self.temporal_stats['hour_mean'] = df['hour'].mean()
                self.temporal_stats['hour_std'] = df['hour'].std()
                self.temporal_stats['day_weekend_ratio'] = df['is_weekend'].mean()
            
            # Temporal deviation features
            if 'hour_deviation' not in df.columns:
                df['hour_deviation'] = np.abs(df['hour'] - self.temporal_stats['hour_mean']) / (self.temporal_stats['hour_std'] + 1e-8)
        
        else:
            # E-commerce dataset - original logic
            # Convert timestamp to datetime
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            
            # Basic temporal features
            df['hour'] = df['transaction_date'].dt.hour
            df['day_of_week'] = df['transaction_date'].dt.dayofweek
            df['day_of_month'] = df['transaction_date'].dt.day
            df['month'] = df['transaction_date'].dt.month
            df['quarter'] = df['transaction_date'].dt.quarter
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Time-based flags
            df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
            df['is_rush_hour'] = df['is_rush_hour'].astype(int)
            
            # Cyclical encoding for hour and day
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Time-based risk periods
            df['is_high_risk_hour'] = ((df['hour'] >= 0) & (df['hour'] <= 4)).astype(int)
            df['is_holiday_period'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int)
            
            # Temporal statistics (fit on training data only)
            if is_training:
                self.temporal_stats['hour_mean'] = df['hour'].mean()
                self.temporal_stats['hour_std'] = df['hour'].std()
                self.temporal_stats['day_weekend_ratio'] = df['is_weekend'].mean()
            
            # Temporal deviation features
            df['hour_deviation'] = np.abs(df['hour'] - self.temporal_stats['hour_mean']) / (self.temporal_stats['hour_std'] + 1e-8)
        
        return df
    
    def _apply_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted temporal features to test data."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset - use 'time' column and derived transaction_date
            if 'transaction_date' not in df.columns and 'time' in df.columns:
                # Create transaction_date from time column if not already done
                df['transaction_date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df['time'], unit='s')
            
            # Basic temporal features
            df['hour'] = df['transaction_date'].dt.hour
            df['day_of_week'] = df['transaction_date'].dt.dayofweek
            df['day_of_month'] = df['transaction_date'].dt.day
            df['month'] = df['transaction_date'].dt.month
            df['quarter'] = df['transaction_date'].dt.quarter
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Time-based flags
            df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
            df['is_rush_hour'] = df['is_rush_hour'].astype(int)
            
            # Cyclical encoding for hour and day
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Time-based risk periods
            df['is_high_risk_hour'] = ((df['hour'] >= 0) & (df['hour'] <= 4)).astype(int)
            df['is_holiday_period'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int)
            
            # Temporal deviation features (using fitted statistics)
            df['hour_deviation'] = np.abs(df['hour'] - self.temporal_stats['hour_mean']) / (self.temporal_stats['hour_std'] + 1e-8)
            
        else:
            # E-commerce dataset - original logic
            # Convert timestamp to datetime
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            
            # Basic temporal features
            df['hour'] = df['transaction_date'].dt.hour
            df['day_of_week'] = df['transaction_date'].dt.dayofweek
            df['day_of_month'] = df['transaction_date'].dt.day
            df['month'] = df['transaction_date'].dt.month
            df['quarter'] = df['transaction_date'].dt.quarter
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Time-based flags
            df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
            df['is_rush_hour'] = df['is_rush_hour'].astype(int)
            
            # Cyclical encoding for hour and day
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Time-based risk periods
            df['is_high_risk_hour'] = ((df['hour'] >= 0) & (df['hour'] <= 4)).astype(int)
            df['is_holiday_period'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int)
            
            # Temporal deviation features (using fitted statistics)
            df['hour_deviation'] = np.abs(df['hour'] - self.temporal_stats['hour_mean']) / (self.temporal_stats['hour_std'] + 1e-8)
        
        return df
    
    def _engineer_customer_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer comprehensive customer-related features."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset doesn't have customer features, skip this section
            logger.info("Credit card dataset detected - skipping customer features")
            return df
        
        # E-commerce dataset - original logic
        # Age-based features
        if 'customer_age' in df.columns:
            df['age_group'] = pd.cut(df['customer_age'], 
                                    bins=[0, 18, 25, 35, 50, 65, 100], 
                                    labels=['teen', 'young', 'adult', 'middle', 'senior', 'elderly'])
            
            if is_training:
                self.label_encoders['age_group'] = LabelEncoder()
                df['age_group_encoded'] = self.label_encoders['age_group'].fit_transform(df['age_group'])
            else:
                df['age_group_encoded'] = self.label_encoders['age_group'].transform(df['age_group'])
            
            # Age risk features
            df['young_customer'] = (df['customer_age'] <= 18).astype(int)
            df['senior_customer'] = (df['customer_age'] >= 65).astype(int)
            df['age_risk'] = np.where(df['customer_age'] <= 18, 2, np.where(df['customer_age'] >= 65, 1, 0))
        
        # Account age features
        if 'account_age_days' in df.columns:
            df['account_age_days_log'] = np.log1p(df['account_age_days'])
            df['account_age_weeks'] = df['account_age_days'] / 7
            df['account_age_months'] = df['account_age_days'] / 30
            
            df['new_account'] = (df['account_age_days'] <= 7).astype(int)
            df['recent_account'] = (df['account_age_days'] <= 30).astype(int)
            df['established_account'] = (df['account_age_days'] > 90).astype(int)
            df['old_account'] = (df['account_age_days'] > 365).astype(int)
            
            # Account age risk
            df['account_age_risk'] = np.where(df['account_age_days'] <= 7, 3, 
                                             np.where(df['account_age_days'] <= 30, 2, 
                                                     np.where(df['account_age_days'] <= 90, 1, 0)))
        
        # Location-based features (fit on training data only)
        if 'customer_location' in df.columns:
            if is_training:
                location_counts = df['customer_location'].value_counts()
                self.location_stats['location_counts'] = location_counts.to_dict()
                self.location_stats['location_risk'] = (location_counts / len(df)).to_dict()
            
            df['location_freq'] = df['customer_location'].map(self.location_stats['location_counts']).fillna(0)
            df['location_risk_score'] = df['customer_location'].map(self.location_stats['location_risk']).fillna(0)
            df['rare_location'] = (df['location_freq'] <= 5).astype(int)
        
        # Categorical encodings
        categorical_cols = ['payment_method', 'product_category', 'device_used']
        
        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                    
                    # Calculate risk scores for categories
                    fraud_rates = df.groupby(col)['is_fraudulent'].mean()
                    self.customer_stats[f'{col}_fraud_rate'] = fraud_rates.to_dict()
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                
                # Add fraud rate features
                df[f'{col}_fraud_rate'] = df[col].map(self.customer_stats[f'{col}_fraud_rate']).fillna(0)
        
        return df
    
    def _apply_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted customer features to test data."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset doesn't have customer features, skip this section
            return df
        
        # E-commerce dataset - original logic
        # Age-based features
        if 'customer_age' in df.columns:
            df['age_group'] = pd.cut(df['customer_age'], 
                                    bins=[0, 18, 25, 35, 50, 65, 100], 
                                    labels=['teen', 'young', 'adult', 'middle', 'senior', 'elderly'])
            df['age_group_encoded'] = self.label_encoders['age_group'].transform(df['age_group'])
            
            # Age risk features
            df['young_customer'] = (df['customer_age'] <= 18).astype(int)
            df['senior_customer'] = (df['customer_age'] >= 65).astype(int)
            df['age_risk'] = np.where(df['customer_age'] <= 18, 2, np.where(df['customer_age'] >= 65, 1, 0))
        
        # Account age features
        if 'account_age_days' in df.columns:
            df['account_age_days_log'] = np.log1p(df['account_age_days'])
            df['account_age_weeks'] = df['account_age_days'] / 7
            df['account_age_months'] = df['account_age_days'] / 30
            
            df['new_account'] = (df['account_age_days'] <= 7).astype(int)
            df['recent_account'] = (df['account_age_days'] <= 30).astype(int)
            df['established_account'] = (df['account_age_days'] > 90).astype(int)
            df['old_account'] = (df['account_age_days'] > 365).astype(int)
            
            # Account age risk
            df['account_age_risk'] = np.where(df['account_age_days'] <= 7, 3, 
                                             np.where(df['account_age_days'] <= 30, 2, 
                                                     np.where(df['account_age_days'] <= 90, 1, 0)))
        
        # Location-based features
        if 'customer_location' in df.columns:
            df['location_freq'] = df['customer_location'].map(self.location_stats['location_counts']).fillna(0)
            df['location_risk_score'] = df['customer_location'].map(self.location_stats['location_risk']).fillna(0)
            df['rare_location'] = (df['location_freq'] <= 5).astype(int)
        
        # Categorical encodings
        categorical_cols = ['payment_method', 'product_category', 'device_used']
        
        for col in categorical_cols:
            if col in df.columns:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                df[f'{col}_fraud_rate'] = df[col].map(self.customer_stats[f'{col}_fraud_rate']).fillna(0)
        
        return df
    
    def _engineer_behavioral_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer behavioral pattern features."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset doesn't have quantity features, skip this section
            logger.info("Credit card dataset detected - skipping behavioral features")
            return df
        
        # E-commerce dataset - original logic
        # Quantity-based features
        if 'quantity' in df.columns and 'account_age_days' in df.columns:
            df['quantity_log'] = np.log1p(df['quantity'])
            df['quantity_per_day'] = df['quantity'] / (df['account_age_days'] + 1)
            
            # Quantity scaling
            if is_training:
                self.quantity_scaler = RobustScaler()
                df['quantity_scaled'] = self.quantity_scaler.fit_transform(df[['quantity']]).flatten()
            else:
                df['quantity_scaled'] = self.quantity_scaler.transform(df[['quantity']]).flatten()
            
            # Quantity thresholds
            if is_training:
                for percentile in [50, 75, 90, 95, 99]:
                    self.percentile_thresholds[f'quantity_{percentile}'] = np.percentile(df['quantity'], percentile)
            
            for percentile in [75, 90, 95, 99]:
                df[f'quantity_above_{percentile}'] = (df['quantity'] > self.percentile_thresholds[f'quantity_{percentile}']).astype(int)
            
            # Behavioral patterns
            if 'transaction_amount' in df.columns:
                df['high_value_low_quantity'] = ((df['transaction_amount'] > self.percentile_thresholds['amount_90']) & 
                                                (df['quantity'] <= 1)).astype(int)
                
                df['low_value_high_quantity'] = ((df['transaction_amount'] < self.percentile_thresholds['amount_50']) & 
                                                (df['quantity'] > self.percentile_thresholds['quantity_90'])).astype(int)
                
                # Purchase behavior features
                df['avg_amount_per_item'] = df['transaction_amount'] / (df['quantity'] + 1)
                df['purchase_efficiency'] = df['transaction_amount'] / (df['quantity'] * df['account_age_days'] + 1)
                
                # Behavioral risk indicators
                df['unusual_purchase_pattern'] = ((df['high_value_low_quantity'] == 1) | 
                                                 (df['low_value_high_quantity'] == 1)).astype(int)
        
        return df
    
    def _apply_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted behavioral features to test data."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset doesn't have quantity features, skip this section
            return df
        
        # E-commerce dataset - original logic
        # Quantity-based features
        if 'quantity' in df.columns and 'account_age_days' in df.columns:
            df['quantity_log'] = np.log1p(df['quantity'])
            df['quantity_per_day'] = df['quantity'] / (df['account_age_days'] + 1)
            
            # Quantity scaling
            df['quantity_scaled'] = self.quantity_scaler.transform(df[['quantity']]).flatten()
            
            # Quantity thresholds
            for percentile in [75, 90, 95, 99]:
                df[f'quantity_above_{percentile}'] = (df['quantity'] > self.percentile_thresholds[f'quantity_{percentile}']).astype(int)
            
            # Behavioral patterns
            if 'transaction_amount' in df.columns:
                df['high_value_low_quantity'] = ((df['transaction_amount'] > self.percentile_thresholds['amount_90']) & 
                                                (df['quantity'] <= 1)).astype(int)
                
                df['low_value_high_quantity'] = ((df['transaction_amount'] < self.percentile_thresholds['amount_50']) & 
                                                (df['quantity'] > self.percentile_thresholds['quantity_90'])).astype(int)
                
                # Purchase behavior features
                df['avg_amount_per_item'] = df['transaction_amount'] / (df['quantity'] + 1)
                df['purchase_efficiency'] = df['transaction_amount'] / (df['quantity'] * df['account_age_days'] + 1)
                
                # Behavioral risk indicators
                df['unusual_purchase_pattern'] = ((df['high_value_low_quantity'] == 1) | 
                                                 (df['low_value_high_quantity'] == 1)).astype(int)
        
        return df
    
    def _engineer_velocity_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer velocity-based features for fraud detection."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset doesn't have account age features, skip this section
            logger.info("Credit card dataset detected - skipping velocity features")
            return df
        
        # E-commerce dataset - original logic
        if 'account_age_days' in df.columns and 'transaction_amount' in df.columns and 'quantity' in df.columns:
            # Transaction velocity
            df['transaction_velocity'] = 1 / (df['account_age_days'] + 1)
            df['amount_velocity'] = df['transaction_amount'] / (df['account_age_days'] + 1)
            df['quantity_velocity'] = df['quantity'] / (df['account_age_days'] + 1)
            
            # Velocity ratios
            df['amount_per_transaction'] = df['transaction_amount']
            df['quantity_per_transaction'] = df['quantity']
            
            # High velocity indicators
            df['high_amount_velocity'] = (df['amount_velocity'] > self.percentile_thresholds['amount_95']).astype(int)
            df['high_quantity_velocity'] = (df['quantity_velocity'] > self.percentile_thresholds['quantity_95']).astype(int)
            
            # Velocity risk scoring
            df['velocity_risk_score'] = (df['high_amount_velocity'] * 2 + 
                                       df['high_quantity_velocity'] * 1.5 + 
                                       df['new_account'] * 2)
        
        return df
    
    def _apply_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted velocity features to test data."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset doesn't have account age features, skip this section
            return df
        
        # E-commerce dataset - original logic
        if 'account_age_days' in df.columns and 'transaction_amount' in df.columns and 'quantity' in df.columns:
            # Transaction velocity
            df['transaction_velocity'] = 1 / (df['account_age_days'] + 1)
            df['amount_velocity'] = df['transaction_amount'] / (df['account_age_days'] + 1)
            df['quantity_velocity'] = df['quantity'] / (df['account_age_days'] + 1)
            
            # Velocity ratios
            df['amount_per_transaction'] = df['transaction_amount']
            df['quantity_per_transaction'] = df['quantity']
            
            # High velocity indicators
            df['high_amount_velocity'] = (df['amount_velocity'] > self.percentile_thresholds['amount_95']).astype(int)
            df['high_quantity_velocity'] = (df['quantity_velocity'] > self.percentile_thresholds['quantity_95']).astype(int)
            
            # Velocity risk scoring
            df['velocity_risk_score'] = (df['high_amount_velocity'] * 2 + 
                                       df['high_quantity_velocity'] * 1.5 + 
                                       df['new_account'] * 2)
        
        return df
    
    def _engineer_interaction_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer interaction features between different variables."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset - focus on V-features and amount interactions
            # Amount-Time interactions
            if 'amount' in df.columns and 'hour' in df.columns:
                df['amount_hour_interaction'] = df['amount'] * df['hour']
                df['amount_day_interaction'] = df['amount'] * df['day_of_week']
            
            # V-features interactions with amount
            if 'amount' in df.columns:
                existing_v_columns = [col for col in v_columns if col in df.columns]
                for i, v_col in enumerate(existing_v_columns[:10]):  # Use first 10 V-features
                    df[f'amount_v{i+1}_interaction'] = df['amount'] * df[v_col]
            
            # V-features interactions with time
            if 'hour' in df.columns:
                for i, v_col in enumerate(existing_v_columns[:5]):  # Use first 5 V-features
                    df[f'hour_v{i+1}_interaction'] = df['hour'] * df[v_col]
            
            return df
        
        # E-commerce dataset - original logic
        # Amount-Quantity interactions
        if 'transaction_amount' in df.columns and 'quantity' in df.columns:
            df['amount_quantity_interaction'] = df['transaction_amount'] * df['quantity']
            df['amount_quantity_ratio'] = df['transaction_amount'] / (df['quantity'] + 1)
        
        # Age-Account interactions
        if 'customer_age' in df.columns and 'account_age_days' in df.columns:
            df['age_account_interaction'] = df['customer_age'] * df['account_age_days']
            df['age_account_ratio'] = df['customer_age'] / (df['account_age_days'] + 1)
        
        # Amount-Time interactions
        if 'transaction_amount' in df.columns and 'hour' in df.columns:
            df['amount_hour_interaction'] = df['transaction_amount'] * df['hour']
            df['amount_day_interaction'] = df['transaction_amount'] * df['day_of_week']
        
        # Risk-Interaction features
        if 'transaction_amount' in df.columns and 'age_risk' in df.columns:
            df['amount_risk_interaction'] = df['transaction_amount'] * df['age_risk']
        if 'quantity' in df.columns and 'account_age_risk' in df.columns:
            df['quantity_risk_interaction'] = df['quantity'] * df['account_age_risk']
        
        # Location-Device interactions
        if 'location_freq' in df.columns and 'device_used_encoded' in df.columns:
            df['location_device_interaction'] = df['location_freq'] * df['device_used_encoded']
        
        # Payment-Product interactions
        if 'payment_method_encoded' in df.columns and 'product_category_encoded' in df.columns:
            df['payment_product_interaction'] = df['payment_method_encoded'] * df['product_category_encoded']
        
        # Complex interactions
        if all(col in df.columns for col in ['customer_age', 'transaction_amount', 'hour']):
            df['age_amount_time_interaction'] = df['customer_age'] * df['transaction_amount'] * df['hour']
        if all(col in df.columns for col in ['account_age_days', 'quantity', 'location_freq']):
            df['account_quantity_location_interaction'] = df['account_age_days'] * df['quantity'] * df['location_freq']
        
        return df
    
    def _apply_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted interaction features to test data."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset - focus on V-features and amount interactions
            # Amount-Time interactions
            if 'amount' in df.columns and 'hour' in df.columns:
                df['amount_hour_interaction'] = df['amount'] * df['hour']
                df['amount_day_interaction'] = df['amount'] * df['day_of_week']
            
            # V-features interactions with amount
            if 'amount' in df.columns:
                existing_v_columns = [col for col in v_columns if col in df.columns]
                for i, v_col in enumerate(existing_v_columns[:10]):  # Use first 10 V-features
                    df[f'amount_v{i+1}_interaction'] = df['amount'] * df[v_col]
            
            # V-features interactions with time
            if 'hour' in df.columns:
                for i, v_col in enumerate(existing_v_columns[:5]):  # Use first 5 V-features
                    df[f'hour_v{i+1}_interaction'] = df['hour'] * df[v_col]
            
            return df
        
        # E-commerce dataset - original logic
        # Amount-Quantity interactions
        if 'transaction_amount' in df.columns and 'quantity' in df.columns:
            df['amount_quantity_interaction'] = df['transaction_amount'] * df['quantity']
            df['amount_quantity_ratio'] = df['transaction_amount'] / (df['quantity'] + 1)
        
        # Age-Account interactions
        if 'customer_age' in df.columns and 'account_age_days' in df.columns:
            df['age_account_interaction'] = df['customer_age'] * df['account_age_days']
            df['age_account_ratio'] = df['customer_age'] / (df['account_age_days'] + 1)
        
        # Amount-Time interactions
        if 'transaction_amount' in df.columns and 'hour' in df.columns:
            df['amount_hour_interaction'] = df['transaction_amount'] * df['hour']
            df['amount_day_interaction'] = df['transaction_amount'] * df['day_of_week']
        
        # Risk-Interaction features
        if 'transaction_amount' in df.columns and 'age_risk' in df.columns:
            df['amount_risk_interaction'] = df['transaction_amount'] * df['age_risk']
        if 'quantity' in df.columns and 'account_age_risk' in df.columns:
            df['quantity_risk_interaction'] = df['quantity'] * df['account_age_risk']
        
        # Location-Device interactions
        if 'location_freq' in df.columns and 'device_used_encoded' in df.columns:
            df['location_device_interaction'] = df['location_freq'] * df['device_used_encoded']
        
        # Payment-Product interactions
        if 'payment_method_encoded' in df.columns and 'product_category_encoded' in df.columns:
            df['payment_product_interaction'] = df['payment_method_encoded'] * df['product_category_encoded']
        
        # Complex interactions
        if all(col in df.columns for col in ['customer_age', 'transaction_amount', 'hour']):
            df['age_amount_time_interaction'] = df['customer_age'] * df['transaction_amount'] * df['hour']
        if all(col in df.columns for col in ['account_age_days', 'quantity', 'location_freq']):
            df['account_quantity_location_interaction'] = df['account_age_days'] * df['quantity'] * df['location_freq']
        
        return df
    
    def _engineer_risk_flags(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Engineer comprehensive risk flag features."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset - focus on amount and V-features risk patterns
            amount_col = 'amount'
            
            # High risk combinations
            df['high_risk_combination'] = ((df[amount_col] > self.percentile_thresholds['amount_95']) & 
                                          (df['is_late_night'] == 1)).astype(int)
            
            # Advanced fraud patterns
            df['suspicious_time_pattern'] = ((df['is_high_risk_hour'] == 1) & 
                                            (df[amount_col] > self.percentile_thresholds['amount_75'])).astype(int)
            
            df['weekend_high_value'] = ((df['is_weekend'] == 1) & 
                                       (df[amount_col] > self.percentile_thresholds['amount_90'])).astype(int)
            
            # df['holiday_period_high_value'] = ((df['is_holiday_period'] == 1) & 
            #                                   (df[amount_col] > self.percentile_thresholds['amount_75'])).astype(int)
            
            # Statistical outlier features
            df['amount_z_score'] = (df[amount_col] - self.percentile_thresholds['amount_50']) / (self.percentile_thresholds['amount_75'] - self.percentile_thresholds['amount_50'] + 1e-8)
            
            # Outlier flags
            df['amount_outlier'] = (np.abs(df['amount_z_score']) > 2).astype(int)
            
            # V-features outlier detection
            existing_v_columns = [col for col in v_columns if col in df.columns]
            if existing_v_columns:
                v_data = df[existing_v_columns]
                df['v_features_outlier'] = (np.abs(v_data - v_data.mean()) > 2 * v_data.std()).any(axis=1).astype(int)
                
                # V-features risk patterns
                df['high_v_features_risk'] = ((df['v_features_outlier'] == 1) & 
                                             (df[amount_col] > self.percentile_thresholds['amount_75'])).astype(int)
            
            # Time-based velocity patterns
            df['rush_hour_high_value'] = ((df['is_rush_hour'] == 1) & 
                                         (df[amount_col] > self.percentile_thresholds['amount_90'])).astype(int)
            
            # Advanced risk scoring for credit card dataset
            risk_score = 0
            risk_score += df['amount_above_95'] * 3
            risk_score += df['is_late_night'] * 2
            risk_score += df['amount_outlier'] * 3
            risk_score += df['high_risk_combination'] * 4
            risk_score += df['suspicious_time_pattern'] * 3
            risk_score += df['weekend_high_value'] * 2.5
            # risk_score += df['holiday_period_high_value'] * 2.5  # Skip holiday period risk
            risk_score += df['rush_hour_high_value'] * 2.5
            
            if 'v_features_outlier' in df.columns:
                risk_score += df['v_features_outlier'] * 3
                risk_score += df['high_v_features_risk'] * 4
            
            # Add V-features statistics to risk score
            if 'v_features_mean' in df.columns:
                v_mean_threshold = df['v_features_mean'].quantile(0.95)
                risk_score += (df['v_features_mean'] > v_mean_threshold).astype(int) * 2
            
            if 'v_features_std' in df.columns:
                v_std_threshold = df['v_features_std'].quantile(0.95)
                risk_score += (df['v_features_std'] > v_std_threshold).astype(int) * 2
            
            df['comprehensive_risk_score'] = risk_score
            
            # Risk categories
            df['low_risk'] = (df['comprehensive_risk_score'] <= 2).astype(int)
            df['medium_risk'] = ((df['comprehensive_risk_score'] > 2) & (df['comprehensive_risk_score'] <= 6)).astype(int)
            df['high_risk'] = (df['comprehensive_risk_score'] > 6).astype(int)
            
            return df
        
        # E-commerce dataset - original logic
        # High risk combinations
        df['high_risk_combination'] = ((df['transaction_amount'] > self.percentile_thresholds['amount_95']) & 
                                      (df['is_late_night'] == 1)).astype(int)
        
        df['new_account_high_amount'] = ((df['new_account'] == 1) & 
                                        (df['transaction_amount'] > self.percentile_thresholds['amount_90'])).astype(int)
        
        df['young_customer_high_amount'] = ((df['young_customer'] == 1) & 
                                           (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        df['rare_location_high_amount'] = ((df['rare_location'] == 1) & 
                                          (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Advanced fraud patterns
        df['suspicious_time_pattern'] = ((df['is_high_risk_hour'] == 1) & 
                                        (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        df['weekend_high_value'] = ((df['is_weekend'] == 1) & 
                                   (df['transaction_amount'] > self.percentile_thresholds['amount_90'])).astype(int)
        
        # df['holiday_period_high_value'] = ((df['is_holiday_period'] == 1) & 
        #                                   (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Device and payment method risk patterns
        df['high_risk_device'] = ((df['device_used_fraud_rate'] > 0.1) & 
                                 (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        df['high_risk_payment'] = ((df['payment_method_fraud_rate'] > 0.1) & 
                                  (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Product category risk patterns
        df['high_risk_product'] = ((df['product_category_fraud_rate'] > 0.1) & 
                                  (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Statistical outlier features (create these first)
        df['amount_z_score'] = (df['transaction_amount'] - self.percentile_thresholds['amount_50']) / (self.percentile_thresholds['amount_75'] - self.percentile_thresholds['amount_50'] + 1e-8)
        df['quantity_z_score'] = (df['quantity'] - self.percentile_thresholds['quantity_50']) / (self.percentile_thresholds['quantity_75'] - self.percentile_thresholds['quantity_50'] + 1e-8)
        
        # Outlier flags
        df['amount_outlier'] = (np.abs(df['amount_z_score']) > 2).astype(int)
        df['quantity_outlier'] = (np.abs(df['quantity_z_score']) > 2).astype(int)
        
        # Behavioral anomaly patterns
        df['unusual_amount_pattern'] = ((df['amount_outlier'] == 1) & 
                                       (df['quantity_outlier'] == 1)).astype(int)
        
        df['velocity_anomaly'] = ((df['velocity_risk_score'] > 3) & 
                                 (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Customer profile risk patterns
        df['high_risk_customer_profile'] = ((df['age_risk'] > 0) & 
                                           (df['account_age_risk'] > 1) & 
                                           (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Location-based risk patterns
        df['foreign_high_value'] = ((df['location_risk_score'] > 0.05) & 
                                   (df['transaction_amount'] > self.percentile_thresholds['amount_90'])).astype(int)
        
        # Time-based velocity patterns
        df['rush_hour_high_value'] = ((df['is_rush_hour'] == 1) & 
                                     (df['transaction_amount'] > self.percentile_thresholds['amount_90'])).astype(int)
        
        # Complex fraud patterns
        df['multi_risk_pattern'] = ((df['new_account'] == 1) & 
                                   (df['is_late_night'] == 1) & 
                                   (df['transaction_amount'] > self.percentile_thresholds['amount_90'])).astype(int)
        
        df['sophisticated_fraud_pattern'] = ((df['young_customer'] == 1) & 
                                            (df['rare_location'] == 1) & 
                                            (df['is_high_risk_hour'] == 1) & 
                                            (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Advanced risk scoring
        risk_score = 0
        risk_score += df['amount_above_95'] * 3
        risk_score += df['is_late_night'] * 2
        risk_score += df['new_account'] * 3
        risk_score += df['young_customer'] * 2
        risk_score += df['rare_location'] * 2
        risk_score += df['quantity_above_95'] * 2
        risk_score += df['high_risk_combination'] * 4
        risk_score += df['new_account_high_amount'] * 4
        risk_score += df['young_customer_high_amount'] * 3
        risk_score += df['rare_location_high_amount'] * 3
        risk_score += df['unusual_purchase_pattern'] * 2
        risk_score += df['velocity_risk_score'] * 2
        
        # New advanced patterns
        risk_score += df['suspicious_time_pattern'] * 3
        risk_score += df['weekend_high_value'] * 2.5
        # risk_score += df['holiday_period_high_value'] * 2.5  # Skip holiday period risk
        risk_score += df['high_risk_device'] * 3
        risk_score += df['high_risk_payment'] * 3
        risk_score += df['high_risk_product'] * 3
        risk_score += df['unusual_amount_pattern'] * 4
        risk_score += df['velocity_anomaly'] * 4
        risk_score += df['high_risk_customer_profile'] * 4
        risk_score += df['foreign_high_value'] * 3.5
        risk_score += df['rush_hour_high_value'] * 2.5
        risk_score += df['multi_risk_pattern'] * 5
        risk_score += df['sophisticated_fraud_pattern'] * 6
        
        df['comprehensive_risk_score'] = risk_score
        
        # Risk categories
        df['low_risk'] = (df['comprehensive_risk_score'] <= 2).astype(int)
        df['medium_risk'] = ((df['comprehensive_risk_score'] > 2) & (df['comprehensive_risk_score'] <= 6)).astype(int)
        df['high_risk'] = (df['comprehensive_risk_score'] > 6).astype(int)
        
        # Statistical outlier features
        df['amount_z_score'] = (df['transaction_amount'] - self.percentile_thresholds['amount_50']) / (self.percentile_thresholds['amount_75'] - self.percentile_thresholds['amount_50'] + 1e-8)
        df['quantity_z_score'] = (df['quantity'] - self.percentile_thresholds['quantity_50']) / (self.percentile_thresholds['quantity_75'] - self.percentile_thresholds['quantity_50'] + 1e-8)
        
        # Outlier flags
        df['amount_outlier'] = (np.abs(df['amount_z_score']) > 2).astype(int)
        df['quantity_outlier'] = (np.abs(df['quantity_z_score']) > 2).astype(int)
        
        return df
    
    def _apply_risk_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted risk flags to test data."""
        
        # Check if this is credit card dataset (has V1-V28 features)
        v_columns = [f'v{i}' for i in range(1, 29)]
        is_credit_card_dataset = any(col in df.columns for col in v_columns)
        
        if is_credit_card_dataset:
            # Credit card dataset - focus on amount and V-features risk patterns
            amount_col = 'amount'
            
            # High risk combinations
            df['high_risk_combination'] = ((df[amount_col] > self.percentile_thresholds['amount_95']) & 
                                          (df['is_late_night'] == 1)).astype(int)
            
            # Advanced fraud patterns
            df['suspicious_time_pattern'] = ((df['is_high_risk_hour'] == 1) & 
                                            (df[amount_col] > self.percentile_thresholds['amount_75'])).astype(int)
            
            df['weekend_high_value'] = ((df['is_weekend'] == 1) & 
                                       (df[amount_col] > self.percentile_thresholds['amount_90'])).astype(int)
            
            # df['holiday_period_high_value'] = ((df['is_holiday_period'] == 1) & 
            #                                   (df[amount_col] > self.percentile_thresholds['amount_75'])).astype(int)
            
            # Statistical outlier features
            df['amount_z_score'] = (df[amount_col] - self.percentile_thresholds['amount_50']) / (self.percentile_thresholds['amount_75'] - self.percentile_thresholds['amount_50'] + 1e-8)
            
            # Outlier flags
            df['amount_outlier'] = (np.abs(df['amount_z_score']) > 2).astype(int)
            
            # V-features outlier detection
            existing_v_columns = [col for col in v_columns if col in df.columns]
            if existing_v_columns:
                v_data = df[existing_v_columns]
                df['v_features_outlier'] = (np.abs(v_data - v_data.mean()) > 2 * v_data.std()).any(axis=1).astype(int)
                
                # V-features risk patterns
                df['high_v_features_risk'] = ((df['v_features_outlier'] == 1) & 
                                             (df[amount_col] > self.percentile_thresholds['amount_75'])).astype(int)
            
            # Time-based velocity patterns
            df['rush_hour_high_value'] = ((df['is_rush_hour'] == 1) & 
                                         (df[amount_col] > self.percentile_thresholds['amount_90'])).astype(int)
            
            # Advanced risk scoring for credit card dataset
            risk_score = 0
            risk_score += df['amount_above_95'] * 3
            risk_score += df['is_late_night'] * 2
            risk_score += df['amount_outlier'] * 3
            risk_score += df['high_risk_combination'] * 4
            risk_score += df['suspicious_time_pattern'] * 3
            risk_score += df['weekend_high_value'] * 2.5
            # risk_score += df['holiday_period_high_value'] * 2.5  # Skip holiday period risk
            risk_score += df['rush_hour_high_value'] * 2.5
            
            if 'v_features_outlier' in df.columns:
                risk_score += df['v_features_outlier'] * 3
                risk_score += df['high_v_features_risk'] * 4
            
            # Add V-features statistics to risk score
            if 'v_features_mean' in df.columns:
                v_mean_threshold = df['v_features_mean'].quantile(0.95)
                risk_score += (df['v_features_mean'] > v_mean_threshold).astype(int) * 2
            
            if 'v_features_std' in df.columns:
                v_std_threshold = df['v_features_std'].quantile(0.95)
                risk_score += (df['v_features_std'] > v_std_threshold).astype(int) * 2
            
            df['comprehensive_risk_score'] = risk_score
            
            # Risk categories
            df['low_risk'] = (df['comprehensive_risk_score'] <= 2).astype(int)
            df['medium_risk'] = ((df['comprehensive_risk_score'] > 2) & (df['comprehensive_risk_score'] <= 6)).astype(int)
            df['high_risk'] = (df['comprehensive_risk_score'] > 6).astype(int)
            
            return df
        
        # E-commerce dataset - original logic
        # High risk combinations
        df['high_risk_combination'] = ((df['transaction_amount'] > self.percentile_thresholds['amount_95']) & 
                                      (df['is_late_night'] == 1)).astype(int)
        
        df['new_account_high_amount'] = ((df['new_account'] == 1) & 
                                        (df['transaction_amount'] > self.percentile_thresholds['amount_90'])).astype(int)
        
        df['young_customer_high_amount'] = ((df['young_customer'] == 1) & 
                                           (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        df['rare_location_high_amount'] = ((df['rare_location'] == 1) & 
                                          (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Advanced fraud patterns
        df['suspicious_time_pattern'] = ((df['is_high_risk_hour'] == 1) & 
                                        (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        df['weekend_high_value'] = ((df['is_weekend'] == 1) & 
                                   (df['transaction_amount'] > self.percentile_thresholds['amount_90'])).astype(int)
        
        # df['holiday_period_high_value'] = ((df['is_holiday_period'] == 1) & 
        #                                   (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Device and payment method risk patterns
        df['high_risk_device'] = ((df['device_used_fraud_rate'] > 0.1) & 
                                 (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        df['high_risk_payment'] = ((df['payment_method_fraud_rate'] > 0.1) & 
                                  (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Product category risk patterns
        df['high_risk_product'] = ((df['product_category_fraud_rate'] > 0.1) & 
                                  (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Statistical outlier features (create these first)
        df['amount_z_score'] = (df['transaction_amount'] - self.percentile_thresholds['amount_50']) / (self.percentile_thresholds['amount_75'] - self.percentile_thresholds['amount_50'] + 1e-8)
        df['quantity_z_score'] = (df['quantity'] - self.percentile_thresholds['quantity_50']) / (self.percentile_thresholds['quantity_75'] - self.percentile_thresholds['quantity_50'] + 1e-8)
        
        # Outlier flags
        df['amount_outlier'] = (np.abs(df['amount_z_score']) > 2).astype(int)
        df['quantity_outlier'] = (np.abs(df['quantity_z_score']) > 2).astype(int)
        
        # Behavioral anomaly patterns
        df['unusual_amount_pattern'] = ((df['amount_outlier'] == 1) & 
                                       (df['quantity_outlier'] == 1)).astype(int)
        
        df['velocity_anomaly'] = ((df['velocity_risk_score'] > 3) & 
                                 (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Customer profile risk patterns
        df['high_risk_customer_profile'] = ((df['age_risk'] > 0) & 
                                           (df['account_age_risk'] > 1) & 
                                           (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Location-based risk patterns
        df['foreign_high_value'] = ((df['location_risk_score'] > 0.05) & 
                                   (df['transaction_amount'] > self.percentile_thresholds['amount_90'])).astype(int)
        
        # Time-based velocity patterns
        df['rush_hour_high_value'] = ((df['is_rush_hour'] == 1) & 
                                     (df['transaction_amount'] > self.percentile_thresholds['amount_90'])).astype(int)
        
        # Complex fraud patterns
        df['multi_risk_pattern'] = ((df['new_account'] == 1) & 
                                   (df['is_late_night'] == 1) & 
                                   (df['transaction_amount'] > self.percentile_thresholds['amount_90'])).astype(int)
        
        df['sophisticated_fraud_pattern'] = ((df['young_customer'] == 1) & 
                                            (df['rare_location'] == 1) & 
                                            (df['is_high_risk_hour'] == 1) & 
                                            (df['transaction_amount'] > self.percentile_thresholds['amount_75'])).astype(int)
        
        # Advanced risk scoring
        risk_score = 0
        risk_score += df['amount_above_95'] * 3
        risk_score += df['is_late_night'] * 2
        risk_score += df['new_account'] * 3
        risk_score += df['young_customer'] * 2
        risk_score += df['rare_location'] * 2
        risk_score += df['quantity_above_95'] * 2
        risk_score += df['high_risk_combination'] * 4
        risk_score += df['new_account_high_amount'] * 4
        risk_score += df['young_customer_high_amount'] * 3
        risk_score += df['rare_location_high_amount'] * 3
        risk_score += df['unusual_purchase_pattern'] * 2
        risk_score += df['velocity_risk_score'] * 2
        
        # New advanced patterns
        risk_score += df['suspicious_time_pattern'] * 3
        risk_score += df['weekend_high_value'] * 2.5
        # risk_score += df['holiday_period_high_value'] * 2.5  # Skip holiday period risk
        risk_score += df['high_risk_device'] * 3
        risk_score += df['high_risk_payment'] * 3
        risk_score += df['high_risk_product'] * 3
        risk_score += df['unusual_amount_pattern'] * 4
        risk_score += df['velocity_anomaly'] * 4
        risk_score += df['high_risk_customer_profile'] * 4
        risk_score += df['foreign_high_value'] * 3.5
        risk_score += df['rush_hour_high_value'] * 2.5
        risk_score += df['multi_risk_pattern'] * 5
        risk_score += df['sophisticated_fraud_pattern'] * 6
        
        df['comprehensive_risk_score'] = risk_score
        
        # Risk categories
        df['low_risk'] = (df['comprehensive_risk_score'] <= 2).astype(int)
        df['medium_risk'] = ((df['comprehensive_risk_score'] > 2) & (df['comprehensive_risk_score'] <= 6)).astype(int)
        df['high_risk'] = (df['comprehensive_risk_score'] > 6).astype(int)
        
        # Statistical outlier features
        df['amount_z_score'] = (df['transaction_amount'] - self.percentile_thresholds['amount_50']) / (self.percentile_thresholds['amount_75'] - self.percentile_thresholds['amount_50'] + 1e-8)
        df['quantity_z_score'] = (df['quantity'] - self.percentile_thresholds['quantity_50']) / (self.percentile_thresholds['quantity_75'] - self.percentile_thresholds['quantity_50'] + 1e-8)
        
        # Outlier flags
        df['amount_outlier'] = (np.abs(df['amount_z_score']) > 2).astype(int)
        df['quantity_outlier'] = (np.abs(df['quantity_z_score']) > 2).astype(int)
        
        return df
    
    def _ensure_numeric_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Ensure all features are numeric and handle any remaining categorical columns."""
        
        # Drop original categorical columns that have been encoded
        categorical_cols = ['payment_method', 'product_category', 'device_used', 'age_group', 'customer_location']
        for col in categorical_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Drop timestamp/date columns (temporal features extracted)
        date_cols = ['timestamp', 'transaction_date']
        for col in date_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
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
            # Amount features
            'amount_log', 'amount_sqrt', 'amount_per_item', 'amount_per_day', 'amount_scaled',
            'amount_above_50', 'amount_above_75', 'amount_above_90', 'amount_above_95', 'amount_above_99',
            'amount_ratio_to_median', 'amount_ratio_to_95th', 'amount_volatility',
            
            # Temporal features
            'hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
            'is_late_night', 'is_business_hours', 'is_rush_hour',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'is_high_risk_hour', 'is_holiday_period', 'hour_deviation',
            
            # Customer features
            'age_group_encoded', 'young_customer', 'senior_customer', 'age_risk',
            'account_age_days_log', 'account_age_weeks', 'account_age_months',
            'new_account', 'recent_account', 'established_account', 'old_account', 'account_age_risk',
            'location_freq', 'location_risk_score', 'rare_location',
            'payment_method_encoded', 'product_category_encoded', 'device_used_encoded',
            'payment_method_fraud_rate', 'product_category_fraud_rate', 'device_used_fraud_rate',
            
            # Behavioral features
            'quantity_log', 'quantity_per_day', 'quantity_scaled',
            'quantity_above_75', 'quantity_above_90', 'quantity_above_95', 'quantity_above_99',
            'high_value_low_quantity', 'low_value_high_quantity',
            'avg_amount_per_item', 'purchase_efficiency', 'unusual_purchase_pattern',
            
            # Velocity features
            'transaction_velocity', 'amount_velocity', 'quantity_velocity',
            'amount_per_transaction', 'quantity_per_transaction',
            'high_amount_velocity', 'high_quantity_velocity', 'velocity_risk_score',
            
            # Interaction features
            'amount_quantity_interaction', 'amount_quantity_ratio',
            'age_account_interaction', 'age_account_ratio',
            'amount_hour_interaction', 'amount_day_interaction',
            'amount_risk_interaction', 'quantity_risk_interaction',
            'location_device_interaction', 'payment_product_interaction',
            'age_amount_time_interaction', 'account_quantity_location_interaction',
            
            # Risk features
            'high_risk_combination', 'new_account_high_amount', 'young_customer_high_amount', 'rare_location_high_amount',
            'suspicious_time_pattern', 'weekend_high_value', 'holiday_period_high_value',
            'high_risk_device', 'high_risk_payment', 'high_risk_product',
            'unusual_amount_pattern', 'velocity_anomaly', 'high_risk_customer_profile',
            'foreign_high_value', 'rush_hour_high_value', 'multi_risk_pattern', 'sophisticated_fraud_pattern',
            'comprehensive_risk_score', 'low_risk', 'medium_risk', 'high_risk',
            'amount_z_score', 'quantity_z_score', 'amount_outlier', 'quantity_outlier'
        ]
        
        return feature_names
    
    def save_fitted_objects(self, filepath: str):
        """Save fitted objects to file."""
        
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before saving")
        
        fitted_objects = {
            'amount_scaler': self.amount_scaler,
            'quantity_scaler': self.quantity_scaler,
            'label_encoders': self.label_encoders,
            'percentile_thresholds': self.percentile_thresholds,
            'customer_stats': self.customer_stats,
            'location_stats': self.location_stats,
            'temporal_stats': self.temporal_stats,
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
        self.quantity_scaler = fitted_objects.get('quantity_scaler')
        self.label_encoders = fitted_objects['label_encoders']
        self.percentile_thresholds = fitted_objects['percentile_thresholds']
        self.customer_stats = fitted_objects.get('customer_stats', {})
        self.location_stats = fitted_objects.get('location_stats', {})
        self.temporal_stats = fitted_objects.get('temporal_stats', {})
        self.config = fitted_objects['config']
        self.is_fitted = fitted_objects['is_fitted']
        
        # Set feature selection flags
        self.use_amount_features = self.config.get('use_amount_features', True)
        self.use_temporal_features = self.config.get('use_temporal_features', True)
        self.use_customer_features = self.config.get('use_customer_features', True)
        self.use_risk_flags = self.config.get('use_risk_flags', True)
        self.use_behavioral_features = self.config.get('use_behavioral_features', True)
        self.use_velocity_features = self.config.get('use_velocity_features', True)
        self.use_interaction_features = self.config.get('use_interaction_features', True)
        
        logger.info(f"Fitted objects loaded from: {filepath}") 