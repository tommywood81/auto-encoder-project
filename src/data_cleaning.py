"""
Data cleaning for E-commerce Fraud Detection dataset.
Based on EDA findings and production best practices.
"""

import pandas as pd
import numpy as np
import os
import logging
from src.config import PipelineConfig
from src.ingest_data import import_fraudulent_ecommerce_data

logger = logging.getLogger(__name__)


class DataCleaner:
    """Data cleaner for E-commerce Fraud Detection dataset."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def clean_data(self, save_output=True):
        """Clean the e-commerce fraud dataset."""
        logger.info("Starting data cleaning for e-commerce fraud dataset...")
        
        # Load raw data
        df = self.load_raw_data()
        logger.info(f"Raw data shape: {df.shape}")
        
        # Clean the data
        df_cleaned = self._clean_dataset(df)
        logger.info(f"Cleaned data shape: {df_cleaned.shape}")
        
        # Save cleaned data if requested
        if save_output:
            self.save_cleaned_data(df_cleaned)
        
        return df_cleaned
    
    def load_raw_data(self):
        """Load raw data using the ingestion module."""
        return import_fraudulent_ecommerce_data()
    
    def _clean_dataset(self, df):
        """Clean the e-commerce dataset based on EDA findings."""
        logger.info("Cleaning e-commerce dataset...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # 0. Clean column names (convert to lowercase with underscores)
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
        logger.info("Cleaned column names to lowercase with underscores")
        
        # 1. Handle datetime column properly (organize by date and time)
        df_clean = self._process_datetime_column(df_clean)
        
        # 2. Handle missing values (EDA showed no missing values, but good practice)
        df_clean = self._handle_missing_values(df_clean)
        
        # 3. Clean categorical variables
        df_clean = self._clean_categorical_variables(df_clean)
        
        # 4. Clean numeric variables (based on EDA findings)
        df_clean = self._clean_numeric_variables(df_clean)
        
        # 5. Remove unnecessary columns (privacy and modeling concerns)
        df_clean = self._remove_unnecessary_columns(df_clean)
        
        # 6. Feature encoding (based on EDA decisions)
        df_clean = self._encode_features(df_clean)
        
        logger.info("Data cleaning completed")
        return df_clean
    
    def _process_datetime_column(self, df):
        """Process the datetime column to organize data by date and time (baseline version)."""
        logger.info("Processing datetime column (baseline version)...")
        
        # Convert to datetime
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        logger.info(f"Converted transaction_date to datetime. Range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
        
        # Sort by datetime to ensure temporal order
        df = df.sort_values('transaction_date').reset_index(drop=True)
        logger.info("Sorted data by transaction date for temporal integrity")
        
        # Extract hour for temporal features
        df['transaction_hour'] = df['transaction_date'].dt.hour
        
        # Create is_between_11pm_and_6am flag (high fraud risk period)
        df['is_between_11pm_and_6am'] = ((df['transaction_hour'] >= 23) | (df['transaction_hour'] <= 6)).astype(int)
        logger.info(f"Created is_between_11pm_and_6am flag. High-risk transactions: {df['is_between_11pm_and_6am'].sum()}")
        
        # Keep only the original transaction_date for baseline (no derived features)
        logger.info("Baseline: keeping only original transaction_date for temporal sorting")
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        logger.info("Handling missing values...")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts[missing_counts > 0]}")
            
            # For categorical columns, fill with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    logger.info(f"Filled missing values in {col} with mode: {mode_val}")
            
            # For numeric columns, fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Filled missing values in {col} with median: {median_val}")
        else:
            logger.info("No missing values found - great for production!")
        
        return df
    
    def _clean_categorical_variables(self, df):
        """Clean categorical variables."""
        logger.info("Cleaning categorical variables...")
        
        # Clean Payment Method (EDA showed 4 payment methods with different fraud rates)
        if 'payment_method' in df.columns:
            df['payment_method'] = df['payment_method'].str.lower().str.strip()
            logger.info(f"Payment methods found: {df['payment_method'].unique()}")
        
        # Clean Product Category (EDA showed 5 categories with different fraud rates)
        if 'product_category' in df.columns:
            df['product_category'] = df['product_category'].str.lower().str.strip()
            logger.info(f"Product categories found: {df['product_category'].unique()}")
        
        # Clean Device Used (EDA showed 3 device types with different fraud rates)
        if 'device_used' in df.columns:
            df['device_used'] = df['device_used'].str.lower().str.strip()
            logger.info(f"Device types found: {df['device_used'].unique()}")
        
        # Clean Customer Location (EDA showed 14,868 unique locations - use frequency encoding)
        # Note: Customer Location is actually the Customer ID in this dataset (misnamed)
        if 'customer_location' in df.columns:
            df['customer_location'] = df['customer_location'].str.strip()
            logger.info(f"Unique customer locations (actually customer IDs): {df['customer_location'].nunique()}")
        
        return df
    
    def _clean_numeric_variables(self, df):
        """Clean numeric variables based on EDA findings."""
        logger.info("Cleaning numeric variables...")
        
        # Clean Transaction Amount (Use robust scaling instead of capping)
        if 'transaction_amount' in df.columns:
            # Remove any negative values (fraud detection context)
            df['transaction_amount'] = df['transaction_amount'].abs()
            
            # Create log transformation for skewed distribution
            df['transaction_amount_log'] = np.log1p(df['transaction_amount'])
            logger.info("Created log transformation of Transaction Amount")
            
            # Calculate robust statistics for scaling
            median_amount = df['transaction_amount'].median()
            q75 = df['transaction_amount'].quantile(0.75)
            q25 = df['transaction_amount'].quantile(0.25)
            iqr = q75 - q25
            
            # Create robust scaled amount (median-centered, IQR-scaled)
            df['transaction_amount_robust_scaled'] = (df['transaction_amount'] - median_amount) / iqr
            logger.info(f"Created robust scaled Transaction Amount (median: ${median_amount:.2f}, IQR: ${iqr:.2f})")
            
            # Log the statistics for monitoring
            logger.info(f"Transaction amount statistics - Mean: ${df['transaction_amount'].mean():.2f}, Std: ${df['transaction_amount'].std():.2f}")
            logger.info(f"Log-transformed amount statistics - Mean: {df['transaction_amount_log'].mean():.2f}, Std: {df['transaction_amount_log'].std():.2f}")
        
        # Clean Customer Age (EDA: found 331 transactions with unrealistic ages, clip to 18-100)
        if 'customer_age' in df.columns:
            # Count unrealistic ages before cleaning
            unrealistic_before = len(df[(df['customer_age'] < 18) | (df['customer_age'] > 100)])
            if unrealistic_before > 0:
                logger.info(f"Found {unrealistic_before} transactions with unrealistic ages (under 18 or over 100)")
            
            # Filter to only keep customers 18+ years old
            df = df[df['customer_age'] >= 18].copy()
            logger.info(f"Filtered dataset to customers 18+ years old. Remaining transactions: {len(df)}")
            
            # Clip remaining ages to realistic range (18-100 years)
            df['customer_age'] = df['customer_age'].clip(upper=100)
            logger.info("Clipped Customer Age to realistic range (18-100 years)")
        
        # Clean Quantity (EDA: max quantity was 5, no extreme outliers found)
        if 'quantity' in df.columns:
            # Remove negative quantities
            df['quantity'] = df['quantity'].abs()
            
            # Cap at 99th percentile (conservative approach)
            q99 = df['quantity'].quantile(0.99)
            df['quantity'] = df['quantity'].clip(upper=q99)
            logger.info(f"Clipped Quantity at 99th percentile: {q99}")
        
        # Clean Account Age Days (EDA: max was 365 days, all accounts < 1 year)
        if 'account_age_days' in df.columns:
            # Remove negative values
            df['account_age_days'] = df['account_age_days'].abs()
            
            # Cap at realistic range (max 10 years = 3650 days)
            df['account_age_days'] = df['account_age_days'].clip(upper=3650)
            logger.info("Clipped Account Age Days to realistic range (max 10 years)")
        
        # Clean Transaction Hour (EDA: hours 0-23, fraud peaks at hours 0, 1, 3, 4, 5)
        if 'transaction_hour' in df.columns:
            # Ensure hours are in 0-23 range
            df['transaction_hour'] = df['transaction_hour'].clip(lower=0, upper=23)
            logger.info("Clipped Transaction Hour to 0-23 range")
            
            # Log fraud patterns by hour (for feature engineering insights)
            hour_fraud = df.groupby('transaction_hour')['is_fraudulent'].mean()
            high_fraud_hours = hour_fraud[hour_fraud > 0.08].index.tolist()
            if high_fraud_hours:
                logger.info(f"High fraud hours (>{8}%): {high_fraud_hours}")
        
        return df
    
    def _remove_unnecessary_columns(self, df):
        """Remove unnecessary columns based on EDA and production considerations."""
        logger.info("Removing unnecessary columns...")
        
        # Remove columns that are not useful for modeling
        columns_to_remove = [
            'transaction_id',      # ALL UNIQUE (100%) - No predictive value for modeling
            'customer_id',         # ALL UNIQUE (100%) - No predictive value for modeling  
            'ip_address',          # ALL UNIQUE (100%) - Privacy concern + no predictive value
            'shipping_address',    # ALL UNIQUE (100%) - Privacy concern + no predictive value
            'billing_address',     # ALL UNIQUE (100%) - Privacy concern + no predictive value
            # Note: transaction_date kept for feature engineering (will extract time-based features)
        ]
        
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
                logger.info(f"Removed column: {col} (100% unique - no predictive value)")
        
        return df
    
    def _encode_features(self, df):
        """Encode categorical features based on EDA decisions."""
        logger.info("Encoding categorical features...")
        
        # Encode categorical columns based on cardinality
        categorical_cols = ['payment_method', 'product_category', 'device_used', 'customer_location']
        
        for col in categorical_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                
                # For high cardinality columns (like Customer Location), use frequency encoding
                if unique_count > 50:
                    freq_encoding = df[col].value_counts(normalize=True)
                    df[f'{col}_freq'] = df[col].map(freq_encoding)
                    logger.info(f"Applied frequency encoding to {col} ({unique_count} unique values)")
                else:
                    # For low cardinality, use label encoding
                    df[col] = df[col].astype('category').cat.codes
                    logger.info(f"Applied label encoding to {col} ({unique_count} unique values)")
        
        return df
    
    def save_cleaned_data(self, df):
        """Save cleaned data to the cleaned directory."""
        logger.info("Saving cleaned data...")
        
        os.makedirs(self.config.data.cleaned_dir, exist_ok=True)
        
        output_file = os.path.join(self.config.data.cleaned_dir, "ecommerce_cleaned.csv")
        df.to_csv(output_file, index=False)
        
        # Log summary statistics
        logger.info(f"Cleaned data saved to {output_file}")
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Log target distribution
        if 'is_fraudulent' in df.columns:
            fraud_dist = df['is_fraudulent'].value_counts()
            logger.info(f"Target distribution: {fraud_dist[0]} legitimate, {fraud_dist[1]} fraudulent")
        
        return output_file 