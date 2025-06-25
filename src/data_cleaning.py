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
        
        # 1. Handle missing values (EDA showed no missing values, but good practice)
        df_clean = self._handle_missing_values(df_clean)
        
        # 2. Clean categorical variables
        df_clean = self._clean_categorical_variables(df_clean)
        
        # 3. Clean numeric variables (based on EDA findings)
        df_clean = self._clean_numeric_variables(df_clean)
        
        # 4. Remove unnecessary columns (privacy and modeling concerns)
        df_clean = self._remove_unnecessary_columns(df_clean)
        
        # 5. Feature encoding (based on EDA decisions)
        df_clean = self._encode_features(df_clean)
        
        logger.info("Data cleaning completed")
        return df_clean
    
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
        if 'Payment Method' in df.columns:
            df['Payment Method'] = df['Payment Method'].str.lower().str.strip()
            logger.info(f"Payment methods found: {df['Payment Method'].unique()}")
        
        # Clean Product Category (EDA showed 5 categories with different fraud rates)
        if 'Product Category' in df.columns:
            df['Product Category'] = df['Product Category'].str.lower().str.strip()
            logger.info(f"Product categories found: {df['Product Category'].unique()}")
        
        # Clean Device Used (EDA showed 3 device types with different fraud rates)
        if 'Device Used' in df.columns:
            df['Device Used'] = df['Device Used'].str.lower().str.strip()
            logger.info(f"Device types found: {df['Device Used'].unique()}")
        
        # Clean Customer Location (EDA showed 14,868 unique locations - use frequency encoding)
        if 'Customer Location' in df.columns:
            df['Customer Location'] = df['Customer Location'].str.strip()
            logger.info(f"Unique locations: {df['Customer Location'].nunique()}")
        
        return df
    
    def _clean_numeric_variables(self, df):
        """Clean numeric variables based on EDA findings."""
        logger.info("Cleaning numeric variables...")
        
        # Clean Transaction Amount (EDA: cap at 99th percentile = $1162.04)
        if 'Transaction Amount' in df.columns:
            # Remove any negative values (fraud detection context)
            df['Transaction Amount'] = df['Transaction Amount'].abs()
            
            # Cap at 99th percentile to handle outliers (EDA finding)
            q99 = df['Transaction Amount'].quantile(0.99)
            df['Transaction Amount'] = df['Transaction Amount'].clip(upper=q99)
            logger.info(f"Clipped Transaction Amount at 99th percentile: ${q99:.2f}")
            
            # Log the impact
            original_mean = df['Transaction Amount'].mean()
            logger.info(f"Transaction amount mean after cleaning: ${original_mean:.2f}")
        
        # Clean Customer Age (EDA: found 331 transactions with unrealistic ages, clip to 18-100)
        if 'Customer Age' in df.columns:
            # Count unrealistic ages before cleaning
            unrealistic_before = len(df[(df['Customer Age'] < 18) | (df['Customer Age'] > 100)])
            if unrealistic_before > 0:
                logger.info(f"Found {unrealistic_before} transactions with unrealistic ages (under 18 or over 100)")
            
            # Filter to only keep customers 18+ years old
            df = df[df['Customer Age'] >= 18].copy()
            logger.info(f"Filtered dataset to customers 18+ years old. Remaining transactions: {len(df)}")
            
            # Clip remaining ages to realistic range (18-100 years)
            df['Customer Age'] = df['Customer Age'].clip(upper=100)
            logger.info("Clipped Customer Age to realistic range (18-100 years)")
        
        # Clean Quantity (EDA: max quantity was 5, no extreme outliers found)
        if 'Quantity' in df.columns:
            # Remove negative quantities
            df['Quantity'] = df['Quantity'].abs()
            
            # Cap at 99th percentile (conservative approach)
            q99 = df['Quantity'].quantile(0.99)
            df['Quantity'] = df['Quantity'].clip(upper=q99)
            logger.info(f"Clipped Quantity at 99th percentile: {q99}")
        
        # Clean Account Age Days (EDA: max was 365 days, all accounts < 1 year)
        if 'Account Age Days' in df.columns:
            # Remove negative values
            df['Account Age Days'] = df['Account Age Days'].abs()
            
            # Cap at realistic range (max 10 years = 3650 days)
            df['Account Age Days'] = df['Account Age Days'].clip(upper=3650)
            logger.info("Clipped Account Age Days to realistic range (max 10 years)")
        
        # Clean Transaction Hour (EDA: hours 0-23, fraud peaks at hours 0, 1, 3, 4, 5)
        if 'Transaction Hour' in df.columns:
            # Ensure hours are in 0-23 range
            df['Transaction Hour'] = df['Transaction Hour'].clip(lower=0, upper=23)
            logger.info("Clipped Transaction Hour to 0-23 range")
            
            # Log fraud patterns by hour (for feature engineering insights)
            hour_fraud = df.groupby('Transaction Hour')['Is Fraudulent'].mean()
            high_fraud_hours = hour_fraud[hour_fraud > 0.08].index.tolist()
            if high_fraud_hours:
                logger.info(f"High fraud hours (>{8}%): {high_fraud_hours}")
        
        return df
    
    def _remove_unnecessary_columns(self, df):
        """Remove unnecessary columns based on EDA and production considerations."""
        logger.info("Removing unnecessary columns...")
        
        # Remove columns that are not useful for modeling
        columns_to_remove = [
            'Transaction ID',      # Unique identifier, not useful for modeling
            'Customer ID',         # Unique identifier, not useful for modeling
            'IP Address',          # Privacy concern, too many unique values (14,868+)
            'Shipping Address',    # Privacy concern, too many unique values
            'Billing Address',     # Privacy concern, too many unique values
            'Transaction Date'     # Will be replaced by derived time features
        ]
        
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
                logger.info(f"Removed column: {col}")
        
        return df
    
    def _encode_features(self, df):
        """Encode categorical features based on EDA decisions."""
        logger.info("Encoding categorical features...")
        
        # Encode categorical columns based on cardinality
        categorical_cols = ['Payment Method', 'Product Category', 'Device Used', 'Customer Location']
        
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
        if 'Is Fraudulent' in df.columns:
            fraud_dist = df['Is Fraudulent'].value_counts()
            logger.info(f"Target distribution: {fraud_dist[0]} legitimate, {fraud_dist[1]} fraudulent")
        
        return output_file 