"""
Data loader utility for fraud detection pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple
from pathlib import Path
import os

logger = logging.getLogger(__name__)


def load_and_split_data(data_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data and perform time-aware split into train/validation/test sets.
    
    Args:
        data_path: Path to the cleaned data file
        train_ratio: Ratio of data to use for training (default: 0.8)
        val_ratio: Ratio of data to use for validation (default: 0.1)
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Loading data from: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Ensure data is sorted by transaction date for time-aware split
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df = df.sort_values('transaction_date').reset_index(drop=True)
        logger.info("Data sorted by transaction date for time-aware split")
    
    # Time-aware split: train (80%) / validation (10%) / test (10%)
    total_samples = len(df)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    
    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:train_size + val_size].copy()
    df_test = df.iloc[train_size + val_size:].copy()
    
    logger.info(f"Time-aware split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    logger.info(f"Train period: transactions 0 to {train_size-1}")
    logger.info(f"Validation period: transactions {train_size} to {train_size + val_size-1}")
    logger.info(f"Test period: transactions {train_size + val_size} to {total_samples-1}")
    
    return df_train, df_val, df_test


def load_and_split_data_80_20(data_path: str, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data and perform clean 80/20 time-aware split.
    
    Args:
        data_path: Path to the cleaned data file
        train_ratio: Ratio of data to use for training (default: 0.8)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Loading data from: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Ensure data is sorted by transaction date for time-aware split
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df = df.sort_values('transaction_date').reset_index(drop=True)
        logger.info("Data sorted by transaction date for time-aware split")
    
    # Clean 80/20 time-aware split
    total_samples = len(df)
    train_size = int(train_ratio * total_samples)
    
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()
    
    logger.info(f"80/20 time-aware split: train={len(df_train)}, test={len(df_test)}")
    logger.info(f"Train period: transactions 0 to {train_size-1}")
    logger.info(f"Test period: transactions {train_size} to {total_samples-1}")
    
    return df_train, df_test


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning for fraud detection.
    
    Args:
        df: Raw dataframe
    
    Returns:
        Cleaned dataframe
    """
    logger.info("Cleaning data...")
    
    df = df.copy()
    
    # Clean column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Handle credit card dataset specifically
    if 'class' in df.columns:
        # Rename target variable for consistency
        df = df.rename(columns={'class': 'is_fraudulent'})
        
        # Clean amount column
        if 'amount' in df.columns:
            # Remove negative amounts
            df['amount'] = df['amount'].abs()
            # Cap at 99th percentile
            q99 = df['amount'].quantile(0.99)
            df['amount'] = df['amount'].clip(upper=q99)
        
        # Add transaction_date based on Time column for time-aware split
        if 'time' in df.columns:
            # Convert time to datetime (assuming time is in seconds from start)
            # Use a base date that spreads transactions across multiple months
            # Scale time to spread across 6 months to avoid constant month features
            max_time = df['time'].max()
            scaled_time = (df['time'] / max_time) * (6 * 30 * 24 * 3600)  # 6 months in seconds
            df['transaction_date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(scaled_time, unit='s')
            df['transaction_hour'] = df['transaction_date'].dt.hour
            # Skip day and month to avoid constant features in test set
            # df['transaction_day'] = df['transaction_date'].dt.day
            # df['transaction_month'] = df['transaction_date'].dt.month
        
        # Ensure V1-V28 columns are properly named
        v_columns = [f'v{i}' for i in range(1, 29)]
        for col in v_columns:
            if col not in df.columns:
                logger.warning(f"Expected column {col} not found in dataset")
        
        logger.info(f"Credit card dataset cleaned. Shape: {df.shape}")
        return df
    
    # Handle e-commerce dataset (original logic)
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['transaction_hour'] = df['transaction_date'].dt.hour
    
    # Clean numeric columns
    if 'transaction_amount' in df.columns:
        # Remove negative amounts
        df['transaction_amount'] = df['transaction_amount'].abs()
        # Cap at 99th percentile
        q99 = df['transaction_amount'].quantile(0.99)
        df['transaction_amount'] = df['transaction_amount'].clip(upper=q99)
    
    if 'customer_age' in df.columns:
        # Clip to realistic range
        df['customer_age'] = df['customer_age'].clip(lower=13, upper=100)
    
    if 'quantity' in df.columns:
        # Remove negative quantities
        df['quantity'] = df['quantity'].abs()
        # Cap at 99th percentile
        q99 = df['quantity'].quantile(0.99)
        df['quantity'] = df['quantity'].clip(upper=q99)
    
    if 'account_age_days' in df.columns:
        # Remove negative values
        df['account_age_days'] = df['account_age_days'].abs()
        # Cap at realistic range
        df['account_age_days'] = df['account_age_days'].clip(upper=3650)
    
    # Remove unnecessary columns
    columns_to_remove = [
        'transaction_id', 'customer_id', 'ip_address',
        'shipping_address', 'billing_address'
    ]
    
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    logger.info(f"Data cleaning completed. Final shape: {df.shape}")
    
    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str):
    """Save cleaned data to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to: {output_path}")


def load_cleaned_data(data_path: str) -> pd.DataFrame:
    """Load cleaned data or clean raw data if needed."""
    if os.path.exists(data_path):
        logger.info(f"Loading cleaned data from: {data_path}")
        return pd.read_csv(data_path)
    else:
        # Try to find raw data
        raw_data_path = data_path.replace('cleaned', 'raw')
        if os.path.exists(raw_data_path):
            logger.info(f"Cleaning raw data from: {raw_data_path}")
            df_raw = pd.read_csv(raw_data_path)
            df_cleaned = clean_data(df_raw)
            save_cleaned_data(df_cleaned, data_path)
            return df_cleaned
        else:
            raise FileNotFoundError(f"Neither cleaned nor raw data found at: {data_path}") 