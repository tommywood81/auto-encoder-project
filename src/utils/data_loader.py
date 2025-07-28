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


def load_and_split_data(data_path: str, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data and perform time-aware split.
    
    Args:
        data_path: Path to the cleaned data file
        train_ratio: Ratio of data to use for training
    
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
    
    # Time-aware split
    total_samples = len(df)
    train_size = int(train_ratio * total_samples)
    
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()
    
    logger.info(f"Time-aware split: train={len(df_train)}, test={len(df_test)}")
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
    
    # Handle datetime
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