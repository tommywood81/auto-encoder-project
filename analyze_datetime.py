"""
Analyze the datetime column to understand temporal structure.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_datetime():
    """Analyze the datetime column structure."""
    
    # Load the raw data
    df = pd.read_csv('data/raw/Fraudulent_E-Commerce_Transaction_Data_2.csv')
    
    print("## DateTime Analysis")
    print()
    
    # Convert to datetime
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    
    print("### Basic DateTime Info:")
    print(f"Date range: {df['Transaction Date'].min()} to {df['Transaction Date'].max()}")
    print(f"Total time span: {df['Transaction Date'].max() - df['Transaction Date'].min()}")
    print(f"Number of unique dates: {df['Transaction Date'].dt.date.nunique()}")
    print(f"Number of unique hours: {df['Transaction Date'].dt.hour.nunique()}")
    print()
    
    print("### Date Distribution:")
    date_counts = df['Transaction Date'].dt.date.value_counts().sort_index()
    print(f"First 10 dates: {date_counts.head(10).to_dict()}")
    print(f"Last 10 dates: {date_counts.tail(10).to_dict()}")
    print()
    
    print("### Hour Distribution:")
    hour_counts = df['Transaction Date'].dt.hour.value_counts().sort_index()
    print(f"Hour distribution: {hour_counts.to_dict()}")
    print()
    
    print("### Fraud by Date:")
    fraud_by_date = df.groupby(df['Transaction Date'].dt.date)['Is Fraudulent'].agg(['count', 'sum', 'mean'])
    fraud_by_date.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
    print("First 10 days:")
    print(fraud_by_date.head(10))
    print()
    print("Last 10 days:")
    print(fraud_by_date.tail(10))
    print()
    
    print("### Fraud by Hour:")
    fraud_by_hour = df.groupby(df['Transaction Date'].dt.hour)['Is Fraudulent'].agg(['count', 'sum', 'mean'])
    fraud_by_hour.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
    print(fraud_by_hour)
    print()
    
    print("### Temporal Split Analysis:")
    # Sort by datetime
    df_sorted = df.sort_values('Transaction Date')
    
    # Calculate split points
    total_rows = len(df_sorted)
    train_size = int(0.8 * total_rows)
    
    split_date = df_sorted.iloc[train_size]['Transaction Date']
    print(f"80/20 split date: {split_date}")
    print(f"Train: {df_sorted[df_sorted['Transaction Date'] < split_date].shape[0]} transactions")
    print(f"Test: {df_sorted[df_sorted['Transaction Date'] >= split_date].shape[0]} transactions")
    print()
    
    # Check fraud distribution in train vs test
    train_fraud = df_sorted[df_sorted['Transaction Date'] < split_date]['Is Fraudulent'].mean()
    test_fraud = df_sorted[df_sorted['Transaction Date'] >= split_date]['Is Fraudulent'].mean()
    print(f"Train fraud rate: {train_fraud:.4f}")
    print(f"Test fraud rate: {test_fraud:.4f}")
    print()
    
    print("### Sample Transactions by Time:")
    print("First 5 transactions:")
    print(df_sorted[['Transaction Date', 'Is Fraudulent']].head())
    print()
    print("Last 5 transactions:")
    print(df_sorted[['Transaction Date', 'Is Fraudulent']].tail())
    print()
    
    return df_sorted

if __name__ == "__main__":
    analyze_datetime() 