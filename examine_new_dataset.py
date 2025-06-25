"""
Examine the new e-commerce fraud dataset.
"""

import pandas as pd
import numpy as np

def examine_dataset():
    """Examine the new e-commerce fraud dataset."""
    print("E-COMMERCE FRAUD DATASET ANALYSIS")
    print("=" * 50)
    
    # Load the dataset
    df = pd.read_csv('data/raw/Fraudulent_E-Commerce_Transaction_Data_2.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Basic info
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Data types
    print(f"\nData type distribution:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Categorize columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumeric columns ({len(numeric_cols)}):")
    for col in numeric_cols:
        print(f"  {col}")
    
    print(f"\nObject/Categorical columns ({len(object_cols)}):")
    for col in object_cols:
        print(f"  {col}")
    
    # Target variable analysis
    print(f"\nTarget variable analysis:")
    if 'Is Fraudulent' in df.columns:
        fraud_counts = df['Is Fraudulent'].value_counts()
        print(f"  Fraudulent: {fraud_counts.get(1, 0)} ({fraud_counts.get(1, 0)/len(df)*100:.2f}%)")
        print(f"  Legitimate: {fraud_counts.get(0, 0)} ({fraud_counts.get(0, 0)/len(df)*100:.2f}%)")
    
    # Sample data
    print(f"\nSample data (first 3 rows):")
    print(df.head(3).to_string())
    
    # Missing values
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        print("  No missing values found!")
    
    # Unique values for categorical columns
    print(f"\nCategorical column analysis:")
    for col in object_cols:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count} unique values")
        if unique_count <= 10:
            print(f"    Values: {sorted(df[col].unique())}")

if __name__ == "__main__":
    examine_dataset() 