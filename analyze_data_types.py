"""
Analyze data types in the IEEE-CIS Fraud Detection dataset.
"""

import pandas as pd
import numpy as np
import os
from src.config import PipelineConfig

def analyze_data_types():
    """Analyze and categorize data types in the dataset."""
    print("DATA TYPE ANALYSIS")
    print("=" * 50)
    
    config = PipelineConfig()
    
    # Try to load from different stages
    data_sources = [
        ("Raw data", os.path.join(config.data.raw_dir, "train_transaction.csv")),
        ("Cleaned data", os.path.join(config.data.cleaned_dir, "train_cleaned.csv")),
        ("Engineered data", os.path.join(config.data.engineered_dir, "train_features.csv"))
    ]
    
    for source_name, file_path in data_sources:
        if os.path.exists(file_path):
            print(f"\n{source_name.upper()}")
            print("-" * 30)
            
            df = pd.read_csv(file_path)
            print(f"Shape: {df.shape}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Analyze data types
            dtype_counts = df.dtypes.value_counts()
            print(f"\nData type distribution:")
            for dtype, count in dtype_counts.items():
                print(f"  {dtype}: {count} columns")
            
            # Categorize columns by type
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            object_cols = df.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            print(f"\nNumeric columns ({len(numeric_cols)}):")
            for col in sorted(numeric_cols):
                print(f"  {col}")
            
            print(f"\nObject/Categorical columns ({len(object_cols)}):")
            for col in sorted(object_cols):
                print(f"  {col}")
            
            if datetime_cols:
                print(f"\nDatetime columns ({len(datetime_cols)}):")
                for col in sorted(datetime_cols):
                    print(f"  {col}")
            
            # Analyze numeric columns in detail
            if len(numeric_cols) > 0:
                print(f"\nNumeric column analysis:")
                numeric_df = df[numeric_cols]
                
                # Check for integer vs float
                int_cols = numeric_df.select_dtypes(include=['int']).columns.tolist()
                float_cols = numeric_df.select_dtypes(include=['float']).columns.tolist()
                
                print(f"  Integer columns: {len(int_cols)}")
                print(f"  Float columns: {len(float_cols)}")
                
                # Check for binary columns (0/1)
                binary_cols = []
                for col in numeric_cols:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
                        binary_cols.append(col)
                
                print(f"  Binary columns (0/1): {len(binary_cols)}")
                if binary_cols:
                    print("    " + ", ".join(binary_cols))
            
            # Analyze object columns in detail
            if len(object_cols) > 0:
                print(f"\nObject column analysis:")
                for col in object_cols[:10]:  # Show first 10
                    unique_count = df[col].nunique()
                    null_count = df[col].isnull().sum()
                    print(f"  {col}: {unique_count} unique values, {null_count} nulls")
                
                if len(object_cols) > 10:
                    print(f"  ... and {len(object_cols) - 10} more object columns")
            
            # Check for specific column patterns
            print(f"\nColumn pattern analysis:")
            
            # V columns (identity features)
            v_cols = [col for col in df.columns if col.startswith('V')]
            print(f"  V columns (identity features): {len(v_cols)}")
            
            # C columns (count features)
            c_cols = [col for col in df.columns if col.startswith('C')]
            print(f"  C columns (count features): {len(c_cols)}")
            
            # D columns (delta features)
            d_cols = [col for col in df.columns if col.startswith('D')]
            print(f"  D columns (delta features): {len(d_cols)}")
            
            # M columns (match features)
            m_cols = [col for col in df.columns if col.startswith('M')]
            print(f"  M columns (match features): {len(m_cols)}")
            
            # id_ columns
            id_cols = [col for col in df.columns if col.startswith('id_')]
            print(f"  id_ columns: {len(id_cols)}")
            
            # card columns
            card_cols = [col for col in df.columns if 'card' in col.lower()]
            print(f"  card columns: {len(card_cols)}")
            
            # addr columns
            addr_cols = [col for col in df.columns if 'addr' in col.lower()]
            print(f"  addr columns: {len(addr_cols)}")
            
            # email columns
            email_cols = [col for col in df.columns if 'email' in col.lower()]
            print(f"  email columns: {len(email_cols)}")
            
            break  # Use the first available data source
        else:
            print(f"\n{source_name} not found at {file_path}")

def analyze_feature_engineering_impact():
    """Analyze how feature engineering affects data types."""
    print(f"\n\nFEATURE ENGINEERING IMPACT ANALYSIS")
    print("=" * 50)
    
    config = PipelineConfig()
    
    # Compare cleaned vs engineered data
    cleaned_file = os.path.join(config.data.cleaned_dir, "train_cleaned.csv")
    engineered_file = os.path.join(config.data.engineered_dir, "train_features.csv")
    
    if os.path.exists(cleaned_file) and os.path.exists(engineered_file):
        cleaned_df = pd.read_csv(cleaned_file)
        engineered_df = pd.read_csv(engineered_file)
        
        print(f"Cleaned data shape: {cleaned_df.shape}")
        print(f"Engineered data shape: {engineered_df.shape}")
        print(f"Additional features: {engineered_df.shape[1] - cleaned_df.shape[1]}")
        
        # Find new columns added by feature engineering
        original_cols = set(cleaned_df.columns)
        engineered_cols = set(engineered_df.columns)
        new_cols = engineered_cols - original_cols
        
        print(f"\nNew features added by feature engineering ({len(new_cols)}):")
        for col in sorted(new_cols):
            dtype = engineered_df[col].dtype
            print(f"  {col} ({dtype})")
        
        # Analyze data type changes
        print(f"\nData type distribution comparison:")
        print(f"Cleaned data:")
        for dtype, count in cleaned_df.dtypes.value_counts().items():
            print(f"  {dtype}: {count}")
        
        print(f"Engineered data:")
        for dtype, count in engineered_df.dtypes.value_counts().items():
            print(f"  {dtype}: {count}")

if __name__ == "__main__":
    analyze_data_types()
    analyze_feature_engineering_impact() 