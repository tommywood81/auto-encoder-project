#!/usr/bin/env python3
"""
Investigate Feature Differences Script

This script compares the feature generation between the recreate script and the pipeline
to identify what's causing the 3 extra features.
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig
from src.feature_factory import FeatureFactory
from src.config_loader import ConfigLoader

def investigate_features():
    """Investigate the feature differences."""
    print("🔍 Investigating feature differences...")
    
    # Load the best configuration
    config_path = "configs/final_optimized_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Get the feature strategy from config
    feature_strategy = config_dict['features']['strategy']
    print(f"📋 Using feature strategy: {feature_strategy}")
    
    # Load cleaned data
    cleaned_file = "data/cleaned/ecommerce_cleaned.csv"
    df_cleaned = pd.read_csv(cleaned_file)
    print(f"📊 Loaded cleaned data: {df_cleaned.shape}")
    
    # Method 1: Using PipelineConfig.get_config (like recreate script)
    print("\n🔧 Method 1: Using PipelineConfig.get_config (recreate script method)")
    pipeline_config = PipelineConfig.get_config(feature_strategy)
    feature_engineer1 = FeatureFactory.create(feature_strategy)
    df1 = feature_engineer1.generate_features(df_cleaned)
    
    # Get numeric features
    numeric_cols1 = df1.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_fraudulent' not in numeric_cols1 and 'is_fraudulent' in df1.columns:
        numeric_cols1.append('is_fraudulent')
    
    df_numeric1 = df1[numeric_cols1]
    X1 = df_numeric1.drop(columns=['is_fraudulent'])
    
    print(f"   Features generated: {len(df1.columns)}")
    print(f"   Numeric features: {len(numeric_cols1)}")
    print(f"   Features for model: {X1.shape[1]}")
    print(f"   Feature columns: {list(X1.columns)}")
    
    # Method 2: Using ConfigLoader (like pipeline)
    print("\n🔧 Method 2: Using ConfigLoader (pipeline method)")
    loader = ConfigLoader(config_dir="configs")
    config_name = "final_optimized_config"
    config = loader.load_config(config_name)
    
    feature_engineer2 = FeatureFactory.create(config['features']['strategy'])
    df2 = feature_engineer2.generate_features(df_cleaned)
    
    # Get numeric features
    numeric_cols2 = df2.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_fraudulent' not in numeric_cols2 and 'is_fraudulent' in df2.columns:
        numeric_cols2.append('is_fraudulent')
    
    df_numeric2 = df2[numeric_cols2]
    X2 = df_numeric2.drop(columns=['is_fraudulent'])
    
    print(f"   Features generated: {len(df2.columns)}")
    print(f"   Numeric features: {len(numeric_cols2)}")
    print(f"   Features for model: {X2.shape[1]}")
    print(f"   Feature columns: {list(X2.columns)}")
    
    # Compare the differences
    print("\n🔍 COMPARISON:")
    print(f"Method 1 (recreate): {X1.shape[1]} features")
    print(f"Method 2 (pipeline): {X2.shape[1]} features")
    print(f"Difference: {X2.shape[1] - X1.shape[1]} features")
    
    # Find the extra features
    features1 = set(X1.columns)
    features2 = set(X2.columns)
    
    extra_in_2 = features2 - features1
    missing_in_2 = features1 - features2
    
    if extra_in_2:
        print(f"\n➕ Extra features in Method 2 (pipeline): {list(extra_in_2)}")
    
    if missing_in_2:
        print(f"\n➖ Missing features in Method 2 (pipeline): {list(missing_in_2)}")
    
    if not extra_in_2 and not missing_in_2:
        print("\n✅ No feature differences found!")
    
    # Check if the feature order is different
    if list(X1.columns) != list(X2.columns):
        print(f"\n⚠️ Feature order is different!")
        print(f"Method 1 order: {list(X1.columns)}")
        print(f"Method 2 order: {list(X2.columns)}")
    
    return X1, X2

if __name__ == "__main__":
    try:
        X1, X2 = investigate_features()
        print("\n🎉 Feature investigation completed!")
    except Exception as e:
        print(f"❌ Error investigating features: {e}")
        import traceback
        traceback.print_exc() 