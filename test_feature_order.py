#!/usr/bin/env python3
"""
Test Feature Order Script

This script tests if the feature order is causing the scaler compatibility issue.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig
from src.feature_factory import FeatureFactory

def test_feature_order():
    """Test if feature order is causing the scaler issue."""
    print("🧪 Testing feature order compatibility...")
    
    # Load the scaler
    scaler_path = "models/autoencoder_scaler.pkl"
    scaler = joblib.load(scaler_path)
    print(f"✅ Loaded scaler from: {scaler_path}")
    
    # Load cleaned data
    cleaned_file = "data/cleaned/ecommerce_cleaned.csv"
    df_cleaned = pd.read_csv(cleaned_file)
    print(f"📊 Loaded cleaned data: {df_cleaned.shape}")
    
    # Generate features using the same method as recreate script
    feature_strategy = "combined"
    pipeline_config = PipelineConfig.get_config(feature_strategy)
    feature_engineer = FeatureFactory.create(feature_strategy)
    df = feature_engineer.generate_features(df_cleaned)
    
    # Get numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_fraudulent' not in numeric_cols and 'is_fraudulent' in df.columns:
        numeric_cols.append('is_fraudulent')
    
    df_numeric = df[numeric_cols]
    X = df_numeric.drop(columns=['is_fraudulent'])
    
    print(f"📈 Features for model: {X.shape[1]}")
    print(f"📋 Feature columns: {list(X.columns)}")
    
    # Test 1: Transform with DataFrame (should work)
    print("\n🧪 Test 1: Transform with DataFrame")
    try:
        X_scaled_df = scaler.transform(X)
        print(f"✅ Success! Scaled shape: {X_scaled_df.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 2: Transform with numpy array (what pipeline does)
    print("\n🧪 Test 2: Transform with numpy array")
    try:
        X_scaled_np = scaler.transform(X.values)
        print(f"✅ Success! Scaled shape: {X_scaled_np.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 3: Check if results are the same
    if 'X_scaled_df' in locals() and 'X_scaled_np' in locals():
        if np.allclose(X_scaled_df, X_scaled_np):
            print("✅ Results are identical!")
        else:
            print("❌ Results are different!")
            print(f"Max difference: {np.max(np.abs(X_scaled_df - X_scaled_np))}")
    
    # Test 4: Check scaler feature names
    print(f"\n🔍 Scaler feature names: {scaler.feature_names_in_}")
    print(f"🔍 Input feature names: {list(X.columns)}")
    
    # Check if they match
    if hasattr(scaler, 'feature_names_in_'):
        scaler_features = list(scaler.feature_names_in_)
        input_features = list(X.columns)
        
        if scaler_features == input_features:
            print("✅ Feature names match exactly!")
        else:
            print("❌ Feature names don't match!")
            print(f"Missing in input: {set(scaler_features) - set(input_features)}")
            print(f"Extra in input: {set(input_features) - set(scaler_features)}")
    
    return X

if __name__ == "__main__":
    try:
        X = test_feature_order()
        print("\n🎉 Feature order test completed!")
    except Exception as e:
        print(f"❌ Error testing feature order: {e}")
        import traceback
        traceback.print_exc() 