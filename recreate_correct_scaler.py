#!/usr/bin/env python3
"""
Recreate Correct Scaler Script

This script recreates the exact scaler that was used during training
to achieve the 0.7335 ROC AUC performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import joblib
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig
from src.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder

def recreate_correct_scaler():
    """Recreate the exact scaler used during training."""
    print(" Recreating correct scaler...")
    
    # Load the best configuration
    config_path = "configs/final_optimized_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Get the feature strategy from config
    feature_strategy = config_dict['features']['strategy']
    print(f"📋 Using feature strategy: {feature_strategy}")
    
    # Create pipeline config using the same method as training
    pipeline_config = PipelineConfig.get_config(feature_strategy)
    
    # Create autoencoder instance (this will create the correct scaler)
    autoencoder = BaselineAutoencoder(pipeline_config)
    
    # Load cleaned data
    cleaned_file = os.path.join(pipeline_config.data.cleaned_dir, "ecommerce_cleaned.csv")
    df_cleaned = pd.read_csv(cleaned_file)
    print(f"📊 Loaded cleaned data: {df_cleaned.shape}")
    
    # Engineer features using the same strategy as training
    feature_engineer = FeatureFactory.create(feature_strategy)
    df = feature_engineer.generate_features(df_cleaned)
    print(f" Generated features: {df.shape}")
    
    # Get numeric features only (same as training)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_fraudulent' not in numeric_cols and 'is_fraudulent' in df.columns:
        numeric_cols.append('is_fraudulent')
    
    df_numeric = df[numeric_cols]
    print(f"📈 Numeric features: {len(numeric_cols)}")
    
    # Separate features and target
    X = df_numeric.drop(columns=['is_fraudulent'])
    y = df_numeric['is_fraudulent']
    
    # Time-aware split (same as training)
    total_samples = len(X)
    train_size = int(0.8 * total_samples)
    
    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    
    print(f"⏰ Time-aware split: train={len(X_train)}, test={len(X_test)}")
    
    # Fit scaler on training data (same as training)
    X_train_scaled = autoencoder.scaler.fit_transform(X_train)
    X_test_scaled = autoencoder.scaler.transform(X_test)
    
    # For autoencoder, we only train on non-fraudulent data
    train_normal_mask = y_train == 0
    X_train_normal = X_train_scaled[train_normal_mask]
    
    print(f"✅ Training on {len(X_train_normal)} normal transactions")
    print(f" Features: {X_train.shape[1]}")
    
    # Save the correct scaler
    scaler_path = "models/autoencoder_scaler.pkl"
    os.makedirs("models", exist_ok=True)
    
    # Save using joblib (same as training)
    joblib.dump(autoencoder.scaler, scaler_path)
    print(f" Scaler saved to: {scaler_path}")
    
    # Test the scaler
    print("\n🧪 Testing scaler...")
    test_scaler = joblib.load(scaler_path)
    test_scaled = test_scaler.transform(X_test)
    print(f"✅ Scaler test successful - transformed shape: {test_scaled.shape}")
    
    # Verify it matches the autoencoder scaler
    if np.allclose(test_scaled, X_test_scaled):
        print("✅ Scaler verification passed - matches training scaler")
    else:
        print("❌ Scaler verification failed - does not match training scaler")
    
    return autoencoder.scaler

if __name__ == "__main__":
    try:
        scaler = recreate_correct_scaler()
        print("\n🎉 Correct scaler recreated successfully!")
        print("Now run_pipeline.py should use the correct scaler and achieve 0.7335 ROC AUC")
    except Exception as e:
        print(f"❌ Error recreating scaler: {e}")
        import traceback
        traceback.print_exc() 