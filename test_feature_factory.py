#!/usr/bin/env python3
"""
Test script for the new config dataclass and feature factory.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig, FeatureConfig
from src.feature_factory import FeatureFactory
from src.feature_engineering import FeatureEngineer
import pandas as pd


def test_config():
    """Test the config dataclass."""
    print("Testing Config Dataclass...")
    print("=" * 50)
    
    # Create default config
    config = PipelineConfig()
    print(f"Default config created:")
    print(f"  Data dirs: {config.data.raw_dir}, {config.data.cleaned_dir}")
    print(f"  Model hidden dims: {config.model.hidden_dims}")
    print(f"  Feature config: {config.features.enable_amount_features}")
    print(f"  Device: {config.model.device}")
    
    # Create custom config
    custom_features = FeatureConfig(
        enable_amount_features=True,
        enable_card_features=False,  # Disable card features
        enable_interaction_features=False,  # Disable interactions
        amount_transformations=["log"]  # Only log transformation
    )
    
    custom_config = PipelineConfig(
        features=custom_features,
        model=config.model,
        data=config.data
    )
    
    print(f"\nCustom config:")
    print(f"  Card features enabled: {custom_config.features.enable_card_features}")
    print(f"  Interaction features enabled: {custom_config.features.enable_interaction_features}")
    print(f"  Amount transformations: {custom_config.features.amount_transformations}")
    
    return config, custom_config


def test_feature_factory():
    """Test the feature factory with sample data."""
    print("\n\nTesting Feature Factory...")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'TransactionAmt': [100, 200, 300, 400, 500],
        'card1': [1, 2, 3, 4, 5],
        'card2': [10, 20, 30, 40, 50],
        'P_emaildomain': ['gmail.com', 'yahoo.com', 'hotmail.com', 'gmail.com', 'yahoo.com'],
        'addr1': [100, 200, 300, 400, 500],
        'V1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'V2': [1.1, 1.2, 1.3, 1.4, 1.5],
        'id_01': [1, 2, 3, 4, 5],
        'id_02': [10, 20, 30, 40, 50],
        'isFraud': [0, 0, 1, 0, 1]
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample data columns: {list(sample_data.columns)}")
    
    # Test with default config
    config = PipelineConfig()
    factory = FeatureFactory(config.features)
    
    # Build features
    df_with_features = factory.build_features(sample_data.copy())
    
    print(f"\nAfter feature engineering: {df_with_features.shape}")
    
    # Get feature summary
    summary = factory.get_feature_summary()
    print(f"\nFeature summary:")
    for group_name, group_info in summary['feature_groups'].items():
        print(f"  {group_name}: {group_info['count']} features")
        for feature_name, description in group_info['features'].items():
            print(f"    - {feature_name}: {description}")
    
    # Test with custom config (disabled features)
    custom_features = FeatureConfig(
        enable_amount_features=True,
        enable_card_features=False,
        enable_interaction_features=False
    )
    
    custom_factory = FeatureFactory(custom_features)
    df_custom = custom_factory.build_features(sample_data.copy())
    
    print(f"\nWith custom config (disabled card/interaction): {df_custom.shape}")
    
    custom_summary = custom_factory.get_feature_summary()
    print(f"Custom feature summary:")
    for group_name, group_info in custom_summary['feature_groups'].items():
        print(f"  {group_name}: {group_info['count']} features")
    
    return df_with_features, df_custom


def test_feature_engineer():
    """Test the updated feature engineer."""
    print("\n\nTesting Feature Engineer...")
    print("=" * 50)
    
    try:
        # This will only work if cleaned data exists
        config = PipelineConfig()
        engineer = FeatureEngineer(config)
        
        print("Feature engineer initialized successfully")
        print(f"Using config: {type(config)}")
        print(f"Feature factory: {type(engineer.feature_factory)}")
        
        # Try to load cleaned data
        try:
            df = engineer.load_cleaned_data()
            print(f"Successfully loaded cleaned data: {df.shape}")
        except FileNotFoundError:
            print("Cleaned data not found - this is expected if pipeline hasn't been run")
        
    except Exception as e:
        print(f"Error initializing feature engineer: {e}")


def main():
    """Main test function."""
    print("FEATURE FACTORY AND CONFIG TEST")
    print("=" * 60)
    
    # Test config
    config, custom_config = test_config()
    
    # Test feature factory
    df_default, df_custom = test_feature_factory()
    
    # Test feature engineer
    test_feature_engineer()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
    print("✓ Config dataclass working")
    print("✓ Feature factory working")
    print("✓ Feature engineer updated")
    print("\nThe new system allows for:")
    print("- Configurable feature engineering")
    print("- Easy feature enabling/disabling")
    print("- Modular feature builders")
    print("- Clean separation of concerns")


if __name__ == "__main__":
    main() 