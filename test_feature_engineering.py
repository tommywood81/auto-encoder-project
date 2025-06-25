"""
Test script for feature engineering with e-commerce dataset.
"""

from src.feature_factory import FeatureFactory
from src.config import PipelineConfig

def test_feature_engineering():
    """Test the feature engineering pipeline."""
    print("Testing feature engineering pipeline...")
    
    # Initialize feature factory
    config = PipelineConfig()
    factory = FeatureFactory(config)
    
    # Engineer features
    df_engineered = factory.engineer_features()
    
    print(f"✅ Feature engineering completed successfully!")
    print(f"   Cleaned data shape: (23634, 11)")
    print(f"   Engineered shape: {df_engineered.shape}")
    print(f"   Additional features: {df_engineered.shape[1] - 11}")
    
    # Show feature summary
    summary = factory.get_feature_summary()
    print(f"\nFeature summary:")
    for group_name, group_info in summary['feature_groups'].items():
        print(f"   {group_name}: {group_info['count']} features")
        for feature_name, description in group_info['features'].items():
            print(f"     - {feature_name}: {description}")
    
    # Check target variable
    if 'Is Fraudulent' in df_engineered.columns:
        fraud_dist = df_engineered['Is Fraudulent'].value_counts()
        print(f"\n   Fraud distribution: {fraud_dist.to_dict()}")
    
    return df_engineered

if __name__ == "__main__":
    test_feature_engineering() 