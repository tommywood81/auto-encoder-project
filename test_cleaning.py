"""
Test script for data cleaning with e-commerce dataset.
"""

from src.data_cleaning import DataCleaner
from src.config import PipelineConfig

def test_cleaning():
    """Test the data cleaning pipeline."""
    print("Testing data cleaning pipeline...")
    
    # Initialize cleaner
    config = PipelineConfig()
    cleaner = DataCleaner(config)
    
    # Clean the data
    df_cleaned = cleaner.clean_data()
    
    print(f"✅ Data cleaning completed successfully!")
    print(f"   Original shape: (23634, 16)")
    print(f"   Cleaned shape: {df_cleaned.shape}")
    print(f"   Columns: {list(df_cleaned.columns)}")
    
    # Check target variable
    if 'Is Fraudulent' in df_cleaned.columns:
        fraud_dist = df_cleaned['Is Fraudulent'].value_counts()
        print(f"   Fraud distribution: {fraud_dist.to_dict()}")
    
    return df_cleaned

if __name__ == "__main__":
    test_cleaning() 