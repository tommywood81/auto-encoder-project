"""
Tests for data loading functionality using real fraud detection data.
"""

import unittest
import pandas as pd
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestDataLoading(unittest.TestCase):
    """Test data loading with real fraud detection data."""
    
    def setUp(self):
        """Set up test environment."""
        self.data_dir = Path(__file__).parent.parent / "data"
        self.cleaned_file = self.data_dir / "cleaned" / "ecommerce_cleaned.csv"
        self.models_dir = Path(__file__).parent.parent / "models"
        
    def test_cleaned_data_exists(self):
        """Test that cleaned data file exists."""
        self.assertTrue(
            self.cleaned_file.exists(), 
            f"Cleaned data file not found: {self.cleaned_file}"
        )
    
    def test_cleaned_data_is_readable(self):
        """Test that cleaned data can be loaded."""
        df = pd.read_csv(self.cleaned_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0, "DataFrame is empty")
    
    def test_cleaned_data_has_expected_columns(self):
        """Test that cleaned data has expected fraud detection columns."""
        df = pd.read_csv(self.cleaned_file)
        
        expected_columns = [
            'transaction_amount', 'customer_age', 'quantity', 
            'account_age_days', 'payment_method', 'product_category',
            'device_used', 'customer_location_freq', 'transaction_amount_log'
        ]
        
        for col in expected_columns:
            self.assertIn(
                col, df.columns, 
                f"Expected column '{col}' not found in cleaned data"
            )
    
    def test_cleaned_data_has_fraud_column(self):
        """Test that cleaned data has the target fraud column."""
        df = pd.read_csv(self.cleaned_file)
        self.assertIn('is_fraudulent', df.columns, "Fraud target column not found")
    
    def test_cleaned_data_fraud_distribution(self):
        """Test that fraud distribution is reasonable (not all fraud or all legitimate)."""
        df = pd.read_csv(self.cleaned_file)
        
        fraud_count = df['is_fraudulent'].sum()
        total_count = len(df)
        fraud_rate = fraud_count / total_count
        
        # Fraud rate should be between 1% and 20% (realistic for fraud detection)
        self.assertGreater(fraud_rate, 0.01, "Fraud rate too low")
        self.assertLess(fraud_rate, 0.20, "Fraud rate too high")
        
        print(f"Fraud rate: {fraud_rate:.3f} ({fraud_count}/{total_count})")
    
    def test_cleaned_data_no_missing_values(self):
        """Test that cleaned data has no missing values in key columns."""
        df = pd.read_csv(self.cleaned_file)
        
        key_columns = [
            'transaction_amount', 'customer_age', 'quantity', 
            'is_fraudulent', 'transaction_amount_log'
        ]
        
        for col in key_columns:
            missing_count = df[col].isnull().sum()
            self.assertEqual(
                missing_count, 0, 
                f"Column '{col}' has {missing_count} missing values"
            )
    
    def test_cleaned_data_customer_age_range(self):
        """Test that customer ages are in reasonable range (18+)."""
        df = pd.read_csv(self.cleaned_file)
        
        min_age = df['customer_age'].min()
        max_age = df['customer_age'].max()
        
        self.assertGreaterEqual(min_age, 18, f"Minimum customer age too low: {min_age}")
        self.assertLessEqual(max_age, 100, f"Maximum customer age too high: {max_age}")
        
        print(f"Customer age range: {min_age} - {max_age}")
    
    def test_cleaned_data_transaction_amounts(self):
        """Test that transaction amounts are positive and reasonable."""
        df = pd.read_csv(self.cleaned_file)
        
        min_amount = df['transaction_amount'].min()
        max_amount = df['transaction_amount'].max()
        
        self.assertGreater(min_amount, 0, "Transaction amounts should be positive")
        self.assertLess(max_amount, 100000, "Transaction amounts seem unreasonably high")
        
        print(f"Transaction amount range: ${min_amount:.2f} - ${max_amount:.2f}")

if __name__ == '__main__':
    unittest.main() 