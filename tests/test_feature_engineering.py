"""
Tests for feature engineering strategies.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.feature_factory.feature_factory import FeatureFactory
from src.config import PipelineConfig


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering strategies."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data for testing
        np.random.seed(42)
        n_samples = 100
        
        self.df = pd.DataFrame({
            'transaction_amount': np.random.exponential(100, n_samples),
            'customer_age': np.random.randint(18, 80, n_samples),
            'quantity': np.random.randint(1, 10, n_samples),
            'account_age_days': np.random.randint(1, 1000, n_samples),
            'payment_method': np.random.choice(['credit_card', 'debit_card', 'digital_wallet'], n_samples),
            'product_category': np.random.choice(['electronics', 'clothing', 'books'], n_samples),
            'device_used': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
            'customer_location': np.random.choice(['US', 'UK', 'CA'], n_samples),
            'transaction_date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
            'transaction_hour': np.random.randint(0, 24, n_samples)
        })
        
        # Available strategies to test
        self.strategies = [
            'baseline_numeric', 'categorical', 'temporal', 'behavioral',
            'demographics', 'fraud_flags', 'rolling', 'rank_encoding', 
            'time_interactions', 'combined'
        ]
    
    def test_baseline_numeric_features_generation(self):
        """Test baseline numeric feature generation with real data."""
        baseline_engineer = FeatureFactory.create('baseline_numeric')
        df_features = baseline_engineer.generate_features(self.df)
        
        # Check that features were added
        self.assertGreater(len(df_features.columns), len(self.df.columns))
        
        # Check specific features
        self.assertIn('transaction_amount_log', df_features.columns)
        self.assertIn('amount_per_item', df_features.columns)
        
        # Check feature info
        feature_info = baseline_engineer.get_feature_info()
        self.assertEqual(feature_info['strategy'], 'baseline_numeric')
        
        print(f"Baseline numeric features: {len(df_features.columns)} features")
    
    def test_behavioral_features_generation(self):
        """Test behavioral feature generation with real data."""
        behavioral_engineer = FeatureFactory.create('behavioral')
        df_features = behavioral_engineer.generate_features(self.df)
        
        # Check that features were added
        self.assertGreater(len(df_features.columns), len(self.df.columns))
        
        # Check specific features
        self.assertIn('amount_per_age', df_features.columns)
        self.assertIn('amount_per_account_age', df_features.columns)
        
        # Check that ratios are reasonable
        self.assertTrue((df_features['amount_per_age'] >= 0).all())
        self.assertTrue((df_features['amount_per_account_age'] >= 0).all())
        
        # Check feature info
        feature_info = behavioral_engineer.get_feature_info()
        self.assertEqual(feature_info['strategy'], 'behavioral')
        
        print(f"Behavioral features: {len(df_features.columns)} features")
    
    def test_demographics_features_generation(self):
        """Test demographics feature generation with real data."""
        demographics_engineer = FeatureFactory.create('demographics')
        df_features = demographics_engineer.generate_features(self.df)
        
        # Check that features were added
        self.assertGreater(len(df_features.columns), len(self.df.columns))
        
        # Check specific features
        self.assertIn('customer_age_band', df_features.columns)
        
        # Check that age bands are reasonable (0-4)
        self.assertTrue((df_features['customer_age_band'] >= 0).all())
        self.assertTrue((df_features['customer_age_band'] <= 4).all())
        
        # Check feature info
        feature_info = demographics_engineer.get_feature_info()
        self.assertEqual(feature_info['strategy'], 'demographics')
        
        print(f"Demographics features: {len(df_features.columns)} features")
    
    def test_combined_features_generation(self):
        """Test combined feature generation with real data."""
        combined_engineer = FeatureFactory.create('combined')
        df_features = combined_engineer.generate_features(self.df)
        
        # Check that features were added
        self.assertGreater(len(df_features.columns), len(self.df.columns))
        
        # Combined should have more features than baseline
        baseline_engineer = FeatureFactory.create('baseline_numeric')
        df_baseline = baseline_engineer.generate_features(self.df)
        
        self.assertGreater(
            len(df_features.columns), len(df_baseline.columns),
            "Combined features should have more columns than baseline"
        )
        
        # Check feature info
        feature_info = combined_engineer.get_feature_info()
        self.assertEqual(feature_info['strategy'], 'combined')
        
        print(f"Combined features: {len(df_features.columns)} features")
    
    def test_all_strategies(self):
        """Test all available strategies."""
        for strategy in self.strategies:
            with self.subTest(strategy=strategy):
                try:
                    engineer = FeatureFactory.create(strategy)
                    df_features = engineer.generate_features(self.df)
                    
                    # Basic checks
                    self.assertIsInstance(df_features, pd.DataFrame)
                    self.assertGreater(len(df_features.columns), 0)
                    
                    # Check feature info
                    feature_info = engineer.get_feature_info()
                    self.assertIn('strategy', feature_info)
                    self.assertIn('description', feature_info)
                    
                    print(f"✓ {strategy}: {len(df_features.columns)} features")
                    
                except Exception as e:
                    self.fail(f"Strategy {strategy} failed: {str(e)}")
    
    def test_feature_factory_creation(self):
        """Test feature factory strategy creation."""
        for strategy in self.strategies:
            with self.subTest(strategy=strategy):
                try:
                    engineer = FeatureFactory.create(strategy)
                    self.assertIsNotNone(engineer)
                except Exception as e:
                    self.fail(f"Failed to create strategy {strategy}: {str(e)}")
    
    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with self.assertRaises(ValueError):
            FeatureFactory.create('invalid_strategy')
    
    def test_strategy_descriptions(self):
        """Test that all strategies have descriptions."""
        descriptions = FeatureFactory.get_available_strategies()
        
        for strategy in self.strategies:
            with self.subTest(strategy=strategy):
                self.assertIn(strategy, descriptions)
                self.assertIsInstance(descriptions[strategy], str)
                self.assertGreater(len(descriptions[strategy]), 0)


if __name__ == '__main__':
    unittest.main() 