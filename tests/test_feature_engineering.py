"""
Tests for feature engineering using real fraud detection data.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.feature_factory import FeatureFactory
from src.config import PipelineConfig

class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering with real fraud detection data."""
    
    def setUp(self):
        """Set up test environment with real data."""
        self.data_dir = Path(__file__).parent.parent / "data"
        self.cleaned_file = self.data_dir / "cleaned" / "ecommerce_cleaned.csv"
        
        # Load real data
        self.df = pd.read_csv(self.cleaned_file)
        
        # Get available strategies
        self.strategies = FeatureFactory.get_available_strategies()
    
    def test_feature_factory_has_strategies(self):
        """Test that feature factory has expected strategies."""
        expected_strategies = [
            'baseline', 'temporal', 'behavioural', 
            'demographic_risk', 'combined'
        ]
        
        for strategy in expected_strategies:
            self.assertIn(
                strategy, self.strategies, 
                f"Strategy '{strategy}' not found in factory"
            )
    
    def test_baseline_features_generation(self):
        """Test baseline feature generation with real data."""
        baseline_engineer = FeatureFactory.create('baseline')
        df_features = baseline_engineer.generate_features(self.df)
        
        # Check that features were generated
        self.assertIsInstance(df_features, pd.DataFrame)
        self.assertGreater(len(df_features), 0)
        
        # Check feature info
        feature_info = baseline_engineer.get_feature_info()
        self.assertEqual(feature_info['strategy'], 'baseline')
        self.assertIn('feature_count', feature_info)
        
        print(f"Baseline features: {feature_info['feature_count']} features")
    
    def test_behavioural_features_generation(self):
        """Test behavioural feature generation with real data."""
        behavioural_engineer = FeatureFactory.create('behavioural')
        df_features = behavioural_engineer.generate_features(self.df)
        
        # Check that features were generated
        self.assertIsInstance(df_features, pd.DataFrame)
        self.assertGreater(len(df_features), 0)
        
        # Check that amount_per_item was created
        if 'amount_per_item' in df_features.columns:
            # Check that amount_per_item is reasonable
            amount_per_item = df_features['amount_per_item']
            self.assertGreater(amount_per_item.min(), 0, "Amount per item should be positive")
            self.assertLess(amount_per_item.max(), 10000, "Amount per item seems too high")
        
        feature_info = behavioural_engineer.get_feature_info()
        print(f"Behavioural features: {feature_info['feature_count']} features")
    
    def test_demographic_risk_features_generation(self):
        """Test demographic risk feature generation with real data."""
        risk_engineer = FeatureFactory.create('demographic_risk')
        df_features = risk_engineer.generate_features(self.df)
        
        # Check that features were generated
        self.assertIsInstance(df_features, pd.DataFrame)
        self.assertGreater(len(df_features), 0)
        
        # Check that customer_age_risk was created
        if 'customer_age_risk' in df_features.columns:
            # Check that risk scores are between 0 and 1
            risk_scores = df_features['customer_age_risk']
            self.assertGreaterEqual(risk_scores.min(), 0, "Risk scores should be >= 0")
            self.assertLessEqual(risk_scores.max(), 1, "Risk scores should be <= 1")
        
        feature_info = risk_engineer.get_feature_info()
        print(f"Demographic risk features: {feature_info['feature_count']} features")
    
    def test_combined_features_generation(self):
        """Test combined feature generation with real data."""
        combined_engineer = FeatureFactory.create('combined')
        df_features = combined_engineer.generate_features(self.df)
        
        # Check that features were generated
        self.assertIsInstance(df_features, pd.DataFrame)
        self.assertGreater(len(df_features), 0)
        
        # Combined should have more features than baseline
        baseline_engineer = FeatureFactory.create('baseline')
        df_baseline = baseline_engineer.generate_features(self.df)
        
        self.assertGreater(
            len(df_features.columns), len(df_baseline.columns),
            "Combined features should have more columns than baseline"
        )
        
        feature_info = combined_engineer.get_feature_info()
        print(f"Combined features: {feature_info['feature_count']} features")
    
    def test_feature_engineering_no_missing_values(self):
        """Test that feature engineering doesn't introduce missing values."""
        for strategy in self.strategies:
            engineer = FeatureFactory.create(strategy)
            df_features = engineer.generate_features(self.df)
            
            # Remove target column for this test
            if 'is_fraudulent' in df_features.columns:
                df_features = df_features.drop(columns=['is_fraudulent'])
            
            missing_count = df_features.isnull().sum().sum()
            self.assertEqual(
                missing_count, 0, 
                f"Strategy '{strategy}' introduced {missing_count} missing values"
            )
    
    def test_feature_engineering_numeric_output(self):
        """Test that all feature engineering outputs are numeric."""
        for strategy in self.strategies:
            engineer = FeatureFactory.create(strategy)
            df_features = engineer.generate_features(self.df)
            
            # Remove target column for this test
            if 'is_fraudulent' in df_features.columns:
                df_features = df_features.drop(columns=['is_fraudulent'])
            
            # Check that all columns are numeric
            for col in df_features.columns:
                self.assertTrue(
                    np.issubdtype(df_features[col].dtype, np.number),
                    f"Column '{col}' in strategy '{strategy}' is not numeric"
                )
    
    def test_feature_engineering_consistency(self):
        """Test that feature engineering produces consistent results."""
        for strategy in self.strategies:
            engineer = FeatureFactory.create(strategy)
            
            # Generate features twice
            df_features_1 = engineer.generate_features(self.df)
            df_features_2 = engineer.generate_features(self.df)
            
            # Results should be identical
            pd.testing.assert_frame_equal(
                df_features_1, df_features_2,
                check_dtype=False,  # Allow for minor dtype differences
                check_names=True
            )

if __name__ == '__main__':
    unittest.main() 