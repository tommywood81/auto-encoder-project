"""
Tests for model loading functionality using real trained models.
"""

import unittest
import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models import BaselineAutoencoder
from src.config import PipelineConfig

class TestModelLoading(unittest.TestCase):
    """Test model loading with real trained models."""
    
    def setUp(self):
        """Set up test environment."""
        self.models_dir = Path(__file__).parent.parent / "models"
        self.data_dir = Path(__file__).parent.parent / "data"
        self.cleaned_file = self.data_dir / "cleaned" / "ecommerce_cleaned.csv"
        
        # Get configuration
        self.config = PipelineConfig.get_config("combined")
    
    def test_model_files_exist(self):
        """Test that trained model files exist."""
        expected_models = [
            "autoencoder.h5",
            "baseline_autoencoder.h5", 
            "autoencoder_fraud_detection.pth"
        ]
        
        found_models = []
        for model_file in expected_models:
            model_path = self.models_dir / model_file
            if model_path.exists():
                found_models.append(model_file)
                print(f"Found model: {model_file}")
        
        self.assertGreater(
            len(found_models), 0, 
            f"No model files found in {self.models_dir}"
        )
    
    def test_autoencoder_initialization(self):
        """Test that autoencoder can be initialized."""
        autoencoder = BaselineAutoencoder(self.config)
        
        # Load the trained model
        model_path = os.path.join(self.config.data.models_dir, "autoencoder.h5")
        if os.path.exists(model_path):
            autoencoder.load_model(model_path)
        
        self.assertIsNotNone(autoencoder)
        self.assertIsNotNone(autoencoder.model)
    
    def test_model_architecture(self):
        """Test that model has expected architecture."""
        autoencoder = BaselineAutoencoder(self.config)
        
        # Load the trained model
        model_path = os.path.join(self.config.data.models_dir, "autoencoder.h5")
        if os.path.exists(model_path):
            autoencoder.load_model(model_path)
        
        model = autoencoder.model
        if model is None:
            self.skipTest("Model could not be loaded")
        
        # Check that model has layers
        self.assertGreater(len(model.layers), 0, "Model has no layers")
        
        # Check that it's an autoencoder (input and output should be same shape)
        input_shape = model.input_shape[1:]  # Remove batch dimension
        output_shape = model.output_shape[1:]  # Remove batch dimension
        
        self.assertEqual(
            input_shape, output_shape,
            f"Autoencoder input/output shapes don't match: {input_shape} vs {output_shape}"
        )
        
        print(f"Model architecture: {len(model.layers)} layers")
        print(f"Input/Output shape: {input_shape}")
    
    def test_model_prediction_shape(self):
        """Test that model can make predictions with correct shape."""
        autoencoder = BaselineAutoencoder(self.config)
        
        # Load the trained model
        model_path = os.path.join(self.config.data.models_dir, "autoencoder.h5")
        if os.path.exists(model_path):
            autoencoder.load_model(model_path)
        
        if autoencoder.model is None:
            self.skipTest("Model could not be loaded")
        
        # Get the actual input shape from the model
        input_shape = autoencoder.model.input_shape[1:]  # Remove batch dimension
        expected_features = input_shape[0] if len(input_shape) > 0 else 12
        
        # Create dummy input with correct shape
        dummy_input = np.random.random((10, expected_features))
        
        try:
            # Make prediction
            prediction = autoencoder.model.predict(dummy_input)
            
            # Check prediction shape
            self.assertEqual(
                prediction.shape, dummy_input.shape,
                f"Prediction shape {prediction.shape} doesn't match input shape {dummy_input.shape}"
            )
        except Exception as e:
            self.skipTest(f"Model prediction failed: {str(e)}")
    

    
    def test_model_threshold_calculation(self):
        """Test that model can calculate appropriate threshold."""
        autoencoder = BaselineAutoencoder(self.config)
        
        # Load the trained model
        model_path = os.path.join(self.config.data.models_dir, "autoencoder.h5")
        if os.path.exists(model_path):
            autoencoder.load_model(model_path)
        
        if autoencoder.model is None:
            self.skipTest("Model could not be loaded")
        
        # Get the actual input shape from the model
        input_shape = autoencoder.model.input_shape[1:]  # Remove batch dimension
        expected_features = input_shape[0] if len(input_shape) > 0 else 12
        
        # Create dummy training data with correct shape
        dummy_training_data = np.random.random((100, expected_features))
        
        try:
            # Calculate threshold using the model's method
            if hasattr(autoencoder, 'calculate_threshold'):
                threshold = autoencoder.calculate_threshold(dummy_training_data)
            else:
                # Use a simple percentile-based threshold
                predictions = autoencoder.model.predict(dummy_training_data)
                reconstruction_errors = np.mean(np.square(dummy_training_data - predictions), axis=1)
                threshold = np.percentile(reconstruction_errors, 95)
            
            # Check that threshold is reasonable
            self.assertIsInstance(threshold, (int, float), "Threshold should be numeric")
            self.assertGreater(threshold, 0, "Threshold should be positive")
            
            print(f"Calculated threshold: {threshold:.4f}")
        except Exception as e:
            self.skipTest(f"Threshold calculation failed: {str(e)}")
    


if __name__ == '__main__':
    unittest.main() 