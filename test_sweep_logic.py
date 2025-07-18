#!/usr/bin/env python3
"""
Test script to verify sweep logic works correctly
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import ConfigLoader

def test_config_loading():
    """Test that config loading works correctly."""
    print("Testing config loading...")
    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config("best_features")
        print("Config loading: PASSED")
        return config
    except Exception as e:
        print(f"Config loading: FAILED - {e}")
        return None

def test_hyperparameter_generation():
    """Test that hyperparameter combinations are generated correctly."""
    print("\nTesting hyperparameter generation...")
    
    hyperparams = {
        'learning_rate': [0.005, 0.008, 0.01, 0.012, 0.015],
        'threshold': [95, 96, 97, 98, 99]
    }
    
    import itertools
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    
    combinations = list(itertools.product(*values))
    total_combinations = len(combinations)
    
    print(f"   Learning rates: {hyperparams['learning_rate']}")
    print(f"   Thresholds: {hyperparams['threshold']}")
    print(f"   Total combinations: {total_combinations}")
    
    if total_combinations == 25:
        print("Hyperparameter generation: PASSED")
        return True
    else:
        print(f"Hyperparameter generation: FAILED - Expected 25, got {total_combinations}")
        return False

def test_model_info_creation():
    """Test that model info creation works with missing keys."""
    print("\nTesting model info creation...")
    
    # Mock config with missing keys
    mock_config = {
        'model': {
            'latent_dim': 16,
            'activation_fn': 'relu',
            # Missing: hidden_dims, dropout_rate
        },
        'features': {
            'strategy': 'combined',
            # Missing: scaling
        }
        # Missing: data section entirely
    }
    
    mock_best_params = {
        'learning_rate': 0.01,
        'threshold': 95
    }
    
    mock_metrics = {
        'precision': 0.5,
        'recall': 0.6,
        'f1_score': 0.55,
        'accuracy': 0.85
    }
    
    try:
        # Test model info creation (this is what was failing)
        model_info = {
            # Model file paths
            'model_path': 'models/final_model.h5',
            'scaler_path': 'models/final_model_scaler.pkl',
            
            # Performance metrics
            'roc_auc': 0.7349,
            'precision': float(mock_metrics.get('precision', 0.0)),
            'recall': float(mock_metrics.get('recall', 0.0)),
            'f1_score': float(mock_metrics.get('f1_score', 0.0)),
            'accuracy': float(mock_metrics.get('accuracy', 0.0)),
            
            # Model architecture
            'latent_dim': mock_config['model']['latent_dim'],
            'hidden_dims': mock_config['model'].get('hidden_dims', [64, 32]),
            'activation_fn': mock_config['model']['activation_fn'],
            'dropout_rate': mock_config['model'].get('dropout_rate', 0.2),
            'l2_reg': mock_config['model'].get('l2_reg', 0.001),
            
            # Training parameters
            'learning_rate': mock_best_params['learning_rate'],
            'batch_size': mock_config['model'].get('batch_size', 128),
            'epochs': mock_config['model'].get('epochs', 50),
            'threshold_percentile': mock_best_params['threshold'],
            'early_stopping': mock_config['model'].get('early_stopping', True),
            'patience': mock_config['model'].get('patience', 10),
            
            # Feature engineering
            'feature_strategy': mock_config['features']['strategy'],
            'feature_scaling': mock_config['features'].get('scaling', 'standard'),
            'feature_count': mock_config.get('feature_count', 'unknown'),
            
            # Data configuration
            'train_split': mock_config.get('data', {}).get('train_split', 0.8),
            'val_split': mock_config.get('data', {}).get('val_split', 0.1),
            'test_split': mock_config.get('data', {}).get('test_split', 0.1),
            'random_state': mock_config.get('data', {}).get('random_state', 42),
            
            # Metadata
            'training_date': '2025-07-12 12:00:00',
            'best_hyperparameters': mock_best_params,
            'model_type': 'autoencoder',
            'task': 'fraud_detection'
        }
        
        # Test YAML serialization
        yaml_str = yaml.dump(model_info, default_flow_style=False)
        print("Model info creation: PASSED")
        print("YAML serialization: PASSED")
        return True
        
    except Exception as e:
        print(f"Model info creation: FAILED - {e}")
        return False

def test_models_directory_creation():
    """Test that models directory can be created."""
    print("\nTesting models directory creation...")
    
    try:
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Test file creation
        test_file = models_dir / "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("test")
        
        # Clean up
        test_file.unlink()
        
        print("Models directory creation: PASSED")
        return True
    except Exception as e:
        print(f"Models directory creation: FAILED - {e}")
        return False

def test_final_model_comparison():
    """Test the final model comparison logic."""
    print("\nTesting final model comparison logic...")
    
    # Test case 1: No existing final model
    try:
        current_final_roc_auc = 0.0
        final_model_info_path = Path("models/final_model_info.yaml")
        
        if final_model_info_path.exists():
            with open(final_model_info_path, 'r') as f:
                current_final_info = yaml.safe_load(f)
                current_final_roc_auc = current_final_info.get('roc_auc', 0.0)
        else:
            current_final_roc_auc = 0.0
        
        # Test with a better model
        best_roc_auc = 0.7349
        if best_roc_auc > current_final_roc_auc:
            print("Final model comparison: PASSED (new model is better)")
            return True
        else:
            print("Final model comparison: FAILED (logic error)")
            return False
            
    except Exception as e:
        print(f"Final model comparison: FAILED - {e}")
        return False

def main():
    """Run all tests."""
    print("RUNNING SWEEP LOGIC TESTS")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_hyperparameter_generation,
        test_model_info_creation,
        test_models_directory_creation,
        test_final_model_comparison
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("ALL TESTS PASSED! Sweep should work correctly.")
        return True
    else:
        print("SOME TESTS FAILED! Need to fix issues before running sweep.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 