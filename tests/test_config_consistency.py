"""
Test to validate configuration consistency and structure.
"""

import pytest
import yaml
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config_loader import ConfigLoader


def test_config_structure():
    """Test that all config files have the required structure."""
    
    config_files = [
        "configs/final_optimized_config.yaml",
        "tests/config/tests_config.yaml"
    ]
    
    for config_path in config_files:
        if os.path.exists(config_path):
            print(f"Testing config: {config_path}")
            
            # Load config
            config_loader = ConfigLoader(config_path)
            
            # Test validation
            assert config_loader.validate_config(), f"Config validation failed for {config_path}"
            
            # Test required sections exist
            assert 'model' in config_loader.config, f"Missing 'model' section in {config_path}"
            assert 'training' in config_loader.config, f"Missing 'training' section in {config_path}"
            assert 'seed' in config_loader.config, f"Missing 'seed' in {config_path}"
            
            # Test required model keys
            model_config = config_loader.get_model_config()
            assert 'latent_dim' in model_config, f"Missing 'latent_dim' in {config_path}"
            assert 'hidden_dims' in model_config, f"Missing 'hidden_dims' in {config_path}"
            
            # Test required training keys
            training_config = config_loader.get_training_config()
            assert 'batch_size' in training_config, f"Missing 'batch_size' in {config_path}"
            assert 'learning_rate' in training_config, f"Missing 'learning_rate' in {config_path}"
            assert 'epochs' in training_config, f"Missing 'epochs' in {config_path}"
            
            print(f"[PASS] {config_path} structure is valid")


def test_config_values():
    """Test that config values are within reasonable ranges."""
    
    config_loader = ConfigLoader("tests/config/tests_config.yaml")
    
    # Test model config values
    model_config = config_loader.get_model_config()
    assert 1 <= model_config['latent_dim'] <= 100, "latent_dim out of reasonable range"
    assert isinstance(model_config['hidden_dims'], list), "hidden_dims should be a list"
    assert all(1 <= dim <= 2048 for dim in model_config['hidden_dims']), "hidden_dims values out of range"
    assert 0.0 <= model_config['dropout_rate'] <= 0.5, "dropout_rate out of range"
    
    # Test training config values
    training_config = config_loader.get_training_config()
    assert 1 <= training_config['batch_size'] <= 1024, "batch_size out of range"
    assert 1e-6 <= training_config['learning_rate'] <= 1.0, "learning_rate out of range"
    assert 1 <= training_config['epochs'] <= 1000, "epochs out of range"
    assert 1 <= training_config['patience'] <= 100, "patience out of range"
    
    # Test seed
    seed = config_loader.config['seed']
    assert isinstance(seed, int), "seed should be an integer"
    assert 0 <= seed <= 2**32 - 1, "seed out of range"
    
    print("[PASS] Config values are within reasonable ranges")


def test_config_get_method():
    """Test config get method functionality."""
    
    config_loader = ConfigLoader("tests/config/tests_config.yaml")
    
    # Test get method with default values
    value = config_loader.get('nonexistent.key', 'default_value')
    assert value == 'default_value', "Default value not returned for nonexistent key"
    
    # Test get method with existing values
    latent_dim = config_loader.get('model.latent_dim')
    assert latent_dim is not None, "Existing key not found"
    
    print("[PASS] Config get method works correctly")


def test_sweep_configs():
    """Test sweep configuration files."""
    
    # Test broad sweep config
    try:
        broad_config = ConfigLoader("configs/sweep_broad.yaml")
        assert broad_config.validate_config(), "Broad sweep config validation failed"
        print("[PASS] Broad sweep config is valid")
    except Exception as e:
        print(f"Broad sweep config error: {e}")
        raise
    
    # Test narrow sweep config
    try:
        narrow_config = ConfigLoader("configs/sweep_narrow.yaml")
        assert narrow_config.validate_config(), "Narrow sweep config validation failed"
        print("[PASS] Narrow sweep config is valid")
    except Exception as e:
        print(f"Narrow sweep config error: {e}")
        raise
    
    # Test final sweep config
    try:
        final_config = ConfigLoader("configs/sweep_final.yaml")
        assert final_config.validate_config(), "Final sweep config validation failed"
        print("[PASS] Final sweep config is valid")
    except Exception as e:
        print(f"Final sweep config error: {e}")
        raise


if __name__ == "__main__":
    test_config_structure()
    test_config_values()
    test_config_get_method()
    test_sweep_configs()
    print("All config consistency tests passed!") 