"""
Test to validate configuration consistency and structure.
"""

import pytest
import yaml
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config_loader import ConfigLoader


def test_config_structure():
    """Test that all config files have the required structure."""
    
    config_files = [
        "configs/sweep_broad.yaml",
        "configs/sweep_narrow.yaml", 
        "configs/sweep_final.yaml",
        "configs/final_optimized_config.yaml"
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
            
            print(f"✅ {config_path} structure is valid")


def test_config_values():
    """Test that config values are within reasonable ranges."""
    
    config_loader = ConfigLoader("configs/final_optimized_config.yaml")
    
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
    
    print("✅ Config values are within reasonable ranges")


def test_config_get_method():
    """Test the config get method with nested keys."""
    
    config_loader = ConfigLoader("configs/final_optimized_config.yaml")
    
    # Test nested key access
    latent_dim = config_loader.get('model.latent_dim')
    assert latent_dim is not None, "Could not access nested key 'model.latent_dim'"
    
    learning_rate = config_loader.get('training.learning_rate')
    assert learning_rate is not None, "Could not access nested key 'training.learning_rate'"
    
    # Test default values
    non_existent = config_loader.get('non.existent.key', default='default_value')
    assert non_existent == 'default_value', "Default value not returned for non-existent key"
    
    print("✅ Config get method works correctly")


def test_sweep_configs():
    """Test sweep-specific config requirements."""
    
    # Test broad sweep config
    if os.path.exists("configs/sweep_broad.yaml"):
        broad_config = ConfigLoader("configs/sweep_broad.yaml")
        assert 'sweep' in broad_config.config, "Broad sweep config missing 'sweep' section"
        assert broad_config.get('sweep.stage') == 'broad', "Broad sweep stage incorrect"
        assert 'model.architectures' in broad_config.config, "Broad sweep missing architectures"
        
        architectures = broad_config.get('model.architectures', {})
        assert len(architectures) > 0, "Broad sweep has no architectures"
        
        for arch_name, arch_config in architectures.items():
            assert 'latent_dim' in arch_config, f"Architecture {arch_name} missing latent_dim"
            assert 'hidden_dims' in arch_config, f"Architecture {arch_name} missing hidden_dims"
            assert 'dropout_rate' in arch_config, f"Architecture {arch_name} missing dropout_rate"
        
        print("✅ Broad sweep config is valid")
    
    # Test narrow sweep config
    if os.path.exists("configs/sweep_narrow.yaml"):
        narrow_config = ConfigLoader("configs/sweep_narrow.yaml")
        assert 'sweep' in narrow_config.config, "Narrow sweep config missing 'sweep' section"
        assert narrow_config.get('sweep.stage') == 'narrow', "Narrow sweep stage incorrect"
        
        # Should have hyperparameter lists
        batch_sizes = narrow_config.get('training.batch_sizes', [])
        learning_rates = narrow_config.get('training.learning_rates', [])
        dropout_rates = narrow_config.get('training.dropout_rates', [])
        
        assert len(batch_sizes) > 0, "Narrow sweep has no batch sizes"
        assert len(learning_rates) > 0, "Narrow sweep has no learning rates"
        assert len(dropout_rates) > 0, "Narrow sweep has no dropout rates"
        
        print("✅ Narrow sweep config is valid")
    
    # Test final sweep config
    if os.path.exists("configs/sweep_final.yaml"):
        final_config = ConfigLoader("configs/sweep_final.yaml")
        assert 'sweep' in final_config.config, "Final sweep config missing 'sweep' section"
        assert final_config.get('sweep.stage') == 'final', "Final sweep stage incorrect"
        
        # Should have fine-tuning parameters
        threshold_percentiles = final_config.get('sweep.threshold_percentiles', [])
        learning_rates = final_config.get('sweep.learning_rates', [])
        epochs_list = final_config.get('sweep.epochs', [])
        
        assert len(threshold_percentiles) > 0, "Final sweep has no threshold percentiles"
        assert len(learning_rates) > 0, "Final sweep has no learning rates"
        assert len(epochs_list) > 0, "Final sweep has no epochs"
        
        print("✅ Final sweep config is valid")


if __name__ == "__main__":
    test_config_structure()
    test_config_values()
    test_config_get_method()
    test_sweep_configs()
    print("All config consistency tests passed!") 