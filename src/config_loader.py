"""
Configuration loader for fraud detection pipeline.
Handles YAML config files with validation and reproducibility enforcement.
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate configuration files with reproducibility enforcement."""
    
    def __init__(self, config_path: str):
        """Initialize config loader with path to YAML file."""
        self.config_path = config_path
        self.config = self._load_config()
        self._enforce_reproducibility()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from: {self.config_path}")
        return config
    
    def _enforce_reproducibility(self):
        """Enforce reproducibility by setting global seeds."""
        seed = self.config.get('seed', 42)
        
        # Set Python random seed
        import random
        random.seed(seed)
        
        # Set NumPy seed
        import numpy as np
        np.random.seed(seed)
        
        # Set TensorFlow seed
        import tensorflow as tf
        tf.random.set_seed(seed)
        
        # Enable deterministic operations
        tf.config.experimental.enable_op_determinism()
        
        logger.info(f"Reproducibility enforced with seed: {seed}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration section."""
        return self.config.get('training', {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration section."""
        return self.config.get('features', {})
    
    def get_sweep_config(self) -> Dict[str, Any]:
        """Get sweep configuration section."""
        return self.config.get('sweep', {})
    
    def validate_config(self) -> bool:
        """Validate configuration structure."""
        required_sections = ['model', 'training']
        required_model_keys = ['latent_dim', 'hidden_dims']
        required_training_keys = ['batch_size', 'learning_rate', 'epochs']
        
        # Check required sections
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required config section: {section}")
                return False
        
        # Check required model keys
        model_config = self.config.get('model', {})
        for key in required_model_keys:
            if key not in model_config:
                logger.error(f"Missing required model config key: {key}")
                return False
        
        # Check required training keys
        training_config = self.config.get('training', {})
        for key in required_training_keys:
            if key not in training_config:
                logger.error(f"Missing required training config key: {key}")
                return False
        
        logger.info("Configuration validation passed")
        return True
    
    def save_config(self, output_path: str):
        """Save current configuration to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {output_path}")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key, value in updates.items():
            keys = key.split('.')
            config = self.config
            
            # Navigate to the nested location
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
        
        logger.info(f"Configuration updated with: {updates}")


def load_config(config_path: str) -> ConfigLoader:
    """Load and validate configuration."""
    config_loader = ConfigLoader(config_path)
    
    if not config_loader.validate_config():
        raise ValueError("Configuration validation failed")
    
    return config_loader 