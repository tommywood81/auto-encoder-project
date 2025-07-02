"""
Configuration loader for W&B experiment tracking.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Load and manage configuration files for experiments."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            
        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> None:
        """
        Save a configuration to file.
        
        Args:
            config: Configuration dictionary
            config_name: Name of the config file (without .yaml extension)
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def update_config(self, config_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing configuration with new values.
        
        Args:
            config_name: Name of the config file to update
            updates: Dictionary of updates to apply
            
        Returns:
            Updated configuration
        """
        config = self.load_config(config_name)
        
        # Recursively update nested dictionaries
        def update_nested(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = update_nested(d[k], v)
                else:
                    d[k] = v
            return d
        
        updated_config = update_nested(config, updates)
        self.save_config(updated_config, config_name)
        
        return updated_config
    
    def get_wandb_config(self, config_name: str, entity: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration formatted for W&B initialization.
        
        Args:
            config_name: Name of the config file
            entity: W&B entity (username/team)
            
        Returns:
            W&B configuration dictionary
        """
        config = self.load_config(config_name)
        
        # Set entity if provided
        if entity:
            config['wandb']['entity'] = entity
            
        return config


def load_experiment_config(config_name: str, entity: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load experiment configuration.
    
    Args:
        config_name: Name of the config file
        entity: W&B entity (username/team)
        
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader()
    return loader.get_wandb_config(config_name, entity) 