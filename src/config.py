"""
Configuration for E-commerce Fraud Detection baseline model.
Simplified configuration focusing on essential settings.
"""

from dataclasses import dataclass
from typing import List, Optional
import os


@dataclass
class DataConfig:
    """Data configuration for baseline model."""
    raw_dir: str = "data/raw"
    cleaned_dir: str = "data/cleaned"
    engineered_dir: str = "data/engineered"
    models_dir: str = "models"
    results_dir: str = "results"
    
    # Data splitting
    test_size: float = 0.2
    random_state: int = 42
    
    # Time-based split (for fraud detection)
    use_time_split: bool = True
    time_split_date: str = "2024-02-01"  # Split data before this date


@dataclass
class ModelConfig:
    """Autoencoder model configuration."""
    # Architecture
    input_dim: Optional[int] = None  # Will be set automatically
    hidden_dim: int = 64
    latent_dim: int = 32
    
    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    
    # Anomaly detection
    threshold_percentile: float = 95.0  # Percentile for anomaly threshold
    
    # Model saving
    save_model: bool = True
    model_name: str = "baseline_autoencoder"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    # Metrics
    primary_metric: str = "roc_auc"
    
    # Threshold optimization
    optimize_threshold: bool = True
    threshold_range: List[float] = None  # Will be set automatically
    
    # Results
    save_results: bool = True
    results_name: str = "baseline_results"


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    data: DataConfig = None
    model: ModelConfig = None
    evaluation: EvaluationConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
    
    @classmethod
    def from_file(cls, config_path: str):
        """Load configuration from file (placeholder for future)."""
        # For now, return default config
        return cls()
    
    def save_config(self, config_path: str):
        """Save configuration to file (placeholder for future)."""
        # For now, just create the directory
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        print(f"Configuration would be saved to {config_path}")


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig() 