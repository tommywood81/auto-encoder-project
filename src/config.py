"""
Configuration settings for the fraud detection pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DataConfig:
    """Configuration for data paths and processing."""
    # Data paths
    raw_dir: str = "data/raw"
    cleaned_dir: str = "data/cleaned"
    engineered_dir: str = "data/engineered"
    processed_dir: str = "data/processed"
    intermediate_dir: str = "data/intermediate"
    
    # Data processing
    random_state: int = 42
    test_size: float = 0.2
    missing_threshold: float = 0.5  # Remove columns with >50% missing values
    
    def __post_init__(self):
        """Ensure all directories exist."""
        for directory in [self.raw_dir, self.cleaned_dir, self.engineered_dir, 
                         self.processed_dir, self.intermediate_dir]:
            os.makedirs(directory, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    # Autoencoder architecture
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 256
    
    # Anomaly detection
    percentile_threshold: float = 90.0  # For anomaly detection threshold
    
    # Device
    device: str = "auto"  # "auto", "cpu", or "cuda"
    
    def __post_init__(self):
        """Set device automatically if not specified."""
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Transaction features
    enable_amount_features: bool = True
    enable_card_features: bool = True
    enable_email_features: bool = True
    enable_addr_features: bool = True
    
    # Identity features
    enable_v_features: bool = True
    enable_id_features: bool = True
    
    # Interaction features
    enable_interaction_features: bool = True
    
    # Statistical features
    enable_statistical_features: bool = True
    
    # Advanced features (not yet implemented)
    enable_temporal_features: bool = False
    enable_behavioral_drift: bool = False
    enable_entity_novelty: bool = False
    
    # Feature transformations
    amount_transformations: List[str] = field(default_factory=lambda: ["log", "sqrt"])
    v_feature_aggregations: List[str] = field(default_factory=lambda: ["count", "mean", "std", "sum"])
    statistical_features: List[str] = field(default_factory=lambda: ["q25", "q75", "iqr", "range"])
    
    # Outlier handling
    outlier_method: str = "iqr"  # "iqr", "zscore", or "none"
    outlier_threshold: float = 1.5  # For IQR method


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    # Metrics to calculate
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "auc"])
    
    # Visualization
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 300
    
    # Results directory
    results_dir: str = "results"


@dataclass
class PipelineConfig:
    """Main configuration class that combines all configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Pipeline control
    force_rerun: bool = False
    verbose: bool = True
    save_intermediate: bool = True


# Legacy support (keeping for backward compatibility)
DATA_RAW = "data/raw"
DATA_CLEANED = "data/cleaned"
DATA_ENGINEERED = "data/engineered"
DATA_PROCESSED = "data/processed"
DATA_INTERMEDIATE = "data/intermediate"
RANDOM_STATE = 42
TEST_SIZE = 0.2
PERCENTILE_THRESHOLD = 90
HIDDEN_DIMS = [64, 32]
LEARNING_RATE = 1e-3
EPOCHS = 20
BATCH_SIZE = 256
DATA_DIR = DATA_RAW
UNZIP_DIR = "ieee_cis"
KAGGLE_COMPETITION = "ieee-fraud-detection" 