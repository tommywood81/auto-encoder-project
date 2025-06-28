"""
Configuration management for the fraud detection pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data configuration."""
    raw_file: str
    cleaned_dir: str
    engineered_dir: str
    models_dir: str
    test_size: float
    random_state: int


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    hidden_dim: int
    latent_dim: int
    learning_rate: float
    epochs: int
    batch_size: int
    validation_split: float
    threshold_percentile: float
    save_model: bool


@dataclass
class FeatureConfig:
    """Feature configuration."""
    transaction_amount: bool
    customer_age: bool
    quantity: bool
    account_age_days: bool
    payment_method: bool
    product_category: bool
    device_used: bool
    customer_location: bool
    transaction_amount_log: bool
    customer_location_freq: bool
    temporal_features: bool
    behavioural_features: bool


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    name: str
    description: str
    feature_strategy: str
    data: DataConfig
    model: ModelConfig
    features: FeatureConfig
    
    @classmethod
    def get_baseline_config(cls) -> 'PipelineConfig':
        """Get baseline configuration."""
        return cls(
            name="baseline",
            description="Basic transaction features only",
            feature_strategy="baseline",
            data=DataConfig(
                raw_file="data/raw/Fraudulent_E-Commerce_Transaction_Data_2.csv",
                cleaned_dir="data/cleaned",
                engineered_dir="data/engineered",
                models_dir="models",
                test_size=0.2,
                random_state=42
            ),
            model=ModelConfig(
                name="autoencoder",
                hidden_dim=64,
                latent_dim=32,
                learning_rate=0.001,
                epochs=10,
                batch_size=32,
                validation_split=0.2,
                threshold_percentile=95.0,
                save_model=True
            ),
            features=FeatureConfig(
                transaction_amount=True,
                customer_age=True,
                quantity=True,
                account_age_days=True,
                payment_method=True,
                product_category=True,
                device_used=True,
                customer_location=True,
                transaction_amount_log=True,
                customer_location_freq=True,
                temporal_features=False,
                behavioural_features=False
            )
        )
    
    @classmethod
    def get_temporal_config(cls) -> 'PipelineConfig':
        """Get temporal configuration."""
        return cls(
            name="temporal",
            description="Basic transaction features + temporal patterns",
            feature_strategy="temporal",
            data=DataConfig(
                raw_file="data/raw/Fraudulent_E-Commerce_Transaction_Data_2.csv",
                cleaned_dir="data/cleaned",
                engineered_dir="data/engineered",
                models_dir="models",
                test_size=0.2,
                random_state=42
            ),
            model=ModelConfig(
                name="autoencoder",
                hidden_dim=64,
                latent_dim=32,
                learning_rate=0.001,
                epochs=10,
                batch_size=32,
                validation_split=0.2,
                threshold_percentile=95.0,
                save_model=True
            ),
            features=FeatureConfig(
                transaction_amount=True,
                customer_age=True,
                quantity=True,
                account_age_days=True,
                payment_method=True,
                product_category=True,
                device_used=True,
                customer_location=True,
                transaction_amount_log=True,
                customer_location_freq=True,
                temporal_features=True,
                behavioural_features=False
            )
        )
    
    @classmethod
    def get_behavioural_config(cls) -> 'PipelineConfig':
        """Get behavioural configuration."""
        return cls(
            name="behavioural",
            description="Core features + amount per item",
            feature_strategy="behavioural",
            data=DataConfig(
                raw_file="data/raw/Fraudulent_E-Commerce_Transaction_Data_2.csv",
                cleaned_dir="data/cleaned",
                engineered_dir="data/engineered",
                models_dir="models",
                test_size=0.2,
                random_state=42
            ),
            model=ModelConfig(
                name="autoencoder",
                hidden_dim=64,
                latent_dim=32,
                learning_rate=0.001,
                epochs=10,
                batch_size=32,
                validation_split=0.2,
                threshold_percentile=95.0,
                save_model=True
            ),
            features=FeatureConfig(
                transaction_amount=True,
                customer_age=True,
                quantity=True,
                account_age_days=True,
                payment_method=True,
                product_category=True,
                device_used=True,
                customer_location=True,
                transaction_amount_log=True,
                customer_location_freq=True,
                temporal_features=False,
                behavioural_features=True
            )
        )
    
    @classmethod
    def get_demographic_risk_config(cls) -> 'PipelineConfig':
        """Get demographic risk configuration."""
        return cls(
            name="demographic_risk",
            description="Core features + customer age risk scores",
            feature_strategy="demographic_risk",
            data=DataConfig(
                raw_file="data/raw/Fraudulent_E-Commerce_Transaction_Data_2.csv",
                test_size=0.2,
                random_state=42
            ),
            model=ModelConfig(
                name="autoencoder",
                hidden_dim=64,
                latent_dim=32,
                epochs=10,
                learning_rate=0.001
            ),
            features=FeatureConfig(
                temporal_features=False,
                behavioural_features=False
            )
        )
    
    @classmethod
    def get_combined_config(cls) -> 'PipelineConfig':
        """Get combined configuration - all features from all strategies."""
        return cls(
            name="combined",
            description="All unique features from all strategies (no duplicates)",
            feature_strategy="combined",
            data=DataConfig(
                raw_file="data/raw/Fraudulent_E-Commerce_Transaction_Data_2.csv",
                cleaned_dir="data/cleaned",
                engineered_dir="data/engineered",
                models_dir="models",
                test_size=0.2,
                random_state=42
            ),
            model=ModelConfig(
                name="autoencoder",
                hidden_dim=64,
                latent_dim=32,
                learning_rate=0.001,
                epochs=10,
                batch_size=32,
                validation_split=0.2,
                threshold_percentile=95.0,
                save_model=True
            ),
            features=FeatureConfig(
                transaction_amount=True,
                customer_age=True,
                quantity=True,
                account_age_days=True,
                payment_method=True,
                product_category=True,
                device_used=True,
                customer_location=True,
                transaction_amount_log=True,
                customer_location_freq=True,
                temporal_features=True,
                behavioural_features=True
            )
        )
    
    @classmethod
    def get_config(cls, strategy: str) -> 'PipelineConfig':
        """Get configuration by strategy name."""
        if strategy == "baseline":
            return cls.get_baseline_config()
        elif strategy == "temporal":
            return cls.get_temporal_config()
        elif strategy == "behavioural":
            return cls.get_behavioural_config()
        elif strategy == "demographic_risk":
            return cls.get_demographic_risk_config()
        elif strategy == "combined":
            return cls.get_combined_config()
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Available: baseline, temporal, behavioural, demographic_risk, combined")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            'name': self.name,
            'description': self.description,
            'feature_strategy': self.feature_strategy,
            'data': {
                'raw_file': self.data.raw_file,
                'test_size': self.data.test_size,
                'random_state': self.data.random_state
            },
            'model': {
                'name': self.model.name,
                'hidden_dim': self.model.hidden_dim,
                'latent_dim': self.model.latent_dim,
                'epochs': self.model.epochs,
                'learning_rate': self.model.learning_rate
            },
            'features': {
                'temporal_features': self.features.temporal_features,
                'behavioural_features': self.features.behavioural_features
            }
        } 