"""
Fraud Detection Autoencoder Package

This package contains modules for implementing a fraud detection system
using autoencoders to detect anomalous transactions.
"""

from .config import PipelineConfig
from .features.feature_engineer import FeatureEngineer
from .models.autoencoder import FraudAutoencoder
from .data import DataCleaner

__version__ = "1.0.0"
__author__ = "Auto-Encoder Project"

__all__ = [
    "PipelineConfig",
    "FeatureEngineer",
    "FraudAutoencoder",
    "DataCleaner"
] 