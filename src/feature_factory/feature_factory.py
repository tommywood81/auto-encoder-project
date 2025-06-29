"""
Feature Factory with Strategy Pattern for Fraud Detection.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np
import logging

# Import the feature strategy classes
from .strategies import (
    BaselineFeatures,
    TemporalFeatures,
    BehaviouralFeatures,
    DemographicRiskFeatures,
    CombinedFeatures
)

logger = logging.getLogger(__name__)


class FeatureEngineer(ABC):
    """Abstract base class for feature engineering strategies."""
    
    @abstractmethod
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for the given dataset."""
        pass
    
    @abstractmethod
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the features generated."""
        pass


class FeatureFactory:
    """Factory for creating feature engineering strategies."""
    
    STRATEGY_DESCRIPTIONS = {
        "baseline": "Basic transaction features only",
        "temporal": "Basic features + temporal patterns",
        "behavioural": "Core features + amount per item",
        "demographic_risk": "Core features + customer age risk scores",
        "combined": "All unique features from all strategies"
    }
    
    STRATEGY_CLASSES = {
        "baseline": BaselineFeatures,
        "temporal": TemporalFeatures,
        "behavioural": BehaviouralFeatures,
        "demographic_risk": DemographicRiskFeatures,
        "combined": CombinedFeatures
    }
    
    @classmethod
    def create(cls, strategy_name: str) -> FeatureEngineer:
        """Create a feature engineering strategy."""
        if strategy_name not in cls.STRATEGY_CLASSES:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(cls.STRATEGY_CLASSES.keys())}")
        
        return cls.STRATEGY_CLASSES[strategy_name]()
    
    @classmethod
    def get_available_strategies(cls) -> list:
        """Get list of available strategies."""
        return list(cls.STRATEGY_CLASSES.keys()) 