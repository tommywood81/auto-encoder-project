"""
Feature Factory with Strategy Pattern for Fraud Detection.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np
import logging

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
    
    _strategies = {
        "baseline": "BaselineFeatures",
        "temporal": "TemporalFeatures", 
        "behavioural": "BehaviouralFeatures",
        "account_age": "AccountAgeRiskFeatures",
        "device_novelty": "DeviceNoveltyFeatures"
    }
    
    @classmethod
    def create(cls, strategy_name: str) -> FeatureEngineer:
        """Create a feature engineering strategy."""
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(cls._strategies.keys())}")
        
        # Import the strategy class dynamically
        from .strategies import BaselineFeatures, TemporalFeatures, BehaviouralFeatures, AccountAgeRiskFeatures, DeviceNoveltyFeatures
        
        strategy_classes = {
            "baseline": BaselineFeatures,
            "temporal": TemporalFeatures,
            "behavioural": BehaviouralFeatures,
            "account_age": AccountAgeRiskFeatures,
            "device_novelty": DeviceNoveltyFeatures
        }
        
        return strategy_classes[strategy_name]()
    
    @classmethod
    def get_available_strategies(cls) -> list:
        """Get list of available strategies."""
        return list(cls._strategies.keys()) 