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
    FeatureEngineer,
    BaselineNumeric,
    CategoricalEncodings,
    TemporalFeatures,
    BehavioralFeatures,
    DemographicFeatures,
    FraudFlags,
    RollingFeatures,
    RankEncodingFeatures,
    TimeInteractionFeatures,
    CombinedFeatures
)

logger = logging.getLogger(__name__)


class FeatureFactory:
    """Factory for creating feature engineering strategies."""
    
    STRATEGY_DESCRIPTIONS = {
        "baseline_numeric": "Log and ratio features from raw numerics",
        "categorical": "Encoded payment, product, and device columns",
        "temporal": "Late-night and burst transaction flags",
        "behavioral": "Behavioral ratios per age/account age",
        "demographics": "Customer age bucketed into risk bands",
        "fraud_flags": "Rule-based fraud risk indicators",
        "rolling": "Rolling mean and std of amount per customer",
        "rank_encoding": "Rank-based encodings of amount and account age",
        "time_interactions": "Crossed and interaction features using hour",
        "combined": "All feature engineering strategies combined"
    }
    
    @classmethod
    def create(cls, strategy_name: str) -> FeatureEngineer:
        """Create a feature engineering strategy."""
        strategies = {
            "baseline_numeric": BaselineNumeric,
            "categorical": CategoricalEncodings,
            "temporal": TemporalFeatures,
            "behavioral": BehavioralFeatures,
            "demographics": DemographicFeatures,
            "fraud_flags": FraudFlags,
            "rolling": RollingFeatures,
            "rank_encoding": RankEncodingFeatures,
            "time_interactions": TimeInteractionFeatures,
            "combined": CombinedFeatures
        }
        
        if strategy_name not in strategies:
            available = ", ".join(strategies.keys())
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {available}")
        
        return strategies[strategy_name]()
    
    @classmethod
    def get_available_strategies(cls) -> Dict[str, str]:
        """Get all available strategies with descriptions."""
        return cls.STRATEGY_DESCRIPTIONS.copy()
    
    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict[str, Any]:
        """Get information about a specific strategy."""
        if strategy_name not in cls.STRATEGY_DESCRIPTIONS:
            raise ValueError(f"Unknown strategy '{strategy_name}'")
        
        strategy = cls.create(strategy_name)
        info = strategy.get_feature_info()
        info['description'] = cls.STRATEGY_DESCRIPTIONS[strategy_name]
        
        return info 