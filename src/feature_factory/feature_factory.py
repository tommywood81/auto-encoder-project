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
    BaselineFeatures,
    TemporalFeatures,
    BehaviouralFeatures,
    DemographicRiskFeatures,
    AdvancedFeatures,
    StatisticalFeatures,
    FraudSpecificFeatures,
    CombinedFeatures,
    EnsembleFeatures,
    DeepFeatures,
    ClusteringFeatures,
    PCAFeatures,
    TimeSeriesFeatures,
    UltraAdvancedFeatures,
    NeuralFeatures,
    GraphFeatures,
    HybridFeatures,
    MetaFeatures,
    QuantumFeatures,
    EvolutionaryFeatures,
    WaveletFeatures,
    EntropyFeatures,
    FractalFeatures,
    TopologicalFeatures,
    HarmonicFeatures,
    MorphologicalFeatures,
    SpectralFeatures,
    GeometricFeatures,
    BestEnsembleFeatures
)

logger = logging.getLogger(__name__)


class FeatureFactory:
    """Factory for creating feature engineering strategies."""
    
    STRATEGY_DESCRIPTIONS = {
        "baseline": "Basic transaction features only",
        "temporal": "Basic features + temporal patterns",
        "behavioural": "Basic features + customer behavior patterns",
        "demographic_risk": "Basic features + demographic risk indicators",
        "advanced": "Basic features + interaction, polynomial, and derived features",
        "statistical": "Basic features + statistical transformations and normalizations",
        "fraud_specific": "Basic features + domain-specific fraud indicators",
        "combined": "All feature engineering strategies combined",
        "ensemble": "Advanced features + ensemble rule-based scores",
        "deep": "Advanced features + complex non-linear transformations",
        "clustering": "Advanced features + unsupervised clustering features",
        "pca": "Advanced features + PCA dimensionality reduction features",
        "time_series": "Advanced features + time series patterns and sequences",
        "ultra_advanced": "All advanced strategies combined + ultra risk and anomaly scores",
        "neural": "Advanced features + neural network based reconstruction features",
        "graph": "Advanced features + graph-based transaction pattern features",
        "hybrid": "Advanced + Statistical + Fraud-specific + Neural features with custom hybrid scoring",
        "meta": "Advanced + Clustering + PCA features with meta-learning ensemble predictions",
        "quantum": "Advanced features + quantum-inspired mathematical transformations",
        "evolutionary": "Advanced features + evolutionary algorithm inspired fitness and adaptation features",
        "wavelet": "Wavelet transform based features for multi-scale analysis",
        "entropy": "Entropy and information theory based features",
        "fractal": "Fractal dimension and self-similarity features",
        "topological": "Topological data analysis inspired features",
        "harmonic": "Harmonic analysis and Fourier transform inspired features",
        "morphological": "Mathematical morphology inspired features",
        "spectral": "Spectral analysis and frequency domain features",
        "geometric": "Geometric and spatial relationship features",
        "best_ensemble": "Combines best performing strategies: hybrid, wavelet, quantum, spectral, fractal"
    }
    
    @classmethod
    def create(cls, strategy_name: str) -> FeatureEngineer:
        """Create a feature engineering strategy."""
        strategies = {
            "baseline": BaselineFeatures,
            "temporal": TemporalFeatures,
            "behavioural": BehaviouralFeatures,
            "demographic_risk": DemographicRiskFeatures,
            "advanced": AdvancedFeatures,
            "statistical": StatisticalFeatures,
            "fraud_specific": FraudSpecificFeatures,
            "combined": CombinedFeatures,
            "ensemble": EnsembleFeatures,
            "deep": DeepFeatures,
            "clustering": ClusteringFeatures,
            "pca": PCAFeatures,
            "time_series": TimeSeriesFeatures,
            "ultra_advanced": UltraAdvancedFeatures,
            "neural": NeuralFeatures,
            "graph": GraphFeatures,
            "hybrid": HybridFeatures,
            "meta": MetaFeatures,
            "quantum": QuantumFeatures,
            "evolutionary": EvolutionaryFeatures,
            "wavelet": WaveletFeatures,
            "entropy": EntropyFeatures,
            "fractal": FractalFeatures,
            "topological": TopologicalFeatures,
            "harmonic": HarmonicFeatures,
            "morphological": MorphologicalFeatures,
            "spectral": SpectralFeatures,
            "geometric": GeometricFeatures,
            "best_ensemble": BestEnsembleFeatures
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