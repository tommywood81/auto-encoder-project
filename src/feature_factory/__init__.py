"""
Feature Factory Package for Fraud Detection.

This package provides a factory pattern for creating different feature engineering strategies.
"""

from .feature_factory import FeatureFactory, FeatureEngineer

__all__ = ['FeatureFactory', 'FeatureEngineer'] 