"""
Evaluation module for fraud detection.

This module handles model evaluation and performance metrics.
"""

from .evaluator import ModelEvaluator
from .evaluate import evaluate_model

__all__ = [
    "ModelEvaluator",
    "evaluate_model"
] 