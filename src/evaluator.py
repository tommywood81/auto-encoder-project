"""
Model evaluation utilities for fraud detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import logging

logger = logging.getLogger(__name__)


class FraudEvaluator:
    """Evaluator for fraud detection models."""
    
    def __init__(self, y_true, y_pred, recon_errors=None):
        """
        Initialize evaluator.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            recon_errors: Reconstruction errors (for autoencoder)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.recon_errors = recon_errors
        
    def print_classification_report(self):
        """Print detailed classification report."""
        logger.info("\nClassification Report:")
        print(classification_report(self.y_true, self.y_pred))
        
    def calculate_metrics(self):
        """Calculate key performance metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred),
            'recall': recall_score(self.y_true, self.y_pred),
            'f1_score': f1_score(self.y_true, self.y_pred)
        }
        
        if self.recon_errors is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.recon_errors)
            
        return metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_error_distribution(self, threshold=None, save_path=None):
        """Plot reconstruction error distribution."""
        if self.recon_errors is None:
            logger.warning("No reconstruction errors provided for plotting")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot histograms
        plt.hist(self.recon_errors[self.y_true == 0], bins=100, alpha=0.5, 
                label='Normal', density=True)
        plt.hist(self.recon_errors[self.y_true == 1], bins=100, alpha=0.5, 
                label='Fraud', density=True)
        
        # Plot threshold line
        if threshold is not None:
            plt.axvline(threshold, color='red', linestyle='--', 
                       label=f'Threshold ({threshold:.4f})')
        
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Reconstruction Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve."""
        if self.recon_errors is None:
            logger.warning("No reconstruction errors provided for ROC curve")
            return
            
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(self.y_true, self.recon_errors)
        auc = roc_auc_score(self.y_true, self.recon_errors)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_precision_recall_curve(self, save_path=None):
        """Plot precision-recall curve."""
        if self.recon_errors is None:
            logger.warning("No reconstruction errors provided for PR curve")
            return
            
        precision, recall, _ = precision_recall_curve(self.y_true, self.recon_errors)
        ap = average_precision_score(self.y_true, self.recon_errors)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def comprehensive_evaluation(self, save_dir=None):
        """Run comprehensive evaluation with all plots and metrics."""
        logger.info("Running comprehensive evaluation...")
        
        # Calculate and print metrics
        metrics = self.calculate_metrics()
        logger.info(f"Performance Metrics: {metrics}")
        
        # Print classification report
        self.print_classification_report()
        
        # Create plots
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
        # Plot confusion matrix
        cm_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
        self.plot_confusion_matrix(cm_path)
        
        # Plot error distribution
        if self.recon_errors is not None:
            error_path = os.path.join(save_dir, 'error_distribution.png') if save_dir else None
            self.plot_error_distribution(error_path)
            
            # Plot ROC curve
            roc_path = os.path.join(save_dir, 'roc_curve.png') if save_dir else None
            self.plot_roc_curve(roc_path)
            
            # Plot PR curve
            pr_path = os.path.join(save_dir, 'pr_curve.png') if save_dir else None
            self.plot_precision_recall_curve(pr_path)
        
        return metrics 