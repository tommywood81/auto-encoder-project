"""
Model evaluation for fraud detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import os
import json
import logging

logger = logging.getLogger(__name__)


class FraudEvaluator:
    """Evaluator for fraud detection models."""
    
    def __init__(self, y_true, y_pred, recon_errors=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.recon_errors = recon_errors
        
    def calculate_basic_metrics(self):
        """Calculate basic classification metrics."""
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_true, self.y_pred, zero_division=0),
            'f1_score': f1_score(self.y_true, self.y_pred, zero_division=0)
        }
        
        # Add AUC if reconstruction errors are available
        if self.recon_errors is not None:
            try:
                metrics['auc'] = roc_auc_score(self.y_true, self.recon_errors)
                metrics['average_precision'] = average_precision_score(self.y_true, self.recon_errors)
            except ValueError:
                logger.warning("Could not calculate AUC - only one class present")
        
        return metrics
    
    def get_confusion_matrix(self):
        """Get confusion matrix."""
        return confusion_matrix(self.y_true, self.y_pred)
    
    def get_classification_report(self):
        """Get detailed classification report."""
        return classification_report(self.y_true, self.y_pred, output_dict=True)
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix."""
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve if reconstruction errors are available."""
        if self.recon_errors is None:
            logger.warning("Reconstruction errors not available for ROC curve")
            return
        
        fpr, tpr, _ = roc_curve(self.y_true, self.recon_errors)
        auc = roc_auc_score(self.y_true, self.recon_errors)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curve(self, save_path=None):
        """Plot precision-recall curve if reconstruction errors are available."""
        if self.recon_errors is None:
            logger.warning("Reconstruction errors not available for PR curve")
            return
        
        precision, recall, _ = precision_recall_curve(self.y_true, self.recon_errors)
        ap = average_precision_score(self.y_true, self.recon_errors)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_reconstruction_error_distribution(self, save_path=None):
        """Plot distribution of reconstruction errors by class."""
        if self.recon_errors is None:
            logger.warning("Reconstruction errors not available for distribution plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot distributions
        plt.hist(self.recon_errors[self.y_true == 0], bins=50, alpha=0.7, 
                label='Normal', density=True)
        plt.hist(self.recon_errors[self.y_true == 1], bins=50, alpha=0.7, 
                label='Fraud', density=True)
        
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Distribution of Reconstruction Errors by Class')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def comprehensive_evaluation(self, save_dir='results'):
        """Perform comprehensive evaluation and save results."""
        logger.info("Performing comprehensive evaluation...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_basic_metrics()
        cm = self.get_confusion_matrix()
        report = self.get_classification_report()
        
        # Print results
        logger.info("Model Performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        if 'auc' in metrics:
            logger.info(f"  AUC: {metrics['auc']:.4f}")
        
        # Create plots
        self.plot_confusion_matrix(os.path.join(save_dir, 'confusion_matrix.png'))
        self.plot_roc_curve(os.path.join(save_dir, 'roc_curve.png'))
        self.plot_precision_recall_curve(os.path.join(save_dir, 'pr_curve.png'))
        self.plot_reconstruction_error_distribution(os.path.join(save_dir, 'error_distribution.png'))
        
        # Save detailed results
        results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {save_dir}")
        
        return metrics 