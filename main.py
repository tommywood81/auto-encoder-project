#!/usr/bin/env python3
"""
Main script for fraud detection using autoencoder.
Based on working code from notebook2.
"""

import os
import logging
import torch
from src.data_loader import FraudDataLoader
from src.autoencoder import Autoencoder, AutoencoderTrainer
from src.evaluator import FraudEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    logger.info("Starting fraud detection pipeline...")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    data_loader = FraudDataLoader()
    data = data_loader.load_processed_data()
    
    # Extract data
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    X_train_ae = data['X_train_ae']
    feature_names = data['feature_names']
    
    logger.info(f"Feature names: {feature_names[:10]}... (showing first 10)")
    logger.info(f"Total features: {len(feature_names)}")
    
    # Initialize autoencoder
    input_dim = X_train_ae.shape[1]
    model = Autoencoder(input_dim=input_dim, hidden_dims=[64, 32])
    logger.info(f"Autoencoder initialized with input dimension: {input_dim}")
    
    # Train autoencoder
    trainer = AutoencoderTrainer(model, device=device, lr=1e-3)
    train_losses = trainer.train(
        X_train_ae, 
        epochs=20, 
        batch_size=256, 
        verbose=True
    )
    
    # Save training losses
    import numpy as np
    np.save('results/training_losses.npy', train_losses)
    
    # Detect anomalies
    y_pred, threshold, recon_errors = trainer.detect_anomalies(
        X_test, X_train_ae, percentile=95
    )
    
    # Evaluate model
    logger.info("Evaluating model performance...")
    evaluator = FraudEvaluator(y_test, y_pred, recon_errors)
    metrics = evaluator.comprehensive_evaluation(save_dir='results')
    
    # Save results
    import json
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': data_loader.scaler,
        'label_encoders': data_loader.label_encoders,
        'feature_names': feature_names,
        'threshold': threshold,
        'metrics': metrics
    }, 'models/autoencoder_fraud_detection.pth')
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Results saved to 'results/' directory")
    logger.info(f"Model saved to 'models/autoencoder_fraud_detection.pth'")
    
    return metrics


if __name__ == "__main__":
    main() 