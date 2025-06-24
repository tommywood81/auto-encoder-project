#!/usr/bin/env python3
"""
Main script for training and evaluating the fraud detection autoencoder.
"""

import sys
import os
import logging
import torch
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import FraudDataLoader
from src.autoencoder import Autoencoder, AutoencoderTrainer
from src.evaluator import FraudEvaluator
from src.config import PERCENTILE_THRESHOLD


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    """Main training and evaluation pipeline."""
    setup_logging()
    
    logging.info("Starting fraud detection model training...")
    logging.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Create output directories
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Load processed data
        logging.info("Loading processed data...")
        data_loader = FraudDataLoader()
        data = data_loader.load_processed_data()
        
        # Extract data
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        X_train_ae = data['X_train_ae']
        feature_names = data['feature_names']
        
        logging.info(f"Features: {len(feature_names)}")
        
        # Initialize and train autoencoder
        input_dim = X_train_ae.shape[1]
        model = Autoencoder(input_dim=input_dim, hidden_dims=[64, 32])
        logging.info(f"Autoencoder initialized with {input_dim} input dimensions")
        
        trainer = AutoencoderTrainer(model, device=device, lr=1e-3)
        train_losses = trainer.train(X_train_ae, epochs=20, batch_size=256, verbose=True)
        
        # Save training losses
        np.save('results/training_losses.npy', train_losses)
        
        # Detect anomalies
        logging.info("Detecting anomalies...")
        y_pred, threshold, recon_errors = trainer.detect_anomalies(
            X_test, X_train_ae, percentile=PERCENTILE_THRESHOLD
        )
        
        # Evaluate model
        logging.info("Evaluating model performance...")
        evaluator = FraudEvaluator(y_test, y_pred, recon_errors)
        metrics = evaluator.comprehensive_evaluation(save_dir='results')
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': data_loader.scaler,
            'label_encoders': data_loader.label_encoders,
            'feature_names': feature_names,
            'threshold': threshold,
            'metrics': metrics
        }, 'models/autoencoder_fraud_detection.pth')
        
        logging.info("Training completed successfully!")
        logging.info(f"Model saved to models/autoencoder_fraud_detection.pth")
        logging.info(f"Results saved to results/")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 