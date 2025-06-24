#!/usr/bin/env python3
"""
Fraud detection pipeline - from raw data to trained model.
"""

import os
import sys
import argparse
import logging
import time
import json
import torch
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.data_loader import FraudDataLoader
from src.autoencoder import Autoencoder, AutoencoderTrainer
from src.evaluator import FraudEvaluator
from src.config import DATA_RAW, DATA_CLEANED, DATA_ENGINEERED, DATA_PROCESSED, PERCENTILE_THRESHOLD


def setup_logging(log_level='INFO', log_file=None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def check_prerequisites():
    """Check if raw data exists."""
    required_files = [
        os.path.join(DATA_RAW, "train_transaction.csv"),
        os.path.join(DATA_RAW, "train_identity.csv")
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logging.error(f"Missing required files: {missing_files}")
        return False
    
    logging.info("All prerequisites met")
    return True


def run_data_cleaning(force_rerun=False):
    """Run data cleaning stage."""
    cleaned_file = os.path.join(DATA_CLEANED, "train_cleaned.csv")
    
    if os.path.exists(cleaned_file) and not force_rerun:
        logging.info("Cleaned data exists, skipping cleaning stage")
        return True
    
    logging.info("=" * 50)
    logging.info("STAGE 1: DATA CLEANING")
    logging.info("=" * 50)
    
    try:
        start_time = time.time()
        
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_data(save_output=True)
        
        elapsed_time = time.time() - start_time
        logging.info(f"Data cleaning completed in {elapsed_time:.1f} seconds")
        logging.info(f"Final shape: {cleaned_df.shape}")
        
        return True
        
    except Exception as e:
        logging.error(f"Data cleaning failed: {str(e)}")
        return False


def run_feature_engineering(force_rerun=False):
    """Run feature engineering stage."""
    engineered_file = os.path.join(DATA_ENGINEERED, "train_features.csv")
    
    if os.path.exists(engineered_file) and not force_rerun:
        logging.info("Engineered data exists, skipping feature engineering stage")
        return True
    
    logging.info("=" * 50)
    logging.info("STAGE 2: FEATURE ENGINEERING")
    logging.info("=" * 50)
    
    try:
        start_time = time.time()
        
        engineer = FeatureEngineer()
        engineered_df = engineer.engineer_features(save_output=True)
        
        elapsed_time = time.time() - start_time
        logging.info(f"Feature engineering completed in {elapsed_time:.1f} seconds")
        logging.info(f"Final shape: {engineered_df.shape}")
        
        return True
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        return False


def run_data_processing(force_rerun=False):
    """Run data processing stage."""
    processed_files = [
        os.path.join(DATA_PROCESSED, "X_train.npy"),
        os.path.join(DATA_PROCESSED, "X_test.npy"),
        os.path.join(DATA_PROCESSED, "y_train.npy"),
        os.path.join(DATA_PROCESSED, "y_test.npy")
    ]
    
    if all(os.path.exists(f) for f in processed_files) and not force_rerun:
        logging.info("Processed data exists, skipping processing stage")
        return True
    
    logging.info("=" * 50)
    logging.info("STAGE 3: DATA PROCESSING")
    logging.info("=" * 50)
    
    try:
        start_time = time.time()
        
        data_loader = FraudDataLoader()
        data_dict = data_loader.load_processed_data()
        
        elapsed_time = time.time() - start_time
        logging.info(f"Data processing completed in {elapsed_time:.1f} seconds")
        logging.info(f"Training samples: {data_dict['X_train'].shape}")
        logging.info(f"Test samples: {data_dict['X_test'].shape}")
        logging.info(f"Features: {len(data_dict['feature_names'])}")
        
        return True
        
    except Exception as e:
        logging.error(f"Data processing failed: {str(e)}")
        return False


def run_model_training(force_rerun=False):
    """Run model training and evaluation stage."""
    model_file = os.path.join('models', 'autoencoder_fraud_detection.pth')
    metrics_file = os.path.join('results', 'metrics.json')
    
    if os.path.exists(model_file) and os.path.exists(metrics_file) and not force_rerun:
        logging.info("Trained model exists, skipping training stage")
        return True
    
    logging.info("=" * 50)
    logging.info("STAGE 4: MODEL TRAINING & EVALUATION")
    logging.info("=" * 50)
    
    try:
        start_time = time.time()
        
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
        
        # Save results
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
        
        elapsed_time = time.time() - start_time
        logging.info(f"Model training completed in {elapsed_time:.1f} seconds")
        logging.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        logging.info(f"Precision: {metrics.get('precision', 0):.4f}")
        logging.info(f"Recall: {metrics.get('recall', 0):.4f}")
        logging.info(f"F1-Score: {metrics.get('f1_score', 0):.4f}")
        
        return True
        
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        return False


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='Fraud detection pipeline')
    parser.add_argument('--stages', nargs='+', 
                       choices=['clean', 'engineer', 'process', 'train', 'all'],
                       default=['all'],
                       help='Pipeline stages to run')
    parser.add_argument('--force-rerun', action='store_true',
                       help='Force rerun all stages')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    parser.add_argument('--log-file',
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    logging.info("Starting fraud detection pipeline...")
    logging.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_prerequisites():
        logging.error("Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Determine stages to run
    if 'all' in args.stages:
        stages_to_run = ['clean', 'engineer', 'process', 'train']
    else:
        stages_to_run = args.stages
    
    logging.info(f"Running stages: {stages_to_run}")
    
    # Run pipeline stages
    start_time = time.time()
    success = True
    
    try:
        if 'clean' in stages_to_run:
            if not run_data_cleaning(args.force_rerun):
                success = False
        
        if 'engineer' in stages_to_run and success:
            if not run_feature_engineering(args.force_rerun):
                success = False
        
        if 'process' in stages_to_run and success:
            if not run_data_processing(args.force_rerun):
                success = False
        
        if 'train' in stages_to_run and success:
            if not run_model_training(args.force_rerun):
                success = False
        
    except KeyboardInterrupt:
        logging.warning("Pipeline interrupted by user")
        success = False
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        success = False
    
    # Final summary
    total_time = time.time() - start_time
    logging.info("=" * 50)
    logging.info("PIPELINE SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Total execution time: {total_time:.1f} seconds")
    logging.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        logging.info("Pipeline completed successfully!")
        if 'train' in stages_to_run:
            logging.info("Model is ready for inference!")
        else:
            logging.info("Data is ready for modeling")
        sys.exit(0)
    else:
        logging.error("Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 