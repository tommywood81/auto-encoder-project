#!/usr/bin/env python3
"""
Production pipeline runner for fraud detection autoencoder.
This script runs the complete data pipeline from raw data to model training and evaluation.
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
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.data_loader import FraudDataLoader
from src.autoencoder import Autoencoder, AutoencoderTrainer
from src.evaluator import FraudEvaluator
from src.config import DATA_RAW, DATA_CLEANED, DATA_ENGINEERED, DATA_PROCESSED, PERCENTILE_THRESHOLD


def setup_logging(log_level='INFO', log_file=None):
    """Set up logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)


def check_prerequisites():
    """Check if all prerequisites are met."""
    logger = logging.getLogger(__name__)
    
    # Check if raw data exists
    required_files = [
        os.path.join(DATA_RAW, "train_transaction.csv"),
        os.path.join(DATA_RAW, "train_identity.csv")
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        logger.error("Please ensure the raw data is downloaded to data/raw/")
        return False
    
    logger.info("✅ All prerequisites met")
    return True


def run_data_cleaning(force_rerun=False):
    """Run data cleaning stage."""
    logger = logging.getLogger(__name__)
    
    cleaned_file = os.path.join(DATA_CLEANED, "train_cleaned.csv")
    
    if os.path.exists(cleaned_file) and not force_rerun:
        logger.info("📂 Cleaned data already exists, skipping cleaning stage")
        return True
    
    logger.info("=" * 60)
    logger.info("🧼 STAGE 1: DATA CLEANING")
    logger.info("=" * 60)
    
    try:
        start_time = time.time()
        
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_data(save_output=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Data cleaning completed in {elapsed_time:.2f} seconds")
        logger.info(f"📊 Final shape: {cleaned_df.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data cleaning failed: {str(e)}")
        return False


def run_feature_engineering(force_rerun=False):
    """Run feature engineering stage."""
    logger = logging.getLogger(__name__)
    
    engineered_file = os.path.join(DATA_ENGINEERED, "train_features.csv")
    
    if os.path.exists(engineered_file) and not force_rerun:
        logger.info("📂 Engineered data already exists, skipping feature engineering stage")
        return True
    
    logger.info("=" * 60)
    logger.info("🧪 STAGE 2: FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    try:
        start_time = time.time()
        
        engineer = FeatureEngineer()
        engineered_df = engineer.engineer_features(save_output=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Feature engineering completed in {elapsed_time:.2f} seconds")
        logger.info(f"📊 Final shape: {engineered_df.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {str(e)}")
        return False


def run_data_processing(force_rerun=False):
    """Run data processing stage."""
    logger = logging.getLogger(__name__)
    
    processed_files = [
        os.path.join(DATA_PROCESSED, "X_train.npy"),
        os.path.join(DATA_PROCESSED, "X_test.npy"),
        os.path.join(DATA_PROCESSED, "y_train.npy"),
        os.path.join(DATA_PROCESSED, "y_test.npy")
    ]
    
    if all(os.path.exists(f) for f in processed_files) and not force_rerun:
        logger.info("📂 Processed data already exists, skipping processing stage")
        return True
    
    logger.info("=" * 60)
    logger.info("📊 STAGE 3: DATA PROCESSING")
    logger.info("=" * 60)
    
    try:
        start_time = time.time()
        
        data_loader = FraudDataLoader()
        data_dict = data_loader.load_processed_data()
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Data processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"📊 Training samples: {data_dict['X_train'].shape}")
        logger.info(f"📊 Test samples: {data_dict['X_test'].shape}")
        logger.info(f"📊 Autoencoder training samples: {data_dict['X_train_ae'].shape}")
        logger.info(f"📊 Features: {len(data_dict['feature_names'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data processing failed: {str(e)}")
        return False


def run_model_training(force_rerun=False):
    """Run model training and evaluation stage."""
    logger = logging.getLogger(__name__)
    
    model_file = os.path.join('models', 'autoencoder_fraud_detection.pth')
    metrics_file = os.path.join('results', 'metrics.json')
    
    if os.path.exists(model_file) and os.path.exists(metrics_file) and not force_rerun:
        logger.info("📂 Trained model already exists, skipping training stage")
        return True
    
    logger.info("=" * 60)
    logger.info("🤖 STAGE 4: MODEL TRAINING & EVALUATION")
    logger.info("=" * 60)
    
    try:
        start_time = time.time()
        
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ Using device: {device}")
        
        # Create output directories
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Load processed data
        logger.info("📂 Loading processed data...")
        data_loader = FraudDataLoader()
        data = data_loader.load_processed_data()
        
        # Extract data
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        X_train_ae = data['X_train_ae']
        feature_names = data['feature_names']
        
        logger.info(f"📊 Feature names: {feature_names[:10]}... (showing first 10)")
        logger.info(f"📊 Total features: {len(feature_names)}")
        
        # Initialize autoencoder
        input_dim = X_train_ae.shape[1]
        model = Autoencoder(input_dim=input_dim, hidden_dims=[64, 32])
        logger.info(f"🤖 Autoencoder initialized with input dimension: {input_dim}")
        
        # Train autoencoder
        logger.info("🎯 Starting model training...")
        trainer = AutoencoderTrainer(model, device=device, lr=1e-3)
        train_losses = trainer.train(
            X_train_ae, 
            epochs=20, 
            batch_size=256, 
            verbose=True
        )
        
        # Save training losses
        np.save('results/training_losses.npy', train_losses)
        logger.info("💾 Training losses saved")
        
        # Detect anomalies
        logger.info("🔍 Detecting anomalies...")
        y_pred, threshold, recon_errors = trainer.detect_anomalies(
            X_test, X_train_ae, percentile=PERCENTILE_THRESHOLD
        )
        
        # Evaluate model
        logger.info("📈 Evaluating model performance...")
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
        logger.info(f"✅ Model training completed in {elapsed_time:.2f} seconds")
        logger.info(f"📊 Model performance: {metrics.get('accuracy', 'N/A'):.4f} accuracy")
        logger.info(f"📊 Precision: {metrics.get('precision', 'N/A'):.4f}")
        logger.info(f"📊 Recall: {metrics.get('recall', 'N/A'):.4f}")
        logger.info(f"📊 F1-Score: {metrics.get('f1_score', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {str(e)}")
        return False


def generate_pipeline_report():
    """Generate a comprehensive pipeline report."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("📋 PIPELINE REPORT")
    logger.info("=" * 60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'pipeline_stages': {},
        'data_sizes': {},
        'feature_counts': {}
    }
    
    # Check each stage
    stages = {
        'raw': DATA_RAW,
        'cleaned': DATA_CLEANED,
        'engineered': DATA_ENGINEERED,
        'processed': DATA_PROCESSED
    }
    
    for stage_name, stage_path in stages.items():
        if os.path.exists(stage_path):
            files = os.listdir(stage_path)
            total_size = sum(
                os.path.getsize(os.path.join(stage_path, f)) 
                for f in files if os.path.isfile(os.path.join(stage_path, f))
            )
            
            report['pipeline_stages'][stage_name] = {
                'status': 'completed',
                'files': files,
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
            
            logger.info(f"✅ {stage_name.upper()}: {len(files)} files, {total_size / (1024 * 1024):.1f} MB")
        else:
            report['pipeline_stages'][stage_name] = {
                'status': 'missing',
                'files': [],
                'total_size_mb': 0
            }
            logger.warning(f"⚠️ {stage_name.upper()}: Directory not found")
    
    # Check model and results
    model_files = ['models/autoencoder_fraud_detection.pth', 'results/metrics.json']
    model_status = 'completed' if all(os.path.exists(f) for f in model_files) else 'missing'
    
    if model_status == 'completed':
        model_size = sum(os.path.getsize(f) for f in model_files if os.path.exists(f))
        logger.info(f"✅ MODEL: 2 files, {model_size / (1024 * 1024):.1f} MB")
    else:
        logger.warning("⚠️ MODEL: Files not found")
    
    report['pipeline_stages']['model'] = {
        'status': model_status,
        'files': [f for f in model_files if os.path.exists(f)],
        'total_size_mb': round(sum(os.path.getsize(f) for f in model_files if os.path.exists(f)) / (1024 * 1024), 2)
    }
    
    # Save report
    report_file = os.path.join('results', 'pipeline_report.json')
    os.makedirs('results', exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📄 Pipeline report saved to {report_file}")
    
    return report


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description='Run the fraud detection data pipeline')
    parser.add_argument('--stages', nargs='+', 
                       choices=['clean', 'engineer', 'process', 'train', 'all'],
                       default=['all'],
                       help='Pipeline stages to run')
    parser.add_argument('--force-rerun', action='store_true',
                       help='Force rerun all stages even if output exists')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    parser.add_argument('--log-file',
                       help='Log file path (optional)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    logger.info("🚀 Starting fraud detection pipeline...")
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🔧 Arguments: {vars(args)}")
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("❌ Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Determine stages to run
    if 'all' in args.stages:
        stages_to_run = ['clean', 'engineer', 'process', 'train']
    else:
        stages_to_run = args.stages
    
    logger.info(f"🎯 Running stages: {stages_to_run}")
    
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
        
        # Generate report
        if success:
            generate_pipeline_report()
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Pipeline interrupted by user")
        success = False
    except Exception as e:
        logger.error(f"❌ Pipeline failed with unexpected error: {str(e)}")
        success = False
    
    # Final summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("🏁 PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"⏱️ Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        logger.info("🎉 Pipeline completed successfully!")
        if 'train' in stages_to_run:
            logger.info("🤖 Model is ready for inference!")
            logger.info("📁 Results saved to 'results/' directory")
            logger.info("📁 Model saved to 'models/autoencoder_fraud_detection.pth'")
        else:
            logger.info("📁 Data is ready for modeling in data/processed/")
        sys.exit(0)
    else:
        logger.error("❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 