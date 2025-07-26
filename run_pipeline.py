"""
Enhanced Main pipeline for fraud detection using autoencoders.
Loads best hyperparameters from final_model_info.yaml and generates comprehensive outputs.
"""

import argparse
import logging
import sys
import os
import yaml
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import joblib
import pickle
import tensorflow as tf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import ConfigLoader
from src.data.data_cleaning import DataCleaner
from src.feature_factory.feature_factory import FeatureFactory
from src.models import BaselineAutoencoder
from src.evaluation.evaluator import FraudEvaluator
from src.config import PipelineConfig, DataConfig, ModelConfig, FeatureConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_final_model_info() -> Dict[str, Any]:
    """Load the best hyperparameters from final_model_info.yaml."""
    model_info_path = Path("models/final_model_info.yaml")
    if not model_info_path.exists():
        raise FileNotFoundError(f"final_model_info.yaml not found at {model_info_path}")
    
    with open(model_info_path, 'r') as f:
        model_info = yaml.safe_load(f)
    
    logger.info(f"Loaded final model info with ROC AUC: {model_info.get('roc_auc', 'unknown')}")
    return model_info

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['predictions', 'intermediate', 'results', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def dict_to_pipeline_config(config_dict: Dict[str, Any]) -> PipelineConfig:
    """Convert dictionary config to PipelineConfig object."""
    # Create DataConfig
    data_config = DataConfig(
        raw_file="data/raw/Fraudulent_E-Commerce_Transaction_Data_2.csv",
        cleaned_dir="data/cleaned",
        engineered_dir="data/engineered",
        models_dir="models",
        test_size=config_dict.get('data', {}).get('test_split', 0.1),
        random_state=config_dict.get('data', {}).get('random_state', 42)
    )
    
    # Create ModelConfig
    model_dict = config_dict.get('model', {})
    model_config = ModelConfig(
        name="autoencoder",
        hidden_dim=model_dict.get('hidden_dims', [64, 32])[0] if isinstance(model_dict.get('hidden_dims'), list) else 64,
        latent_dim=model_dict.get('latent_dim', 16),
        learning_rate=model_dict.get('learning_rate', 0.01),
        epochs=model_dict.get('epochs', 50),
        batch_size=model_dict.get('batch_size', 32),
        validation_split=0.2,
        threshold_percentile=model_dict.get('threshold', 95.0),
        save_model=True
    )
    
    # Create FeatureConfig
    feature_dict = config_dict.get('features', {})
    feature_config = FeatureConfig(
        transaction_amount=True,
        customer_age=True,
        quantity=True,
        account_age_days=True,
        payment_method=True,
        product_category=True,
        device_used=True,
        customer_location=True,
        transaction_amount_log=True,
        customer_location_freq=True,
        temporal_features=False,
        behavioral_features=False
    )
    
    # Create PipelineConfig
    pipeline_config = PipelineConfig(
        name="enhanced_pipeline",
        description="Enhanced pipeline with best hyperparameters",
        feature_strategy=feature_dict.get('strategy', 'combined'),
        data=data_config,
        model=model_config,
        features=feature_config
    )
    
    return pipeline_config

def load_existing_model_and_scaler(autoencoder: BaselineAutoencoder, model_info: Dict[str, Any]) -> bool:
    """Load the existing trained model and scaler. Returns True if scaler was loaded successfully."""
    logger.info("Loading existing model and scaler...")
    
    # Load model
    model_path = Path("models/autoencoder.h5")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    autoencoder.model = tf.keras.models.load_model(model_path, compile=False)
    logger.info(f"Model loaded from: {model_path}")
    
    # Load scaler
    scaler_path = Path("models/autoencoder_scaler.pkl")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    scaler_loaded = False
    try:
        # Try loading with joblib first (how we saved it)
        autoencoder.scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from: {scaler_path}")
        scaler_loaded = True
    except Exception as e:
        logger.warning(f"Failed to load scaler with joblib from {scaler_path}: {e}")
        try:
            # Try loading with pickle as fallback
            with open(scaler_path, 'rb') as f:
                autoencoder.scaler = pickle.load(f)
            logger.info(f"Scaler loaded from: {scaler_path} (using pickle)")
            scaler_loaded = True
        except Exception as e2:
            logger.warning(f"Failed to load scaler with pickle from {scaler_path}: {e2}")
            logger.info("Creating new scaler and fitting on data...")
            # Create a new scaler and fit it on the data
            from sklearn.preprocessing import StandardScaler
            autoencoder.scaler = StandardScaler()
            # We'll need to fit it later when we have the data
    
    # Update model_info with actual paths
    model_info['model_path'] = str(model_path)
    model_info['scaler_path'] = str(scaler_path)
    
    # Save updated model_info
    model_info_path = Path("models/final_model_info.yaml")
    with open(model_info_path, 'w') as f:
        yaml.dump(model_info, f, default_flow_style=False)
    logger.info(f"Updated model info saved to: {model_info_path}")
    
    return scaler_loaded

def generate_predictions(autoencoder: BaselineAutoencoder, df_features: pd.DataFrame, 
                        threshold: float, model_info: Dict[str, Any]) -> pd.DataFrame:
    """Generate predictions and save to predictions/final_predictions.csv and predictions/labelled_predictions.csv."""
    logger.info("Generating predictions...")
    
    # Get numeric features
    df_numeric = df_features.select_dtypes(include=[np.number])
    if 'is_fraudulent' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=['is_fraudulent'])
    
    # Apply time-aware split (first 80% train, last 20% test)
    # Data is already sorted by transaction_date, so we can use index for time split
    total_samples = len(df_features)
    train_size = int(0.8 * total_samples)  # 80% for training, 20% for testing
    
    # Split based on temporal order (first 80% for training, last 20% for testing)
    df_train = df_features.iloc[:train_size]
    df_test = df_features.iloc[train_size:]
    df_numeric_train = df_numeric.iloc[:train_size]
    df_numeric_test = df_numeric.iloc[train_size:]
    
    logger.info(f"Time-aware split: train={len(df_train)}, test={len(df_test)}")
    logger.info(f"Train period: transactions 0 to {train_size-1}")
    logger.info(f"Test period: transactions {train_size} to {total_samples-1}")
    
    # Get anomaly scores for test set only (for evaluation)
    test_anomaly_scores = autoencoder.predict_anomaly_scores(df_numeric_test.values)
    
    # Apply threshold to get binary predictions for test set
    test_binary_predictions = (test_anomaly_scores > threshold).astype(int)
    
    # Create test predictions dataframe
    test_predictions_df = df_test.copy()
    test_predictions_df['anomaly_score'] = test_anomaly_scores
    test_predictions_df['predicted_fraud'] = test_binary_predictions
    test_predictions_df['predicted_label'] = test_predictions_df['predicted_fraud'].map({0: 'not_fraud', 1: 'fraud'})
    
    # Save test predictions for evaluation
    test_predictions_path = Path("predictions/test_predictions.csv")
    test_predictions_df.to_csv(test_predictions_path, index=False)
    logger.info(f"Test predictions saved to: {test_predictions_path}")
    
    # Save labelled test predictions (simplified version with key columns)
    labelled_columns = ['anomaly_score', 'predicted_fraud', 'predicted_label']
    if 'is_fraudulent' in test_predictions_df.columns:
        labelled_columns.insert(0, 'is_fraudulent')
    
    labelled_test_df = test_predictions_df[labelled_columns].copy()
    labelled_path = Path("predictions/labelled_predictions.csv")
    labelled_test_df.to_csv(labelled_path, index=False)
    logger.info(f"Labelled test predictions saved to: {labelled_path}")
    
    return test_predictions_df

def save_intermediate_files(autoencoder: BaselineAutoencoder, df_features: pd.DataFrame, 
                          threshold: float, model_info: Dict[str, Any]):
    """Save intermediate files for visualizations."""
    logger.info("Saving intermediate files...")
    
    # Get numeric features
    df_numeric = df_features.select_dtypes(include=[np.number])
    if 'is_fraudulent' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=['is_fraudulent'])
    
    # Apply time-aware split (first 80% train, last 20% test)
    # Data is already sorted by transaction_date, so we can use index for time split
    total_samples = len(df_features)
    train_size = int(0.8 * total_samples)  # 80% for training, 20% for testing
    
    # Split based on temporal order (first 80% for training, last 20% for testing)
    df_train = df_features.iloc[:train_size]
    df_test = df_features.iloc[train_size:]
    df_numeric_train = df_numeric.iloc[:train_size]
    df_numeric_test = df_numeric.iloc[train_size:]
    
    logger.info(f"Time-aware split for intermediate files: train={len(df_train)}, test={len(df_test)}")
    logger.info(f"Train period: transactions 0 to {train_size-1}")
    logger.info(f"Test period: transactions {train_size} to {total_samples-1}")
    
    # Save test set anomaly scores (for evaluation and graphs)
    test_anomaly_scores = autoencoder.predict_anomaly_scores(df_numeric_test.values)
    test_anomaly_df = pd.DataFrame({
        'anomaly_score': test_anomaly_scores,
        'is_fraudulent': df_test['is_fraudulent'] if 'is_fraudulent' in df_test.columns else None
    })
    anomaly_path = Path("intermediate/anomaly_scores.csv")
    test_anomaly_df.to_csv(anomaly_path, index=False)
    logger.info(f"Test set anomaly scores saved to: {anomaly_path}")
    
    # Save test set latent space (get encoder part of autoencoder)
    X_test_scaled = autoencoder.scaler.transform(df_numeric_test.values)
    encoder = autoencoder.model.layers[0]  # First layer is encoder
    test_latent_space = encoder.predict(X_test_scaled)
    latent_path = Path("intermediate/latent_space.npy")
    np.save(latent_path, test_latent_space)
    logger.info(f"Test set latent space saved to: {latent_path}")
    
    # Save threshold
    threshold_info = {
        'threshold_value': float(threshold),
        'threshold_percentile': model_info.get('threshold_percentile', 95),
        'model_info_path': str(Path("models/final_model_info.yaml"))
    }
    threshold_path = Path("intermediate/threshold_info.yaml")
    with open(threshold_path, 'w') as f:
        yaml.dump(threshold_info, f, default_flow_style=False)
    logger.info(f"Threshold info saved to: {threshold_path}")

def compute_evaluation_metrics(predictions_df: pd.DataFrame, model_info: Dict[str, Any]) -> Dict[str, float]:
    """Compute and save evaluation metrics."""
    logger.info("Computing evaluation metrics...")
    
    if 'is_fraudulent' not in predictions_df.columns:
        logger.warning("No ground truth labels found - skipping evaluation metrics")
        return {}
    
    y_true = predictions_df['is_fraudulent']
    y_pred = predictions_df['predicted_fraud']
    anomaly_scores = predictions_df['anomaly_score']
    
    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(y_true, anomaly_scores),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred)
    }
    
    # Save metrics as JSON
    metrics_path = Path("results/metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation metrics saved to: {metrics_path}")
    
    # Also save as YAML for consistency
    metrics_yaml_path = Path("results/evaluation_metrics.yaml")
    with open(metrics_yaml_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    logger.info(f"Evaluation metrics also saved to: {metrics_yaml_path}")
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    print("="*60)
    
    return metrics

def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete fraud detection pipeline with existing model."""
    logger.info(f"Starting enhanced fraud detection pipeline with existing model...")
    
    # Load best hyperparameters from final_model_info.yaml
    model_info = load_final_model_info()
    
    # Create necessary directories
    create_directories()
    
    # Update config with best hyperparameters
    config['model']['learning_rate'] = model_info['learning_rate']
    config['model']['threshold'] = model_info['threshold_percentile']
    config['features']['strategy'] = model_info['feature_strategy']
    
    logger.info(f"Using best hyperparameters:")
    logger.info(f"  Learning Rate: {model_info['learning_rate']}")
    logger.info(f"  Threshold: {model_info['threshold_percentile']}%")
    logger.info(f"  Feature Strategy: {model_info['feature_strategy']}")
    
    # Convert dictionary config to PipelineConfig object
    pipeline_config = dict_to_pipeline_config(config)
    
    # Step 1: Data cleaning
    logger.info("Step 1: Data cleaning...")
    cleaner = DataCleaner(pipeline_config)
    df_cleaned = cleaner.clean_data(save_output=True)
    logger.info(f"Data cleaned: {len(df_cleaned)} transactions")
    
    # Step 2: Feature engineering
    logger.info("Step 2: Feature engineering...")
    feature_engineer = FeatureFactory.create(config['features']['strategy'])
    df_features = feature_engineer.generate_features(df_cleaned)
    logger.info(f"Features generated: {len(df_features.columns)} features")
    
    # Step 3: Load existing model
    logger.info("Step 3: Loading existing model...")
    autoencoder = BaselineAutoencoder(pipeline_config)
    scaler_loaded = load_existing_model_and_scaler(autoencoder, model_info)
    logger.info("Model loading completed")
    
    # If scaler wasn't loaded successfully, fit it on the data
    if not scaler_loaded:
        logger.info("Fitting new scaler on data...")
        df_numeric = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in df_numeric.columns:
            df_numeric = df_numeric.drop(columns=['is_fraudulent'])
        autoencoder.scaler.fit(df_numeric.values)
        logger.info("New scaler fitted successfully")
    
    # Create a mock results dict for compatibility with existing code
    results = {
        'roc_auc': model_info.get('roc_auc', 0.0),
        'threshold': model_info.get('threshold_value', 0.0)
    }
    
    # Step 4: Generate predictions
    # Calculate threshold the same way as original training
    threshold_percentile = model_info.get('threshold_percentile', 95)
    
    # Calculate threshold from training data (same as original training)
    logger.info(f"Calculating threshold using {threshold_percentile}th percentile...")
    
    # Get training data for threshold calculation
    df_numeric = df_features.select_dtypes(include=[np.number])
    if 'is_fraudulent' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=['is_fraudulent'])
    
    # Use time-aware split (first 80% train, last 20% test)
    # Data is already sorted by transaction_date, so we can use index for time split
    total_samples = len(df_features)
    train_size = int(0.8 * total_samples)  # 80% for training, 20% for testing
    
    # Split based on temporal order (first 80% for training, last 20% for testing)
    df_train = df_features.iloc[:train_size]
    df_test = df_features.iloc[train_size:]
    df_numeric_train = df_numeric.iloc[:train_size]
    df_numeric_test = df_numeric.iloc[train_size:]
    
    logger.info(f"Time-aware split: train={len(df_train)}, test={len(df_test)}")
    logger.info(f"Train period: transactions 0 to {train_size-1}")
    logger.info(f"Test period: transactions {train_size} to {total_samples-1}")
    
    # Get only normal (non-fraudulent) training data for threshold calculation
    train_normal_mask = df_train['is_fraudulent'] == 0
    df_numeric_train_normal = df_numeric_train[train_normal_mask]
    
    # Calculate anomaly scores on normal training data
    train_anomaly_scores = autoencoder.predict_anomaly_scores(df_numeric_train_normal.values)
    
    # Calculate threshold as 95th percentile of normal training data
    threshold = np.percentile(train_anomaly_scores, threshold_percentile)
    logger.info(f"Threshold calculated: {threshold:.6f} (percentile {threshold_percentile})")
    
    predictions_df = generate_predictions(autoencoder, df_features, threshold, model_info)
    
    # Step 5: Save intermediate files
    save_intermediate_files(autoencoder, df_features, threshold, model_info)
    
    # Step 6: Compute evaluation metrics
    metrics = compute_evaluation_metrics(predictions_df, model_info)
    
    # Step 7: Model evaluation summary
    roc_auc = results.get('roc_auc', 0.0)
    logger.info(f"Model evaluation completed - ROC AUC: {roc_auc:.4f}")
    
    return {
        'roc_auc': roc_auc,
        'config': config,
        'results': results,
        'metrics': metrics,
        'model_info': model_info
    }

def main():
    parser = argparse.ArgumentParser(description="Enhanced Fraud Detection Pipeline (using existing model)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/final_optimized_config.yaml",
        help="Path to the config YAML file to use for the pipeline"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file {args.config} does not exist. Please run the sweep first or provide a valid config.")
        sys.exit(1)
    
    # Check if final_model_info.yaml exists
    if not os.path.exists("models/final_model_info.yaml"):
        print("models/final_model_info.yaml not found. Please run the final model sweep first.")
        sys.exit(1)
    
    loader = ConfigLoader(config_dir=os.path.dirname(args.config) or '.')
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    config = loader.load_config(config_name)
    
    try:
        results = run_pipeline(config)
        print(f"\nEnhanced pipeline completed successfully!")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print(f"Model loaded from: models/autoencoder.h5")
        print(f"Scaler loaded from: models/autoencoder_scaler.pkl")
        print(f"Files saved:")
        print(f"  - predictions/test_predictions.csv")
        print(f"  - predictions/labelled_predictions.csv")
        print(f"  - intermediate/anomaly_scores.csv (test set only)")
        print(f"  - intermediate/latent_space.npy (test set only)")
        print(f"  - intermediate/threshold_info.yaml")
        print(f"  - results/metrics.json")
        print(f"  - results/evaluation_metrics.yaml")
        print(f"Run python src/evaluation/graph_generator.py to create visualizations")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()