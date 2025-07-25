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

def save_model_and_scaler(autoencoder: BaselineAutoencoder, model_info: Dict[str, Any]):
    """Save the trained model and scaler."""
    logger.info("Saving model and scaler...")
    
    # Save model
    model_path = Path("models/final_model.h5")
    autoencoder.model.save(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = Path("models/final_model_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(autoencoder.scaler, f)
    logger.info(f"Scaler saved to: {scaler_path}")
    
    # Update model_info with actual paths
    model_info['model_path'] = str(model_path)
    model_info['scaler_path'] = str(scaler_path)
    
    # Save updated model_info
    model_info_path = Path("models/final_model_info.yaml")
    with open(model_info_path, 'w') as f:
        yaml.dump(model_info, f, default_flow_style=False)
    logger.info(f"Updated model info saved to: {model_info_path}")

def generate_predictions(autoencoder: BaselineAutoencoder, df_features: pd.DataFrame, 
                        threshold: float, model_info: Dict[str, Any]) -> pd.DataFrame:
    """Generate predictions and save to predictions/final_predictions.csv and predictions/labelled_predictions.csv."""
    logger.info("Generating predictions...")
    
    # Get numeric features
    df_numeric = df_features.select_dtypes(include=[np.number])
    if 'is_fraudulent' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=['is_fraudulent'])
    
    # Get anomaly scores
    anomaly_scores = autoencoder.predict_anomaly_scores(df_numeric.values)
    
    # Apply threshold to get binary predictions
    binary_predictions = (anomaly_scores > threshold).astype(int)
    
    # Create predictions dataframe
    predictions_df = df_features.copy()
    predictions_df['anomaly_score'] = anomaly_scores
    predictions_df['predicted_fraud'] = binary_predictions
    predictions_df['predicted_label'] = predictions_df['predicted_fraud'].map({0: 'not_fraud', 1: 'fraud'})
    
    # Save final_predictions.csv (with all original data + predictions)
    predictions_path = Path("predictions/final_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Final predictions saved to: {predictions_path}")
    
    # Save labelled_predictions.csv (simplified version with key columns)
    labelled_columns = ['anomaly_score', 'predicted_fraud', 'predicted_label']
    if 'is_fraudulent' in predictions_df.columns:
        labelled_columns.insert(0, 'is_fraudulent')
    
    labelled_df = predictions_df[labelled_columns].copy()
    labelled_path = Path("predictions/labelled_predictions.csv")
    labelled_df.to_csv(labelled_path, index=False)
    logger.info(f"Labelled predictions saved to: {labelled_path}")
    
    return predictions_df

def save_intermediate_files(autoencoder: BaselineAutoencoder, df_features: pd.DataFrame, 
                          threshold: float, model_info: Dict[str, Any]):
    """Save intermediate files for visualizations."""
    logger.info("Saving intermediate files...")
    
    # Get numeric features
    df_numeric = df_features.select_dtypes(include=[np.number])
    if 'is_fraudulent' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=['is_fraudulent'])
    
    # Save anomaly scores
    anomaly_scores = autoencoder.predict_anomaly_scores(df_numeric.values)
    anomaly_df = pd.DataFrame({
        'anomaly_score': anomaly_scores,
        'is_fraudulent': df_features['is_fraudulent'] if 'is_fraudulent' in df_features.columns else None
    })
    anomaly_path = Path("intermediate/anomaly_scores.csv")
    anomaly_df.to_csv(anomaly_path, index=False)
    logger.info(f"Anomaly scores saved to: {anomaly_path}")
    
    # Save latent space (get encoder part of autoencoder)
    X_scaled = autoencoder.scaler.transform(df_numeric.values)
    encoder = autoencoder.model.layers[0]  # First layer is encoder
    latent_space = encoder.predict(X_scaled)
    latent_path = Path("intermediate/latent_space.npy")
    np.save(latent_path, latent_space)
    logger.info(f"Latent space saved to: {latent_path}")
    
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
    """Run the complete fraud detection pipeline with given config."""
    logger.info(f"Starting enhanced fraud detection pipeline...")
    
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
    
    # Step 3: Model training
    logger.info("Step 3: Model training...")
    autoencoder = BaselineAutoencoder(pipeline_config)
    results = autoencoder.train()
    logger.info("Model training completed")
    
    # Extract training history for loss curve generation
    if 'history' in results:
        history = results['history']
        training_history = {
            'loss': history.history['loss'],
            'val_loss': history.history.get('val_loss', []),
            'reconstruction_error': history.history.get('reconstruction_error', [])
        }
        model_info['training_history'] = training_history
        logger.info(f"Training history saved: {len(training_history['loss'])} epochs")
    
    # Step 4: Save model and scaler
    save_model_and_scaler(autoencoder, model_info)
    
    # Step 5: Generate predictions
    threshold = model_info.get('threshold_value', results.get('threshold', 0))
    predictions_df = generate_predictions(autoencoder, df_features, threshold, model_info)
    
    # Step 6: Save intermediate files
    save_intermediate_files(autoencoder, df_features, threshold, model_info)
    
    # Step 7: Compute evaluation metrics
    metrics = compute_evaluation_metrics(predictions_df, model_info)
    
    # Step 8: Model evaluation summary
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
    parser = argparse.ArgumentParser(description="Enhanced Fraud Detection Pipeline")
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
        print(f"Files saved:")
        print(f"  - predictions/final_predictions.csv")
        print(f"  - predictions/labelled_predictions.csv")
        print(f"  - intermediate/anomaly_scores.csv")
        print(f"  - intermediate/latent_space.npy")
        print(f"  - intermediate/threshold_info.yaml")
        print(f"  - results/metrics.json")
        print(f"  - results/evaluation_metrics.yaml")
        print(f"  - models/final_model.h5")
        print(f"  - models/final_model_scaler.pkl")
        print(f"Run python src/evaluation/graph_generator.py to create visualizations")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()