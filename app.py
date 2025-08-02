#!/usr/bin/env python3
"""
Fraud Detection Inference Server - Production Ready
Simplified inference server that loads trained model and engineered test data for predictions.
"""

import os
import sys
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from contextlib import asynccontextmanager

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from src.config_loader import ConfigLoader
from src.models.autoencoder import FraudAutoencoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/inference.log')
    ]
)
logger = logging.getLogger(__name__)

# Global variables for loaded components
fraud_detector: Optional[FraudAutoencoder] = None
inference_config: Optional[Dict] = None
engineered_test_data: Optional[pd.DataFrame] = None
anomaly_scores: Optional[np.ndarray] = None


def load_inference_config(config_path: str = "configs/inference_config.yaml") -> Dict:
    """1. Load configuration from dedicated inference config file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Inference config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded inference config from: {config_path}")
    return config


def load_trained_components(config: Dict) -> FraudAutoencoder:
    """2. Load trained autoencoder model and scaler from disk."""
    logger.info("Loading trained components...")
    
    # Load autoencoder model
    model_path = config['inference']['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    autoencoder = FraudAutoencoder({})
    autoencoder.load_model(model_path)
    logger.info(f"Loaded trained autoencoder model from: {model_path}")
    
    # Verify scaler is loaded with the model
    if not hasattr(autoencoder, 'scaler') or autoencoder.scaler is None:
        raise ValueError("Model scaler not found - ensure model was saved with scaler")
    
    logger.info(f"Model scaler loaded with {autoencoder.scaler.n_features_in_} features")
    return autoencoder


def load_engineered_test_data(config: Dict) -> pd.DataFrame:
    """3. Load engineered test data (already processed by main.py pipeline)."""
    logger.info("Loading engineered test data...")
    
    engineered_data_path = config['inference']['engineered_test_data_path']
    if not os.path.exists(engineered_data_path):
        raise FileNotFoundError(f"Engineered test data not found: {engineered_data_path}")
    
    # Load the engineered test data
    df_engineered = pd.read_csv(engineered_data_path)
    
    logger.info(f"Loaded engineered test data: {df_engineered.shape}")
    logger.info(f"Columns: {list(df_engineered.columns)}")
    
    return df_engineered


def scale_test_data(autoencoder: FraudAutoencoder, df_engineered: pd.DataFrame) -> np.ndarray:
    """4. Scale the test data using the already-fitted scaler."""
    logger.info("Scaling test data...")
    
    # Prepare feature columns (exclude non-feature columns)
    feature_columns = [col for col in df_engineered.columns 
                      if col not in ['transaction_id', 'is_fraudulent']]
    
    X_engineered = df_engineered[feature_columns].values
    
    logger.info(f"Feature columns: {len(feature_columns)}")
    logger.info(f"Engineered features shape: {X_engineered.shape}")
    
    # Handle feature count mismatch
    expected_features = autoencoder.scaler.n_features_in_
    actual_features = X_engineered.shape[1]
    
    if actual_features != expected_features:
        logger.warning(f"Feature count mismatch: expected {expected_features}, got {actual_features}")
        
        if actual_features < expected_features:
            # Add missing features with zeros
            missing_features = expected_features - actual_features
            padding = np.zeros((X_engineered.shape[0], missing_features))
            X_engineered = np.hstack([X_engineered, padding])
            logger.info(f"Added {missing_features} zero-padded features")
        else:
            # Remove extra features (take first expected_features)
            X_engineered = X_engineered[:, :expected_features]
            logger.info(f"Removed {actual_features - expected_features} extra features")
    
    # Scale using fitted scaler (do not refit)
    X_scaled = autoencoder.scaler.transform(X_engineered)
    
    logger.info(f"Test data scaled: {X_scaled.shape}")
    return X_scaled


def generate_predictions(autoencoder: FraudAutoencoder, X_scaled: np.ndarray) -> np.ndarray:
    """5. Generate model predictions (reconstructions)."""
    logger.info("Generating model predictions...")
    
    # Get reconstructions from the autoencoder
    reconstructions = autoencoder.model.predict(X_scaled, verbose=0)
    
    logger.info(f"Generated reconstructions: {reconstructions.shape}")
    return reconstructions


def calculate_reconstruction_errors(X_scaled: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """6. Calculate reconstruction errors (mean squared error per row)."""
    logger.info("Calculating reconstruction errors...")
    
    # Calculate mean squared error for each sample
    reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
    
    logger.info(f"Calculated reconstruction errors: {len(reconstruction_errors)} samples")
    logger.info(f"Error range: {np.min(reconstruction_errors):.6f} - {np.max(reconstruction_errors):.6f}")
    logger.info(f"Mean error: {np.mean(reconstruction_errors):.6f}")
    
    return reconstruction_errors


def get_anomaly_threshold(config: Dict, reconstruction_errors: np.ndarray) -> float:
    """Get anomaly threshold from config or calculate from errors."""
    threshold_percentile = config['inference'].get('anomaly_threshold_percentile', 95.0)
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    
    logger.info(f"Anomaly threshold ({threshold_percentile}th percentile): {threshold:.6f}")
    return threshold


def classify_anomalies(reconstruction_errors: np.ndarray, threshold: float, 
                      df_engineered: pd.DataFrame) -> List[Dict]:
    """Classify transactions as normal or anomaly based on threshold."""
    logger.info("Classifying anomalies...")
    
    # Classify based on threshold
    predictions = (reconstruction_errors > threshold).astype(int)
    
    # Create results
    results = []
    for i, (_, row) in enumerate(df_engineered.iterrows()):
        # Calculate normalized fraud probability (0-100%)
        max_error = np.max(reconstruction_errors)
        min_error = np.min(reconstruction_errors)
        normalized_error = (reconstruction_errors[i] - min_error) / (max_error - min_error) if max_error > min_error else 0.5
        fraud_probability = normalized_error * 100
        
        result = {
            "transaction_id": str(row.get('transaction_id', f"TXN_{i}")),
            "reconstruction_error": float(reconstruction_errors[i]),
            "predicted_fraud": bool(predictions[i]),
            "actual_fraud": bool(row.get('is_fraudulent', False)),
            "fraud_probability": fraud_probability,
            "amount": float(row.get('amount', 0)),
            "time": float(row.get('time', 0))
        }
        results.append(result)
    
    # Sort by reconstruction error (highest first)
    results.sort(key=lambda x: x["reconstruction_error"], reverse=True)
    
    fraud_detected = sum(1 for r in results if r["predicted_fraud"])
    actual_fraud = sum(1 for r in results if r["actual_fraud"])
    
    logger.info(f"Classified {len(results)} transactions")
    logger.info(f"Fraud detected: {fraud_detected}, Actual fraud: {actual_fraud}")
    
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global fraud_detector, inference_config, engineered_test_data, anomaly_scores
    
    # Startup
    logger.info("Starting Fraud Detection Inference Server...")
    
    try:
        # 1. Load configuration
        inference_config = load_inference_config()
        
        # 2. Load trained components
        fraud_detector = load_trained_components(inference_config)
        
        # 3. Load engineered test data
        engineered_test_data = load_engineered_test_data(inference_config)
        
        # 4. Scale test data
        X_scaled = scale_test_data(fraud_detector, engineered_test_data)
        
        # 5. Generate predictions
        reconstructions = generate_predictions(fraud_detector, X_scaled)
        
        # 6. Calculate reconstruction errors
        anomaly_scores = calculate_reconstruction_errors(X_scaled, reconstructions)
        
        logger.info("All inference components loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load components: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Fraud Detection Inference Server...")


# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection Inference Server",
    description="Inference server using trained autoencoder for fraud detection",
    version="1.0.0",
    lifespan=lifespan
)

# Setup templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": fraud_detector is not None,
        "data_loaded": engineered_test_data is not None,
        "scores_calculated": anomaly_scores is not None,
        "transaction_count": len(engineered_test_data) if engineered_test_data is not None else 0
    }


@app.get("/api/predict")
async def get_predictions():
    """Get all predictions with current threshold."""
    if fraud_detector is None or engineered_test_data is None or anomaly_scores is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    threshold = get_anomaly_threshold(inference_config, anomaly_scores)
    results = classify_anomalies(anomaly_scores, threshold, engineered_test_data)
    
    return {
        "threshold": threshold,
        "threshold_percentile": inference_config['inference'].get('anomaly_threshold_percentile', 95.0),
        "predictions": results[:100],  # Return top 100 results
        "summary": {
            "total_transactions": len(results),
            "fraud_detected": sum(1 for r in results if r["predicted_fraud"]),
            "actual_fraud": sum(1 for r in results if r["actual_fraud"]),
            "true_positives": sum(1 for r in results if r["predicted_fraud"] and r["actual_fraud"]),
            "false_positives": sum(1 for r in results if r["predicted_fraud"] and not r["actual_fraud"]),
            "true_negatives": sum(1 for r in results if not r["predicted_fraud"] and not r["actual_fraud"]),
            "false_negatives": sum(1 for r in results if not r["predicted_fraud"] and r["actual_fraud"])
        }
    }


@app.get("/api/predict/threshold/{threshold}")
async def get_predictions_with_threshold(threshold: float):
    """Get predictions using custom threshold."""
    if fraud_detector is None or engineered_test_data is None or anomaly_scores is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    results = classify_anomalies(anomaly_scores, threshold, engineered_test_data)
    
    return {
        "threshold": threshold,
        "predictions": results[:100],  # Return top 100 results
        "summary": {
            "total_transactions": len(results),
            "fraud_detected": sum(1 for r in results if r["predicted_fraud"]),
            "actual_fraud": sum(1 for r in results if r["actual_fraud"]),
            "true_positives": sum(1 for r in results if r["predicted_fraud"] and r["actual_fraud"]),
            "false_positives": sum(1 for r in results if r["predicted_fraud"] and not r["actual_fraud"]),
            "true_negatives": sum(1 for r in results if not r["predicted_fraud"] and not r["actual_fraud"]),
            "false_negatives": sum(1 for r in results if not r["predicted_fraud"] and r["actual_fraud"])
        }
    }


@app.get("/api/stats")
async def get_statistics():
    """Get reconstruction error statistics."""
    if anomaly_scores is None:
        raise HTTPException(status_code=503, detail="Anomaly scores not calculated")
    
    return {
        "reconstruction_errors": {
            "min": float(np.min(anomaly_scores)),
            "max": float(np.max(anomaly_scores)),
            "mean": float(np.mean(anomaly_scores)),
            "std": float(np.std(anomaly_scores)),
            "median": float(np.median(anomaly_scores))
        },
        "percentiles": {
            "25": float(np.percentile(anomaly_scores, 25)),
            "50": float(np.percentile(anomaly_scores, 50)),
            "75": float(np.percentile(anomaly_scores, 75)),
            "90": float(np.percentile(anomaly_scores, 90)),
            "95": float(np.percentile(anomaly_scores, 95)),
            "99": float(np.percentile(anomaly_scores, 99))
        }
    }


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 