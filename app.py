#!/usr/bin/env python3
"""
Fraud Detection Inference Server - Production Ready
Simplified inference server that loads trained model and engineered test data for predictions.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from src.models.autoencoder import FraudAutoencoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for inference components
fraud_detector: Optional[FraudAutoencoder] = None
inference_config: Optional[Dict] = None
engineered_test_data: Optional[pd.DataFrame] = None
raw_anomaly_scores: Optional[np.ndarray] = None
ground_truth_labels: Optional[pd.Series] = None


def load_inference_config(config_path: str = "configs/inference_config.yaml") -> Dict:
    """1. Load inference configuration."""
    import yaml
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    logger.info(f"Loaded inference config from: {config_path}")
    return config


def load_trained_components(config: Dict) -> FraudAutoencoder:
    """2. Load trained autoencoder model and scaler."""
    model_path = config['inference']['model_path']
    scaler_path = config['inference']['scaler_path']
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    # Load the autoencoder with trained model and scaler
    autoencoder = FraudAutoencoder({})  # Empty config for inference
    autoencoder.load_model(model_path)
    
    logger.info(f"Loaded trained autoencoder model from: {model_path}")
    logger.info(f"Model scaler loaded with {autoencoder.scaler.n_features_in_} features")
    
    return autoencoder


def load_engineered_test_data(config: Dict) -> pd.DataFrame:
    """3. Load engineered test data (features only, no fraud labels)."""
    logger.info("Loading engineered test data...")
    
    engineered_data_path = config['inference']['engineered_test_data_path']
    if not os.path.exists(engineered_data_path):
        raise FileNotFoundError(f"Engineered test data not found: {engineered_data_path}")
    
    # Load the engineered test data with index column
    df_engineered = pd.read_csv(engineered_data_path, index_col=0)
    
    # Apply sampling if configured for faster dashboard response
    sample_size = config['inference'].get('test_data_sample_size', 1.0)
    if sample_size < 1.0:
        original_size = len(df_engineered)
        sample_count = int(original_size * sample_size)
        df_engineered = df_engineered.sample(n=sample_count, random_state=42)
        logger.info(f"Sampled {sample_count} transactions ({sample_size*100:.1f}% of {original_size} total)")
    
    logger.info(f"Loaded engineered test data: {df_engineered.shape}")
    logger.info(f"Columns: {list(df_engineered.columns)}")
    
    return df_engineered


def load_ground_truth_labels(config: Dict) -> Optional[pd.Series]:
    """Load ground truth fraud labels separately for evaluation only."""
    logger.info("Loading ground truth labels for evaluation...")
    
    try:
        # Try to load from test labels file first (properly aligned)
        test_labels_path = "data/engineered/test_labels.csv"
        if os.path.exists(test_labels_path):
            df_labels = pd.read_csv(test_labels_path)
            if 'is_fraudulent' in df_labels.columns:
                # Create a mapping from index to label
                labels_dict = dict(zip(df_labels['index'], df_labels['is_fraudulent']))
                logger.info(f"Loaded ground truth labels from test labels: {len(labels_dict)} samples")
                logger.info(f"Fraud distribution: {pd.Series(list(labels_dict.values())).value_counts().to_dict()}")
                return labels_dict
        
        # Fallback to cleaned data (less reliable alignment)
        cleaned_data_path = "data/cleaned/creditcard_cleaned.csv"
        if os.path.exists(cleaned_data_path):
            df_cleaned = pd.read_csv(cleaned_data_path)
            if 'is_fraudulent' in df_cleaned.columns:
                labels = df_cleaned['is_fraudulent']
                logger.info(f"Loaded ground truth labels from cleaned data: {len(labels)} samples")
                logger.info(f"Fraud distribution: {labels.value_counts().to_dict()}")
                return labels
        
        # Fallback to raw data
        raw_data_path = "data/raw/creditcard.csv"
        if os.path.exists(raw_data_path):
            df_raw = pd.read_csv(raw_data_path)
            if 'Class' in df_raw.columns:
                labels = df_raw['Class']
                logger.info(f"Loaded ground truth labels from raw data: {len(labels)} samples")
                logger.info(f"Fraud distribution: {labels.value_counts().to_dict()}")
                return labels
        
        logger.warning("No ground truth labels found - evaluation metrics will not be available")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to load ground truth labels: {e}")
        return None


def scale_test_data(autoencoder: FraudAutoencoder, df_engineered: pd.DataFrame) -> np.ndarray:
    """4. Scale the test data using the already-fitted scaler (features only)."""
    logger.info("Scaling test data...")
    
    # Prepare feature columns (exclude non-feature columns)
    # IMPORTANT: Never include fraud labels in feature scaling
    feature_columns = [col for col in df_engineered.columns 
                      if col not in ['transaction_id', 'is_fraudulent', 'Class']]
    
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


def calculate_raw_reconstruction_errors(X_scaled: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """6. Calculate raw reconstruction errors (no normalization)."""
    logger.info("Calculating raw reconstruction errors...")
    
    # Calculate mean squared error for each sample (raw scores)
    reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
    
    logger.info(f"Calculated reconstruction errors: {len(reconstruction_errors)} samples")
    logger.info(f"Raw error range: {np.min(reconstruction_errors):.6f} - {np.max(reconstruction_errors):.6f}")
    logger.info(f"Mean error: {np.mean(reconstruction_errors):.6f}")
    
    return reconstruction_errors


def get_anomaly_threshold(config: Dict, reconstruction_errors: np.ndarray) -> float:
    """Get anomaly threshold from config or calculate from errors."""
    threshold_percentile = config['inference'].get('anomaly_threshold_percentile', 95.0)
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    
    logger.info(f"Anomaly threshold ({threshold_percentile}th percentile): {threshold:.6f}")
    return threshold


def classify_anomalies(reconstruction_errors: np.ndarray, threshold: float, 
                      df_engineered: pd.DataFrame, ground_truth: Optional[pd.Series] = None) -> List[Dict]:
    """Classify transactions as normal or anomaly based on threshold."""
    logger.info("Classifying anomalies...")
    
    # Classify based on threshold
    predictions = (reconstruction_errors > threshold).astype(int)
    
    # Create results
    results = []
    for i, (_, row) in enumerate(df_engineered.iterrows()):
        # Use raw reconstruction error as anomaly score
        raw_anomaly_score = float(reconstruction_errors[i])
        
        # Get engineered features for this transaction
        engineered_features = {}
        for col in df_engineered.columns:
            if col not in ['transaction_id', 'amount', 'time', 'is_fraudulent', 'Class']:
                engineered_features[col] = float(row.get(col, 0))
        
        # Get ground truth label if available
        actual_fraud = None
        if ground_truth is not None:
            # Get the original index from the test data
            original_index = df_engineered.index[i]
            if original_index in ground_truth:
                actual_fraud = bool(ground_truth[original_index])
                # Debug: log first few fraud cases
                if actual_fraud and i < 10:
                    logger.info(f"Found actual fraud at index {i}, original_index {original_index}")
        
        result = {
            "transaction_id": str(row.get('transaction_id', f"TXN_{i}")),
            "reconstruction_error": raw_anomaly_score,
            "anomaly_score": raw_anomaly_score,  # Raw score, not normalized
            "predicted_fraud": bool(predictions[i]),
            "actual_fraud": actual_fraud,
            "fraud_probability": raw_anomaly_score,  # Use raw score as probability
            "amount": float(row.get('amount', 0)),
            "time": float(row.get('time', 0)),
            "engineered_features": engineered_features
        }
        results.append(result)
    
    # Sort by reconstruction error (highest first)
    results.sort(key=lambda x: x["reconstruction_error"], reverse=True)
    
    fraud_detected = sum(1 for r in results if r["predicted_fraud"])
    
    if ground_truth is not None:
        actual_fraud_count = sum(1 for r in results if r["actual_fraud"])
        logger.info(f"Classified {len(results)} transactions")
        logger.info(f"Fraud detected: {fraud_detected}, Actual fraud: {actual_fraud_count}")
        # Debug: log some sample indices
        logger.info(f"Sample indices checked: {list(df_engineered.index[:5])}")
        logger.info(f"Ground truth keys sample: {list(ground_truth.keys())[:5]}")
    else:
        logger.info(f"Classified {len(results)} transactions (no ground truth labels)")
        logger.info(f"Fraud detected: {fraud_detected}")
    
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global fraud_detector, inference_config, engineered_test_data, raw_anomaly_scores, ground_truth_labels
    
    # Startup
    logger.info("Starting Fraud Detection Inference Server...")
    
    try:
        # 1. Load configuration
        inference_config = load_inference_config()
        
        # 2. Load trained components
        fraud_detector = load_trained_components(inference_config)
        
        # 3. Load engineered test data (features only)
        engineered_test_data = load_engineered_test_data(inference_config)
        
        # 4. Load ground truth labels separately (for evaluation only)
        ground_truth_labels = load_ground_truth_labels(inference_config)
        
        # 5. Scale test data (features only, no fraud labels)
        X_scaled = scale_test_data(fraud_detector, engineered_test_data)
        
        # 6. Generate predictions
        reconstructions = generate_predictions(fraud_detector, X_scaled)
        
        # 7. Calculate raw reconstruction errors
        raw_anomaly_scores = calculate_raw_reconstruction_errors(X_scaled, reconstructions)
        
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
        "anomaly_scores_loaded": raw_anomaly_scores is not None,
        "ground_truth_loaded": ground_truth_labels is not None
    }


@app.get("/api/predict")
async def get_predictions():
    """Get all predictions with default threshold."""
    if raw_anomaly_scores is None or engineered_test_data is None:
        return {"error": "Model not loaded"}
    
    threshold = get_anomaly_threshold(inference_config, raw_anomaly_scores)
    results = classify_anomalies(raw_anomaly_scores, threshold, engineered_test_data, ground_truth_labels)
    
    return {
        "threshold": threshold,
        "predictions": results[:100],  # Return top 100
        "summary": {
            "total_transactions": len(results),
            "fraud_detected": sum(1 for r in results if r["predicted_fraud"]),
            "actual_fraud": sum(1 for r in results if r["actual_fraud"]) if ground_truth_labels is not None else None,
            "true_positives": sum(1 for r in results if r["predicted_fraud"] and r["actual_fraud"]) if ground_truth_labels is not None else None,
            "false_positives": sum(1 for r in results if r["predicted_fraud"] and not r["actual_fraud"]) if ground_truth_labels is not None else None,
            "true_negatives": sum(1 for r in results if not r["predicted_fraud"] and not r["actual_fraud"]) if ground_truth_labels is not None else None,
            "false_negatives": sum(1 for r in results if not r["predicted_fraud"] and r["actual_fraud"]) if ground_truth_labels is not None else None
        }
    }


@app.get("/api/predict/threshold/{threshold}")
async def get_predictions_with_threshold(threshold: float):
    """Get predictions with custom threshold."""
    if raw_anomaly_scores is None or engineered_test_data is None:
        return {"error": "Model not loaded"}
    
    results = classify_anomalies(raw_anomaly_scores, threshold, engineered_test_data, ground_truth_labels)
    
    # For high thresholds (corresponding to high percentiles), return all flagged transactions
    # For lower thresholds, limit to first 100 to avoid overwhelming the frontend
    if len(results) <= 1000:  # If we have 1000 or fewer flagged transactions, return all
        predictions_to_return = results
    else:
        predictions_to_return = results[:100]  # Return top 100
    
    return {
        "threshold": threshold,
        "predictions": predictions_to_return,
        "total_predictions": len(results),  # Total number of flagged transactions
        "returned_predictions": len(predictions_to_return),  # Number of transactions returned
        "summary": {
            "total_transactions": len(results),
            "fraud_detected": sum(1 for r in results if r["predicted_fraud"]),
            "actual_fraud": sum(1 for r in results if r["actual_fraud"]) if ground_truth_labels is not None else None,
            "true_positives": sum(1 for r in results if r["predicted_fraud"] and r["actual_fraud"]) if ground_truth_labels is not None else None,
            "false_positives": sum(1 for r in results if r["predicted_fraud"] and not r["actual_fraud"]) if ground_truth_labels is not None else None,
            "true_negatives": sum(1 for r in results if not r["predicted_fraud"] and not r["actual_fraud"]) if ground_truth_labels is not None else None,
            "false_negatives": sum(1 for r in results if not r["predicted_fraud"] and r["actual_fraud"]) if ground_truth_labels is not None else None
        }
    }


@app.get("/api/stats")
async def get_statistics():
    """Get anomaly score statistics."""
    if raw_anomaly_scores is None:
        return {"error": "Anomaly scores not loaded"}
    
    # Calculate fraud percentages
    total_fraud_rate = None
    test_fraud_rate = None
    
    if ground_truth_labels is not None:
        # Calculate fraud rate in test set
        test_fraud_count = sum(ground_truth_labels.values())
        test_total_count = len(ground_truth_labels)
        test_fraud_rate = (test_fraud_count / test_total_count) * 100 if test_total_count > 0 else 0
        
        # Try to get total dataset fraud rate from cleaned data
        try:
            cleaned_data_path = "data/cleaned/creditcard_cleaned.csv"
            if os.path.exists(cleaned_data_path):
                df_cleaned = pd.read_csv(cleaned_data_path)
                if 'is_fraudulent' in df_cleaned.columns:
                    total_fraud_count = df_cleaned['is_fraudulent'].sum()
                    total_count = len(df_cleaned)
                    total_fraud_rate = (total_fraud_count / total_count) * 100 if total_count > 0 else 0
        except Exception as e:
            logger.warning(f"Could not calculate total fraud rate: {e}")
    
    return {
        "min_score": float(np.min(raw_anomaly_scores)),
        "max_score": float(np.max(raw_anomaly_scores)),
        "mean_score": float(np.mean(raw_anomaly_scores)),
        "median_score": float(np.median(raw_anomaly_scores)),
        "std_score": float(np.std(raw_anomaly_scores)),
        "percentiles": {
            "25th": float(np.percentile(raw_anomaly_scores, 25)),
            "50th": float(np.percentile(raw_anomaly_scores, 50)),
            "75th": float(np.percentile(raw_anomaly_scores, 75)),
            "90th": float(np.percentile(raw_anomaly_scores, 90)),
            "95th": float(np.percentile(raw_anomaly_scores, 95)),
            "99th": float(np.percentile(raw_anomaly_scores, 99))
        },
        "fraud_rates": {
            "total_dataset_fraud_percent": total_fraud_rate,
            "test_set_fraud_percent": test_fraud_rate
        }
    }


@app.get("/api/percentiles")
async def get_percentiles():
    """Get available percentiles for threshold selection."""
    if raw_anomaly_scores is None:
        return {"error": "Anomaly scores not loaded"}
    
    percentiles = [50, 75, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99]
    thresholds = {}
    
    for p in percentiles:
        thresholds[p] = float(np.percentile(raw_anomaly_scores, p))
    
    return {
        "percentiles": percentiles,
        "thresholds": thresholds
    }


@app.get("/api/predict/percentile/{percentile}")
async def get_predictions_with_percentile(percentile: int):
    """Get predictions using percentile-based threshold."""
    if raw_anomaly_scores is None or engineered_test_data is None:
        return {"error": "Model not loaded"}
    
    if percentile < 0 or percentile > 100:
        return {"error": "Percentile must be between 0 and 100"}
    
    # Calculate threshold based on percentile
    threshold = np.percentile(raw_anomaly_scores, percentile)
    
    # Make predictions: transactions above threshold are flagged as fraud
    predictions = (raw_anomaly_scores > threshold).astype(int)
    
    # Get indices of transactions flagged as fraud (above threshold)
    fraud_indices = np.where(predictions == 1)[0]
    
    if len(fraud_indices) == 0:
        # No transactions flagged as fraud
        return {
            "threshold": threshold,
            "percentile": percentile,
            "predictions": [],
            "total_predictions": 0,
            "returned_predictions": 0,
            "summary": {
                "total_transactions": len(raw_anomaly_scores),
                "fraud_detected": 0,
                "actual_fraud": 0 if ground_truth_labels is not None else None,
                "true_positives": 0 if ground_truth_labels is not None else None,
                "false_positives": 0 if ground_truth_labels is not None else None,
                "true_negatives": len(raw_anomaly_scores) if ground_truth_labels is not None else None,
                "false_negatives": 0 if ground_truth_labels is not None else None
            }
        }
    
    # Sort fraud indices by anomaly score (highest first)
    sorted_fraud_indices = fraud_indices[np.argsort(raw_anomaly_scores[fraud_indices])[::-1]]
    
    # Limit results for performance
    max_results = 1000 if percentile >= 95 else 100
    selected_indices = sorted_fraud_indices[:max_results]
    
    # Define PCA and professional features for display
    display_features = [
        # Core transaction features
        'amount', 'time',
        
        # PCA features (V1-V28)
        'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',
        'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
        'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28',
        
        # Professional business features
        'amount_log', 'amount_sqrt', 'amount_scaled',
        'amount_above_95', 'amount_ratio_to_median',
        'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
        'comprehensive_risk_score', 'high_risk'
    ]
    
    # Create results for selected transactions only
    results = []
    for i in selected_indices:
        row = engineered_test_data.iloc[i]
        raw_anomaly_score = float(raw_anomaly_scores[i])
        
        # Get only PCA and professional features
        engineered_features = {}
        for col in display_features:
            if col in engineered_test_data.columns:
                engineered_features[col] = float(row.get(col, 0))
        
        # Get ground truth label if available
        actual_fraud = None
        if ground_truth_labels is not None:
            original_index = engineered_test_data.index[i]
            if original_index in ground_truth_labels:
                actual_fraud = bool(ground_truth_labels[original_index])
        
        result = {
            "transaction_id": str(row.get('transaction_id', f"TXN_{i}")),
            "reconstruction_error": raw_anomaly_score,
            "anomaly_score": raw_anomaly_score,
            "predicted_fraud": bool(predictions[i]),  # Use actual prediction
            "actual_fraud": actual_fraud,
            "fraud_probability": raw_anomaly_score,
            "amount": float(row.get('amount', 0)),
            "time": float(row.get('time', 0)),
            "engineered_features": engineered_features
        }
        results.append(result)
    
    # Calculate summary statistics
    total_transactions = len(raw_anomaly_scores)
    fraud_detected = len(fraud_indices)
    
    # Calculate actual fraud count if ground truth is available
    actual_fraud_count = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    if ground_truth_labels is not None:
        actual_fraud_count = sum(ground_truth_labels.values())
        
        # Calculate confusion matrix
        for i in range(len(raw_anomaly_scores)):
            predicted_fraud = predictions[i] == 1
            original_index = engineered_test_data.index[i]
            actual_fraud = ground_truth_labels.get(original_index, False)
            
            if predicted_fraud and actual_fraud:
                true_positives += 1
            elif predicted_fraud and not actual_fraud:
                false_positives += 1
            elif not predicted_fraud and not actual_fraud:
                true_negatives += 1
            elif not predicted_fraud and actual_fraud:
                false_negatives += 1
    
    return {
        "threshold": threshold,
        "percentile": percentile,
        "predictions": results,
        "total_predictions": fraud_detected,
        "returned_predictions": len(results),
        "summary": {
            "total_transactions": total_transactions,
            "fraud_detected": fraud_detected,
            "actual_fraud": actual_fraud_count if ground_truth_labels is not None else None,
            "true_positives": true_positives if ground_truth_labels is not None else None,
            "false_positives": false_positives if ground_truth_labels is not None else None,
            "true_negatives": true_negatives if ground_truth_labels is not None else None,
            "false_negatives": false_negatives if ground_truth_labels is not None else None
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 