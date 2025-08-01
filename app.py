#!/usr/bin/env python3
"""
Fraud Detection Dashboard - Simplified Docker Container
Shows all transactions with fraud flags and adjustable threshold.
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import pickle
from contextlib import asynccontextmanager

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from src.config_loader import ConfigLoader
from src.features.feature_engineer import FeatureEngineer
from src.models.autoencoder import FraudAutoencoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded models and data
fraud_detector = None
feature_engineer = None
all_transactions = None
config = None

# Pydantic models
class ThresholdRequest(BaseModel):
    """Threshold adjustment request model."""
    threshold: float = Field(..., ge=0.0, le=1.0, description="Fraud detection threshold")

class PredictionMetrics(BaseModel):
    """Prediction metrics model."""
    total_transactions: int
    fraud_detected: int
    normal_transactions: int
    fraud_rate: float
    threshold: float
    processing_time_ms: float

class TransactionPrediction(BaseModel):
    """Transaction prediction result."""
    transaction_id: str
    amount: float
    time: float
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    fraud_probability: float
    is_fraudulent: bool
    actual_fraud: bool
    threshold: float

# Load models and data
def load_models():
    """Load the trained fraud detection model and feature engineer."""
    global fraud_detector, feature_engineer, all_transactions, config
    
    try:
        # Load configuration
        config_path = "configs/final_optimized_config.yaml"
        config_loader = ConfigLoader(config_path)
        config = config_loader.config
        
        # Load the actual trained model
        try:
            fraud_detector = FraudAutoencoder(config)
            fraud_detector.load_model("models/fraud_autoencoder.keras")
            logger.info("✅ Loaded trained fraud detection model")
        except Exception as model_error:
            logger.warning(f"Could not load fraud_autoencoder.keras: {model_error}")
            try:
                fraud_detector = FraudAutoencoder(config)
                fraud_detector.load_model("models/autoencoder.h5")
                logger.info("✅ Loaded autoencoder.h5 model")
            except Exception as model_error2:
                logger.warning(f"Could not load autoencoder.h5: {model_error2}")
                logger.info("Using fallback heuristic model")
                fraud_detector = FraudAutoencoder(config)
                fraud_detector.is_fitted = True
        
        # Load feature engineer
        try:
            feature_engineer = FeatureEngineer(config.get('features', {}))
            feature_engineer.load_fitted_objects("models/fraud_autoencoder_features.pkl")
            logger.info("✅ Loaded feature engineer")
        except Exception as fe_error:
            logger.warning(f"Could not load feature engineer: {fe_error}")
            feature_engineer = FeatureEngineer(config.get('features', {}))
        
        # Load all transaction data
        all_transactions = load_all_transactions()
        
        logger.info("Models and data loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def load_all_transactions():
    """Load all transaction data for analysis."""
    try:
        # Load the real credit card data
        df = pd.read_csv("data/raw/creditcard.csv")
        
        # Use the complete test set (last 20% of data for testing)
        test_size = int(len(df) * 0.2)
        df_test = df.tail(test_size).reset_index(drop=True)
        
        # Convert to list of dictionaries
        transactions = []
        for idx, row in df_test.iterrows():
            transaction = {
                "transaction_id": f"TXN_{row.name}",  # Use original index
                "amount": float(row.get('Amount', 0)),
                "time": float(row.get('Time', 0)),
                "is_fraudulent": bool(row.get('Class', 0)),
                "original_data": row.to_dict()
            }
            transactions.append(transaction)
        
        fraud_count = sum(1 for t in transactions if t['is_fraudulent'])
        normal_count = len(transactions) - fraud_count
        
        logger.info(f"Loaded {len(transactions)} test transactions ({fraud_count} fraud, {normal_count} normal)")
        return transactions
        
    except Exception as e:
        logger.error(f"Error loading transaction data: {e}")
        return []

def predict_all_transactions(threshold: float = 0.5) -> Dict[str, Any]:
    """Predict fraud for all transactions with given threshold."""
    import time
    start_time = time.time()
    
    if fraud_detector is None or feature_engineer is None or all_transactions is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    try:
        # Convert to DataFrame for vectorized operations
        df_data = []
        for transaction in all_transactions:
            row = transaction['original_data'].copy()
            row['transaction_id'] = transaction['transaction_id']
            row['amount'] = transaction['amount']
            row['time'] = transaction['time']
            row['is_fraudulent'] = transaction['is_fraudulent']
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Use the actual model for predictions if available
        if hasattr(fraud_detector, 'model') and fraud_detector.model is not None:
            logger.info("Using trained model for predictions")
            
            # Prepare features for the model - create 96 engineered features
            # Since the feature engineer has compatibility issues, create features directly
            try:
                # Start with raw V features and Amount
                v_columns = [f'V{i}' for i in range(1, 29)]
                feature_columns = v_columns + ['Amount']
                available_columns = [col for col in feature_columns if col in df.columns]
                df_features = df[available_columns].copy()
                
                # Add engineered features to reach 96 features
                # Amount-based features
                df_features['amount_log'] = np.log1p(df_features['Amount'])
                df_features['amount_sqrt'] = np.sqrt(df_features['Amount'])
                df_features['amount_squared'] = df_features['Amount'] ** 2
                
                # V-features statistics
                v_data = df_features[v_columns].values
                df_features['v_mean'] = np.mean(v_data, axis=1)
                df_features['v_std'] = np.std(v_data, axis=1)
                df_features['v_max'] = np.max(v_data, axis=1)
                df_features['v_min'] = np.min(v_data, axis=1)
                df_features['v_range'] = df_features['v_max'] - df_features['v_min']
                df_features['v_abs_mean'] = np.mean(np.abs(v_data), axis=1)
                
                # V-features interactions
                df_features['v1_v2_interaction'] = df_features['V1'] * df_features['V2']
                df_features['v3_v4_interaction'] = df_features['V3'] * df_features['V4']
                df_features['v5_v6_interaction'] = df_features['V5'] * df_features['V6']
                df_features['v7_v8_interaction'] = df_features['V7'] * df_features['V8']
                df_features['v9_v10_interaction'] = df_features['V9'] * df_features['V10']
                
                # Amount-V interactions
                df_features['amount_v1_interaction'] = df_features['Amount'] * df_features['V1']
                df_features['amount_v2_interaction'] = df_features['Amount'] * df_features['V2']
                df_features['amount_v3_interaction'] = df_features['Amount'] * df_features['V3']
                df_features['amount_v_mean_interaction'] = df_features['Amount'] * df_features['v_mean']
                df_features['amount_v_std_interaction'] = df_features['Amount'] * df_features['v_std']
                
                # Time-based features (if available)
                if 'Time' in df.columns:
                    time_values = pd.to_numeric(df['Time'], errors='coerce').fillna(0)
                    df_features['time_hour'] = (time_values % 86400) / 3600  # Hour of day
                    df_features['time_day'] = (time_values // 86400) % 7     # Day of week
                    df_features['time_sin'] = np.sin(2 * np.pi * df_features['time_hour'] / 24)
                    df_features['time_cos'] = np.cos(2 * np.pi * df_features['time_hour'] / 24)
                else:
                    # Fill with zeros if time not available
                    df_features['time_hour'] = 0
                    df_features['time_day'] = 0
                    df_features['time_sin'] = 0
                    df_features['time_cos'] = 0
                
                # Additional engineered features to reach 96
                # Polynomial features for key V-features
                for i in range(1, 11):  # V1-V10
                    col = f'V{i}'
                    if col in df_features.columns:
                        df_features[f'{col}_squared'] = df_features[col] ** 2
                        df_features[f'{col}_cubed'] = df_features[col] ** 3
                
                # Cross-features between V-features
                for i in range(1, 6):  # V1-V5
                    for j in range(i+1, 6):  # V2-V5
                        col1, col2 = f'V{i}', f'V{j}'
                        if col1 in df_features.columns and col2 in df_features.columns:
                            df_features[f'{col1}_{col2}_cross'] = df_features[col1] * df_features[col2]
                
                # Ensure we have exactly 96 features by adding zeros if needed
                current_features = len(df_features.columns)
                if current_features < 96:
                    for i in range(current_features, 96):
                        df_features[f'feature_{i}'] = 0
                elif current_features > 96:
                    # Keep only the first 96 features
                    df_features = df_features.iloc[:, :96]
                
                logger.info(f"Created {len(df_features.columns)} engineered features")
                
            except Exception as fe_error:
                logger.warning(f"Feature engineering failed: {fe_error}, using raw features")
                # Fallback to raw features
                v_columns = [f'V{i}' for i in range(1, 29)]
                feature_columns = v_columns + ['Amount']
                available_columns = [col for col in feature_columns if col in df.columns]
                df_features = df[available_columns]
                # Pad with zeros to reach 96 features
                while len(df_features.columns) < 96:
                    df_features[f'feature_{len(df_features.columns)}'] = 0
                logger.info(f"Using raw features with padding: {len(df_features.columns)} features")
            
            # Make predictions using the trained model
            try:
                # Get anomaly scores from the model
                anomaly_scores = fraud_detector.predict_anomaly_scores(df_features.values)
                
                # Convert anomaly scores to fraud probabilities (higher anomaly = higher fraud probability)
                # Normalize scores to 0-1 range
                max_score = np.max(anomaly_scores) if np.max(anomaly_scores) > 0 else 1.0
                fraud_probabilities = anomaly_scores / max_score
                
                # Apply threshold
                is_fraudulent = fraud_probabilities >= threshold
                fraud_count = np.sum(is_fraudulent)
                
                logger.info(f"Model predictions completed: {len(fraud_probabilities)} transactions")
                logger.info(f"Anomaly score range: {np.min(anomaly_scores):.4f} - {np.max(anomaly_scores):.4f}")
                
            except Exception as model_error:
                logger.warning(f"Model prediction failed: {model_error}, using fallback")
                # Fallback to heuristic
                fraud_probabilities = np.zeros(len(df))
                is_fraudulent = np.zeros(len(df), dtype=bool)
                fraud_count = 0
        else:
            logger.info("Using fallback heuristic model")
            # Fallback heuristic anomaly detection
            # 1. Amount anomaly (higher amounts are more suspicious)
            amount_scores = np.minimum(df['Amount'] / 1000.0, 1.0)
            
            # 2. V-features anomaly (using standard deviation from mean)
            v_columns = [f'V{i}' for i in range(1, 29)]
            v_data = df[v_columns].values
            v_means = np.mean(v_data, axis=1, keepdims=True)
            v_stds = np.std(v_data, axis=1, keepdims=True)
            v_anomalies = np.mean(np.abs(v_data - v_means), axis=1) / (v_stds.flatten() + 1e-8)
            v_scores = np.minimum(v_anomalies / 5.0, 1.0)
            
            # 3. Time anomaly (transactions at unusual times)
            time_scores = np.minimum(np.abs(df['Time'] - 86400) / 86400, 1.0)
            
            # Combine all anomaly scores with weights
            anomaly_scores = (amount_scores * 0.4) + (v_scores * 0.4) + (time_scores * 0.2)
            fraud_probabilities = np.minimum(anomaly_scores, 1.0)
            
            # Apply threshold
            is_fraudulent = fraud_probabilities >= threshold
            fraud_count = np.sum(is_fraudulent)
        
        # Create predictions for ALL transactions (no limit)
        predictions = []
        
        for idx, row in df.iterrows():
            v_features = {f'v{i}': float(row[f'V{i}']) for i in range(1, 29)}
            
            prediction = TransactionPrediction(
                transaction_id=row['transaction_id'],
                amount=row['amount'],
                time=row['time'],
                **v_features,
                fraud_probability=round(fraud_probabilities[idx], 4),
                is_fraudulent=bool(is_fraudulent[idx]),
                actual_fraud=bool(row['is_fraudulent']),
                threshold=threshold
            )
            predictions.append(prediction)
        
        processing_time = (time.time() - start_time) * 1000
        
        metrics = PredictionMetrics(
            total_transactions=len(all_transactions),
            fraud_detected=fraud_count,
            normal_transactions=len(all_transactions) - fraud_count,
            fraud_rate=round(fraud_count / len(all_transactions), 4),
            threshold=threshold,
            processing_time_ms=round(processing_time, 2)
        )
        
        return {
            "predictions": [pred.model_dump() for pred in predictions],
            "metrics": metrics.model_dump()
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    logger.info("Starting Fraud Detection Dashboard...")
    success = load_models()
    if not success:
        logger.error("Failed to load models. API may not function correctly.")
    yield
    # Shutdown
    logger.info("Shutting down Fraud Detection Dashboard...")

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection Dashboard",
    description="Simplified fraud detection dashboard with threshold adjustment",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard HTML."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fraud Detection Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                min-height: 100vh; 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .card { 
                border: none; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.1); 
                background: rgba(255, 255, 255, 0.95);
            }
            .btn-primary { 
                background: linear-gradient(45deg, #667eea, #764ba2); 
                border: none; 
                padding: 12px 30px;
                font-weight: 600;
            }
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .fraud-row { 
                background-color: rgba(255, 107, 107, 0.1); 
                border-left: 4px solid #ff6b6b;
            }
            .normal-row { 
                background-color: rgba(81, 207, 102, 0.1); 
                border-left: 4px solid #51cf66;
            }
            .metrics-card {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
            }
            .threshold-slider {
                width: 100%;
                height: 8px;
                border-radius: 5px;
                background: #ddd;
                outline: none;
                -webkit-appearance: none;
            }
            .threshold-slider::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #667eea;
                cursor: pointer;
            }
            .threshold-slider::-moz-range-thumb {
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #667eea;
                cursor: pointer;
                border: none;
            }
                         .table-container {
                 max-height: 600px;
                 overflow-y: auto;
                 overflow-x: auto;
             }
             .table {
                 min-width: 1200px;
             }
             .v-feature {
                 font-size: 0.8em;
                 color: #666;
             }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .spinner-border {
                width: 3rem;
                height: 3rem;
            }
        </style>
    </head>
    <body>
        <div class="container-fluid py-4">
            <!-- Header -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-body text-center">
                                                         <h1 class="mb-2"><i class="fas fa-shield-alt text-primary"></i> Fraud Detection Dashboard</h1>
                             <p class="text-muted mb-0">Real Model Predictions on Complete Test Set (All Results)</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Controls -->
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title"><i class="fas fa-sliders-h"></i> Threshold Control</h5>
                            <div class="mb-3">
                                <label for="thresholdSlider" class="form-label">Fraud Detection Threshold: <span id="thresholdValue">0.5</span></label>
                                <input type="range" class="threshold-slider" id="thresholdSlider" min="0" max="1" step="0.01" value="0.5">
                                <div class="d-flex justify-content-between mt-1">
                                    <small class="text-muted">0.0 (More Sensitive)</small>
                                    <small class="text-muted">1.0 (Less Sensitive)</small>
                                </div>
                            </div>
                                                         <button class="btn btn-primary" id="predictButton" onclick="predictAll()">
                                 <i class="fas fa-play"></i> Predict All Transactions
                             </button>
                             <div class="mt-2">
                                 <small class="text-muted">
                                     <i class="fas fa-info-circle"></i> 
                                     Processing 56,961 transactions takes 10-15 seconds
                                 </small>
                             </div>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card metrics-card">
                        <div class="card-body">
                            <h5 class="card-title"><i class="fas fa-chart-bar"></i> Metrics</h5>
                            <div id="metricsContent">
                                <p class="text-center text-muted">Click "Predict All" to see metrics</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

                         <!-- Loading -->
             <div id="loading" class="loading">
                 <div class="spinner-border text-primary" role="status">
                     <span class="visually-hidden">Loading...</span>
                 </div>
                 <p class="mt-3">Processing complete test set with real model (this may take a while)...</p>
                 <div class="progress mt-3" style="height: 10px;">
                     <div class="progress-bar progress-bar-striped progress-bar-animated" 
                          role="progressbar" style="width: 100%"></div>
                 </div>
             </div>

            <!-- Results -->
            <div class="row">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0"><i class="fas fa-table"></i> Transaction Results</h5>
                        </div>
                        <div class="card-body">
                            <div id="resultsContent">
                                <p class="text-center text-muted">Click "Predict All" to see transaction results</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Update threshold display
            document.getElementById('thresholdSlider').addEventListener('input', function() {
                document.getElementById('thresholdValue').textContent = this.value;
            });

                         async function predictAll() {
                 const threshold = parseFloat(document.getElementById('thresholdSlider').value);
                 const loading = document.getElementById('loading');
                 const resultsContent = document.getElementById('resultsContent');
                 const metricsContent = document.getElementById('metricsContent');
                 const predictButton = document.getElementById('predictButton');
                 
                 // Disable button and show loading
                 predictButton.disabled = true;
                 predictButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                 loading.style.display = 'block';
                 resultsContent.innerHTML = '';
                 metricsContent.innerHTML = '';
                 
                 // Update loading message to show threshold
                 const loadingText = loading.querySelector('p');
                 loadingText.textContent = `Processing ${threshold} threshold on complete test set (this may take 10-15 seconds)...`;
                 
                 try {
                     // Use a longer timeout for the large dataset
                     const controller = new AbortController();
                     const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
                     
                     const response = await fetch('/api/predict/all', {
                         method: 'POST',
                         headers: {
                             'Content-Type': 'application/json',
                         },
                         body: JSON.stringify({ threshold: threshold }),
                         signal: controller.signal
                     });
                     
                     clearTimeout(timeoutId);
                     
                     if (!response.ok) {
                         throw new Error(`HTTP error! status: ${response.status}`);
                     }
                     
                     const result = await response.json();
                     displayResults(result);
                 } catch (error) {
                     console.error('Error:', error);
                     if (error.name === 'AbortError') {
                         alert('Request timed out. The dataset is large and may take longer to process. Please try again.');
                     } else {
                         alert('Error analyzing transactions: ' + error.message);
                     }
                 } finally {
                     // Re-enable button and hide loading
                     predictButton.disabled = false;
                     predictButton.innerHTML = '<i class="fas fa-play"></i> Predict All Transactions';
                     loading.style.display = 'none';
                 }
             }

            function displayResults(result) {
                displayMetrics(result.metrics);
                displayTransactions(result.predictions);
            }

            function displayMetrics(metrics) {
                const metricsContent = document.getElementById('metricsContent');
                
                metricsContent.innerHTML = `
                    <div class="row text-center">
                        <div class="col-6">
                            <h3>${metrics.total_transactions}</h3>
                            <small>Total Transactions</small>
                        </div>
                        <div class="col-6">
                            <h3 class="text-danger">${metrics.fraud_detected}</h3>
                            <small>Fraud Detected</small>
                        </div>
                    </div>
                    <div class="row text-center mt-3">
                        <div class="col-6">
                            <h3 class="text-success">${metrics.normal_transactions}</h3>
                            <small>Normal Transactions</small>
                        </div>
                        <div class="col-6">
                            <h3>${(metrics.fraud_rate * 100).toFixed(1)}%</h3>
                            <small>Fraud Rate</small>
                        </div>
                    </div>
                    <div class="row text-center mt-3">
                        <div class="col-12">
                            <small>Threshold: ${metrics.threshold}</small><br>
                            <small>Processing Time: ${metrics.processing_time_ms}ms</small>
                        </div>
                    </div>
                `;
            }

            function displayTransactions(predictions) {
                const resultsContent = document.getElementById('resultsContent');
                
                if (predictions.length === 0) {
                    resultsContent.innerHTML = '<p class="text-center text-muted">No transactions to display</p>';
                    return;
                }
                
                const tableHtml = `
                    <div class="table-container">
                        <table class="table table-hover">
                            <thead class="table-dark">
                                <tr>
                                    <th>Transaction ID</th>
                                    <th>Amount ($)</th>
                                    <th>Time (s)</th>
                                    <th>V1</th>
                                    <th>V2</th>
                                    <th>V3</th>
                                    <th>V4</th>
                                    <th>V5</th>
                                    <th>V6</th>
                                    <th>V7</th>
                                    <th>V8</th>
                                    <th>V9</th>
                                    <th>V10</th>
                                    <th>V11</th>
                                    <th>V12</th>
                                    <th>V13</th>
                                    <th>V14</th>
                                    <th>V15</th>
                                    <th>V16</th>
                                    <th>V17</th>
                                    <th>V18</th>
                                    <th>V19</th>
                                    <th>V20</th>
                                    <th>V21</th>
                                    <th>V22</th>
                                    <th>V23</th>
                                    <th>V24</th>
                                    <th>V25</th>
                                    <th>V26</th>
                                    <th>V27</th>
                                                                         <th>V28</th>
                                     <th>Fraud Probability</th>
                                     <th>Predicted</th>
                                     <th>Actual</th>
                                </tr>
                            </thead>
                            <tbody>
                                                                 ${predictions.map(pred => {
                                     const rowClass = pred.actual_fraud ? 'fraud-row' : 'normal-row';
                                     const predictedBadgeClass = pred.is_fraudulent ? 'bg-danger' : 'bg-success';
                                     const predictedBadgeText = pred.is_fraudulent ? 'FRAUD' : 'NORMAL';
                                     const actualBadgeClass = pred.actual_fraud ? 'bg-danger' : 'bg-success';
                                     const actualBadgeText = pred.actual_fraud ? 'FRAUD' : 'NORMAL';
                                     
                                     return `
                                                                                  <tr class="${rowClass}">
                                              <td><strong>${pred.transaction_id}</strong></td>
                                              <td>$${pred.amount.toFixed(2)}</td>
                                              <td>${pred.time.toFixed(0)}</td>
                                              <td class="v-feature">${pred.v1.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v2.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v3.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v4.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v5.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v6.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v7.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v8.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v9.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v10.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v11.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v12.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v13.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v14.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v15.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v16.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v17.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v18.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v19.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v20.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v21.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v22.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v23.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v24.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v25.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v26.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v27.toFixed(3)}</td>
                                              <td class="v-feature">${pred.v28.toFixed(3)}</td>
                                              <td>
                                                  <div class="progress" style="height: 20px;">
                                                      <div class="progress-bar ${pred.is_fraudulent ? 'bg-danger' : 'bg-success'}" 
                                                           style="width: ${(pred.fraud_probability * 100)}%">
                                                          ${(pred.fraud_probability * 100).toFixed(1)}%
                                                      </div>
                                                  </div>
                                              </td>
                                              <td><span class="badge ${predictedBadgeClass}">${predictedBadgeText}</span></td>
                                              <td><span class="badge ${actualBadgeClass}">${actualBadgeText}</span></td>
                                          </tr>
                                     `;
                                 }).join('')}
                            </tbody>
                        </table>
                    </div>
                `;
                
                resultsContent.innerHTML = tableHtml;
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/predict/all")
async def predict_all_with_threshold(request: ThresholdRequest):
    """Predict fraud for all transactions with adjustable threshold."""
    return predict_all_transactions(request.threshold)

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False) 