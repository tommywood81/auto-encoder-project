#!/usr/bin/env python3
"""
Fraud Detection Dashboard - FastAPI Application
Professional fraud detection system with real-time prediction capabilities.
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

# Global variables for loaded models
fraud_detector = None
feature_engineer = None
sample_data = None
config = None

# Pydantic models
class TransactionData(BaseModel):
    """Transaction data model for prediction."""
    amount: float = Field(..., description="Transaction amount")
    customer_age: int = Field(..., description="Customer age")
    payment_method: str = Field(..., description="Payment method")
    merchant_category: str = Field(..., description="Merchant category")
    transaction_time: str = Field(..., description="Transaction timestamp")
    location: str = Field(..., description="Transaction location")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 150.50,
                "customer_age": 35,
                "payment_method": "credit_card",
                "merchant_category": "electronics",
                "transaction_time": "2024-01-15T14:30:00",
                "location": "New York, NY"
            }
        }
    }

class PredictionResponse(BaseModel):
    """Prediction response model."""
    transaction_id: str
    fraud_probability: float
    is_fraudulent: bool
    confidence_score: float
    risk_level: str
    anomaly_score: float
    processing_time_ms: float
    timestamp: str

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    transactions: List[TransactionData]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[PredictionResponse]
    total_transactions: int
    fraud_count: int
    processing_time_ms: float

class SystemStatus(BaseModel):
    """System status model."""
    status: str
    model_loaded: bool
    last_model_update: str
    total_predictions: int
    system_uptime: str
    version: str

# Load models and data
def load_models():
    """Load the trained fraud detection model and feature engineer."""
    global fraud_detector, feature_engineer, sample_data, config
    
    try:
        # Load configuration
        config_path = "configs/final_optimized_config.yaml"
        config_loader = ConfigLoader(config_path)
        config = config_loader.config
        
        # Load trained model
        model_path = "models/fraud_autoencoder.keras"
        fraud_detector = FraudAutoencoder(config)
        fraud_detector.load_model(model_path)
        
        # Load feature engineer
        feature_engineer = FeatureEngineer(config.get('features', {}))
        feature_engineer.load_fitted_objects("models/fraud_autoencoder_features.pkl")
        
        # Load sample data for demonstration
        sample_data = load_sample_data()
        
        logger.info("Models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def load_sample_data():
    """Load sample transaction data for demonstration."""
    try:
        # Load the cleaned data
        df = pd.read_csv("data/cleaned/creditcard_cleaned.csv")
        
        # Create sample transactions (both fraudulent and non-fraudulent)
        fraud_samples = df[df['is_fraudulent'] == 1].head(5)
        normal_samples = df[df['is_fraudulent'] == 0].head(5)
        
        samples = []
        
        # Convert fraud samples
        for idx, row in fraud_samples.iterrows():
            sample = {
                "id": f"FRAUD_{idx}",
                "amount": float(row.get('amount', 0)),
                "customer_age": int(row.get('customer_age', 30)),
                "payment_method": "credit_card",
                "merchant_category": "electronics",
                "transaction_time": "2024-01-15T14:30:00",
                "location": "New York, NY",
                "is_fraudulent": True,
                "original_data": row.to_dict()
            }
            samples.append(sample)
        
        # Convert normal samples
        for idx, row in normal_samples.iterrows():
            sample = {
                "id": f"NORMAL_{idx}",
                "amount": float(row.get('amount', 0)),
                "customer_age": int(row.get('customer_age', 30)),
                "payment_method": "credit_card",
                "merchant_category": "electronics",
                "transaction_time": "2024-01-15T14:30:00",
                "location": "New York, NY",
                "is_fraudulent": False,
                "original_data": row.to_dict()
            }
            samples.append(sample)
        
        return samples
        
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return []

def predict_single_transaction(transaction: TransactionData) -> PredictionResponse:
    """Predict fraud for a single transaction."""
    import time
    start_time = time.time()
    
    try:
        # Convert transaction to DataFrame
        transaction_dict = transaction.dict()
        
        # Create a DataFrame with the transaction data
        # We'll need to map the input fields to the expected feature columns
        df_transaction = pd.DataFrame([transaction_dict])
        
        # Apply feature engineering
        df_features = feature_engineer.transform(df_transaction)
        
        # Remove target column if present
        if 'is_fraudulent' in df_features.columns:
            df_features = df_features.drop(columns=['is_fraudulent'])
        
        # Get numeric features only
        df_numeric = df_features.select_dtypes(include=[np.number])
        
        # Make prediction
        anomaly_score = fraud_detector.predict_anomaly_scores(df_numeric.values)[0]
        is_fraudulent = fraud_detector.predict(df_numeric.values)[0]
        
        # Calculate fraud probability (normalize anomaly score)
        fraud_probability = min(anomaly_score / 10.0, 1.0)  # Normalize to 0-1
        
        # Determine risk level
        if fraud_probability > 0.8:
            risk_level = "HIGH"
        elif fraud_probability > 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Calculate confidence score
        confidence_score = abs(fraud_probability - 0.5) * 2  # Distance from 0.5
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return PredictionResponse(
            transaction_id=f"TXN_{int(time.time())}",
            fraud_probability=round(fraud_probability, 4),
            is_fraudulent=bool(is_fraudulent),
            confidence_score=round(confidence_score, 4),
            risk_level=risk_level,
            anomaly_score=round(anomaly_score, 6),
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    logger.info("Starting FraudGuard Pro API...")
    success = load_models()
    if not success:
        logger.error("Failed to load models. API may not function correctly.")
    yield
    # Shutdown
    logger.info("Shutting down FraudGuard Pro API...")

# Initialize FastAPI app
app = FastAPI(
    title="FraudGuard Pro - AI Fraud Detection System",
    description="Enterprise-grade fraud detection powered by deep learning autoencoders",
    version="2.0.0",
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
        <title>FraudGuard Pro - AI Fraud Detection Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .card { border: none; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
            .btn-primary { background: linear-gradient(45deg, #667eea, #764ba2); border: none; }
            .btn-danger { background: linear-gradient(45deg, #ff6b6b, #ee5a52); border: none; }
            .btn-success { background: linear-gradient(45deg, #51cf66, #40c057); border: none; }
            .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
            .status-online { background-color: #51cf66; }
            .status-offline { background-color: #ff6b6b; }
            .transaction-card { transition: transform 0.2s; }
            .transaction-card:hover { transform: translateY(-2px); }
            .fraud-indicator { background: linear-gradient(45deg, #ff6b6b, #ee5a52); color: white; }
            .normal-indicator { background: linear-gradient(45deg, #51cf66, #40c057); color: white; }
        </style>
    </head>
    <body>
        <div class="container-fluid py-4">
            <!-- Header -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-body text-center">
                            <h1 class="mb-2"><i class="fas fa-shield-alt text-primary"></i> FraudGuard Pro</h1>
                            <p class="text-muted mb-0">Enterprise AI Fraud Detection System</p>
                            <div class="mt-2">
                                <span class="status-indicator status-online"></span>
                                <span class="text-success">System Online</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Main Dashboard -->
            <div class="row">
                <!-- Real-time Prediction -->
                <div class="col-lg-8">
                    <div class="card mb-4">
                        <div class="card-header">
                            <h5 class="mb-0"><i class="fas fa-bolt"></i> Real-time Transaction Analysis</h5>
                        </div>
                        <div class="card-body">
                            <form id="predictionForm">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label class="form-label">Transaction Amount ($)</label>
                                            <input type="number" class="form-control" id="amount" step="0.01" required>
                                        </div>
                                        <div class="mb-3">
                                            <label class="form-label">Customer Age</label>
                                            <input type="number" class="form-control" id="customerAge" required>
                                        </div>
                                        <div class="mb-3">
                                            <label class="form-label">Payment Method</label>
                                            <select class="form-select" id="paymentMethod" required>
                                                <option value="credit_card">Credit Card</option>
                                                <option value="debit_card">Debit Card</option>
                                                <option value="bank_transfer">Bank Transfer</option>
                                            </select>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label class="form-label">Merchant Category</label>
                                            <select class="form-select" id="merchantCategory" required>
                                                <option value="electronics">Electronics</option>
                                                <option value="clothing">Clothing</option>
                                                <option value="food">Food & Dining</option>
                                                <option value="travel">Travel</option>
                                            </select>
                                        </div>
                                        <div class="mb-3">
                                            <label class="form-label">Location</label>
                                            <input type="text" class="form-control" id="location" required>
                                        </div>
                                        <div class="mb-3">
                                            <label class="form-label">Transaction Time</label>
                                            <input type="datetime-local" class="form-control" id="transactionTime" required>
                                        </div>
                                    </div>
                                </div>
                                <button type="submit" class="btn btn-primary">
                                    <i class="fas fa-search"></i> Analyze Transaction
                                </button>
                            </form>
                            
                            <div id="predictionResult" class="mt-4" style="display: none;">
                                <div class="card">
                                    <div class="card-body">
                                        <h6>Analysis Result</h6>
                                        <div id="resultContent"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Sample Transactions -->
                <div class="col-lg-4">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0"><i class="fas fa-database"></i> Sample Transactions</h5>
                        </div>
                        <div class="card-body">
                            <p class="text-muted small">Click on any transaction to test the system</p>
                            <div id="sampleTransactions"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Batch Analysis -->
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0"><i class="fas fa-layer-group"></i> Batch Analysis</h5>
                        </div>
                        <div class="card-body">
                            <button class="btn btn-success" onclick="analyzeAllSamples()">
                                <i class="fas fa-play"></i> Analyze All Sample Transactions
                            </button>
                            <div id="batchResults" class="mt-3" style="display: none;">
                                <div class="card">
                                    <div class="card-body">
                                        <h6>Batch Analysis Results</h6>
                                        <div id="batchContent"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Load sample transactions on page load
            document.addEventListener('DOMContentLoaded', function() {
                loadSampleTransactions();
                setDefaultDateTime();
            });

            function setDefaultDateTime() {
                const now = new Date();
                const localDateTime = new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
                document.getElementById('transactionTime').value = localDateTime;
            }

            async function loadSampleTransactions() {
                try {
                    const response = await fetch('/api/samples');
                    const samples = await response.json();
                    
                    const container = document.getElementById('sampleTransactions');
                    container.innerHTML = '';
                    
                    samples.forEach(sample => {
                        const card = document.createElement('div');
                        card.className = 'card transaction-card mb-2';
                        card.style.cursor = 'pointer';
                        
                        const badgeClass = sample.is_fraudulent ? 'fraud-indicator' : 'normal-indicator';
                        const badgeText = sample.is_fraudulent ? 'FRAUD' : 'NORMAL';
                        
                        card.innerHTML = `
                            <div class="card-body p-3">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <strong>$${sample.amount.toFixed(2)}</strong>
                                        <br><small class="text-muted">Age: ${sample.customer_age}</small>
                                    </div>
                                    <span class="badge ${badgeClass}">${badgeText}</span>
                                </div>
                            </div>
                        `;
                        
                        card.onclick = () => fillFormWithSample(sample);
                        container.appendChild(card);
                    });
                } catch (error) {
                    console.error('Error loading samples:', error);
                }
            }

            function fillFormWithSample(sample) {
                document.getElementById('amount').value = sample.amount;
                document.getElementById('customerAge').value = sample.customer_age;
                document.getElementById('paymentMethod').value = sample.payment_method;
                document.getElementById('merchantCategory').value = sample.merchant_category;
                document.getElementById('location').value = sample.location;
                document.getElementById('transactionTime').value = sample.transaction_time.replace('Z', '');
            }

            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = {
                    amount: parseFloat(document.getElementById('amount').value),
                    customer_age: parseInt(document.getElementById('customerAge').value),
                    payment_method: document.getElementById('paymentMethod').value,
                    merchant_category: document.getElementById('merchantCategory').value,
                    location: document.getElementById('location').value,
                    transaction_time: document.getElementById('transactionTime').value
                };
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(formData)
                    });
                    
                    const result = await response.json();
                    displayPredictionResult(result);
                } catch (error) {
                    console.error('Error:', error);
                    alert('Error analyzing transaction');
                }
            });

            function displayPredictionResult(result) {
                const resultDiv = document.getElementById('predictionResult');
                const contentDiv = document.getElementById('resultContent');
                
                const riskColor = result.risk_level === 'HIGH' ? 'danger' : 
                                result.risk_level === 'MEDIUM' ? 'warning' : 'success';
                
                contentDiv.innerHTML = `
                    <div class="row">
                        <div class="col-md-6">
                            <div class="d-flex justify-content-between mb-2">
                                <span>Fraud Probability:</span>
                                <span class="badge bg-${riskColor}">${(result.fraud_probability * 100).toFixed(1)}%</span>
                            </div>
                            <div class="d-flex justify-content-between mb-2">
                                <span>Risk Level:</span>
                                <span class="badge bg-${riskColor}">${result.risk_level}</span>
                            </div>
                            <div class="d-flex justify-content-between mb-2">
                                <span>Confidence:</span>
                                <span>${(result.confidence_score * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="d-flex justify-content-between mb-2">
                                <span>Decision:</span>
                                <span class="badge ${result.is_fraudulent ? 'bg-danger' : 'bg-success'}">
                                    ${result.is_fraudulent ? 'FRAUD DETECTED' : 'NORMAL'}
                                </span>
                            </div>
                            <div class="d-flex justify-content-between mb-2">
                                <span>Anomaly Score:</span>
                                <span>${result.anomaly_score.toFixed(4)}</span>
                            </div>
                            <div class="d-flex justify-content-between mb-2">
                                <span>Processing Time:</span>
                                <span>${result.processing_time_ms}ms</span>
                            </div>
                        </div>
                    </div>
                `;
                
                resultDiv.style.display = 'block';
            }

            async function analyzeAllSamples() {
                try {
                    const response = await fetch('/api/samples');
                    const samples = await response.json();
                    
                    const batchData = {
                        transactions: samples.map(s => ({
                            amount: s.amount,
                            customer_age: s.customer_age,
                            payment_method: s.payment_method,
                            merchant_category: s.merchant_category,
                            location: s.location,
                            transaction_time: s.transaction_time
                        }))
                    };
                    
                    const batchResponse = await fetch('/api/predict/batch', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(batchData)
                    });
                    
                    const batchResult = await batchResponse.json();
                    displayBatchResults(batchResult);
                } catch (error) {
                    console.error('Error:', error);
                    alert('Error in batch analysis');
                }
            }

            function displayBatchResults(result) {
                const resultDiv = document.getElementById('batchResults');
                const contentDiv = document.getElementById('batchContent');
                
                let fraudCount = 0;
                let normalCount = 0;
                
                const predictionsHtml = result.predictions.map(pred => {
                    if (pred.is_fraudulent) fraudCount++;
                    else normalCount++;
                    
                    const riskColor = pred.risk_level === 'HIGH' ? 'danger' : 
                                    pred.risk_level === 'MEDIUM' ? 'warning' : 'success';
                    
                    return `
                        <div class="row mb-2">
                            <div class="col-md-3">${pred.transaction_id}</div>
                            <div class="col-md-2">${(pred.fraud_probability * 100).toFixed(1)}%</div>
                            <div class="col-md-2">
                                <span class="badge bg-${riskColor}">${pred.risk_level}</span>
                            </div>
                            <div class="col-md-2">
                                <span class="badge ${pred.is_fraudulent ? 'bg-danger' : 'bg-success'}">
                                    ${pred.is_fraudulent ? 'FRAUD' : 'NORMAL'}
                                </span>
                            </div>
                            <div class="col-md-3">${pred.processing_time_ms}ms</div>
                        </div>
                    `;
                }).join('');
                
                contentDiv.innerHTML = `
                    <div class="row mb-3">
                        <div class="col-md-3">
                            <strong>Total Transactions:</strong> ${result.total_transactions}
                        </div>
                        <div class="col-md-3">
                            <strong>Fraud Detected:</strong> <span class="text-danger">${result.fraud_count}</span>
                        </div>
                        <div class="col-md-3">
                            <strong>Normal:</strong> <span class="text-success">${normalCount}</span>
                        </div>
                        <div class="col-md-3">
                            <strong>Processing Time:</strong> ${result.processing_time_ms}ms
                        </div>
                    </div>
                    <div class="table-responsive">
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>Transaction ID</th>
                                    <th>Fraud Probability</th>
                                    <th>Risk Level</th>
                                    <th>Decision</th>
                                    <th>Processing Time</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${result.predictions.map(pred => `
                                    <tr>
                                        <td>${pred.transaction_id}</td>
                                        <td>${(pred.fraud_probability * 100).toFixed(1)}%</td>
                                        <td><span class="badge bg-${pred.risk_level === 'HIGH' ? 'danger' : pred.risk_level === 'MEDIUM' ? 'warning' : 'success'}">${pred.risk_level}</span></td>
                                        <td><span class="badge ${pred.is_fraudulent ? 'bg-danger' : 'bg-success'}">${pred.is_fraudulent ? 'FRAUD' : 'NORMAL'}</span></td>
                                        <td>${pred.processing_time_ms}ms</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                `;
                
                resultDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """

@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and health information."""
    return SystemStatus(
        status="online" if fraud_detector is not None else "offline",
        model_loaded=fraud_detector is not None,
        last_model_update=datetime.now().isoformat(),
        total_predictions=0,  # Could track this in a database
        system_uptime="0 days, 0 hours, 0 minutes",
        version="2.0.0"
    )

@app.get("/api/samples")
async def get_sample_transactions():
    """Get sample transactions for demonstration."""
    if sample_data is None:
        raise HTTPException(status_code=500, detail="Sample data not loaded")
    return sample_data

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_transaction(transaction: TransactionData):
    """Predict fraud for a single transaction."""
    if fraud_detector is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return predict_single_transaction(transaction)

@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_transactions(batch_request: BatchPredictionRequest):
    """Predict fraud for multiple transactions."""
    if fraud_detector is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    predictions = []
    for transaction in batch_request.transactions:
        prediction = predict_single_transaction(transaction)
        predictions.append(prediction)
    
    processing_time = (time.time() - start_time) * 1000
    fraud_count = sum(1 for p in predictions if p.is_fraudulent)
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_transactions=len(predictions),
        fraud_count=fraud_count,
        processing_time_ms=round(processing_time, 2)
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False) 