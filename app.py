#!/usr/bin/env python3
"""
Fraud Detection API Server using FastAPI.
Serves the trained autoencoder model for real-time fraud detection.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig
from src.models import BaselineAutoencoder
from src.feature_factory import FeatureFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="AI-powered fraud detection using autoencoders",
    version="1.0.0"
)

# Global variables for model and data
model = None
scaler = None
threshold = None
test_data = None
feature_columns = None

# Pydantic models for request/response
class Features(BaseModel):
    features: Dict[str, float]

class BatchRequest(BaseModel):
    transactions: List[Dict[str, float]]

class PredictionResponse(BaseModel):
    is_fraudulent: bool
    anomaly_score: float
    threshold: float
    confidence: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    model_type: str
    strategy: str
    feature_count: int
    feature_columns: List[str]
    threshold: Optional[float]
    model_loaded: bool

def load_model():
    """Load the trained model and prepare test data."""
    global model, scaler, threshold, test_data, feature_columns
    
    try:
        logger.info("Loading fraud detection model...")
        
        # Load configuration for combined strategy (best performing)
        config = PipelineConfig.get_config("combined")
        
        # Initialize autoencoder
        autoencoder = BaselineAutoencoder(config)
        
        # Load the trained model
        model_path = os.path.join(config.data.models_dir, "autoencoder.h5")
        if not os.path.exists(model_path):
            # Try alternative model names
            for model_file in ["baseline_autoencoder.h5", "autoencoder_fraud_detection.pth"]:
                model_path = os.path.join(config.data.models_dir, model_file)
                if os.path.exists(model_path):
                    break
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found in {config.data.models_dir}")
        
        autoencoder.load_model(model_path)
        
        # Load scaler and threshold
        logger.info("Preparing test data and recreating scaler...")
        
        # Load cleaned data and engineer features
        cleaned_file = os.path.join(config.data.cleaned_dir, "ecommerce_cleaned.csv")
        if not os.path.exists(cleaned_file):
            raise FileNotFoundError(f"Cleaned data not found: {cleaned_file}")
        
        df_cleaned = pd.read_csv(cleaned_file)
        feature_engineer = FeatureFactory.create(config.feature_strategy)
        df_features = feature_engineer.generate_features(df_cleaned)
        
        # Get numeric features
        df_numeric = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in df_numeric.columns:
            df_numeric = df_numeric.drop(columns=['is_fraudulent'])
        
        # Store feature columns
        feature_columns = df_numeric.columns.tolist()
        
        # Prepare test data (last 20% of data for testing)
        total_samples = len(df_numeric)
        test_start = int(0.8 * total_samples)
        test_data_full = df_numeric.iloc[test_start:test_start+20]  # First 20 test samples
        
        # Fit scaler on training data (first 80%)
        train_data = df_numeric.iloc[:test_start]
        autoencoder.scaler.fit(train_data)
        
        # Store components
        model = autoencoder.model
        scaler = autoencoder.scaler
        threshold = autoencoder.threshold or np.percentile(
            autoencoder.predict_anomaly_scores(train_data), 95.0
        )
        test_data = test_data_full
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"Feature columns: {feature_columns}")
        logger.info(f"Test data shape: {test_data.shape}")
        logger.info(f"Threshold: {threshold:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    if not load_model():
        logger.error("Failed to load model. Application may not function properly.")

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main web interface."""
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None
    )

@app.get("/test-data")
async def get_test_data():
    """Get the first 20 test data points for inference."""
    if test_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert test data to list of dictionaries
        test_data_list = []
        for i, row in test_data.iterrows():
            data_point = {
                'id': i,
                'features': row.to_dict()
            }
            test_data_list.append(data_point)
        
        return {
            'test_data': test_data_list,
            'feature_columns': feature_columns,
            'count': len(test_data_list)
        }
        
    except Exception as e:
        logger.error(f"Error getting test data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=Dict)
async def predict(request: Features):
    """Make fraud prediction for a single transaction."""
    if model is None or scaler is None or threshold is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        features = request.features
        
        # Validate features
        if not all(col in features for col in feature_columns):
            missing = [col for col in feature_columns if col not in features]
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing}",
                headers={"required_features": str(feature_columns)}
            )
        
        # Create feature vector
        feature_vector = np.array([[features[col] for col in feature_columns]])
        
        # Make prediction
        anomaly_score = model.predict_anomaly_scores(feature_vector)[0]
        is_fraudulent = int(anomaly_score > threshold)
        
        # Calculate confidence (inverse of anomaly score, normalized)
        confidence = max(0.0, min(1.0, 1.0 - (anomaly_score / (threshold * 2))))
        
        return {
            'prediction': {
                'is_fraudulent': bool(is_fraudulent),
                'anomaly_score': float(anomaly_score),
                'threshold': float(threshold),
                'confidence': float(confidence)
            },
            'input_features': features
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(request: BatchRequest):
    """Make fraud predictions for multiple transactions."""
    if model is None or scaler is None or threshold is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        transactions = request.transactions
        
        if not transactions:
            raise HTTPException(status_code=400, detail="No transactions provided")
        
        predictions = []
        
        for i, transaction in enumerate(transactions):
            if not isinstance(transaction, dict):
                raise HTTPException(status_code=400, detail=f"Transaction {i} must be a dictionary")
            
            features = transaction
            
            # Validate features
            if not all(col in features for col in feature_columns):
                missing = [col for col in feature_columns if col not in features]
                raise HTTPException(
                    status_code=400,
                    detail=f"Transaction {i} missing features: {missing}",
                    headers={"required_features": str(feature_columns)}
                )
            
            # Create feature vector
            feature_vector = np.array([[features[col] for col in feature_columns]])
            
            # Make prediction
            anomaly_score = model.predict_anomaly_scores(feature_vector)[0]
            is_fraudulent = int(anomaly_score > threshold)
            confidence = max(0.0, min(1.0, 1.0 - (anomaly_score / (threshold * 2))))
            
            predictions.append({
                'transaction_id': i,
                'prediction': {
                    'is_fraudulent': bool(is_fraudulent),
                    'anomaly_score': float(anomaly_score),
                    'threshold': float(threshold),
                    'confidence': float(confidence)
                },
                'input_features': features
            })
        
        return {
            'predictions': predictions,
            'total_transactions': len(predictions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_type="Autoencoder",
        strategy="combined",
        feature_count=len(feature_columns) if feature_columns else 0,
        feature_columns=feature_columns or [],
        threshold=float(threshold) if threshold else None,
        model_loaded=True
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 