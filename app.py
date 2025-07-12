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
from datetime import datetime, timedelta
import json

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
full_data = None
data_dates = None

# Pydantic models for request/response
class Features(BaseModel):
    features: Dict[str, float]

class BatchRequest(BaseModel):
    transactions: List[Dict[str, float]]

class DateAnalysisRequest(BaseModel):
    date: str
    threshold: Optional[float] = None

class ThresholdUpdateRequest(BaseModel):
    threshold: float

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

class DateAnalysisResponse(BaseModel):
    date: str
    total_transactions: int
    flagged_transactions: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    threshold_used: float
    potential_savings: float
    avg_transaction_amount: float
    sample_anomaly_scores: List[float]

# Add new Pydantic models for business-focused responses
class AnomalousTransaction(BaseModel):
    transaction_id: str
    customer_id: str
    transaction_amount: float
    anomaly_score: float
    is_fraudulent: bool
    transaction_date: str
    merchant_category: str
    risk_level: str
    review_priority: str

class BusinessAnalysisResponse(BaseModel):
    date: str
    total_transactions: int
    flagged_transactions: int
    top_anomalous_count: int
    potential_savings: float
    avg_transaction_amount: float
    risk_distribution: Dict[str, int]
    anomalous_transactions: List[AnomalousTransaction]
    business_insights: Dict[str, str]

def load_model():
    """Load the trained model and prepare test data."""
    global model, scaler, threshold, test_data, feature_columns, full_data, data_dates
    
    try:
        logger.info("Loading fraud detection model...")
        
        # Load configuration for combined strategy (best performing)
        config = PipelineConfig.get_config("combined")
        
        # Initialize autoencoder
        autoencoder = BaselineAutoencoder(config)
        
        # Load the best trained model (final_model.h5)
        model_path = os.path.join(config.data.models_dir, "final_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Best model not found: {model_path}")
        
        autoencoder.load_model(model_path)
        
        # Load the correct threshold from final_model_info.yaml
        model_info_path = os.path.join(config.data.models_dir, "final_model_info.yaml")
        trained_threshold = 86.0  # Default from final_config.yaml
        
        if os.path.exists(model_info_path):
            try:
                import yaml
                # Try to load with safe_load first
                with open(model_info_path, 'r') as f:
                    model_info = yaml.safe_load(f)
                if model_info and 'threshold' in model_info:
                    trained_threshold = float(model_info['threshold'])
                    logger.info(f"Loaded threshold from final_model_info.yaml: {trained_threshold}")
                else:
                    logger.warning("No threshold found in final_model_info.yaml, using default: 86.0")
            except Exception as e:
                logger.warning(f"Could not load final_model_info.yaml: {e}")
                logger.warning("Using default threshold: 86.0")
        else:
            logger.warning("final_model_info.yaml not found, using default threshold: 86.0")
        
        # Load scaler and threshold
        logger.info("Preparing test data and recreating scaler...")
        
        # Load cleaned data and engineer features
        cleaned_file = os.path.join(config.data.cleaned_dir, "ecommerce_cleaned.csv")
        if not os.path.exists(cleaned_file):
            raise FileNotFoundError(f"Cleaned data not found: {cleaned_file}")
        
        df_cleaned = pd.read_csv(cleaned_file)
        
        # Parse transaction dates
        df_cleaned['transaction_date'] = pd.to_datetime(df_cleaned['transaction_date'])
        df_cleaned['date'] = df_cleaned['transaction_date'].dt.date.astype(str)
        
        # Store full data and available dates
        full_data = df_cleaned
        data_dates = sorted(df_cleaned['date'].unique().tolist())
        
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
        model = autoencoder  # Store the full autoencoder object
        scaler = autoencoder.scaler
        threshold = trained_threshold  # Use the trained threshold, not percentile-based
        test_data = test_data_full
        
        logger.info(f"Best model loaded successfully!")
        logger.info(f"Model: final_model.h5 (combined strategy)")
        logger.info(f"Feature columns: {feature_columns}")
        logger.info(f"Test data shape: {test_data.shape}")
        logger.info(f"Threshold: {threshold:.6f}")
        logger.info(f"Available dates: {len(data_dates)} dates from {data_dates[0]} to {data_dates[-1]}")
        
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

@app.get("/sample-anomaly-scores")
async def get_sample_anomaly_scores():
    """Get sample anomaly scores for debugging threshold issues."""
    if model is None or scaler is None or full_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get a sample of data (first 100 transactions)
        sample_data = full_data.head(100).copy()
        
        # Prepare features
        feature_engineer = FeatureFactory.create("combined")
        df_features = feature_engineer.generate_features(sample_data)
        
        # Get numeric features
        df_numeric = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in df_numeric.columns:
            actual_labels = df_numeric['is_fraudulent'].values
            df_numeric = df_numeric.drop(columns=['is_fraudulent'])
        else:
            actual_labels = np.zeros(len(df_numeric))
        
        # Scale features
        scaled_features = scaler.transform(df_numeric)
        
        # Get anomaly scores
        anomaly_scores = model.predict_anomaly_scores(scaled_features)
        
        # Calculate percentiles
        percentiles = {
            'min': float(anomaly_scores.min()),
            'max': float(anomaly_scores.max()),
            'mean': float(anomaly_scores.mean()),
            'std': float(anomaly_scores.std()),
            'p50': float(np.percentile(anomaly_scores, 50)),
            'p75': float(np.percentile(anomaly_scores, 75)),
            'p90': float(np.percentile(anomaly_scores, 90)),
            'p95': float(np.percentile(anomaly_scores, 95)),
            'p99': float(np.percentile(anomaly_scores, 99))
        }
        
        return {
            "sample_size": len(anomaly_scores),
            "current_threshold": float(threshold),
            "anomaly_score_stats": percentiles,
            "sample_scores": anomaly_scores[:10].tolist()  # First 10 scores
        }
        
    except Exception as e:
        logger.error(f"Error getting sample anomaly scores: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-dates")
async def get_available_dates():
    """Get available dates for analysis."""
    if data_dates is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return {
        "dates": data_dates,
        "date_range": {
            "start": data_dates[0],
            "end": data_dates[-1],
            "total_days": len(data_dates)
        }
    }

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

@app.post("/analyze-date", response_model=DateAnalysisResponse)
async def analyze_date(request: DateAnalysisRequest):
    """Analyze transactions for a specific date with business focus."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Filter data for the requested date
        date_data = full_data[full_data['date'] == request.date]
        
        if len(date_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data found for date {request.date}")
        
        logger.info(f"Analyzing {len(date_data)} transactions for date {request.date}")
        
        # Engineer features for the date
        feature_engineer = FeatureFactory.create("combined")
        df_features = feature_engineer.generate_features(date_data)
        
        # Get numeric features
        df_numeric = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in df_numeric.columns:
            df_numeric = df_numeric.drop(columns=['is_fraudulent'])
        
        # Scale the features
        scaled_features = scaler.transform(df_numeric)
        
        # Get reconstruction errors
        reconstructed = model.model.predict(scaled_features)
        mse = np.mean(np.power(scaled_features - reconstructed, 2), axis=1)
        
        # Calculate anomaly scores (normalized reconstruction error)
        anomaly_scores = mse / np.max(mse)
        
        # Get actual fraud labels
        actual_fraud = date_data['is_fraudulent'].values
        
        # Calculate threshold-based predictions
        predictions = anomaly_scores > (threshold / 100.0)
        
        # Calculate metrics
        true_positives = np.sum((predictions == 1) & (actual_fraud == 1))
        false_positives = np.sum((predictions == 1) & (actual_fraud == 0))
        true_negatives = np.sum((predictions == 0) & (actual_fraud == 0))
        false_negatives = np.sum((predictions == 0) & (actual_fraud == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AUC-ROC
        from sklearn.metrics import roc_auc_score
        auc_roc = roc_auc_score(actual_fraud, anomaly_scores)
        
        # Calculate business metrics
        avg_transaction_amount = date_data['transaction_amount'].mean()
        potential_savings = true_positives * avg_transaction_amount
        
        # Get top 5% most anomalous transactions for manual review
        top_5_percent_count = max(1, int(len(anomaly_scores) * 0.05))
        top_indices = np.argsort(anomaly_scores)[-top_5_percent_count:]
        
        # Create anomalous transactions list for manual review
        anomalous_transactions = []
        for idx in reversed(top_indices):  # Most anomalous first
            transaction = date_data.iloc[idx]
            anomaly_score = anomaly_scores[idx]
            is_fraud = actual_fraud[idx]
            
            # Determine risk level and review priority
            if anomaly_score > 0.9:
                risk_level = "Critical"
                review_priority = "Immediate"
            elif anomaly_score > 0.7:
                risk_level = "High"
                review_priority = "High"
            elif anomaly_score > 0.5:
                risk_level = "Medium"
                review_priority = "Medium"
            else:
                risk_level = "Low"
                review_priority = "Low"
            
            anomalous_transactions.append({
                "transaction_id": str(transaction.get('transaction_id', f"TXN_{idx}")),
                "customer_id": str(transaction.get('customer_id', f"CUST_{idx}")),
                "transaction_amount": float(transaction['transaction_amount']),
                "anomaly_score": float(anomaly_score),
                "is_fraudulent": bool(is_fraud),
                "transaction_date": str(transaction['transaction_date']),
                "merchant_category": str(transaction.get('merchant_category', 'Unknown')),
                "risk_level": risk_level,
                "review_priority": review_priority
            })
        
        # Calculate risk distribution
        risk_distribution = {
            "Critical": len([t for t in anomalous_transactions if t["risk_level"] == "Critical"]),
            "High": len([t for t in anomalous_transactions if t["risk_level"] == "High"]),
            "Medium": len([t for t in anomalous_transactions if t["risk_level"] == "Medium"]),
            "Low": len([t for t in anomalous_transactions if t["risk_level"] == "Low"])
        }
        
        # Generate business insights
        business_insights = {
            "fraud_rate": f"{(np.sum(actual_fraud) / len(actual_fraud) * 100):.2f}%",
            "detection_rate": f"{(true_positives / np.sum(actual_fraud) * 100):.2f}%" if np.sum(actual_fraud) > 0 else "0%",
            "false_alarm_rate": f"{(false_positives / (len(actual_fraud) - np.sum(actual_fraud)) * 100):.2f}%" if (len(actual_fraud) - np.sum(actual_fraud)) > 0 else "0%",
            "avg_anomaly_score": f"{np.mean(anomaly_scores):.3f}",
            "max_anomaly_score": f"{np.max(anomaly_scores):.3f}"
        }
        
        return DateAnalysisResponse(
            date=request.date,
            total_transactions=len(date_data),
            flagged_transactions=int(np.sum(predictions)),
            true_positives=int(true_positives),
            false_positives=int(false_positives),
            true_negatives=int(true_negatives),
            false_negatives=int(false_negatives),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1_score),
            auc_roc=float(auc_roc),
            threshold_used=float(threshold),
            potential_savings=float(potential_savings),
            avg_transaction_amount=float(avg_transaction_amount),
            sample_anomaly_scores=anomaly_scores[:10].tolist()
        )
        
    except Exception as e:
        logger.error(f"Error analyzing date {request.date}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/get-anomalous-transactions", response_model=BusinessAnalysisResponse)
async def get_anomalous_transactions(request: DateAnalysisRequest):
    """Get the top 5% most anomalous transactions for manual fraud review."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Filter data for the requested date
        date_data = full_data[full_data['date'] == request.date]
        
        if len(date_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data found for date {request.date}")
        
        logger.info(f"Getting anomalous transactions for {len(date_data)} transactions on {request.date}")
        
        # Engineer features for the date
        feature_engineer = FeatureFactory.create("combined")
        df_features = feature_engineer.generate_features(date_data)
        
        # Get numeric features
        df_numeric = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in df_numeric.columns:
            df_numeric = df_numeric.drop(columns=['is_fraudulent'])
        
        # Scale the features
        scaled_features = scaler.transform(df_numeric)
        
        # Get reconstruction errors
        reconstructed = model.model.predict(scaled_features)
        mse = np.mean(np.power(scaled_features - reconstructed, 2), axis=1)
        
        # Calculate anomaly scores (normalized reconstruction error)
        anomaly_scores = mse / np.max(mse)
        
        # Get actual fraud labels
        actual_fraud = date_data['is_fraudulent'].values
        
        # Get top 5% most anomalous transactions for manual review
        top_5_percent_count = max(1, int(len(anomaly_scores) * 0.05))
        top_indices = np.argsort(anomaly_scores)[-top_5_percent_count:]
        
        # Create anomalous transactions list for manual review
        anomalous_transactions = []
        for idx in reversed(top_indices):  # Most anomalous first
            transaction = date_data.iloc[idx]
            anomaly_score = anomaly_scores[idx]
            is_fraud = actual_fraud[idx]
            
            # Determine risk level and review priority
            if anomaly_score > 0.9:
                risk_level = "Critical"
                review_priority = "Immediate"
            elif anomaly_score > 0.7:
                risk_level = "High"
                review_priority = "High"
            elif anomaly_score > 0.5:
                risk_level = "Medium"
                review_priority = "Medium"
            else:
                risk_level = "Low"
                review_priority = "Low"
            
            anomalous_transactions.append(AnomalousTransaction(
                transaction_id=str(transaction.get('transaction_id', f"TXN_{idx}")),
                customer_id=str(transaction.get('customer_id', f"CUST_{idx}")),
                transaction_amount=float(transaction['transaction_amount']),
                anomaly_score=float(anomaly_score),
                is_fraudulent=bool(is_fraud),
                transaction_date=str(transaction['transaction_date']),
                merchant_category=str(transaction.get('merchant_category', 'Unknown')),
                risk_level=risk_level,
                review_priority=review_priority
            ))
        
        # Calculate risk distribution
        risk_distribution = {
            "Critical": len([t for t in anomalous_transactions if t.risk_level == "Critical"]),
            "High": len([t for t in anomalous_transactions if t.risk_level == "High"]),
            "Medium": len([t for t in anomalous_transactions if t.risk_level == "Medium"]),
            "Low": len([t for t in anomalous_transactions if t.risk_level == "Low"])
        }
        
        # Calculate business metrics
        avg_transaction_amount = date_data['transaction_amount'].mean()
        predictions = anomaly_scores > (threshold / 100.0)
        true_positives = np.sum((predictions == 1) & (actual_fraud == 1))
        potential_savings = true_positives * avg_transaction_amount
        
        # Generate business insights
        business_insights = {
            "fraud_rate": f"{(np.sum(actual_fraud) / len(actual_fraud) * 100):.2f}%",
            "detection_rate": f"{(true_positives / np.sum(actual_fraud) * 100):.2f}%" if np.sum(actual_fraud) > 0 else "0%",
            "false_alarm_rate": f"{(np.sum((predictions == 1) & (actual_fraud == 0)) / (len(actual_fraud) - np.sum(actual_fraud)) * 100):.2f}%" if (len(actual_fraud) - np.sum(actual_fraud)) > 0 else "0%",
            "avg_anomaly_score": f"{np.mean(anomaly_scores):.3f}",
            "max_anomaly_score": f"{np.max(anomaly_scores):.3f}",
            "review_efficiency": f"{(len([t for t in anomalous_transactions if t.is_fraudulent]) / len(anomalous_transactions) * 100):.1f}%"
        }
        
        return BusinessAnalysisResponse(
            date=request.date,
            total_transactions=len(date_data),
            flagged_transactions=int(np.sum(predictions)),
            top_anomalous_count=len(anomalous_transactions),
            potential_savings=float(potential_savings),
            avg_transaction_amount=float(avg_transaction_amount),
            risk_distribution=risk_distribution,
            anomalous_transactions=anomalous_transactions,
            business_insights=business_insights
        )
        
    except Exception as e:
        logger.error(f"Error getting anomalous transactions for {request.date}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get anomalous transactions: {str(e)}")

@app.post("/update-threshold")
async def update_threshold(request: ThresholdUpdateRequest):
    """Update the fraud detection threshold."""
    global threshold
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        threshold = request.threshold
        logger.info(f"Threshold updated to: {threshold}")
        
        return {
            "message": "Threshold updated successfully",
            "new_threshold": threshold
        }
        
    except Exception as e:
        logger.error(f"Error updating threshold: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/current-threshold")
async def get_current_threshold():
    """Get the current model threshold."""
    return {"threshold": 86.0}  # Fixed threshold from model

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
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        
        # Create feature array
        feature_array = np.array([[features[col] for col in feature_columns]])
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        
        # Get prediction
        anomaly_score = model.predict_anomaly_scores(scaled_features)[0]
        is_fraudulent = anomaly_score > threshold
        
        # Calculate confidence (distance from threshold)
        confidence = min(1.0, abs(anomaly_score - threshold) / threshold)
        
        return {
            "prediction": {
                "is_fraudulent": bool(is_fraudulent),
                "anomaly_score": float(anomaly_score),
                "threshold": float(threshold),
                "confidence": float(confidence)
            }
        }
        
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
        
        # Validate all transactions have required features
        for i, transaction in enumerate(transactions):
            if not all(col in transaction for col in feature_columns):
                missing = [col for col in feature_columns if col not in transaction]
                raise HTTPException(status_code=400, detail=f"Transaction {i} missing features: {missing}")
        
        # Create feature array
        feature_arrays = []
        for transaction in transactions:
            feature_array = [transaction[col] for col in feature_columns]
            feature_arrays.append(feature_array)
        
        feature_matrix = np.array(feature_arrays)
        
        # Scale features
        scaled_features = scaler.transform(feature_matrix)
        
        # Get predictions
        anomaly_scores = model.predict_anomaly_scores(scaled_features)
        predictions = anomaly_scores > threshold
        
        # Calculate confidences
        confidences = np.minimum(1.0, np.abs(anomaly_scores - threshold) / threshold)
        
        # Format results
        results = []
        for i, (score, pred, conf) in enumerate(zip(anomaly_scores, predictions, confidences)):
            results.append({
                "transaction_id": i,
                "is_fraudulent": bool(pred),
                "anomaly_score": float(score),
                "threshold": float(threshold),
                "confidence": float(conf)
            })
        
        return {
            "predictions": results,
            "summary": {
                "total_transactions": len(results),
                "fraud_count": int(np.sum(predictions)),
                "legitimate_count": int(np.sum(~predictions)),
                "fraud_rate": float(np.mean(predictions))
            }
        }
        
    except Exception as e:
        logger.error(f"Error making batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        return ModelInfoResponse(
            model_type="Autoencoder",
            strategy="combined",
            feature_count=len(feature_columns) if feature_columns else 0,
            feature_columns=feature_columns or [],
            threshold=86.0,  # Fixed threshold from model
            model_loaded=True
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 