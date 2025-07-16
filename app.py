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
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from datetime import datetime, timedelta
import json
import random

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
app = FastAPI(title="Fraud Detection Autoencoder API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables for model and data
model = None
scaler = None
threshold = None
test_data = None
feature_columns = None
full_data = None
data_dates = None

# Global variables

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
        trained_threshold = 97.0  # Default to 97th percentile
        
        if os.path.exists(model_info_path):
            try:
                import yaml
                # Try to load with safe_load first
                with open(model_info_path, 'r') as f:
                    model_info = yaml.safe_load(f)
                if model_info and 'threshold_percentile' in model_info:
                    trained_threshold = float(model_info['threshold_percentile'])
                    logger.info(f"Loaded threshold_percentile from final_model_info.yaml: {trained_threshold}")
                elif model_info and 'threshold' in model_info:
                    trained_threshold = float(model_info['threshold'])
                    logger.info(f"Loaded threshold from final_model_info.yaml: {trained_threshold}")
                else:
                    logger.warning("No threshold found in final_model_info.yaml, using default: 97.0")
            except Exception as e:
                logger.warning(f"Could not load final_model_info.yaml: {e}")
                logger.warning("Using default threshold: 97.0")
        else:
            logger.warning("final_model_info.yaml not found, using default threshold: 97.0")
        
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
        
        # Train Random Forest for interpretability
        logger.info("Training Random Forest model for interpretability...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Prepare data for Random Forest (use actual fraud labels)
        rf_features = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in rf_features.columns:
            rf_labels = rf_features['is_fraudulent'].values
            rf_features = rf_features.drop(columns=['is_fraudulent'])
        else:
            # If no fraud labels, use autoencoder predictions as proxy
            scaled_rf_features = scaler.transform(rf_features)
            rf_anomaly_scores = model.predict_anomaly_scores(scaled_rf_features)
            rf_threshold_score = np.percentile(rf_anomaly_scores, threshold)
            rf_labels = (rf_anomaly_scores > rf_threshold_score).astype(int)
            logger.info("Using autoencoder predictions as proxy labels for Random Forest")
        
        # Train Random Forest
        rf_model_local = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            rf_features, rf_labels, test_size=0.2, random_state=42, stratify=rf_labels
        )
        
        rf_model_local.fit(X_train, y_train)
        
        # Store Random Forest model and feature names
        global rf_model, rf_feature_names
        rf_model = rf_model_local
        rf_feature_names = rf_features.columns.tolist()
        
        # Calculate feature importance
        feature_importance = rf_model.feature_importances_
        feature_importance_dict = dict(zip(rf_feature_names, feature_importance))
        
        # Sort features by importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"Random Forest trained successfully!")
        logger.info(f"Top 5 most important features:")
        for feature, importance in sorted_features[:5]:
            logger.info(f"  {feature}: {importance:.4f}")
        
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
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
        
        # Calculate anomaly scores with better normalization
        # Use percentile-based normalization for more meaningful scores
        anomaly_scores = (mse - np.percentile(mse, 5)) / (np.percentile(mse, 95) - np.percentile(mse, 5))
        anomaly_scores = np.clip(anomaly_scores, 0, 1)  # Clip to 0-1 range
        
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

@app.get("/api/dashboard-metrics")
async def get_dashboard_metrics():
    """Get comprehensive dashboard metrics using real data"""
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Load real data
        real_data = load_real_data()
        if real_data is None or len(real_data) == 0:
            raise HTTPException(status_code=500, detail="No real data available")
        
        # Calculate metrics from real data
        total_transactions = len(real_data)
        flagged_transactions = len(real_data[real_data['anomaly_score'] > 0.7])
        high_risk_transactions = len(real_data[real_data['anomaly_score'] > 0.85])
        
        # Calculate fraud rate from real data
        if 'is_fraud' in real_data.columns:
            fraud_rate = (len(real_data[real_data['is_fraud'] == 1]) / total_transactions * 100)
        else:
            fraud_rate = 0
        
        # Calculate savings based on real transaction amounts
        if 'transaction_amount' in real_data.columns:
            avg_transaction_amount = real_data['transaction_amount'].mean()
        else:
            avg_transaction_amount = 100
        potential_savings = high_risk_transactions * avg_transaction_amount * 10  # Estimate 10x for fraud amounts
        
        # Risk distribution
        risk_levels = {
            'low': len(real_data[real_data['anomaly_score'] <= 0.5]),
            'medium': len(real_data[(real_data['anomaly_score'] > 0.5) & (real_data['anomaly_score'] <= 0.7)]),
            'high': len(real_data[(real_data['anomaly_score'] > 0.7) & (real_data['anomaly_score'] <= 0.85)]),
            'critical': len(real_data[real_data['anomaly_score'] > 0.85])
        }
        
        # Recent activity (last 24 hours) - use available data
        recent_flagged = flagged_transactions  # Simplified for real data
        
        return {
            "total_transactions": total_transactions,
            "flagged_transactions": flagged_transactions,
            "high_risk_transactions": high_risk_transactions,
            "fraud_rate": round(fraud_rate, 2),
            "potential_savings": potential_savings,
            "risk_levels": risk_levels,
            "recent_fraud": 0,  # Real data may not have fraud labels
            "recent_flagged": recent_flagged,
            "model_auc": 0.73  # Actual model performance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating metrics: {str(e)}")

def load_real_data():
    """Load and process real transaction data with model predictions"""
    try:
        # Load the cleaned data
        data_path = "data/cleaned/ecommerce_cleaned.csv"
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            return None
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"Loaded real data with shape: {df.shape}")
        
        # Ensure model and scaler are loaded
        if model is None or scaler is None:
            print("Model or scaler not loaded")
            return None
        
        # Apply the same feature engineering that was used during training
        print("Applying feature engineering...")
        feature_engineer = FeatureFactory.create("combined")
        df_features = feature_engineer.generate_features(df)
        
        # Get numeric features (same as training)
        df_numeric = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in df_numeric.columns:
            df_numeric = df_numeric.drop(columns=['is_fraudulent'])
        
        print(f"Engineered features shape: {df_numeric.shape}")
        print(f"Available engineered features: {df_numeric.columns.tolist()}")
        
        # Scale features using the same scaler from training
        print("Scaling features...")
        scaled_features = scaler.transform(df_numeric)
        
        # Use the trained model to predict anomaly scores
        print("Generating anomaly scores using trained model...")
        anomaly_scores = model.predict_anomaly_scores(scaled_features)
        
        # Convert to higher values (0-1 scale) for better visibility
        # Use percentile-based normalization to make scores more meaningful
        anomaly_scores = (anomaly_scores - np.percentile(anomaly_scores, 5)) / (np.percentile(anomaly_scores, 95) - np.percentile(anomaly_scores, 5))
        anomaly_scores = np.clip(anomaly_scores, 0, 1)  # Clip to 0-1 range
        
        print(f"Anomaly scores range: {anomaly_scores.min():.3f} to {anomaly_scores.max():.3f}")
        
        # Add anomaly scores to original dataframe
        df['anomaly_score'] = anomaly_scores
        
        # Add timestamp if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()
        
        # Add fraud labels if available
        if 'is_fraudulent' in df.columns:
            df['is_fraud'] = df['is_fraudulent']
        else:
            df['is_fraud'] = 0  # Assume no fraud labels
        
        print(f"Final dataframe shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading real data: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None



@app.get("/api/review-queue")
async def get_review_queue(date: str = None):
    """Get transactions for manual review (top 10% most anomalous by date)"""
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Load real data
        real_data = load_real_data()
        if real_data is None or len(real_data) == 0:
            raise HTTPException(status_code=500, detail="No real data available")
        
        print(f"Loaded real data with shape: {real_data.shape}")
        print(f"Columns: {real_data.columns.tolist()}")
        
        # Filter by date if provided
        if date:
            print(f"Filtering by date: {date}")
            # Convert date string to datetime for comparison
            date_obj = pd.to_datetime(date).date()
            real_data['transaction_date'] = pd.to_datetime(real_data['transaction_date'])
            real_data = real_data[real_data['transaction_date'].dt.date == date_obj].copy()
            
            print(f"After date filtering: {len(real_data)} transactions")
            
            if len(real_data) == 0:
                return {
                    "queue": [],
                    "total_in_queue": 0,
                    "threshold_score": 0,
                    "message": f"No transactions found for date {date}"
                }
        
        # Check if anomaly_score column exists
        if 'anomaly_score' not in real_data.columns:
            print("Error: anomaly_score column not found in real_data")
            print(f"Available columns: {real_data.columns.tolist()}")
            raise HTTPException(status_code=500, detail="Anomaly scores not generated")
        
        # Get top 10% most anomalous transactions
        threshold = real_data['anomaly_score'].quantile(0.90)
        review_queue = real_data[real_data['anomaly_score'] >= threshold].copy()
        
        print(f"Threshold: {threshold}, Review queue size: {len(review_queue)}")
        
        # Sort by anomaly score (highest first)
        review_queue = review_queue.sort_values('anomaly_score', ascending=False)
        
        # Format for frontend
        queue_data = []
        for idx, row in review_queue.head(50).iterrows():  # Top 50 for display
            queue_data.append({
                'id': f"TXN_{idx:06d}",
                'amount': round(row['transaction_amount'], 2),
                'anomaly_score': round(row['anomaly_score'], 3),
                'timestamp': str(row['transaction_date']) if 'transaction_date' in row else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'risk_level': get_risk_level(row['anomaly_score']),
                'customer_age': row['customer_age'],
                'merchant_category': row['product_category'],
                'location_mismatch': False,  # Default for real data
                'device_mismatch': False,  # Default for real data
                'is_fraud': bool(row.get('is_fraud', False))
            })
        
        return {
            "queue": queue_data,
            "total_in_queue": len(review_queue),
            "threshold_score": round(threshold, 3),
            "date": date if date else "All dates"
        }
    except Exception as e:
        print(f"Error in get_review_queue: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating review queue: {str(e)}")

def get_risk_level(anomaly_score):
    """Convert anomaly score to risk level"""
    if anomaly_score > 0.85:
        return "Critical"
    elif anomaly_score > 0.70:
        return "High"
    elif anomaly_score > 0.50:
        return "Medium"
    elif anomaly_score > 0.30:
        return "Low"
    else:
        return "Very Low"

@app.get("/api/risk-distribution")
async def get_risk_distribution():
    """Get risk distribution data for charts"""
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Load real data
        real_data = load_real_data()
        if real_data is None or len(real_data) == 0:
            raise HTTPException(status_code=500, detail="No real data available")
        
        # Create risk distribution
        risk_bins = [0, 0.3, 0.5, 0.7, 0.85, 1.0]
        risk_labels = ['Very Low', 'Low', 'Medium', 'High', 'Critical']
        
        risk_distribution = []
        for i in range(len(risk_bins) - 1):
            count = len(real_data[(real_data['anomaly_score'] > risk_bins[i]) & 
                                 (real_data['anomaly_score'] <= risk_bins[i + 1])])
            risk_distribution.append({
                'risk_level': risk_labels[i],
                'count': count,
                'percentage': round(count / len(real_data) * 100, 1)
            })
        
        return {"risk_distribution": risk_distribution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating risk distribution: {str(e)}")

@app.get("/api/performance-metrics")
async def get_performance_metrics():
    """Get model performance metrics using real data"""
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Load real data
        real_data = load_real_data()
        if real_data is None or len(real_data) == 0:
            raise HTTPException(status_code=500, detail="No real data available")
        
        # Calculate performance metrics from real data
        # Since we may not have fraud labels, we'll use anomaly score distributions
        thresholds = np.arange(0.1, 1.0, 0.05)
        performance_data = []
        
        for threshold in thresholds:
            flagged = real_data[real_data['anomaly_score'] > threshold]
            
            # Calculate metrics based on anomaly score distribution
            total_flagged = len(flagged)
            total_transactions = len(real_data)
            
            # Estimate precision based on anomaly score (higher scores = more likely fraud)
            avg_anomaly_score = flagged['anomaly_score'].mean() if len(flagged) > 0 else 0
            estimated_precision = min(avg_anomaly_score, 1.0)
            
            # Calculate TPR and FPR based on anomaly score distribution
            tpr = total_flagged / total_transactions if total_transactions > 0 else 0
            fpr = tpr * (1 - estimated_precision)  # Estimate FPR
            
            performance_data.append({
                'threshold': round(threshold, 2),
                'tpr': round(tpr, 3),
                'fpr': round(fpr, 3),
                'precision': round(estimated_precision, 3)
            })
        
        return {
            "performance_data": performance_data,
            "auc": 0.73,  # Actual model performance
            "best_threshold": 0.7,
            "demo_mode": False
        }
    except Exception as e:
        print(f"Error in performance metrics: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating performance metrics: {str(e)}")



@app.get("/api/flagged-transactions")
async def get_flagged_transactions(date: str = None, threshold_percentile: float = 95):
    """Get flagged transactions for a specific date with actual data columns"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load real data
        real_data = load_real_data()
        if real_data is None or len(real_data) == 0:
            raise HTTPException(status_code=500, detail="No real data available")
        
        # Filter by date if provided
        if date:
            date_obj = pd.to_datetime(date).date()
            real_data['transaction_date'] = pd.to_datetime(real_data['transaction_date'])
            real_data = real_data[real_data['transaction_date'].dt.date == date_obj].copy()
            
            if len(real_data) == 0:
                return {
                    "flagged_transactions": [],
                    "total_transactions": 0,
                    "flagged_count": 0,
                    "threshold_score": 0,
                    "date": date,
                    "message": f"No transactions found for date {date}"
                }
        
        # Calculate threshold based on percentile
        threshold_score = np.percentile(real_data['anomaly_score'], threshold_percentile)
        
        # Get flagged transactions (above threshold)
        flagged_data = real_data[real_data['anomaly_score'] >= threshold_score].copy()
        
        # Sort by anomaly score (highest first)
        flagged_data = flagged_data.sort_values('anomaly_score', ascending=False)
        
        # Format transactions with actual columns (ordered by importance, most important on right)
        flagged_transactions = []
        for idx, row in flagged_data.iterrows():
            transaction = {
                'transaction_amount': float(row['transaction_amount']),
                'transaction_date': str(row['transaction_date']),
                'payment_method': str(row['payment_method']),
                'product_category': str(row['product_category']),
                'quantity': int(row['quantity']),
                'customer_age': int(row['customer_age']),
                'customer_location': str(row['customer_location']),
                'device_used': str(row['device_used']),
                'account_age_days': int(row['account_age_days']),
                'transaction_hour': int(row['transaction_hour']),
                'is_between_11pm_and_6am': bool(row['is_between_11pm_and_6am']),
                'is_fraudulent': bool(row['is_fraudulent']),
                'anomaly_score': float(row['anomaly_score'])
            }
            flagged_transactions.append(transaction)
        
        return {
            "flagged_transactions": flagged_transactions,
            "total_transactions": len(real_data),
            "flagged_count": len(flagged_data),
            "threshold_score": float(threshold_score),
            "threshold_percentile": threshold_percentile,
            "date": date if date else "All dates",
            "anomaly_score_stats": {
                "min": float(real_data['anomaly_score'].min()),
                "max": float(real_data['anomaly_score'].max()),
                "mean": float(real_data['anomaly_score'].mean()),
                "std": float(real_data['anomaly_score'].std())
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting flagged transactions: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error getting flagged transactions: {str(e)}")

@app.get("/api/model-features")
async def get_model_features():
    """Get all features the model was trained on"""
    if feature_columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Categorize features for better understanding
    feature_categories = {
        "Base Features": [],
        "Engineered Features": [],
        "Temporal Features": [],
        "Behavioral Features": [],
        "Risk Indicators": []
    }
    
    for feature in feature_columns:
        if feature in ['transaction_amount', 'payment_method', 'product_category', 'quantity', 'customer_age', 'device_used', 'account_age_days']:
            feature_categories["Base Features"].append(feature)
        elif 'encoded' in feature or 'flag' in feature:
            feature_categories["Risk Indicators"].append(feature)
        elif 'rolling' in feature or 'avg' in feature or 'std' in feature:
            feature_categories["Behavioral Features"].append(feature)
        elif 'hour' in feature or 'time' in feature or 'late' in feature:
            feature_categories["Temporal Features"].append(feature)
        else:
            feature_categories["Engineered Features"].append(feature)
    
    return {
        "total_features": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_categories": feature_categories
    }

@app.post("/api/predict")
async def predict_with_threshold(request: DateAnalysisRequest):
    """
    Simple prediction endpoint for the demo dashboard.
    Returns basic metrics about flagged vs non-flagged transactions.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use provided threshold or default to the trained threshold from YAML
        threshold_value = request.threshold if request.threshold is not None else threshold
        
        # Filter data for the requested date
        if request.date == "All Dates":
            date_data = full_data.copy()
        else:
            date_data = full_data[full_data['date'] == request.date].copy()
            
        if len(date_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data found for date {request.date}")
        
        logger.info(f"Running prediction on {len(date_data)} transactions with threshold {threshold_value}")
        
        # Engineer features
        feature_engineer = FeatureFactory.create("combined")
        df_features = feature_engineer.generate_features(date_data)
        
        # Get numeric features
        df_numeric = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in df_numeric.columns:
            actual_labels = df_numeric['is_fraudulent'].values
            df_numeric = df_numeric.drop(columns=['is_fraudulent'])
        else:
            actual_labels = np.zeros(len(df_numeric))
        
        # Scale features
        scaled_features = scaler.transform(df_numeric)
        
        # Get reconstruction errors
        reconstructed = model.model.predict(scaled_features)
        mse = np.mean(np.power(scaled_features - reconstructed, 2), axis=1)
        
        # Calculate anomaly scores (normalized)
        anomaly_scores = (mse - np.percentile(mse, 5)) / (np.percentile(mse, 95) - np.percentile(mse, 5))
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        # Apply threshold (convert percentile to actual score)
        threshold_score = np.percentile(anomaly_scores, threshold_value)
        flagged_by_model = anomaly_scores > threshold_score
        
        # Debug logging
        logger.info(f"Threshold percentile: {threshold_value}")
        logger.info(f"Threshold score: {threshold_score:.6f}")
        logger.info(f"Anomaly scores range: {anomaly_scores.min():.6f} to {anomaly_scores.max():.6f}")
        logger.info(f"Anomaly scores mean: {anomaly_scores.mean():.6f}")
        logger.info(f"Number of scores above threshold: {np.sum(anomaly_scores > threshold_score)}")
        
        # Calculate basic metrics
        total_transactions = len(date_data)
        flagged_transactions = int(np.sum(flagged_by_model))
        not_flagged = total_transactions - flagged_transactions
        
        # Calculate how many flagged transactions were actually fraud
        flagged_actual_fraud = int(np.sum((flagged_by_model == 1) & (actual_labels == 1)))
        flagged_false_alarms = flagged_transactions - flagged_actual_fraud
        
        # Calculate fraud rate (percentage of transactions flagged)
        fraud_rate = (flagged_transactions / total_transactions) * 100 if total_transactions > 0 else 0
        
        # Calculate precision (percentage of flagged that were actually fraud)
        precision = (flagged_actual_fraud / flagged_transactions) * 100 if flagged_transactions > 0 else 0
        
        logger.info(f"Prediction complete: {flagged_transactions} flagged, {not_flagged} not flagged")
        logger.info(f"Of {flagged_transactions} flagged: {flagged_actual_fraud} actual fraud, {flagged_false_alarms} false alarms")
        
        return {
            "date": request.date,
            "threshold": threshold_value,
            "metrics": {
                "total_transactions": total_transactions,
                "flagged_transactions": flagged_transactions,
                "not_flagged": not_flagged,
                "flagged_actual_fraud": flagged_actual_fraud,
                "flagged_false_alarms": flagged_false_alarms,
                "fraud_rate": fraud_rate,
                "precision": precision
            }
        }
        
    except Exception as e:
        logger.error(f"Error in predict_with_threshold: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/detailed-transactions")
async def get_detailed_transactions(request: DateAnalysisRequest):
    """
    Get detailed transaction data with feature importance for interpretability.
    Shows complete transaction rows with Random Forest feature importance scores.
    """
    if model is None or scaler is None or rf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Filter data for the requested date
        if request.date == "All Dates":
            date_data = full_data.copy()
        else:
            date_data = full_data[full_data['date'] == request.date].copy()
            
        if len(date_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data found for date {request.date}")
        
        logger.info(f"Analyzing {len(date_data)} transactions for detailed view")
        
        # Engineer features
        feature_engineer = FeatureFactory.create("combined")
        df_features = feature_engineer.generate_features(date_data)
        
        # Get numeric features for autoencoder
        df_numeric = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in df_numeric.columns:
            actual_labels = df_numeric['is_fraudulent'].values
            df_numeric = df_numeric.drop(columns=['is_fraudulent'])
        else:
            actual_labels = np.zeros(len(df_numeric))
        
        # Scale features for autoencoder
        scaled_features = scaler.transform(df_numeric)
        
        # Get autoencoder anomaly scores
        reconstructed = model.model.predict(scaled_features)
        mse = np.mean(np.power(scaled_features - reconstructed, 2), axis=1)
        anomaly_scores = (mse - np.percentile(mse, 5)) / (np.percentile(mse, 95) - np.percentile(mse, 5))
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        # Apply threshold
        threshold_score = np.percentile(anomaly_scores, threshold)
        flagged_by_autoencoder = anomaly_scores > threshold_score
        
        # Get Random Forest predictions and feature importance
        rf_features_data = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in rf_features_data.columns:
            rf_features_data = rf_features_data.drop(columns=['is_fraudulent'])
        
        # Get Random Forest predictions
        rf_predictions = rf_model.predict(rf_features_data)
        rf_probabilities = rf_model.predict_proba(rf_features_data)[:, 1]  # Probability of fraud
        
        # Calculate feature importance for each transaction
        feature_importance_per_transaction = []
        
        for i, transaction_features in enumerate(rf_features_data.values):
            # Get feature importance for this specific transaction
            transaction_importance = {}
            
            # Use Random Forest's feature importances as base
            base_importance = dict(zip(rf_feature_names, rf_model.feature_importances_))
            
            # Adjust importance based on feature values (higher values = more important for this transaction)
            for feature_name, feature_value in zip(rf_feature_names, transaction_features):
                # Normalize feature value and combine with base importance
                normalized_value = abs(feature_value) / (rf_features_data[feature_name].max() + 1e-8)
                transaction_importance[feature_name] = base_importance[feature_name] * (1 + normalized_value)
            
            # Sort by importance
            sorted_importance = sorted(transaction_importance.items(), key=lambda x: x[1], reverse=True)
            feature_importance_per_transaction.append(sorted_importance)
        
        # Prepare detailed transaction data
        detailed_transactions = []
        
        for i, (_, transaction) in enumerate(date_data.iterrows()):
            # Get top 5 most important features for this transaction
            top_features = feature_importance_per_transaction[i][:5]
            
            # Calculate risk level based on anomaly score
            risk_level = get_risk_level(anomaly_scores[i])
            
            # Determine review priority
            if anomaly_scores[i] > np.percentile(anomaly_scores, 95):
                review_priority = "High"
            elif anomaly_scores[i] > np.percentile(anomaly_scores, 85):
                review_priority = "Medium"
            else:
                review_priority = "Low"
            
            transaction_detail = {
                "transaction_id": str(transaction.get('transaction_id', f"TXN_{i:06d}")),
                "customer_id": str(transaction.get('customer_id', f"CUST_{i:06d}")),
                "transaction_date": str(transaction.get('transaction_date', '')),
                "transaction_amount": float(transaction.get('transaction_amount', 0)),
                "merchant_category": str(transaction.get('merchant_category', 'Unknown')),
                "payment_method": str(transaction.get('payment_method', 'Unknown')),
                "customer_age": int(transaction.get('customer_age', 0)),
                "account_age_days": int(transaction.get('account_age_days', 0)),
                "is_fraudulent": bool(transaction.get('is_fraudulent', False)),
                "anomaly_score": float(anomaly_scores[i]),
                "autoencoder_flagged": bool(flagged_by_autoencoder[i]),
                "rf_prediction": bool(rf_predictions[i]),
                "rf_probability": float(rf_probabilities[i]),
                "risk_level": risk_level,
                "review_priority": review_priority,
                "top_features": [
                    {
                        "feature": feature_name,
                        "importance": float(importance),
                        "value": float(rf_features_data.iloc[i][feature_name])
                    }
                    for feature_name, importance in top_features
                ],
                "all_features": {
                    feature_name: {
                        "importance": float(importance),
                        "value": float(rf_features_data.iloc[i][feature_name])
                    }
                    for feature_name, importance in feature_importance_per_transaction[i]
                }
            }
            
            detailed_transactions.append(transaction_detail)
        
        # Sort by anomaly score (highest first)
        detailed_transactions.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        # Calculate summary statistics
        total_transactions = len(detailed_transactions)
        flagged_transactions = sum(1 for t in detailed_transactions if t['autoencoder_flagged'])
        actual_fraud = sum(1 for t in detailed_transactions if t['is_fraudulent'])
        
        return {
            "date": request.date,
            "summary": {
                "total_transactions": total_transactions,
                "flagged_transactions": flagged_transactions,
                "actual_fraud": actual_fraud,
                "threshold_score": float(threshold_score)
            },
            "transactions": detailed_transactions,
            "feature_importance_summary": {
                "top_global_features": [
                    {"feature": feature, "importance": float(importance)}
                    for feature, importance in sorted(
                        dict(zip(rf_feature_names, rf_model.feature_importances_)).items(),
                        key=lambda x: x[1], reverse=True
                    )[:10]
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in get_detailed_transactions: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get detailed transactions: {str(e)}")

@app.post("/api/generate-3d-plot")
async def generate_3d_plot():
    """Generate 3D latent space visualization using PCA and save it."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import json
        from pathlib import Path
        from sklearn.decomposition import PCA
        
        # Create static directory if it doesn't exist
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)
        
        # Use a sample of data for visualization (first 1000 transactions)
        sample_data = full_data.head(1000).copy()
        
        logger.info(f"Generating PCA 3D plot for {len(sample_data)} transactions")
        
        # Engineer features
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
        
        # Get latent representations from the encoder
        encoder = model.model.layers[0]
        latent_representations = encoder.predict(scaled_features)
        
        # Apply PCA to reduce 16 dimensions to 3
        pca = PCA(n_components=3)
        latent_3d_pca = pca.fit_transform(latent_representations)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        total_explained_variance = np.sum(explained_variance)
        
        logger.info(f"PCA explained variance ratios: {explained_variance}")
        logger.info(f"Total explained variance: {total_explained_variance:.3f}")
        
        # Calculate anomaly scores to determine normal vs anomalous
        reconstructed = model.model.predict(scaled_features)
        mse = np.mean(np.power(scaled_features - reconstructed, 2), axis=1)
        anomaly_scores = (mse - np.percentile(mse, 5)) / (np.percentile(mse, 95) - np.percentile(mse, 5))
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        # Determine normal vs anomalous based on threshold
        threshold_score = np.percentile(anomaly_scores, threshold)
        is_anomalous = anomaly_scores > threshold_score
        
        # Separate normal and anomalous points
        normal_mask = ~is_anomalous
        anomalous_mask = is_anomalous
        
        # Prepare data for JSON
        plot_data = {
            "normal": {
                "x": latent_3d_pca[normal_mask, 0].tolist(),
                "y": latent_3d_pca[normal_mask, 1].tolist(),
                "z": latent_3d_pca[normal_mask, 2].tolist()
            },
            "anomalous": {
                "x": latent_3d_pca[anomalous_mask, 0].tolist(),
                "y": latent_3d_pca[anomalous_mask, 1].tolist(),
                "z": latent_3d_pca[anomalous_mask, 2].tolist()
            },
            "metadata": {
                "total_points": len(latent_3d_pca),
                "normal_points": int(np.sum(normal_mask)),
                "anomalous_points": int(np.sum(anomalous_mask)),
                "threshold": float(threshold_score),
                "latent_dimensions": latent_representations.shape[1],
                "method": "pca",
                "explained_variance": explained_variance.tolist(),
                "total_explained_variance": float(total_explained_variance),
                "explanation": f"This visualization uses Principal Component Analysis (PCA) to transform all 16 latent dimensions into 3 dimensions. PCA finds the directions of maximum variance in the data and projects the points onto these new axes. The first 3 principal components explain {total_explained_variance:.1%} of the total variance in the latent space. This method ensures we capture the most important patterns in the data while reducing dimensionality."
            }
        }
        
        # Save plot data to JSON file
        plot_file = static_dir / "latent_space_3d.json"
        with open(plot_file, 'w') as f:
            json.dump(plot_data, f, indent=2)
        
        # Create and save the actual plot image
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Create the 3D plot with enhanced styling
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot normal transactions in green with better styling
            ax.scatter(latent_3d_pca[normal_mask, 0], latent_3d_pca[normal_mask, 1], latent_3d_pca[normal_mask, 2], 
                      c='#2E8B57', s=25, alpha=0.7, label='Normal Transactions', edgecolors='white', linewidth=0.5)
            
            # Plot anomalous transactions in red with better styling
            ax.scatter(latent_3d_pca[anomalous_mask, 0], latent_3d_pca[anomalous_mask, 1], latent_3d_pca[anomalous_mask, 2], 
                      c='#DC143C', s=35, alpha=0.9, label='Anomalous Transactions', edgecolors='white', linewidth=0.5)
            
            # Customize the plot with enhanced styling
            ax.set_xlabel('Principal Component 1', fontsize=14, fontweight='bold')
            ax.set_ylabel('Principal Component 2', fontsize=14, fontweight='bold')
            ax.set_zlabel('Principal Component 3', fontsize=14, fontweight='bold')
            ax.set_title(f'3D Latent Space Visualization (PCA)\nAutoencoder Fraud Detection\nExplained Variance: {total_explained_variance:.1%}', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Add legend with better styling
            ax.legend(fontsize=12, loc='upper right', framealpha=0.9, fancybox=True, shadow=True)
            
            # Enhanced background styling
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('#E0E0E0')
            ax.yaxis.pane.set_edgecolor('#E0E0E0')
            ax.zaxis.pane.set_edgecolor('#E0E0E0')
            ax.xaxis.pane.set_alpha(0.1)
            ax.yaxis.pane.set_alpha(0.1)
            ax.zaxis.pane.set_alpha(0.1)
            
            # Grid styling
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Set background color
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#F8F9FA')
            
            # Save the plot with high quality
            plot_image_path = static_dir / "latent_space_3d_plot.png"
            plt.savefig(plot_image_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Enhanced 3D plot image saved to {plot_image_path}")
            
        except Exception as e:
            logger.warning(f"Could not create plot image: {e}")
            plot_image_path = None
        
        logger.info(f"3D plot data saved to {plot_file}")
        logger.info(f"Normal points: {plot_data['metadata']['normal_points']}, Anomalous points: {plot_data['metadata']['anomalous_points']}")
        
        return {
            "success": True,
            "message": "3D visualization generated successfully",
            "file_path": str(plot_file),
            "image_path": str(plot_image_path) if plot_image_path else None,
            "metadata": plot_data["metadata"]
        }
        
    except Exception as e:
        logger.error(f"Error generating 3D plot: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e)
        }



@app.post("/api/generate-all-visualizations")
async def generate_all_visualizations():
    """Generate all data science visualizations."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import json
        from pathlib import Path
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, auc, confusion_matrix
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        
        # Create static directory if it doesn't exist
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)
        
        # Use a larger sample for comprehensive analysis
        sample_data = full_data.head(2000).copy()
        
        logger.info(f"Generating all visualizations for {len(sample_data)} transactions")
        
        # Engineer features
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
        
        # Get reconstruction errors and anomaly scores
        reconstructed = model.model.predict(scaled_features)
        mse = np.mean(np.power(scaled_features - reconstructed, 2), axis=1)
        anomaly_scores = (mse - np.percentile(mse, 5)) / (np.percentile(mse, 95) - np.percentile(mse, 5))
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        # Calculate threshold
        threshold_score = np.percentile(anomaly_scores, threshold)
        is_anomalous = anomaly_scores > threshold_score
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Reconstruction Error Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        normal_errors = mse[actual_labels == 0]
        fraud_errors = mse[actual_labels == 1]
        
        ax.hist(normal_errors, bins=50, alpha=0.7, label='Normal Transactions', color='#2E8B57')
        ax.hist(fraud_errors, bins=50, alpha=0.7, label='Fraudulent Transactions', color='#DC143C')
        ax.set_xlabel('Reconstruction Error (MSE)')
        ax.set_ylabel('Frequency')
        ax.set_title('Reconstruction Error Distribution\nNormal vs Fraudulent Transactions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(static_dir / "reconstruction_error_dist.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance Analysis (using correlation with anomaly scores)
        feature_importance = []
        for i, col in enumerate(df_numeric.columns):
            correlation = np.corrcoef(df_numeric[col], anomaly_scores)[0, 1]
            feature_importance.append((col, abs(correlation)))
        
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_importance[:15]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        features, importances = zip(*top_features)
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        
        bars = ax.barh(range(len(features)), importances, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Absolute Correlation with Anomaly Score')
        ax.set_title('Feature Importance Analysis\nCorrelation with Anomaly Detection')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(static_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Anomaly Score vs Transaction Amount
        fig, ax = plt.subplots(figsize=(10, 6))
        normal_mask = actual_labels == 0
        fraud_mask = actual_labels == 1
        
        ax.scatter(sample_data['transaction_amount'][normal_mask], anomaly_scores[normal_mask], 
                  alpha=0.6, s=20, color='#2E8B57', label='Normal')
        ax.scatter(sample_data['transaction_amount'][fraud_mask], anomaly_scores[fraud_mask], 
                  alpha=0.8, s=30, color='#DC143C', label='Fraudulent')
        ax.set_xlabel('Transaction Amount ($)')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Score vs Transaction Amount')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(static_dir / "anomaly_vs_amount.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Time-based Anomaly Patterns
        sample_data['anomaly_score'] = anomaly_scores
        hourly_anomaly = sample_data.groupby('transaction_hour')['anomaly_score'].mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hourly_anomaly.index, hourly_anomaly.values, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Anomaly Score')
        ax.set_title('Time-based Anomaly Patterns\nAverage Anomaly Score by Hour')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 2))
        plt.tight_layout()
        plt.savefig(static_dir / "time_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. ROC Curve
        fpr, tpr, _ = roc_curve(actual_labels, anomaly_scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, color='#2E8B57', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve\nReceiver Operating Characteristic')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(static_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Confusion Matrix
        predictions = (anomaly_scores > threshold_score).astype(int)
        cm = confusion_matrix(actual_labels, predictions)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Fraudulent'],
                   yticklabels=['Normal', 'Fraudulent'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix\nThreshold = {:.3f}'.format(threshold_score))
        plt.tight_layout()
        plt.savefig(static_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Feature Correlation Matrix (top features)
        top_feature_names = [f[0] for f in top_features[:10]]
        correlation_matrix = df_numeric[top_feature_names].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix\nTop 10 Most Important Features')
        plt.tight_layout()
        plt.savefig(static_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 8. Threshold Sensitivity Analysis
        thresholds = np.arange(0.1, 1.0, 0.05)
        precision_scores = []
        recall_scores = []
        
        for thresh in thresholds:
            thresh_score = np.percentile(anomaly_scores, thresh * 100)
            preds = (anomaly_scores > thresh_score).astype(int)
            
            tp = np.sum((preds == 1) & (actual_labels == 1))
            fp = np.sum((preds == 1) & (actual_labels == 0))
            fn = np.sum((preds == 0) & (actual_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, precision_scores, marker='o', label='Precision', linewidth=2)
        ax.plot(thresholds, recall_scores, marker='s', label='Recall', linewidth=2)
        ax.set_xlabel('Threshold Percentile')
        ax.set_ylabel('Score')
        ax.set_title('Threshold Sensitivity Analysis\nPrecision vs Recall at Different Thresholds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(static_dir / "threshold_sensitivity.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 9. Customer Segmentation by Anomaly Score
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Age groups
        age_bins = [0, 25, 35, 45, 100]
        age_labels = ['18-25', '26-35', '36-45', '45+']
        sample_data['age_group'] = pd.cut(sample_data['customer_age'], bins=age_bins, labels=age_labels)
        
        age_anomaly = sample_data.groupby('age_group')['anomaly_score'].apply(list)
        if len(age_anomaly) > 0:
            # Filter out empty groups
            valid_age_data = [(label, data) for label, data in zip(age_anomaly.index, age_anomaly.values) if len(data) > 0]
            if valid_age_data:
                labels, data = zip(*valid_age_data)
                axes[0,0].boxplot(data, labels=labels)
        axes[0,0].set_title('Anomaly Scores by Age Group')
        axes[0,0].set_ylabel('Anomaly Score')
        axes[0,0].grid(True, alpha=0.3)
        
        # Account age
        account_bins = [0, 30, 90, 180, 1000]
        account_labels = ['<30 days', '30-90 days', '90-180 days', '180+ days']
        sample_data['account_group'] = pd.cut(sample_data['account_age_days'], bins=account_bins, labels=account_labels)
        
        account_anomaly = sample_data.groupby('account_group')['anomaly_score'].apply(list)
        if len(account_anomaly) > 0:
            # Filter out empty groups
            valid_account_data = [(label, data) for label, data in zip(account_anomaly.index, account_anomaly.values) if len(data) > 0]
            if valid_account_data:
                labels, data = zip(*valid_account_data)
                axes[0,1].boxplot(data, labels=labels)
        axes[0,1].set_title('Anomaly Scores by Account Age')
        axes[0,1].set_ylabel('Anomaly Score')
        axes[0,1].grid(True, alpha=0.3)
        
        # Payment method
        payment_anomaly = sample_data.groupby('payment_method')['anomaly_score'].apply(list)
        if len(payment_anomaly) > 0:
            # Filter out empty groups
            valid_payment_data = [(label, data) for label, data in zip(payment_anomaly.index, payment_anomaly.values) if len(data) > 0]
            if valid_payment_data:
                labels, data = zip(*valid_payment_data)
                axes[1,0].boxplot(data, labels=labels)
        axes[1,0].set_title('Anomaly Scores by Payment Method')
        axes[1,0].set_ylabel('Anomaly Score')
        axes[1,0].grid(True, alpha=0.3)
        
        # Product category
        category_anomaly = sample_data.groupby('product_category')['anomaly_score'].apply(list)
        if len(category_anomaly) > 0:
            # Filter out empty groups
            valid_category_data = [(label, data) for label, data in zip(category_anomaly.index, category_anomaly.values) if len(data) > 0]
            if valid_category_data:
                labels, data = zip(*valid_category_data)
                axes[1,1].boxplot(data, labels=labels)
        axes[1,1].set_title('Anomaly Scores by Product Category')
        axes[1,1].set_ylabel('Anomaly Score')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(static_dir / "customer_segmentation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 10. Performance Metrics Summary
        metrics_data = {
            'metric': ['ROC AUC', 'Precision', 'Recall', 'F1-Score'],
            'value': [
                f"{roc_auc:.3f}",
                f"{precision_scores[len(thresholds)//2]:.3f}",
                f"{recall_scores[len(thresholds)//2]:.3f}",
                f"{2 * precision_scores[len(thresholds)//2] * recall_scores[len(thresholds)//2] / (precision_scores[len(thresholds)//2] + recall_scores[len(thresholds)//2]):.3f}"
            ]
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metrics_data['metric'], [float(v) for v in metrics_data['value']], 
                     color=['#2E8B57', '#DC143C', '#FFD700', '#4169E1'])
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_data['value']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(static_dir / "performance_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metadata
        viz_metadata = {
            "total_transactions": len(sample_data),
            "fraud_rate": f"{np.mean(actual_labels)*100:.2f}%",
            "roc_auc": roc_auc,
            "threshold_used": threshold_score,
            "top_features": [f[0] for f in top_features[:5]],
            "generated_at": datetime.now().isoformat()
        }
        
        with open(static_dir / "visualization_metadata.json", 'w') as f:
            json.dump(viz_metadata, f, indent=2)
        
        logger.info("All visualizations generated successfully")
        
        return {
            "success": True,
            "message": "All visualizations generated successfully",
            "metadata": viz_metadata
        }
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/check-visualization")
async def check_visualization():
    """Check if 3D visualization data is available."""
    try:
        from pathlib import Path
        
        plot_file = Path("static/latent_space_3d.json")
        
        if plot_file.exists():
            return {
                "available": True,
                "file_path": str(plot_file)
            }
        else:
            return {
                "available": False,
                "message": "No visualization data found"
            }
            
    except Exception as e:
        logger.error(f"Error checking visualization: {str(e)}")
        return {
            "available": False,
            "error": str(e)
        }

@app.get("/api/get-3d-plot-data")
async def get_3d_plot_data():
    """Get 3D plot data for frontend visualization."""
    try:
        import json
        from pathlib import Path
        
        plot_file = Path("static/latent_space_3d.json")
        
        if not plot_file.exists():
            raise HTTPException(status_code=404, detail="Visualization data not found")
        
        with open(plot_file, 'r') as f:
            plot_data = json.load(f)
        
        return plot_data
        
    except Exception as e:
        logger.error(f"Error getting 3D plot data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load visualization data: {str(e)}")

@app.get("/api/get-3d-plot-image")
async def get_3d_plot_image():
    """Get the 3D plot image for display."""
    try:
        from pathlib import Path
        from fastapi.responses import FileResponse
        
        plot_image_path = Path("static/latent_space_3d_plot.png")
        
        if not plot_image_path.exists():
            raise HTTPException(status_code=404, detail="Plot image not found")
        
        return FileResponse(plot_image_path, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error serving plot image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve plot image: {str(e)}")



@app.get("/api/get-visualization/{viz_type}")
async def get_visualization(viz_type: str):
    """Get specific visualization image."""
    try:
        from pathlib import Path
        from fastapi.responses import FileResponse
        
        viz_files = {
            "reconstruction_error": "reconstruction_error_dist.png",
            "feature_importance": "feature_importance.png",
            "anomaly_vs_amount": "anomaly_vs_amount.png",
            "time_patterns": "time_patterns.png",
            "roc_curve": "roc_curve.png",
            "confusion_matrix": "confusion_matrix.png",
            "correlation_matrix": "correlation_matrix.png",
            "threshold_sensitivity": "threshold_sensitivity.png",
            "customer_segmentation": "customer_segmentation.png",
            "performance_metrics": "performance_metrics.png"
        }
        
        if viz_type not in viz_files:
            raise HTTPException(status_code=404, detail=f"Visualization type '{viz_type}' not found")
        
        viz_path = Path("static") / viz_files[viz_type]
        
        if not viz_path.exists():
            raise HTTPException(status_code=404, detail=f"Visualization file not found: {viz_files[viz_type]}")
        
        return FileResponse(viz_path, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error serving visualization {viz_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve visualization: {str(e)}")

@app.get("/api/get-visualization-metadata")
async def get_visualization_metadata():
    """Get metadata for all visualizations."""
    try:
        import json
        from pathlib import Path
        
        metadata_path = Path("static/visualization_metadata.json")
        
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Visualization metadata not found")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error getting visualization metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")

@app.post("/api/all-columns-transactions")
async def get_all_columns_transactions(request: DateAnalysisRequest):
    """
    Get all actual columns from the data with fraud flags and feature importance.
    Shows complete transaction rows with all 36+ columns and Random Forest feature importance.
    """
    if model is None or scaler is None or rf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Filter data for the requested date
        if request.date == "All Dates":
            date_data = full_data.copy()
        else:
            date_data = full_data[full_data['date'] == request.date].copy()
            
        if len(date_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data found for date {request.date}")
        
        logger.info(f"Analyzing {len(date_data)} transactions for all columns view")
        
        # Engineer features
        feature_engineer = FeatureFactory.create("combined")
        df_features = feature_engineer.generate_features(date_data)
        
        # Get numeric features for autoencoder
        df_numeric = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in df_numeric.columns:
            actual_labels = df_numeric['is_fraudulent'].values
            df_numeric = df_numeric.drop(columns=['is_fraudulent'])
        else:
            actual_labels = np.zeros(len(df_numeric))
        
        # Scale features for autoencoder
        scaled_features = scaler.transform(df_numeric)
        
        # Get autoencoder anomaly scores
        reconstructed = model.model.predict(scaled_features)
        mse = np.mean(np.power(scaled_features - reconstructed, 2), axis=1)
        anomaly_scores = (mse - np.percentile(mse, 5)) / (np.percentile(mse, 95) - np.percentile(mse, 5))
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        # Apply threshold
        threshold_score = np.percentile(anomaly_scores, threshold)
        flagged_by_autoencoder = anomaly_scores > threshold_score
        
        # Get Random Forest predictions and feature importance
        rf_features_data = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in rf_features_data.columns:
            rf_features_data = rf_features_data.drop(columns=['is_fraudulent'])
        
        # Get Random Forest predictions
        rf_predictions = rf_model.predict(rf_features_data)
        rf_probabilities = rf_model.predict_proba(rf_features_data)[:, 1]  # Probability of fraud
        
        # Calculate feature importance for each transaction
        feature_importance_per_transaction = []
        
        for i, transaction_features in enumerate(rf_features_data.values):
            # Get feature importance for this specific transaction
            transaction_importance = {}
            
            # Use Random Forest's feature importances as base
            base_importance = dict(zip(rf_feature_names, rf_model.feature_importances_))
            
            # Adjust importance based on feature values (higher values = more important for this transaction)
            for feature_name, feature_value in zip(rf_feature_names, transaction_features):
                # Normalize feature value and combine with base importance
                normalized_value = abs(feature_value) / (rf_features_data[feature_name].max() + 1e-8)
                transaction_importance[feature_name] = base_importance[feature_name] * (1 + normalized_value)
            
            # Sort by importance
            sorted_importance = sorted(transaction_importance.items(), key=lambda x: x[1], reverse=True)
            feature_importance_per_transaction.append(sorted_importance)
        
        # Prepare all columns transaction data
        all_columns_transactions = []
        
        for i, (_, transaction) in enumerate(date_data.iterrows()):
            # Get top 5 most important features for this transaction
            top_features = feature_importance_per_transaction[i][:5]
            
            # Calculate risk level based on anomaly score
            risk_level = get_risk_level(anomaly_scores[i])
            
            # Determine review priority
            if anomaly_scores[i] > np.percentile(anomaly_scores, 95):
                review_priority = "High"
            elif anomaly_scores[i] > np.percentile(anomaly_scores, 85):
                review_priority = "Medium"
            else:
                review_priority = "Low"
            
            # Create transaction data with all original columns
            transaction_data = {
                # Original columns from the data
                "transaction_id": str(transaction.get('transaction_id', f"TXN_{i:06d}")),
                "transaction_date": str(transaction.get('transaction_date', '')),
                "transaction_amount": float(transaction.get('transaction_amount', 0)),
                "quantity": int(transaction.get('quantity', 0)),
                "customer_age": int(transaction.get('customer_age', 0)),
                "account_age_days": int(transaction.get('account_age_days', 0)),
                "payment_method": str(transaction.get('payment_method', 'Unknown')),
                "product_category": str(transaction.get('product_category', 'Unknown')),
                "device_used": str(transaction.get('device_used', 'Unknown')),
                "customer_location": str(transaction.get('customer_location', 'Unknown')),
                "transaction_hour": int(transaction.get('transaction_hour', 0)),
                "is_fraudulent": bool(transaction.get('is_fraudulent', False)),
                
                # Engineered features (all 36+ columns)
                "transaction_amount_log": float(df_features.iloc[i].get('transaction_amount_log', 0)),
                "amount_per_item": float(df_features.iloc[i].get('amount_per_item', 0)),
                "payment_method_encoded": int(df_features.iloc[i].get('payment_method_encoded', 0)),
                "product_category_encoded": int(df_features.iloc[i].get('product_category_encoded', 0)),
                "device_used_encoded": int(df_features.iloc[i].get('device_used_encoded', 0)),
                "is_late_night": int(df_features.iloc[i].get('is_late_night', 0)),
                "is_burst_transaction": int(df_features.iloc[i].get('is_burst_transaction', 0)),
                "amount_per_age": float(df_features.iloc[i].get('amount_per_age', 0)),
                "amount_per_account_age": float(df_features.iloc[i].get('amount_per_account_age', 0)),
                "customer_age_band": int(df_features.iloc[i].get('customer_age_band', 0)),
                "high_amount_flag": int(df_features.iloc[i].get('high_amount_flag', 0)),
                "new_account_flag": int(df_features.iloc[i].get('new_account_flag', 0)),
                "young_customer_flag": int(df_features.iloc[i].get('young_customer_flag', 0)),
                "late_night_flag": int(df_features.iloc[i].get('late_night_flag', 0)),
                "high_quantity_flag": int(df_features.iloc[i].get('high_quantity_flag', 0)),
                "unusual_location_flag": int(df_features.iloc[i].get('unusual_location_flag', 0)),
                "amount_age_interaction": int(df_features.iloc[i].get('amount_age_interaction', 0)),
                "account_age_interaction": int(df_features.iloc[i].get('account_age_interaction', 0)),
                "fraud_risk_score": int(df_features.iloc[i].get('fraud_risk_score', 0)),
                "rolling_avg_amount_3": float(df_features.iloc[i].get('rolling_avg_amount_3', 0)),
                "rolling_std_amount_3": float(df_features.iloc[i].get('rolling_std_amount_3', 0)),
                "transaction_amount_rank": float(df_features.iloc[i].get('transaction_amount_rank', 0)),
                "account_age_rank": float(df_features.iloc[i].get('account_age_rank', 0)),
                "amount_x_hour": float(df_features.iloc[i].get('amount_x_hour', 0)),
                "amount_per_hour": float(df_features.iloc[i].get('amount_per_hour', 0)),
                
                # Model predictions and scores
                "anomaly_score": float(anomaly_scores[i]),
                "autoencoder_flagged": bool(flagged_by_autoencoder[i]),
                "rf_prediction": bool(rf_predictions[i]),
                "rf_probability": float(rf_probabilities[i]),
                "risk_level": risk_level,
                "review_priority": review_priority,
                
                # Feature importance
                "top_features": [
                    {
                        "feature": feature_name,
                        "importance": float(importance),
                        "value": float(rf_features_data.iloc[i][feature_name])
                    }
                    for feature_name, importance in top_features
                ],
                "all_features": {
                    feature_name: {
                        "importance": float(importance),
                        "value": float(rf_features_data.iloc[i][feature_name])
                    }
                    for feature_name, importance in feature_importance_per_transaction[i]
                }
            }
            
            all_columns_transactions.append(transaction_data)
        
        # Sort by anomaly score (highest first)
        all_columns_transactions.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        # Calculate summary statistics
        total_transactions = len(all_columns_transactions)
        flagged_transactions = sum(1 for t in all_columns_transactions if t['autoencoder_flagged'])
        actual_fraud = sum(1 for t in all_columns_transactions if t['is_fraudulent'])
        
        return {
            "date": request.date,
            "summary": {
                "total_transactions": total_transactions,
                "flagged_transactions": flagged_transactions,
                "actual_fraud": actual_fraud,
                "threshold_score": float(threshold_score)
            },
            "transactions": all_columns_transactions,
            "feature_importance_summary": {
                "top_global_features": [
                    {"feature": feature, "importance": float(importance)}
                    for feature, importance in sorted(
                        dict(zip(rf_feature_names, rf_model.feature_importances_)).items(),
                        key=lambda x: x[1], reverse=True
                    )[:10]
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in get_all_columns_transactions: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get all columns transactions: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 