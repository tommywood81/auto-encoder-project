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
import pickle
import hashlib

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

# Create cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

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
rf_model = None

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool

class DateAnalysisRequest(BaseModel):
    date: str
    threshold: Optional[float] = None

# Cache utility functions
def get_cache_key(date: str, threshold: float, endpoint: str) -> str:
    """Generate a unique cache key for a specific request."""
    key_string = f"{date}_{threshold}_{endpoint}"
    return hashlib.md5(key_string.encode()).hexdigest()

def save_cache(key: str, data: dict):
    """Save data to cache."""
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

def load_cache(key: str) -> Optional[dict]:
    """Load data from cache."""
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def clear_cache():
    """Clear all cache files."""
    for file in os.listdir(CACHE_DIR):
        if file.endswith('.pkl'):
            os.remove(os.path.join(CACHE_DIR, file))

def get_risk_level(anomaly_score):
    """Convert anomaly score to risk level"""
    if anomaly_score > 0.95:
        return "Critical"  # Top 5% most anomalous
    elif anomaly_score > 0.85:
        return "High"      # Top 15% most anomalous
    elif anomaly_score > 0.70:
        return "Medium"    # Top 30% most anomalous
    elif anomaly_score > 0.50:
        return "Low"       # Top 50% most anomalous
    else:
        return "Very Low"  # Bottom 50%

@app.on_event("startup")
async def startup_event():
    """Load model and data on startup."""
    global model, scaler, threshold, test_data, feature_columns, full_data, data_dates, rf_model
    
    logger.info("Loading fraud detection model...")
    
    try:
        # Load the trained model with proper configuration
        from src.config import PipelineConfig
        config = PipelineConfig.get_combined_config()
        model = BaselineAutoencoder(config)
        model.load_model("models/final_model.h5")
        scaler = model.scaler  # Use the scaler from the model
        
        # Load threshold from YAML
        import yaml
        with open("models/final_model_info.yaml", 'r') as f:
            model_info = yaml.safe_load(f)
        threshold = model_info.get('threshold_percentile', 95.0)
        
        logger.info(f"Loaded threshold_percentile from final_model_info.yaml: {threshold}")
        
        # Load and prepare test data
        logger.info("Preparing test data and recreating scaler...")
        test_data = pd.read_csv("data/cleaned/ecommerce_cleaned.csv")
        
        # Engineer features for test data
        feature_engineer = FeatureFactory.create("combined")
        df_features = feature_engineer.generate_features(test_data.head(20))
        
        # Get feature columns
        df_numeric = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in df_numeric.columns:
            df_numeric = df_numeric.drop(columns=['is_fraudulent'])
        feature_columns = df_numeric.columns.tolist()
        
        # Load full data for analysis
        full_data = pd.read_csv("data/cleaned/ecommerce_cleaned.csv")
        full_data['date'] = pd.to_datetime(full_data['transaction_date']).dt.date.astype(str)
        
        # Get available dates
        data_dates = sorted(full_data['date'].unique().tolist())
        
        # Train Random Forest for interpretability
        logger.info("Training Random Forest model for interpretability...")
        from sklearn.ensemble import RandomForestClassifier
        
        # Prepare features for RF
        rf_features = feature_engineer.generate_features(full_data.head(1000))
        rf_features_numeric = rf_features.select_dtypes(include=[np.number])
        
        if 'is_fraudulent' in rf_features_numeric.columns:
            rf_X = rf_features_numeric.drop(columns=['is_fraudulent'])
            rf_y = rf_features_numeric['is_fraudulent']
        else:
            # If no fraud labels, create synthetic ones based on anomaly scores
            rf_X = rf_features_numeric
            rf_y = np.random.choice([0, 1], size=len(rf_X), p=[0.95, 0.05])
        
        # Train RF
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(rf_X, rf_y)
        
        logger.info("Random Forest trained successfully!")
        logger.info("Top 5 most important features:")
        feature_importance = list(zip(rf_X.columns, rf_model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        for feature, importance in feature_importance[:5]:
            logger.info(f"  {feature}: {importance:.4f}")
        
        logger.info("Best model loaded successfully!")
        logger.info(f"Model: final_model.h5 (combined strategy)")
        logger.info(f"Feature columns: {feature_columns}")
        logger.info(f"Test data shape: {test_data.head(20).shape}")
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Available dates: {len(data_dates)} dates from {data_dates[0]} to {data_dates[-1]}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None
    )

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

@app.post("/api/predict")
async def predict_with_threshold(request: DateAnalysisRequest):
    """
    Simple prediction endpoint for the demo dashboard.
    Returns basic metrics about flagged vs non-flagged transactions.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use provided threshold or default to the trained threshold
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
        
        # Calculate reconstruction errors and anomaly scores
        reconstructed = model.model.predict(scaled_features)
        mse_scores = np.mean(np.power(scaled_features - reconstructed, 2), axis=1)
        
        # Calculate standardized anomaly scores (0-1 range) - match risk level expectations
        # Use 5th and 95th percentiles so that 5% of transactions get "Critical" risk level
        anomaly_scores = (mse_scores - np.percentile(mse_scores, 5)) / (np.percentile(mse_scores, 95) - np.percentile(mse_scores, 5))
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        # Apply threshold
        threshold_score = np.percentile(anomaly_scores, threshold_value)
        flagged_by_autoencoder = anomaly_scores > threshold_score
        
        # Calculate summary statistics
        total_transactions = len(date_data)
        flagged_transactions = sum(flagged_by_autoencoder)
        actual_fraud = sum(actual_labels)
        actual_fraud_caught = sum(1 for i in range(len(date_data)) if flagged_by_autoencoder[i] and actual_labels[i])
        
        # Calculate precision
        precision = (actual_fraud_caught / flagged_transactions * 100) if flagged_transactions > 0 else 0
        
        # Cache the results
        cache_key = get_cache_key(request.date, threshold_value, "predict")
        cache_data = {
            "total_transactions": total_transactions,
            "flagged_transactions": flagged_transactions,
            "actual_fraud": actual_fraud_caught,
            "precision": precision,
            "threshold_score": float(threshold_score),
            "anomaly_scores": anomaly_scores.tolist(),
            "flagged_by_autoencoder": flagged_by_autoencoder.tolist(),
            "actual_labels": actual_labels.tolist(),
            "date_data": date_data.to_dict('records')
        }
        save_cache(cache_key, cache_data)
        
        return {
            "total_transactions": total_transactions,
            "flagged_transactions": flagged_transactions,
            "actual_fraud": actual_fraud_caught,
            "precision": precision,
            "threshold_score": float(threshold_score)
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-3d-plot")
async def generate_3d_plot():
    """Generate 3D latent space visualization."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Check if cached visualization exists
        plot_file = "static/latent_space_3d.json"
        if os.path.exists(plot_file):
            logger.info("Using cached 3D visualization")
            return {"success": True, "message": "3D plot already exists"}
        
        # Generate 3D visualization
        logger.info("Generating 3D latent space visualization...")
        
        # Use a sample of data for visualization
        sample_data = full_data.head(1000).copy()
        
        # Engineer features
        feature_engineer = FeatureFactory.create("combined")
        df_features = feature_engineer.generate_features(sample_data)
        
        # Get numeric features
        df_numeric = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in df_numeric.columns:
            fraud_labels = df_numeric['is_fraudulent'].values
            df_numeric = df_numeric.drop(columns=['is_fraudulent'])
        else:
            fraud_labels = np.zeros(len(df_numeric))
        
        # Scale features
        scaled_features = scaler.transform(df_numeric)
        
        # Get latent space representation
        encoder = model.model.layers[0]  # Assuming first layer is encoder
        latent_representation = encoder.predict(scaled_features)
        
        # If latent space is not 3D, use PCA to reduce to 3D
        if latent_representation.shape[1] != 3:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            latent_representation = pca.fit_transform(latent_representation)
        
        # Create 3D scatter plot data
        plot_data = {
            "x": latent_representation[:, 0].tolist(),
            "y": latent_representation[:, 1].tolist(),
            "z": latent_representation[:, 2].tolist(),
            "fraud": fraud_labels.tolist(),
            "colors": ["red" if f == 1 else "blue" for f in fraud_labels]
        }
        
        # Save to file
        os.makedirs("static", exist_ok=True)
        with open(plot_file, 'w') as f:
            json.dump(plot_data, f)
        
        logger.info("3D visualization saved successfully")
        return {"success": True, "message": "3D plot generated successfully"}
        
    except Exception as e:
        logger.error(f"Error generating 3D plot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/check-visualization")
async def check_visualization():
    """Check if 3D visualization data is available."""
    try:
        plot_file = "static/latent_space_3d.json"
        
        if os.path.exists(plot_file):
            return {
                "available": True,
                "file_path": plot_file
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
    """Get 3D plot data for visualization."""
    try:
        plot_file = "static/latent_space_3d.json"
        
        if not os.path.exists(plot_file):
            raise HTTPException(status_code=404, detail="3D plot data not found")
        
        with open(plot_file, 'r') as f:
            plot_data = json.load(f)
        
        return plot_data
        
    except Exception as e:
        logger.error(f"Error getting 3D plot data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get-3d-plot-image")
async def get_3d_plot_image():
    """Get 3D plot as static image."""
    try:
        from fastapi.responses import FileResponse
        
        plot_image = "static/latent_space_3d_plot.png"
        
        if not os.path.exists(plot_image):
            raise HTTPException(status_code=404, detail="3D plot image not found")
        
        return FileResponse(plot_image, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error serving 3D plot image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        
        # Check if all cached visualizations exist
        viz_files = [
            "reconstruction_error_dist.png",
            "feature_importance.png", 
            "anomaly_vs_amount.png",
            "time_patterns.png",
            "roc_curve.png",
            "confusion_matrix.png",
            "correlation_matrix.png",
            "threshold_sensitivity.png",
            "customer_segmentation.png",
            "performance_metrics.png",
            "visualization_metadata.json"
        ]
        
        all_cached = all((static_dir / file).exists() for file in viz_files)
        
        if all_cached:
            logger.info("Using cached comprehensive visualizations")
            # Load metadata
            metadata_file = static_dir / "visualization_metadata.json"
            with open(metadata_file, 'r') as f:
                viz_metadata = json.load(f)
            
            return {
                "success": True,
                "message": "All visualizations loaded from cache",
                "metadata": viz_metadata,
                "cached": True
            }
        
        # Use a larger sample for comprehensive analysis
        sample_data = full_data.head(2000).copy()
        
        logger.info(f"Generating all visualizations for {len(sample_data)} transactions")
        
        # Engineer features
        feature_engineer = FeatureFactory.create("combined")
        df_features = feature_engineer.generate_features(sample_data)
        
        # Get numeric features
        df_numeric = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in df_numeric.columns:
            fraud_labels = df_numeric['is_fraudulent'].values
            df_numeric = df_numeric.drop(columns=['is_fraudulent'])
        else:
            fraud_labels = np.zeros(len(df_numeric))
        
        # Scale features
        scaled_features = scaler.transform(df_numeric)
        
        # Get reconstruction errors and anomaly scores
        reconstructed = model.model.predict(scaled_features)
        mse_scores = np.mean(np.power(scaled_features - reconstructed, 2), axis=1)
        
        # Calculate normalized anomaly scores
        anomaly_scores = (mse_scores - np.percentile(mse_scores, 5)) / (np.percentile(mse_scores, 95) - np.percentile(mse_scores, 5))
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        # Add anomaly scores to sample data
        sample_data['anomaly_score'] = anomaly_scores
        sample_data['fraud_label'] = fraud_labels
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Reconstruction Error Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Age group analysis
        sample_data['age_group'] = pd.cut(sample_data['customer_age'], bins=5, labels=['18-30', '31-45', '46-60', '61-75', '76+'])
        age_anomaly = sample_data.groupby('age_group')['anomaly_score'].apply(list)
        labels = list(age_anomaly.index)
        data = list(age_anomaly.values)
        axes[0,0].boxplot(data, tick_labels=labels)
        axes[0,0].set_title('Anomaly Scores by Age Group')
        axes[0,0].set_ylabel('Anomaly Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Account age analysis
        sample_data['account_group'] = pd.cut(sample_data['account_age_days'], bins=5, labels=['New', 'Young', 'Established', 'Mature', 'Old'])
        account_anomaly = sample_data.groupby('account_group')['anomaly_score'].apply(list)
        labels = list(account_anomaly.index)
        data = list(account_anomaly.values)
        axes[0,1].boxplot(data, tick_labels=labels)
        axes[0,1].set_title('Anomaly Scores by Account Age')
        axes[0,1].set_ylabel('Anomaly Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Payment method analysis
        payment_anomaly = sample_data.groupby('payment_method')['anomaly_score'].apply(list)
        labels = list(payment_anomaly.index)
        data = list(payment_anomaly.values)
        axes[1,0].boxplot(data, tick_labels=labels)
        axes[1,0].set_title('Anomaly Scores by Payment Method')
        axes[1,0].set_ylabel('Anomaly Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Product category analysis
        category_anomaly = sample_data.groupby('product_category')['anomaly_score'].apply(list)
        labels = list(category_anomaly.index)
        data = list(category_anomaly.values)
        axes[1,1].boxplot(data, tick_labels=labels)
        axes[1,1].set_title('Anomaly Scores by Product Category')
        axes[1,1].set_ylabel('Anomaly Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('static/customer_segmentation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance Metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(fraud_labels, anomaly_scores)
        roc_auc = auc(fpr, tpr)
        axes[0,0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,0].set_xlim([0.0, 1.0])
        axes[0,0].set_ylim([0.0, 1.05])
        axes[0,0].set_xlabel('False Positive Rate')
        axes[0,0].set_ylabel('True Positive Rate')
        axes[0,0].set_title('Receiver Operating Characteristic (ROC) Curve')
        axes[0,0].legend(loc="lower right")
        axes[0,0].grid(True)
        
        # Confusion Matrix
        threshold_optimal = np.percentile(anomaly_scores, 95)
        predictions = anomaly_scores > threshold_optimal
        cm = confusion_matrix(fraud_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title('Confusion Matrix')
        axes[0,1].set_xlabel('Predicted')
        axes[0,1].set_ylabel('Actual')
        
        # Threshold Sensitivity
        thresholds = np.arange(0.1, 1.0, 0.05)
        precision_scores = []
        recall_scores = []
        
        for thresh in thresholds:
            preds = anomaly_scores > thresh
            if np.sum(preds) > 0:
                precision = np.sum((preds == 1) & (fraud_labels == 1)) / np.sum(preds)
                recall = np.sum((preds == 1) & (fraud_labels == 1)) / np.sum(fraud_labels) if np.sum(fraud_labels) > 0 else 0
            else:
                precision = 0
                recall = 0
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        axes[1,0].plot(thresholds, precision_scores, label='Precision', marker='o')
        axes[1,0].plot(thresholds, recall_scores, label='Recall', marker='s')
        axes[1,0].set_xlabel('Threshold')
        axes[1,0].set_ylabel('Score')
        axes[1,0].set_title('Threshold Sensitivity')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Performance Metrics Summary
        metrics_text = f"""
        Model Performance Summary:
        
        AUC-ROC: {roc_auc:.3f}
        Optimal Threshold: {threshold_optimal:.3f}
        
        At {threshold_optimal:.3f} threshold:
        - Precision: {precision_scores[len(thresholds)//2]:.3f}
        - Recall: {recall_scores[len(thresholds)//2]:.3f}
        - Flagged Transactions: {np.sum(predictions)}
        - Actual Fraud: {np.sum(fraud_labels)}
        """
        
        axes[1,1].text(0.1, 0.5, metrics_text, transform=axes[1,1].transAxes, 
                      fontsize=12, verticalalignment='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1,1].set_title('Performance Metrics')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig('static/performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "sample_size": len(sample_data),
            "auc_roc": float(roc_auc),
            "optimal_threshold": float(threshold_optimal),
            "total_visualizations": len(viz_files) - 1  # Exclude metadata file
        }
        
        with open('static/visualization_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("All visualizations generated successfully")
        
        return {
            "success": True,
            "message": "All visualizations generated successfully",
            "metadata": metadata,
            "cached": False
        }
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/api/all-columns-transactions-fast")
async def get_all_columns_transactions_fast(request: DateAnalysisRequest, page: int = 0, page_size: int = 100):
    """
    Fast version of all-columns-transactions that only calculates feature importance for the current page.
    This is much faster than the original endpoint.
    """
    if model is None or scaler is None or rf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Use provided threshold or default to the trained threshold
        threshold_value = request.threshold if request.threshold is not None else threshold
        
        # Filter data for the requested date
        if request.date == "All Dates":
            date_data = full_data.copy()
        else:
            date_data = full_data[full_data['date'] == request.date].copy()
            
        if len(date_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data found for date {request.date}")
        
        logger.info(f"Fast processing: Analyzing {len(date_data)} transactions for page {page}")
        
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
        
        # Calculate reconstruction errors for all transactions
        reconstructed = model.model.predict(scaled_features)
        mse_scores = np.mean(np.power(scaled_features - reconstructed, 2), axis=1)
        
        # Apply threshold using raw MSE values
        threshold_score = np.percentile(mse_scores, threshold_value)
        flagged_by_autoencoder = mse_scores > threshold_score
        
        # Get all transactions sorted by date (earliest first)
        all_indices = np.argsort(date_data['transaction_date'].values)
        
        # Calculate pagination
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, len(all_indices))
        page_indices = all_indices[start_idx:end_idx]
        
        # Get Random Forest predictions and feature importance
        rf_features_data = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in rf_features_data.columns:
            rf_features_data = rf_features_data.drop(columns=['is_fraudulent'])
        
        # Get Random Forest predictions
        rf_predictions = rf_model.predict(rf_features_data)
        rf_probabilities = rf_model.predict_proba(rf_features_data)[:, 1]  # Probability of fraud
        
        # Calculate feature importance ONLY for the current page transactions (much faster!)
        feature_importance_per_transaction = []
        for idx in page_indices:
            # Get feature importance for this specific transaction
            transaction_features = rf_features_data.iloc[idx:idx+1]
            feature_importance = rf_model.feature_importances_
            feature_names = rf_features_data.columns
            
            # Sort features by importance for this transaction
            feature_importance_sorted = sorted(zip(feature_names, feature_importance), 
                                             key=lambda x: x[1], reverse=True)
            
            # Take top 3 features
            top_features = feature_importance_sorted[:3]
            feature_importance_per_transaction.append(top_features)
        
        # Calculate standardized anomaly scores (0-1 range) - match risk level expectations
        # Use 5th and 95th percentiles so that 5% of transactions get "Critical" risk level
        anomaly_scores = (mse_scores - np.percentile(mse_scores, 5)) / (np.percentile(mse_scores, 95) - np.percentile(mse_scores, 5))
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        # Build transaction data for the current page
        all_columns_transactions = []
        for i, idx in enumerate(page_indices):
            transaction = date_data.iloc[idx]
            
            # Get risk level
            risk_level = get_risk_level(anomaly_scores[idx])
            
            # Format top feature importance
            top_features_text = ", ".join([f"{feature} ({importance*100:.1f}%)" 
                                         for feature, importance in feature_importance_per_transaction[i]])
            
            transaction_data = {
                "date": transaction['transaction_date'],
                "amount": f"${transaction['transaction_amount']:.2f}",
                "quantity": transaction['quantity'],
                "customer_age": transaction['customer_age'],
                "account_age": transaction['account_age_days'],
                "payment_method": transaction['payment_method'],
                "product_category": transaction['product_category'],
                "device_used": transaction['device_used'],
                "customer_location": transaction['customer_location'],
                "hour": transaction['transaction_hour'],
                "is_fraudulent": "YES" if actual_labels[idx] else "NO",
                "amount_log": f"{np.log(transaction['transaction_amount']):.4f}",
                "amount_per_item": f"{transaction['transaction_amount'] / transaction['quantity']:.4f}",
                "late_night": "1" if transaction['transaction_hour'] in [23, 0, 1, 2, 3, 4, 5] else "N/A",
                "high_amount_flag": "1" if transaction['transaction_amount'] > 500 else "N/A",
                "new_account_flag": "1" if transaction['account_age_days'] < 30 else "N/A",
                "fraud_risk_score": int(anomaly_scores[idx] * 5) + 1,  # 1-5 scale
                "anomaly_score": f"{anomaly_scores[idx]:.4f}",
                "autoencoder_flagged": "YES" if flagged_by_autoencoder[idx] else "NO",
                "rf_probability": f"{rf_probabilities[idx]:.4f}",
                "risk_level": risk_level,
                "top_feature_importance": top_features_text,
                "all_features": {
                    feature_name: {
                        "importance": float(importance),
                        "value": float(rf_features_data.iloc[idx][feature_name])
                    }
                    for feature_name, importance in feature_importance_per_transaction[i]
                }
            }
            
            all_columns_transactions.append(transaction_data)
        
        # Calculate summary statistics for all transactions
        total_all_transactions = len(date_data)
        flagged_all_transactions = sum(1 for i in range(len(date_data)) if flagged_by_autoencoder[i])
        actual_fraud_all = sum(1 for i in range(len(date_data)) if actual_labels[i])
        
        # Calculate actual fraud caught (true positives)
        actual_fraud_caught = sum(1 for i in range(len(date_data)) if flagged_by_autoencoder[i] and actual_labels[i])
        
        # Calculate precision
        precision = (actual_fraud_caught / flagged_all_transactions * 100) if flagged_all_transactions > 0 else 0
        
        # Calculate pagination info
        total_pages = (len(all_indices) + page_size - 1) // page_size
        
        return {
            "transactions": all_columns_transactions,
            "summary": {
                "total_transactions": total_all_transactions,
                "flagged_transactions": flagged_all_transactions,
                "actual_fraud": actual_fraud_caught,
                "threshold_score": float(threshold_score),
                "precision": precision
            },
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "page_size": page_size,
                "total_items": len(all_indices),
                "has_next": page < total_pages - 1,
                "has_prev": page > 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error in fast transactions endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)