import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.config import PipelineConfig
from src.feature_factory import FeatureFactory
import joblib
import tensorflow as tf
import yaml

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load config, model, scaler, and test data on startup
CONFIG = PipelineConfig.get_combined_config()
MODEL_PATH = "models/final_model.h5"
SCALER_PATH = "models/final_model_scaler.pkl"
MODEL_INFO_PATH = "models/final_model_info.yaml"
DATA_PATH = os.path.join(CONFIG.data.cleaned_dir, "ecommerce_cleaned.csv")

model = None
scaler = None
test_features = None
original_data = None
model_metrics = None
threshold = None
all_predictions = None

@app.on_event("startup")
def load_everything():
    global model, scaler, test_features, original_data, model_metrics, threshold, all_predictions
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    # Load model metrics
    with open(MODEL_INFO_PATH, 'r') as file:
        model_metrics = yaml.safe_load(file)
    # Load and process test data
    df = pd.read_csv(DATA_PATH)
    original_data = df.to_dict('records')  # Store all rows
    feature_engineer = FeatureFactory.create("combined")
    df_features = feature_engineer.generate_features(df)
    df_numeric = df_features.select_dtypes(include=[np.number])
    if 'is_fraudulent' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=['is_fraudulent'])
    test_features = scaler.transform(df_numeric)
    
    # Calculate threshold once using the same method as during training
    # The 95th percentile threshold is already baked into the model
    threshold_percentile = model_metrics.get('threshold_percentile', 95)
    all_reconstructions = model.predict(test_features, verbose=0)
    all_mse = np.mean(np.square(test_features - all_reconstructions), axis=1)
    threshold = np.percentile(all_mse, threshold_percentile)
    
    # Pre-calculate all predictions
    all_predictions = all_mse > threshold
    
    print(f"Model loaded successfully with {len(test_features)} transactions available for prediction")
    print(f"Threshold set at: {threshold:.4f} ({threshold_percentile}th percentile)")
    print(f"Total transactions flagged as fraud: {np.sum(all_predictions)}")

@app.get("/", response_class=HTMLResponse)
def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_test_set():
    if model is None or scaler is None or test_features is None or original_data is None or model_metrics is None or threshold is None or all_predictions is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    try:
        # Get all reconstruction errors (anomaly scores)
        all_reconstructions = model.predict(test_features, verbose=0)
        all_mse_scores = np.mean(np.square(test_features - all_reconstructions), axis=1)
        
        # Add fraud prediction column to each transaction
        transactions_with_predictions = []
        for i, transaction in enumerate(original_data):
            transaction_copy = transaction.copy()
            transaction_copy['AE_Predicts_Fraud'] = bool(all_predictions[i])
            transaction_copy['Anomaly_Score'] = float(all_mse_scores[i])
            transactions_with_predictions.append(transaction_copy)
        
        # Calculate dataset statistics
        total_transactions = len(original_data)
        flagged_as_fraud = np.sum(all_predictions)
        fraud_percentage = (flagged_as_fraud / total_transactions) * 100
        
        # Add prediction results to the response
        result = {
            "transaction_data": transactions_with_predictions,
            "model_performance": {
                "roc_auc": float(model_metrics.get('roc_auc', 0)),
                "accuracy": float(model_metrics.get('accuracy', 0)),
                "precision": float(model_metrics.get('precision', 0)),
                "recall": float(model_metrics.get('recall', 0)),
                "f1_score": float(model_metrics.get('f1_score', 0)),
                "training_date": model_metrics.get('training_date', 'Unknown'),
                "model_type": model_metrics.get('model_type', 'Unknown'),
                "feature_strategy": model_metrics.get('feature_strategy', 'Unknown'),
                "latent_dim": int(model_metrics.get('latent_dim', 0)),
                "threshold_percentile": int(model_metrics.get('threshold_percentile', 0)),
                "total_transactions": total_transactions,
                "flagged_as_fraud": int(flagged_as_fraud),
                "fraud_percentage": float(fraud_percentage),
                "model_threshold": float(threshold)
            }
        }
        
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") 