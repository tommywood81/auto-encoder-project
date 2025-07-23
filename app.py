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

@app.on_event("startup")
def load_everything():
    global model, scaler, test_features, original_data, model_metrics, threshold
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    # Load model metrics
    with open(MODEL_INFO_PATH, 'r') as file:
        model_metrics = yaml.safe_load(file)
    # Load and process test data
    df = pd.read_csv(DATA_PATH)
    original_data = df.iloc[0:1].to_dict('records')[0]  # Store first row
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
    
    print(f"Model loaded successfully with {len(test_features)} transactions available for prediction")
    print(f"Threshold set at: {threshold:.4f} ({threshold_percentile}th percentile)")

@app.get("/", response_class=HTMLResponse)
def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_first_row():
    if model is None or scaler is None or test_features is None or original_data is None or model_metrics is None or threshold is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    try:
        # Make prediction using the loaded model
        # Get reconstruction error (anomaly score) for the first transaction
        reconstructions = model.predict(test_features[0:1], verbose=0)
        mse_score = np.mean(np.square(test_features[0:1] - reconstructions), axis=1)[0]
        
        # Determine if it's flagged as fraud based on the model's threshold
        # The 95th percentile threshold is already baked into the model during training
        is_flagged = mse_score > threshold
        
        # Add prediction results to the response
        result = {
            "transaction_data": original_data,
            "prediction": {
                "anomaly_score": float(mse_score),
                "threshold": float(threshold),
                "is_flagged": bool(is_flagged),
                "fraud_probability": float(min(mse_score / threshold, 2.0))  # Normalized score
            },
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
                "threshold_percentile": int(model_metrics.get('threshold_percentile', 0))
            }
        }
        
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") 