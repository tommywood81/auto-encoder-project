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

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load config, model, scaler, and test data on startup
CONFIG = PipelineConfig.get_combined_config()
MODEL_PATH = "models/final_model.h5"
SCALER_PATH = "models/final_model_scaler.pkl"
DATA_PATH = os.path.join(CONFIG.data.cleaned_dir, "ecommerce_cleaned.csv")

model = None
scaler = None
test_features = None
original_data = None
threshold = None

@app.on_event("startup")
def load_everything():
    global model, scaler, test_features, original_data, threshold
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    # Load and process test data
    df = pd.read_csv(DATA_PATH)
    original_data = df.iloc[0:1].to_dict('records')[0]  # Store first row
    feature_engineer = FeatureFactory.create("combined")
    df_features = feature_engineer.generate_features(df)
    df_numeric = df_features.select_dtypes(include=[np.number])
    if 'is_fraudulent' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=['is_fraudulent'])
    test_features = scaler.transform(df_numeric)
    
    # Calculate threshold from training data (95th percentile)
    # For demo purposes, we'll use a reasonable threshold
    threshold = np.percentile(test_features.flatten(), 95)
    print(f"Model loaded successfully. Threshold set at: {threshold:.4f}")

@app.get("/", response_class=HTMLResponse)
def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_first_row():
    if model is None or scaler is None or test_features is None or original_data is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    try:
        # Make prediction using the loaded model
        # Get reconstruction error (anomaly score)
        reconstructions = model.predict(test_features, verbose=0)
        mse_scores = np.mean(np.square(test_features - reconstructions), axis=1)
        anomaly_score = mse_scores[0]  # First row
        
        # Determine if it's flagged as fraud based on threshold
        is_flagged = anomaly_score > threshold
        
        # Add prediction results to the response
        result = {
            "transaction_data": original_data,
            "prediction": {
                "anomaly_score": float(anomaly_score),
                "threshold": float(threshold),
                "is_flagged": bool(is_flagged),
                "fraud_probability": float(min(anomaly_score / threshold, 2.0))  # Normalized score
            }
        }
        
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") 