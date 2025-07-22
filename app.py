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
test_labels = None
original_data = None

@app.on_event("startup")
def load_everything():
    global model, scaler, test_features, test_labels, original_data
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
        test_labels = df_numeric['is_fraudulent'].values
        df_numeric = df_numeric.drop(columns=['is_fraudulent'])
    else:
        test_labels = np.zeros(len(df_numeric))
    test_features = scaler.transform(df_numeric)

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
    
    # Predict reconstruction for the first row
    x = test_features[0:1]
    reconstructed = model.predict(x)
    mse = float(np.mean(np.square(x - reconstructed)))
    
    # Combine original data with prediction results
    result = {
        "transaction_data": original_data,
        "anomaly_score": mse,
        "actual_label": int(test_labels[0]) if test_labels is not None else None
    }
    return JSONResponse(result) 