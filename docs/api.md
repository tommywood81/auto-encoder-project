# API Documentation

## FastAPI Implementation

The fraud detection dashboard is powered by a FastAPI application that provides real-time analysis capabilities.

```python
# app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

app = FastAPI(title="Fraud Detection Dashboard")

class FraudDetector:
    def __init__(self, config: Dict):
        self.model = load_model(config['model_path'])
        self.scaler = load_scaler(config['scaler_path'])
        self.feature_engineer = FraudFeatureEngineer(config)
        self.threshold_cache = {}
        
    async def predict_batch(
        self, 
        transactions: pd.DataFrame, 
        percentile: float = 99.0
    ) -> Dict:
        """Process transactions and return anomaly scores."""
        try:
            # Engineer features
            features = self.feature_engineer.engineer_features(transactions)
            
            # Scale features
            X_scaled = self.scaler.transform(features)
            
            # Get anomaly scores
            anomaly_scores = self.model.predict_anomaly_scores(X_scaled)
            
            # Calculate threshold
            threshold = np.percentile(anomaly_scores, percentile)
            
            return {
                'scores': anomaly_scores,
                'is_fraud': anomaly_scores > threshold,
                'threshold': threshold
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Model inference failed"
            )
```

## API Endpoints

### POST /api/analyze
Analyzes transactions for potential fraud.

```python
@app.post("/api/analyze")
async def analyze_transactions(
    background_tasks: BackgroundTasks,
    percentile: float = 99.0
) -> Dict:
    """Analyze transactions for potential fraud."""
    try:
        # Load test transactions
        transactions = load_transactions(config['test_data_path'])
        
        # Process in background for large datasets
        if len(transactions) > 10000:
            background_tasks.add_task(
                detector.predict_batch, 
                transactions, 
                percentile
            )
            return {"status": "processing"}
        
        # Process immediately for smaller datasets
        results = await detector.predict_batch(transactions, percentile)
        
        return {
            "total_transactions": len(transactions),
            "fraud_detected": int(results['is_fraud'].sum()),
            "threshold_value": float(results['threshold']),
            "percentile": percentile,
            "results": format_results(transactions, results)
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### GET /api/metrics
Returns model performance metrics.

```python
@app.get("/api/metrics")
async def get_performance_metrics() -> Dict:
    """Get model performance metrics."""
    try:
        metrics = calculate_metrics(
            y_true=test_data['fraud_label'],
            y_pred=detector.latest_predictions['is_fraud'],
            anomaly_scores=detector.latest_predictions['scores']
        )
        
        return {
            "auc_roc": metrics['auc_roc'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1_score": metrics['f1'],
            "confusion_matrix": metrics['confusion_matrix'].tolist()
        }
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Could not calculate metrics"
        )
```

### GET /api/health
Health check endpoint for monitoring.

```python
@app.get("/api/health")
async def health_check() -> Dict:
    """Health check endpoint."""
    try:
        # Verify model loaded
        test_input = np.random.random((1, detector.model.input_dim))
        detector.model.predict(test_input)
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "feature_engineer_loaded": True,
            "memory_usage": get_memory_usage()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

## Configuration

```yaml
# configs/inference_config.yaml
inference:
  model_path: models/fraud_autoencoder.keras
  scaler_path: models/fraud_scaler.pkl
  engineered_test_data_path: data/engineered/test_features_90_10.csv
  test_data_sample_size: 1.0
  cache_anomaly_scores: true
```
