import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from src.config import PERCENTILE_THRESHOLD


def evaluate_autoencoder(model, X_train_ae, X_test, y_test, percentile=None):
    """
    Evaluate the autoencoder for fraud detection using reconstruction error.
    
    Args:
        model (Autoencoder): Trained autoencoder model
        X_train_ae (np.ndarray): Training data (non-fraudulent)
        X_test (np.ndarray): Test data
        y_test (np.ndarray): True labels for test data
        percentile (int, optional): Percentile for threshold calculation. Defaults to config value.
    """
    if percentile is None:
        percentile = PERCENTILE_THRESHOLD
        
    print("🔎 Detecting anomalies...")
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Get reconstructions for test data
        recon_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        
        # Get reconstructions for training data (non-fraudulent)
        recon_train = model(torch.tensor(X_train_ae, dtype=torch.float32)).numpy()

    # Calculate reconstruction errors
    test_errors = ((X_test - recon_test) ** 2).mean(axis=1)
    train_errors = ((X_train_ae - recon_train) ** 2).mean(axis=1)
    
    # Calculate threshold based on training data errors
    threshold = np.percentile(train_errors, percentile)
    
    print(f"📏 Threshold ({percentile}th percentile): {threshold:.4f}")
    print(f"📊 Mean reconstruction error (normal): {train_errors.mean():.4f}")
    print(f"📊 Mean reconstruction error (test): {test_errors.mean():.4f}")
    
    # Make predictions
    y_pred = (test_errors > threshold).astype(int)
    
    # Calculate metrics
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    print(f"📈 ROC-AUC Score: {roc_auc_score(y_test, test_errors):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"True Negatives: {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives: {cm[1, 1]}")
    
    # Additional statistics
    fraud_detected = np.sum(y_pred == 1)
    total_fraud = np.sum(y_test == 1)
    total_normal = np.sum(y_test == 0)
    
    print(f"\n📋 Summary:")
    print(f"Total transactions: {len(y_test)}")
    print(f"Fraudulent transactions: {total_fraud}")
    print(f"Normal transactions: {total_normal}")
    print(f"Anomalies detected: {fraud_detected}")
    print(f"Detection rate: {fraud_detected/len(y_test)*100:.2f}%") 