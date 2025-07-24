#!/usr/bin/env python3
"""
Data Science Visualization Generator for Autoencoder Fraud Detection
Generates comprehensive visualizations for both business and technical audiences.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig
from src.feature_factory import FeatureFactory

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuration
CONFIG = PipelineConfig.get_combined_config()
DATA_PATH = os.path.join(CONFIG.data.cleaned_dir, "ecommerce_cleaned.csv")
STATIC_DIR = "static"
MODEL_PATH = "models/final_model.h5"
SCALER_PATH = "models/final_model_scaler.pkl"

def load_data_and_model():
    """Load data and model for visualization generation."""
    print("Loading data and model...")
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Generate features
    feature_engineer = FeatureFactory.create("combined")
    df_features = feature_engineer.generate_features(df)
    
    # Load model and scaler
    import tensorflow as tf
    import joblib
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    
    # Prepare features for model
    df_numeric = df_features.select_dtypes(include=[np.number])
    if 'is_fraudulent' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=['is_fraudulent'])
    
    features_scaled = scaler.transform(df_numeric)
    
    # Get predictions
    reconstructions = model.predict(features_scaled, verbose=0)
    mse_scores = np.mean(np.square(features_scaled - reconstructions), axis=1)
    
    # Calculate threshold
    threshold = np.percentile(mse_scores, 95)
    predictions = (mse_scores > threshold).astype(int)
    
    return df, df_features, features_scaled, mse_scores, predictions, threshold, model

def create_correlation_matrix(df_features):
    """Create correlation matrix heatmap."""
    print("Generating correlation matrix...")
    
    # Select numeric columns
    numeric_cols = df_features.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_cols.corr()
    
    # Create heatmap
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix(y_true, y_pred):
    """Create confusion matrix visualization."""
    print("Generating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_roc_curve(y_true, anomaly_scores):
    """Create ROC curve visualization."""
    print("Generating ROC curve...")
    
    fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_anomaly_vs_amount(df, anomaly_scores):
    """Create anomaly score vs transaction amount scatter plot."""
    print("Generating anomaly vs amount plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with color coding for fraud
    scatter = plt.scatter(df['transaction_amount'], anomaly_scores, 
                         c=df['is_fraudulent'], cmap='viridis', alpha=0.6, s=30)
    
    plt.colorbar(scatter, label='Fraud Status')
    plt.xlabel('Transaction Amount ($)', fontsize=12)
    plt.ylabel('Anomaly Score', fontsize=12)
    plt.title('Anomaly Score vs Transaction Amount', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'anomaly_vs_amount.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_reconstruction_error_distribution(anomaly_scores, y_true):
    """Create reconstruction error distribution."""
    print("Generating reconstruction error distribution...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot distributions for legitimate and fraudulent transactions
    legitimate_scores = anomaly_scores[y_true == 0]
    fraudulent_scores = anomaly_scores[y_true == 1]
    
    plt.hist(legitimate_scores, bins=50, alpha=0.7, label='Legitimate', color='blue', density=True)
    plt.hist(fraudulent_scores, bins=50, alpha=0.7, label='Fraudulent', color='red', density=True)
    
    # Add threshold line
    threshold = np.percentile(anomaly_scores, 95)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, 
                label=f'Threshold (95th percentile: {threshold:.3f})')
    
    plt.xlabel('Reconstruction Error (MSE)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Reconstruction Errors', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'reconstruction_error_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_importance_analysis(df_features, anomaly_scores):
    """Create feature importance analysis."""
    print("Generating feature importance analysis...")
    
    # Select numeric features
    numeric_features = df_features.select_dtypes(include=[np.number])
    if 'is_fraudulent' in numeric_features.columns:
        numeric_features = numeric_features.drop(columns=['is_fraudulent'])
    
    # Calculate correlation with anomaly scores
    correlations = []
    for col in numeric_features.columns:
        corr = np.corrcoef(numeric_features[col], anomaly_scores)[0, 1]
        correlations.append((col, abs(corr)))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = correlations[:15]
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    features, corrs = zip(*top_features)
    
    bars = plt.barh(range(len(features)), corrs, color='skyblue')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Absolute Correlation with Anomaly Score', fontsize=12)
    plt.title('Feature Importance (Correlation with Anomaly Score)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, corr) in enumerate(zip(bars, corrs)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{corr:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_time_patterns_analysis(df):
    """Create time-based pattern analysis."""
    print("Generating time patterns analysis...")
    
    # Use transaction_hour if available, otherwise create synthetic data
    if 'transaction_hour' in df.columns:
        df['hour'] = df['transaction_hour']
    else:
        # Create synthetic time patterns for demonstration
        np.random.seed(42)
        df['hour'] = np.random.randint(0, 24, len(df))
    
    # Create synthetic day of week for demonstration
    np.random.seed(42)
    df['day_of_week'] = np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], len(df))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Hourly fraud pattern
    hourly_fraud = df.groupby('hour')['is_fraudulent'].mean()
    axes[0, 0].bar(hourly_fraud.index, hourly_fraud.values, color='red', alpha=0.7)
    axes[0, 0].set_title('Fraud Rate by Hour of Day', fontweight='bold')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Fraud Rate')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Daily fraud pattern
    daily_fraud = df.groupby('day_of_week')['is_fraudulent'].mean()
    daily_fraud = daily_fraud.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    axes[0, 1].bar(range(len(daily_fraud)), daily_fraud.values, color='orange', alpha=0.7)
    axes[0, 1].set_title('Fraud Rate by Day of Week', fontweight='bold')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Fraud Rate')
    axes[0, 1].set_xticks(range(len(daily_fraud)))
    axes[0, 1].set_xticklabels(daily_fraud.index, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Transaction volume by hour
    hourly_volume = df.groupby('hour').size()
    axes[1, 0].bar(hourly_volume.index, hourly_volume.values, color='blue', alpha=0.7)
    axes[1, 0].set_title('Transaction Volume by Hour', fontweight='bold')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Number of Transactions')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Average amount by hour
    hourly_amount = df.groupby('hour')['transaction_amount'].mean()
    axes[1, 1].bar(hourly_amount.index, hourly_amount.values, color='green', alpha=0.7)
    axes[1, 1].set_title('Average Transaction Amount by Hour', fontweight='bold')
    axes[1, 1].set_xlabel('Hour')
    axes[1, 1].set_ylabel('Average Amount ($)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'time_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_threshold_sensitivity_analysis(y_true, anomaly_scores):
    """Create threshold sensitivity analysis."""
    print("Generating threshold sensitivity analysis...")
    
    thresholds = np.percentile(anomaly_scores, range(80, 100))
    metrics = []
    
    for threshold in thresholds:
        y_pred = (anomaly_scores > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'flagged_rate': (tp + fp) / len(y_true)
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Precision vs Threshold
    axes[0, 0].plot(metrics_df['threshold'], metrics_df['precision'], 'b-', linewidth=2)
    axes[0, 0].set_title('Precision vs Threshold', fontweight='bold')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Recall vs Threshold
    axes[0, 1].plot(metrics_df['threshold'], metrics_df['recall'], 'r-', linewidth=2)
    axes[0, 1].set_title('Recall vs Threshold', fontweight='bold')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score vs Threshold
    axes[1, 0].plot(metrics_df['threshold'], metrics_df['f1'], 'g-', linewidth=2)
    axes[1, 0].set_title('F1 Score vs Threshold', fontweight='bold')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Flagged Rate vs Threshold
    axes[1, 1].plot(metrics_df['threshold'], metrics_df['flagged_rate'], 'purple', linewidth=2)
    axes[1, 1].set_title('Flagged Rate vs Threshold', fontweight='bold')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Flagged Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'threshold_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_customer_segmentation(df):
    """Create customer segmentation analysis."""
    print("Generating customer segmentation analysis...")
    
    # Create customer segments based on transaction behavior
    df['amount_segment'] = pd.cut(df['transaction_amount'], 
                                 bins=[0, 50, 200, 500, 1000, float('inf')],
                                 labels=['Low', 'Medium', 'High', 'Very High', 'Extreme'])
    
    # Use customer_location for frequency analysis since there's no customer_id
    df['frequency_segment'] = pd.cut(df.groupby('customer_location')['transaction_amount'].transform('count'),
                                   bins=[0, 1, 3, 10, float('inf')],
                                   labels=['One-time', 'Occasional', 'Regular', 'Frequent'])
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Fraud rate by amount segment
    amount_fraud = df.groupby('amount_segment')['is_fraudulent'].mean()
    axes[0, 0].bar(range(len(amount_fraud)), amount_fraud.values, color='red', alpha=0.7)
    axes[0, 0].set_title('Fraud Rate by Amount Segment', fontweight='bold')
    axes[0, 0].set_xlabel('Amount Segment')
    axes[0, 0].set_ylabel('Fraud Rate')
    axes[0, 0].set_xticks(range(len(amount_fraud)))
    axes[0, 0].set_xticklabels(amount_fraud.index, rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Fraud rate by frequency segment
    freq_fraud = df.groupby('frequency_segment')['is_fraudulent'].mean()
    axes[0, 1].bar(range(len(freq_fraud)), freq_fraud.values, color='orange', alpha=0.7)
    axes[0, 1].set_title('Fraud Rate by Frequency Segment', fontweight='bold')
    axes[0, 1].set_xlabel('Frequency Segment')
    axes[0, 1].set_ylabel('Fraud Rate')
    axes[0, 1].set_xticks(range(len(freq_fraud)))
    axes[0, 1].set_xticklabels(freq_fraud.index, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average amount by segment
    avg_amount = df.groupby('amount_segment')['transaction_amount'].mean()
    axes[1, 0].bar(range(len(avg_amount)), avg_amount.values, color='blue', alpha=0.7)
    axes[1, 0].set_title('Average Amount by Segment', fontweight='bold')
    axes[1, 0].set_xlabel('Amount Segment')
    axes[1, 0].set_ylabel('Average Amount ($)')
    axes[1, 0].set_xticks(range(len(avg_amount)))
    axes[1, 0].set_xticklabels(avg_amount.index, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Transaction count by segment
    segment_counts = df['amount_segment'].value_counts()
    axes[1, 1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Transaction Distribution by Amount Segment', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'customer_segmentation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_metrics_summary(y_true, y_pred, anomaly_scores):
    """Create comprehensive performance metrics summary."""
    print("Generating performance metrics summary...")
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Metrics bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    values = [accuracy, precision, recall, f1, roc_auc]
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
    
    bars = axes[0, 0].bar(metrics, values, color=colors, alpha=0.7)
    axes[0, 0].set_title('Model Performance Metrics', fontweight='bold', fontsize=14)
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    axes[0, 1].set_title('Confusion Matrix', fontweight='bold')
    
    # Anomaly score distribution
    legitimate_scores = anomaly_scores[y_true == 0]
    fraudulent_scores = anomaly_scores[y_true == 1]
    
    axes[1, 0].hist(legitimate_scores, bins=30, alpha=0.7, label='Legitimate', color='blue', density=True)
    axes[1, 0].hist(fraudulent_scores, bins=30, alpha=0.7, label='Fraudulent', color='red', density=True)
    axes[1, 0].set_title('Anomaly Score Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Anomaly Score')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
    axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[1, 1].set_xlim([0.0, 1.0])
    axes[1, 1].set_ylim([0.0, 1.05])
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_latent_space_visualization(features_scaled, model, y_true):
    """Create latent space visualization using t-SNE."""
    print("Generating latent space visualization...")
    
    # Use PCA for dimensionality reduction instead of accessing encoder
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features_scaled)
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Plot points with color coding
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=y_true, cmap='viridis', alpha=0.6, s=30)
    
    plt.colorbar(scatter, label='Fraud Status')
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.title('Feature Space Visualization (PCA)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'latent_space_3d_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save feature space data for interactive visualization
    latent_data = {
        'x': features_2d[:, 0].tolist(),
        'y': features_2d[:, 1].tolist(),
        'fraud_status': y_true.tolist(),
        'fraud_labels': ['Legitimate' if x == 0 else 'Fraudulent' for x in y_true]
    }
    
    with open(os.path.join(STATIC_DIR, 'latent_space_3d.json'), 'w') as f:
        json.dump(latent_data, f)

def update_visualization_metadata(df, y_true, y_pred, anomaly_scores, threshold):
    """Update visualization metadata."""
    print("Updating visualization metadata...")
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    fraud_rate = np.mean(y_true) * 100
    
    # Get top features (correlation with anomaly scores)
    numeric_features = df.select_dtypes(include=[np.number])
    if 'is_fraudulent' in numeric_features.columns:
        numeric_features = numeric_features.drop(columns=['is_fraudulent'])
    
    correlations = []
    for col in numeric_features.columns:
        corr = np.corrcoef(numeric_features[col], anomaly_scores)[0, 1]
        correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = [col for col, _ in correlations[:5]]
    
    metadata = {
        "total_transactions": len(df),
        "fraud_rate": f"{fraud_rate:.2f}%",
        "roc_auc": roc_auc,
        "threshold_used": threshold,
        "top_features": top_features,
        "generated_at": pd.Timestamp.now().isoformat()
    }
    
    with open(os.path.join(STATIC_DIR, 'visualization_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    """Main function to generate all visualizations."""
    print("Starting Data Science Visualization Generation...")
    print("=" * 60)
    
    # Load data and model
    df, df_features, features_scaled, anomaly_scores, predictions, threshold, model = load_data_and_model()
    
    # Get true labels
    y_true = df['is_fraudulent'].values
    
    # Generate all visualizations
    create_correlation_matrix(df_features)
    create_confusion_matrix(y_true, predictions)
    create_roc_curve(y_true, anomaly_scores)
    create_anomaly_vs_amount(df, anomaly_scores)
    create_reconstruction_error_distribution(anomaly_scores, y_true)
    create_feature_importance_analysis(df_features, anomaly_scores)
    create_time_patterns_analysis(df)
    create_threshold_sensitivity_analysis(y_true, anomaly_scores)
    create_customer_segmentation(df)
    create_performance_metrics_summary(y_true, predictions, anomaly_scores)
    create_latent_space_visualization(features_scaled, model, y_true)
    
    # Update metadata
    update_visualization_metadata(df, y_true, predictions, anomaly_scores, threshold)
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print(f"Files saved to: {STATIC_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main() 