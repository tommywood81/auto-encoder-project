import os
import sys
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import json
import joblib
import tensorflow as tf
import yaml
import logging
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

STATIC_DIR = "static"
SCALER_PATH = "models/final_model_scaler.pkl"
MODEL_PATH = "models/final_model.h5"
MODEL_INFO_PATH = "models/final_model_info.yaml"
DATA_PATH = "data/cleaned/ecommerce_cleaned.csv"


def get_file_size(path):
    return os.path.getsize(path) if os.path.exists(path) else -1

def save_if_changed(fig, path):
    try:
        # Ensure static directory exists
        os.makedirs(STATIC_DIR, exist_ok=True)
        
        tmp_path = path + ".tmp"
        fig.savefig(tmp_path, dpi=300, bbox_inches='tight')
        fig.clf()
        fig.clear()
        plt.close(fig)
        old_size = get_file_size(path)
        new_size = get_file_size(tmp_path)
        if old_size != new_size:
            os.replace(tmp_path, path)
            return True
        else:
            os.remove(tmp_path)
            return False
    except Exception as e:
        logger.error(f"Error saving figure to {path}: {e}")
        return False

def generate_autoencoder_graphs():
    """Generate autoencoder-focused graphs using the output of run_pipeline.py with the 'combined' strategy."""
    try:
        logger.info("Running run_pipeline.py with --strategy combined ...")
        result = subprocess.run([
            sys.executable, os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'run_pipeline.py'),
            '--strategy', 'combined'
        ], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"run_pipeline.py failed: {result.stderr}")
            raise RuntimeError(f"run_pipeline.py failed: {result.stderr}")
        logger.info(result.stdout)
        # After pipeline, proceed as before
        # Check if required files exist
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not os.path.exists(MODEL_INFO_PATH):
            raise FileNotFoundError(f"Model info file not found: {MODEL_INFO_PATH}")
        # Load data and model directly (same as app.py)
        logger.info("Loading data...")
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Loaded {len(df)} transactions")
        # Load scaler
        logger.info("Loading scaler...")
        scaler = joblib.load(SCALER_PATH)
        # Load model
        logger.info("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        # Load model metrics
        logger.info("Loading model metrics...")
        with open(MODEL_INFO_PATH, 'r') as file:
            model_metrics = yaml.safe_load(file)
        # Generate features using the same strategy as the app
        logger.info("Generating features...")
        from src.feature_factory import FeatureFactory
        feature_engineer = FeatureFactory.create("combined")
        df_features = feature_engineer.generate_features(df)
        logger.info(f"Generated {len(df_features.columns)} features")
        # Prepare features for model
        df_numeric = df_features.select_dtypes(include=[np.number])
        if 'is_fraudulent' in df_numeric.columns:
            df_numeric = df_numeric.drop(columns=['is_fraudulent'])
        features_scaled = scaler.transform(df_numeric)
        # Get autoencoder predictions and reconstructions
        logger.info("Running autoencoder predictions...")
        reconstructions = model.predict(features_scaled, verbose=0)
        mse_scores = np.mean(np.square(features_scaled - reconstructions), axis=1)
        threshold = np.percentile(mse_scores, 95)
        predictions = (mse_scores > threshold).astype(int)
        y_true = df['is_fraudulent'].values
        logger.info("Generating autoencoder-focused graphs...")
        # 1. ROC Curve
        try:
            fig = plt.figure(figsize=(10, 8))
            fpr, tpr, _ = roc_curve(y_true, mse_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('Autoencoder ROC Curve', fontsize=16, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(STATIC_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info("Generated ROC curve")
        except Exception as e:
            logger.error(f"Error generating ROC curve: {e}")
        # 2. Confusion Matrix
        try:
            fig = plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
            plt.title('Autoencoder Confusion Matrix', fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            fig.savefig(os.path.join(STATIC_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info("Generated confusion matrix")
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {e}")
        # 3. Latent Space Visualization (using encoder part)
        try:
            encoder = tf.keras.Model(inputs=model.input, outputs=model.get_layer('latent').output)
            latent_representations = encoder.predict(features_scaled, verbose=0)
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_representations)
            fig = plt.figure(figsize=(12, 8))
            plt.scatter(latent_2d[y_true == 0, 0], latent_2d[y_true == 0, 1], alpha=0.6, label='Legitimate', color='blue', s=20)
            plt.scatter(latent_2d[y_true == 1, 0], latent_2d[y_true == 1, 1], alpha=0.8, label='Fraudulent', color='red', s=30)
            plt.xlabel('First Principal Component', fontsize=12)
            plt.ylabel('Second Principal Component', fontsize=12)
            plt.title('Autoencoder Latent Space Visualization', fontsize=16, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(STATIC_DIR, 'latent_space.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info("Generated latent space visualization")
        except Exception as e:
            logger.error(f"Could not generate latent space visualization: {e}")
        # 4. Training History (if available)
        try:
            if 'training_history' in model_metrics:
                history = model_metrics['training_history']
                fig = plt.figure(figsize=(12, 8))
                plt.plot(history['loss'], label='Training Loss', color='blue')
                if 'val_loss' in history:
                    plt.plot(history['val_loss'], label='Validation Loss', color='red')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.title('Autoencoder Training History', fontsize=16, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fig.savefig(os.path.join(STATIC_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info("Generated training history plot")
        except Exception as e:
            logger.error(f"Could not generate training history: {e}")
        # 5. Autoencoder Model Architecture
        try:
            from tensorflow.keras.utils import plot_model
            plot_path = os.path.join(STATIC_DIR, 'autoencoder_architecture.png')
            plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True, dpi=120)
            logger.info("Generated autoencoder model architecture plot")
        except Exception as e:
            logger.error(f"Could not generate model architecture plot: {e}")
        logger.info(f"Graph generation completed.")
        return True
    except Exception as e:
        logger.error(f"Error in generate_autoencoder_graphs: {e}")
        raise

def generate_all_graphs():
    """Legacy function for backward compatibility - now calls autoencoder-focused graphs"""
    return generate_autoencoder_graphs()

if __name__ == "__main__":
    try:
        generate_autoencoder_graphs()
        print("Graph generation completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 