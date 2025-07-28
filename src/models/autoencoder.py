"""
Simplified Autoencoder for Fraud Detection
Production-grade implementation with proper train/test separation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import logging
from typing import Dict, Any, Tuple, Optional
import pickle
import os

logger = logging.getLogger(__name__)


class FraudAutoencoder:
    """Production-grade autoencoder for fraud detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize autoencoder with configuration."""
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_fitted = False
        
    def build_model(self, input_dim: int) -> keras.Model:
        """Build autoencoder architecture."""
        
        # Get architecture parameters
        latent_dim = self.config.get('latent_dim', 16)
        hidden_dims = self.config.get('hidden_dims', [64, 32])
        activation = self.config.get('activation', 'relu')
        dropout_rate = self.config.get('dropout_rate', 0.1)
        
        # Encoder
        encoder_input = keras.Input(shape=(input_dim,))
        x = encoder_input
        
        for dim in hidden_dims:
            x = layers.Dense(dim, activation=activation)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.BatchNormalization()(x)
        
        # Latent space
        latent = layers.Dense(latent_dim, activation=activation, name='latent')(x)
        
        # Decoder
        for dim in reversed(hidden_dims):
            x = layers.Dense(dim, activation=activation)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.BatchNormalization()(x)
        
        # Output
        decoder_output = layers.Dense(input_dim, activation='linear')(x)
        
        # Create model
        model = keras.Model(encoder_input, decoder_output, name='fraud_autoencoder')
        
        return model
    
    def prepare_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training with proper scaling."""
        
        # Get numeric features only
        df_train_numeric = df_train.select_dtypes(include=[np.number])
        df_test_numeric = df_test.select_dtypes(include=[np.number])
        
        # Remove target variable if present
        if 'is_fraudulent' in df_train_numeric.columns:
            df_train_numeric = df_train_numeric.drop(columns=['is_fraudulent'])
        if 'is_fraudulent' in df_test_numeric.columns:
            df_test_numeric = df_test_numeric.drop(columns=['is_fraudulent'])
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(df_train_numeric.values)
        X_test_scaled = self.scaler.transform(df_test_numeric.values)
        
        logger.info(f"Data prepared: train={X_train_scaled.shape}, test={X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled
    
    def train(self, X_train: np.ndarray, X_test: np.ndarray, 
              y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train the autoencoder."""
        
        logger.info("Training autoencoder...")
        
        # Build model
        input_dim = X_train.shape[1]
        self.model = self.build_model(input_dim)
        
        # Compile model
        learning_rate = self.config.get('learning_rate', 0.001)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = []
        
        # Early stopping
        if self.config.get('early_stopping', True):
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('patience', 10),
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Reduce learning rate
        if self.config.get('reduce_lr', True):
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
            callbacks.append(reduce_lr)
        
        # Custom callback for AUC monitoring
        auc_callback = AUCCallback(X_test, y_test, self.config.get('threshold_percentile', 95))
        callbacks.append(auc_callback)
        
        # Train model
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        validation_split = self.config.get('validation_split', 0.2)
        
        history = self.model.fit(
            X_train,
            X_train,  # Autoencoder learns to reconstruct input
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate threshold using training data
        self._calculate_threshold(X_train, y_train)
        
        # Evaluate on test set
        test_auc = self._evaluate(X_test, y_test)
        
        self.is_fitted = True
        
        results = {
            'test_auc': test_auc,
            'threshold': self.threshold,
            'history': history.history
        }
        
        logger.info(f"Training completed. Test AUC: {test_auc:.4f}")
        
        return results
    
    def _calculate_threshold(self, X_train: np.ndarray, y_train: np.ndarray):
        """Calculate threshold using normal training data."""
        
        # Get only normal (non-fraudulent) data
        normal_mask = y_train == 0
        X_normal = X_train[normal_mask]
        
        # Calculate reconstruction error for normal data
        X_reconstructed = self.model.predict(X_normal)
        reconstruction_errors = np.mean(np.square(X_normal - X_reconstructed), axis=1)
        
        # Calculate threshold as percentile
        threshold_percentile = self.config.get('threshold_percentile', 95)
        self.threshold = np.percentile(reconstruction_errors, threshold_percentile)
        
        logger.info(f"Threshold calculated: {self.threshold:.6f} (percentile {threshold_percentile})")
        
        # Set fitted flag
        self.is_fitted = True
    
    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale input
        X_scaled = self.scaler.transform(X)
        
        # Get reconstruction
        X_reconstructed = self.model.predict(X_scaled)
        
        # Calculate reconstruction error
        anomaly_scores = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
        
        return anomaly_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        anomaly_scores = self.predict_anomaly_scores(X)
        predictions = (anomaly_scores > self.threshold).astype(int)
        
        return predictions
    
    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate model performance."""
        anomaly_scores = self.predict_anomaly_scores(X_test)
        auc = roc_auc_score(y_test, anomaly_scores)
        return auc
    
    def save_model(self, filepath: str):
        """Save model and scaler."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Save model
        model_path = f"{filepath}_model.h5"
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = f"{filepath}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save threshold
        threshold_path = f"{filepath}_threshold.pkl"
        with open(threshold_path, 'wb') as f:
            pickle.dump(self.threshold, f)
        
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and scaler."""
        # Load model
        model_path = f"{filepath}_model.h5"
        self.model = keras.models.load_model(model_path)
        
        # Load scaler
        scaler_path = f"{filepath}_scaler.pkl"
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load threshold
        threshold_path = f"{filepath}_threshold.pkl"
        with open(threshold_path, 'rb') as f:
            self.threshold = pickle.load(f)
        
        self.is_fitted = True
        logger.info(f"Model loaded from: {filepath}")


class AUCCallback(keras.callbacks.Callback):
    """Custom callback to monitor AUC during training."""
    
    def __init__(self, X_val, y_val, threshold_percentile=95):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.threshold_percentile = threshold_percentile
        self.best_auc = 0
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        # Calculate anomaly scores
        X_reconstructed = self.model.predict(self.X_val)
        reconstruction_errors = np.mean(np.square(self.X_val - X_reconstructed), axis=1)
        
        # Calculate threshold
        threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        
        # Calculate AUC
        auc = roc_auc_score(self.y_val, reconstruction_errors)
        
        # Log AUC
        logs['val_auc'] = auc
        
        # Track best AUC
        if auc > self.best_auc:
            self.best_auc = auc
            self.best_epoch = epoch
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: AUC = {auc:.4f}, Threshold = {threshold:.6f}")
    
    def on_train_end(self, logs=None):
        logger.info(f"Best AUC: {self.best_auc:.4f} at epoch {self.best_epoch}") 