"""
Production-grade autoencoder for fraud detection.
Config-driven implementation with clean W&B logging.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import pickle
import os
import logging
import wandb
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)


class AUCCallback(keras.callbacks.Callback):
    """Custom callback to monitor AUC during training."""
    
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_auc = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        # Calculate validation AUC using anomaly scores
        val_predictions = self.model.predict(self.X_val, verbose=0)
        val_anomaly_scores = np.mean(np.square(self.X_val - val_predictions), axis=1)
        val_auc = roc_auc_score(self.y_val, val_anomaly_scores)
        
        # Update best AUC
        if val_auc > self.best_auc:
            self.best_auc = val_auc
        
        # Log to W&B if available
        if wandb.run is not None:
            wandb.log({
                'val_auc': val_auc,
                'best_val_auc': self.best_auc,
                'epoch': epoch
            })
        
        # Update logs
        logs['val_auc'] = val_auc
        logs['best_val_auc'] = self.best_auc


class FraudAutoencoder:
    """Production-grade autoencoder for fraud detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize autoencoder with configuration."""
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.threshold = None
        self.is_fitted = False
        
        # Model parameters
        self.latent_dim = self.config.get('latent_dim', 16)
        self.hidden_dims = self.config.get('hidden_dims', [128, 64, 32])
        self.dropout_rate = self.config.get('dropout_rate', 0.2)
        
        # Training parameters
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 64)
        self.epochs = self.config.get('epochs', 100)
        self.early_stopping = self.config.get('early_stopping', True)
        self.patience = self.config.get('patience', 15)
        self.reduce_lr = self.config.get('reduce_lr', True)
        self.validation_split = self.config.get('validation_split', 0.2)
        
        # Threshold parameters
        self.threshold_percentile = self.config.get('threshold_percentile', 95)
        
        logger.info(f"Autoencoder initialized with config: latent_dim={self.latent_dim}, "
                   f"hidden_dims={self.hidden_dims}, dropout_rate={self.dropout_rate}")
    
    def build_model(self, input_dim: int) -> keras.Model:
        """Build the autoencoder architecture."""
        
        # Input layer
        input_layer = layers.Input(shape=(input_dim,))
        
        # Encoder with improved architecture
        encoded = input_layer
        
        # First layer with more capacity
        encoded = layers.Dense(self.hidden_dims[0], activation='relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(self.dropout_rate)(encoded)
        
        # Additional layers with decreasing capacity
        for dim in self.hidden_dims[1:]:
            encoded = layers.Dense(dim, activation='relu')(encoded)
            encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dropout(self.dropout_rate)(encoded)
        
        # Latent space with regularization
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(encoded)
        latent = layers.BatchNormalization()(latent)
        
        # Decoder with symmetric architecture
        decoded = latent
        
        # Additional layers with increasing capacity
        for dim in reversed(self.hidden_dims[1:]):
            decoded = layers.Dense(dim, activation='relu')(decoded)
            decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Dropout(self.dropout_rate)(decoded)
        
        # Final layer with more capacity
        decoded = layers.Dense(self.hidden_dims[0], activation='relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(self.dropout_rate)(decoded)
        
        # Output layer
        output_layer = layers.Dense(input_dim, activation='linear', name='output')(decoded)
        
        # Create model
        model = keras.Model(inputs=input_layer, outputs=output_layer, name='fraud_autoencoder')
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Use 'mse' for proper serialization
            metrics=['mae']
        )
        
        logger.info(f"Model built with {model.count_params():,} parameters")
        
        return model
    
    def prepare_data(self, df_train_features: pd.DataFrame, df_test_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training and testing."""
        
        # Get numeric features (excluding target)
        train_numeric = df_train_features.select_dtypes(include=[np.number])
        test_numeric = df_test_features.select_dtypes(include=[np.number])
        
        if 'is_fraudulent' in train_numeric.columns:
            train_numeric = train_numeric.drop(columns=['is_fraudulent'])
        if 'is_fraudulent' in test_numeric.columns:
            test_numeric = test_numeric.drop(columns=['is_fraudulent'])
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(train_numeric.values)
        X_test_scaled = self.scaler.transform(test_numeric.values)
        
        logger.info(f"Data prepared: train={X_train_scaled.shape}, test={X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled
    
    def train(self, X_train: np.ndarray, X_test: np.ndarray, 
              y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Train the autoencoder model."""
        
        logger.info("Starting model training...")
        
        # Build model
        self.model = self.build_model(X_train.shape[1])
        
        # Callbacks
        callbacks = []
        
        # Early stopping
        if self.early_stopping:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Learning rate reduction
        if self.reduce_lr:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.patience // 2,
                min_lr=1e-7,
                verbose=1
            )
            callbacks.append(reduce_lr)
        
        # AUC callback
        auc_callback = AUCCallback(X_test, y_test)
        callbacks.append(auc_callback)
        
        # Training
        history = self.model.fit(
            X_train, X_train,  # Autoencoder: input = target
            validation_data=(X_test, X_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate threshold
        self._calculate_threshold(X_train)
        
        # Evaluate on test set
        test_metrics = self._evaluate_test_set(X_test, y_test)
        
        # Log final metrics
        if wandb.run is not None:
            wandb.log({
                'final_test_auc': test_metrics['auc'],
                'final_threshold': self.threshold,
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_f1_score': test_metrics['f1_score'],
                'final_precision': test_metrics['precision'],
                'final_recall': test_metrics['recall']
            })
        
        results = {
            'test_auc': test_metrics['auc'],
            'f1_score': test_metrics['f1_score'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'threshold': self.threshold,
            'train_loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
            'best_val_auc': auc_callback.best_auc
        }
        
        logger.info(f"Training completed. Test AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1_score']:.4f}, Threshold: {self.threshold:.6f}")
        
        return results
    
    def _calculate_threshold(self, X_train: np.ndarray):
        """Calculate anomaly threshold from training data."""
        
        # Get reconstruction errors for training data
        train_reconstructions = self.model.predict(X_train, verbose=0)
        train_errors = np.mean(np.square(X_train - train_reconstructions), axis=1)
        
        # Calculate threshold based on percentile
        self.threshold = np.percentile(train_errors, self.threshold_percentile)
        
        logger.info(f"Threshold calculated: {self.threshold:.6f} (percentile: {self.threshold_percentile})")
        
        # Set fitted state
        self.is_fitted = True
    
    def _evaluate_test_set(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set."""
        
        # Get reconstruction errors
        test_reconstructions = self.model.predict(X_test, verbose=0)
        test_errors = np.mean(np.square(X_test - test_reconstructions), axis=1)
        
        # Calculate AUC
        test_auc = roc_auc_score(y_test, test_errors)
        
        # Calculate predictions using threshold
        predictions = (test_errors > self.threshold).astype(int)
        
        # Calculate additional metrics
        f1 = f1_score(y_test, predictions, zero_division=0)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        
        return {
            'auc': test_auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }
    
    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores for input data."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale input if needed
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Get reconstruction errors
        reconstructions = self.model.predict(X_scaled, verbose=0)
        errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        return errors
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud labels for input data."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get anomaly scores
        scores = self.predict_anomaly_scores(X)
        
        # Apply threshold
        predictions = (scores > self.threshold).astype(int)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Ensure filepath has correct extension
        if not filepath.endswith('.keras'):
            filepath = filepath + '.keras'
        
        # Save model using native Keras format
        self.model.save(filepath)
        
        # Save scaler and threshold separately
        scaler_path = filepath.replace('.keras', '_scaler.pkl')
        threshold_path = filepath.replace('.keras', '_threshold.pkl')
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(threshold_path, 'wb') as f:
            pickle.dump(self.threshold, f)
        
        logger.info(f"Model saved to {filepath}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Threshold saved to {threshold_path}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        # Ensure filepath has correct extension
        if not filepath.endswith('.keras'):
            filepath = filepath + '.keras'
        
        # Load model using native Keras format
        self.model = keras.models.load_model(filepath)
        
        # Load scaler and threshold
        scaler_path = filepath.replace('.keras', '_scaler.pkl')
        threshold_path = filepath.replace('.keras', '_threshold.pkl')
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(threshold_path, 'rb') as f:
            self.threshold = pickle.load(f)
        
        # Set fitted flag to True after successful loading
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Scaler loaded from {scaler_path}")
        logger.info(f"Threshold loaded from {threshold_path}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        
        if self.model is None:
            return "Model not built yet"
        
        # Capture model summary
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        
        return '\n'.join(summary_list)
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance based on reconstruction error sensitivity."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # This is a simplified approach - in practice, you might want more sophisticated methods
        # For now, we'll use the magnitude of weights in the first layer as a proxy
        
        if self.model is None:
            return {}
        
        # Get weights from first layer
        first_layer_weights = self.model.layers[1].get_weights()[0]  # First dense layer
        
        # Calculate importance as mean absolute weight
        importance = np.mean(np.abs(first_layer_weights), axis=1)
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature in enumerate(feature_names):
            if i < len(importance):
                feature_importance[feature] = float(importance[i])
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance 