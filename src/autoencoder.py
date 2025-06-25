"""
Baseline autoencoder for E-commerce Fraud Detection.
Simple architecture focused on anomaly detection.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import PipelineConfig
from src.feature_factory import BaselineFeatureFactory

logger = logging.getLogger(__name__)


class BaselineAutoencoder:
    """Baseline autoencoder for fraud detection."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.feature_factory = BaselineFeatureFactory(self.config)
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training with time-aware split."""
        logger.info("Preparing data for baseline model...")
        
        # Engineer features
        df = self.feature_factory.engineer_features(save_output=False)
        
        # Get numeric features only
        df_numeric = self.feature_factory.get_numeric_features(df)
        
        # Separate features and target
        X = df_numeric.drop(columns=['is_fraudulent'])
        y = df_numeric['is_fraudulent']
        
        # Time-aware split (if transaction_date features exist)
        if 'hour' in X.columns and self.config.data.use_time_split:
            # Use hour as proxy for time (simplified time split)
            # Split based on hour: train on first 80% of hours, test on last 20%
            hours = X['hour'].values
            split_hour = np.percentile(hours, 80)
            
            train_mask = hours <= split_hour
            test_mask = hours > split_hour
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
            logger.info(f"Time-aware split: train={len(X_train)}, test={len(X_test)}")
        else:
            # Fallback to random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.data.test_size, 
                random_state=self.config.data.random_state, stratify=y
            )
            logger.info(f"Random split: train={len(X_train)}, test={len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # For autoencoder, we only train on non-fraudulent data
        train_normal_mask = y_train == 0
        X_train_normal = X_train_scaled[train_normal_mask]
        
        logger.info(f"Training on {len(X_train_normal)} normal transactions")
        logger.info(f"Features: {X_train.shape[1]}")
        
        return X_train_normal, X_test_scaled, y_train, y_test
    
    def build_model(self, input_dim: int):
        """Build simple autoencoder architecture."""
        logger.info(f"Building autoencoder with input_dim={input_dim}")
        
        # Encoder
        encoder = keras.Sequential([
            layers.Dense(self.config.model.hidden_dim, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(self.config.model.latent_dim, activation='relu')
        ])
        
        # Decoder
        decoder = keras.Sequential([
            layers.Dense(self.config.model.hidden_dim, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(input_dim, activation='linear')
        ])
        
        # Autoencoder
        self.model = keras.Sequential([encoder, decoder])
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.model.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Autoencoder built and compiled")
        return self.model
    
    def train(self):
        """Train the baseline autoencoder."""
        logger.info("Starting baseline autoencoder training...")
        
        # Prepare data
        X_train_normal, X_test_scaled, y_train, y_test = self.prepare_data()
        
        # Build model
        self.build_model(input_dim=X_train_normal.shape[1])
        
        # Train model
        history = self.model.fit(
            X_train_normal, X_train_normal,
            epochs=self.config.model.epochs,
            batch_size=self.config.model.batch_size,
            validation_split=self.config.model.validation_split,
            verbose=1
        )
        
        # Calculate reconstruction error for threshold
        train_reconstructions = self.model.predict(X_train_normal)
        train_mse = np.mean(np.square(X_train_normal - train_reconstructions), axis=1)
        
        # Set threshold based on percentile
        self.threshold = np.percentile(train_mse, self.config.model.threshold_percentile)
        logger.info(f"Anomaly threshold set at {self.threshold:.6f} (percentile {self.config.model.threshold_percentile})")
        
        # Evaluate on test set
        test_reconstructions = self.model.predict(X_test_scaled)
        test_mse = np.mean(np.square(X_test_scaled - test_reconstructions), axis=1)
        
        # Predict anomalies
        y_pred = (test_mse > self.threshold).astype(int)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, test_mse)
        logger.info(f"Test ROC AUC: {roc_auc:.4f}")
        
        # Save model if requested
        if self.config.model.save_model:
            self.save_model()
        
        return {
            'roc_auc': roc_auc,
            'threshold': self.threshold,
            'y_test': y_test,
            'y_pred': y_pred,
            'test_mse': test_mse,
            'history': history
        }
    
    def save_model(self):
        """Save the trained model."""
        os.makedirs(self.config.data.models_dir, exist_ok=True)
        model_path = os.path.join(self.config.data.models_dir, f"{self.config.model.model_name}.h5")
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores for new data."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        X_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        return mse
    
    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies for new data."""
        scores = self.predict_anomaly_scores(X)
        return (scores > self.threshold).astype(int) 