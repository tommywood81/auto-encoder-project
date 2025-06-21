"""
Autoencoder model for fraud detection.
Based on working code from notebook2.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """Autoencoder neural network for anomaly detection."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers[:-1])  # Remove last ReLU
        
        # Decoder layers (reverse of encoder)
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        prev_dim = hidden_dims_reversed[0]
        for hidden_dim in hidden_dims_reversed[1:]:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        """Forward pass through encoder and decoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderTrainer:
    """Trainer for autoencoder model."""
    
    def __init__(self, model, device='cpu', lr=1e-3):
        """
        Initialize trainer.
        
        Args:
            model: Autoencoder model
            device: Device to train on ('cpu' or 'cuda')
            lr: Learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            x_batch = batch[0].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x_batch)
            loss = self.criterion(output, x_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss
    
    def train(self, X_train_ae, epochs=20, batch_size=256, verbose=True):
        """
        Train the autoencoder.
        
        Args:
            X_train_ae: Training data (non-fraudulent only)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print progress
        """
        logger.info(f"Starting training on {self.device}...")
        
        # Create data loader
        train_dataset = TensorDataset(torch.tensor(X_train_ae, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = self.train_epoch(train_loader)
            self.train_losses.append(total_loss)
            
            if verbose:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
        
        logger.info("Training completed!")
        return self.train_losses
    
    def get_reconstruction_error(self, X):
        """Get reconstruction error for given data."""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            x_reconstructed = self.model(x_tensor).cpu().numpy()
            
        # Calculate mean squared error
        recon_errors = ((X - x_reconstructed) ** 2).mean(axis=1)
        return recon_errors
    
    def detect_anomalies(self, X_test, X_train_ae, percentile=95):
        """
        Detect anomalies using reconstruction error.
        
        Args:
            X_test: Test data
            X_train_ae: Training data (for threshold calculation)
            percentile: Percentile for threshold calculation
            
        Returns:
            y_pred: Predicted labels (0=normal, 1=anomaly)
            threshold: Calculated threshold
            recon_errors: Reconstruction errors
        """
        logger.info("Performing anomaly detection...")
        
        # Get reconstruction errors for test data
        recon_errors = self.get_reconstruction_error(X_test)
        
        # Calculate threshold from training data
        recon_train = self.get_reconstruction_error(X_train_ae)
        threshold = np.percentile(recon_train, percentile)
        
        # Predict anomalies
        y_pred = (recon_errors > threshold).astype(int)
        
        logger.info(f"Threshold ({percentile}th percentile): {threshold:.4f}")
        
        return y_pred, threshold, recon_errors 