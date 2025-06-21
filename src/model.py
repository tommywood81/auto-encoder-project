import torch.nn as nn
from src.config import ENCODER_DIMS, DECODER_DIMS


class Autoencoder(nn.Module):
    """
    Autoencoder neural network for anomaly detection.
    
    The model learns to reconstruct normal transactions and uses reconstruction
    error to detect fraudulent transactions as anomalies.
    """
    
    def __init__(self, input_dim):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim (int): Number of input features
        """
        super().__init__()
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in ENCODER_DIMS:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers[:-1])  # Remove last ReLU
        
        # Decoder layers (reverse of encoder)
        decoder_layers = []
        decoder_dims = DECODER_DIMS + [input_dim]  # Add input_dim at the end
        prev_dim = ENCODER_DIMS[-1]
        
        for dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        
        # Remove last ReLU and replace with identity for final layer
        decoder_layers = decoder_layers[:-1]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Reconstructed tensor
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded 