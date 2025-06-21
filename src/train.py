import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import Autoencoder
from src.config import BATCH_SIZE, LEARNING_RATE, EPOCHS


def train_autoencoder(X_train_ae, input_dim, epochs=None):
    """
    Train the autoencoder on non-fraudulent data.
    
    Args:
        X_train_ae (np.ndarray): Training data (only non-fraudulent transactions)
        input_dim (int): Number of input features
        epochs (int, optional): Number of training epochs. Defaults to config value.
        
    Returns:
        Autoencoder: Trained autoencoder model
    """
    if epochs is None:
        epochs = EPOCHS
        
    print("🏋️ Training autoencoder...")
    
    # Initialize model, optimizer, and loss function
    model = Autoencoder(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    # Create data loader
    dataset = TensorDataset(torch.tensor(X_train_ae, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in loader:
            x = batch[0]
            
            # Forward pass
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"📉 Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
    
    print("✅ Training completed!")
    return model 