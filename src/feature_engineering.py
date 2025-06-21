from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE, TEST_SIZE


def preprocess_data(df):
    """
    Preprocess data for autoencoder training.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        tuple: (X_train_ae, X_test, y_train, y_test) where X_train_ae contains only non-fraudulent data
    """
    print("🧪 Preprocessing and splitting data...")
    
    # Separate features and target
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # For autoencoder, we only train on non-fraudulent data (normal transactions)
    X_train_ae = X_train[y_train == 0]
    
    print(f"✅ Training samples (non-fraudulent): {X_train_ae.shape[0]}")
    print(f"✅ Test samples: {X_test.shape[0]}")
    
    return X_train_ae, X_test, y_train, y_test 