import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from src.config import DATA_DIR


def load_and_clean_data():
    """
    Load and clean the fraud detection dataset.
    
    Returns:
        pd.DataFrame: Cleaned and preprocessed dataset
    """
    print("🧼 Loading and cleaning data...")
    
    # Load transaction and identity data
    trans = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"))
    iden = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"))

    # Merge transaction and identity data
    df = trans.merge(iden, on="TransactionID", how="left")

    # Drop columns that are not useful for modeling
    drop_cols = [
        "TransactionID", "TransactionDT", "ProductCD", "DeviceType", "DeviceInfo",
        "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38"
    ]
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Remove columns with too many missing values (>50%)
    df = df.dropna(thresh=len(df) * 0.5, axis=1)
    
    # Fill remaining missing values with 0
    df = df.fillna(0)

    # Encode categorical variables
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    print(f"✅ Final shape: {df.shape}")
    return df 