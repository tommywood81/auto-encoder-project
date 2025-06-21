"""
Data loader for IEEE-CIS Fraud Detection dataset.
Based on working code from notebook2.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDataLoader:
    """Data loader for IEEE-CIS Fraud Detection dataset."""
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_merge_data(self):
        """Load and merge transaction + identity data."""
        logger.info("Loading and merging transaction + identity data...")
        
        # Load transaction data
        trans_path = os.path.join(self.data_dir, "train_transaction.csv")
        trans = pd.read_csv(trans_path)
        logger.info(f"Transaction data shape: {trans.shape}")
        
        # Load identity data
        iden_path = os.path.join(self.data_dir, "train_identity.csv")
        iden = pd.read_csv(iden_path)
        logger.info(f"Identity data shape: {iden.shape}")
        
        # Merge on TransactionID
        df = trans.merge(iden, on="TransactionID", how="left")
        logger.info(f"Data loaded. Shape after merge: {df.shape}")
        logger.info(f"Fraud distribution: {df['isFraud'].value_counts()}")
        
        return df
    
    def clean_and_preprocess(self, df):
        """Clean and preprocess the data."""
        logger.info("Cleaning and preprocessing data...")
        
        # Drop problematic columns
        drop_cols = [
            "TransactionID", "TransactionDT", "ProductCD", "DeviceType", "DeviceInfo",
            "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38"
        ]
        df = df.drop(columns=drop_cols, errors='ignore')
        
        # Drop columns with too many missing values
        df = df.dropna(thresh=df.shape[0]*0.5, axis=1)
        
        # Fill remaining NaN values with 0
        df = df.fillna(0)
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include="object").columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df
    
    def prepare_features_and_target(self, df):
        """Prepare features and target variables."""
        X = df.drop(columns=["isFraud"])
        y = df["isFraud"]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts()}")
        
        return X, y
    
    def scale_and_split(self, X, y, test_size=0.2, random_state=42):
        """Scale features and split into train/test sets."""
        logger.info("Scaling features and splitting train/test sets...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # For autoencoder, we only train on non-fraudulent data
        X_train_ae = X_train[y_train == 0]
        
        logger.info(f"Split complete:")
        logger.info(f"  X_train: {X_train.shape}")
        logger.info(f"  X_test: {X_test.shape}")
        logger.info(f"  X_train_ae (non-fraud only): {X_train_ae.shape}")
        
        return X_train, X_test, y_train, y_test, X_train_ae
    
    def get_feature_names(self, df):
        """Get feature names after preprocessing."""
        return df.drop(columns=["isFraud"]).columns.tolist()
    
    def load_processed_data(self):
        """Complete data loading pipeline."""
        # Load and merge
        df = self.load_and_merge_data()
        
        # Clean and preprocess
        df_clean = self.clean_and_preprocess(df)
        
        # Prepare features and target
        X, y = self.prepare_features_and_target(df_clean)
        
        # Scale and split
        X_train, X_test, y_train, y_test, X_train_ae = self.scale_and_split(X, y)
        
        # Get feature names
        feature_names = self.get_feature_names(df_clean)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_ae': X_train_ae,
            'feature_names': feature_names,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        } 