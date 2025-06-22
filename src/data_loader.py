"""
Data loader for IEEE-CIS Fraud Detection dataset.
Updated to work with the new pipeline structure.
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging
from src.config import DATA_RAW, DATA_CLEANED, DATA_ENGINEERED, DATA_PROCESSED, RANDOM_STATE, TEST_SIZE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDataLoader:
    """Data loader for IEEE-CIS Fraud Detection dataset."""
    
    def __init__(self, use_pipeline=True):
        """
        Initialize the data loader.
        
        Args:
            use_pipeline (bool): Whether to use the new pipeline structure or legacy approach
        """
        self.use_pipeline = use_pipeline
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_raw_data(self):
        """Load raw transaction + identity data."""
        logger.info("Loading raw transaction + identity data...")
        
        # Load transaction data
        trans_path = os.path.join(DATA_RAW, "train_transaction.csv")
        trans = pd.read_csv(trans_path)
        logger.info(f"Transaction data shape: {trans.shape}")
        
        # Load identity data
        iden_path = os.path.join(DATA_RAW, "train_identity.csv")
        iden = pd.read_csv(iden_path)
        logger.info(f"Identity data shape: {iden.shape}")
        
        # Merge on TransactionID
        df = trans.merge(iden, on="TransactionID", how="left")
        logger.info(f"Data loaded. Shape after merge: {df.shape}")
        logger.info(f"Fraud distribution: {df['isFraud'].value_counts()}")
        
        return df
    
    def load_cleaned_data(self):
        """Load cleaned data from the cleaned directory."""
        logger.info("Loading cleaned data...")
        
        cleaned_file = os.path.join(DATA_CLEANED, "train_cleaned.csv")
        if not os.path.exists(cleaned_file):
            raise FileNotFoundError(f"Cleaned data not found at {cleaned_file}. Run data cleaning first.")
        
        df = pd.read_csv(cleaned_file)
        logger.info(f"Cleaned data shape: {df.shape}")
        
        return df
    
    def load_engineered_data(self):
        """Load engineered data from the engineered directory."""
        logger.info("Loading engineered data...")
        
        engineered_file = os.path.join(DATA_ENGINEERED, "train_features.csv")
        if not os.path.exists(engineered_file):
            raise FileNotFoundError(f"Engineered data not found at {engineered_file}. Run feature engineering first.")
        
        df = pd.read_csv(engineered_file)
        logger.info(f"Engineered data shape: {df.shape}")
        
        return df
    
    def load_processed_data_from_disk(self):
        """Load processed data from the processed directory."""
        logger.info("Loading processed data from disk...")
        
        # Check if processed data exists
        processed_files = [
            os.path.join(DATA_PROCESSED, "X_train.npy"),
            os.path.join(DATA_PROCESSED, "X_test.npy"),
            os.path.join(DATA_PROCESSED, "y_train.npy"),
            os.path.join(DATA_PROCESSED, "y_test.npy"),
            os.path.join(DATA_PROCESSED, "scaler.pkl"),
            os.path.join(DATA_PROCESSED, "label_encoders.pkl")
        ]
        
        if not all(os.path.exists(f) for f in processed_files):
            raise FileNotFoundError("Processed data not found. Run data processing first.")
        
        # Load processed data
        X_train = np.load(os.path.join(DATA_PROCESSED, "X_train.npy"))
        X_test = np.load(os.path.join(DATA_PROCESSED, "X_test.npy"))
        y_train = np.load(os.path.join(DATA_PROCESSED, "y_train.npy"))
        y_test = np.load(os.path.join(DATA_PROCESSED, "y_test.npy"))
        
        # Load scaler and label encoders
        with open(os.path.join(DATA_PROCESSED, "scaler.pkl"), 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(os.path.join(DATA_PROCESSED, "label_encoders.pkl"), 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        # Load feature names
        feature_names_file = os.path.join(DATA_PROCESSED, "feature_names.txt")
        if os.path.exists(feature_names_file):
            with open(feature_names_file, 'r') as f:
                feature_names = f.read().splitlines()
        else:
            # Fallback: load from engineered data
            df = self.load_engineered_data()
            feature_names = df.drop(columns=["isFraud"]).columns.tolist()
        
        # For autoencoder, we only train on non-fraudulent data
        X_train_ae = X_train[y_train == 0]
        
        logger.info(f"Processed data loaded:")
        logger.info(f"  X_train: {X_train.shape}")
        logger.info(f"  X_test: {X_test.shape}")
        logger.info(f"  X_train_ae (non-fraud only): {X_train_ae.shape}")
        logger.info(f"  Features: {len(feature_names)}")
        
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
    
    def save_processed_data(self, data_dict):
        """Save processed data to the processed directory."""
        logger.info("Saving processed data...")
        
        # Create processed directory if it doesn't exist
        os.makedirs(DATA_PROCESSED, exist_ok=True)
        
        # Save numpy arrays
        np.save(os.path.join(DATA_PROCESSED, "X_train.npy"), data_dict['X_train'])
        np.save(os.path.join(DATA_PROCESSED, "X_test.npy"), data_dict['X_test'])
        np.save(os.path.join(DATA_PROCESSED, "y_train.npy"), data_dict['y_train'])
        np.save(os.path.join(DATA_PROCESSED, "y_test.npy"), data_dict['y_test'])
        
        # Save scaler and label encoders
        with open(os.path.join(DATA_PROCESSED, "scaler.pkl"), 'wb') as f:
            pickle.dump(data_dict['scaler'], f)
        
        with open(os.path.join(DATA_PROCESSED, "label_encoders.pkl"), 'wb') as f:
            pickle.dump(data_dict['label_encoders'], f)
        
        # Save feature names
        with open(os.path.join(DATA_PROCESSED, "feature_names.txt"), 'w') as f:
            f.write('\n'.join(data_dict['feature_names']))
        
        logger.info(f"Processed data saved to {DATA_PROCESSED}")
    
    def process_engineered_data(self, df):
        """Process engineered data for modeling."""
        logger.info("Processing engineered data for modeling...")
        
        # Prepare features and target
        X = df.drop(columns=["isFraud"])
        y = df["isFraud"]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts()}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        
        # For autoencoder, we only train on non-fraudulent data
        X_train_ae = X_train[y_train == 0]
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        logger.info(f"Processing complete:")
        logger.info(f"  X_train: {X_train.shape}")
        logger.info(f"  X_test: {X_test.shape}")
        logger.info(f"  X_train_ae (non-fraud only): {X_train_ae.shape}")
        logger.info(f"  Features: {len(feature_names)}")
        
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
    
    def load_processed_data(self):
        """
        Complete data loading pipeline.
        Tries to load from processed directory first, falls back to processing from engineered data.
        """
        try:
            # Try to load from processed directory
            return self.load_processed_data_from_disk()
        except FileNotFoundError:
            logger.info("Processed data not found, processing from engineered data...")
            
            # Load engineered data
            df = self.load_engineered_data()
            
            # Process for modeling
            data_dict = self.process_engineered_data(df)
            
            # Save processed data for future use
            self.save_processed_data(data_dict)
            
            return data_dict
    
    # Legacy methods for backward compatibility
    def load_and_merge_data(self):
        """Legacy method - load raw data."""
        return self.load_raw_data()
    
    def clean_and_preprocess(self, df):
        """Legacy method - now handled by DataCleaner."""
        logger.warning("clean_and_preprocess is deprecated. Use DataCleaner instead.")
        return df
    
    def prepare_features_and_target(self, df):
        """Legacy method - prepare features and target."""
        X = df.drop(columns=["isFraud"])
        y = df["isFraud"]
        return X, y
    
    def scale_and_split(self, X, y, test_size=0.2, random_state=42):
        """Legacy method - scale and split data."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # For autoencoder, we only train on non-fraudulent data
        X_train_ae = X_train[y_train == 0]
        
        return X_train, X_test, y_train, y_test, X_train_ae
    
    def get_feature_names(self, df):
        """Legacy method - get feature names."""
        return df.drop(columns=["isFraud"]).columns.tolist() 