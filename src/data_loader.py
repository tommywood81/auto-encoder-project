"""
Data loader for IEEE-CIS Fraud Detection dataset.
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging
from src.config import PipelineConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for IEEE-CIS Fraud Detection dataset."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_raw_data(self):
        """Load raw transaction + identity data."""
        logger.info("Loading raw transaction + identity data...")
        
        trans_path = os.path.join(self.config.data.raw_dir, "train_transaction.csv")
        iden_path = os.path.join(self.config.data.raw_dir, "train_identity.csv")
        
        trans = pd.read_csv(trans_path)
        iden = pd.read_csv(iden_path)
        
        logger.info(f"Transaction data: {trans.shape}")
        logger.info(f"Identity data: {iden.shape}")
        
        # Merge on TransactionID
        df = trans.merge(iden, on="TransactionID", how="left")
        logger.info(f"Data loaded. Shape after merge: {df.shape}")
        logger.info(f"Fraud distribution: {df['isFraud'].value_counts()}")
        
        return df
    
    def load_cleaned_data(self):
        """Load cleaned data from the cleaned directory."""
        logger.info("Loading cleaned data...")
        
        cleaned_file = os.path.join(self.config.data.cleaned_dir, "train_cleaned.csv")
        if not os.path.exists(cleaned_file):
            raise FileNotFoundError(f"Cleaned data not found at {cleaned_file}. Run data cleaning first.")
        
        df = pd.read_csv(cleaned_file)
        logger.info(f"Cleaned data: {df.shape}")
        
        return df
    
    def load_engineered_data(self):
        """Load engineered data from the engineered directory."""
        logger.info("Loading engineered data...")
        
        engineered_file = os.path.join(self.config.data.engineered_dir, "train_features.csv")
        if not os.path.exists(engineered_file):
            raise FileNotFoundError(f"Engineered data not found at {engineered_file}. Run feature engineering first.")
        
        df = pd.read_csv(engineered_file)
        logger.info(f"Engineered data: {df.shape}")
        
        return df
    
    def load_processed_data_from_disk(self):
        """Load processed data from the processed directory."""
        logger.info("Loading processed data from disk...")
        
        # Check if processed data exists
        processed_files = [
            os.path.join(self.config.data.processed_dir, "X_train.npy"),
            os.path.join(self.config.data.processed_dir, "X_test.npy"),
            os.path.join(self.config.data.processed_dir, "y_train.npy"),
            os.path.join(self.config.data.processed_dir, "y_test.npy"),
            os.path.join(self.config.data.processed_dir, "scaler.pkl"),
            os.path.join(self.config.data.processed_dir, "label_encoders.pkl")
        ]
        
        if not all(os.path.exists(f) for f in processed_files):
            raise FileNotFoundError("Processed data not found. Run data processing first.")
        
        # Load processed data
        X_train = np.load(os.path.join(self.config.data.processed_dir, "X_train.npy"))
        X_test = np.load(os.path.join(self.config.data.processed_dir, "X_test.npy"))
        y_train = np.load(os.path.join(self.config.data.processed_dir, "y_train.npy"))
        y_test = np.load(os.path.join(self.config.data.processed_dir, "y_test.npy"))
        
        # Load scaler and label encoders
        with open(os.path.join(self.config.data.processed_dir, "scaler.pkl"), 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(os.path.join(self.config.data.processed_dir, "label_encoders.pkl"), 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        # Load feature names
        feature_names_file = os.path.join(self.config.data.processed_dir, "feature_names.txt")
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
        
        os.makedirs(self.config.data.processed_dir, exist_ok=True)
        
        # Save numpy arrays
        np.save(os.path.join(self.config.data.processed_dir, "X_train.npy"), data_dict['X_train'])
        np.save(os.path.join(self.config.data.processed_dir, "X_test.npy"), data_dict['X_test'])
        np.save(os.path.join(self.config.data.processed_dir, "y_train.npy"), data_dict['y_train'])
        np.save(os.path.join(self.config.data.processed_dir, "y_test.npy"), data_dict['y_test'])
        
        # Save scaler and label encoders
        with open(os.path.join(self.config.data.processed_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(data_dict['scaler'], f)
        
        with open(os.path.join(self.config.data.processed_dir, "label_encoders.pkl"), 'wb') as f:
            pickle.dump(data_dict['label_encoders'], f)
        
        # Save feature names
        with open(os.path.join(self.config.data.processed_dir, "feature_names.txt"), 'w') as f:
            f.write('\n'.join(data_dict['feature_names']))
        
        logger.info(f"Processed data saved to {self.config.data.processed_dir}")
    
    def process_engineered_data(self, df):
        """Process engineered data for modeling."""
        logger.info("Processing engineered data for modeling...")
        
        # Prepare features and target
        X = df.drop(columns=["isFraud"])
        y = df["isFraud"]
        
        logger.info(f"Features: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts()}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.data.test_size, 
            stratify=y, random_state=self.config.data.random_state
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
        """Complete data loading pipeline."""
        try:
            return self.load_processed_data_from_disk()
        except FileNotFoundError:
            logger.info("Processed data not found, processing from engineered data...")
            df = self.load_engineered_data()
            data_dict = self.process_engineered_data(df)
            self.save_processed_data(data_dict)
            return data_dict


# Legacy class for backward compatibility
class FraudDataLoader:
    """Legacy data loader class for backward compatibility."""
    
    def __init__(self, use_pipeline=True):
        self.config = PipelineConfig()
        self.loader = DataLoader(self.config)
        self.use_pipeline = use_pipeline
        self.scaler = self.loader.scaler
        self.label_encoders = self.loader.label_encoders
    
    def load_raw_data(self):
        return self.loader.load_raw_data()
    
    def load_cleaned_data(self):
        return self.loader.load_cleaned_data()
    
    def load_engineered_data(self):
        return self.loader.load_engineered_data()
    
    def load_processed_data_from_disk(self):
        return self.loader.load_processed_data_from_disk()
    
    def save_processed_data(self, data_dict):
        return self.loader.save_processed_data(data_dict)
    
    def process_engineered_data(self, df):
        return self.loader.process_engineered_data(df)
    
    def load_processed_data(self):
        return self.loader.load_processed_data() 