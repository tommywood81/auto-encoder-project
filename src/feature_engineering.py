import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import PipelineConfig, FeatureConfig
from src.feature_factory import FeatureFactory


class FeatureEngineer:
    """Feature engineering pipeline for fraud detection dataset."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.feature_factory = FeatureFactory(self.config.features)
        self.scaler = None
        
    def load_cleaned_data(self):
        """Load cleaned data from the cleaned directory."""
        print("Loading cleaned data...")
        
        cleaned_file = os.path.join(self.config.data.cleaned_dir, "train_cleaned.csv")
        if not os.path.exists(cleaned_file):
            raise FileNotFoundError(f"Cleaned data not found at {cleaned_file}. Run data cleaning first.")
        
        df = pd.read_csv(cleaned_file)
        print(f"Cleaned data: {df.shape}")
        
        return df
    
    def engineer_features(self, save_output=True):
        """Complete feature engineering pipeline using feature factory."""
        print("Starting feature engineering pipeline...")
        
        # Load cleaned data
        df = self.load_cleaned_data()
        
        # Use feature factory to build features
        df = self.feature_factory.build_features(df)
        
        print(f"Final engineered shape: {df.shape}")
        
        # Save engineered data if requested
        if save_output:
            self.save_engineered_data(df)
        
        return df
    
    def save_engineered_data(self, df, suffix=""):
        """Save engineered data to the engineered directory."""
        print("Saving engineered data...")
        
        os.makedirs(self.config.data.engineered_dir, exist_ok=True)
        
        # Save engineered data
        output_file = os.path.join(self.config.data.engineered_dir, f"train_features{suffix}.csv")
        df.to_csv(output_file, index=False)
        
        # Convert NumPy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save feature information from factory
        feature_summary = self.feature_factory.get_feature_summary()
        info_file = os.path.join(self.config.data.engineered_dir, f"feature_info{suffix}.json")
        with open(info_file, 'w') as f:
            json.dump(convert_numpy_types(feature_summary), f, indent=2)
        
        print(f"Saved engineered data to {output_file}")
        
        return output_file
    
    def preprocess_for_modeling(self, df):
        """Preprocess engineered data for autoencoder training."""
        print("Preprocessing and splitting data...")
        
        # Separate features and target
        X = df.drop(columns=["isFraud"])
        y = df["isFraud"]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.data.test_size, 
            stratify=y, random_state=self.config.data.random_state
        )

        # For autoencoder, we only train on non-fraudulent data
        X_train_ae = X_train[y_train == 0]
        
        print(f"Training samples (non-fraudulent): {X_train_ae.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        
        return X_train_ae, X_test, y_train, y_test, self.scaler


def preprocess_data(df):
    """Legacy function for backward compatibility."""
    engineer = FeatureEngineer()
    return engineer.preprocess_for_modeling(df) 