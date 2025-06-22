import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE, TEST_SIZE, DATA_CLEANED, DATA_ENGINEERED


class FeatureEngineer:
    """Feature engineering pipeline for fraud detection dataset."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_info = {}
        self.scaler = None
        
    def load_cleaned_data(self):
        """Load cleaned data from the cleaned directory."""
        print("📂 Loading cleaned data...")
        
        cleaned_file = os.path.join(DATA_CLEANED, "train_cleaned.csv")
        if not os.path.exists(cleaned_file):
            raise FileNotFoundError(f"Cleaned data not found at {cleaned_file}. Run data cleaning first.")
        
        df = pd.read_csv(cleaned_file)
        print(f"📊 Loaded cleaned data shape: {df.shape}")
        
        return df
    
    def create_transaction_features(self, df):
        """Create transaction-related features."""
        print("💳 Creating transaction features...")
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['TransactionAmt'])
        df['amount_sqrt'] = np.sqrt(df['TransactionAmt'])
        
        # Card features (if available)
        card_cols = [col for col in df.columns if 'card' in col.lower()]
        if card_cols:
            df['card_count'] = df[card_cols].notna().sum(axis=1)
        
        # Email features (if available)
        email_cols = [col for col in df.columns if 'email' in col.lower()]
        if email_cols:
            df['email_count'] = df[email_cols].notna().sum(axis=1)
        
        # Address features (if available)
        addr_cols = [col for col in df.columns if 'addr' in col.lower()]
        if addr_cols:
            df['addr_count'] = df[addr_cols].notna().sum(axis=1)
        
        self.feature_info['transaction_features'] = {
            'amount_log': 'Log transformation of transaction amount',
            'amount_sqrt': 'Square root transformation of transaction amount',
            'card_count': f'Number of card-related fields filled ({len(card_cols)} columns)',
            'email_count': f'Number of email-related fields filled ({len(email_cols)} columns)',
            'addr_count': f'Number of address-related fields filled ({len(addr_cols)} columns)'
        }
        
        print(f"💳 Created {len(self.feature_info['transaction_features'])} transaction features")
        return df
    
    def create_identity_features(self, df):
        """Create identity-related features."""
        print("🆔 Creating identity features...")
        
        # Identity features (V columns)
        v_cols = [col for col in df.columns if col.startswith('V')]
        if v_cols:
            # Count of V features that are not null
            df['v_features_count'] = df[v_cols].notna().sum(axis=1)
            
            # Mean of V features
            df['v_features_mean'] = df[v_cols].mean(axis=1)
            
            # Std of V features
            df['v_features_std'] = df[v_cols].std(axis=1)
            
            # Sum of V features
            df['v_features_sum'] = df[v_cols].sum(axis=1)
        
        # ID features
        id_cols = [col for col in df.columns if col.startswith('id_')]
        if id_cols:
            df['id_features_count'] = df[id_cols].notna().sum(axis=1)
        
        self.feature_info['identity_features'] = {
            'v_features_count': f'Count of V features filled ({len(v_cols)} columns)',
            'v_features_mean': f'Mean of V features ({len(v_cols)} columns)',
            'v_features_std': f'Standard deviation of V features ({len(v_cols)} columns)',
            'v_features_sum': f'Sum of V features ({len(v_cols)} columns)',
            'id_features_count': f'Count of ID features filled ({len(id_cols)} columns)'
        }
        
        print(f"🆔 Created {len(self.feature_info['identity_features'])} identity features")
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between different feature groups."""
        print("🔗 Creating interaction features...")
        
        # Amount interactions
        if 'TransactionAmt' in df.columns:
            # Amount * card count
            if 'card_count' in df.columns:
                df['amount_card_interaction'] = df['TransactionAmt'] * df['card_count']
            
            # Amount * email count
            if 'email_count' in df.columns:
                df['amount_email_interaction'] = df['TransactionAmt'] * df['email_count']
        
        # V features interactions
        if 'v_features_count' in df.columns and 'v_features_mean' in df.columns:
            df['v_count_mean_interaction'] = df['v_features_count'] * df['v_features_mean']
        
        self.feature_info['interaction_features'] = {
            'amount_card_interaction': 'Transaction amount * card count',
            'amount_email_interaction': 'Transaction amount * email count',
            'v_count_mean_interaction': 'V features count * V features mean'
        }
        
        print(f"🔗 Created {len(self.feature_info['interaction_features'])} interaction features")
        return df
    
    def create_statistical_features(self, df):
        """Create statistical features."""
        print("📊 Creating statistical features...")
        
        # V features statistics
        v_cols = [col for col in df.columns if col.startswith('V')]
        if v_cols:
            # Percentiles
            df['v_features_q25'] = df[v_cols].quantile(0.25, axis=1)
            df['v_features_q75'] = df[v_cols].quantile(0.75, axis=1)
            df['v_features_iqr'] = df['v_features_q75'] - df['v_features_q25']
            
            # Range
            df['v_features_range'] = df[v_cols].max(axis=1) - df[v_cols].min(axis=1)
        
        self.feature_info['statistical_features'] = {
            'v_features_q25': '25th percentile of V features',
            'v_features_q75': '75th percentile of V features',
            'v_features_iqr': 'Interquartile range of V features',
            'v_features_range': 'Range of V features'
        }
        
        print(f"📊 Created {len(self.feature_info['statistical_features'])} statistical features")
        return df
    
    def handle_infinite_values(self, df):
        """Handle infinite values in the dataset."""
        print("♾️ Handling infinite values...")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        print(f"♾️ Handled infinite values in {len(numeric_cols)} numeric columns")
        return df
    
    def save_engineered_data(self, df, suffix=""):
        """Save engineered data to the engineered directory."""
        print("💾 Saving engineered data...")
        
        # Create engineered directory if it doesn't exist
        os.makedirs(DATA_ENGINEERED, exist_ok=True)
        
        # Save engineered data
        output_file = os.path.join(DATA_ENGINEERED, f"train_features{suffix}.csv")
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
        
        # Save feature information
        info_file = os.path.join(DATA_ENGINEERED, f"feature_info{suffix}.json")
        with open(info_file, 'w') as f:
            json.dump(convert_numpy_types(self.feature_info), f, indent=2)
        
        print(f"💾 Saved engineered data to {output_file}")
        print(f"💾 Saved feature info to {info_file}")
        
        return output_file
    
    def engineer_features(self, save_output=True):
        """
        Complete feature engineering pipeline.
        
        Args:
            save_output (bool): Whether to save engineered data to disk
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        print("🧪 Starting feature engineering pipeline...")
        
        # Load cleaned data
        df = self.load_cleaned_data()
        
        # Create features
        df = self.create_transaction_features(df)
        df = self.create_identity_features(df)
        df = self.create_interaction_features(df)
        df = self.create_statistical_features(df)
        
        # Handle infinite values
        df = self.handle_infinite_values(df)
        
        print(f"✅ Final engineered shape: {df.shape}")
        
        # Save engineered data if requested
        if save_output:
            self.save_engineered_data(df)
        
        return df
    
    def preprocess_for_modeling(self, df):
        """
        Preprocess engineered data for autoencoder training.
        
        Args:
            df (pd.DataFrame): Engineered dataset
            
        Returns:
            tuple: (X_train_ae, X_test, y_train, y_test, scaler) where X_train_ae contains only non-fraudulent data
        """
        print("🧪 Preprocessing and splitting data...")
        
        # Separate features and target
        X = df.drop(columns=["isFraud"])
        y = df["isFraud"]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )

        # For autoencoder, we only train on non-fraudulent data (normal transactions)
        X_train_ae = X_train[y_train == 0]
        
        print(f"✅ Training samples (non-fraudulent): {X_train_ae.shape[0]}")
        print(f"✅ Test samples: {X_test.shape[0]}")
        print(f"✅ Features: {X_train.shape[1]}")
        
        return X_train_ae, X_test, y_train, y_test, self.scaler


def preprocess_data(df):
    """
    Legacy function for backward compatibility.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        tuple: (X_train_ae, X_test, y_train, y_test) where X_train_ae contains only non-fraudulent data
    """
    engineer = FeatureEngineer()
    return engineer.preprocess_for_modeling(df) 