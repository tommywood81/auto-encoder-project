import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import LabelEncoder
from src.config import DATA_RAW, DATA_CLEANED


class DataCleaner:
    """Data cleaning pipeline for fraud detection dataset."""
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.cleaning_stats = {}
        self.label_encoders = {}
        
    def load_raw_data(self):
        """Load raw transaction and identity data."""
        print("📂 Loading raw data...")
        
        # Load transaction and identity data
        trans = pd.read_csv(os.path.join(DATA_RAW, "train_transaction.csv"))
        iden = pd.read_csv(os.path.join(DATA_RAW, "train_identity.csv"))
        
        print(f"📊 Transaction data shape: {trans.shape}")
        print(f"📊 Identity data shape: {iden.shape}")
        
        return trans, iden
    
    def merge_data(self, trans, iden):
        """Merge transaction and identity data."""
        print("🔗 Merging transaction and identity data...")
        
        # Merge on TransactionID
        df = trans.merge(iden, on="TransactionID", how="left")
        
        print(f"📊 Merged data shape: {df.shape}")
        return df
    
    def remove_unnecessary_columns(self, df):
        """Remove columns that are not useful for modeling."""
        print("🗑️ Removing unnecessary columns...")
        
        # Columns to drop (not useful for modeling)
        drop_cols = [
            "TransactionID", "TransactionDT", "ProductCD", 
            "DeviceType", "DeviceInfo", "id_30", "id_31", 
            "id_33", "id_34", "id_35", "id_36", "id_37", "id_38"
        ]
        
        # Only drop columns that exist
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        df = df.drop(columns=existing_drop_cols)
        
        self.cleaning_stats['dropped_columns'] = existing_drop_cols
        print(f"🗑️ Dropped {len(existing_drop_cols)} columns")
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        print("🔍 Handling missing values...")
        
        # Calculate missing value percentages
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
        
        # Remove columns with >50% missing values
        df = df.drop(columns=high_missing_cols)
        
        # Fill remaining missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric columns with median
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Fill categorical columns with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        self.cleaning_stats['high_missing_columns'] = high_missing_cols
        self.cleaning_stats['missing_values_filled'] = {
            'numeric_median': len(numeric_cols),
            'categorical_mode': len(categorical_cols)
        }
        
        print(f"🔍 Removed {len(high_missing_cols)} columns with >50% missing values")
        print(f"🔍 Filled missing values in {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
        
        return df
    
    def handle_outliers(self, df):
        """Handle outliers in numeric columns using IQR method."""
        print("📊 Handling outliers...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_stats = {}
        
        for col in numeric_cols:
            if col == 'isFraud':  # Skip target variable
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers > 0:
                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                outlier_stats[col] = {
                    'outliers_count': outliers,
                    'outliers_pct': (outliers / len(df)) * 100
                }
        
        self.cleaning_stats['outliers_handled'] = outlier_stats
        print(f"📊 Handled outliers in {len(outlier_stats)} columns")
        
        return df
    
    def encode_categorical_variables(self, df):
        """Encode categorical variables using LabelEncoder."""
        print("🔤 Encoding categorical variables...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        self.cleaning_stats['categorical_columns_encoded'] = list(categorical_cols)
        print(f"🔤 Encoded {len(categorical_cols)} categorical columns")
        
        return df
    
    def save_cleaned_data(self, df, suffix=""):
        """Save cleaned data to the cleaned directory."""
        print("💾 Saving cleaned data...")
        
        # Create cleaned directory if it doesn't exist
        os.makedirs(DATA_CLEANED, exist_ok=True)
        
        # Save cleaned data
        output_file = os.path.join(DATA_CLEANED, f"train_cleaned{suffix}.csv")
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
        
        # Save cleaning statistics
        stats_file = os.path.join(DATA_CLEANED, f"cleaning_stats{suffix}.json")
        with open(stats_file, 'w') as f:
            json.dump(convert_numpy_types(self.cleaning_stats), f, indent=2)
        
        # Save label encoders
        encoders_file = os.path.join(DATA_CLEANED, f"label_encoders{suffix}.json")
        encoders_dict = {}
        for col, le in self.label_encoders.items():
            encoders_dict[col] = {
                'classes': le.classes_.tolist(),
                'n_classes': len(le.classes_)
            }
        
        with open(encoders_file, 'w') as f:
            json.dump(encoders_dict, f, indent=2)
        
        print(f"💾 Saved cleaned data to {output_file}")
        print(f"💾 Saved cleaning stats to {stats_file}")
        print(f"💾 Saved label encoders to {encoders_file}")
        
        return output_file
    
    def clean_data(self, save_output=True):
        """
        Complete data cleaning pipeline.
        
        Args:
            save_output (bool): Whether to save cleaned data to disk
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        print("🧼 Starting data cleaning pipeline...")
        
        # Load raw data
        trans, iden = self.load_raw_data()
        
        # Merge data
        df = self.merge_data(trans, iden)
        
        # Clean data
        df = self.remove_unnecessary_columns(df)
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df)
        df = self.encode_categorical_variables(df)
        
        print(f"✅ Final cleaned shape: {df.shape}")
        
        # Save cleaned data if requested
        if save_output:
            self.save_cleaned_data(df)
        
        return df


def load_and_clean_data():
    """
    Legacy function for backward compatibility.
    
    Returns:
        pd.DataFrame: Cleaned and preprocessed dataset
    """
    cleaner = DataCleaner()
    return cleaner.clean_data(save_output=True) 