"""
Data ingestion module for loading and preprocessing raw data.
"""

import logging
import pandas as pd
import os
from pathlib import Path
from src.config import PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


class DataIngestion:
    """Data ingestion class for the pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def ingest_data(self, input_file: str = None) -> pd.DataFrame:
        """Ingest data from a CSV file."""
        try:
            # Use default e-commerce fraud dataset if none provided
            if input_file is None:
                df = import_fraudulent_ecommerce_data()
            else:
                logger.info(f"Loading data from {input_file}")
                df = pd.read_csv(input_file)
            
            # Create ingested directory if it doesn't exist
            ingested_dir = Path("data/ingested")
            ingested_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to ingested directory
            output_file = ingested_dir / "raw_ecommerce_data.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved ingested data to {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting data: {str(e)}")
            raise


def list_available_files() -> list:
    """
    List all available files in the raw data directory.
    Returns:
        list: List of filenames in the raw data directory.
    """
    return [f.name for f in RAW_DATA_DIR.glob("*") if f.is_file()]


def load_csv_from_raw(filename: str) -> pd.DataFrame:
    """
    Load a CSV file from the raw data directory.
    Args:
        filename (str): Name of the CSV file to load.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def import_fraudulent_ecommerce_data() -> pd.DataFrame:
    """
    Import the 'Fraudulent_E-Commerce_Transaction_Data_2.csv' file from data/raw.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    filename = "Fraudulent_E-Commerce_Transaction_Data_2.csv"
    df = load_csv_from_raw(filename)
    
    # Log basic information about the dataset
    logger.info(f"E-commerce fraud dataset loaded:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Fraud distribution: {df['Is Fraudulent'].value_counts().to_dict()}")
    
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the ingested dataset.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        
    Returns:
        dict: Dictionary containing dataset information.
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'data_types': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Add target variable info if present
    if 'Is Fraudulent' in df.columns:
        info['fraud_distribution'] = df['Is Fraudulent'].value_counts().to_dict()
    
    return info


def main():
    """Main function to test data ingestion."""
    try:
        logger.info("Starting data ingestion pipeline...")
        
        # List available files
        available_files = list_available_files()
        logger.info(f"Available files in raw data directory: {available_files}")
        
        # Ingest the e-commerce fraud dataset
        df = import_fraudulent_ecommerce_data()
        
        # Get and log dataset information
        info = get_data_info(df)
        logger.info(f"Dataset information:")
        logger.info(f"  Shape: {info['shape']}")
        logger.info(f"  Memory usage: {info['memory_usage_mb']:.2f} MB")
        logger.info(f"  Data types: {info['data_types']}")
        
        if 'fraud_distribution' in info:
            logger.info(f"  Fraud distribution: {info['fraud_distribution']}")
        
        logger.info(f"Successfully ingested {len(df)} rows")
        
    except Exception as e:
        logger.error(f"Error in ingestion pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 