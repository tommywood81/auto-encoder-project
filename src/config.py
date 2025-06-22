import os

# Data pipeline directories
DATA_RAW = "data/raw"                    # Original downloaded data
DATA_CLEANED = "data/cleaned"            # After data cleaning
DATA_ENGINEERED = "data/engineered"      # After feature engineering
DATA_PROCESSED = "data/processed"        # Final processed data for modeling
DATA_INTERMEDIATE = "data/intermediate"  # Temporary files during processing

# Legacy support (keeping for backward compatibility)
DATA_DIR = DATA_RAW
UNZIP_DIR = "ieee_cis"
KAGGLE_COMPETITION = "ieee-fraud-detection"

# Model parameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 30
RANDOM_STATE = 42
TEST_SIZE = 0.2
PERCENTILE_THRESHOLD = 90  # Changed from 95 to 90 for more sensitive fraud detection

# Autoencoder architecture
ENCODER_DIMS = [64, 32]
DECODER_DIMS = [64]  # Will be reversed and used with input_dim 