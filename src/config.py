import os

# Data directories
DATA_DIR = "data/raw"
UNZIP_DIR = "ieee_cis"
KAGGLE_COMPETITION = "ieee-fraud-detection"

# Model parameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 30
RANDOM_STATE = 42
TEST_SIZE = 0.2
PERCENTILE_THRESHOLD = 90

# Autoencoder architecture
ENCODER_DIMS = [64, 32]
DECODER_DIMS = [64]  # Will be reversed and used with input_dim 