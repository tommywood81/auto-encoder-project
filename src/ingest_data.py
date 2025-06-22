import os
import json
import subprocess
from src.config import DATA_DIR, UNZIP_DIR, KAGGLE_COMPETITION


def download_data_if_needed():
    """
    Download the IEEE-CIS fraud detection dataset from Kaggle if not already present.
    Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.
    
    NOTE: This function is currently commented out since data is already loaded in data/raw
    """
    # Data is already loaded in data/raw, so this function is not needed
    print("✅ Data already exists in data/raw, skipping download.")
    return
    
    # # Original download code (commented out)
    # if os.path.exists(os.path.join(DATA_DIR, "train_transaction.csv")):
    #     print("✅ Data already exists, skipping download.")
    #     return

    # print("📦 Downloading data from Kaggle...")
    
    # # Create Kaggle credentials directory
    # os.makedirs("/root/.kaggle", exist_ok=True)
    
    # # Write Kaggle credentials
    # with open("/root/.kaggle/kaggle.json", "w") as f:
    #     json.dump({
    #         "username": os.environ["KAGGLE_USERNAME"],
    #         "key": os.environ["KAGGLE_KEY"]
    #     }, f)
    
    # # Set proper permissions
    # os.chmod("/root/.kaggle/kaggle.json", 0o600)

    # # Download competition data
    # subprocess.run([
    #     "kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION
    # ], check=True)
    
    # # Unzip the downloaded file
    # subprocess.run([
    #     "unzip", "-q", "ieee-fraud-detection.zip", "-d", UNZIP_DIR
    # ], check=True)
    
    # # Create data directory and move files
    # os.makedirs(DATA_DIR, exist_ok=True)
    # os.rename(f"{UNZIP_DIR}/train_transaction.csv", f"{DATA_DIR}/train_transaction.csv")
    # os.rename(f"{UNZIP_DIR}/train_identity.csv", f"{DATA_DIR}/train_identity.csv")
    
    # print("✅ Data ingested and saved to disk.") 