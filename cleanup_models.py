#!/usr/bin/env python3
"""
Clean up old model files and keep only final_model files
"""

import os
from pathlib import Path

def cleanup_models():
    """Remove old 'best' model files and keep only 'final' model files."""
    models_dir = Path("models")
    
    # Files to remove (old "best" naming)
    files_to_remove = [
        "best_autoencoder.h5",
        "best_scaler.pkl", 
        "best_model_info.yaml"
    ]
    
    print("🧹 Cleaning up old model files...")
    
    for filename in files_to_remove:
        file_path = models_dir / filename
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✅ Removed: {filename}")
            except Exception as e:
                print(f"❌ Failed to remove {filename}: {e}")
        else:
            print(f"ℹ️  File not found: {filename}")
    
    print("\n📁 Current models folder contents:")
    for file in models_dir.iterdir():
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   {file.name} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    cleanup_models() 