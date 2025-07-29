#!/usr/bin/env python3
"""
Sweep Runner for Fraud Detection Pipeline
Config-driven three-stage hyperparameter optimization
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.sweeps.sweep_manager import SweepManager
from src.utils.data_loader import load_and_split_data, clean_data, save_cleaned_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the sweep process for hyperparameter optimization."""

    parser = argparse.ArgumentParser(description="Fraud Detection Sweep Runner")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["broad", "narrow", "final", "all"],
        default="all",
        help="Sweep stage to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (auto-determined if not specified)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/cleaned/ecommerce_cleaned.csv",
        help="Path to the cleaned data file"
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/raw/Fraudulent_E-Commerce_Transaction_Data_2.csv",
        help="Path to the raw data file (if cleaned data doesn't exist)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FRAUD DETECTION SWEEP RUNNER")
    print("=" * 80)
    print(f"Stage: {args.stage}")
    print("=" * 80)

    # Determine config file based on stage
    if args.config:
        config_path = args.config
    else:
        if args.stage == "broad":
            config_path = "configs/sweep_broad.yaml"
        elif args.stage == "narrow":
            config_path = "configs/sweep_narrow.yaml"
        elif args.stage == "final":
            config_path = "configs/sweep_final.yaml"
        else:
            config_path = "configs/sweep_broad.yaml"  # Default for "all"

    print(f"Config: {config_path}")

    # Check if cleaned data exists, if not clean raw data
    if not os.path.exists(args.data_path):
        logger.info(f"Cleaned data not found at {args.data_path}")
        if os.path.exists(args.raw_data_path):
            logger.info(f"Cleaning raw data from {args.raw_data_path}")
            df_raw = pd.read_csv(args.raw_data_path)
            df_cleaned = clean_data(df_raw)
            save_cleaned_data(df_cleaned, args.data_path)
            logger.info(f"Cleaned data saved to {args.data_path}")
        else:
            raise FileNotFoundError(f"Neither cleaned nor raw data found")

    # Initialize sweep manager
    try:
        sweep_manager = SweepManager(config_path)
        logger.info(f"Sweep manager initialized with config: {config_path}")
    except Exception as e:
        logger.error(f"Failed to initialize sweep manager: {e}")
        sys.exit(1)

    # Run appropriate stage(s)
    if args.stage == "all":
        logger.info("Running complete three-stage sweep process")
        results = sweep_manager.run_complete_sweep()
        
        print("\n" + "=" * 60)
        print("COMPLETE SWEEP PROCESS FINISHED")
        print("=" * 60)
        print(f"Best AUC: {results['best_auc']:.4f}")
        print(f"Best config saved to: configs/final_optimized_config.yaml")
        
        # Print next steps
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. Train final model:")
        print("   python main.py --mode train --config configs/final_optimized_config.yaml")
        print("\n2. Test model performance:")
        print("   python main.py --mode test --config configs/final_optimized_config.yaml")
        print("\n3. Make predictions:")
        print("   python main.py --mode predict --config configs/final_optimized_config.yaml")

    elif args.stage == "broad":
        logger.info("Running broad sweep")
        results = sweep_manager.run_broad_sweep()
        
        print("\n" + "=" * 60)
        print("BROAD SWEEP COMPLETED")
        print("=" * 60)
        print("Next command to run:")
        print("python run_sweeps.py --stage narrow --config configs/sweep_narrow.yaml")

    elif args.stage == "narrow":
        logger.info("Running narrow sweep")
        results = sweep_manager.run_narrow_sweep()
        
        print("\n" + "=" * 60)
        print("NARROW SWEEP COMPLETED")
        print("=" * 60)
        print("Next command to run:")
        print("python run_sweeps.py --stage final --config configs/sweep_final.yaml")

    elif args.stage == "final":
        logger.info("Running final sweep")
        results = sweep_manager.run_final_sweep()
        
        print("\n" + "=" * 60)
        print("FINAL SWEEP COMPLETED")
        print("=" * 60)
        print("Next command to run:")
        print("python main.py --mode train --config configs/final_optimized_config.yaml")

    print("\nSweep process completed successfully!")


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    main() 