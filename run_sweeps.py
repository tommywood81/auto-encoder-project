#!/usr/bin/env python3
"""
Main Sweep Runner for Fraud Detection
Three-stage optimization: broad → narrow → final tuning
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.sweeps.sweep_manager import SweepManager
from src.utils.data_loader import load_cleaned_data, clean_data, save_cleaned_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete sweep process."""
    
    parser = argparse.ArgumentParser(description="Fraud Detection Sweep Runner")
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
    parser.add_argument(
        "--stage",
        type=str,
        choices=["broad", "narrow", "final", "all"],
        default="all",
        help="Which stage to run"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FRAUD DETECTION SWEEP RUNNER")
    print("=" * 80)
    
    # Check if cleaned data exists, if not clean raw data
    if not os.path.exists(args.data_path):
        logger.info(f"Cleaned data not found at {args.data_path}")
        if os.path.exists(args.raw_data_path):
            logger.info(f"Cleaning raw data from {args.raw_data_path}")
            df_raw = load_cleaned_data(args.raw_data_path)
            df_cleaned = clean_data(df_raw)
            save_cleaned_data(df_cleaned, args.data_path)
            logger.info(f"Cleaned data saved to {args.data_path}")
        else:
            raise FileNotFoundError(f"Neither cleaned nor raw data found")
    
    # Initialize sweep manager
    sweep_manager = SweepManager()
    
    if args.stage == "all":
        # Run complete three-stage sweep
        logger.info("Running complete three-stage sweep process...")
        best_result = sweep_manager.run_complete_sweep(args.data_path)
        
        print("\n" + "=" * 80)
        print("SWEEP PROCESS COMPLETED")
        print("=" * 80)
        print(f"Best AUC achieved: {best_result['test_auc']:.4f}")
        print(f"Best configuration: {best_result['name']}")
        print(f"Threshold: {best_result['threshold']:.6f}")
        
    elif args.stage == "broad":
        # Run only broad sweep
        logger.info("Running broad sweep only...")
        results = sweep_manager.run_broad_sweep(args.data_path)
        
        print("\n" + "=" * 50)
        print("BROAD SWEEP COMPLETED")
        print("=" * 50)
        for i, result in enumerate(results[:3]):
            print(f"{i+1}. {result['name']}: AUC = {result['test_auc']:.4f}")
    
    elif args.stage == "narrow":
        # Run narrow sweep (requires broad results)
        logger.info("Running narrow sweep...")
        # This would need to load previous broad results
        print("Narrow sweep requires broad sweep results. Run 'all' or 'broad' first.")
    
    elif args.stage == "final":
        # Run final tuning (requires narrow results)
        logger.info("Running final tuning...")
        # This would need to load previous narrow results
        print("Final tuning requires narrow sweep results. Run 'all' first.")
    
    print("\nSweep process completed successfully!")


if __name__ == "__main__":
    main() 