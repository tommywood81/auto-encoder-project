#!/usr/bin/env python3
"""
Main Controller for Fraud Detection Experiment Pipeline

This script orchestrates the entire experiment pipeline:
1. Feature sweep to find best feature strategy
2. Hyperparameter tuning with best features
3. Final model training and deployment

Uses W&B for experiment tracking and automated decision making.
"""

import wandb
import yaml
import os
import sys
import time
import logging
import subprocess
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentController:
    """Main controller for orchestrating experiments."""
    
    def __init__(self, entity: Optional[str] = None, project: str = "fraud-detection-autoencoder"):
        self.entity = entity
        self.project = project
        self.config_loader = ConfigLoader()
        
        # Set W&B project
        os.environ["WANDB_PROJECT"] = project
    
    def get_best_feature_auc(self) -> Optional[float]:
        """
        Get the best ROC AUC from feature sweep using W&B API.
        
        Returns:
            Best ROC AUC or None if no runs found
        """
        try:
            api = wandb.Api()
            runs = api.runs(f"{self.entity}/{self.project}" if self.entity else self.project)
            
            # Filter for feature sweep runs
            feature_runs = [run for run in runs if "feature_sweep" in run.tags]
            
            if not feature_runs:
                logger.warning("No feature sweep runs found in W&B")
                return None
            
            # Get best AUC
            best_auc = 0.0
            for run in feature_runs:
                if run.summary and 'final_auc' in run.summary:
                    auc = run.summary['final_auc']
                    if auc > best_auc:
                        best_auc = auc
            
            return best_auc if best_auc > 0 else None
            
        except Exception as e:
            logger.error(f"Failed to get best feature AUC from W&B: {str(e)}")
            return None
    
    def check_feature_improvement(self, baseline_threshold: float = 0.72) -> bool:
        """
        Check if feature sweep improved over baseline.
        
        Args:
            baseline_threshold: Minimum AUC improvement threshold
            
        Returns:
            True if improvement found, False otherwise
        """
        best_auc = self.get_best_feature_auc()
        
        if best_auc is None:
            logger.warning("Could not determine best feature AUC - proceeding with hyperparameter tuning")
            return True
        
        improvement = best_auc - baseline_threshold
        
        if improvement > 0:
            logger.info(f"✅ Feature improvement found! Best AUC: {best_auc:.4f} (improvement: +{improvement:.4f})")
            return True
        else:
            logger.info(f"❌ No significant feature improvement. Best AUC: {best_auc:.4f} (vs baseline: {baseline_threshold:.4f})")
            return False
    
    def run_feature_sweep(self) -> bool:
        """
        Run feature sweep and check results.
        
        Returns:
            True if sweep completed successfully
        """
        print("🍰 Step 1: Feature Sweep")
        print("=" * 50)
        
        try:
            # Run feature sweep
            cmd = [sys.executable, "sweep_features_wandb.py"]
            if self.entity:
                cmd.extend(["--entity", self.entity])
            
            print(f"🚀 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Feature sweep failed: {result.stderr}")
                print(f"❌ Feature sweep failed: {result.stderr}")
                return False
            
            print("✅ Feature sweep completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Feature sweep failed: {str(e)}")
            print(f"❌ Feature sweep failed: {str(e)}")
            return False
    
    def run_hyperparameter_sweep(self) -> bool:
        """
        Run hyperparameter sweep.
        
        Returns:
            True if sweep completed successfully
        """
        print("\n🔧 Step 2: Hyperparameter Tuning")
        print("=" * 50)
        
        try:
            # Run hyperparameter sweep
            cmd = [sys.executable, "sweep_parameters_wandb.py"]
            if self.entity:
                cmd.extend(["--entity", self.entity])
            
            print(f"🚀 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Hyperparameter sweep failed: {result.stderr}")
                print(f"❌ Hyperparameter sweep failed: {result.stderr}")
                return False
            
            print("✅ Hyperparameter sweep completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Hyperparameter sweep failed: {str(e)}")
            print(f"❌ Hyperparameter sweep failed: {str(e)}")
            return False
    
    def run_final_training(self) -> bool:
        """
        Run final model training.
        
        Returns:
            True if training completed successfully
        """
        print("\n🏆 Step 3: Final Model Training")
        print("=" * 50)
        
        try:
            # Run final training
            cmd = [sys.executable, "train_final_model.py"]
            if self.entity:
                cmd.extend(["--entity", self.entity])
            
            print(f"🚀 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Final training failed: {result.stderr}")
                print(f"❌ Final training failed: {result.stderr}")
                return False
            
            print("✅ Final training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Final training failed: {str(e)}")
            print(f"❌ Final training failed: {str(e)}")
            return False
    
    def run_full_pipeline(self, baseline_threshold: float = 0.72, force_hyperparam: bool = False) -> bool:
        """
        Run the complete experiment pipeline.
        
        Args:
            baseline_threshold: Minimum AUC improvement threshold
            force_hyperparam: Force hyperparameter tuning even without improvement
            
        Returns:
            True if pipeline completed successfully
        """
        print("🚀 Starting Complete Experiment Pipeline")
        print("=" * 80)
        
        # Step 1: Feature Sweep
        if not self.run_feature_sweep():
            return False
        
        # Wait a bit for W&B to sync
        print("⏳ Waiting for W&B to sync...")
        time.sleep(10)
        
        # Step 2: Check if hyperparameter tuning is worth it
        if not force_hyperparam:
            improvement_found = self.check_feature_improvement(baseline_threshold)
            
            if not improvement_found:
                print("\n❌ No significant feature improvement found.")
                print("   Skipping hyperparameter tuning to save resources.")
                print("   You can force hyperparameter tuning with --force-hyperparam")
                return True
        
        # Step 2: Hyperparameter Tuning
        if not self.run_hyperparameter_sweep():
            return False
        
        # Wait a bit for W&B to sync
        print("⏳ Waiting for W&B to sync...")
        time.sleep(10)
        
        # Step 3: Final Training
        if not self.run_final_training():
            return False
        
        print("\n🎉 Complete pipeline finished successfully!")
        print("   📊 Check W&B dashboard for detailed results")
        print("   🏆 Final model saved and ready for deployment")
        
        return True
    
    def get_experiment_summary(self) -> Dict:
        """
        Get summary of all experiments.
        
        Returns:
            Dictionary with experiment summary
        """
        try:
            api = wandb.Api()
            runs = api.runs(f"{self.entity}/{self.project}" if self.entity else self.project)
            
            summary = {
                "total_runs": len(runs),
                "feature_sweep_runs": 0,
                "param_sweep_runs": 0,
                "final_runs": 0,
                "best_feature_auc": 0.0,
                "best_param_auc": 0.0,
                "best_final_auc": 0.0
            }
            
            for run in runs:
                # Count runs by type
                if "feature_sweep" in run.tags:
                    summary["feature_sweep_runs"] += 1
                    if run.summary and 'final_auc' in run.summary:
                        auc = run.summary['final_auc']
                        if auc > summary["best_feature_auc"]:
                            summary["best_feature_auc"] = auc
                
                elif "param_sweep" in run.tags:
                    summary["param_sweep_runs"] += 1
                    if run.summary and 'final_auc' in run.summary:
                        auc = run.summary['final_auc']
                        if auc > summary["best_param_auc"]:
                            summary["best_param_auc"] = auc
                
                elif "final_model" in run.tags:
                    summary["final_runs"] += 1
                    if run.summary and 'final_auc' in run.summary:
                        auc = run.summary['final_auc']
                        if auc > summary["best_final_auc"]:
                            summary["best_final_auc"] = auc
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get experiment summary: {str(e)}")
            return {}

def main():
    """Main function to run the experiment controller."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Main controller for fraud detection experiments")
    parser.add_argument("--entity", type=str, help="W&B entity (username/team)")
    parser.add_argument("--project", type=str, default="fraud-detection-autoencoder", 
                       help="W&B project name")
    parser.add_argument("--baseline-threshold", type=float, default=0.72,
                       help="Minimum AUC improvement threshold")
    parser.add_argument("--force-hyperparam", action="store_true",
                       help="Force hyperparameter tuning even without improvement")
    parser.add_argument("--summary", action="store_true",
                       help="Show experiment summary only")
    
    args = parser.parse_args()
    
    # Initialize controller
    controller = ExperimentController(args.entity, args.project)
    
    if args.summary:
        # Show summary only
        summary = controller.get_experiment_summary()
        if summary:
            print("\n📊 Experiment Summary")
            print("=" * 50)
            print(f"Total runs: {summary['total_runs']}")
            print(f"Feature sweep runs: {summary['feature_sweep_runs']}")
            print(f"Parameter sweep runs: {summary['param_sweep_runs']}")
            print(f"Final runs: {summary['final_runs']}")
            print(f"Best feature AUC: {summary['best_feature_auc']:.4f}")
            print(f"Best parameter AUC: {summary['best_param_auc']:.4f}")
            print(f"Best final AUC: {summary['best_final_auc']:.4f}")
        else:
            print("❌ Could not retrieve experiment summary")
    else:
        # Run full pipeline
        success = controller.run_full_pipeline(
            baseline_threshold=args.baseline_threshold,
            force_hyperparam=args.force_hyperparam
        )
        
        if success:
            print("\n✅ Pipeline completed successfully!")
        else:
            print("\n❌ Pipeline failed!")
            sys.exit(1)

if __name__ == "__main__":
    main() 