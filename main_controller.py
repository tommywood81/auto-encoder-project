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
    
    def __init__(self, entity: Optional[str] = None):
        """
        Initialize the experiment controller.
        
        Args:
            entity: W&B entity (team/organization name). 
                   Defaults to 'tommywood81-fractal-dynamics' for your team.
        """
        self.entity = entity or "tommywood81-fractal-dynamics"  # Use team name by default
        self.project = "fraud-detection-autoencoder"
        self.config_loader = ConfigLoader()
        
        # Set W&B project
        os.environ["WANDB_PROJECT"] = self.project
    
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
    
    def check_feature_improvement(self) -> bool:
        """
        Check if the current feature sweep produced a new best result.
        
        Returns:
            True if current sweep produced a new best result, False otherwise
        """
        try:
            # Get the best feature AUC from current sweep
            best_feature_auc = self.get_best_feature_auc()
            
            if best_feature_auc is None:
                logger.warning("Could not determine best feature AUC - proceeding with hyperparameter tuning")
                return True
            
            # Get the best AUC from all previous runs in W&B
            api = wandb.Api()
            all_runs = api.runs(f"{self.entity}/{self.project}")
            
            best_previous_auc = 0.0
            if all_runs:
                # Get the best AUC from all runs (excluding current sweep)
                current_sweep_runs = [run for run in all_runs if "feature_sweep" in run.tags]
                other_runs = [run for run in all_runs if "feature_sweep" not in run.tags]
                
                if other_runs:
                    best_previous_auc = max([run.summary.get('final_auc', 0) for run in other_runs])
            
            # Compare current best with previous best
            if best_feature_auc > best_previous_auc:
                improvement = best_feature_auc - best_previous_auc
                logger.info(f"New best result found! Current AUC: {best_feature_auc:.4f} (improvement: +{improvement:.4f} over previous best {best_previous_auc:.4f})")
                print(f"   New best result found! Current AUC: {best_feature_auc:.4f} (improvement: +{improvement:.4f} over previous best {best_previous_auc:.4f})")
                return True
            else:
                logger.info(f"No new best result. Current AUC: {best_feature_auc:.4f} (vs previous best: {best_previous_auc:.4f})")
                print(f"   No new best result. Current AUC: {best_feature_auc:.4f} (vs previous best: {best_previous_auc:.4f})")
                return False
                
        except Exception as e:
            logger.error(f"Failed to check feature improvement: {str(e)}")
            print(f"   Could not check feature improvement - proceeding with hyperparameter tuning")
            return True
    
    def run_feature_sweep(self) -> bool:
        """
        Run feature sweep and check results.
        
        Returns:
            True if sweep completed successfully
        """
        print("Step 1: Feature Sweep")
        print("=" * 50)
        
        try:
            # Run feature sweep
            cmd = [sys.executable, "sweep_features_wandb.py"]
            if self.entity:
                cmd.extend(["--entity", self.entity])
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Feature sweep failed: {result.stderr}")
                print(f"Feature sweep failed: {result.stderr}")
                return False
            
            print("Feature sweep completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Feature sweep failed: {str(e)}")
            print(f"Feature sweep failed: {str(e)}")
            return False
    
    def run_hyperparameter_sweep(self) -> bool:
        """
        Run hyperparameter sweep.
        
        Returns:
            True if sweep completed successfully
        """
        print("\nStep 2: Hyperparameter Tuning")
        print("=" * 50)
        
        try:
            # Run hyperparameter sweep
            cmd = [sys.executable, "sweep_parameters_wandb.py"]
            if self.entity:
                cmd.extend(["--entity", self.entity])
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Hyperparameter sweep failed: {result.stderr}")
                print(f"Hyperparameter sweep failed: {result.stderr}")
                return False
            
            print("Hyperparameter sweep completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Hyperparameter sweep failed: {str(e)}")
            print(f"Hyperparameter sweep failed: {str(e)}")
            return False
    
    def run_final_training(self) -> bool:
        """
        Run final model training with multiple reconstruction thresholds.
        
        Returns:
            True if training completed successfully
        """
        print("\nStep 3: Final Model Training with Multiple Thresholds")
        print("=" * 60)
        
        # Define thresholds to test
        thresholds = [86, 89, 93]
        best_auc = 0.0
        best_threshold = None
        best_run_success = False
        
        for threshold in thresholds:
            print(f"\nTesting threshold: {threshold}%")
            print("-" * 40)
            
            try:
                # Update the final config with current threshold
                config = self.config_loader.load_config("final_config")
                config['model']['threshold'] = threshold
                
                # Save updated config temporarily
                temp_config_path = "configs/temp_final_config.yaml"
                with open(temp_config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                
                # Run final training with current threshold
                cmd = [sys.executable, "train_final_model.py"]
                if self.entity:
                    cmd.extend(["--entity", self.entity])
                
                print(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"Threshold {threshold}% training completed successfully")
                    
                    # Get the AUC from the training output
                    try:
                        # First try to get from stdout (most reliable)
                        if "Best ROC AUC:" in result.stdout:
                            import re
                            auc_match = re.search(r"Best ROC AUC:\s+([\d.]+)", result.stdout)
                            if auc_match:
                                current_auc = float(auc_match.group(1))
                                print(f"   AUC from output: {current_auc:.4f}")
                                
                                if current_auc > best_auc:
                                    best_auc = current_auc
                                    best_threshold = threshold
                                    best_run_success = True
                                    print(f"   NEW BEST! Threshold {threshold}% with AUC {current_auc:.4f}")
                                else:
                                    print(f"   Not best (current best: {best_auc:.4f})")
                            else:
                                print(f"   Could not parse AUC from output for threshold {threshold}%")
                        else:
                            # Fallback to W&B API
                            api = wandb.Api()
                            runs = api.runs(f"{self.entity}/{self.project}", filters={"tags": "final_model"})
                            
                            if runs:
                                # Get the most recent run for this threshold
                                latest_run = runs[0]  # Most recent run
                                if latest_run.summary and 'final_auc' in latest_run.summary:
                                    current_auc = latest_run.summary['final_auc']
                                    print(f"   AUC from W&B: {current_auc:.4f}")
                                    
                                    if current_auc > best_auc:
                                        best_auc = current_auc
                                        best_threshold = threshold
                                        best_run_success = True
                                        print(f"   NEW BEST! Threshold {threshold}% with AUC {current_auc:.4f}")
                                    else:
                                        print(f"   Not best (current best: {best_auc:.4f})")
                                else:
                                    print(f"   Could not retrieve AUC from W&B for threshold {threshold}%")
                            else:
                                print(f"   No final model runs found for threshold {threshold}%")
                    
                    except Exception as e:
                        logger.error(f"Failed to get AUC for threshold {threshold}%: {str(e)}")
                        print(f"   Could not retrieve AUC for threshold {threshold}%")
                
                else:
                    logger.error(f"Final training failed for threshold {threshold}%: {result.stderr}")
                    print(f"Final training failed for threshold {threshold}%: {result.stderr}")
                
                # Clean up temp config
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
                
                # Small delay between runs
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Final training failed for threshold {threshold}%: {str(e)}")
                print(f"Final training failed for threshold {threshold}%: {str(e)}")
        
        # Summary
        print(f"\nFinal Training Summary")
        print("=" * 40)
        if best_run_success:
            print(f"Best threshold: {best_threshold}%")
            print(f"Best AUC: {best_auc:.4f}")
            
            # Update final config with best threshold
            config = self.config_loader.load_config("final_config")
            config['model']['threshold'] = best_threshold
            self.config_loader.update_config("final_config", {"model": config['model']})
            print(f"Updated final config with best threshold: {best_threshold}%")
            
            return True
        else:
            print("No successful training runs found")
            return False
    
    def run_full_pipeline(self, force_hyperparam: bool = False) -> bool:
        """
        Run the complete experiment pipeline.
        
        Args:
            force_hyperparam: Force hyperparameter tuning even without improvement
            
        Returns:
            True if pipeline completed successfully
        """
        print("Starting Complete Experiment Pipeline")
        print("=" * 80)
        
        # Step 1: Feature Sweep
        if not self.run_feature_sweep():
            return False
        
        # Wait a bit for W&B to sync
        print("Waiting for W&B to sync...")
        time.sleep(10)
        
        # Step 2: Check if hyperparameter tuning is worth it
        if not force_hyperparam:
            improvement_found = self.check_feature_improvement()
            
            if not improvement_found:
                print("\nNo new best result found.")
                print("   Skipping hyperparameter tuning to save resources.")
                print("   You can force hyperparameter tuning with --force-hyperparam")
                return True
        
        # Step 2: Hyperparameter Tuning
        if not self.run_hyperparameter_sweep():
            return False
        
        # Wait a bit for W&B to sync
        print("Waiting for W&B to sync...")
        time.sleep(10)
        
        # Step 3: Final Training
        if not self.run_final_training():
            return False
        
        print("\nComplete pipeline finished successfully!")
        print("   Check W&B dashboard for detailed results")
        print("   Final model saved and ready for deployment")
        
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
    parser.add_argument("--force-hyperparam", action="store_true",
                       help="Force hyperparameter tuning even without improvement")
    parser.add_argument("--summary", action="store_true",
                       help="Show experiment summary only")
    
    args = parser.parse_args()
    
    # Initialize controller
    controller = ExperimentController(args.entity)
    
    if args.summary:
        # Show summary only
        summary = controller.get_experiment_summary()
        if summary:
            print("\nExperiment Summary")
            print("=" * 50)
            
            # Get latest runs for each stage
            api = wandb.Api()
            
            # Get feature sweep runs
            feature_runs = api.runs(f"{args.entity}/{args.project}", filters={"tags": "feature_sweep"})
            if feature_runs:
                best_feature_run = max(feature_runs, key=lambda r: r.summary.get('final_auc', 0))
                print(f"Best Feature Strategy: {best_feature_run.config.get('features', {}).get('strategy', 'Unknown')}")
                print(f"Feature AUC: {best_feature_run.summary.get('final_auc', 0):.4f}")
            
            # Get hyperparameter sweep runs
            param_runs = api.runs(f"{args.entity}/{args.project}", filters={"tags": "param_sweep"})
            if param_runs:
                best_param_run = max(param_runs, key=lambda r: r.summary.get('final_auc', 0))
                print(f"Best Hyperparameters: {best_param_run.config.get('model', {})}")
                print(f"Param AUC: {best_param_run.summary.get('final_auc', 0):.4f}")
            
            # Get final model runs
            final_runs = api.runs(f"{args.entity}/{args.project}", filters={"tags": "final_model"})
            if final_runs:
                best_final_run = max(final_runs, key=lambda r: r.summary.get('final_auc', 0))
                print(f"Final Model AUC: {best_final_run.summary.get('final_auc', 0):.4f}")
            
        else:
            print("Could not retrieve experiment summary")
    else:
        # Run full pipeline
        success = controller.run_full_pipeline(
            force_hyperparam=args.force_hyperparam
        )
        
        if success:
            print("\nPipeline completed successfully!")
        else:
            print("\nPipeline failed!")
            sys.exit(1)

if __name__ == "__main__":
    main() 