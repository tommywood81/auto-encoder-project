#!/usr/bin/env python3
"""
Feature Sweep Script for Fraud Detection Pipeline

This script runs the fraud detection pipeline with all available feature strategies
and compares their performance to find the best approach.
"""

import subprocess
import sys
import time
import logging
from typing import Dict, List, Tuple
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available strategies
STRATEGIES = [
    "baseline",
    "temporal", 
    "behavioural",
    "demographic_risk",
    "combined"
]

# Strategy descriptions for display
STRATEGY_DESCRIPTIONS = {
    "baseline": "Basic transaction features only (9 features)",
    "temporal": "Basic features + temporal patterns (10 features)",
    "behavioural": "Core features + amount per item (10 features)",
    "demographic_risk": "Core features + customer age risk scores (10 features)",
    "combined": "All unique features from all strategies (no duplicates)"
}

def run_pipeline(strategy: str) -> Tuple[bool, float, str]:
    """
    Run the pipeline with a specific strategy.
    
    Args:
        strategy: Feature strategy to use
        
    Returns:
        Tuple of (success, roc_auc, error_message)
    """
    logger.info(f"Starting pipeline with strategy: {strategy}")
    print(f"   📋 Strategy: {strategy}")
    
    try:
        # Run the pipeline with real-time output
        cmd = [sys.executable, "run_pipeline.py", "--strategy", strategy]
        logger.info(f"📋 Executing command: {' '.join(cmd)}")
        print(f"   Running: python run_pipeline.py --strategy {strategy}")
        print(f"   Pipeline output:")
        print("   " + "-" * 60)
        
        # Run with output capture for parsing
        result = subprocess.run(
            cmd, 
            timeout=300,  # 5 minute timeout
            text=True,
            capture_output=True
        )
        
        # Print the output in real-time style
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"   {line}")
        
        if result.stderr:
            for line in result.stderr.split('\n'):
                if line.strip():
                    print(f"   [STDERR] {line}")
        
        print("   " + "-" * 60)
        
        if result.returncode != 0:
            error_msg = f"Pipeline failed for {strategy}"
            logger.error(error_msg)
            print(f"   FAILED: {strategy}")
            return False, 0.0, error_msg
        
        # Extract ROC AUC from the output
        roc_auc = extract_roc_auc(result.stdout)
        
        # If not found in stdout, try stderr (where logging output goes)
        if roc_auc == 0.0 and result.stderr:
            roc_auc = extract_roc_auc(result.stderr)
        
        if roc_auc > 0:
            logger.info(f"Strategy {strategy} completed successfully with ROC AUC: {roc_auc:.4f}")
            print(f"   SUCCESS: {strategy} - Pipeline completed (ROC AUC: {roc_auc:.4f})")
            return True, roc_auc, "Success"
        else:
            logger.warning(f"Strategy {strategy} completed but ROC AUC not found")
            print(f"   WARNING: {strategy} - Pipeline completed but ROC AUC not found")
            return True, 0.0, "ROC AUC not found in output"
            
    except subprocess.TimeoutExpired:
        error_msg = f"Pipeline timed out for {strategy}"
        logger.error(error_msg)
        print(f"   TIMEOUT: {strategy}")
        return False, 0.0, error_msg
    except Exception as e:
        error_msg = f"Unexpected error for {strategy}: {str(e)}"
        logger.error(error_msg)
        print(f"   ERROR: {strategy} - {str(e)}")
        return False, 0.0, error_msg

def extract_roc_auc(output: str) -> float:
    """
    Extract ROC AUC value from pipeline output.
    
    Args:
        output: The stdout from the pipeline
        
    Returns:
        ROC AUC value as float, or 0.0 if not found
    """
    # Look for ROC AUC in the output
    # Pattern: "ROC AUC: 0.6291" or "Test ROC AUC: 0.6291"
    patterns = [
        r'ROC AUC:\s*([0-9]+\.[0-9]+)',
        r'Test ROC AUC:\s*([0-9]+\.[0-9]+)',
        r'roc_auc:\s*([0-9]+\.[0-9]+)',
        r'roc auc:\s*([0-9]+\.[0-9]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    return 0.0

def print_results(results: Dict[str, Tuple[bool, float, str]]):
    """Print the results in a formatted table."""
    print("\n" + "="*80)
    print("FEATURE SWEEP RESULTS")
    print("="*80)
    
    # Sort by ROC AUC (descending)
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1][1] if x[1][0] else 0.0, 
        reverse=True
    )
    
    print(f"{'Strategy':<20} {'Status':<10} {'ROC AUC':<10} {'Notes':<30}")
    print("-" * 80)
    
    for strategy, (success, roc_auc, error_msg) in sorted_results:
        status = "SUCCESS" if success else "FAILED"
        roc_str = f"{roc_auc:.4f}" if success else "N/A"
        notes = error_msg[:27] + "..." if len(error_msg) > 30 else error_msg
        
        print(f"{strategy:<20} {status:<10} {roc_str:<10} {notes:<30}")
    
    print("-" * 80)
    
    # Find best performing strategy
    successful_results = [(s, r[1]) for s, r in sorted_results if r[0]]
    
    if successful_results:
        best_strategy, best_roc = successful_results[0]
        print(f"\nBEST PERFORMING STRATEGY: {best_strategy}")
        print(f"   ROC AUC: {best_roc:.4f}")
        
        # Compare with baseline
        baseline_result = results.get("baseline")
        if baseline_result and baseline_result[0]:
            baseline_roc = baseline_result[1]
            if baseline_roc > 0:
                improvement = ((best_roc - baseline_roc) / baseline_roc) * 100
                
                if best_strategy == "baseline":
                    print(f"   Baseline is the best strategy!")
                elif improvement > 0:
                    print(f"   Improvement over baseline: +{improvement:.2f}%")
                else:
                    print(f"   Performance vs baseline: {improvement:.2f}%")
            else:
                print(f"   Baseline ROC AUC is zero - cannot calculate improvement")
        else:
            print(f"   Could not compare with baseline (baseline failed)")
    else:
        print(f"\nNo strategies completed successfully!")

def main():
    """Main function to run the feature sweep."""
    print("Starting Feature Sweep for Fraud Detection Pipeline")
    print("="*80)
    print("This will test all feature strategies with 10 epochs each")
    print("Estimated time: ~10-15 minutes")
    print("="*80)
    
    # Show what strategies will be tested
    print("\nFEATURE STRATEGIES TO TEST:")
    for i, strategy in enumerate(STRATEGIES, 1):
        print(f"   {i}. {strategy}")
    print()
    
    results = {}
    start_time = time.time()
    
    for i, strategy in enumerate(STRATEGIES, 1):
        print(f"\n{'='*20} STAGE {i}/{len(STRATEGIES)}: {strategy.upper()} {'='*20}")
        logger.info(f"Running stage {i}/{len(STRATEGIES)}: {strategy}")
        print(f"STAGE {i}/{len(STRATEGIES)}: Testing {strategy.upper()} strategy")
        print(f"   Description: {STRATEGY_DESCRIPTIONS[strategy]}")
        
        success, roc_auc, error_msg = run_pipeline(strategy)
        results[strategy] = (success, roc_auc, error_msg)
        
        # Progress update
        elapsed_time = time.time() - start_time
        avg_time_per_strategy = elapsed_time / i
        remaining_strategies = len(STRATEGIES) - i
        estimated_remaining = remaining_strategies * avg_time_per_strategy
        
        print(f"Elapsed: {elapsed_time:.1f}s | Est. remaining: {estimated_remaining:.1f}s")
        print(f"Progress: {i}/{len(STRATEGIES)} strategies completed")
        
        # Small delay between runs
        if i < len(STRATEGIES):  # Don't delay after the last one
            print("Waiting 2 seconds before next strategy...")
            time.sleep(2)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print results
    print_results(results)
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"   Total time: {total_time:.1f} seconds")
    print(f"   Strategies tested: {len(STRATEGIES)}")
    print(f"   Successful runs: {sum(1 for r in results.values() if r[0])}")
    print(f"   Failed runs: {sum(1 for r in results.values() if not r[0])}")
    print(f"   Average time per strategy: {total_time/len(STRATEGIES):.1f} seconds")
    
    print(f"\nFeature sweep completed!")
    print("="*80)

if __name__ == "__main__":
    main() 