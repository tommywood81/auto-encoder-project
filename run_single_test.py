#!/usr/bin/env python3
"""
Single Test Runner for Fraud Detection Pipeline
Run individual tests for debugging and monitoring.
"""

import os
import sys
import time
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def run_test(test_name):
    """Run a single test by name."""
    print(f"\n{'='*60}")
    print(f"RUNNING TEST: {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        if test_name == "auc_test":
            from tests.test_auc_75 import run_auc_test
            result = run_auc_test()
            if result['success']:
                print(f"✅ AUC Test PASSED: {result['test_auc']:.4f}")
            else:
                print(f"❌ AUC Test FAILED")
            return result['success']
            
        elif test_name == "config_test":
            from tests.test_auc_75 import test_config_driven_approach
            test_config_driven_approach()
            return True
            
        elif test_name == "reproducibility_test":
            from tests.test_auc_75 import test_reproducibility
            test_reproducibility()
            return True
            
        elif test_name == "model_reproducibility":
            from tests.test_model_reproducibility import test_model_reproducibility
            test_model_reproducibility()
            return True
            
        elif test_name == "prediction_consistency":
            from tests.test_prediction_consistency import test_model_save_load_consistency
            test_model_save_load_consistency()
            return True
            
        elif test_name == "config_consistency":
            from tests.test_config_consistency import test_config_structure
            test_config_structure()
            return True
            
        elif test_name == "no_data_leak":
            from tests.test_no_data_leak import test_no_data_leakage
            test_no_data_leakage()
            return True
            
        else:
            print(f"❌ Unknown test: {test_name}")
            print("Available tests:")
            print("  - auc_test")
            print("  - config_test")
            print("  - reproducibility_test")
            print("  - model_reproducibility")
            print("  - prediction_consistency")
            print("  - config_consistency")
            print("  - no_data_leak")
            return False
            
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"⏱️  Test duration: {duration:.2f} seconds")


def main():
    """Main function to run tests."""
    if len(sys.argv) < 2:
        print("Usage: python run_single_test.py <test_name>")
        print("\nAvailable tests:")
        print("  auc_test              - Run AUC performance test (3 epochs)")
        print("  config_test           - Test config-driven approach")
        print("  reproducibility_test  - Test pipeline reproducibility")
        print("  model_reproducibility - Test model reproducibility")
        print("  prediction_consistency - Test prediction consistency")
        print("  config_consistency    - Test configuration structure")
        print("  no_data_leak          - Test for data leakage")
        print("\nExample: python run_single_test.py auc_test")
        return
    
    test_name = sys.argv[1]
    success = run_test(test_name)
    
    if success:
        print(f"\n🎉 Test '{test_name}' PASSED!")
        sys.exit(0)
    else:
        print(f"\n❌ Test '{test_name}' FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main() 