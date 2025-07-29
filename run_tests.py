#!/usr/bin/env python3
"""
Test runner for fraud detection pipeline.
Runs all comprehensive tests to ensure system quality.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_test(test_file: str) -> bool:
    """Run a single test file."""
    
    logger.info(f"Running test: {test_file}")
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, check=True)
        logger.info(f"✅ {test_file} PASSED")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {test_file} FAILED")
        logger.error(f"Error: {e.stderr}")
        return False


def main():
    """Run all tests."""
    
    print("=" * 80)
    print("FRAUD DETECTION PIPELINE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # List of tests to run
    tests = [
        "tests/test_config_consistency.py",
        "tests/test_no_data_leak.py", 
        "tests/test_model_reproducibility.py",
        "tests/test_prediction_consistency.py",
        "tests/test_auc_75.py"
    ]
    
    # Check if tests exist
    existing_tests = []
    for test in tests:
        if os.path.exists(test):
            existing_tests.append(test)
        else:
            logger.warning(f"Test file not found: {test}")
    
    if not existing_tests:
        logger.error("No test files found!")
        return False
    
    # Run tests
    passed = 0
    failed = 0
    
    for test in existing_tests:
        if run_test(test):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {len(existing_tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ System is ready for production use")
        return True
    else:
        print(f"\n❌ {failed} TEST(S) FAILED!")
        print("❌ System needs fixes before production use")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 