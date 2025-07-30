#!/usr/bin/env python3
"""
Comprehensive Test Runner for Fraud Detection Pipeline
Runs all tests with config-driven approach and detailed logging.
"""

import os
import sys
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import ConfigLoader


def setup_test_logging():
    """Setup comprehensive logging for test execution."""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/test_run_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def run_single_test(test_module, test_function, logger):
    """Run a single test function and return results."""
    start_time = time.time()
    
    try:
        logger.info(f"Starting test: {test_module}.{test_function}")
        
        # Import and run test
        module = __import__(f"tests.{test_module}", fromlist=[test_function])
        test_func = getattr(module, test_function)
        
        # Run the test
        result = test_func()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"✅ PASSED: {test_module}.{test_function} ({duration:.2f}s)")
        
        return {
            'test': f"{test_module}.{test_function}",
            'status': 'PASSED',
            'duration': duration,
            'error': None
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        error_msg = f"❌ FAILED: {test_module}.{test_function} - {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            'test': f"{test_module}.{test_function}",
            'status': 'FAILED',
            'duration': duration,
            'error': str(e)
        }


def run_auc_test(logger):
    """Run the AUC test specifically."""
    return run_single_test('test_auc_75', 'run_auc_test', logger)


def run_all_tests():
    """Run all tests in the test suite."""
    logger = setup_test_logging()
    
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE TEST SUITE EXECUTION")
    logger.info("=" * 80)
    
    # Load test configuration
    try:
        config_loader = ConfigLoader("tests/config/tests_config.yaml")
        test_settings = config_loader.config.get('test_settings', {})
        logger.info(f"Test configuration loaded: {test_settings}")
    except Exception as e:
        logger.error(f"Failed to load test configuration: {e}")
        return False
    
    # Define test suite
    test_suite = [
        # Core functionality tests
        ('test_auc_75', 'run_auc_test'),
        ('test_auc_75', 'test_config_driven_approach'),
        ('test_auc_75', 'test_reproducibility'),
        
        # Model quality tests
        ('test_model_reproducibility', 'test_model_reproducibility'),
        ('test_model_reproducibility', 'test_seed_enforcement'),
        ('test_model_reproducibility', 'test_deterministic_operations'),
        
        # Prediction consistency tests
        ('test_prediction_consistency', 'test_model_save_load_consistency'),
        ('test_prediction_consistency', 'test_feature_engineer_save_load_consistency'),
        ('test_prediction_consistency', 'test_prediction_stability'),
        ('test_prediction_consistency', 'test_model_persistence_integrity'),
        
        # Configuration tests
        ('test_config_consistency', 'test_config_structure'),
        ('test_config_consistency', 'test_config_values'),
        ('test_config_consistency', 'test_config_get_method'),
        
        # Data integrity tests
        ('test_no_data_leak', 'test_no_data_leakage'),
        ('test_no_data_leak', 'test_feature_engineering_fit_only'),
    ]
    
    # Track results
    results = []
    total_tests = len(test_suite)
    passed_tests = 0
    failed_tests = 0
    
    logger.info(f"Starting execution of {total_tests} tests...")
    logger.info("=" * 80)
    
    # Run each test
    for test_module, test_function in test_suite:
        result = run_single_test(test_module, test_function, logger)
        results.append(result)
        
        if result['status'] == 'PASSED':
            passed_tests += 1
        else:
            failed_tests += 1
    
    # Summary
    logger.info("=" * 80)
    logger.info("TEST EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Detailed results
    logger.info("\nDETAILED RESULTS:")
    logger.info("-" * 80)
    
    for result in results:
        status_icon = "✅" if result['status'] == 'PASSED' else "❌"
        logger.info(f"{status_icon} {result['test']} ({result['duration']:.2f}s)")
        if result['error']:
            logger.info(f"   Error: {result['error']}")
    
    # Final status
    if failed_tests == 0:
        logger.info("\n🎉 ALL TESTS PASSED! 🎉")
        logger.info("Pipeline is ready for production use.")
        return True
    else:
        logger.error(f"\n⚠️  {failed_tests} TESTS FAILED!")
        logger.error("Please review the failed tests before proceeding.")
        return False


def main():
    """Main test runner function."""
    print("=" * 80)
    print("FRAUD DETECTION PIPELINE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("This will run all tests with 3 epochs each for quick validation.")
    print("=" * 80)
    
    success = run_all_tests()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main() 