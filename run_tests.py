#!/usr/bin/env python3
"""
Test runner for fraud detection system using real data.
"""

import unittest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_all_tests():
    """Run all tests and return results."""
    print("Running Fraud Detection System Tests")
    print("=" * 50)
    print("Using real fraud detection data and models")
    print()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Overall result
    if result.wasSuccessful():
        print("\nALL TESTS PASSED!")
        return True
    else:
        print("\nSOME TESTS FAILED!")
        return False

def run_specific_test(test_name):
    """Run a specific test module."""
    print(f"Running specific test: {test_name}")
    print("=" * 50)
    
    # Import and run specific test
    test_module = f"tests.{test_name}"
    try:
        module = __import__(test_module, fromlist=[''])
        suite = unittest.TestLoader().loadTestsFromModule(module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    except ImportError as e:
        print(f"Could not import test module {test_name}: {e}")
        return False

def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # Run all tests
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 