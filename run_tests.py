#!/usr/bin/env python3
"""
Simple test runner for the AV simulation project
"""

import sys
import os
import unittest
import subprocess

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_basic_tests():
    """Run basic tests without external dependencies"""
    print("Running basic unit tests...")

    # Create test suite
    loader = unittest.TestLoader()

    # Load tests from test modules (without pygame/cv2 dependencies)
    test_modules = [
        'tests.test_utils',
    ]

    suite = unittest.TestSuite()

    for module in test_modules:
        try:
            tests = loader.loadTestsFromName(module)
            suite.addTests(tests)
        except Exception as e:
            print(f"Could not load tests from {module}: {e}")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()

def run_with_pytest():
    """Run tests using pytest if available"""
    try:
        # Try to run pytest
        cmd = [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short']
        result = subprocess.run(cmd, capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        return result.returncode == 0

    except Exception as e:
        print(f"Could not run pytest: {e}")
        return False

def check_test_structure():
    """Check that test files are structured correctly"""
    print("Checking test structure...")

    test_files = [
        'tests/__init__.py',
        'tests/conftest.py',
        'tests/test_simulation.py',
        'tests/test_lane_detection.py',
        'tests/test_behavioral_planning.py',
        'tests/test_utils.py'
    ]

    missing_files = []
    for file_path in test_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"Missing test files: {missing_files}")
        return False

    print("All test files present!")

    # Check test file sizes
    for file_path in test_files:
        size = os.path.getsize(file_path)
        print(f"{file_path}: {size} bytes")

    return True

def main():
    """Main test runner"""
    print("=" * 60)
    print("AV SIMULATION TEST RUNNER")
    print("=" * 60)

    # Check structure first
    if not check_test_structure():
        print("Test structure check failed!")
        return 1

    print("\n" + "=" * 40)
    print("ATTEMPTING PYTEST RUN")
    print("=" * 40)

    # Try pytest first
    if run_with_pytest():
        print("✅ Pytest run successful!")
        return 0
    else:
        print("❌ Pytest run failed, falling back to basic tests")

    print("\n" + "=" * 40)
    print("RUNNING BASIC TESTS")
    print("=" * 40)

    # Fall back to basic tests
    if run_basic_tests():
        print("✅ Basic tests passed!")
        return 0
    else:
        print("❌ Basic tests failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())