# Testing Guide for AV Simulation Project

## Overview

This project includes comprehensive unit tests providing excellent coverage across all major components of the autonomous vehicle simulation system.

## Test Statistics

- **Total Test Classes**: 30
- **Total Test Methods**: 86
- **Coverage Level**: EXCELLENT 🟢
- **Test Files**: 5 comprehensive test modules

## Test Structure

```
tests/
├── __init__.py                     # Test package initialization
├── conftest.py                     # Pytest fixtures and configuration
├── test_simulation.py              # Core simulation tests (19 methods, 9 classes)
├── test_lane_detection.py          # Lane detection tests (22 methods, 6 classes)
├── test_behavioral_planning.py     # Behavioral planning tests (33 methods, 10 classes)
├── test_basic_structure.py         # Package structure tests (12 methods, 4 classes)
└── test_utils.py                   # Test utilities and helpers
```

## Test Categories

### 1. Core Simulation Tests (`test_simulation.py`)
- **VehicleState** - Data structure validation
- **Vehicle** - Movement, collision detection, state management
- **Action** - Enum validation
- **Environment** - Base environment functionality
- **HighwayEnvironment** - Highway-specific tests
- **MergingEnvironment** - Lane merging scenarios
- **RoundaboutEnvironment** - Roundabout navigation
- **BehaviorPlanner** - Action planning logic
- **Simulation** - Main simulation loop and controls

### 2. Lane Detection Tests (`test_lane_detection.py`)
- **LaneDetector** - Base detector functionality
- **StraightLaneDetector** - Hough line transform testing
- **CurvedLaneDetector** - Advanced curve detection algorithms
- **LaneDetectionDemo** - Interactive demo functionality
- **Integration Tests** - End-to-end lane detection pipeline
- **Error Handling** - Invalid input and edge cases

### 3. Behavioral Planning Tests (`test_behavioral_planning.py`)
- **VehicleAction** - Action enum validation
- **VehicleState** - Planning state representation
- **DynamicsModel** - Neural network dynamics modeling
- **ExperienceBuffer** - Reinforcement learning experience storage
- **ModelBasedRL** - Model-based reinforcement learning
- **CrossEntropyMethod** - Optimization algorithm testing
- **RobustControl** - Uncertainty-aware control
- **POMDP** - Partially observable decision processes
- **Integration** - Complete planning pipeline
- **Error Handling** - Invalid parameters and edge cases

### 4. Structure Tests (`test_basic_structure.py`)
- **Package Structure** - Import validation and file organization
- **Code Quality** - Syntax validation and docstring presence
- **Test Coverage** - Ensuring comprehensive test coverage
- **Configuration** - Setup and configuration file validation

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements-test.txt
```

### Running All Tests

Using pytest (recommended):
```bash
pytest tests/ -v
```

With coverage reporting:
```bash
pytest tests/ --cov=src/av_simulation --cov-report=html --cov-report=term-missing
```

### Running Specific Test Categories

Core simulation tests only:
```bash
pytest tests/test_simulation.py -v
```

Lane detection tests only:
```bash
pytest tests/test_lane_detection.py -v
```

Behavioral planning tests only:
```bash
pytest tests/test_behavioral_planning.py -v
```

### Running Tests by Markers

Run unit tests only:
```bash
pytest -m "unit" -v
```

Run integration tests only:
```bash
pytest -m "integration" -v
```

Skip slow tests:
```bash
pytest -m "not slow" -v
```

### Alternative Test Runners

Using unittest directly:
```bash
python -m unittest discover tests/ -v
```

Using the custom test runner:
```bash
python run_tests.py
```

Generate test summary:
```bash
python test_summary.py
```

## Test Configuration

### pytest.ini
- Configures test discovery patterns
- Sets coverage thresholds (80% minimum)
- Defines test markers (unit, integration, slow, gpu)
- Configures output format and warnings

### tox.ini
- Multi-environment testing (Python 3.8-3.11)
- Linting and code formatting checks
- Automated coverage reporting

### conftest.py
- Shared fixtures for all tests
- Mock objects for external dependencies
- Test data factories and utilities

## Key Testing Features

### 1. Comprehensive Mocking
- **pygame** - Mocked for headless testing
- **OpenCV** - Mocked computer vision functions
- **Neural networks** - Controlled torch operations
- **File I/O** - Temporary file fixtures

### 2. Test Data Generation
- Synthetic road images for lane detection
- Vehicle trajectory simulation
- Multi-vehicle scenario creation
- Performance benchmarking datasets

### 3. Error Handling Validation
- Invalid input parameter testing
- Edge case scenario coverage
- Exception handling verification
- Boundary condition testing

### 4. Performance Testing
- Execution time benchmarking
- Memory usage validation
- Algorithm efficiency verification
- Scalability testing

## Coverage Goals

- **Line Coverage**: >80% (enforced by pytest-cov)
- **Branch Coverage**: Comprehensive decision path testing
- **Integration Coverage**: Component interaction testing
- **Error Coverage**: Exception and edge case handling

## Adding New Tests

### Test File Naming
- Use `test_*.py` pattern
- Mirror source module structure
- Group related functionality

### Test Method Naming
- Use `test_*` prefix
- Descriptive action-based names
- Include expected behavior

### Test Organization
- One test class per major component
- setUp/tearDown for test isolation
- Meaningful assertions with custom messages

### Example Test Structure

```python
class TestNewComponent(unittest.TestCase):
    """Test NewComponent functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.component = NewComponent()

    def test_component_creation(self):
        """Test component can be created with valid parameters"""
        self.assertIsNotNone(self.component)
        self.assertEqual(self.component.state, "initialized")

    def test_component_method_with_valid_input(self):
        """Test component method with valid input produces expected output"""
        result = self.component.process(valid_input)
        self.assertEqual(result, expected_output)

    def test_component_method_with_invalid_input(self):
        """Test component method handles invalid input gracefully"""
        with self.assertRaises(ValueError):
            self.component.process(invalid_input)
```

## Continuous Integration

The test suite is designed to run in CI/CD environments:

- **Headless execution** - No GUI dependencies required
- **Deterministic results** - Fixed random seeds for reproducibility
- **Fast execution** - Optimized for automated testing
- **Clear reporting** - Detailed failure messages and coverage reports

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed and src/ is in PYTHONPATH

**Display Errors**: Run tests in headless mode or with mocked display functions

**Timeout Issues**: Use pytest-timeout for long-running tests

**Memory Issues**: Run tests with --maxfail=1 to stop on first failure

### Debug Mode

Run tests with additional debugging:
```bash
pytest tests/ -v -s --tb=long --pdb
```

## Contributing

When adding new functionality:

1. Write tests first (TDD approach)
2. Ensure >80% coverage for new code
3. Include both positive and negative test cases
4. Add integration tests for component interactions
5. Update this documentation as needed

## Summary

This test suite provides **excellent coverage** with 86 test methods across 30 test classes, ensuring the reliability and robustness of the autonomous vehicle simulation system. The comprehensive testing approach covers unit tests, integration tests, error handling, and performance validation across all major components.