# Mock Testing Examples

This document provides comprehensive examples for using the mock testing framework in the SerDes Validation Framework.

## Overview

The mock testing framework allows you to develop, test, and validate your code without requiring physical hardware. This is particularly useful for:

- Development environments
- CI/CD pipelines
- Unit testing
- Integration testing
- Demonstration purposes

## Basic Mock Mode Usage

### Example 1: Enabling Mock Mode

```python
import os

# Method 1: Environment variable
os.environ['SVF_MOCK_MODE'] = '1'

# Method 2: Programmatic configuration
from serdes_validation_framework.config import set_mock_mode
set_mock_mode(True)

# Verify mock mode is enabled
from serdes_validation_framework.config import is_mock_mode_enabled
print(f"Mock mode enabled: {is_mock_mode_enabled()}")
```

### Example 2: Basic Mock Validation

```python
# With mock mode enabled, all classes use mock implementations
from serdes_validation_framework.protocols.usb4 import USB4Validator

validator = USB4Validator()

# Generate mock signal data
mock_signal = validator.generate_mock_signal_data(
    duration=1e-6,  # 1 microsecond
    sample_rate=100e9,  # 100 GSa/s
    noise_level=0.01
)

# Run validation (uses mock implementations)
results = validator.validate_compliance(mock_signal)

print(f"Mock validation completed with {len(results)} results")
for result in results[:5]:  # Show first 5 results
    print(f"  {result.test_name}: {'PASS' if result.status else 'FAIL'}")
```

## Mock Data Generation

### Example 3: Custom Mock Signal Generation

```python
from serdes_validation_framework.mocks import MockSignalGenerator
from serdes_validation_framework.protocols.usb4.constants import USB4SignalMode

# Create mock signal generator
generator = MockSignalGenerator()

# Generate realistic USB4 signal
usb4_signal = generator.generate_usb4_signal(
    mode=USB4SignalMode.GEN3_X2,
    duration=2e-6,  # 2 microseconds
    sample_rate=200e9,  # 200 GSa/s
    lane_skew_ps=5.0,  # 5 ps lane skew
    jitter_rms=0.02,   # 2% RMS jitter
    noise_amplitude=0.05
)

print(f"Generated signal with {len(usb4_signal.lane0_data)} samples")
print(f"Lane skew: {usb4_signal.lane_skew_ps:.1f} ps")
```

### Example 4: Mock Eye Diagram Data

```python
from serdes_validation_framework.mocks import MockEyeDiagramGenerator

# Generate mock eye diagram data
eye_generator = MockEyeDiagramGenerator()

eye_data = eye_generator.generate_eye_diagram(
    signal_mode=USB4SignalMode.GEN3_X2,
    eye_height=0.8,     # 80% of ideal
    eye_width=0.75,     # 75% of ideal
    jitter_rms=0.015,   # 1.5% RMS jitter
    noise_level=0.02
)

print(f"Mock eye diagram:")
print(f"  Height: {eye_data.eye_height:.3f}")
print(f"  Width: {eye_data.eye_width:.3f}")
print(f"  Quality Factor: {eye_data.quality_factor:.2f}")
```

## Mock Testing Patterns

### Example 5: Unit Testing with Mocks

```python
import unittest
from unittest.mock import patch
import os

class TestUSB4Validation(unittest.TestCase):
    
    def setUp(self):
        """Set up mock mode for each test"""
        os.environ['SVF_MOCK_MODE'] = '1'
    
    def test_compliance_validation(self):
        """Test USB4 compliance validation"""
        from serdes_validation_framework.protocols.usb4 import USB4ComplianceValidator
        
        validator = USB4ComplianceValidator()
        
        # Generate mock signal data
        signal_data = validator.generate_mock_signal_data()
        
        # Run compliance tests
        results = validator.validate_full_compliance(signal_data)
        
        # Assertions
        self.assertGreater(len(results), 0, "Should have test results")
        
        # In mock mode, most tests should pass
        pass_rate = sum(1 for r in results if r.status) / len(results)
        self.assertGreater(pass_rate, 0.8, "Mock tests should have high pass rate")
    
    def test_signal_analysis(self):
        """Test signal analysis functionality"""
        from serdes_validation_framework.protocols.usb4 import USB4SignalAnalyzer
        
        analyzer = USB4SignalAnalyzer()
        signal_data = analyzer.generate_mock_signal_data()
        
        # Test eye diagram analysis
        eye_results = analyzer.analyze_eye_diagram(signal_data)
        self.assertIsNotNone(eye_results)
        self.assertGreater(eye_results.eye_height, 0)
        self.assertGreater(eye_results.eye_width, 0)
        
        # Test jitter analysis
        jitter_results = analyzer.analyze_jitter(signal_data)
        self.assertIsNotNone(jitter_results)
        self.assertGreater(jitter_results.rms_jitter, 0)

if __name__ == '__main__':
    unittest.main()
```

### Example 6: Integration Testing

```python
def test_end_to_end_usb4_workflow():
    """Test complete USB4 validation workflow in mock mode"""
    
    # Enable mock mode
    os.environ['SVF_MOCK_MODE'] = '1'
    
    # Import all required modules
    from serdes_validation_framework.protocols.usb4 import (
        USB4Validator,
        USB4SignalAnalyzer,
        USB4ComplianceValidator
    )
    from serdes_validation_framework.reporting import USB4TestReporter
    from serdes_validation_framework.visualization import USB4Visualizer
    
    # Step 1: Generate mock signal data
    validator = USB4Validator()
    signal_data = validator.generate_mock_signal_data(
        mode=USB4SignalMode.GEN3_X2
    )
    
    # Step 2: Analyze signal
    analyzer = USB4SignalAnalyzer()
    eye_results = analyzer.analyze_eye_diagram(signal_data)
    skew_results = analyzer.analyze_lane_skew(signal_data)
    
    # Step 3: Run compliance tests
    compliance_validator = USB4ComplianceValidator()
    compliance_results = compliance_validator.validate_full_compliance(signal_data)
    
    # Step 4: Generate visualization
    visualizer = USB4Visualizer()
    eye_plot, _ = visualizer.plot_eye_diagram(signal_data)
    
    # Step 5: Create report
    reporter = USB4TestReporter()
    
    session = TestSession(
        session_id="integration_test_001",
        timestamp=datetime.now(),
        test_type=ReportType.COMPLIANCE,
        signal_mode=USB4SignalMode.GEN3_X2
    )
    
    reporter.add_test_session(session)
    reporter.add_compliance_results(session.session_id, compliance_results)
    
    report_path = reporter.generate_compliance_report(session.session_id)
    
    # Verify all steps completed successfully
    assert eye_results is not None
    assert skew_results is not None
    assert len(compliance_results) > 0
    assert os.path.exists(eye_plot)
    assert os.path.exists(report_path)
    
    print("✓ End-to-end integration test passed")

# Run the test
test_end_to_end_usb4_workflow()
```

## CI/CD Integration Examples

### Example 7: GitHub Actions Workflow

```yaml
# .github/workflows/usb4-validation.yml
name: USB4 Validation Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run USB4 validation tests
      env:
        SVF_MOCK_MODE: "1"
      run: |
        python -m pytest tests/test_usb4_comprehensive.py -v
        python -m pytest tests/test_usb4_compliance.py -v
        python -m pytest tests/test_thunderbolt_certification.py -v
    
    - name: Generate test report
      env:
        SVF_MOCK_MODE: "1"
      run: |
        python scripts/generate_ci_report.py
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      with:
        name: test-results-${{ matrix.python-version }}
        path: test_reports/
```

### Example 8: CI/CD Test Script

```python
#!/usr/bin/env python3
"""
CI/CD test script for USB4 validation
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

def run_ci_tests():
    """Run comprehensive CI/CD tests"""
    
    # Ensure mock mode is enabled
    os.environ['SVF_MOCK_MODE'] = '1'
    
    # Import test modules
    from serdes_validation_framework.protocols.usb4 import (
        USB4Validator,
        USB4ComplianceValidator
    )
    from serdes_validation_framework.protocols.usb4.thunderbolt import (
        ThunderboltSecurityValidator,
        IntelCertificationSuite
    )
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'mock_mode': True,
        'tests': {}
    }
    
    # Test 1: Basic USB4 validation
    try:
        validator = USB4Validator()
        signal_data = validator.generate_mock_signal_data()
        results = validator.validate_compliance(signal_data)
        
        test_results['tests']['basic_validation'] = {
            'status': 'PASS',
            'test_count': len(results),
            'pass_rate': sum(1 for r in results if r.status) / len(results)
        }
        print("✓ Basic USB4 validation test passed")
        
    except Exception as e:
        test_results['tests']['basic_validation'] = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"✗ Basic USB4 validation test failed: {e}")
    
    # Test 2: Compliance testing
    try:
        compliance_validator = USB4ComplianceValidator()
        signal_data = compliance_validator.generate_mock_signal_data()
        compliance_results = compliance_validator.validate_full_compliance(signal_data)
        
        test_results['tests']['compliance_testing'] = {
            'status': 'PASS',
            'test_count': len(compliance_results),
            'pass_rate': sum(1 for r in compliance_results if r.status) / len(compliance_results)
        }
        print("✓ Compliance testing passed")
        
    except Exception as e:
        test_results['tests']['compliance_testing'] = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"✗ Compliance testing failed: {e}")
    
    # Test 3: Thunderbolt security
    try:
        security_validator = ThunderboltSecurityValidator()
        security_results = security_validator.validate_security_features({
            'dma_protection': True,
            'device_authentication': True
        })
        
        test_results['tests']['thunderbolt_security'] = {
            'status': 'PASS',
            'security_features': security_results
        }
        print("✓ Thunderbolt security test passed")
        
    except Exception as e:
        test_results['tests']['thunderbolt_security'] = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"✗ Thunderbolt security test failed: {e}")
    
    # Test 4: Certification suite
    try:
        cert_suite = IntelCertificationSuite()
        cert_results = cert_suite.run_pre_certification_checks()
        
        test_results['tests']['certification_suite'] = {
            'status': 'PASS',
            'pre_check_results': cert_results
        }
        print("✓ Certification suite test passed")
        
    except Exception as e:
        test_results['tests']['certification_suite'] = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"✗ Certification suite test failed: {e}")
    
    # Save results
    results_dir = Path('test_reports')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'ci_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Determine overall status
    failed_tests = [name for name, result in test_results['tests'].items() 
                   if result['status'] == 'FAIL']
    
    if failed_tests:
        print(f"\n✗ CI tests failed. Failed tests: {', '.join(failed_tests)}")
        return 1
    else:
        print("\n✓ All CI tests passed successfully!")
        return 0

if __name__ == '__main__':
    sys.exit(run_ci_tests())
```

## Mock Performance Testing

### Example 9: Performance Benchmarking with Mocks

```python
import time
from serdes_validation_framework.protocols.usb4.performance import USB4PerformanceBenchmark

def benchmark_mock_performance():
    """Benchmark mock implementation performance"""
    
    os.environ['SVF_MOCK_MODE'] = '1'
    
    benchmark = USB4PerformanceBenchmark()
    
    # Test signal generation performance
    start_time = time.time()
    
    for i in range(100):
        signal_data = benchmark.generate_mock_signal_data(
            duration=1e-6,
            sample_rate=100e9
        )
    
    generation_time = time.time() - start_time
    print(f"Mock signal generation: {generation_time:.2f}s for 100 signals")
    
    # Test validation performance
    start_time = time.time()
    
    validator = USB4Validator()
    for i in range(50):
        results = validator.validate_compliance(signal_data)
    
    validation_time = time.time() - start_time
    print(f"Mock validation: {validation_time:.2f}s for 50 validations")
    
    # Performance metrics
    print(f"Signal generation rate: {100/generation_time:.1f} signals/sec")
    print(f"Validation rate: {50/validation_time:.1f} validations/sec")

benchmark_mock_performance()
```

### Example 10: Memory Usage Testing

```python
import psutil
import os

def test_mock_memory_usage():
    """Test memory usage of mock implementations"""
    
    os.environ['SVF_MOCK_MODE'] = '1'
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create multiple validators
    validators = []
    for i in range(10):
        validator = USB4Validator()
        signal_data = validator.generate_mock_signal_data()
        results = validator.validate_compliance(signal_data)
        validators.append((validator, results))
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory
    
    print(f"Initial memory: {initial_memory:.1f} MB")
    print(f"Peak memory: {peak_memory:.1f} MB")
    print(f"Memory increase: {memory_increase:.1f} MB")
    print(f"Memory per validator: {memory_increase/10:.1f} MB")
    
    # Cleanup
    del validators
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Final memory: {final_memory:.1f} MB")

test_mock_memory_usage()
```

## Advanced Mock Scenarios

### Example 11: Custom Mock Behaviors

```python
from serdes_validation_framework.mocks import MockUSB4Validator

class CustomMockValidator(MockUSB4Validator):
    """Custom mock validator with specific behaviors"""
    
    def __init__(self, failure_rate=0.1):
        super().__init__()
        self.failure_rate = failure_rate
    
    def validate_compliance(self, signal_data, config=None):
        """Override to introduce controlled failures"""
        results = super().validate_compliance(signal_data, config)
        
        # Introduce failures based on failure rate
        import random
        for result in results:
            if random.random() < self.failure_rate:
                result.status = False
                result.measured_value *= 0.8  # Simulate poor measurement
        
        return results

# Use custom mock validator
custom_validator = CustomMockValidator(failure_rate=0.2)  # 20% failure rate
signal_data = custom_validator.generate_mock_signal_data()
results = custom_validator.validate_compliance(signal_data)

failed_tests = [r for r in results if not r.status]
print(f"Custom mock validation: {len(failed_tests)}/{len(results)} tests failed")
```

### Example 12: Mock Data Persistence

```python
import pickle
from pathlib import Path

def save_mock_data_for_replay():
    """Save mock data for consistent replay"""
    
    os.environ['SVF_MOCK_MODE'] = '1'
    
    validator = USB4Validator()
    
    # Generate and save multiple mock datasets
    mock_datasets = {}
    
    for mode in [USB4SignalMode.GEN2_X2, USB4SignalMode.GEN3_X2]:
        signal_data = validator.generate_mock_signal_data(mode=mode)
        results = validator.validate_compliance(signal_data)
        
        mock_datasets[mode.name] = {
            'signal_data': signal_data,
            'results': results,
            'timestamp': datetime.now()
        }
    
    # Save to file
    mock_data_dir = Path('mock_data')
    mock_data_dir.mkdir(exist_ok=True)
    
    with open(mock_data_dir / 'usb4_mock_datasets.pkl', 'wb') as f:
        pickle.dump(mock_datasets, f)
    
    print(f"Saved {len(mock_datasets)} mock datasets")

def load_and_replay_mock_data():
    """Load and replay saved mock data"""
    
    mock_data_dir = Path('mock_data')
    
    with open(mock_data_dir / 'usb4_mock_datasets.pkl', 'rb') as f:
        mock_datasets = pickle.load(f)
    
    for mode_name, dataset in mock_datasets.items():
        print(f"Replaying {mode_name} dataset:")
        print(f"  Signal samples: {len(dataset['signal_data'].lane0_data)}")
        print(f"  Test results: {len(dataset['results'])}")
        print(f"  Generated: {dataset['timestamp']}")

# Save and replay mock data
save_mock_data_for_replay()
load_and_replay_mock_data()
```

These examples demonstrate the comprehensive mock testing capabilities of the framework. Mock mode enables efficient development, testing, and CI/CD integration without requiring physical hardware.

For more information on mock testing, see the [Mock Testing Tutorial](mock_testing.md) and [API Documentation](../api/mock_controller.md).