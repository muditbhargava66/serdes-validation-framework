# Mock Testing Guide

The SerDes Validation Framework includes a comprehensive mock testing infrastructure that enables testing without physical hardware. This is essential for CI/CD pipelines, development environments, and automated testing.

## Overview

Mock testing allows you to:
- Run tests in CI/CD environments without hardware
- Develop and debug validation algorithms
- Perform regression testing with 91+ core tests
- Test error conditions safely with proper isolation
- Validate framework functionality with intelligent protocol detection
- Execute multi-protocol integration tests
- Perform performance regression testing

## Enabling Mock Mode

### Environment Variable

The simplest way to enable mock mode is through an environment variable:

```bash
export SVF_MOCK_MODE=1
python your_test_script.py
```

### Programmatic Control

You can also enable mock mode programmatically:

```python
import os
os.environ['SVF_MOCK_MODE'] = '1'

# Now import and use the framework
from serdes_validation_framework import USB4Validator
```

### Test Runner Integration

The framework's test runner automatically enables mock mode:

```bash
# Run core tests (91 tests, 0.83s execution time)
python -m pytest tests/ -v --tb=short --ignore=tests/integration --ignore=tests/performance --ignore=tests/legacy

# Run multi-protocol integration tests
python -m pytest tests/integration/test_multi_protocol_integration.py -v

# Run all comprehensive tests
python -m pytest tests/test_*_comprehensive.py -v

# Run with full output (includes some integration test failures)
python -m pytest tests/ -v --tb=short
```

## Mock Architecture

The mock system is built with realistic implementations that provide:

### Realistic Data Generation
- Signal data with proper noise characteristics
- Protocol-specific timing and amplitude
- Configurable test scenarios

### Complete API Coverage
- All public methods are mocked
- Consistent return value formats
- Proper error handling

### Deterministic Results
- Reproducible test outcomes
- Configurable success/failure scenarios
- Realistic performance metrics

## Available Mock Classes

### Core Analysis Mocks

#### DataAnalyzer Mock
```python
from tests.mocks.analyzer import DataAnalyzer

# Mock data analyzer with realistic statistics
analyzer = DataAnalyzer(sample_data)
stats = analyzer.compute_statistics('signal_strength')
print(f"Mean: {stats['mean']}, Std: {stats['std']}")
```

#### SignalProcessor Mock
```python
from tests.mocks.analyzer import SignalProcessor

processor = SignalProcessor()
filtered_signal = processor.filter_signal(signal, filter_type="lowpass")
fft_result = processor.apply_fft(signal)
```

### Protocol-Specific Mocks

#### USB4 Validation Mock
```python
from tests.mocks.usb4_comprehensive import USB4Validator

validator = USB4Validator()
results = validator.validate_usb4_compliance()
print(f"Compliance Status: {results['overall_status']}")
```

#### PCIe Analyzer Mock
```python
from tests.mocks.pcie_analyzer import PCIeAnalyzer

config = PCIeConfig(mode="NRZ", sample_rate=100e9)
analyzer = PCIeAnalyzer(config)
results = analyzer.analyze_signal(test_data)
```

#### Framework Integration Mock
```python
from tests.mocks.framework_integration import FrameworkIntegrator

framework = FrameworkIntegrator()
protocols = framework.get_supported_protocols()
test_results = framework.run_test_sequence("USB4", test_params)
```

## Mock Data Generation

### Signal Data Generation

The framework provides utilities for generating realistic test signals:

```python
import numpy as np

def generate_usb4_signal(duration=5e-6, sample_rate=80e9):
    """Generate realistic USB4 dual-lane signal"""
    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples)
    
    # Generate NRZ data for both lanes
    lane0_data = np.random.choice([-0.4, 0.4], size=num_samples)
    lane1_data = np.random.choice([-0.4, 0.4], size=num_samples)
    
    # Add realistic noise
    noise_level = 0.02
    lane0_data += np.random.normal(0, noise_level, num_samples)
    lane1_data += np.random.normal(0, noise_level, num_samples)
    
    return {
        'time': time,
        'lane0_data': lane0_data,
        'lane1_data': lane1_data,
        'sample_rate': sample_rate
    }

# Use in tests
signal_data = generate_usb4_signal()
```

### Protocol-Specific Data

#### USB4 Test Data
```python
def generate_usb4_test_data():
    """Generate USB4-specific test data"""
    return {
        'signal_mode': 'Gen3x2',
        'bandwidth': 40e9,  # 40 Gbps
        'lane_count': 2,
        'power_state': 'U0',
        'tunneling_protocols': ['PCIe', 'DisplayPort', 'USB32']
    }
```

#### PCIe Test Data
```python
def generate_pcie_test_data():
    """Generate PCIe-specific test data"""
    return {
        'generation': 'Gen5',
        'lanes': 16,
        'speed': '32 GT/s',
        'signal_mode': 'PAM4',
        'link_training': True
    }
```

## Writing Mock Tests

### Basic Mock Test Structure

```python
import pytest
import os
from unittest.mock import patch

# Enable mock mode
os.environ['SVF_MOCK_MODE'] = '1'

class TestUSB4Validation:
    """Test USB4 validation with mocks"""
    
    def test_usb4_compliance_basic(self):
        """Test basic USB4 compliance validation"""
        from serdes_validation_framework.protocols.usb4 import USB4Validator
        
        validator = USB4Validator()
        signal_data = generate_usb4_signal()
        
        results = validator.validate_compliance(signal_data)
        
        assert results is not None
        assert 'overall_status' in results
        assert results['overall_status'] in ['PASS', 'FAIL']
    
    def test_usb4_tunneling(self):
        """Test USB4 tunneling validation"""
        from serdes_validation_framework.protocols.usb4 import USB4TunnelingManager
        
        manager = USB4TunnelingManager()
        tunnel_id = manager.create_tunnel('PCIe', bandwidth='16 Gbps')
        
        assert tunnel_id is not None
        
        performance = manager.test_tunnel_performance(tunnel_id)
        assert performance['status'] == 'OPTIMAL'
```

### Advanced Mock Scenarios

#### Error Condition Testing
```python
def test_error_handling():
    """Test error handling with invalid data"""
    from serdes_validation_framework.protocols.usb4 import USB4Validator
    
    validator = USB4Validator()
    
    # Test with None data
    with pytest.raises((ValueError, TypeError)):
        validator.validate_compliance(None)
    
    # Test with invalid signal data
    invalid_data = {'invalid': 'data'}
    with pytest.raises((ValueError, TypeError)):
        validator.validate_compliance(invalid_data)
```

#### Performance Testing
```python
def test_performance_benchmarks():
    """Test performance benchmarks with mocks"""
    from serdes_validation_framework.protocols.usb4 import USB4PerformanceBenchmark
    
    benchmark = USB4PerformanceBenchmark()
    signal_data = generate_usb4_signal()
    
    results = benchmark.run_performance_tests(signal_data)
    
    assert results['throughput'] > 0
    assert results['latency'] > 0
    assert results['efficiency'] > 0.8
```

#### Multi-Protocol Testing
```python
def test_multi_protocol_validation():
    """Test multi-protocol validation"""
    from tests.mocks.multi_protocol import MultiProtocolAnalyzer
    
    analyzer = MultiProtocolAnalyzer()
    
    # Detect protocols
    detection_results = analyzer.detect_protocols()
    assert 'detected_protocols' in detection_results
    
    # Run cross-protocol tests
    cross_results = analyzer.run_cross_protocol_tests()
    assert cross_results['overall_score'] > 90
```

## Mock Configuration

### Configurable Mock Behavior

You can configure mock behavior for different test scenarios:

```python
class ConfigurableMockValidator:
    """Configurable mock validator for testing"""
    
    def __init__(self, success_rate=0.95, response_time=0.1):
        self.success_rate = success_rate
        self.response_time = response_time
    
    def validate_compliance(self, signal_data):
        """Mock validation with configurable outcomes"""
        import random
        import time
        
        # Simulate processing time
        time.sleep(self.response_time)
        
        # Determine outcome based on success rate
        success = random.random() < self.success_rate
        
        return {
            'overall_status': 'PASS' if success else 'FAIL',
            'confidence': self.success_rate,
            'test_duration': self.response_time
        }

# Use in tests
validator = ConfigurableMockValidator(success_rate=0.8)
results = validator.validate_compliance(test_data)
```

### Environment-Specific Configuration

```python
def get_mock_config():
    """Get mock configuration based on environment"""
    if os.environ.get('CI') == 'true':
        # CI environment - fast, deterministic
        return {
            'response_time': 0.01,
            'success_rate': 1.0,
            'enable_noise': False
        }
    else:
        # Development environment - realistic
        return {
            'response_time': 0.1,
            'success_rate': 0.95,
            'enable_noise': True
        }
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: SerDes Framework Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run mock tests
      env:
        SVF_MOCK_MODE: 1
      run: |
        python tests/run_tests.py --category all
    
    - name: Generate test report
      run: |
        python -m pytest tests/ --html=report.html --self-contained-html
    
    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: report.html
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    
    environment {
        SVF_MOCK_MODE = '1'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install -e .'
            }
        }
        
        stage('Test') {
            steps {
                sh 'python tests/run_tests.py --category comprehensive'
            }
        }
        
        stage('Report') {
            steps {
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'test_reports',
                    reportFiles: 'index.html',
                    reportName: 'Test Report'
                ])
            }
        }
    }
}
```

## Best Practices

### 1. Realistic Mock Data
- Use realistic signal parameters
- Include appropriate noise levels
- Model actual hardware behavior

### 2. Comprehensive Coverage
- Test both success and failure scenarios
- Include edge cases and error conditions
- Validate all public API methods

### 3. Performance Considerations
- Keep mock operations fast for CI/CD
- Use deterministic results when possible
- Avoid unnecessary complexity

### 4. Maintainability
- Keep mocks synchronized with real implementations
- Use clear naming conventions
- Document mock behavior and limitations

### 5. Test Isolation
- Ensure tests don't interfere with each other
- Clean up resources properly
- Use fresh mock instances for each test

## Debugging Mock Tests

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your mock test code here
```

### Mock Inspection

```python
def inspect_mock_behavior():
    """Inspect mock behavior for debugging"""
    from tests.mocks.analyzer import DataAnalyzer
    
    analyzer = DataAnalyzer({'test': [1, 2, 3, 4, 5]})
    
    # Check available methods
    methods = [method for method in dir(analyzer) if not method.startswith('_')]
    print(f"Available methods: {methods}")
    
    # Test method behavior
    stats = analyzer.compute_statistics('test')
    print(f"Statistics result: {stats}")
```

### Validation Against Real Hardware

When possible, validate mock behavior against real hardware:

```python
def validate_mock_accuracy():
    """Compare mock results with real hardware (when available)"""
    
    # Real hardware test (when available)
    if os.environ.get('SVF_MOCK_MODE') != '1':
        real_results = run_real_hardware_test()
    
    # Mock test
    os.environ['SVF_MOCK_MODE'] = '1'
    mock_results = run_mock_test()
    
    # Compare results (structure, not values)
    assert set(real_results.keys()) == set(mock_results.keys())
    assert type(real_results['status']) == type(mock_results['status'])
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure mock mode is enabled before importing
2. **Missing Methods**: Check that all required methods are implemented in mocks
3. **Type Mismatches**: Verify mock return types match real implementations
4. **Performance Issues**: Optimize mock operations for CI/CD speed

### Solutions

```python
# Fix import issues
import os
os.environ['SVF_MOCK_MODE'] = '1'  # Set BEFORE importing

# Handle missing methods gracefully
class RobustMock:
    def __getattr__(self, name):
        def mock_method(*args, **kwargs):
            return {'status': 'MOCK', 'method': name}
        return mock_method

# Type checking
def validate_mock_types(mock_result, expected_types):
    for key, expected_type in expected_types.items():
        assert isinstance(mock_result.get(key), expected_type)
```

## Next Steps

- Explore [Test Framework Guide](../guides/testing.md) for advanced testing strategies
- Learn about [CI/CD Integration](../guides/cicd.md) for automated testing
- Check out [API Reference](../api/index.md) for complete mock coverage
- See [Troubleshooting Guide](../guides/troubleshooting.md) for common issues