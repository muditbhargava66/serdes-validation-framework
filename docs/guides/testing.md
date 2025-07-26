# Testing Guide

The SerDes Validation Framework includes a comprehensive testing infrastructure with 91+ core tests, advanced mock implementations, and multi-protocol integration testing.

## Test Infrastructure Overview

### Test Categories

The framework organizes tests into several categories:

#### 1. Core Tests (91 tests) ✅
**Status**: All passing, 0.83s execution time
**Purpose**: Essential framework functionality validation

```bash
# Recommended command for development and CI/CD
python -m pytest tests/ -v --tb=short --ignore=tests/integration --ignore=tests/performance --ignore=tests/legacy
```

**Test Coverage**:
- **Basic Functionality**: 16 tests - Core framework operations
- **Basic Imports**: 7 tests - Import system validation
- **Certification Comprehensive**: 5 tests - Certification workflows
- **Daisy Chain Comprehensive**: 5 tests - Multi-device testing
- **Data Analysis**: 4 tests - Signal analysis algorithms
- **Framework Integration**: 5 tests - Component integration
- **Multi-Protocol**: 4 tests - Cross-protocol functionality
- **PCIe Analyzer**: 4 tests - PCIe-specific validation
- **Security Comprehensive**: 5 tests - Security testing
- **Summary Tests**: 7 tests - Framework organization
- **Tunneling Comprehensive**: 20 tests - Protocol tunneling
- **USB4 Comprehensive**: 9 tests - USB4 functionality

#### 2. Integration Tests (Partially Working)
**Multi-Protocol Integration**: 6/6 tests ✅ (Fully working)
**USB4 Integration**: 0/15 tests ❌ (Needs enhanced mocks)

```bash
# Multi-protocol integration (working)
python -m pytest tests/integration/test_multi_protocol_integration.py -v

# USB4 integration (needs work)
python -m pytest tests/integration/test_usb4_integration.py -v
```

#### 3. Performance Tests (Needs Work)
**USB4 Performance Regression**: 1/8 tests ✅

```bash
# Performance tests (partial functionality)
python -m pytest tests/performance/ -v
```

#### 4. Legacy Tests (Properly Isolated)
**Status**: All properly skipped
**Purpose**: Maintain backward compatibility while avoiding conflicts

```bash
# Legacy tests are automatically ignored
python -m pytest tests/legacy/ -v  # Will be skipped
```

## Test Execution Commands

### Recommended Commands

```bash
# Primary development command (91 tests passing)
python -m pytest tests/ -v --tb=short --ignore=tests/integration --ignore=tests/performance --ignore=tests/legacy

# Multi-protocol integration tests (6 tests passing)
python -m pytest tests/integration/test_multi_protocol_integration.py -v

# All comprehensive tests
python -m pytest tests/test_*_comprehensive.py -v

# Specific test categories
python -m pytest tests/test_basic_imports.py -v
python -m pytest tests/test_framework_integration.py -v
python -m pytest tests/test_usb4_comprehensive.py -v
```

### Full Test Suite

```bash
# Full command (91 passed, 25 failed from integration/performance)
python -m pytest tests/ -v --tb=short
```

## Mock Testing Infrastructure

### Automatic Mock Mode

The framework automatically enables mock mode through environment variables:

```python
# Set in pytest.ini
env = 
    SVF_MOCK_MODE=1
    PYTHONPATH=src
```

### Mock Implementation Features

#### Intelligent Protocol Detection
```python
class ProtocolDetector:
    def detect_protocol_from_signal(self, signal_data, sample_rate, voltage_range):
        # Enhanced mock logic based on signal characteristics
        if isinstance(signal_data, dict):
            # USB4 multi-lane format - use first lane
            signal_array = signal_data[0]['voltage']
        else:
            # Single signal array
            signal_array = signal_data
        
        signal_std = np.std(signal_array)
        signal_levels = self._estimate_signal_levels(signal_array)
        
        # PCIe: Higher voltage range (1.2V) and PAM4 levels
        if voltage_range >= 1.0 and len(signal_levels) >= 3:
            return ProtocolType.PCIE
        # Ethernet: Medium voltage range (0.8V) and PAM4 levels
        elif voltage_range == 0.8 and len(signal_levels) >= 3 and signal_std > 0.15:
            return ProtocolType.ETHERNET_224G
        # USB4: Lower voltage range (0.8V) and NRZ (2 levels)
        elif voltage_range <= 0.8 and len(signal_levels) <= 2:
            return ProtocolType.USB4
        else:
            return ProtocolType.USB4
```

#### Realistic Signal Generation
```python
def generate_pcie_signal(self, duration: float = 5e-6):
    """Generate PCIe-like signal for testing"""
    sample_rate = 200e9
    num_samples = int(duration * sample_rate)
    
    # PCIe 6.0 PAM4 signal (~32 GBaud)
    symbol_rate = 32e9
    pam4_levels = np.array([-0.6, -0.2, 0.2, 0.6])
    symbols = np.random.choice(4, num_symbols)
    
    # Generate realistic signal with noise
    voltage = np.zeros(num_samples)
    for i, symbol in enumerate(symbols):
        start_idx = int(i * symbol_period * sample_rate)
        end_idx = min(int((i + 1) * symbol_period * sample_rate), num_samples)
        if end_idx > start_idx:
            voltage[start_idx:end_idx] = pam4_levels[symbol]
    
    # Add noise
    voltage += 0.03 * np.random.randn(num_samples)
    
    return voltage, {
        'sample_rate': sample_rate,
        'voltage_range': 1.2,
        'expected_protocol': ProtocolType.PCIE,
        'symbol_rate': symbol_rate,
        'modulation': 'PAM4'
    }
```

#### Comprehensive Mock Classes
```python
# Available mock implementations
from tests.mocks.analyzer import DataAnalyzer
from tests.mocks.framework_integration import FrameworkIntegrator
from tests.mocks.pcie_analyzer import PCIeAnalyzer
from tests.mocks.multi_protocol import MultiProtocolComparator
from tests.mocks.usb4_comprehensive import USB4Validator
```

## Test Configuration

### pytest.ini Configuration
```ini
[tool:pytest]
testpaths = tests
python_paths = src
addopts = -v --tb=short --ignore=tests/legacy --ignore=tests/hardware --ignore=tests/integration --ignore=tests/performance
env = 
    SVF_MOCK_MODE=1
    PYTHONPATH=src
markers =
    hardware: marks tests as requiring hardware (deselect with '-m "not hardware"')
    visualization: marks tests as requiring visualization libraries
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
```

### Test Markers

```python
# Hardware tests (automatically skipped in mock mode)
@pytest.mark.hardware
def test_real_hardware():
    pass

# Visualization tests (conditional based on dependencies)
@pytest.mark.visualization
def test_plotting():
    pass

# Slow tests (can be excluded for fast CI)
@pytest.mark.slow
def test_performance_benchmark():
    pass

# Integration tests
@pytest.mark.integration
def test_multi_protocol_integration():
    pass
```

## Writing Tests

### Basic Test Structure

```python
import pytest
import numpy as np
from serdes_validation_framework import create_validation_framework

class TestFrameworkIntegration:
    """Framework integration tests"""
    
    def test_framework_creation(self):
        """Test framework creation and basic functionality"""
        framework = create_validation_framework()
        assert framework is not None
        assert hasattr(framework, 'detect_protocol')
    
    def test_protocol_detection(self):
        """Test protocol detection functionality"""
        framework = create_validation_framework()
        
        # Generate test signal
        signal_data = np.random.randn(1000) * 0.1
        
        protocol = framework.detect_protocol(
            signal_data=signal_data,
            sample_rate=100e9,
            voltage_range=0.8
        )
        
        assert protocol is not None
```

### Multi-Protocol Integration Tests

```python
class TestMultiProtocolIntegration:
    """Multi-protocol integration tests"""
    
    def test_protocol_detection_accuracy(self, protocol_detector):
        """Test protocol detection accuracy across different protocols"""
        test_cases = []
        
        # Generate test signals for different protocols
        pcie_signal, pcie_params = self.generate_pcie_signal()
        ethernet_signal, ethernet_params = self.generate_ethernet_signal()
        usb4_signal_data, usb4_params = self.generate_usb4_signal()
        
        # Test detection for each protocol
        detection_results = []
        for protocol_name, signal, params in test_cases:
            detected_protocol = protocol_detector.detect_protocol_from_signal(
                signal_data=signal,
                sample_rate=params['sample_rate'],
                voltage_range=params['voltage_range']
            )
            
            detection_results.append({
                'protocol_name': protocol_name,
                'expected': params['expected_protocol'],
                'detected': detected_protocol,
                'correct': detected_protocol == params['expected_protocol']
            })
        
        # Verify detection accuracy
        correct_detections = sum(1 for result in detection_results if result['correct'])
        total_detections = len(detection_results)
        accuracy = correct_detections / total_detections
        
        # Require at least 66% accuracy (2 out of 3 protocols)
        assert accuracy >= 0.66, f"Protocol detection accuracy {accuracy:.2%} below threshold"
```

### Performance Testing

```python
class TestPerformanceRegression:
    """Performance regression tests"""
    
    def test_validation_performance(self):
        """Test validation performance benchmarks"""
        framework = create_validation_framework()
        
        # Generate large signal dataset
        duration = 10e-6  # 10 μs
        sample_rate = 200e9
        num_samples = int(duration * sample_rate)
        signal_data = np.random.randn(num_samples) * 0.1
        
        start_time = time.time()
        results = framework.run_auto_validation(
            signal_data=signal_data,
            sample_rate=sample_rate,
            voltage_range=0.8
        )
        validation_time = time.time() - start_time
        
        # Performance requirements
        assert validation_time < 5.0, f"Validation took {validation_time:.2f}s, expected < 5.0s"
        assert results is not None
```

## Error Handling and Testing

### Conditional Import Testing

```python
def test_conditional_imports():
    """Test conditional import system"""
    from serdes_validation_framework.protocols.usb4 import VISUALIZATION_AVAILABLE
    
    # Test availability flags
    assert isinstance(VISUALIZATION_AVAILABLE, bool)
    
    # Test graceful degradation
    if VISUALIZATION_AVAILABLE:
        from serdes_validation_framework.protocols.usb4.visualization import USB4Visualizer
        viz = USB4Visualizer()
        assert viz is not None
    else:
        # Should handle missing dependencies gracefully
        with pytest.raises(ImportError):
            from serdes_validation_framework.protocols.usb4.visualization import USB4Visualizer
```

### Error Condition Testing

```python
def test_error_handling():
    """Test error handling with invalid inputs"""
    framework = create_validation_framework()
    
    # Test with None data
    with pytest.raises((ValueError, TypeError)):
        framework.detect_protocol(None, 100e9, 0.8)
    
    # Test with invalid parameters
    with pytest.raises((ValueError, TypeError)):
        framework.detect_protocol(np.array([1, 2, 3]), -100, 0.8)
```

## CI/CD Integration

### GitHub Actions Configuration

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
    
    - name: Run core tests
      run: |
        python -m pytest tests/ -v --tb=short --ignore=tests/integration --ignore=tests/performance --ignore=tests/legacy
    
    - name: Run integration tests
      run: |
        python -m pytest tests/integration/test_multi_protocol_integration.py -v
    
    - name: Generate coverage report
      run: |
        pip install pytest-cov
        python -m pytest tests/ --cov=serdes_validation_framework --cov-report=html
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

### Test Reporting

```python
# Generate test reports
python -m pytest tests/ --html=report.html --self-contained-html

# Generate coverage report
python -m pytest tests/ --cov=serdes_validation_framework --cov-report=html

# Performance profiling
python -m pytest tests/ --profile
```

## Debugging Tests

### Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable framework debug logging
logger = logging.getLogger('serdes_validation_framework')
logger.setLevel(logging.DEBUG)
```

### Test Isolation

```python
# Use fresh instances for each test
@pytest.fixture
def fresh_framework():
    """Provide fresh framework instance for each test"""
    return create_validation_framework()

# Clean up resources
@pytest.fixture(autouse=True)
def cleanup():
    """Automatic cleanup after each test"""
    yield
    # Cleanup code here
    import gc
    gc.collect()
```

### Mock Inspection

```python
def test_mock_behavior():
    """Inspect mock behavior for debugging"""
    from tests.mocks.analyzer import DataAnalyzer
    
    analyzer = DataAnalyzer({'test': [1, 2, 3, 4, 5]})
    
    # Check available methods
    methods = [method for method in dir(analyzer) if not method.startswith('_')]
    print(f"Available methods: {methods}")
    
    # Test method behavior
    stats = analyzer.compute_statistics('test')
    print(f"Statistics result: {stats}")
    
    assert 'mean' in stats
    assert 'std' in stats
```

## Best Practices

### 1. Test Organization
- Keep core tests fast and reliable
- Isolate integration tests that may be flaky
- Use proper test markers for categorization

### 2. Mock Implementation
- Make mocks realistic and consistent
- Handle different input formats gracefully
- Provide configurable behavior for different scenarios

### 3. Error Handling
- Test both success and failure paths
- Validate error messages and types
- Ensure graceful degradation

### 4. Performance
- Keep core tests under 1 second total execution time
- Use appropriate timeouts for integration tests
- Profile and optimize slow tests

### 5. Maintainability
- Keep tests simple and focused
- Use descriptive test names and docstrings
- Maintain test documentation

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH is set correctly
2. **Mock Mode Issues**: Verify SVF_MOCK_MODE=1 is set
3. **Test Isolation**: Use fresh fixtures for each test
4. **Performance Issues**: Check for resource leaks

### Solutions

```python
# Fix import issues
import sys
sys.path.insert(0, 'src')

# Verify mock mode
import os
assert os.environ.get('SVF_MOCK_MODE') == '1'

# Resource cleanup
@pytest.fixture(autouse=True)
def cleanup_resources():
    yield
    # Cleanup code
    import gc
    gc.collect()
```

## Next Steps

- Explore [Mock Testing Guide](../tutorials/mock_testing.md) for detailed mock usage
- Check [API Reference](../api/index.md) for complete testing API
- See [CI/CD Integration](../guides/cicd.md) for automated testing setup
- Review [Troubleshooting Guide](../guides/troubleshooting.md) for common issues