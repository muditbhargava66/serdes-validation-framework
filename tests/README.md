# SerDes Validation Framework Test Suite v1.4.1

This directory contains comprehensive tests for the SerDes Validation Framework v1.4.1, covering all supported protocols, REST API functionality, Jupyter Dashboard features, and stress testing capabilities.

## 📁 Test Structure

### Core Test Files
- **`conftest.py`** - Test configuration and shared fixtures
- **`run_tests.py`** - Comprehensive test runner script

### Unit Tests
- **`test_data_analysis.py`** - Data analysis module tests
- **`test_framework_integration.py`** - Unified framework integration tests
- **`test_pcie_analyzer.py`** - PCIe 6.0 analyzer tests
- **`test_usb4_comprehensive.py`** - Comprehensive USB4/Thunderbolt 4 tests
- **`test_multi_protocol.py`** - Multi-protocol comparison tests

### 🆕 New Tests in v1.4.1
- **`test_api.py`** - Complete REST API functionality tests (15 tests)
- **`test_jupyter_dashboard.py`** - Jupyter dashboard system tests (22 tests)
- **`test_stress_testing.py`** - Loopback stress testing validation (16 tests)

### Integration Tests
- **`integration/test_multi_protocol_integration.py`** - Cross-protocol integration
- **`integration/test_usb4_integration.py`** - USB4 system integration

### Performance Tests
- **`performance/test_usb4_performance_regression.py`** - USB4 performance regression

### Legacy Tests
- Various legacy test files for backward compatibility

## 🚀 Quick Start

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Specific Test Categories
```bash
# Unit tests only
python tests/run_tests.py --category unit

# Integration tests
python tests/run_tests.py --category integration

# Performance tests
python tests/run_tests.py --category performance

# Include legacy tests (may have import issues)
python tests/run_tests.py --include-legacy
```

### Run Individual Test Files
```bash
# Using pytest directly
python -m pytest tests/test_data_analysis.py -v

# Using the test runner
python tests/run_tests.py --category unit --verbose
```

## 🧪 Test Categories

### 1. Unit Tests
**Purpose**: Test individual components and modules in isolation

**Coverage**:
- Data analysis functionality
- Framework integration features
- PCIe 6.0 analyzer capabilities
- USB4/Thunderbolt 4 comprehensive validation
- Multi-protocol comparison algorithms
- 🆕 **REST API endpoints** (all 8 endpoints tested)
- 🆕 **Jupyter dashboard functionality** (eye diagrams, waveform analysis)
- 🆕 **Stress testing systems** (loopback and USB4-specific)

**Characteristics**:
- Fast execution (< 50 seconds total for 130 tests)
- Mock mode enabled by default
- No external dependencies required
- High test coverage for core functionality
- 🆕 **API endpoint validation** with request/response testing
- 🆕 **Dashboard component testing** with mock data
- 🆕 **Stress test simulation** with progressive degradation

### 2. Integration Tests
**Purpose**: Test interaction between different components

**Coverage**:
- Multi-protocol integration workflows
- USB4 system-level integration
- Cross-protocol compatibility
- End-to-end validation scenarios

**Characteristics**:
- Medium execution time (30-120 seconds)
- Tests component interactions
- May require specific configurations
- Validates complete workflows

### 3. Performance Tests
**Purpose**: Ensure performance requirements are met

**Coverage**:
- USB4 performance regression testing
- Throughput benchmarking
- Memory usage validation
- Processing time verification

**Characteristics**:
- Longer execution time (2-10 minutes)
- Performance metrics collection
- Regression detection
- Resource usage monitoring

### 4. Legacy Tests
**Purpose**: Maintain backward compatibility

**Coverage**:
- Legacy module interfaces
- Deprecated functionality
- Migration path validation

**Characteristics**:
- May have import issues
- Maintained for compatibility
- Run separately from main test suite

## 🔧 Test Configuration

### Environment Variables
```bash
# Mock mode (enabled by default for tests)
export SVF_MOCK_MODE=1

# Logging level
export SVF_LOG_LEVEL=INFO

# Test timeout
export PYTEST_TIMEOUT=300
```

### Mock Mode
All tests run in mock mode by default, which means:
- No real hardware required
- Simulated instrument responses
- Predictable test data
- Fast execution
- Isolated testing environment

### Test Fixtures
Common test fixtures are provided in `conftest.py`:
- `sample_signal_data` - Basic NRZ signal data
- `sample_pam4_signal` - PAM4 signal data
- `sample_usb4_dual_lane` - USB4 dual-lane signal data
- `mock_instrument_controller` - Mock instrument controller

## 📊 Test Coverage

### Protocol Coverage
- ✅ **PCIe 6.0**: NRZ/PAM4 dual-mode, link training, compliance
- ✅ **224G Ethernet**: PAM4 analysis, equalization, compliance
- ✅ **USB4/Thunderbolt 4**: Dual-lane, tunneling, power management, certification
- ✅ **Multi-Protocol**: Cross-protocol comparison, detection, integration

### 🆕 New Coverage in v1.4.1
- ✅ **REST API**: All 8 endpoints with full request/response validation
- ✅ **Jupyter Dashboard**: Eye diagram analysis, waveform analysis, multi-protocol support
- ✅ **Stress Testing**: Loopback testing, progressive degradation, multi-cycle validation

### Functionality Coverage
- ✅ **Signal Analysis**: Eye diagrams, jitter analysis, BER estimation
- ✅ **Data Analysis**: Statistical analysis, visualization, processing
- ✅ **Instrument Control**: Mock controllers, real hardware interfaces
- ✅ **Test Sequences**: Automated test workflows, validation sequences
- ✅ **Framework Integration**: Unified validation, protocol detection

### Quality Metrics
- **Unit Test Coverage**: >90% for core modules
- **Integration Coverage**: Key workflows and interactions
- **Performance Coverage**: Critical performance paths
- **Error Handling**: Exception scenarios and edge cases
- 🆕 **API Test Coverage**: 100% endpoint coverage (15/15 tests passing)
- 🆕 **Dashboard Test Coverage**: Complete component testing (22/22 tests passing)
- 🆕 **Stress Test Coverage**: Full simulation testing (16/16 tests passing)
- 🆕 **Total Test Count**: 130 tests (up from ~90 in v1.4.0)

## 🎯 Running Tests in Different Modes

### Development Mode
```bash
# Quick unit tests during development
python tests/run_tests.py --category unit --verbose

# Specific test file
python -m pytest tests/test_data_analysis.py -v
```

### CI/CD Mode
```bash
# Complete test suite for CI/CD
python tests/run_tests.py --verbose

# With coverage reporting
python -m pytest tests/ --cov=src --cov-report=xml
```

### Performance Testing
```bash
# Performance regression tests
python tests/run_tests.py --category performance --verbose

# Memory profiling
python -m pytest tests/performance/ --profile
```

### Debug Mode
```bash
# Debug failing tests
python -m pytest tests/test_specific.py -v -s --tb=long

# Drop into debugger on failure
python -m pytest tests/test_specific.py --pdb
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure src is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   
   # Or use the test runner which handles this automatically
   python tests/run_tests.py
   ```

2. **Missing Dependencies**
   ```bash
   # Install test dependencies
   pip install -r requirements-dev.txt
   
   # Install framework in development mode
   pip install -e .
   ```

3. **Mock Mode Issues**
   ```bash
   # Ensure mock mode is enabled
   export SVF_MOCK_MODE=1
   
   # Check mock mode status
   python -c "import os; print('Mock mode:', os.environ.get('SVF_MOCK_MODE', 'Not set'))"
   ```

4. **Test Timeouts**
   ```bash
   # Increase timeout for slow tests
   python -m pytest tests/ --timeout=600
   ```

### Debug Information
```bash
# Verbose test output
python tests/run_tests.py --verbose

# Show test collection
python -m pytest tests/ --collect-only

# Run specific test method
python -m pytest tests/test_data_analysis.py::TestDataAnalyzer::test_compute_statistics -v
```

## 📈 Performance Expectations

### Unit Tests
- **Execution Time**: < 50 seconds total (130 tests)
- **Memory Usage**: < 150 MB peak
- **Test Count**: 130 individual tests (including 53 new v1.4.1 tests)

### Integration Tests
- **Execution Time**: 30-120 seconds total
- **Memory Usage**: < 200 MB peak
- **Test Count**: ~20-40 integration scenarios

### Performance Tests
- **Execution Time**: 2-10 minutes total
- **Memory Usage**: < 500 MB peak
- **Test Count**: ~10-20 performance benchmarks

## 🤝 Contributing Tests

### Adding New Tests
1. Follow existing test patterns and naming conventions
2. Use appropriate test fixtures from `conftest.py`
3. Include both positive and negative test cases
4. Add performance tests for critical paths
5. Update test categories in `run_tests.py`

### Test Guidelines
- **Naming**: Use descriptive test names (`test_usb4_dual_lane_skew_compensation`)
- **Structure**: Arrange-Act-Assert pattern
- **Mocking**: Use mocks for external dependencies
- **Assertions**: Clear, specific assertions with good error messages
- **Documentation**: Document complex test scenarios

### Test Template
```python
"""
Test Suite for New Module

Description of what this test suite covers.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from serdes_validation_framework.new_module import NewClass


class TestNewClass:
    """Test cases for NewClass"""
    
    @pytest.fixture
    def new_instance(self):
        """Create NewClass instance for testing"""
        return NewClass()
    
    def test_basic_functionality(self, new_instance):
        """Test basic functionality"""
        result = new_instance.basic_method()
        assert result is not None
    
    def test_error_handling(self, new_instance):
        """Test error handling"""
        with pytest.raises(ValueError):
            new_instance.method_with_validation(invalid_input)
    
    @patch('serdes_validation_framework.new_module.external_dependency')
    def test_with_mocking(self, mock_dependency, new_instance):
        """Test with external dependency mocked"""
        mock_dependency.return_value = "mocked_result"
        result = new_instance.method_using_dependency()
        assert result == "expected_result"
```

## 📄 License

All tests are provided under the same MIT license as the main framework.

---

**Happy Testing!** 🧪

For questions about testing or to report test issues, please refer to the main project documentation or open an issue on GitHub.