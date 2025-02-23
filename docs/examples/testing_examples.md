# Testing Examples

## Overview

This guide demonstrates testing techniques for:
- Unit testing with mock instruments
- Integration testing with mock/real hardware
- Performance validation testing
- Type safety verification

## Basic Unit Tests

### Mock Controller Tests

```python
import unittest
from typing import Dict, Optional
import numpy as np
from serdes_validation_framework.instrument_control.mock_controller import (
    get_instrument_controller,
    MockInstrumentController
)

class TestMockController(unittest.TestCase):
    """Test mock controller functionality"""
    
    def setUp(self) -> None:
        """Set up test fixtures"""
        # Force mock mode
        os.environ['SVF_MOCK_MODE'] = '1'
        self.controller = get_instrument_controller()
        
    def test_basic_communication(self) -> None:
        """Test basic instrument communication"""
        resource = 'GPIB::1::INSTR'
        
        try:
            # Connect
            self.controller.connect_instrument(resource)
            
            # Query ID
            response = self.controller.query_instrument(resource, '*IDN?')
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            
        finally:
            # Cleanup
            self.controller.disconnect_instrument(resource)
            
    def test_custom_responses(self) -> None:
        """Test custom response configuration"""
        # Configure response
        self.controller.add_mock_response(
            'TEST:VALUE?',
            lambda: f"{np.random.normal(0, 0.1):.6f}",
            delay=0.1
        )
        
        resource = 'GPIB::1::INSTR'
        self.controller.connect_instrument(resource)
        
        try:
            # Query multiple times
            values = []
            for _ in range(5):
                response = self.controller.query_instrument(
                    resource,
                    'TEST:VALUE?'
                )
                values.append(float(response))
            
            # Check variation
            self.assertGreater(np.std(values), 0)
            
        finally:
            self.controller.disconnect_instrument(resource)
```

### Data Type Tests

```python
class TestDataValidation(unittest.TestCase):
    """Test data type validation"""
    
    def setUp(self) -> None:
        """Set up test data"""
        self.test_data = {
            'time': np.arange(1000, dtype=np.float64) / 256e9,
            'voltage': np.random.normal(0, 1, 1000).astype(np.float64)
        }
    
    def test_array_validation(self) -> None:
        """Test array type validation"""
        from serdes_validation_framework.data_analysis import validate_array
        
        # Test valid array
        valid_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        validate_array(valid_array, "test_array")
        
        # Test invalid type
        with self.assertRaises(AssertionError):
            invalid_array = np.array([1, 2, 3])  # Integer array
            validate_array(invalid_array, "test_array")
        
        # Test empty array
        with self.assertRaises(AssertionError):
            empty_array = np.array([], dtype=np.float64)
            validate_array(empty_array, "test_array")
            
    def test_parameter_validation(self) -> None:
        """Test numeric parameter validation"""
        def validate_parameter(
            value: float,
            name: str,
            min_val: Optional[float] = None,
            max_val: Optional[float] = None
        ) -> None:
            assert isinstance(value, float), \
                f"{name} must be float"
            if min_val is not None:
                assert value >= min_val, \
                    f"{name} must be >= {min_val}"
            if max_val is not None:
                assert value <= max_val, \
                    f"{name} must be <= {max_val}"
        
        # Test valid value
        validate_parameter(1.0, "test_param", 0.0, 2.0)
        
        # Test invalid type
        with self.assertRaises(AssertionError):
            validate_parameter(1, "test_param")  # Integer
        
        # Test range violation
        with self.assertRaises(AssertionError):
            validate_parameter(3.0, "test_param", 0.0, 2.0)
```

## Integration Tests

### Mock Scope Tests

```python
class TestMockScope(unittest.TestCase):
    """Test scope functionality with mock controller"""
    
    def setUp(self) -> None:
        """Set up test fixtures"""
        os.environ['SVF_MOCK_MODE'] = '1'
        self.scope = HighBandwidthScope('GPIB0::7::INSTR')
        
    def tearDown(self) -> None:
        """Clean up test fixtures"""
        self.scope.cleanup()
        
    def test_waveform_capture(self) -> None:
        """Test waveform capture functionality"""
        # Configure scope
        config = ScopeConfig(
            sampling_rate=256e9,
            bandwidth=120e9,
            timebase=5e-12,
            voltage_range=0.8
        )
        self.scope.configure_for_224g(config)
        
        # Capture waveform
        waveform = self.scope.capture_waveform(
            duration=1e-6,
            sample_rate=config.sampling_rate
        )
        
        # Validate data
        self.assertIsInstance(waveform.time, np.ndarray)
        self.assertIsInstance(waveform.voltage, np.ndarray)
        self.assertEqual(len(waveform.time), len(waveform.voltage))
        self.assertTrue(np.issubdtype(waveform.voltage.dtype, np.floating))
        
    def test_eye_measurement(self) -> None:
        """Test eye diagram measurement"""
        result = self.scope.measure_eye_diagram()
        
        # Validate results
        self.assertIsInstance(result.eye_heights, list)
        self.assertEqual(len(result.eye_heights), 3)  # PAM4 has 3 eyes
        self.assertTrue(all(isinstance(h, float) for h in result.eye_heights))
```

### Signal Analysis Tests

```python
class TestSignalAnalysis(unittest.TestCase):
    """Test signal analysis with mock data"""
    
    def setUp(self) -> None:
        """Set up test data"""
        # Generate test signal
        num_points = 10000
        time = np.arange(num_points, dtype=np.float64) / 256e9
        
        # PAM4 levels
        levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
        symbols = np.random.choice(levels, size=num_points)
        voltage = symbols + np.random.normal(0, 0.1, num_points)
        
        self.test_data = {
            'time': time,
            'voltage': voltage.astype(np.float64)
        }
        
    def test_level_analysis(self) -> None:
        """Test PAM4 level analysis"""
        analyzer = PAM4Analyzer(self.test_data)
        levels = analyzer.analyze_level_separation()
        
        # Check level count
        self.assertEqual(len(levels.level_means), 4)
        
        # Check level ordering
        sorted_levels = np.sort(levels.level_means)
        self.assertTrue(np.all(np.diff(sorted_levels) > 0))
        
        # Check uniformity
        self.assertLess(levels.uniformity, 0.2)
        
    def test_evm_calculation(self) -> None:
        """Test EVM calculation"""
        analyzer = PAM4Analyzer(self.test_data)
        evm = analyzer.calculate_evm()
        
        # Check EVM ranges
        self.assertGreater(evm.rms_evm_percent, 0)
        self.assertLess(evm.rms_evm_percent, 10)
        self.assertGreater(evm.peak_evm_percent, evm.rms_evm_percent)
```

## Performance Tests

### Mock Performance Testing

```python
class TestPerformance(unittest.TestCase):
    """Test performance with mock controller"""
    
    def setUp(self) -> None:
        """Set up test environment"""
        os.environ['SVF_MOCK_MODE'] = '1'
        self.controller = get_instrument_controller()
        
    def test_response_timing(self) -> None:
        """Test response timing"""
        import time
        
        # Configure delayed response
        delay = 0.1  # 100 ms
        self.controller.add_mock_response(
            'TEST:DELAY?',
            'response',
            delay=delay
        )
        
        resource = 'GPIB::1::INSTR'
        self.controller.connect_instrument(resource)
        
        try:
            # Measure response time
            start = time.time()
            self.controller.query_instrument(resource, 'TEST:DELAY?')
            elapsed = time.time() - start
            
            # Check timing
            self.assertGreaterEqual(elapsed, delay)
            self.assertLess(elapsed, delay * 1.5)
            
        finally:
            self.controller.disconnect_instrument(resource)
            
    def test_throughput(self) -> None:
        """Test data throughput"""
        # Configure waveform response
        num_points = 1000000
        self.controller.add_mock_response(
            ':WAVeform:DATA?',
            lambda: ','.join(['0.0'] * num_points),
            delay=0.5
        )
        
        resource = 'GPIB::1::INSTR'
        self.controller.connect_instrument(resource)
        
        try:
            # Measure transfer time
            start = time.time()
            response = self.controller.query_instrument(
                resource,
                ':WAVeform:DATA?'
            )
            elapsed = time.time() - start
            
            # Calculate throughput
            data_size = len(response.encode())
            throughput = data_size / elapsed / 1e6  # MB/s
            
            print(f"Throughput: {throughput:.1f} MB/s")
            
        finally:
            self.controller.disconnect_instrument(resource)
```

## Error Handling Tests

### Mock Error Tests

```python
class TestErrorHandling(unittest.TestCase):
    """Test error handling with mock controller"""
    
    def setUp(self) -> None:
        """Set up test environment"""
        self.controller = get_instrument_controller()
        
        # Configure high error rates
        self.controller.set_error_rates(
            connection_error_rate=0.5,
            command_error_rate=0.5
        )
        
    def test_connection_errors(self) -> None:
        """Test connection error handling"""
        attempts = 10
        failures = 0
        
        for _ in range(attempts):
            try:
                self.controller.connect_instrument('GPIB::1::INSTR')
                self.controller.disconnect_instrument('GPIB::1::INSTR')
            except Exception:
                failures += 1
        
        # Check failure rate
        failure_rate = failures / attempts
        self.assertGreater(failure_rate, 0.2)
        self.assertLess(failure_rate, 0.8)
        
    def test_command_errors(self) -> None:
        """Test command error handling"""
        resource = 'GPIB::1::INSTR'
        self.controller.connect_instrument(resource)
        
        try:
            attempts = 10
            failures = 0
            
            for _ in range(attempts):
                try:
                    self.controller.query_instrument(resource, '*IDN?')
                except Exception:
                    failures += 1
            
            # Check failure rate
            failure_rate = failures / attempts
            self.assertGreater(failure_rate, 0.2)
            self.assertLess(failure_rate, 0.8)
            
        finally:
            self.controller.disconnect_instrument(resource)
```

## Running Tests

### Test Execution

```python
def run_test_suite() -> None:
    """Run complete test suite"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(unittest.makeSuite(TestMockController))
    suite.addTests(unittest.makeSuite(TestDataValidation))
    suite.addTests(unittest.makeSuite(TestMockScope))
    suite.addTests(unittest.makeSuite(TestSignalAnalysis))
    suite.addTests(unittest.makeSuite(TestPerformance))
    suite.addTests(unittest.makeSuite(TestErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    run_test_suite()
```

## Best Practices

1. Always validate numeric types:
   ```python
   def test_numeric_validation(self) -> None:
       def validate_float(value: float) -> None:
           assert isinstance(value, float), \
               f"Value must be float, got {type(value)}"
       
       # Test cases
       validate_float(1.0)  # OK
       with self.assertRaises(AssertionError):
           validate_float(1)  # Integer not allowed
   ```

2. Use proper cleanup:
   ```python
   def setUp(self) -> None:
       self.resources = []
       
   def tearDown(self) -> None:
       for resource in self.resources:
           try:
               self.controller.disconnect_instrument(resource)
           except Exception as e:
               print(f"Cleanup error: {e}")
   ```

3. Test error conditions:
   ```python
   def test_error_handling(self) -> None:
       with self.assertRaises(ValueError):
           # Test invalid input
           self.controller.connect_instrument("")
   ```

## See Also

- [Mock Controller Examples](mock_controller_examples.md)
- [Mock Testing Tutorial](../tutorials/mock_testing.md)
- [Testing Guide](../guides/testing.md)