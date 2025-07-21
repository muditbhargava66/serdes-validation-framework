# Testing Guide

## Overview

This guide covers:
- Unit testing best practices
- Type safety in tests 
- Mock testing patterns
- Integration testing
- Performance validation

## Type-Safe Test Design

### Input Validation

All test functions should validate numeric inputs:

```python
from typing import Dict, List, Optional, Union
import numpy as np
import numpy.typing as npt

def validate_test_data(
    data: npt.NDArray[np.float64],
    name: str = "data"
) -> None:
    """
    Validate numeric test data
    
    Args:
        data: Numeric data array
        name: Array name for error messages
        
    Raises:
        AssertionError: If validation fails
    """
    # Type validation
    assert isinstance(data, np.ndarray), \
        f"{name} must be numpy array, got {type(data)}"
    
    # Data type validation  
    assert np.issubdtype(data.dtype, np.floating), \
        f"{name} must be floating-point, got {data.dtype}"
    
    # Value validation
    assert len(data) > 0, f"{name} cannot be empty"
    assert not np.any(np.isnan(data)), f"{name} contains NaN values"
    assert not np.any(np.isinf(data)), f"{name} contains infinite values"
```

### Parameter Validation

```python
def validate_test_parameter(
    value: float,
    name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> None:
    """
    Validate numeric test parameter
    
    Args:
        value: Parameter value
        name: Parameter name
        min_value: Optional minimum value
        max_value: Optional maximum value
        
    Raises:
        AssertionError: If validation fails
    """
    # Type validation
    assert isinstance(value, float), \
        f"{name} must be float, got {type(value)}"
    
    # Range validation
    if min_value is not None:
        assert value >= min_value, \
            f"{name} must be >= {min_value}"
    
    if max_value is not None:
        assert value <= max_value, \
            f"{name} must be <= {max_value}"
```

## Unit Testing

### Test Case Structure

```python
import unittest
from typing import Dict, List, Optional

class TestMeasurements(unittest.TestCase):
    """Test measurement calculations"""
    
    def setUp(self) -> None:
        """Set up test data"""
        self.test_data = {
            'voltage': np.array([0.1, 0.2, 0.3], dtype=np.float64),
            'current': np.array([1.0, 2.0, 3.0], dtype=np.float64)
        }
    
    def test_power_calculation(self) -> None:
        """Test power calculation with validation"""
        # Validate inputs
        validate_test_data(self.test_data['voltage'], 'voltage')
        validate_test_data(self.test_data['current'], 'current')
        
        # Calculate power
        power = self.test_data['voltage'] * self.test_data['current']
        
        # Validate result
        self.assertTrue(np.issubdtype(power.dtype, np.floating))
        self.assertEqual(len(power), len(self.test_data['voltage']))
        self.assertTrue(np.all(power >= 0))
```

### Test Helper Functions

```python
class TestHelpers:
    """Type-safe test helper functions"""
    
    @staticmethod
    def generate_test_signal(
        num_points: int,
        amplitude: float,
        noise: float
    ) -> npt.NDArray[np.float64]:
        """
        Generate test signal with validation
        
        Args:
            num_points: Number of data points
            amplitude: Signal amplitude
            noise: Noise amplitude
            
        Returns:
            Test signal array
            
        Raises:
            AssertionError: If parameters are invalid
        """
        # Validate inputs
        assert isinstance(num_points, int), \
            "num_points must be integer"
        assert isinstance(amplitude, float), \
            "amplitude must be float"
        assert isinstance(noise, float), \
            "noise must be float"
        
        assert num_points > 0, "num_points must be positive"
        assert amplitude > 0, "amplitude must be positive"
        assert noise >= 0, "noise must be non-negative"
        
        # Generate signal
        time = np.arange(num_points, dtype=np.float64)
        signal = amplitude * np.sin(2 * np.pi * time / num_points)
        noise_data = np.random.normal(0, noise, num_points)
        
        return (signal + noise_data).astype(np.float64)
    
    @staticmethod
    def validate_signal_properties(
        signal: npt.NDArray[np.float64],
        expected_rms: float,
        tolerance: float
    ) -> None:
        """
        Validate signal properties
        
        Args:
            signal: Signal array to validate
            expected_rms: Expected RMS value
            tolerance: Validation tolerance
            
        Raises:
            AssertionError: If validation fails
        """
        # Validate inputs
        validate_test_data(signal, 'signal')
        validate_test_parameter(expected_rms, 'expected_rms', min_value=0)
        validate_test_parameter(tolerance, 'tolerance', min_value=0)
        
        # Calculate RMS
        rms = np.sqrt(np.mean(signal**2))
        
        # Validate within tolerance
        assert abs(rms - expected_rms) <= tolerance, \
            f"RMS {rms:.3f} differs from expected {expected_rms:.3f}"
```

## Mock Testing

### Mock Configuration

```python
from unittest.mock import MagicMock, patch
from typing import Dict, Optional

class MockTestConfig:
    """Type-safe mock test configuration"""
    
    def __init__(
        self,
        connection_error_rate: float = 0.0,
        command_error_rate: float = 0.0
    ) -> None:
        """
        Initialize mock configuration
        
        Args:
            connection_error_rate: Connection failure probability 
            command_error_rate: Command failure probability
        """
        # Validate inputs
        validate_test_parameter(
            connection_error_rate,
            'connection_error_rate',
            min_value=0.0,
            max_value=1.0
        )
        validate_test_parameter(
            command_error_rate, 
            'command_error_rate',
            min_value=0.0,
            max_value=1.0
        )
        
        self.connection_error_rate = connection_error_rate
        self.command_error_rate = command_error_rate

def configure_mock_controller(
    config: MockTestConfig
) -> MagicMock:
    """
    Configure mock controller with error simulation
    
    Args:
        config: Mock test configuration
        
    Returns:
        Configured mock controller
    """
    controller = MagicMock()
    
    def mock_connect(*args):
        if np.random.random() < config.connection_error_rate:
            raise ConnectionError("Simulated connection failure")
    
    def mock_command(*args):
        if np.random.random() < config.command_error_rate:
            raise RuntimeError("Simulated command failure")
            
    controller.connect.side_effect = mock_connect
    controller.send_command.side_effect = mock_command
    
    return controller
```

### Mock Response Generation

```python
def configure_mock_responses(
    controller: MagicMock,
    base_value: float,
    noise_amplitude: float
) -> None:
    """
    Configure mock measurement responses
    
    Args:
        controller: Mock controller instance
        base_value: Base measurement value
        noise_amplitude: Response noise amplitude
    """
    # Validate inputs
    validate_test_parameter(base_value, 'base_value')
    validate_test_parameter(
        noise_amplitude,
        'noise_amplitude',
        min_value=0
    )
    
    def mock_measure():
        noise = np.random.normal(0, noise_amplitude)
        return f"{base_value + noise:.6f}"
    
    # Configure responses
    controller.query.side_effect = mock_measure
```

## Integration Testing

### Test Fixtures

```python
@pytest.fixture
def test_configuration() -> Dict[str, float]:
    """Provide validated test configuration"""
    config = {
        'sample_rate': 256e9,
        'bandwidth': 120e9,
        'timebase': 5e-12,
        'voltage_range': 0.8
    }
    
    # Validate all parameters
    for name, value in config.items():
        validate_test_parameter(value, name, min_value=0)
    
    return config

@pytest.fixture
def test_data() -> Dict[str, npt.NDArray[np.float64]]:
    """Provide validated test data"""
    # Generate data
    num_points = 1000
    time = np.arange(num_points, dtype=np.float64) / 256e9
    voltage = TestHelpers.generate_test_signal(
        num_points=num_points,
        amplitude=0.5,
        noise=0.05
    )
    
    data = {
        'time': time,
        'voltage': voltage
    }
    
    # Validate arrays
    for name, array in data.items():
        validate_test_data(array, name)
    
    return data
```

### Integration Tests

```python
def test_measurement_chain(
    test_configuration: Dict[str, float],
    test_data: Dict[str, npt.NDArray[np.float64]]
) -> None:
    """
    Test complete measurement chain
    
    Args:
        test_configuration: Test configuration parameters
        test_data: Test signal data
    """
    try:
        # Initialize components
        scope = configure_scope(test_configuration)
        analyzer = configure_analyzer(test_data)
        
        # Run measurement chain
        waveform = scope.capture_waveform(
            duration=1e-6,
            sample_rate=test_configuration['sample_rate']
        )
        
        results = analyzer.analyze_signal(waveform)
        
        # Validate results
        assert isinstance(results, dict), \
            "Results must be dictionary"
        
        for name, value in results.items():
            assert isinstance(value, float), \
                f"Result {name} must be float"
            assert not np.isnan(value), \
                f"Result {name} is NaN"
            assert not np.isinf(value), \
                f"Result {name} is infinite"
        
    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")
```

## Performance Testing

### Response Time Tests

```python
def measure_response_time(
    controller: Any,
    num_iterations: int = 100
) -> Dict[str, float]:
    """
    Measure command response times
    
    Args:
        controller: Controller instance
        num_iterations: Number of test iterations
        
    Returns:
        Dictionary of timing statistics
    """
    # Validate input
    assert isinstance(num_iterations, int), \
        "num_iterations must be integer"
    assert num_iterations > 0, \
        "num_iterations must be positive"
    
    response_times = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        controller.query_instrument('GPIB::1::INSTR', '*IDN?')
        response_times.append(time.time() - start_time)
    
    # Convert to float array
    times = np.array(response_times, dtype=np.float64)
    
    return {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'min_ms': float(np.min(times) * 1000),
        'max_ms': float(np.max(times) * 1000)
    }
```

### Resource Usage Tests

```python
def monitor_resource_usage(
    test_function: Callable,
    duration: float
) -> Dict[str, float]:
    """
    Monitor resource usage during test
    
    Args:
        test_function: Test function to monitor
        duration: Test duration in seconds
        
    Returns:
        Dictionary of resource metrics
    """
    # Validate inputs
    assert callable(test_function), \
        "test_function must be callable"
    validate_test_parameter(duration, 'duration', min_value=0)
    
    import psutil
    import threading
    
    # Initialize metrics
    metrics = {
        'cpu_percent': [],
        'memory_mb': [],
        'io_counters': []
    }
    
    # Monitoring thread
    def monitor():
        start_time = time.time()
        while time.time() - start_time < duration:
            metrics['cpu_percent'].append(
                float(psutil.cpu_percent())
            )
            metrics['memory_mb'].append(
                float(psutil.Process().memory_info().rss / 1e6)
            )
            metrics['io_counters'].append(
                float(psutil.Process().io_counters().read_bytes / 1e6)
            )
            time.sleep(0.1)
    
    # Run monitoring
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()
    
    # Run test function
    test_function()
    
    # Wait for monitoring
    monitor_thread.join()
    
    # Calculate statistics
    results = {}
    for metric, values in metrics.items():
        array = np.array(values, dtype=np.float64)
        results[f"{metric}_mean"] = float(np.mean(array))
        results[f"{metric}_max"] = float(np.max(array))
    
    return results
```

## Best Practices

### 1. Type Validation

Always validate numeric types:
```python
assert isinstance(value, float), f"Expected float, got {type(value)}"
```

### 2. Test Organization

Use clear test case structure:
```python
class TestModule(unittest.TestCase):
    """Test module functionality"""
    
    def setUp(self) -> None:
        """Set up test fixtures"""
        pass
        
    def tearDown(self) -> None:
        """Clean up test fixtures"""
        pass
```

### 3. Error Handling

Use proper error hierarchy:
```python
class TestError(Exception):
    """Base test error"""
    pass

class ValidationError(TestError):
    """Validation error"""
    pass

class ResourceError(TestError):
    """Resource error"""
    pass
```

### 4. Mock Testing

Configure realistic mock behavior:
```python
def configure_realistic_mock(
    base_value: float,
    noise: float,
    delay: float
) -> None:
    """Configure realistic mock responses"""
    validate_test_parameter(base_value, 'base_value')
    validate_test_parameter(noise, 'noise', min_value=0)
    validate_test_parameter(delay, 'delay', min_value=0)
```

## See Also

- [Mock Testing Guide](../tutorials/mock_testing.md)
- [Testing Examples](../examples/testing_examples.md)