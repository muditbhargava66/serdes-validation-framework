# Mock Controller Examples

## Basic Usage

### Simple Mock Testing

```python
from typing import Dict, List, Optional, Union
import numpy as np
from serdes_validation_framework.instrument_control.mock_controller import (
    get_instrument_controller,
    MockInstrumentController
)

def basic_mock_example() -> None:
    """Basic mock controller demonstration"""
    # Force mock mode
    os.environ['SVF_MOCK_MODE'] = '1'
    
    try:
        # Initialize controller
        controller = get_instrument_controller()
        
        # Connect to mock instrument
        controller.connect_instrument('GPIB::1::INSTR')
        
        # Query identification
        response = controller.query_instrument('GPIB::1::INSTR', '*IDN?')
        print(f"Mock instrument: {response}")
        
    finally:
        # Cleanup
        controller.disconnect_instrument('GPIB::1::INSTR')
```

### Type-Safe Configuration

```python
def configure_mock_controller(
    noise_amplitude: float = 0.05,
    delay_ms: float = 100.0
) -> MockInstrumentController:
    """
    Configure mock controller with type validation
    
    Args:
        noise_amplitude: Noise amplitude (0-1)
        delay_ms: Response delay in milliseconds
        
    Returns:
        Configured controller
    """
    # Validate inputs
    assert isinstance(noise_amplitude, float), \
        "Noise amplitude must be float"
    assert isinstance(delay_ms, float), \
        "Delay must be float"
    assert 0 <= noise_amplitude <= 1, \
        "Noise amplitude must be between 0 and 1"
    assert delay_ms > 0, \
        "Delay must be positive"
    
    try:
        # Create controller
        controller = MockInstrumentController()
        
        # Configure responses
        controller.add_mock_response(
            '*IDN?',
            'Mock Instrument v1.0',
            delay=delay_ms/1000.0
        )
        
        controller.add_mock_response(
            'MEASure:VOLTage:DC?',
            lambda: f"{np.random.normal(0, noise_amplitude):.6f}",
            delay=delay_ms/1000.0
        )
        
        return controller
        
    except Exception as e:
        raise RuntimeError(f"Controller configuration failed: {e}")
```

## Advanced Examples

### Custom Data Generation

```python
def generate_pam4_data(
    num_points: int = 1000000,
    sample_rate: float = 256e9,
    noise_amplitude: float = 0.05
) -> str:
    """
    Generate PAM4 waveform data
    
    Args:
        num_points: Number of data points
        sample_rate: Sample rate in Hz
        noise_amplitude: Noise amplitude
        
    Returns:
        Comma-separated waveform data
    """
    # Validate inputs
    assert isinstance(num_points, int), "num_points must be integer"
    assert isinstance(sample_rate, float), "sample_rate must be float"
    assert isinstance(noise_amplitude, float), "noise_amplitude must be float"
    assert num_points > 0, "num_points must be positive"
    assert sample_rate > 0, "sample_rate must be positive"
    assert noise_amplitude >= 0, "noise_amplitude must be non-negative"
    
    try:
        # Generate PAM4 levels
        levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
        symbols = np.random.choice(levels, size=num_points)
        
        # Add noise
        noise = np.random.normal(0, noise_amplitude, num_points)
        waveform = (symbols + noise).astype(np.float64)
        
        # Format data
        return ','.join(f"{x:.8f}" for x in waveform)
        
    except Exception as e:
        raise ValueError(f"Data generation failed: {e}")

def configure_waveform_generation(
    controller: MockInstrumentController
) -> None:
    """
    Configure waveform generation responses
    
    Args:
        controller: Mock controller instance
    """
    # Add waveform response
    controller.add_mock_response(
        ':WAVeform:DATA?',
        lambda: generate_pam4_data(),
        delay=0.5
    )
    
    # Add parameters response
    controller.add_mock_response(
        ':WAVeform:XINCrement?',
        lambda: f"{1/256e9:.12e}",  # Sample period
        delay=0.1
    )
```

### Error Simulation

```python
def configure_error_simulation(
    controller: MockInstrumentController,
    connection_error_rate: float = 0.1,
    command_error_rate: float = 0.05
) -> None:
    """
    Configure error simulation
    
    Args:
        controller: Mock controller instance
        connection_error_rate: Connection failure probability
        command_error_rate: Command failure probability
    """
    # Validate inputs
    assert isinstance(connection_error_rate, float), \
        "Connection error rate must be float"
    assert isinstance(command_error_rate, float), \
        "Command error rate must be float"
    assert 0 <= connection_error_rate <= 1, \
        "Connection error rate must be between 0 and 1"
    assert 0 <= command_error_rate <= 1, \
        "Command error rate must be between 0 and 1"
    
    # Set error rates
    controller.set_error_rates(
        connection_error_rate=connection_error_rate,
        command_error_rate=command_error_rate
    )
```

## Complete Examples

### Mock Oscilloscope

```python
from serdes_validation_framework.instrument_control.scope_224g import (
    HighBandwidthScope,
    ScopeConfig
)

def mock_scope_example() -> None:
    """Demonstrate complete mock scope usage"""
    # Force mock mode
    os.environ['SVF_MOCK_MODE'] = '1'
    
    try:
        # Create mock controller
        controller = configure_mock_controller(
            noise_amplitude=0.05,
            delay_ms=100.0
        )
        
        # Configure mock responses
        configure_waveform_generation(controller)
        
        # Create scope
        scope = HighBandwidthScope(
            'GPIB0::7::INSTR',
            controller=controller
        )
        
        # Configure scope
        config = ScopeConfig(
            sampling_rate=256e9,  # 256 GSa/s
            bandwidth=120e9,      # 120 GHz
            timebase=5e-12,      # 5 ps/div
            voltage_range=0.8    # 0.8V
        )
        scope.configure_for_224g(config)
        
        # Capture waveform
        waveform = scope.capture_waveform(
            duration=1e-6,  # 1 µs
            sample_rate=config.sampling_rate
        )
        
        print(f"Captured {len(waveform.voltage)} points")
        print(f"Voltage range: {np.min(waveform.voltage):.2f} to "
              f"{np.max(waveform.voltage):.2f}")
        
    finally:
        # Cleanup
        scope.cleanup()
```

### Mock Pattern Generator 

```python
def mock_pattern_generator_example() -> None:
    """Demonstrate mock pattern generator usage"""
    # Force mock mode
    os.environ['SVF_MOCK_MODE'] = '1'
    
    try:
        # Configure controller
        controller = configure_mock_controller()
        
        # Add pattern responses
        controller.add_mock_response(
            ':PATTern:TYPE?',
            'PRBS31',
            delay=0.1
        )
        
        controller.add_mock_response(
            ':PATTern:LENGth?',
            '2147483647',  # 2^31 - 1
            delay=0.1
        )
        
        # Connect to instrument
        controller.connect_instrument('GPIB::10::INSTR')
        
        # Configure pattern
        controller.send_command(
            'GPIB::10::INSTR',
            ':PATTern:TYPE PRBS31'
        )
        
        # Verify configuration
        pattern = controller.query_instrument(
            'GPIB::10::INSTR',
            ':PATTern:TYPE?'
        )
        print(f"Pattern type: {pattern}")
        
    finally:
        # Cleanup
        controller.disconnect_instrument('GPIB::10::INSTR')
```

### Complete Test Setup

```python
def complete_mock_setup() -> None:
    """Demonstrate complete mock test setup"""
    # Force mock mode
    os.environ['SVF_MOCK_MODE'] = '1'
    
    try:
        # Create and configure controller
        controller = configure_mock_controller(
            noise_amplitude=0.05,
            delay_ms=100.0
        )
        
        # Configure error simulation
        configure_error_simulation(
            controller,
            connection_error_rate=0.1,
            command_error_rate=0.05
        )
        
        # Configure waveform generation
        configure_waveform_generation(controller)
        
        # Create scope
        scope = HighBandwidthScope(
            'GPIB0::7::INSTR',
            controller=controller
        )
        
        # Configure scope
        config = ScopeConfig(
            sampling_rate=256e9,
            bandwidth=120e9,
            timebase=5e-12,
            voltage_range=0.8
        )
        scope.configure_for_224g(config)
        
        # Capture and analyze data
        waveform = scope.capture_waveform(
            duration=1e-6,
            sample_rate=config.sampling_rate
        )
        
        # Create analyzer
        analyzer = PAM4Analyzer({
            'time': waveform.time,
            'voltage': waveform.voltage
        })
        
        # Analyze levels
        levels = analyzer.analyze_level_separation()
        print("\nPAM4 Levels:")
        print(f"Means: {levels.level_means}")
        print(f"Separations: {levels.level_separations}")
        print(f"Uniformity: {levels.uniformity:.3f}")
        
        # Calculate EVM
        evm = analyzer.calculate_evm()
        print("\nEVM Results:")
        print(f"RMS EVM: {evm.rms_evm_percent:.2f}%")
        print(f"Peak EVM: {evm.peak_evm_percent:.2f}%")
        
        # Analyze eye diagram
        eye = analyzer.analyze_eye_diagram()
        print("\nEye Measurements:")
        print(f"Heights: {eye.eye_heights}")
        print(f"Worst height: {eye.worst_eye_height:.3f}")
        
    finally:
        # Cleanup
        scope.cleanup()
```

## Best Practices

### 1. Type Validation

```python
def validate_mock_parameters(
    value: float,
    name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> None:
    """
    Validate numeric parameters
    
    Args:
        value: Parameter value
        name: Parameter name
        min_value: Optional minimum value
        max_value: Optional maximum value
    """
    assert isinstance(value, float), \
        f"{name} must be float, got {type(value)}"
    
    if min_value is not None:
        assert value >= min_value, \
            f"{name} must be >= {min_value}"
    
    if max_value is not None:
        assert value <= max_value, \
            f"{name} must be <= {max_value}"
```

### 2. Error Handling

```python
def safe_mock_operation(func):
    """Decorator for safe mock operations"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            print(f"Connection error: {e}")
            raise
        except Exception as e:
            print(f"Operation failed: {e}")
            raise
    return wrapper
```

### 3. Resource Cleanup

```python
from contextlib import contextmanager

@contextmanager
def mock_instrument(
    address: str,
    controller: Optional[MockInstrumentController] = None
):
    """Context manager for mock instrument"""
    controller = controller or configure_mock_controller()
    try:
        controller.connect_instrument(address)
        yield controller
    finally:
        controller.disconnect_instrument(address)
```

## See Also

- [Mock Testing Tutorial](../tutorials/mock_testing.md)
- [Mock Controller API](../api/mock_controller.md)
- [Testing Examples](testing_examples.md)