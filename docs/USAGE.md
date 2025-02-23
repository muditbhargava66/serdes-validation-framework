# Usage Guide

## Overview

This guide covers:
- Basic framework usage
- Mock testing setup
- Hardware integration
- Type-safe development

## Quick Start

### Basic Controller Setup

```python
from typing import Dict, Optional, Union
import numpy as np
from serdes_validation_framework import get_instrument_controller

def initialize_controller(
    mock_mode: Optional[bool] = None
) -> Any:
    """
    Initialize instrument controller
    
    Args:
        mock_mode: Force mock mode if specified
        
    Returns:
        Configured controller instance
    """
    # Set mock mode if specified
    if mock_mode is not None:
        os.environ['SVF_MOCK_MODE'] = '1' if mock_mode else '0'
    
    # Get controller
    controller = get_instrument_controller()
    print(f"Operating in {controller.get_mode()} mode")
    
    return controller
```

### Basic Operations

```python
def basic_instrument_operations(
    resource_name: str,
    timeout_ms: float = 1000.0
) -> None:
    """
    Demonstrate basic operations
    
    Args:
        resource_name: VISA resource identifier
        timeout_ms: Operation timeout in milliseconds
    """
    # Validate inputs
    assert isinstance(timeout_ms, float), "Timeout must be float"
    assert timeout_ms > 0, "Timeout must be positive"
    
    try:
        # Get controller
        controller = get_instrument_controller()
        
        # Connect to instrument
        controller.connect_instrument(resource_name)
        
        # Basic operations
        controller.send_command(resource_name, '*RST')
        response = controller.query_instrument(resource_name, '*IDN?')
        print(f"Instrument: {response}")
        
    finally:
        # Cleanup
        controller.disconnect_instrument(resource_name)
```

## Mock Testing

### Mock Configuration

```python
def configure_mock_testing(
    noise_amplitude: float = 0.05,
    delay_ms: float = 100.0
) -> Any:
    """
    Configure mock testing environment
    
    Args:
        noise_amplitude: Noise amplitude (0-1)
        delay_ms: Response delay in milliseconds
        
    Returns:
        Configured mock controller
    """
    # Validate inputs
    assert isinstance(noise_amplitude, float), "Noise amplitude must be float"
    assert isinstance(delay_ms, float), "Delay must be float"
    assert 0 <= noise_amplitude <= 1, "Noise amplitude must be between 0 and 1"
    assert delay_ms > 0, "Delay must be positive"
    
    # Force mock mode
    os.environ['SVF_MOCK_MODE'] = '1'
    
    # Get controller
    controller = get_instrument_controller()
    
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
```

### Mock Data Generation

```python
def configure_waveform_generation(
    controller: Any,
    num_points: int = 1000000,
    noise_amplitude: float = 0.05
) -> None:
    """
    Configure waveform data generation
    
    Args:
        controller: Mock controller instance
        num_points: Number of data points
        noise_amplitude: Noise amplitude
    """
    def generate_pam4_waveform() -> str:
        """Generate PAM4 test data"""
        # Generate PAM4 levels
        levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
        symbols = np.random.choice(levels, size=num_points)
        
        # Add noise
        noise = np.random.normal(0, noise_amplitude, num_points)
        waveform = symbols + noise
        
        return ','.join(f"{x:.8f}" for x in waveform)
    
    # Add waveform response
    controller.add_mock_response(
        ':WAVeform:DATA?',
        generate_pam4_waveform,
        delay=0.5
    )
```

### Error Simulation

```python
def configure_error_simulation(
    controller: Any,
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

## Hardware Integration

### Oscilloscope Setup

```python
from serdes_validation_framework.instrument_control.scope_224g import (
    HighBandwidthScope,
    ScopeConfig
)

def setup_oscilloscope(
    resource_name: str,
    sample_rate: float = 256e9,
    bandwidth: float = 120e9
) -> HighBandwidthScope:
    """
    Configure oscilloscope
    
    Args:
        resource_name: VISA resource identifier
        sample_rate: Sample rate in Hz
        bandwidth: Bandwidth in Hz
        
    Returns:
        Configured scope instance
    """
    # Validate inputs
    assert isinstance(sample_rate, float), "Sample rate must be float"
    assert isinstance(bandwidth, float), "Bandwidth must be float"
    assert sample_rate > 0, "Sample rate must be positive"
    assert bandwidth > 0, "Bandwidth must be positive"
    
    # Create configuration
    config = ScopeConfig(
        sampling_rate=sample_rate,
        bandwidth=bandwidth,
        timebase=5e-12,  # 5 ps/div
        voltage_range=0.8  # 0.8V
    )
    
    # Initialize scope
    scope = HighBandwidthScope(resource_name)
    scope.configure_for_224g(config)
    
    return scope
```

### Data Collection

```python
def collect_waveform_data(
    scope: HighBandwidthScope,
    duration: float = 1e-6
) -> Dict[str, np.ndarray]:
    """
    Collect waveform data
    
    Args:
        scope: Configured scope instance
        duration: Capture duration in seconds
        
    Returns:
        Dictionary with time and voltage arrays
    """
    # Validate input
    assert isinstance(duration, float), "Duration must be float"
    assert duration > 0, "Duration must be positive"
    
    try:
        # Capture waveform
        waveform = scope.capture_waveform(
            duration=duration,
            sample_rate=scope.config.sampling_rate
        )
        
        # Return data
        return {
            'time': waveform.time,
            'voltage': waveform.voltage
        }
        
    except Exception as e:
        raise RuntimeError(f"Data collection failed: {e}")
```

## Analysis Examples

### PAM4 Analysis

```python
from serdes_validation_framework.data_analysis import PAM4Analyzer

def analyze_pam4_signal(
    time_data: np.ndarray,
    voltage_data: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze PAM4 signal
    
    Args:
        time_data: Time points array
        voltage_data: Voltage measurements array
        
    Returns:
        Analysis results dictionary
    """
    # Validate inputs
    assert isinstance(time_data, np.ndarray), "Time data must be numpy array"
    assert isinstance(voltage_data, np.ndarray), "Voltage data must be numpy array"
    assert len(time_data) == len(voltage_data), "Array lengths must match"
    assert len(time_data) > 0, "Arrays cannot be empty"
    
    try:
        # Create analyzer
        analyzer = PAM4Analyzer({
            'time': time_data,
            'voltage': voltage_data
        })
        
        # Analyze levels
        levels = analyzer.analyze_level_separation()
        
        # Calculate EVM
        evm = analyzer.calculate_evm()
        
        # Analyze eye diagram
        eye = analyzer.analyze_eye_diagram()
        
        # Return results
        return {
            'levels': {
                'means': levels.level_means,
                'separations': levels.level_separations,
                'uniformity': levels.uniformity
            },
            'evm': {
                'rms_percent': evm.rms_evm_percent,
                'peak_percent': evm.peak_evm_percent
            },
            'eye': {
                'heights': eye.eye_heights,
                'widths': eye.eye_widths,
                'worst_height': eye.worst_eye_height
            }
        }
        
    except Exception as e:
        raise RuntimeError(f"Analysis failed: {e}")
```

## Complete Examples

### Mock Testing Example

```python
def run_mock_example() -> None:
    """Complete mock testing example"""
    try:
        # Configure mock environment
        controller = configure_mock_testing(
            noise_amplitude=0.05,
            delay_ms=100.0
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
        
        # Collect data
        waveform = scope.capture_waveform(
            duration=1e-6,
            sample_rate=config.sampling_rate
        )
        
        # Analyze data
        results = analyze_pam4_signal(
            waveform.time,
            waveform.voltage
        )
        
        # Print results
        print("\nAnalysis Results:")
        print(f"RMS EVM: {results['evm']['rms_percent']:.2f}%")
        print(f"Worst Eye Height: {results['eye']['worst_height']:.3f}")
        
    finally:
        # Cleanup
        scope.cleanup()
```

### Hardware Example

```python
def run_hardware_example() -> None:
    """Complete hardware testing example"""
    # Force real hardware mode
    os.environ['SVF_MOCK_MODE'] = '0'
    
    try:
        # Create scope
        scope = setup_oscilloscope(
            'GPIB0::7::INSTR',
            sample_rate=256e9,
            bandwidth=120e9
        )
        
        # Collect data
        data = collect_waveform_data(
            scope,
            duration=1e-6
        )
        
        # Analyze data
        results = analyze_pam4_signal(
            data['time'],
            data['voltage']
        )
        
        # Print results
        print("\nAnalysis Results:")
        print(f"RMS EVM: {results['evm']['rms_percent']:.2f}%")
        print(f"Worst Eye Height: {results['eye']['worst_height']:.3f}")
        
    finally:
        # Cleanup
        scope.cleanup()
```

## Best Practices

1. Always validate numeric inputs:
   ```python
   assert isinstance(value, float), f"Value must be float, got {type(value)}"
   ```

2. Use proper cleanup:
   ```python
   try:
       # ... operations
   finally:
       controller.disconnect_instrument(resource_name)
   ```

3. Handle errors appropriately:
   ```python
   try:
       result = controller.query_instrument(resource, query)
   except Exception as e:
       logger.error(f"Query failed: {e}")
       raise
   ```

## See Alsos

- [Installation Guide](INSTALL.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Mock Testing Tutorial](tutorials/mock_testing.md)

---