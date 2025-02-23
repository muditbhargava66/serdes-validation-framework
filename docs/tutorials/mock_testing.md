# Mock Testing Tutorial

## Introduction

This tutorial demonstrates how to use the mock testing capabilities of the SerDes Validation Framework for development and testing without physical hardware.

## Environment Setup

First, configure your environment for mock testing:

```bash
# Force mock mode for development
export SVF_MOCK_MODE=1

# For automated testing, set in your test script:
import os
os.environ['SVF_MOCK_MODE'] = '1'
```

## Basic Mock Testing

### Initializing the Controller

```python
from typing import Dict, List, Optional, Union
import numpy as np
import numpy.typing as npt
from serdes_validation_framework.instrument_control.mock_controller import (
    get_instrument_controller,
    get_instrument_mode
)

def initialize_mock_controller() -> Any:
    """
    Initialize mock controller with validation
    
    Returns:
        Mock instrument controller instance
    """
    # Force mock mode
    os.environ['SVF_MOCK_MODE'] = '1'
    
    # Get controller
    controller = get_instrument_controller()
    
    # Verify mock mode
    mode = controller.get_mode()
    assert mode == 'mock', f"Expected mock mode, got {mode}"
    
    return controller
```

### Basic Commands

```python
def test_basic_commands() -> None:
    """Demonstrate basic mock command execution"""
    controller = initialize_mock_controller()
    
    try:
        # Connect to mock instrument
        controller.connect_instrument('GPIB::1::INSTR')
        
        # Send command
        controller.send_command('GPIB::1::INSTR', '*RST')
        
        # Query instrument
        response = controller.query_instrument('GPIB::1::INSTR', '*IDN?')
        print(f"Mock response: {response}")
        
    finally:
        # Cleanup
        controller.disconnect_instrument('GPIB::1::INSTR')
```

## Custom Response Configuration

### Simple Responses

```python
def configure_basic_responses(controller: Any) -> None:
    """
    Configure basic mock responses
    
    Args:
        controller: Mock controller instance
    """
    # Static response
    controller.add_mock_response(
        '*IDN?',
        'Mock Instrument v1.0',
        delay=0.1
    )
    
    # Dynamic response
    controller.add_mock_response(
        'MEASure:VOLTage:DC?',
        lambda: f"{np.random.normal(0, 0.1):.6f}",
        delay=0.2
    )
    
    # Response with error simulation
    controller.add_mock_response(
        'TEST:ERRor?',
        'Error: Invalid command',
        delay=0.1,
        error_rate=0.2  # 20% error rate
    )
```

### PAM4 Signal Generation

```python
def configure_pam4_responses(controller: Any) -> None:
    """
    Configure realistic PAM4 signal responses
    
    Args:
        controller: Mock controller instance
    """
    def generate_pam4_waveform(
        num_points: int = 1000000,
        noise_amplitude: float = 0.05
    ) -> str:
        """
        Generate synthetic PAM4 waveform
        
        Args:
            num_points: Number of data points
            noise_amplitude: Noise amplitude (0-1)
            
        Returns:
            Comma-separated waveform data
        """
        # Validate inputs
        assert isinstance(num_points, int), "num_points must be integer"
        assert isinstance(noise_amplitude, float), "noise_amplitude must be float"
        assert noise_amplitude > 0, "noise_amplitude must be positive"
        
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

## Error Simulation

### Basic Error Rates

```python
def configure_error_simulation(controller: Any) -> None:
    """
    Configure error simulation rates
    
    Args:
        controller: Mock controller instance
    """
    controller.set_error_rates(
        connection_error_rate=0.1,  # 10% connection failures
        command_error_rate=0.05,    # 5% command failures
        data_error_rate=0.01        # 1% data corruption
    )
```

### Error Handling Example

```python
def test_error_handling() -> None:
    """Demonstrate error handling with mock controller"""
    controller = initialize_mock_controller()
    
    # Configure high error rates for testing
    controller.set_error_rates(
        connection_error_rate=0.5,
        command_error_rate=0.5
    )
    
    try:
        # Attempt connections
        for i in range(5):
            try:
                print(f"\nAttempt {i+1}:")
                controller.connect_instrument('GPIB::1::INSTR')
                print("  Connection successful")
                
                response = controller.query_instrument(
                    'GPIB::1::INSTR',
                    '*IDN?'
                )
                print(f"  Query successful: {response}")
                
                controller.disconnect_instrument('GPIB::1::INSTR')
                print("  Disconnection successful")
                
            except Exception as e:
                print(f"  Error: {e}")
                
    finally:
        # Ensure cleanup
        try:
            controller.disconnect_instrument('GPIB::1::INSTR')
        except:
            pass
```

## Complete Test Example

### Mock Scope Testing

```python
from serdes_validation_framework.instrument_control.scope_224g import (
    HighBandwidthScope,
    ScopeConfig
)

def test_mock_scope() -> None:
    """Demonstrate complete mock scope testing"""
    # Force mock mode
    os.environ['SVF_MOCK_MODE'] = '1'
    
    try:
        # Initialize scope
        scope = HighBandwidthScope('GPIB0::7::INSTR')
        
        # Configure scope
        config = ScopeConfig(
            sampling_rate=256e9,  # 256 GSa/s
            bandwidth=120e9,      # 120 GHz
            timebase=5e-12,       # 5 ps/div
            voltage_range=0.8     # 0.8V
        )
        scope.configure_for_224g(config)
        
        # Configure mock responses
        configure_pam4_responses(scope.controller)
        
        # Capture waveform
        waveform = scope.capture_waveform(
            duration=1e-6,  # 1 µs
            sample_rate=config.sampling_rate
        )
        
        # Validate data
        assert isinstance(waveform.time, np.ndarray), \
            "Time data must be numpy array"
        assert isinstance(waveform.voltage, np.ndarray), \
            "Voltage data must be numpy array"
        
        print(f"Captured {len(waveform.voltage)} points")
        print(f"Voltage range: {np.min(waveform.voltage):.2f} to "
              f"{np.max(waveform.voltage):.2f}")
        
    finally:
        # Cleanup
        scope.cleanup()
```

### Mock Compliance Testing

```python
from serdes_validation_framework.test_sequence.eth_224g_sequence import (
    Ethernet224GTestSequence,
    ComplianceResults
)

def test_mock_compliance() -> None:
    """Demonstrate mock compliance testing"""
    # Force mock mode
    os.environ['SVF_MOCK_MODE'] = '1'
    
    try:
        # Initialize sequence
        sequence = Ethernet224GTestSequence()
        
        # Set up instruments
        scope_address = 'GPIB0::7::INSTR'
        pattern_gen_address = 'GPIB0::10::INSTR'
        
        sequence.setup_instruments([
            scope_address,
            pattern_gen_address
        ])
        
        # Configure mock responses
        controller = sequence.instrument_controller
        configure_pam4_responses(controller)
        
        # Run compliance tests
        results = sequence.run_compliance_test_suite(
            scope_address,
            pattern_gen_address
        )
        
        # Validate results
        assert isinstance(results, ComplianceResults), \
            "Invalid results type"
        
        # Print results
        print("\nCompliance Test Results:")
        print(f"Status: {results.test_status}")
        print(f"RMS EVM: {results.evm_results.rms_evm_percent:.2f}%")
        print(f"Worst Eye Height: {results.eye_results.worst_eye_height:.3f}")
        
    finally:
        # Cleanup
        sequence.cleanup([scope_address, pattern_gen_address])
```

## Best Practices

### 1. Type Safety

Always validate numeric inputs:
```python
def configure_mock_measurement(
    value: float,
    noise: float
) -> None:
    """Configure mock measurement response"""
    assert isinstance(value, float), "value must be float"
    assert isinstance(noise, float), "noise must be float"
    assert noise >= 0, "noise must be non-negative"
```

### 2. Error Handling

Use proper error handling:
```python
try:
    controller.connect_instrument(resource_name)
    # ... perform operations
except ConnectionError as e:
    logger.error(f"Connection failed: {e}")
    raise
finally:
    controller.disconnect_instrument(resource_name)
```

### 3. Resource Cleanup

Always cleanup resources:
```python
def test_with_cleanup() -> None:
    """Example with proper cleanup"""
    controller = None
    try:
        controller = initialize_mock_controller()
        # ... perform tests
    finally:
        if controller is not None:
            for resource in controller.connected_instruments:
                try:
                    controller.disconnect_instrument(resource)
                except Exception as e:
                    logger.warning(f"Cleanup error: {e}")
```

### 4. Mock Configuration

Organize mock responses:
```python
def configure_mock_controller(controller: Any) -> None:
    """Centralized mock configuration"""
    # Basic responses
    configure_basic_responses(controller)
    
    # PAM4 responses
    configure_pam4_responses(controller)
    
    # Error simulation
    configure_error_simulation(controller)
```

## See Also

- [Mock Controller API](../api/mock_controller.md)
- [Instrument Control API](../api/instrument_control.md)
- [224G Validation Tutorial](224g_validation.md)