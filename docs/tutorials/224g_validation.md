# 224G Ethernet Validation Tutorial

## Overview

This tutorial demonstrates how to validate 224G Ethernet interfaces using the SerDes Validation Framework, focusing on:
- Equipment setup and configuration
- Link training validation
- Compliance testing
- Performance analysis

## Type-Safe Setup

### Equipment Configuration

```python
from typing import Dict, List, Optional, Union
import numpy as np
import numpy.typing as npt
from serdes_validation_framework.instrument_control.scope_224g import (
    HighBandwidthScope,
    ScopeConfig
)

def configure_validation_setup(
    scope_address: str,
    sample_rate: float = 256e9,
    bandwidth: float = 120e9
) -> HighBandwidthScope:
    """
    Configure scope for 224G validation with type checking
    
    Args:
        scope_address: VISA resource identifier
        sample_rate: Sample rate in Hz
        bandwidth: Bandwidth in Hz
        
    Returns:
        Configured scope instance
        
    Raises:
        AssertionError: If parameters are invalid
        ConnectionError: If setup fails
    """
    # Validate inputs
    assert isinstance(scope_address, str), "Scope address must be string"
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
    
    try:
        # Initialize scope
        scope = HighBandwidthScope(scope_address)
        scope.configure_for_224g(config)
        return scope
        
    except Exception as e:
        raise ConnectionError(f"Failed to configure scope: {e}")
```

### Data Validation

```python
def validate_waveform_data(
    time_data: npt.NDArray[np.float64],
    voltage_data: npt.NDArray[np.float64]
) -> None:
    """
    Validate waveform data arrays
    
    Args:
        time_data: Time points array
        voltage_data: Voltage measurements array
        
    Raises:
        AssertionError: If validation fails
    """
    # Type validation
    assert isinstance(time_data, np.ndarray), \
        f"Time data must be numpy array, got {type(time_data)}"
    assert isinstance(voltage_data, np.ndarray), \
        f"Voltage data must be numpy array, got {type(voltage_data)}"
    
    # Data type validation
    assert np.issubdtype(time_data.dtype, np.floating), \
        f"Time data must be floating-point, got {time_data.dtype}"
    assert np.issubdtype(voltage_data.dtype, np.floating), \
        f"Voltage data must be floating-point, got {voltage_data.dtype}"
    
    # Array validation
    assert len(time_data) == len(voltage_data), \
        f"Array length mismatch: {len(time_data)} != {len(voltage_data)}"
    assert len(time_data) > 0, "Arrays cannot be empty"
    
    # Value validation
    assert not np.any(np.isnan(time_data)), "Time data contains NaN values"
    assert not np.any(np.isnan(voltage_data)), "Voltage data contains NaN values"
    assert not np.any(np.isinf(time_data)), "Time data contains infinite values"
    assert not np.any(np.isinf(voltage_data)), "Voltage data contains infinite values"
```

## Link Training Validation

### Training Configuration

```python
from serdes_validation_framework.protocols.ethernet_224g.training import (
    EqualizerConfig,
    EnhancedEqualizer
)

def setup_link_training(
    adaptation_rate: float = 0.01,
    max_iterations: int = 1000
) -> EnhancedEqualizer:
    """
    Configure link training parameters
    
    Args:
        adaptation_rate: Equalizer adaptation rate
        max_iterations: Maximum training iterations
        
    Returns:
        Configured equalizer instance
    """
    # Validate inputs
    assert isinstance(adaptation_rate, float), \
        "Adaptation rate must be float"
    assert isinstance(max_iterations, int), \
        "Max iterations must be integer"
    assert 0 < adaptation_rate < 1, \
        "Adaptation rate must be between 0 and 1"
    assert max_iterations > 0, \
        "Max iterations must be positive"
    
    # Create configuration
    config = EqualizerConfig(
        fir_taps=11,          # Odd number for symmetric FIR
        dfe_taps=3,           # Typical for 224G
        adaptation_rate=adaptation_rate,
        max_iterations=max_iterations,
        convergence_threshold=1e-6,
        min_snr=20.0,
        algorithm='lms'
    )
    
    return EnhancedEqualizer(config)
```

### Running Training

```python
def run_link_training(
    scope: HighBandwidthScope,
    equalizer: EnhancedEqualizer,
    duration: float = 1e-6
) -> Dict[str, Union[float, List[float]]]:
    """
    Run link training sequence
    
    Args:
        scope: Configured scope instance
        equalizer: Configured equalizer
        duration: Training duration in seconds
        
    Returns:
        Dictionary of training results
    """
    # Validate inputs
    assert isinstance(duration, float), "Duration must be float"
    assert duration > 0, "Duration must be positive"
    
    try:
        # Capture training data
        waveform = scope.capture_waveform(
            duration=duration,
            sample_rate=scope.config.sampling_rate
        )
        
        # Validate captured data
        validate_waveform_data(waveform.time, waveform.voltage)
        
        # Run training
        trained_signal, state = equalizer.train(
            waveform.voltage,
            np.roll(waveform.voltage, -1)  # Simple target
        )
        
        # Return results
        return {
            'training_time': float(duration),
            'convergence_status': state.error_history[-1] < equalizer.config.convergence_threshold,
            'final_error': float(state.error_history[-1]),
            'tap_weights': list(map(float, state.fir_taps))
        }
        
    except Exception as e:
        raise RuntimeError(f"Link training failed: {e}")
```

## Compliance Testing

### Test Configuration

```python
from serdes_validation_framework.protocols.ethernet_224g.compliance import (
    ComplianceSpecification,
    ComplianceTestConfig
)

def setup_compliance_tests() -> Tuple[ComplianceSpecification, Dict[str, float]]:
    """
    Set up compliance testing configuration
    
    Returns:
        Tuple of (ComplianceSpecification, test limits)
    """
    # Create specification checker
    spec = ComplianceSpecification()
    
    # Define test limits
    limits = {
        'eye_height_min': 0.2,    # Minimum eye height
        'eye_width_min': 0.4,     # Minimum eye width
        'rms_evm_max': 5.0,       # Maximum RMS EVM percentage
        'jitter_max': 0.3e-12     # Maximum total jitter
    }
    
    return spec, limits
```

### Running Tests

```python
def run_compliance_tests(
    scope: HighBandwidthScope,
    spec: ComplianceSpecification,
    limits: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Run compliance test suite
    
    Args:
        scope: Configured scope instance
        spec: Compliance specification
        limits: Test limits
        
    Returns:
        Dictionary of test results
    """
    try:
        results = {}
        
        # Level compliance
        levels = scope.measure_pam4_levels()
        level_status, level_measurements = spec.check_pam4_levels(
            levels.level_means
        )
        results['levels'] = level_measurements
        
        # Eye diagram
        eye = scope.capture_eye_diagram()
        eye_status, eye_measurements = spec.check_eye_diagram(
            eye.worst_eye_height,
            eye.worst_eye_width
        )
        results['eye'] = eye_measurements
        
        # Jitter
        jitter = scope.measure_jitter()
        results['jitter'] = jitter
        
        # Overall status
        results['status'] = all([
            level_status,
            eye_status,
            jitter['tj'] < limits['jitter_max']
        ])
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Compliance testing failed: {e}")
```

## Complete Example

### Full Validation Sequence

```python
def run_validation_sequence() -> None:
    """Run complete 224G validation sequence"""
    # Force mock mode for example
    os.environ['SVF_MOCK_MODE'] = '1'
    
    try:
        # 1. Setup equipment
        scope = configure_validation_setup('GPIB0::7::INSTR')
        
        # 2. Run link training
        equalizer = setup_link_training()
        training_results = run_link_training(scope, equalizer)
        
        print("\nLink Training Results:")
        print(f"Status: {'Converged' if training_results['convergence_status'] else 'Failed'}")
        print(f"Final Error: {training_results['final_error']:.6f}")
        
        # 3. Run compliance tests
        spec, limits = setup_compliance_tests()
        compliance_results = run_compliance_tests(scope, spec, limits)
        
        print("\nCompliance Test Results:")
        print(f"Overall Status: {'PASS' if compliance_results['status'] else 'FAIL'}")
        
        print("\nDetailed Measurements:")
        for category, measurements in compliance_results.items():
            if category != 'status':
                print(f"\n{category.upper()}:")
                for param, value in measurements.items():
                    print(f"  {param}: {value}")
        
    except Exception as e:
        print(f"Validation failed: {e}")
        
    finally:
        # Cleanup
        scope.cleanup()
```

## Best Practices

### 1. Type Validation

Always validate numeric inputs:
```python
def validate_numeric_parameter(
    value: float,
    name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> None:
    """Validate numeric parameter"""
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

Use proper error hierarchy:
```python
class ValidationError(Exception):
    """Base validation error"""
    pass

class ConfigError(ValidationError):
    """Configuration error"""
    pass

class MeasurementError(ValidationError):
    """Measurement error"""
    pass
```

### 3. Results Validation

Validate measurement results:
```python
def validate_results(
    results: Dict[str, Dict[str, float]],
    limits: Dict[str, float]
) -> bool:
    """Validate results against limits"""
    try:
        for category, measurements in results.items():
            for param, value in measurements.items():
                limit = limits.get(f"{category}.{param}")
                if limit is not None and value > limit:
                    return False
        return True
    except Exception as e:
        raise ValidationError(f"Results validation failed: {e}")
```

## See Also

- [PAM4 Analysis Tutorial](pam4_analysis.md)
- [Mock Testing Tutorial](mock_testing.md)
- [224G Ethernet API](../api/eth_224g.md)

## References

- IEEE 802.3 224G Ethernet Specification
- High-Speed Serial Link Design Guide
- Equipment Manuals and Specifications