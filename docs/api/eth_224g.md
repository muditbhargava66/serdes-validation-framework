# 224G Ethernet Protocol API

## Overview

This module provides comprehensive support for validating 224G Ethernet interfaces, including:
- PAM4 signal generation and analysis
- Advanced link training algorithms
- Compliance testing
- Mock testing capabilities

## Type Safety

All numeric parameters use explicit type validation:

```python
from typing import Dict, List, Union, Optional
import numpy as np
import numpy.typing as npt

def validate_signal_parameters(
    time_data: npt.NDArray[np.float64],
    voltage_data: npt.NDArray[np.float64]
) -> None:
    """
    Validate signal array parameters
    
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

## Equipment Setup

### Scope Configuration

```python
from serdes_validation_framework.instrument_control.scope_224g import (
    HighBandwidthScope,
    ScopeConfig
)

def configure_scope_for_224g(
    scope_address: str,
    sampling_rate: float = 256e9,
    bandwidth: float = 120e9
) -> HighBandwidthScope:
    """
    Configure scope for 224G measurements
    
    Args:
        scope_address: VISA resource identifier
        sampling_rate: Sample rate in Hz
        bandwidth: Bandwidth in Hz
        
    Returns:
        Configured scope instance
        
    Raises:
        AssertionError: If parameters are invalid
        ConnectionError: If scope connection fails
    """
    # Validate parameters
    assert isinstance(sampling_rate, float), "Sample rate must be float"
    assert isinstance(bandwidth, float), "Bandwidth must be float"
    assert sampling_rate > 0, "Sample rate must be positive"
    assert bandwidth > 0, "Bandwidth must be positive"
    
    # Create configuration
    config = ScopeConfig(
        sampling_rate=sampling_rate,
        bandwidth=bandwidth,
        timebase=5e-12,  # 5 ps/div
        voltage_range=0.8  # 0.8V
    )
    
    # Initialize and configure scope
    scope = HighBandwidthScope(scope_address)
    scope.configure_for_224g(config)
    
    return scope
```

### Mock Mode Configuration

```python
from serdes_validation_framework.instrument_control.mock_controller import (
    get_instrument_controller,
    MockInstrumentController
)

def configure_mock_responses(controller: MockInstrumentController) -> None:
    """Configure realistic mock responses for 224G testing"""
    
    def generate_pam4_waveform() -> str:
        """Generate PAM4 test data"""
        num_points = 1000000
        levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
        symbols = np.random.choice(levels, size=num_points)
        noise = np.random.normal(0, 0.05, num_points)
        waveform = (symbols + noise).astype(np.float64)
        return ','.join(f"{x:.8f}" for x in waveform)
    
    # Add mock responses
    controller.add_mock_response(
        ':WAVeform:DATA?',
        generate_pam4_waveform,
        delay=0.5
    )
    
    controller.add_mock_response(
        ':MEASure:EYE:HEIGht?',
        lambda: f"{np.random.normal(0.4, 0.05):.6f}",
        delay=0.1
    )
    
    controller.add_mock_response(
        ':MEASure:JITTer:TJ?',
        lambda: f"{np.random.normal(1e-12, 1e-13):.3e}",
        delay=0.1
    )
```

## Link Training

### Training Configuration

```python
@dataclass
class TrainingConfig:
    """Link training configuration with validation"""
    adaptation_rate: float
    max_iterations: int
    convergence_threshold: float
    min_snr: float
    
    def __post_init__(self) -> None:
        """Validate configuration parameters"""
        assert isinstance(self.adaptation_rate, float), \
            "Adaptation rate must be float"
        assert isinstance(self.max_iterations, int), \
            "Max iterations must be integer"
        assert isinstance(self.convergence_threshold, float), \
            "Convergence threshold must be float"
        assert isinstance(self.min_snr, float), \
            "Minimum SNR must be float"
        
        assert 0 < self.adaptation_rate < 1, \
            "Adaptation rate must be between 0 and 1"
        assert self.max_iterations > 0, \
            "Max iterations must be positive"
        assert self.convergence_threshold > 0, \
            "Convergence threshold must be positive"
        assert self.min_snr > 0, \
            "Minimum SNR must be positive"
```

### Training Execution

```python
def run_link_training(
    scope: HighBandwidthScope,
    config: TrainingConfig
) -> Dict[str, Union[float, List[float]]]:
    """
    Run link training with validation
    
    Args:
        scope: Configured scope instance
        config: Training configuration
        
    Returns:
        Dictionary containing:
            - training_time: Training duration in seconds
            - final_error: Final adaptation error
            - tap_weights: List of equalizer tap weights 
    """
    try:
        # Capture training data
        waveform_data = scope.capture_waveform(
            duration=1e-6,  # 1 µs
            sample_rate=scope.config.sampling_rate
        )
        
        # Validate captured data
        validate_signal_parameters(
            waveform_data.time,
            waveform_data.voltage
        )
        
        # Run training algorithm
        tap_weights = train_equalizer(
            waveform_data.voltage,
            config.adaptation_rate,
            config.max_iterations,
            config.convergence_threshold
        )
        
        # Calculate final error
        error = calculate_error(
            waveform_data.voltage,
            tap_weights
        )
        
        return {
            'training_time': float(waveform_data.time[-1]),
            'final_error': float(error),
            'tap_weights': list(map(float, tap_weights))
        }
        
    except Exception as e:
        raise TrainingError(f"Link training failed: {e}")
```

## Compliance Testing

### Test Configuration

```python
@dataclass
class ComplianceConfig:
    """Compliance test configuration"""
    test_patterns: Dict[str, str]
    eye_mask: Dict[str, float]
    jitter_limits: Dict[str, float]
    evm_limits: Dict[str, float]
    
    def __post_init__(self) -> None:
        """Validate configuration"""
        # Validate patterns
        assert all(isinstance(k, str) and isinstance(v, str) 
                  for k, v in self.test_patterns.items()), \
            "Test patterns must be strings"
        
        # Validate numeric limits
        for limits in [self.eye_mask, self.jitter_limits, self.evm_limits]:
            assert all(isinstance(k, str) and isinstance(v, float)
                      for k, v in limits.items()), \
                "Limits must be floating-point numbers"
```

### Running Tests

```python
def run_compliance_tests(
    scope: HighBandwidthScope,
    config: ComplianceConfig
) -> Dict[str, Dict[str, float]]:
    """
    Run compliance test suite
    
    Args:
        scope: Configured scope instance
        config: Test configuration
        
    Returns:
        Dictionary of test results by category
    """
    results = {}
    
    try:
        # Level compliance
        level_data = scope.measure_pam4_levels()
        results['levels'] = validate_pam4_levels(level_data)
        
        # Eye diagram
        eye_data = scope.capture_eye_diagram()
        results['eye'] = validate_eye_parameters(
            eye_data,
            config.eye_mask
        )
        
        # Jitter
        jitter_data = scope.measure_jitter()
        results['jitter'] = validate_jitter_components(
            jitter_data,
            config.jitter_limits
        )
        
        # EVM
        evm_data = scope.measure_evm()
        results['evm'] = validate_evm_results(
            evm_data,
            config.evm_limits
        )
        
        return results
        
    except Exception as e:
        raise ComplianceError(f"Compliance testing failed: {e}")
```

## Mock Testing Example

```python
def run_mock_compliance_test() -> None:
    """Example of running compliance tests with mock hardware"""
    
    # Force mock mode
    os.environ['SVF_MOCK_MODE'] = '1'
    
    try:
        # Get mock controller
        controller = get_instrument_controller()
        configure_mock_responses(controller)
        
        # Configure scope
        scope = configure_scope_for_224g('GPIB0::7::INSTR')
        
        # Create test configuration
        config = ComplianceConfig(
            test_patterns={
                'level': 'PRBS13',
                'eye': 'PRBS31'
            },
            eye_mask={
                'height': 0.2,
                'width': 0.4
            },
            jitter_limits={
                'tj': 0.3e-12,
                'rj': 0.15e-12
            },
            evm_limits={
                'rms': 5.0,
                'peak': 10.0
            }
        )
        
        # Run tests
        results = run_compliance_tests(scope, config)
        
        # Print results
        print("\nCompliance Test Results:")
        for category, measurements in results.items():
            print(f"\n{category.upper()}:")
            for param, value in measurements.items():
                print(f"  {param}: {value}")
                
    except Exception as e:
        print(f"Test failed: {e}")
        
    finally:
        # Cleanup
        scope.cleanup()
```

## Error Handling

```python
class Eth224GError(Exception):
    """Base class for 224G Ethernet errors"""
    pass

class TrainingError(Eth224GError):
    """Error in link training"""
    pass

class ComplianceError(Eth224GError):
    """Error in compliance testing"""
    pass

def validate_measurements(
    results: Dict[str, Dict[str, float]],
    limits: Dict[str, float]
) -> bool:
    """
    Validate measurement results against limits
    
    Args:
        results: Measurement results
        limits: Test limits
        
    Returns:
        True if all measurements pass
        
    Raises:
        ValidationError: If results structure is invalid
    """
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

- [PAM4 Analysis API](pam4_analysis.md)
- [Mock Controller API](mock_controller.md)
- [224G Validation Tutorial](../tutorials/224g_validation.md)

---