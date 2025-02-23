# Instrument Setup Guide

## Overview

This guide covers instrument setup for both:
- Physical hardware configuration
- Mock testing environment

## Hardware Requirements

### Minimum Requirements

For 224G Ethernet validation:
- Oscilloscope (120+ GHz bandwidth)
- Pattern Generator (PAM4 capable)
- GPIB/USB interface

### Recommended Equipment

```python
RECOMMENDED_EQUIPMENT = {
    'oscilloscope': {
        'bandwidth': 120e9,      # 120 GHz
        'sample_rate': 256e9,    # 256 GSa/s
        'channels': 4,           # Minimum channels
        'memory': 1e6           # Minimum points
    },
    'pattern_generator': {
        'data_rate': 112e9,     # 112 GBaud
        'modulation': 'PAM4',    # PAM4 support
        'patterns': [
            'PRBS7',
            'PRBS13',
            'PRBS31'
        ]
    }
}
```

## Physical Setup

### GPIB Configuration

1. Address Assignment:
```python
def configure_gpib_addresses(
    scope_address: int = 7,
    pattern_gen_address: int = 10
) -> None:
    """
    Configure GPIB addresses
    
    Args:
        scope_address: Scope GPIB address
        pattern_gen_address: Pattern generator address
        
    Raises:
        ValueError: If addresses are invalid
    """
    # Validate addresses
    assert isinstance(scope_address, int), "Scope address must be integer"
    assert isinstance(pattern_gen_address, int), "Pattern gen address must be integer"
    assert 0 <= scope_address <= 30, "Invalid scope address"
    assert 0 <= pattern_gen_address <= 30, "Invalid pattern gen address"
    assert scope_address != pattern_gen_address, "Addresses must be unique"
    
    print(f"Configure scope to address {scope_address}")
    print(f"Configure pattern generator to address {pattern_gen_address}")
```

2. Connection Verification:
```python
from typing import Dict, List
import time

def verify_gpib_connections(
    addresses: List[int],
    timeout_ms: float = 1000.0
) -> Dict[int, bool]:
    """
    Verify GPIB connections
    
    Args:
        addresses: List of GPIB addresses
        timeout_ms: Query timeout in milliseconds
        
    Returns:
        Dictionary mapping addresses to connection status
    """
    # Validate inputs
    assert isinstance(timeout_ms, float), "Timeout must be float"
    assert timeout_ms > 0, "Timeout must be positive"
    
    results = {}
    for addr in addresses:
        try:
            resource = f"GPIB0::{addr}::INSTR"
            controller.connect_instrument(resource)
            response = controller.query_instrument(resource, '*IDN?')
            results[addr] = True
            print(f"Address {addr}: {response}")
        except Exception as e:
            print(f"Address {addr} failed: {e}")
            results[addr] = False
    return results
```

### Signal Connections

1. Cable Requirements:
```python
def validate_cable_specs(
    bandwidth_ghz: float,
    length_m: float,
    loss_db_per_m: float
) -> bool:
    """
    Validate cable specifications
    
    Args:
        bandwidth_ghz: Cable bandwidth in GHz
        length_m: Cable length in meters
        loss_db_per_m: Loss per meter in dB
        
    Returns:
        True if specifications are adequate
    """
    # Validate inputs
    assert isinstance(bandwidth_ghz, float), "Bandwidth must be float"
    assert isinstance(length_m, float), "Length must be float"
    assert isinstance(loss_db_per_m, float), "Loss must be float"
    
    # Check specifications
    total_loss = length_m * loss_db_per_m
    return (
        bandwidth_ghz >= 120.0 and    # Minimum 120 GHz
        length_m <= 1.0 and           # Maximum 1 meter
        total_loss <= 3.0             # Maximum 3 dB loss
    )
```

2. Connection Diagram:
```
[Scope]---(Ch1)--[DUT]---(Ch2)---[Pattern Gen]
           (Trig)----------(Trig)
```

## Mock Setup

### Environment Configuration

1. Force Mock Mode:
```python
import os
from typing import Optional

def setup_mock_environment(
    debug_level: Optional[str] = None
) -> None:
    """
    Configure environment for mock testing
    
    Args:
        debug_level: Optional debug level override
    """
    # Set mock mode
    os.environ['SVF_MOCK_MODE'] = '1'
    
    # Configure debug level
    if debug_level:
        assert debug_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR'], \
            "Invalid debug level"
        os.environ['SVF_DEBUG_LEVEL'] = debug_level
```

2. Mock Response Configuration:
```python
def configure_mock_responses(
    controller: Any,
    noise_level: float = 0.05
) -> None:
    """
    Configure realistic mock responses
    
    Args:
        controller: Mock controller instance
        noise_level: Noise amplitude (0-1)
    """
    # Validate input
    assert isinstance(noise_level, float), "Noise level must be float"
    assert 0 <= noise_level <= 1, "Noise level must be between 0 and 1"
    
    # Add responses
    controller.add_mock_response(
        '*IDN?',
        'Mock Instrument v1.0',
        delay=0.1
    )
    
    controller.add_mock_response(
        'MEASure:VOLTage:DC?',
        lambda: f"{np.random.normal(0, noise_level):.6f}",
        delay=0.2
    )
    
    controller.add_mock_response(
        ':WAVeform:DATA?',
        lambda: generate_pam4_waveform(noise_level),
        delay=0.5
    )
```

### Mock Instrument Creation

```python
def create_mock_instruments(
    scope_config: Dict[str, float],
    pattern_gen_config: Dict[str, Any]
) -> Tuple[Any, Any]:
    """
    Create mock instrument instances
    
    Args:
        scope_config: Scope configuration
        pattern_gen_config: Pattern generator configuration
        
    Returns:
        Tuple of (scope, pattern generator)
    """
    # Validate configurations
    required_scope = {'sampling_rate', 'bandwidth'}
    required_pg = {'data_rate', 'patterns'}
    
    missing_scope = required_scope - set(scope_config.keys())
    missing_pg = required_pg - set(pattern_gen_config.keys())
    
    if missing_scope or missing_pg:
        raise ValueError(
            f"Missing configuration: scope={missing_scope}, pg={missing_pg}"
        )
    
    try:
        # Create scope
        scope = HighBandwidthScope(
            'GPIB0::7::INSTR',
            config=ScopeConfig(**scope_config)
        )
        
        # Create pattern generator
        pattern_gen = MockPatternGenerator(
            'GPIB0::10::INSTR',
            **pattern_gen_config
        )
        
        return scope, pattern_gen
        
    except Exception as e:
        raise RuntimeError(f"Failed to create mock instruments: {e}")
```

## Validation

### Hardware Validation

```python
def validate_hardware_setup(
    scope_address: str,
    pattern_gen_address: str
) -> bool:
    """
    Validate physical hardware setup
    
    Args:
        scope_address: Scope VISA address
        pattern_gen_address: Pattern generator address
        
    Returns:
        True if setup is valid
    """
    try:
        # Check scope
        scope = HighBandwidthScope(scope_address)
        scope_info = scope.query_instrument('*IDN?')
        print(f"Scope: {scope_info}")
        
        # Check pattern generator
        pg = PatternGenerator(pattern_gen_address)
        pg_info = pg.query_instrument('*IDN?')
        print(f"Pattern Generator: {pg_info}")
        
        # Test basic communication
        scope.send_command('*RST')
        pg.send_command('*RST')
        
        return True
        
    except Exception as e:
        print(f"Hardware validation failed: {e}")
        return False
```

### Mock Validation

```python
def validate_mock_setup(
    scope: Any,
    pattern_gen: Any
) -> bool:
    """
    Validate mock setup
    
    Args:
        scope: Mock scope instance
        pattern_gen: Mock pattern generator instance
        
    Returns:
        True if setup is valid
    """
    try:
        # Verify mock mode
        assert scope.controller.get_mode() == 'mock', \
            "Scope not in mock mode"
        assert pattern_gen.controller.get_mode() == 'mock', \
            "Pattern generator not in mock mode"
        
        # Test responses
        scope_response = scope.query_instrument('*IDN?')
        pg_response = pattern_gen.query_instrument('*IDN?')
        
        print(f"Mock scope: {scope_response}")
        print(f"Mock pattern generator: {pg_response}")
        
        return True
        
    except Exception as e:
        print(f"Mock validation failed: {e}")
        return False
```

## Common Issues

### Hardware Issues

1. GPIB Connection Failures:
```python
def diagnose_gpib_issues(
    address: int,
    timeout_ms: float = 1000.0
) -> str:
    """
    Diagnose GPIB connection issues
    
    Args:
        address: GPIB address
        timeout_ms: Query timeout
        
    Returns:
        Diagnostic message
    """
    try:
        resource = f"GPIB0::{address}::INSTR"
        controller.connect_instrument(resource)
        return "Connection OK"
    except Exception as e:
        if "timeout" in str(e).lower():
            return "Device not responding - check power and cables"
        elif "not found" in str(e).lower():
            return "Device not found - check address configuration"
        else:
            return f"Unknown error: {e}"
```

2. Signal Quality Issues:
```python
def check_signal_quality(
    voltage_data: np.ndarray,
    min_amplitude: float = 0.1,
    max_noise: float = 0.05
) -> Dict[str, bool]:
    """
    Check basic signal quality
    
    Args:
        voltage_data: Voltage measurements
        min_amplitude: Minimum peak-to-peak amplitude
        max_noise: Maximum noise level
        
    Returns:
        Dictionary of quality checks
    """
    # Validate inputs
    assert isinstance(min_amplitude, float), "Min amplitude must be float"
    assert isinstance(max_noise, float), "Max noise must be float"
    
    # Calculate metrics
    peak_to_peak = np.ptp(voltage_data)
    noise_level = np.std(voltage_data)
    
    return {
        'amplitude_ok': peak_to_peak >= min_amplitude,
        'noise_ok': noise_level <= max_noise
    }
```

### Mock Issues

1. Configuration Problems:
```python
def check_mock_config() -> Dict[str, bool]:
    """
    Check mock configuration issues
    
    Returns:
        Dictionary of configuration checks
    """
    return {
        'mock_mode': os.environ.get('SVF_MOCK_MODE') == '1',
        'debug_level': os.environ.get('SVF_DEBUG_LEVEL') in [
            'DEBUG', 'INFO', 'WARNING', 'ERROR'
        ],
        'buffer_size': is_valid_buffer_size(
            os.environ.get('SVF_MAX_BUFFER_SIZE')
        )
    }
```

2. Response Issues:
```python
def verify_mock_responses(
    controller: Any
) -> Dict[str, bool]:
    """
    Verify mock response configuration
    
    Args:
        controller: Mock controller instance
        
    Returns:
        Dictionary of response checks
    """
    required_responses = [
        '*IDN?',
        '*RST',
        'MEASure:VOLTage:DC?',
        ':WAVeform:DATA?'
    ]
    
    return {
        cmd: cmd in controller.mock_responses
        for cmd in required_responses
    }
```

## See Also

- [Environment Configuration Guide](environment_vars.md)
- [Troubleshooting Guide](troubleshooting.md)
- [Mock Testing Tutorial](../tutorials/mock_testing.md)

---