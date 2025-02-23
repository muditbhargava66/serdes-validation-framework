# Troubleshooting Guide

## Overview

This guide covers common issues and solutions for:
- Hardware configuration problems
- Mock testing issues
- Data analysis errors
- Type validation failures

## Diagnostic Tools

### System Check

```python
from typing import Dict, Any, Optional
import os
import numpy as np

def run_system_check() -> Dict[str, bool]:
    """
    Run comprehensive system check
    
    Returns:
        Dictionary of check results
    """
    results = {}
    
    # Check environment
    results['environment'] = {
        'mock_mode': os.environ.get('SVF_MOCK_MODE') in ['0', '1'],
        'debug_level': os.environ.get('SVF_DEBUG_LEVEL') in [
            'DEBUG', 'INFO', 'WARNING', 'ERROR'
        ]
    }
    
    # Check Python version
    import sys
    results['python'] = {
        'version': sys.version_info >= (3, 10),
        'numpy': 'numpy' in sys.modules,
        'scipy': 'scipy' in sys.modules
    }
    
    # Check hardware/mock status
    try:
        controller = get_instrument_controller()
        results['controller'] = {
            'initialized': True,
            'mode': controller.get_mode()
        }
    except Exception as e:
        results['controller'] = {
            'initialized': False,
            'error': str(e)
        }
    
    return results
```

### Hardware Diagnostics

```python
def diagnose_hardware_issues(
    resource_name: str,
    timeout_ms: float = 1000.0
) -> Dict[str, Any]:
    """
    Diagnose hardware connection issues
    
    Args:
        resource_name: VISA resource identifier
        timeout_ms: Query timeout in milliseconds
        
    Returns:
        Diagnostic results dictionary
    """
    # Validate inputs
    assert isinstance(timeout_ms, float), "Timeout must be float"
    assert timeout_ms > 0, "Timeout must be positive"
    
    results = {}
    
    try:
        # Check connection
        controller = get_instrument_controller()
        controller.connect_instrument(resource_name)
        results['connection'] = 'OK'
        
        # Basic queries
        try:
            id_response = controller.query_instrument(resource_name, '*IDN?')
            results['identification'] = id_response
        except Exception as e:
            results['identification'] = f"Failed: {e}"
            
        try:
            controller.send_command(resource_name, '*RST')
            results['reset'] = 'OK'
        except Exception as e:
            results['reset'] = f"Failed: {e}"
            
        return results
        
    except Exception as e:
        return {
            'connection': f"Failed: {e}",
            'identification': 'Not attempted',
            'reset': 'Not attempted'
        }
```

### Mock Testing Diagnostics

```python
def diagnose_mock_issues(
    controller: Any
) -> Dict[str, Any]:
    """
    Diagnose mock testing issues
    
    Args:
        controller: Mock controller instance
        
    Returns:
        Diagnostic results dictionary
    """
    results = {}
    
    # Check mock mode
    results['mode'] = {
        'current': controller.get_mode(),
        'forced': os.environ.get('SVF_MOCK_MODE') is not None
    }
    
    # Check responses
    required_responses = {
        '*IDN?': 'identification',
        '*RST': 'reset command',
        'MEASure:VOLTage:DC?': 'voltage measurement',
        ':WAVeform:DATA?': 'waveform data'
    }
    
    results['responses'] = {}
    for cmd, description in required_responses.items():
        if cmd in controller.mock_responses:
            response = controller.mock_responses[cmd]
            results['responses'][cmd] = {
                'configured': True,
                'has_delay': response.delay > 0,
                'has_error_rate': response.error_rate > 0
            }
        else:
            results['responses'][cmd] = {
                'configured': False,
                'missing': description
            }
    
    return results
```

## Common Issues

### Hardware Problems

1. Connection Failures:
```python
def check_connection_issues(
    resource_name: str
) -> str:
    """
    Check connection issues
    
    Args:
        resource_name: VISA resource identifier
        
    Returns:
        Diagnostic message
    """
    try:
        controller = get_instrument_controller()
        controller.connect_instrument(resource_name)
        return "Connection successful"
        
    except Exception as e:
        error_str = str(e).lower()
        
        if "timeout" in error_str:
            return (
                "Device not responding - Check:\n"
                "1. Power connection\n"
                "2. GPIB/USB cable\n"
                "3. Device address"
            )
        elif "not found" in error_str:
            return (
                "Device not found - Check:\n"
                "1. Resource name format\n"
                "2. Device address settings\n"
                "3. VISA installation"
            )
        else:
            return f"Unknown error: {e}"
```

2. Data Quality Issues:
```python
def check_data_quality(
    voltage_data: np.ndarray,
    sample_rate: float
) -> Dict[str, Any]:
    """
    Check data quality issues
    
    Args:
        voltage_data: Voltage measurements
        sample_rate: Sample rate in Hz
        
    Returns:
        Quality check results
    """
    # Validate inputs
    assert isinstance(sample_rate, float), "Sample rate must be float"
    assert sample_rate > 0, "Sample rate must be positive"
    
    results = {}
    
    # Check basic statistics
    results['statistics'] = {
        'mean': float(np.mean(voltage_data)),
        'std': float(np.std(voltage_data)),
        'min': float(np.min(voltage_data)),
        'max': float(np.max(voltage_data))
    }
    
    # Check for issues
    results['issues'] = {
        'has_nans': bool(np.any(np.isnan(voltage_data))),
        'has_infs': bool(np.any(np.isinf(voltage_data))),
        'low_amplitude': bool(np.ptp(voltage_data) < 0.1),
        'high_noise': bool(np.std(voltage_data) > 0.5)
    }
    
    return results
```

### Mock Testing Problems

1. Configuration Issues:
```python
def diagnose_mock_config() -> Dict[str, str]:
    """
    Diagnose mock configuration issues
    
    Returns:
        Dictionary of diagnostics
    """
    issues = {}
    
    # Check environment
    mock_mode = os.environ.get('SVF_MOCK_MODE')
    if mock_mode not in ['0', '1']:
        issues['mock_mode'] = (
            f"Invalid SVF_MOCK_MODE: {mock_mode}\n"
            "Must be '0' or '1'"
        )
    
    # Check debug level
    debug_level = os.environ.get('SVF_DEBUG_LEVEL', 'INFO')
    if debug_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
        issues['debug_level'] = (
            f"Invalid debug level: {debug_level}\n"
            "Must be DEBUG, INFO, WARNING, or ERROR"
        )
    
    return issues if issues else {'status': 'All configurations valid'}
```

2. Response Problems:
```python
def verify_mock_responses(
    controller: Any
) -> Dict[str, Any]:
    """
    Verify mock response configuration
    
    Args:
        controller: Mock controller instance
        
    Returns:
        Response verification results
    """
    results = {
        'mode': controller.get_mode(),
        'responses': {}
    }
    
    # Check basic responses
    for cmd in ['*IDN?', '*RST', 'MEASure:VOLTage:DC?']:
        if cmd in controller.mock_responses:
            response = controller.mock_responses[cmd]
            results['responses'][cmd] = {
                'configured': True,
                'type': 'static' if isinstance(response.value, str) else 'dynamic'
            }
        else:
            results['responses'][cmd] = {
                'configured': False,
                'missing': True
            }
    
    # Check waveform generation
    waveform_cmd = ':WAVeform:DATA?'
    if waveform_cmd in controller.mock_responses:
        try:
            data = controller.query_instrument('GPIB::1::INSTR', waveform_cmd)
            values = np.array(data.split(','), dtype=np.float64)
            results['waveform'] = {
                'points': len(values),
                'range': float(np.ptp(values)),
                'valid': True
            }
        except Exception as e:
            results['waveform'] = {
                'error': str(e),
                'valid': False
            }
    
    return results
```

### Data Analysis Issues

1. Type Validation Failures:
```python
def check_type_issues(
    data: Dict[str, Any]
) -> Dict[str, str]:
    """
    Check type validation issues
    
    Args:
        data: Data dictionary to check
        
    Returns:
        Dictionary of type issues
    """
    issues = {}
    
    for key, value in data.items():
        if key.endswith(('_ns', '_ps', '_ms')):
            if not isinstance(value, float):
                issues[key] = f"Must be float, got {type(value)}"
                
        elif key.endswith(('_v', '_mv')):
            if not isinstance(value, float):
                issues[key] = f"Must be float, got {type(value)}"
                
        elif key.endswith('_array'):
            if not isinstance(value, np.ndarray):
                issues[key] = f"Must be numpy array, got {type(value)}"
            elif not np.issubdtype(value.dtype, np.floating):
                issues[key] = f"Must be floating-point, got {value.dtype}"
    
    return issues
```

2. Analysis Pipeline Issues:
```python
def diagnose_analysis_pipeline(
    data: Dict[str, np.ndarray],
    analysis_steps: List[str]
) -> Dict[str, Any]:
    """
    Diagnose analysis pipeline issues
    
    Args:
        data: Input data dictionary
        analysis_steps: List of analysis steps
        
    Returns:
        Pipeline diagnostics
    """
    results = {'steps': {}}
    
    try:
        # Check input data
        for name, array in data.items():
            results['input'] = check_type_issues({f"{name}_array": array})
        
        # Run pipeline steps
        analyzer = PAM4Analyzer(data)
        
        for step in analysis_steps:
            try:
                if step == 'levels':
                    result = analyzer.analyze_level_separation()
                    results['steps'][step] = {
                        'status': 'OK',
                        'levels': len(result.level_means)
                    }
                    
                elif step == 'evm':
                    result = analyzer.calculate_evm()
                    results['steps'][step] = {
                        'status': 'OK',
                        'rms_evm': float(result.rms_evm_percent)
                    }
                    
                elif step == 'eye':
                    result = analyzer.analyze_eye_diagram()
                    results['steps'][step] = {
                        'status': 'OK',
                        'heights': len(result.eye_heights)
                    }
                    
            except Exception as e:
                results['steps'][step] = {
                    'status': 'Failed',
                    'error': str(e)
                }
        
        return results
        
    except Exception as e:
        return {
            'status': 'Failed',
            'error': str(e)
        }
```

## Solution Steps

### Hardware Solutions

1. Connection Problems:
   - Check physical connections
   - Verify GPIB/USB addresses
   - Test with simple commands
   - Increase timeout if needed

2. Data Quality Problems:
   - Check signal amplitude
   - Verify bandwidth settings
   - Adjust vertical scale
   - Check triggering

### Mock Testing Solutions

1. Configuration:
   - Set environment variables correctly
   - Verify mock mode setting
   - Check debug level

2. Response Issues:
   - Add missing responses
   - Verify response formats
   - Test with error simulation
   - Check data generation

### Type Validation Solutions

1. Data Types:
   - Convert numeric inputs to float
   - Use numpy float64 arrays
   - Validate array shapes
   - Check for NaN/Inf

2. Analysis Issues:
   - Verify input formats
   - Check array dimensions
   - Monitor signal quality
   - Validate results

## See Also

- [Instrument Setup Guide](instrument_setup.md)
- [Environment Configuration](environment_vars.md)
- [Mock Testing Tutorial](../tutorials/mock_testing.md)