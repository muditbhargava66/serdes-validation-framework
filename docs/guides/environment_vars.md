# Environment Configuration Guide

## Overview

The SerDes Validation Framework uses environment variables to control:
- Hardware/mock mode selection
- Debug settings
- Test configurations
- Performance tuning

## Core Environment Variables

### Mock Mode Control

```bash
# Force mock mode (no hardware required)
export SVF_MOCK_MODE=1

# Force real hardware mode
export SVF_MOCK_MODE=0

# Auto-detect mode (default)
unset SVF_MOCK_MODE
```

### Script Usage

```python
from typing import Optional
import os
from enum import Enum

class InstrumentMode(Enum):
    """Valid instrument modes"""
    REAL = 'real'
    MOCK = 'mock'
    AUTO = 'auto'

def validate_mock_mode(mode_value: Optional[str]) -> None:
    """
    Validate mock mode environment variable
    
    Args:
        mode_value: Environment variable value
        
    Raises:
        ValueError: If value is invalid
    """
    if mode_value is not None:
        try:
            mode_int = int(mode_value)
            if mode_int not in [0, 1]:
                raise ValueError(
                    f"SVF_MOCK_MODE must be 0 or 1, got {mode_value}"
                )
        except ValueError:
            raise ValueError(
                f"Invalid SVF_MOCK_MODE value: {mode_value}"
            )

def get_instrument_mode() -> InstrumentMode:
    """
    Get current instrument mode
    
    Returns:
        InstrumentMode enum value
    """
    mode_value = os.environ.get('SVF_MOCK_MODE')
    
    try:
        validate_mock_mode(mode_value)
        
        if mode_value is not None:
            return InstrumentMode.MOCK if int(mode_value) == 1 \
                else InstrumentMode.REAL
        
        return InstrumentMode.AUTO
        
    except ValueError as e:
        print(f"Warning: {e}, using auto mode")
        return InstrumentMode.AUTO
```

## Test Configuration

### Debug Level

```bash
# Set debug level
export SVF_DEBUG_LEVEL=DEBUG    # Most verbose
export SVF_DEBUG_LEVEL=INFO     # Normal operation
export SVF_DEBUG_LEVEL=WARNING  # Warnings and errors only
export SVF_DEBUG_LEVEL=ERROR    # Errors only
```

### Python Usage

```python
import logging
from typing import Dict, Optional

def configure_logging(level: Optional[str] = None) -> None:
    """
    Configure logging based on environment
    
    Args:
        level: Optional override level
    """
    # Get level from environment or parameter
    log_level = level or os.environ.get('SVF_DEBUG_LEVEL', 'INFO')
    
    # Validate level
    valid_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    
    if log_level not in valid_levels:
        print(f"Invalid debug level: {log_level}, using INFO")
        log_level = 'INFO'
    
    # Configure logging
    logging.basicConfig(
        level=valid_levels[log_level],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
```

## Performance Configuration

### Memory Settings

```bash
# Control buffer sizes
export SVF_MAX_BUFFER_SIZE=1000000  # Maximum buffer size
export SVF_MIN_BUFFER_SIZE=1000     # Minimum buffer size

# Control data precision
export SVF_DATA_PRECISION=64  # Use 64-bit floats
export SVF_DATA_PRECISION=32  # Use 32-bit floats
```

### Implementation

```python
def validate_numeric_env(
    name: str,
    min_value: float,
    max_value: float
) -> float:
    """
    Validate numeric environment variable
    
    Args:
        name: Variable name
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If validation fails
    """
    value = os.environ.get(name)
    if value is not None:
        try:
            float_value = float(value)
            if not min_value <= float_value <= max_value:
                raise ValueError(
                    f"{name} must be between {min_value} and {max_value}"
                )
            return float_value
        except ValueError:
            raise ValueError(f"Invalid {name} value: {value}")
    return min_value  # Default to minimum

def get_buffer_config() -> Dict[str, int]:
    """
    Get buffer configuration from environment
    
    Returns:
        Dictionary of buffer settings
    """
    try:
        max_size = int(validate_numeric_env(
            'SVF_MAX_BUFFER_SIZE',
            1000,
            10000000
        ))
        min_size = int(validate_numeric_env(
            'SVF_MIN_BUFFER_SIZE',
            100,
            max_size
        ))
        return {
            'max_size': max_size,
            'min_size': min_size
        }
    except ValueError as e:
        print(f"Warning: {e}, using defaults")
        return {
            'max_size': 1000000,
            'min_size': 1000
        }
```

## Hardware Configuration

### GPIB Settings

```bash
# GPIB configuration
export SVF_GPIB_TIMEOUT=10000  # Timeout in milliseconds
export SVF_GPIB_ADDRESS=1      # Default GPIB address
```

### Validation

```python
def validate_gpib_config() -> Dict[str, int]:
    """
    Validate GPIB configuration
    
    Returns:
        Dictionary of GPIB settings
    """
    try:
        timeout = int(validate_numeric_env(
            'SVF_GPIB_TIMEOUT',
            1000,
            60000
        ))
        address = int(validate_numeric_env(
            'SVF_GPIB_ADDRESS',
            0,
            30
        ))
        return {
            'timeout': timeout,
            'address': address
        }
    except ValueError as e:
        print(f"Warning: {e}, using defaults")
        return {
            'timeout': 10000,
            'address': 1
        }
```

## Complete Example

### Environment Setup

```bash
#!/bin/bash

# Mock mode configuration
export SVF_MOCK_MODE=1

# Debug configuration
export SVF_DEBUG_LEVEL=DEBUG

# Performance settings
export SVF_MAX_BUFFER_SIZE=1000000
export SVF_MIN_BUFFER_SIZE=1000
export SVF_DATA_PRECISION=64

# Hardware settings
export SVF_GPIB_TIMEOUT=10000
export SVF_GPIB_ADDRESS=1

# Run validation script
python validation_script.py
```

### Python Implementation

```python
def configure_environment() -> Dict[str, Any]:
    """
    Configure complete environment
    
    Returns:
        Dictionary of all settings
    """
    try:
        # Get instrument mode
        mode = get_instrument_mode()
        
        # Configure logging
        configure_logging()
        
        # Get configurations
        buffer_config = get_buffer_config()
        gpib_config = validate_gpib_config()
        
        return {
            'mode': mode,
            'buffer': buffer_config,
            'gpib': gpib_config
        }
        
    except Exception as e:
        logging.error(f"Environment configuration failed: {e}")
        raise
```

## Best Practices

1. Always validate environment variables:
   ```python
   value = os.environ.get('SVF_VARIABLE')
   if value is not None:
       try:
           validated = validate_value(value)
       except ValueError as e:
           logging.warning(f"Invalid value: {e}")
           validated = default_value
   ```

2. Use type hints and assertions:
   ```python
   def validate_value(value: str) -> float:
       """Validate numeric value"""
       try:
           float_value = float(value)
           assert float_value > 0, "Value must be positive"
           return float_value
       except (ValueError, AssertionError) as e:
           raise ValueError(f"Validation failed: {e}")
   ```

3. Handle missing variables gracefully:
   ```python
   def get_env_value(
       name: str,
       default: Any
   ) -> Any:
       """Get environment value with default"""
       return os.environ.get(name, default)
   ```

4. Document all environment variables:
   ```python
   ENVIRONMENT_VARS = {
       'SVF_MOCK_MODE': 'Control mock/real mode (0/1)',
       'SVF_DEBUG_LEVEL': 'Set debug level',
       'SVF_MAX_BUFFER_SIZE': 'Maximum buffer size',
       # ...
   }
   ```

## See Also

- [Mock Testing Guide](mock_testing.md)
- [Instrument Setup Guide](instrument_setup.md)
- [Troubleshooting Guide](troubleshooting.md)

---