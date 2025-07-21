# SerDes Validation Framework API Documentation

## Overview

The SerDes Validation Framework provides comprehensive tools for validating high-speed serial interfaces. This documentation covers the core APIs, type safety requirements, and integration patterns.

## Core Components

### Instrument Control

Core functionality for controlling test equipment:

- [Mock Controller](mock_controller.md)
  - Environment-controlled mock/real hardware switching
  - Type-safe mock response configuration
  - Error simulation capabilities
  - Custom response handlers

- [Instrument Control](instrument_control.md)
  - Base instrument control interfaces
  - GPIB/USB communication
  - Command validation
  - Error handling

### Signal Analysis

Signal processing and analysis tools:

- [PAM4 Analysis](pam4_analysis.md)
  - Level separation analysis
  - Error Vector Magnitude (EVM) calculation
  - Eye diagram measurements
  - Type-safe signal processing

- [224G Ethernet](eth_224g.md)
  - Protocol-specific measurements
  - Compliance testing
  - Link training
  - Performance validation

- [PCIe 6.0](pcie.md) **NEW**
  - Complete PCIe 6.0 specification support
  - NRZ/PAM4 dual-mode capabilities
  - Advanced link training and equalization
  - Comprehensive compliance testing
  - Multi-lane analysis with skew detection

## Type Safety

All APIs enforce strict type checking:

```python
from typing import Dict, List, Optional, Union
import numpy as np
import numpy.typing as npt

# Type aliases
FloatArray = npt.NDArray[np.float64]
SignalData = Dict[str, FloatArray]
Measurements = Dict[str, Union[float, List[float]]]

def validate_numeric_data(
    data: FloatArray,
    name: str = "data"
) -> None:
    """
    Validate numeric array properties
    
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

## Mock Testing Support

The framework provides comprehensive mock testing capabilities:

```python
from serdes_validation_framework.instrument_control.mock_controller import (
    get_instrument_controller,
    get_instrument_mode
)

# Control mock mode via environment
os.environ['SVF_MOCK_MODE'] = '1'  # Force mock mode
os.environ['SVF_MOCK_MODE'] = '0'  # Force real mode
# Auto-detect mode (default)
unset SVF_MOCK_MODE

# Initialize controller
controller = get_instrument_controller()
print(f"Operating in {controller.get_mode()} mode")
```

## Error Handling

Standardized error hierarchy:

```python
class ValidationError(Exception):
    """Base class for validation errors"""
    pass

class InstrumentError(Exception):
    """Base class for instrument errors"""
    pass

class AnalysisError(Exception):
    """Base class for analysis errors"""
    pass
```

## Data Classes

Type-safe data containers:

```python
from dataclasses import dataclass
from typing import List

@dataclass
class MeasurementResults:
    """Type-safe measurement results"""
    timestamp: float
    values: List[float]
    metadata: Dict[str, Union[str, float]]
    
    def __post_init__(self) -> None:
        """Validate measurement data"""
        assert isinstance(self.timestamp, float), \
            "Timestamp must be float"
        assert all(isinstance(x, float) for x in self.values), \
            "Values must be floats"
```

## Module Organization

```
serdes_validation_framework/
├── instrument_control/
│   ├── controller.py        # Base controller
│   ├── mock_controller.py   # Mock testing support
│   └── scope_224g.py       # High-bandwidth scope
├── data_analysis/
│   ├── analyzer.py         # Base analyzer
│   └── pam4_analyzer.py    # PAM4 signal analysis
├── protocols/
│   └── ethernet_224g/      # 224G protocol support
└── test_sequence/          # Test automation
```

## Quick Start

### Basic Usage

```python
from serdes_validation_framework.instrument_control import get_instrument_controller
from serdes_validation_framework.data_analysis import PAM4Analyzer

# Initialize controller
controller = get_instrument_controller()

# Connect to instrument
controller.connect_instrument('GPIB::1::INSTR')

# Collect data
response = controller.query_instrument(
    'GPIB::1::INSTR',
    ':WAVeform:DATA?'
)

# Analyze results
analyzer = PAM4Analyzer({
    'voltage': np.array(response.split(','), dtype=np.float64)
})
results = analyzer.analyze_level_separation()
```

### Mock Testing

```python
# Force mock mode
os.environ['SVF_MOCK_MODE'] = '1'

# Configure mock responses
controller = get_instrument_controller()
controller.add_mock_response(
    'TEST:MEASURE?',
    lambda: f"{np.random.normal(0, 0.1):.6f}",
    delay=0.1
)

# Use controller normally
response = controller.query_instrument('GPIB::1::INSTR', 'TEST:MEASURE?')
```

## Best Practices

1. Always validate numeric types:
   ```python
   assert isinstance(value, float), f"Expected float, got {type(value)}"
   ```

2. Use type hints:
   ```python
   def calculate_snr(signal: FloatArray, noise: FloatArray) -> float:
   ```

3. Handle cleanup properly:
   ```python
   try:
       controller.connect_instrument(resource_name)
       # ... perform operations
   finally:
       controller.disconnect_instrument(resource_name)
   ```

4. Use logging for debugging:
   ```python
   logger.debug(f"Measurement result: {value:.6f}")
   ```

## Examples

Complete examples are available in the tutorials:

- [Getting Started](../tutorials/getting_started.md)
- [Mock Testing Guide](../tutorials/mock_testing.md)
- [224G Validation](../tutorials/224g_validation.md)
- [PAM4 Analysis](../tutorials/pam4_analysis.md)

## See Also

- [Installation Guide](../INSTALL.md)
- [Usage Guide](../USAGE.md)
- [Contributing Guide](../CONTRIBUTING.md)