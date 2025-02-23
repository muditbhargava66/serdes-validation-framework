# Getting Started with SerDes Validation Framework

## Overview

This tutorial will guide you through:
1. Setting up your environment
2. Basic framework usage
3. Mock testing for development
4. Running your first validation tests

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- git (for source installation)

### Setup Steps

1. **Clone the repository:**
```bash
git clone https://github.com/muditbhargava66/serdes-validation-framework.git
cd serdes-validation-framework
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Basic Usage

### Controller Setup

The framework supports both real hardware and mock testing modes:

```python
from typing import Dict, Optional, Union
import numpy as np
from serdes_validation_framework.instrument_control import get_instrument_controller

# Initialize controller (auto-detects mode)
controller = get_instrument_controller()

# Check current mode
print(f"Operating in {controller.get_mode()} mode")
```

### Instrument Connection

```python
def connect_to_instrument(resource_name: str) -> None:
    """
    Connect to test instrument
    
    Args:
        resource_name: VISA resource identifier
        
    Raises:
        ConnectionError: If connection fails
    """
    try:
        controller.connect_instrument(resource_name)
        
        # Verify connection
        response = controller.query_instrument(resource_name, '*IDN?')
        print(f"Connected to: {response}")
        
    except Exception as e:
        raise ConnectionError(f"Failed to connect to {resource_name}: {e}")
```

## Mock Testing

### Enabling Mock Mode

For development without hardware:

```python
# Method 1: Environment variable
import os
os.environ['SVF_MOCK_MODE'] = '1'

# Method 2: Command line
# Run your script with:
# SVF_MOCK_MODE=1 python your_script.py
```

### Basic Mock Example

```python
def run_basic_test() -> None:
    """Demonstrate basic framework usage"""
    # Force mock mode for example
    os.environ['SVF_MOCK_MODE'] = '1'
    
    try:
        # Initialize
        controller = get_instrument_controller()
        
        # Connect
        resource = 'GPIB::1::INSTR'
        controller.connect_instrument(resource)
        
        # Send commands
        controller.send_command(resource, '*RST')
        response = controller.query_instrument(resource, '*IDN?')
        print(f"Instrument response: {response}")
        
    finally:
        # Cleanup
        controller.disconnect_instrument(resource)
```

### Custom Mock Responses

```python
def configure_mock_measurements() -> None:
    """Configure custom mock responses"""
    controller = get_instrument_controller()
    
    # Add voltage measurement response
    controller.add_mock_response(
        'MEASure:VOLTage:DC?',
        lambda: f"{np.random.normal(1.0, 0.1):.6f}",
        delay=0.1
    )
    
    # Add temperature measurement
    controller.add_mock_response(
        'MEASure:TEMPerature?',
        lambda: f"{np.random.normal(25.0, 0.5):.2f}",
        delay=0.2
    )
```

## Data Collection

### Basic Measurement

```python
def measure_voltage(channel: int) -> float:
    """
    Measure DC voltage
    
    Args:
        channel: Channel number
        
    Returns:
        Measured voltage in volts
    """
    # Validate input
    assert isinstance(channel, int), "Channel must be integer"
    assert channel > 0, "Channel must be positive"
    
    # Get voltage reading
    response = controller.query_instrument(
        'GPIB::1::INSTR',
        f':MEASure:VOLTage:DC? (@{channel})'
    )
    
    try:
        return float(response)
    except ValueError:
        raise ValueError(f"Invalid voltage response: {response}")
```

### Waveform Capture

```python
def capture_waveform(
    duration: float,
    sample_rate: float
) -> Dict[str, np.ndarray]:
    """
    Capture waveform data
    
    Args:
        duration: Capture duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Dictionary with time and voltage arrays
    """
    # Validate inputs
    assert isinstance(duration, float), "Duration must be float"
    assert isinstance(sample_rate, float), "Sample rate must be float"
    assert duration > 0, "Duration must be positive"
    assert sample_rate > 0, "Sample rate must be positive"
    
    try:
        # Configure acquisition
        num_points = int(duration * sample_rate)
        controller.send_command(
            'GPIB::1::INSTR',
            f':ACQuire:POINts {num_points}'
        )
        
        # Get data
        response = controller.query_instrument(
            'GPIB::1::INSTR',
            ':WAVeform:DATA?'
        )
        
        # Convert to arrays
        voltage = np.array(response.split(','), dtype=np.float64)
        time = np.arange(len(voltage)) / sample_rate
        
        return {
            'time': time,
            'voltage': voltage
        }
        
    except Exception as e:
        raise RuntimeError(f"Waveform capture failed: {e}")
```

## Data Analysis

### Basic Statistics

```python
def analyze_measurements(
    values: np.ndarray,
    name: str = "data"
) -> Dict[str, float]:
    """
    Calculate basic statistics
    
    Args:
        values: Measurement values array
        name: Data name for error messages
        
    Returns:
        Dictionary of statistics
    """
    # Validate input
    assert isinstance(values, np.ndarray), \
        f"{name} must be numpy array"
    assert np.issubdtype(values.dtype, np.floating), \
        f"{name} must be floating-point"
    assert len(values) > 0, f"{name} cannot be empty"
    
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }
```

## Complete Example

### PAM4 Signal Analysis

```python
from serdes_validation_framework.data_analysis import PAM4Analyzer

def analyze_pam4_signal() -> None:
    """Demonstrate complete PAM4 analysis"""
    # Force mock mode for example
    os.environ['SVF_MOCK_MODE'] = '1'
    
    try:
        # Capture data
        data = capture_waveform(
            duration=1e-6,    # 1 µs
            sample_rate=256e9  # 256 GSa/s
        )
        
        # Create analyzer
        analyzer = PAM4Analyzer(data)
        
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
        
    except Exception as e:
        print(f"Analysis failed: {e}")
```

## Next Steps

- [Mock Testing Tutorial](mock_testing.md) - Detailed mock testing guide
- [224G Ethernet Tutorial](224g_validation.md) - Protocol-specific validation
- [PAM4 Analysis Tutorial](pam4_analysis.md) - Advanced signal analysis

## Troubleshooting

### Common Issues

1. **Mode Selection:**
   ```python
   # Check current mode
   mode = controller.get_mode()
   if mode != 'mock':
       print("Warning: Using real hardware mode")
   ```

2. **Connection Problems:**
   ```python
   try:
       controller.connect_instrument(resource_name)
   except ConnectionError as e:
       print(f"Connection failed: {e}")
       # Try alternative resource or mock mode
   ```

3. **Data Validation:**
   ```python
   # Validate numeric data
   if not np.issubdtype(data.dtype, np.floating):
       raise TypeError(f"Expected float data, got {data.dtype}")
   ```

### Getting Help

- Check the [API Documentation](../api/index.md)
- Review the [Troubleshooting Guide](../guides/troubleshooting.md)
- File issues on GitHub

## See Also

- [Mock Controller Documentation](../api/mock_controller.md)
- [PAM4 Analysis Documentation](../api/pam4_analysis.md)
- [Environment Configuration](../guides/environment_vars.md)