# Installation Guide

## Overview

This guide covers:
- Basic installation
- Development setup
- Mock mode configuration
- Hardware requirements (optional)

## Prerequisites

### Required Software

- Python 3.10 or higher
- pip package manager
- git (for source installation)

### Optional Hardware

For physical hardware testing:
- GPIB interface card/adapter
- NI-VISA or equivalent driver
- Lab instruments

## Basic Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/muditbhargava66/serdes-validation-framework.git
   cd serdes-validation-framework
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```python
   from serdes_validation_framework import get_instrument_controller
   
   # Initialize controller
   controller = get_instrument_controller()
   print(f"Controller mode: {controller.get_mode()}")
   ```

## Mock Mode Setup

### Environment Configuration

1. **Force mock mode (recommended for development):**
   ```bash
   # Linux/macOS
   export SVF_MOCK_MODE=1
   
   # Windows (Command Prompt)
   set SVF_MOCK_MODE=1
   
   # Windows (PowerShell)
   $env:SVF_MOCK_MODE = "1"
   ```

2. **Configure debug level:**
   ```bash
   export SVF_DEBUG_LEVEL=DEBUG  # Most verbose
   export SVF_DEBUG_LEVEL=INFO   # Normal operation
   ```

### Python Configuration

```python
import os
from typing import Optional

def setup_development_environment(
    mock_mode: bool = True,
    debug_level: Optional[str] = None
) -> None:
    """
    Configure development environment
    
    Args:
        mock_mode: Enable mock mode if True
        debug_level: Optional debug level
    """
    # Set mock mode
    os.environ['SVF_MOCK_MODE'] = '1' if mock_mode else '0'
    
    # Set debug level
    if debug_level:
        assert debug_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR'], \
            "Invalid debug level"
        os.environ['SVF_DEBUG_LEVEL'] = debug_level
```

## Development Installation

1. **Install development dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Configure pre-commit hooks:**
   ```bash
   pre-commit install
   ```

3. **Install type checking tools:**
   ```bash
   pip install mypy
   ```

4. **Set up mock environment:**
   ```python
   # setup_dev.py
   from serdes_validation_framework import setup_mock_environment
   
   setup_mock_environment(
       mock_mode=True,
       debug_level='DEBUG'
   )
   ```

## Hardware Setup (Optional)

For physical hardware testing:

1. **Install VISA drivers:**
   - Download NI-VISA from National Instruments website
   - Install with GPIB support
   - Verify installation:
     ```python
     import pyvisa
     rm = pyvisa.ResourceManager()
     print(rm.list_resources())
     ```

2. **Configure GPIB:**
   ```python
   def verify_gpib_setup() -> bool:
       """
       Verify GPIB configuration
       
       Returns:
           True if GPIB is properly configured
       """
       try:
           import pyvisa
           rm = pyvisa.ResourceManager()
           resources = rm.list_resources()
           return any('GPIB' in r for r in resources)
       except Exception:
           return False
   ```

## Version Compatibility

### Python Versions

| Version | Python | Dependencies |
|---------|--------|--------------|
| 1.2.0   | ≥3.10  | numpy ≥1.21  |
| 1.1.0   | ≥3.10  | numpy ≥1.20  |
| 1.0.0   | ≥3.7   | numpy ≥1.19  |

### Hardware Compatibility

| Equipment     | Requirements         | Mock Support |
|--------------|---------------------|--------------|
| Oscilloscope | 120+ GHz bandwidth | ✓ Full      |
| Pattern Gen  | PAM4 capable       | ✓ Full      |
| BERT         | 224G support       | ✓ Basic     |

## Configuration

### Mock Configuration

```python
from typing import Dict, Optional

def configure_mock_testing(
    config: Dict[str, float]
) -> None:
    """
    Configure mock testing environment
    
    Args:
        config: Configuration parameters
    """
    # Validate configuration
    required = {'noise_amplitude', 'delay_ms'}
    missing = required - set(config.keys())
    if missing:
        raise ValueError(f"Missing configuration: {missing}")
    
    # Set environment
    os.environ['SVF_MOCK_MODE'] = '1'
    
    # Configure mock responses
    controller = get_instrument_controller()
    controller.add_mock_response(
        '*IDN?',
        'Mock Instrument v1.0',
        delay=config['delay_ms']/1000.0
    )
```

### Hardware Configuration

```python
def configure_hardware_testing(
    gpib_timeout: float = 1000.0
) -> None:
    """
    Configure hardware testing environment
    
    Args:
        gpib_timeout: GPIB timeout in milliseconds
    """
    # Validate input
    assert isinstance(gpib_timeout, float), "Timeout must be float"
    assert gpib_timeout > 0, "Timeout must be positive"
    
    # Set environment
    os.environ['SVF_MOCK_MODE'] = '0'
    os.environ['SVF_GPIB_TIMEOUT'] = f"{gpib_timeout}"
```

## Troubleshooting

### Mock Mode Issues

1. **Environment Variables:**
   ```python
   def check_environment() -> Dict[str, bool]:
       """Check environment configuration"""
       return {
           'mock_mode': os.environ.get('SVF_MOCK_MODE') in ['0', '1'],
           'debug_level': os.environ.get('SVF_DEBUG_LEVEL') in [
               'DEBUG', 'INFO', 'WARNING', 'ERROR'
           ]
       }
   ```

2. **Import Issues:**
   ```python
   def verify_installation() -> Dict[str, bool]:
       """Verify package installation"""
       try:
           import serdes_validation_framework
           from serdes_validation_framework import (
               get_instrument_controller,
               MockInstrumentController
           )
           return {'status': True}
       except ImportError as e:
           return {'status': False, 'error': str(e)}
   ```

### Hardware Issues

1. **VISA Problems:**
   ```python
   def diagnose_visa_issues() -> str:
       """Diagnose VISA installation issues"""
       try:
           import pyvisa
           rm = pyvisa.ResourceManager()
           resources = rm.list_resources()
           return f"Found resources: {resources}"
       except Exception as e:
           return f"VISA error: {e}"
   ```

2. **GPIB Issues:**
   ```python
   def check_gpib_connection(
       address: int
   ) -> Dict[str, Any]:
       """
       Check GPIB connection
       
       Args:
           address: GPIB address to check
       """
       assert isinstance(address, int), "Address must be integer"
       assert 0 <= address <= 30, "Invalid GPIB address"
       
       try:
           resource = f"GPIB0::{address}::INSTR"
           controller = get_instrument_controller()
           controller.connect_instrument(resource)
           return {'status': 'connected'}
       except Exception as e:
           return {'status': 'failed', 'error': str(e)}
   ```

## Next Steps

1. Read the [Usage Guide](USAGE.md)
2. Try the [Mock Testing Tutorial](tutorials/mock_testing.md)
3. Check the [API Documentation](api/index.md)

## See Also

- [Contributing Guide](CONTRIBUTING.md)
- [Environment Configuration](guides/environment_vars.md)
- [Troubleshooting Guide](guides/troubleshooting.md)

---