# Mock Controller

## Overview

The mock controller provides comprehensive simulation capabilities for instrument control during development and testing, eliminating the need for physical hardware. It ensures type safety through strict validation and offers configurable behavior via environment variables.

## Features

- Environment variable-based mode control
- Type-safe mock data generation
- Configurable error simulation
- Custom response configuration
- Runtime hardware detection

## Installation

The mock controller is included in the SerDes Validation Framework. No additional installation is required.

## Environment Configuration

Control mock mode via environment variables:

```bash
# Force mock mode (no hardware required)
export SVF_MOCK_MODE=1

# Force real hardware mode
export SVF_MOCK_MODE=0

# Auto-detect mode (default)
unset SVF_MOCK_MODE
```

## Basic Usage

```python
from serdes_validation_framework.instrument_control.mock_controller import (
    get_instrument_controller,
    get_instrument_mode,
    InstrumentMode
)

# Initialize controller (automatically detects mode)
controller = get_instrument_controller()

# Check current mode
mode = controller.get_mode()
print(f"Operating in {mode} mode")

# Basic operations
controller.connect_instrument('GPIB::1::INSTR')
response = controller.query_instrument('GPIB::1::INSTR', '*IDN?')
print(f"Response: {response}")
```

## Type Safety

The mock controller enforces strict type checking:

```python
def send_command(self, resource_name: str, command: str) -> None:
    """
    Send command to mock instrument with type validation
    
    Args:
        resource_name: VISA resource identifier
        command: SCPI command string
        
    Raises:
        TypeError: If arguments have invalid types
        ValueError: If instrument not connected
    """
    assert isinstance(resource_name, str), "Resource name must be a string"
    assert isinstance(command, str), "Command must be a string"
    
    if resource_name not in self.connected_instruments:
        raise ValueError(f"Instrument {resource_name} not connected")
```

## Mock Response Configuration

### Basic Responses

```python
# Add simple response
controller.add_mock_response(
    '*IDN?',
    'Mock Instrument v1.0',
    delay=0.1
)

# Add dynamic response
controller.add_mock_response(
    'MEASure:VOLTage:DC?',
    lambda: f"{np.random.normal(0, 0.1):.6f}",
    delay=0.2
)
```

### PAM4 Signal Generation

```python
# Configure PAM4 waveform generation
controller.add_mock_response(
    ':WAVeform:DATA?',
    lambda: generate_pam4_waveform(
        num_points=1000000,
        noise_amplitude=0.05
    ),
    delay=0.5
)

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
        Comma-separated waveform data string
    """
    assert isinstance(num_points, int), "num_points must be an integer"
    assert isinstance(noise_amplitude, float), "noise_amplitude must be a float"
    assert 0 <= noise_amplitude <= 1, "noise_amplitude must be between 0 and 1"
    
    # Generate PAM4 levels
    levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
    symbols = np.random.choice(levels, size=num_points)
    
    # Add noise
    noise = np.random.normal(0, noise_amplitude, num_points)
    waveform = symbols + noise
    
    return ','.join(f"{x:.8f}" for x in waveform)
```

## Error Simulation

Configure error rates for testing error handling:

```python
# Set error rates
controller.set_error_rates(
    connection_error_rate=0.1,  # 10% connection failures
    command_error_rate=0.05,    # 5% command failures
    data_error_rate=0.01       # 1% data corruption
)
```

## Advanced Usage

### Custom Error Handling

```python
try:
    controller.connect_instrument('GPIB::1::INSTR')
    controller.send_command('GPIB::1::INSTR', '*RST')
except MockConnectionError as e:
    print(f"Connection failed: {e}")
except MockCommandError as e:
    print(f"Command failed: {e}")
```

### Response Validation

```python
def validate_response(
    command: str,
    response: Union[str, Callable[[], str]],
    delay: float,
    error_rate: float
) -> None:
    """
    Validate mock response configuration
    
    Args:
        command: SCPI command string
        response: Response string or generator function
        delay: Response delay in seconds
        error_rate: Error probability (0-1)
        
    Raises:
        AssertionError: If parameters are invalid
    """
    assert isinstance(command, str), "Command must be a string"
    assert callable(response) or isinstance(response, str), \
        "Response must be string or callable"
    assert isinstance(delay, float), "Delay must be a float"
    assert isinstance(error_rate, float), "Error rate must be a float"
    assert delay >= 0, "Delay must be non-negative"
    assert 0 <= error_rate <= 1, "Error rate must be between 0 and 1"
```

## Best Practices

1. Always validate input types:
   ```python
   assert isinstance(value, float), f"Value must be float, got {type(value)}"
   ```

2. Use type hints:
   ```python
   def query_float(self, resource: str, query: str) -> float:
   ```

3. Handle cleanup properly:
   ```python
   try:
       controller.connect_instrument('GPIB::1::INSTR')
       # ... perform operations
   finally:
       controller.disconnect_instrument('GPIB::1::INSTR')
   ```

4. Use logging for debugging:
   ```python
   logger.debug(f"Mock response for {command}: {response}")
   ```

## API Reference

### Core Functions

#### get_instrument_controller()

```python
def get_instrument_controller():
    """
    Get appropriate instrument controller based on environment and availability
    
    Returns:
        Real or mock instrument controller
        
    Raises:
        RuntimeError: If real mode forced but hardware not available
    """
```

#### get_instrument_mode()

```python
def get_instrument_mode() -> InstrumentMode:
    """
    Determine instrument mode based on environment variables
    
    Environment Variables:
        SVF_MOCK_MODE: Control mock mode (1=mock, 0=real, unset=auto)
    
    Returns:
        InstrumentMode enum value
        
    Raises:
        ValueError: If SVF_MOCK_MODE contains invalid value
    """
```

### MockInstrumentController Class

```python
class MockInstrumentController:
    """
    Mock controller for GPIB instruments in test environments
    
    This class provides a mock implementation of instrument control functions
    for testing without physical hardware. It generates realistic synthetic
    data and simulates common instrument behaviors.
    """
    
    def connect_instrument(self, resource_name: str) -> None:
        """
        Simulate instrument connection
        
        Args:
            resource_name: VISA resource identifier
            
        Raises:
            MockConnectionError: If connection simulation fails
            ValueError: If resource name is invalid
        """
    
    def disconnect_instrument(self, resource_name: str) -> None:
        """
        Simulate instrument disconnection
        
        Args:
            resource_name: VISA resource identifier
            
        Raises:
            ValueError: If instrument not connected
        """
    
    def send_command(self, resource_name: str, command: str) -> None:
        """
        Simulate sending command to instrument
        
        Args:
            resource_name: VISA resource identifier
            command: SCPI command string
            
        Raises:
            ValueError: If instrument not connected
            MockCommandError: If command simulation fails
        """
    
    def query_instrument(self, resource_name: str, query: str) -> str:
        """
        Simulate querying instrument with realistic delays
        
        Args:
            resource_name: VISA resource identifier
            query: SCPI query string
            
        Returns:
            Simulated response string
            
        Raises:
            ValueError: If instrument not connected or query invalid
            MockCommandError: If query simulation fails
        """
```

## Testing

Run the built-in tests:

```bash
python -m src.serdes_validation_framework.instrument_control.mock_controller
```

Example test output:
```
Running mock controller tests...
Test 1: Basic connectivity... PASS
Test 2: Waveform generation... PASS
Test 3: Custom responses... PASS
Test 4: Error simulation... PASS
Test 5: Cleanup operations... PASS
All tests completed successfully!
```

## See Also

- [Instrument Control API](instrument_control.md)
- [Testing Guide](../guides/testing.md)
- [Environment Configuration](../guides/environment_vars.md)

---