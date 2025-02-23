# Instrument Control

## Overview

The instrument control module provides a unified interface for controlling lab equipment, supporting both physical hardware and mock testing environments. It emphasizes type safety and robust error handling.

## Key Components

### Controller Types

1. Real Hardware Controller
   - Direct GPIB/USB/Network instrument control
   - Hardware-specific communication
   - Real-time measurements

2. Mock Controller
   - Hardware simulation for testing
   - Configurable responses
   - Error condition simulation

## Basic Usage

### Controller Initialization

```python
from serdes_validation_framework.instrument_control import get_instrument_controller
from typing import Dict, Optional, Union

# Initialize controller (auto-detects mode)
controller = get_instrument_controller()

# Check controller type
if controller.get_mode() == 'mock':
    print("Using mock controller")
else:
    print("Using real hardware controller")
```

### Type-Safe Instrument Connection

```python
def connect_to_instrument(
    resource_name: str,
    timeout_ms: float = 1000.0
) -> None:
    """
    Connect to instrument with type validation
    
    Args:
        resource_name: VISA resource identifier
        timeout_ms: Connection timeout in milliseconds
        
    Raises:
        TypeError: If arguments have invalid types
        ValueError: If resource name is invalid
        ConnectionError: If connection fails
    """
    # Validate input types
    assert isinstance(resource_name, str), \
        f"Resource name must be string, got {type(resource_name)}"
    assert isinstance(timeout_ms, float), \
        f"Timeout must be float, got {type(timeout_ms)}"
    
    # Validate resource name format
    if not any(x in resource_name for x in ['GPIB', 'USB', 'TCPIP']):
        raise ValueError(f"Invalid resource name format: {resource_name}")
    
    try:
        controller.connect_instrument(resource_name)
        controller.send_command(resource_name, '*IDN?')
    except Exception as e:
        raise ConnectionError(f"Failed to connect to {resource_name}: {e}")
```

### Command Execution

```python
def execute_command(
    resource_name: str,
    command: str,
    timeout_ms: Optional[float] = None
) -> None:
    """
    Execute instrument command with validation
    
    Args:
        resource_name: VISA resource identifier
        command: SCPI command string
        timeout_ms: Optional command timeout
        
    Raises:
        TypeError: If arguments have invalid types
        ValueError: If instrument not connected
    """
    # Type validation
    assert isinstance(resource_name, str), "Resource name must be string"
    assert isinstance(command, str), "Command must be string"
    if timeout_ms is not None:
        assert isinstance(timeout_ms, float), "Timeout must be float"
    
    if timeout_ms is not None:
        controller.set_timeout(resource_name, timeout_ms)
    
    controller.send_command(resource_name, command)
```

### Query Execution

```python
def query_float(
    resource_name: str,
    query: str,
    timeout_ms: Optional[float] = None
) -> float:
    """
    Query floating-point value from instrument
    
    Args:
        resource_name: VISA resource identifier
        query: SCPI query string
        timeout_ms: Optional query timeout
        
    Returns:
        Query result as float
        
    Raises:
        TypeError: If arguments have invalid types
        ValueError: If response cannot be converted to float
    """
    # Type validation
    assert isinstance(resource_name, str), "Resource name must be string"
    assert isinstance(query, str), "Query must be string"
    if timeout_ms is not None:
        assert isinstance(timeout_ms, float), "Timeout must be float"
    
    response = controller.query_instrument(resource_name, query)
    
    try:
        return float(response)
    except ValueError:
        raise ValueError(f"Response '{response}' cannot be converted to float")
```

## Advanced Usage

### Custom Response Types

```python
class TypedResponse:
    """Type-safe response handler"""
    
    @staticmethod
    def to_float(response: str) -> float:
        """Convert response to float"""
        try:
            return float(response)
        except ValueError:
            raise ValueError(f"Cannot convert '{response}' to float")
    
    @staticmethod
    def to_bool(response: str) -> bool:
        """Convert response to boolean"""
        try:
            value = int(response)
            return bool(value)
        except ValueError:
            raise ValueError(f"Cannot convert '{response}' to boolean")
    
    @staticmethod
    def to_int(response: str) -> int:
        """Convert response to integer"""
        try:
            return int(response)
        except ValueError:
            raise ValueError(f"Cannot convert '{response}' to integer")

def query_typed(
    resource_name: str,
    query: str,
    response_type: str
) -> Union[float, bool, int, str]:
    """
    Query with type conversion
    
    Args:
        resource_name: VISA resource identifier
        query: SCPI query string
        response_type: One of 'float', 'bool', 'int', 'str'
        
    Returns:
        Query result converted to specified type
    """
    response = controller.query_instrument(resource_name, query)
    
    converters = {
        'float': TypedResponse.to_float,
        'bool': TypedResponse.to_bool,
        'int': TypedResponse.to_int,
        'str': str
    }
    
    if response_type not in converters:
        raise ValueError(f"Invalid response type: {response_type}")
    
    return converters[response_type](response)
```

### Error Handling

```python
class InstrumentError(Exception):
    """Base class for instrument errors"""
    pass

class CommunicationError(InstrumentError):
    """Error in instrument communication"""
    pass

class TimeoutError(InstrumentError):
    """Operation timeout"""
    pass

def safe_query(
    resource_name: str,
    query: str,
    max_retries: int = 3,
    timeout_ms: float = 1000.0
) -> str:
    """
    Query with retry and error handling
    
    Args:
        resource_name: VISA resource identifier
        query: SCPI query string
        max_retries: Maximum retry attempts
        timeout_ms: Query timeout in milliseconds
        
    Returns:
        Query response
        
    Raises:
        CommunicationError: If query fails after retries
        TimeoutError: If operation times out
    """
    assert isinstance(max_retries, int), "max_retries must be integer"
    assert isinstance(timeout_ms, float), "timeout_ms must be float"
    
    for attempt in range(max_retries):
        try:
            controller.set_timeout(resource_name, timeout_ms)
            return controller.query_instrument(resource_name, query)
        except Exception as e:
            if attempt == max_retries - 1:
                raise CommunicationError(f"Query failed after {max_retries} attempts: {e}")
            continue
```

## Mock Configuration 

### Custom Mock Responses

```python
def configure_mock_responses(
    controller: Any,
    response_map: Dict[str, Union[str, float]]
) -> None:
    """
    Configure mock responses for testing
    
    Args:
        controller: Instrument controller instance
        response_map: Mapping of commands to responses
        
    Example:
        configure_mock_responses(controller, {
            '*IDN?': 'Mock Instrument',
            'MEASure:VOLTage?': 1.23
        })
    """
    for command, response in response_map.items():
        if isinstance(response, (int, float)):
            response = f"{float(response):.6f}"
        controller.add_mock_response(command, response)
```

### Error Simulation

```python
def configure_error_simulation(
    controller: Any,
    connection_error_rate: float = 0.0,
    command_error_rate: float = 0.0
) -> None:
    """
    Configure error simulation rates
    
    Args:
        controller: Instrument controller instance
        connection_error_rate: Connection failure probability
        command_error_rate: Command failure probability
    """
    assert 0 <= connection_error_rate <= 1, \
        "Connection error rate must be between 0 and 1"
    assert 0 <= command_error_rate <= 1, \
        "Command error rate must be between 0 and 1"
    
    controller.set_error_rates(
        connection_error_rate=connection_error_rate,
        command_error_rate=command_error_rate
    )
```

## Best Practices

1. Always validate input types:
   ```python
   assert isinstance(value, float), f"Expected float, got {type(value)}"
   ```

2. Use type hints:
   ```python
   def measure_voltage(channel: int) -> float:
   ```

3. Handle cleanup properly:
   ```python
   try:
       controller.connect_instrument(resource_name)
       # ... perform operations
   finally:
       controller.disconnect_instrument(resource_name)
   ```

4. Set appropriate timeouts:
   ```python
   controller.set_timeout(resource_name, operation_time * 1.5)
   ```

5. Use mock mode for testing:
   ```python
   os.environ['SVF_MOCK_MODE'] = '1'  # Force mock mode
   ```

## Examples

### Basic Measurement

```python
def measure_dc_voltage(
    resource_name: str,
    channel: int,
    range_v: float
) -> float:
    """
    Measure DC voltage with type validation
    
    Args:
        resource_name: VISA resource identifier
        channel: Channel number
        range_v: Voltage range in volts
        
    Returns:
        Measured voltage in volts
    """
    # Validate inputs
    assert isinstance(channel, int), "Channel must be integer"
    assert isinstance(range_v, float), "Range must be float"
    assert range_v > 0, "Range must be positive"
    
    try:
        # Configure measurement
        execute_command(resource_name, f":CHANnel{channel}:RANGe {range_v:.6f}")
        execute_command(resource_name, f":CHANnel{channel}:COUPling DC")
        
        # Make measurement
        voltage = query_float(resource_name, f":MEASure:VOLTage:DC? (@{channel})")
        return voltage
        
    except Exception as e:
        raise InstrumentError(f"Voltage measurement failed: {e}")
```

### Automated Test Sequence

```python
def run_test_sequence(
    resource_names: List[str],
    test_points: List[float]
) -> Dict[str, List[float]]:
    """
    Run automated test sequence
    
    Args:
        resource_names: List of instrument identifiers
        test_points: List of test voltages
        
    Returns:
        Dictionary of test results
    """
    results: Dict[str, List[float]] = {}
    
    try:
        # Connect instruments
        for resource in resource_names:
            connect_to_instrument(resource)
        
        # Run tests
        for voltage in test_points:
            # Set source
            execute_command(
                resource_names[0],
                f":SOURce:VOLTage {voltage:.6f}"
            )
            
            # Measure result
            measurement = query_float(
                resource_names[1],
                ":MEASure:VOLTage:DC?"
            )
            results.setdefault('voltages', []).append(measurement)
            
    finally:
        # Cleanup
        for resource in resource_names:
            try:
                controller.disconnect_instrument(resource)
            except Exception as e:
                logger.warning(f"Cleanup error for {resource}: {e}")
    
    return results
```

## See Also

- [Mock Controller Documentation](mock_controller.md)
- [Testing Guide](../guides/testing.md)
- [Equipment Setup](../guides/instrument_setup.md)

---