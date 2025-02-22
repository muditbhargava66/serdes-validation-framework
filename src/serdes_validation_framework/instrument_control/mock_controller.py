"""
Mock Controller Module

This module provides mock instrument control capabilities for testing and development
without requiring physical hardware. It includes support for environment variable
control of mock/real modes and realistic synthetic data generation.

Environment Variables:
    SVF_MOCK_MODE:
        - Set to 1 to force mock mode (no hardware required)
        - Set to 0 to force real hardware mode (will fail if hardware not available)
        - If not set, automatically detects available hardware

Example:
    >>> controller = get_instrument_controller()
    >>> controller.connect_instrument('GPIB::1::INSTR')
    >>> response = controller.query_instrument('GPIB::1::INSTR', '*IDN?')
    >>> print(response)
    'Mock Instrument'
"""

import os
import logging
import warnings
import time
from typing import Dict, Optional, Any, Union, Callable, List
from enum import Enum
import numpy as np
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class InstrumentMode(Enum):
    """Enumeration for instrument control modes"""
    REAL = 'real'
    MOCK = 'mock'
    AUTO = 'auto'  # Automatically detect based on hardware availability

@dataclass
class MockResponse:
    """Data class for configurable mock responses"""
    value: Union[str, Callable[[], str]]
    delay: float = 0.0  # Simulated response delay in seconds
    error_rate: float = 0.0  # Probability of generating an error

class MockInstrumentError(Exception):
    """Base exception for mock instrument errors"""
    pass

class MockConnectionError(MockInstrumentError):
    """Exception for mock connection failures"""
    pass

class MockCommandError(MockInstrumentError):
    """Exception for mock command failures"""
    pass

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
    # Check environment variable
    mock_mode = os.environ.get('SVF_MOCK_MODE')
    
    if mock_mode is not None:
        try:
            mock_mode_int = int(mock_mode)
            if mock_mode_int == 1:
                logger.info("Forcing mock mode from environment variable")
                return InstrumentMode.MOCK
            elif mock_mode_int == 0:
                logger.info("Forcing real hardware mode from environment variable")
                return InstrumentMode.REAL
            else:
                raise ValueError(f"SVF_MOCK_MODE must be 0 or 1, got {mock_mode}")
        except ValueError:
            warnings.warn(f"Invalid SVF_MOCK_MODE value: {mock_mode}, using auto detection")
    
    return InstrumentMode.AUTO

def get_instrument_controller():
    """
    Get appropriate instrument controller based on environment and availability
    
    Returns:
        Real or mock instrument controller
        
    Raises:
        RuntimeError: If real mode forced but hardware not available
    """
    mode = get_instrument_mode()
    
    if mode == InstrumentMode.MOCK:
        logger.info("Using mock controller (forced by environment)")
        return MockInstrumentController()
    elif mode == InstrumentMode.REAL:
        try:
            import pyvisa
            rm = pyvisa.ResourceManager()
            # Test GPIB availability
            resources = rm.list_resources()
            if not any('GPIB' in str(r) for r in resources):
                raise RuntimeError("No GPIB resources found")
            logger.info("Using real GPIB controller (forced by environment)")
            from .controller import InstrumentController
            return InstrumentController()
        except Exception as e:
            raise RuntimeError(
                "Real hardware mode forced but GPIB not available: "
                f"{str(e)}"
            )
    else:  # AUTO mode
        try:
            import pyvisa
            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()
            if any('GPIB' in str(r) for r in resources):
                logger.info("Using real GPIB controller (auto-detected)")
                from .controller import InstrumentController
                return InstrumentController()
        except Exception as e:
            logger.info(f"GPIB not available ({e}), using mock controller")
        return MockInstrumentController()

class MockInstrumentController:
    """
    Mock controller for GPIB instruments in test environments
    
    This class provides a mock implementation of instrument control functions
    for testing without physical hardware. It generates realistic synthetic
    data and simulates common instrument behaviors.
    """
    
    def __init__(self) -> None:
        """Initialize mock controller with simulated instruments"""
        self.mode = InstrumentMode.MOCK
        self.connected_instruments: Dict[str, bool] = {}
        self.mock_responses = self._initialize_mock_responses()
        self.error_simulation = {
            'connection_error_rate': 0.0,
            'command_error_rate': 0.0,
            'data_error_rate': 0.0
        }
        logger.info("Mock instrument controller initialized")

    def _initialize_mock_responses(self) -> Dict[str, MockResponse]:
        """
        Initialize default mock responses for common commands
        
        Returns:
            Dictionary mapping commands to MockResponse objects
        """
        return {
            '*IDN?': MockResponse('Mock Instrument', delay=0.1),
            '*RST': MockResponse('OK', delay=0.2),
            '*TST?': MockResponse('0', delay=0.5),
            '*OPC?': MockResponse('1', delay=0.1),
            ':WAVeform:DATA?': MockResponse(self._generate_mock_waveform, delay=1.0),
            'MEASure:VOLTage:DC?': MockResponse(
                lambda: f"{np.random.normal(0, 0.1):.6f}", 
                delay=0.3
            ),
            ':MEASure:EYE:HEIGht?': MockResponse(
                lambda: f"{np.random.normal(0.4, 0.05):.6f}",
                delay=0.5
            ),
            ':MEASure:EYE:WIDTh?': MockResponse(
                lambda: f"{np.random.normal(0.6, 0.05):.6f}",
                delay=0.5
            ),
            ':MEASure:JITTer:TJ?': MockResponse(
                lambda: f"{np.random.normal(1e-12, 1e-13):.3e}",
                delay=0.5
            ),
            ':MEASure:JITTer:RJ?': MockResponse(
                lambda: f"{np.random.normal(0.5e-12, 5e-14):.3e}",
                delay=0.5
            ),
            ':MEASure:JITTer:DJ?': MockResponse(
                lambda: f"{np.random.normal(0.8e-12, 8e-14):.3e}",
                delay=0.5
            )
        }

    def get_mode(self) -> str:
        """
        Get current operation mode
        
        Returns:
            String indicating current mode ('mock' or 'real')
        """
        return self.mode.value

    def set_error_rates(
        self,
        connection_error_rate: float = 0.0,
        command_error_rate: float = 0.0,
        data_error_rate: float = 0.0
    ) -> None:
        """
        Set error simulation rates for testing error handling
        
        Args:
            connection_error_rate: Probability of connection failures
            command_error_rate: Probability of command failures
            data_error_rate: Probability of data corruption
            
        Raises:
            ValueError: If rates are not between 0 and 1
        """
        for name, rate in [
            ('connection', connection_error_rate),
            ('command', command_error_rate),
            ('data', data_error_rate)
        ]:
            if not 0 <= rate <= 1:
                raise ValueError(f"{name} error rate must be between 0 and 1")
        
        self.error_simulation = {
            'connection_error_rate': connection_error_rate,
            'command_error_rate': command_error_rate,
            'data_error_rate': data_error_rate
        }

    def add_mock_response(
        self,
        command: str,
        response: Union[str, Callable[[], str]],
        delay: float = 0.0,
        error_rate: float = 0.0
    ) -> None:
        """
        Add or update mock response for a command
        
        Args:
            command: SCPI command string
            response: Response string or function generating response
            delay: Simulated response delay in seconds
            error_rate: Probability of response error
            
        Raises:
            ValueError: If delay or error_rate is invalid
        """
        assert isinstance(command, str), "Command must be a string"
        assert callable(response) or isinstance(response, str), \
            "Response must be string or callable"
        assert delay >= 0, "Delay must be non-negative"
        assert 0 <= error_rate <= 1, "Error rate must be between 0 and 1"
        
        self.mock_responses[command] = MockResponse(response, delay, error_rate)

    def connect_instrument(self, resource_name: str) -> None:
        """
        Simulate instrument connection
        
        Args:
            resource_name: VISA resource identifier
            
        Raises:
            MockConnectionError: If connection simulation fails
            ValueError: If resource name is invalid
        """
        assert isinstance(resource_name, str), "Resource name must be a string"
        
        # Validate resource name format
        if not any(x in resource_name for x in ['GPIB', 'USB', 'TCPIP', 'VXI', 'ASRL']):
            raise ValueError(f"Invalid resource name format: {resource_name}")
        
        # Simulate connection failures
        if np.random.random() < self.error_simulation['connection_error_rate']:
            raise MockConnectionError(f"Simulated connection failure to {resource_name}")
        
        logger.info(f"Mock connecting to {resource_name}")
        self.connected_instruments[resource_name] = True

    def disconnect_instrument(self, resource_name: str) -> None:
        """
        Simulate instrument disconnection
        
        Args:
            resource_name: VISA resource identifier
            
        Raises:
            ValueError: If instrument not connected
        """
        assert isinstance(resource_name, str), "Resource name must be a string"
        
        if resource_name not in self.connected_instruments:
            raise ValueError(f"Instrument {resource_name} not connected")
            
        logger.info(f"Mock disconnecting from {resource_name}")
        del self.connected_instruments[resource_name]

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
        assert isinstance(resource_name, str), "Resource name must be a string"
        assert isinstance(command, str), "Command must be a string"
        
        if resource_name not in self.connected_instruments:
            raise ValueError(f"Instrument {resource_name} not connected")
            
        # Simulate command failures
        if np.random.random() < self.error_simulation['command_error_rate']:
            raise MockCommandError(f"Simulated command failure: {command}")
            
        logger.debug(f"Mock command to {resource_name}: {command}")

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
        assert isinstance(resource_name, str), "Resource name must be a string"
        assert isinstance(query, str), "Query must be a string"
        
        if resource_name not in self.connected_instruments:
            raise ValueError(f"Instrument {resource_name} not connected")
            
        # Simulate query failures
        if np.random.random() < self.error_simulation['command_error_rate']:
            raise MockCommandError(f"Simulated query failure: {query}")
            
        # Get mock response
        if query in self.mock_responses:
            mock_response = self.mock_responses[query]
            
            # Simulate response delay
            if mock_response.delay > 0:
                time.sleep(mock_response.delay)
            
            # Handle response generation
            if callable(mock_response.value):
                response = mock_response.value()
            else:
                response = mock_response.value
                
            # Simulate response error
            if np.random.random() < mock_response.error_rate:
                response = "ERROR"
                
            return response
            
        logger.warning(f"No mock response defined for query: {query}")
        return "0.0"  # Default response

    def _generate_mock_waveform(self) -> str:
        """
        Generate synthetic PAM4 waveform data with proper voltage levels
        
        Returns:
            Comma-separated waveform data string with proper PAM4 levels
        """
        try:
            # Generate PAM4 signal with realistic characteristics
            num_points = 1000000
            
            # Define proper PAM4 levels with good separation
            levels = np.array([-0.6, -0.2, 0.2, 0.6])  # Four distinct levels
            
            # Generate random symbols with equal probability
            symbols = np.random.choice(len(levels), num_points, p=[0.25, 0.25, 0.25, 0.25])
            voltage_levels = levels[symbols]
            
            # Add minimal noise to maintain level separation
            noise_amplitude = 0.02  # Small noise
            noise = np.random.normal(0, noise_amplitude, num_points)
            
            # Add timing jitter simulation
            t = np.linspace(0, num_points/256e9, num_points)  # 256 GSa/s
            jitter = 0.01 * np.sin(2 * np.pi * 1e9 * t)  # Reduced jitter
            
            # Combine signal components
            waveform = voltage_levels + noise + jitter
            
            # Add transition effects and ensure level separation
            for i in range(1, num_points):
                if symbols[i] != symbols[i-1]:
                    # Add small transition effects
                    waveform[i] += 0.05 * np.random.randn()
            
            # Scale and center the waveform
            waveform = waveform * 0.8  # Scale to typical voltage range
            
            return ','.join(map(lambda x: f"{x:.6f}", waveform))
            
        except Exception as e:
            logger.error(f"Failed to generate mock waveform: {e}")
            return ','.join(['0.0'] * 1000)  # Fallback data

    def __repr__(self) -> str:
        """
        String representation of controller state
        
        Returns:
            String with controller information
        """
        return (
            f"MockInstrumentController(mode={self.mode.value}, "
            f"connected_instruments={list(self.connected_instruments.keys())})"
        )

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test mock controller
    try:
        controller = MockInstrumentController()
        print(f"\nController: {controller}")
        
        # Test basic operations
        controller.connect_instrument('GPIB::1::INSTR')
        response = controller.query_instrument('GPIB::1::INSTR', '*IDN?')
        print(f"Mock ID response: {response}")
        
        # Test waveform generation
        waveform = controller.query_instrument('GPIB::1::INSTR', ':WAVeform:DATA?')
        waveform_data = list(map(float, waveform.split(',')))
        print(f"\nGenerated waveform points: {len(waveform_data)}")
        print(f"Waveform range: {min(waveform_data):.2f} to {max(waveform_data):.2f}")
        
        # Test measurement queries
        eye_height = controller.query_instrument('GPIB::1::INSTR', ':MEASure:EYE:HEIGht?')
        eye_width = controller.query_instrument('GPIB::1::INSTR', ':MEASure:EYE:WIDTh?')
        print(f"\nEye measurements:")
        print(f"Height: {eye_height}")
        print(f"Width: {eye_width}")
        
        # Test error simulation
        print("\nTesting error simulation:")
        controller.set_error_rates(
            connection_error_rate=0.5,
            command_error_rate=0.5,
            data_error_rate=0.5
        )
        
        # Try multiple operations to demonstrate error handling
        for i in range(5):
            try:
                print(f"\nTest iteration {i+1}:")
                controller.connect_instrument('GPIB::2::INSTR')
                print("  Connection successful")
                response = controller.query_instrument('GPIB::2::INSTR', '*IDN?')
                print(f"  Query successful: {response}")
                controller.disconnect_instrument('GPIB::2::INSTR')
                print("  Disconnection successful")
            except MockConnectionError as e:
                print(f"  Connection error: {e}")
            except MockCommandError as e:
                print(f"  Command error: {e}")
            except Exception as e:
                print(f"  Unexpected error: {e}")
        
        # Test custom response addition
        print("\nTesting custom responses:")
        controller.add_mock_response(
            'TEST:CUSTOM?',
            lambda: f"{np.random.random():.3f}",
            delay=0.1,
            error_rate=0.2
        )
        
        for i in range(3):
            try:
                response = controller.query_instrument('GPIB::1::INSTR', 'TEST:CUSTOM?')
                print(f"Custom response {i+1}: {response}")
            except Exception as e:
                print(f"Custom response error: {e}")
        
        # Test disconnection and cleanup
        controller.disconnect_instrument('GPIB::1::INSTR')
        print("\nTest complete")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

# test section
def run_mock_controller_tests():
    """
    Run comprehensive tests of mock controller functionality
    
    This function tests all major features of the mock controller including:
    - Basic connection and communication
    - Waveform generation
    - Error simulation
    - Custom response handling
    - Error conditions and recovery
    
    Returns:
        True if all tests pass, raises exception otherwise
    """
    try:
        # Test 1: Basic connectivity with no errors
        logger.info("\nTest 1: Basic connectivity")
        controller = MockInstrumentController()
        controller.set_error_rates(0, 0, 0)  # Disable errors for basic tests
        
        assert controller.get_mode() == 'mock', "Wrong controller mode"
        controller.connect_instrument('GPIB::1::INSTR')
        assert 'GPIB::1::INSTR' in controller.connected_instruments, \
            "Instrument not connected"
            
        # Test basic queries
        response = controller.query_instrument('GPIB::1::INSTR', '*IDN?')
        print(f"ID Query response: {response}")
        assert isinstance(response, str) and len(response) > 0, \
            "Invalid ID response"
        
        # Test 2: Waveform generation
        logger.info("\nTest 2: Waveform generation")
        waveform = controller.query_instrument('GPIB::1::INSTR', ':WAVeform:DATA?')
        waveform_data = list(map(float, waveform.split(',')))
        print(f"Generated {len(waveform_data)} waveform points")
        assert len(waveform_data) > 0, "No waveform data generated"
        
        data_range = (min(waveform_data), max(waveform_data))
        print(f"Data range: {data_range[0]:.2f} to {data_range[1]:.2f}")
        assert -5.0 <= data_range[0] and data_range[1] <= 5.0, \
            "Waveform data out of range"
        
        # Test 3: Custom responses
        logger.info("\nTest 3: Custom responses")
        test_value = "CustomResponse123"
        controller.add_mock_response('TEST:CUSTOM?', test_value, error_rate=0)
        response = controller.query_instrument('GPIB::1::INSTR', 'TEST:CUSTOM?')
        print(f"Custom response test: {response}")
        assert response == test_value, "Custom response failed"
        
        # Test 4: Error simulation for connection
        logger.info("\nTest 4: Connection error simulation")
        controller.set_error_rates(connection_error_rate=1.0, command_error_rate=0)
        
        try:
            controller.connect_instrument('GPIB::2::INSTR')
            assert False, "Connection should have failed"
        except MockConnectionError as e:
            print(f"Expected connection error caught: {e}")
        
        # Test 5: Error simulation for commands
        logger.info("\nTest 5: Command error simulation")
        controller.set_error_rates(connection_error_rate=0, command_error_rate=1.0)
        
        try:
            controller.query_instrument('GPIB::1::INSTR', '*IDN?')
            assert False, "Command should have failed"
        except MockCommandError as e:
            print(f"Expected command error caught: {e}")
        
        # Test 6: Dynamic response function
        logger.info("\nTest 6: Dynamic response function")
        controller.set_error_rates(0, 0, 0)  # Reset error rates
        controller.add_mock_response(
            'TEST:RANDOM?',
            lambda: f"{np.random.random():.3f}",
            error_rate=0
        )
        
        responses = set()
        for _ in range(3):
            response = controller.query_instrument('GPIB::1::INSTR', 'TEST:RANDOM?')
            responses.add(response)
            print(f"Dynamic response: {response}")
        
        assert len(responses) > 1, "Dynamic responses not varying"
        
        # Test 7: Cleanup
        logger.info("\nTest 7: Cleanup")
        controller.disconnect_instrument('GPIB::1::INSTR')
        assert len(controller.connected_instruments) == 0, \
            "Instrument not properly disconnected"
        
        logger.info("All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Mock controller tests failed: {e}")
        raise

def run_interactive_demo():
    """Run interactive demonstration of mock controller features"""
    try:
        logger.info("\nStarting mock controller interactive demo...")
        controller = MockInstrumentController()
        
        # Basic operations demo
        controller.connect_instrument('GPIB::1::INSTR')
        print(f"\nController: {controller}")
        
        response = controller.query_instrument('GPIB::1::INSTR', '*IDN?')
        print(f"Mock ID response: {response}")
        
        # Waveform demo
        waveform = controller.query_instrument('GPIB::1::INSTR', ':WAVeform:DATA?')
        waveform_data = list(map(float, waveform.split(',')))
        print(f"\nGenerated waveform points: {len(waveform_data)}")
        print(f"Waveform range: {min(waveform_data):.2f} to {max(waveform_data):.2f}")
        
        # Measurement demo
        eye_height = controller.query_instrument('GPIB::1::INSTR', ':MEASure:EYE:HEIGht?')
        eye_width = controller.query_instrument('GPIB::1::INSTR', ':MEASure:EYE:WIDTh?')
        print(f"\nEye measurements:")
        print(f"Height: {eye_height}")
        print(f"Width: {eye_width}")
        
        # Error simulation demo with lower rates
        print("\nDemonstrating error simulation (with 20% error rates):")
        controller.set_error_rates(
            connection_error_rate=0.2,
            command_error_rate=0.2,
            data_error_rate=0.2
        )
        
        for i in range(5):
            try:
                print(f"\nTest {i+1}:")
                controller.connect_instrument('GPIB::2::INSTR')
                print("  Connection successful")
                response = controller.query_instrument('GPIB::2::INSTR', '*IDN?')
                print(f"  Query successful: {response}")
                controller.disconnect_instrument('GPIB::2::INSTR')
                print("  Disconnection successful")
            except MockConnectionError as e:
                print(f"  Connection error: {e}")
            except MockCommandError as e:
                print(f"  Command error: {e}")
            except Exception as e:
                print(f"  Unexpected error: {e}")
        
        # Custom response demo
        print("\nDemonstrating custom responses:")
        controller.set_error_rates(0, 0, 0)  # Reset error rates
        controller.add_mock_response(
            'TEST:CUSTOM?',
            lambda: f"{np.random.random():.3f}",
            error_rate=0
        )
        
        for i in range(3):
            response = controller.query_instrument('GPIB::1::INSTR', 'TEST:CUSTOM?')
            print(f"Custom response {i+1}: {response}")
        
        # Cleanup
        controller.disconnect_instrument('GPIB::1::INSTR')
        print("\nDemo complete")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

# Run tests and demo if module is run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Run automated tests first
        run_mock_controller_tests()
        
        # Then run interactive demo
        run_interactive_demo()
        
    except Exception as e:
        print(f"Error: {e}")
        raise