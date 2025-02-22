# src/serdes_validation_framework/data_collection/data_collector.py

import logging
from typing import Dict, Any, Optional
from ..instrument_control.controller import InstrumentController
from ..instrument_control.mock_controller import MockInstrumentController, get_instrument_controller

logger = logging.getLogger(__name__)

class DataCollector:
    """Data collection interface with support for both real and mock hardware"""
    
    def __init__(self, controller: Optional[Any] = None) -> None:
        """
        Initialize data collector
        
        Args:
            controller: Optional instrument controller (uses auto-detection if None)
        """
        self.instruments: Dict[str, Any] = {}
        self.controller = controller if controller is not None else get_instrument_controller()
        logger.info("DataCollector initialized")
        
    def connect_instrument(self, resource_name: str) -> None:
        """
        Connect to an instrument
        
        Args:
            resource_name: VISA resource identifier
            
        Raises:
            ValueError: If resource name is invalid
            Exception: If connection fails
        """
        # Validate resource name
        if not resource_name:
            raise ValueError("Resource name cannot be empty")
            
        valid_prefixes = ('GPIB', 'USB', 'TCPIP', 'VXI', 'ASRL')
        if not any(resource_name.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f"Invalid resource name format: {resource_name}. "
                f"Must start with one of: {valid_prefixes}"
            )
        
        try:
            self.controller.connect_instrument(resource_name)
            self.instruments[resource_name] = self.controller
            logger.info(f"Connected to instrument {resource_name}")
        except Exception as e:
            logger.error(f"Failed to connect to {resource_name}: {e}")
            raise
            
    def disconnect_instrument(self, resource_name: str) -> None:
        """
        Disconnect from an instrument
        
        Args:
            resource_name: VISA resource identifier
        """
        if resource_name not in self.instruments:
            raise ValueError(f"Instrument {resource_name} not connected")
            
        try:
            self.controller.disconnect_instrument(resource_name)
            del self.instruments[resource_name]
            logger.info(f"Disconnected from instrument {resource_name}")
        except Exception as e:
            logger.error(f"Failed to disconnect from {resource_name}: {e}")
            raise
            
    def collect_data(self, resource_name: str, command: str) -> str:
        """
        Collect data from an instrument
        
        Args:
            resource_name: VISA resource identifier
            command: SCPI command to send
            
        Returns:
            Response from instrument
        """
        if resource_name not in self.instruments:
            raise ValueError(f"Instrument {resource_name} not connected")
            
        try:
            response = self.controller.query_instrument(resource_name, command)
            logger.debug(f"Data collected from {resource_name}: {response}")
            return response
        except Exception as e:
            logger.error(f"Failed to collect data from {resource_name}: {e}")
            raise