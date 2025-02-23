# tests/test_data_collection.py

import logging
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.data_collection.data_collector import DataCollector


class MockController:
    """Mock controller for testing"""
    def __init__(self) -> None:
        self.connected_instruments = {}
        self.query_responses = {}
        self.mock_instrument = MagicMock()

    def _validate_resource_name(self, resource_name: str) -> None:
        """
        Validate VISA resource name format
        
        Args:
            resource_name: Resource identifier to validate
            
        Raises:
            ValueError: If resource name is invalid
        """
        if not resource_name:
            raise ValueError("Resource name cannot be empty")

        valid_prefixes = ('GPIB', 'USB', 'TCPIP', 'VXI', 'ASRL')
        if not any(resource_name.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f"Invalid resource name format: {resource_name}. "
                f"Must start with one of: {valid_prefixes}"
            )

    def connect_instrument(self, resource_name: str) -> None:
        """
        Mock instrument connection with validation
        
        Args:
            resource_name: VISA resource identifier
            
        Raises:
            ValueError: If resource name is invalid
        """
        self._validate_resource_name(resource_name)
        self.connected_instruments[resource_name] = True

    def disconnect_instrument(self, resource_name: str) -> None:
        """
        Mock instrument disconnection
        
        Args:
            resource_name: VISA resource identifier
            
        Raises:
            ValueError: If instrument not connected
        """
        if resource_name not in self.connected_instruments:
            raise ValueError(f"Instrument {resource_name} not connected")
        del self.connected_instruments[resource_name]

    def query_instrument(self, resource_name: str, query: str) -> str:
        """
        Mock instrument query
        
        Args:
            resource_name: VISA resource identifier
            query: Command to query
            
        Returns:
            Mock response string
            
        Raises:
            ValueError: If instrument not connected
        """
        if resource_name not in self.connected_instruments:
            raise ValueError(f"Instrument {resource_name} not connected")
        return self.query_responses.get(query, "Mock Response")

class TestDataCollector(unittest.TestCase):
    """Test cases for DataCollector class"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        # Create mock controller
        self.mock_controller = MockController()

        # Create test responses
        self.mock_controller.query_responses = {
            '*IDN?': 'Test Instrument',
            'TEST:QUERY?': 'Test Response',
            'MEASure:VOLTage:DC?': '1.234'
        }

        # Initialize data collector with mock controller
        self.data_collector = DataCollector(controller=self.mock_controller)

        # Test resource name
        self.test_resource = 'GPIB::1::INSTR'

    def test_initialization(self) -> None:
        """Test data collector initialization"""
        self.assertIsNotNone(self.data_collector.controller)
        self.assertEqual(len(self.data_collector.instruments), 0)

    def test_connect_instrument(self) -> None:
        """Test instrument connection"""
        # Connect instrument
        self.data_collector.connect_instrument(self.test_resource)

        # Verify connection
        self.assertIn(self.test_resource, self.data_collector.instruments)
        self.assertTrue(
            self.test_resource in self.mock_controller.connected_instruments
        )

    def test_collect_data(self) -> None:
        """Test data collection"""
        # Connect and collect data
        self.data_collector.connect_instrument(self.test_resource)

        # Test basic query
        response = self.data_collector.collect_data(self.test_resource, '*IDN?')
        self.assertEqual(response, 'Test Instrument')

        # Test measurement query
        response = self.data_collector.collect_data(
            self.test_resource,
            'MEASure:VOLTage:DC?'
        )
        self.assertEqual(response, '1.234')

    def test_disconnect_instrument(self) -> None:
        """Test instrument disconnection"""
        # Connect then disconnect
        self.data_collector.connect_instrument(self.test_resource)
        self.data_collector.disconnect_instrument(self.test_resource)

        # Verify disconnection
        self.assertNotIn(self.test_resource, self.data_collector.instruments)
        self.assertNotIn(
            self.test_resource,
            self.mock_controller.connected_instruments
        )

    def test_collect_data_not_connected(self) -> None:
        """Test error handling for collecting from disconnected instrument"""
        with self.assertRaises(ValueError) as context:
            self.data_collector.collect_data(self.test_resource, '*IDN?')
        self.assertIn("not connected", str(context.exception))

    def test_disconnect_instrument_not_connected(self) -> None:
        """Test error handling for disconnecting an unconnected instrument"""
        with self.assertRaises(ValueError) as context:
            self.data_collector.disconnect_instrument(self.test_resource)
        self.assertIn("not connected", str(context.exception))

    def test_error_handling(self) -> None:
        """Test error handling in various scenarios"""
        # Test empty resource name
        with self.assertRaises(ValueError) as ctx:
            self.data_collector.connect_instrument("")
        self.assertIn("cannot be empty", str(ctx.exception))

        # Test invalid resource name format
        with self.assertRaises(ValueError) as ctx:
            self.data_collector.connect_instrument("INVALID::RESOURCE")
        self.assertIn("Invalid resource name format", str(ctx.exception))

        # Test duplicate connection (should work, but log a warning)
        self.data_collector.connect_instrument(self.test_resource)
        self.data_collector.connect_instrument(self.test_resource)  # Should work
        self.assertIn(self.test_resource, self.data_collector.instruments)

        # Test disconnect non-existent instrument
        with self.assertRaises(ValueError) as ctx:
            self.data_collector.disconnect_instrument("GPIB::99::INSTR")
        self.assertIn("not connected", str(ctx.exception))

        # Test query to non-existent instrument
        with self.assertRaises(ValueError) as ctx:
            self.data_collector.collect_data("GPIB::99::INSTR", "*IDN?")
        self.assertIn("not connected", str(ctx.exception))

        # Test cleanup
        self.data_collector.disconnect_instrument(self.test_resource)

    def test_multiple_instruments(self) -> None:
        """Test handling multiple instruments"""
        resources = ['GPIB::1::INSTR', 'GPIB::2::INSTR', 'GPIB::3::INSTR']

        # Connect multiple instruments
        for resource in resources:
            self.data_collector.connect_instrument(resource)

        # Verify all connections
        for resource in resources:
            self.assertIn(resource, self.data_collector.instruments)

        # Collect data from each
        for resource in resources:
            response = self.data_collector.collect_data(resource, '*IDN?')
            self.assertEqual(response, 'Test Instrument')

        # Disconnect all
        for resource in resources:
            self.data_collector.disconnect_instrument(resource)

        # Verify all disconnected
        self.assertEqual(len(self.data_collector.instruments), 0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
