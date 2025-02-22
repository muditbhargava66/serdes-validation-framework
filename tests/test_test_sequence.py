# tests/test_test_sequence.py

import unittest
from unittest.mock import patch, MagicMock
import logging
import numpy as np
from typing import Dict, List, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.test_sequence.sequencer import TestSequencer
from src.serdes_validation_framework.data_analysis.analyzer import DataAnalyzer

class MockInstrumentController:
    """Mock controller for testing"""
    def __init__(self) -> None:
        self.connected_instruments: Dict[str, bool] = {}
        # Define mock responses with proper numeric values
        self.responses = {
            '*IDN?': 'Mock Instrument',
            '*RST': 'OK',
            'MEASure:VOLTage:DC?': '0.123',
            '*TST?': '0',
            '*OPC?': '1'
        }

    def connect_instrument(self, resource_name: str) -> None:
        """Mock connect instrument"""
        self.connected_instruments[resource_name] = True
        
    def disconnect_instrument(self, resource_name: str) -> None:
        """Mock disconnect instrument"""
        if resource_name in self.connected_instruments:
            del self.connected_instruments[resource_name]
            
    def send_command(self, resource_name: str, command: str) -> None:
        """Mock send command"""
        if resource_name not in self.connected_instruments:
            raise ValueError(f"Instrument {resource_name} not connected")
            
    def query_instrument(self, resource_name: str, command: str) -> str:
        """Mock query with numeric responses"""
        if resource_name not in self.connected_instruments:
            raise ValueError(f"Instrument {resource_name} not connected")
        return self.responses.get(command, '0.0')

class TestTestSequencer(unittest.TestCase):
    """Test cases for TestSequencer class"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        # Create mock controller with proper responses
        self.mock_controller = MockInstrumentController()
        
        # Initialize test sequencer with mock controller
        self.test_sequencer = TestSequencer(controller=self.mock_controller)
        
        # Test resource names
        self.test_resources = ['GPIB::1::INSTR', 'GPIB::2::INSTR']
        
        # Pre-connect instruments for some tests
        for resource in self.test_resources:
            self.mock_controller.connect_instrument(resource)
            self.test_sequencer.instruments[resource] = self.mock_controller

    def test_initialization(self) -> None:
        """Test sequencer initialization"""
        sequencer = TestSequencer(controller=self.mock_controller)
        self.assertIsNotNone(sequencer.instrument_controller)
        self.assertEqual(len(sequencer.instruments), 0)

    def test_setup_instruments(self) -> None:
        """Test instrument setup"""
        sequencer = TestSequencer(controller=self.mock_controller)
        sequencer.setup_instruments(['GPIB::1::INSTR'])
        self.assertIn('GPIB::1::INSTR', sequencer.instruments)
        self.assertIn('GPIB::1::INSTR', self.mock_controller.connected_instruments)

    def test_run_sequence(self) -> None:
        """Test sequence execution"""
        sequence = [
            {'resource': 'GPIB::1::INSTR', 'command': '*RST', 'action': 'send'},
            {'resource': 'GPIB::2::INSTR', 'command': '*IDN?', 'action': 'query'},
            {'resource': 'GPIB::2::INSTR', 'command': 'MEASure:VOLTage:DC?', 'action': 'query'}
        ]
        
        results = self.test_sequencer.run_sequence(sequence)
        self.assertEqual(results['GPIB::2::INSTR'], '0.123')
        self.assertIn('GPIB::2::INSTR', self.test_sequencer.instruments)

    def test_collect_and_analyze_data(self) -> None:
        """Test data collection and analysis"""
        # Test with voltage measurement
        stats = self.test_sequencer.collect_and_analyze_data(
            'GPIB::2::INSTR',
            'MEASure:VOLTage:DC?',
            'voltage'
        )
        
        self.assertIsNotNone(stats)
        self.assertIn('mean', stats)
        self.assertEqual(float(stats['mean']), 0.123)

    def test_cleanup(self) -> None:
        """Test cleanup functionality"""
        # Ensure instruments are connected
        self.test_sequencer.setup_instruments(self.test_resources)
        
        # Verify initial state
        for resource in self.test_resources:
            self.assertIn(resource, self.test_sequencer.instruments)
            self.assertIn(resource, self.mock_controller.connected_instruments)
            
        # Perform cleanup
        self.test_sequencer.cleanup(self.test_resources)
        
        # Verify cleanup
        for resource in self.test_resources:
            self.assertNotIn(resource, self.test_sequencer.instruments)
            self.assertNotIn(resource, self.mock_controller.connected_instruments)

    def test_error_handling(self) -> None:
        """Test error handling scenarios"""
        # Test invalid resource name
        with self.assertRaises(ValueError):
            self.test_sequencer.setup_instruments([''])
            
        # Test invalid sequence
        with self.assertRaises(ValueError):
            self.test_sequencer.run_sequence([])
            
        # Test disconnected instrument
        with self.assertRaises(ValueError):
            self.test_sequencer.collect_and_analyze_data(
                'INVALID::RESOURCE',
                'MEASure:VOLTage:DC?',
                'voltage'
            )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()