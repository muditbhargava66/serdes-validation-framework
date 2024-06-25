import unittest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.test_sequence.sequencer import TestSequencer

class TestTestSequencer(unittest.TestCase):

    @patch('src.serdes_validation_framework.data_collection.data_collector.DataCollector')
    @patch('src.serdes_validation_framework.instrument_control.controller.InstrumentController')
    def setUp(self, MockInstrumentController, MockDataCollector):
        self.mock_data_collector = MockDataCollector.return_value
        self.mock_instrument_controller = MockInstrumentController.return_value

        # Mock the instruments
        self.mock_instrument = MagicMock()
        self.mock_instrument_controller.connect_instrument.return_value = self.mock_instrument
        self.mock_instrument_controller.query_instrument.return_value = 'Mock Response'
        self.mock_data_collector.connect_instrument.return_value = self.mock_instrument
        self.mock_data_collector.collect_data.return_value = '0.1 0.2 0.3 0.4 0.5'

        self.test_sequencer = TestSequencer()
        self.test_sequencer.instrument_controller = self.mock_instrument_controller
        self.test_sequencer.data_collector = self.mock_data_collector

        # Connect the instruments
        self.test_sequencer.instrument_controller.connect_instrument('GPIB::1::INSTR')
        self.test_sequencer.instrument_controller.connect_instrument('GPIB::2::INSTR')

    def tearDown(self):
        # Disconnect the instruments
        self.test_sequencer.instrument_controller.disconnect_instrument('GPIB::1::INSTR')
        self.test_sequencer.instrument_controller.disconnect_instrument('GPIB::2::INSTR')

    def test_setup_instruments(self):
        self.test_sequencer.setup_instruments(['GPIB::1::INSTR'])
        self.mock_instrument_controller.connect_instrument.assert_called_with('GPIB::1::INSTR')

    def test_run_sequence(self):
        sequence = [
            {'resource': 'GPIB::1::INSTR', 'command': '*RST', 'action': 'send'},
            {'resource': 'GPIB::2::INSTR', 'command': '*IDN?', 'action': 'query'}
        ]
        results = self.test_sequencer.run_sequence(sequence)
        self.mock_instrument_controller.send_command.assert_called_with('GPIB::1::INSTR', '*RST')
        self.mock_instrument_controller.query_instrument.assert_called_with('GPIB::2::INSTR', '*IDN?')
        self.assertEqual(results['GPIB::2::INSTR'], 'Mock Response')

    def test_collect_and_analyze_data(self):
        stats = self.test_sequencer.collect_and_analyze_data('GPIB::2::INSTR', 'MEASure:VOLTage:DC?', 'voltage')
        self.mock_data_collector.collect_data.assert_called_with('GPIB::2::INSTR', 'MEASure:VOLTage:DC?')
        self.assertIsNotNone(stats)

    def test_cleanup(self):
        self.test_sequencer.cleanup(['GPIB::1::INSTR', 'GPIB::2::INSTR'])
        self.mock_instrument_controller.disconnect_instrument.assert_any_call('GPIB::1::INSTR')
        self.mock_instrument_controller.disconnect_instrument.assert_any_call('GPIB::2::INSTR')

if __name__ == '__main__':
    unittest.main()
