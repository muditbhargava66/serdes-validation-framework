"""
Test Sequence Example Script

This script demonstrates the usage of the TestSequencer class for automated
instrument control and data collection. It showcases:
- Mock instrument setup for testing
- GPIB communication simulation
- Basic test sequence execution
- Data collection and analysis workflow

The example uses PyVISA's mock functionality to simulate instrument
communication without requiring physical hardware.

Dependencies:
    - logging
    - unittest.mock
    - pyvisa
    - src.serdes_validation_framework

Author: Mudit Bhargava
Date: February 2025
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch

from src.serdes_validation_framework.test_sequence.sequencer import TestSequencer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@patch('pyvisa.ResourceManager')
def main(mock_rm):
    try:
        # Setup the mock
        mock_instrument = MagicMock()
        mock_rm.return_value.open_resource.return_value = mock_instrument

        ts = TestSequencer()
        instruments = ['GPIB::1::INSTR', 'GPIB::2::INSTR']
        ts.setup_instruments(instruments)

        sequence = [
            {'resource': 'GPIB::1::INSTR', 'command': '*RST', 'action': 'send'},
            {'resource': 'GPIB::2::INSTR', 'command': '*IDN?', 'action': 'query'}
        ]
        mock_instrument.query.return_value = 'Mock Response'
        results = ts.run_sequence(sequence)
        print(f"Sequence results: {results}")

        # Ensure DataCollector uses the same mock instrument
        ts.data_collector.instruments['GPIB::2::INSTR'] = mock_instrument
        mock_instrument.query.return_value = '0.1 0.2 0.3 0.4 0.5'

        # Simulate collected data as a space-separated string and convert it to a list of floats
        collected_data = mock_instrument.query.return_value.split()
        numerical_data = list(map(float, collected_data))
        mock_instrument.query.return_value = ' '.join(map(str, numerical_data))

        stats = ts.collect_and_analyze_data('GPIB::2::INSTR', 'MEASure:VOLTage:DC?', 'voltage')
        print(f"Data statistics: {stats}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        try:
            ts.cleanup(instruments)
        except Exception as cleanup_error:
            logger.error(f"Cleanup error: {cleanup_error}")

if __name__ == "__main__":
    main()
