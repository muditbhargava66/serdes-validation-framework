import unittest
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tests.mock_pyvisa import mock_pyvisa
from src.serdes_validation_framework.data_collection.data_collector import DataCollector

class TestDataCollector(unittest.TestCase):

    @patch('pyvisa.ResourceManager')
    def setUp(self, MockResourceManager):
        self.mock_rm, self.mock_instrument = mock_pyvisa()
        MockResourceManager.return_value = self.mock_rm
        self.data_collector = DataCollector()

    def test_connect_instrument(self):
        self.data_collector.connect_instrument('GPIB::1::INSTR')
        self.mock_rm.open_resource.assert_called_with('GPIB::1::INSTR')
        self.assertIn('GPIB::1::INSTR', self.data_collector.instruments)

    def test_collect_data(self):
        self.data_collector.connect_instrument('GPIB::1::INSTR')
        self.mock_instrument.query.return_value = 'Mock Response'
        response = self.data_collector.collect_data('GPIB::1::INSTR', '*IDN?')
        self.mock_instrument.query.assert_called_with('*IDN?')
        self.assertEqual(response, 'Mock Response')

    def test_disconnect_instrument(self):
        self.data_collector.connect_instrument('GPIB::1::INSTR')
        self.data_collector.disconnect_instrument('GPIB::1::INSTR')
        self.mock_instrument.close.assert_called_once()
        self.assertNotIn('GPIB::1::INSTR', self.data_collector.instruments)

    def test_collect_data_not_connected(self):
        with self.assertRaises(ValueError):
            self.data_collector.collect_data('GPIB::1::INSTR', '*IDN?')

    def test_disconnect_instrument_not_connected(self):
        with self.assertRaises(ValueError):
            self.data_collector.disconnect_instrument('GPIB::1::INSTR')

if __name__ == '__main__':
    unittest.main()
