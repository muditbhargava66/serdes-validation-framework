import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.instrument_control.controller import InstrumentController


class TestInstrumentController(unittest.TestCase):
    @patch("pyvisa.ResourceManager")
    def setUp(self, MockResourceManager):
        self.mock_rm = MockResourceManager.return_value
        self.mock_instrument = MagicMock()
        self.mock_rm.open_resource.return_value = self.mock_instrument
        self.controller = InstrumentController()

    def test_connect_instrument(self):
        self.controller.connect_instrument("GPIB::2::INSTR")
        self.mock_rm.open_resource.assert_called_with("GPIB::2::INSTR")
        self.assertIn("GPIB::2::INSTR", self.controller.instruments)

    def test_send_command(self):
        self.controller.connect_instrument("GPIB::2::INSTR")
        self.controller.send_command("GPIB::2::INSTR", "*RST")
        self.mock_instrument.write.assert_called_with("*RST")

    def test_query_instrument(self):
        self.controller.connect_instrument("GPIB::2::INSTR")
        self.mock_instrument.query.return_value = "Mock Response"
        response = self.controller.query_instrument("GPIB::2::INSTR", "*IDN?")
        self.mock_instrument.query.assert_called_with("*IDN?")
        self.assertEqual(response, "Mock Response")

    def test_disconnect_instrument(self):
        self.controller.connect_instrument("GPIB::2::INSTR")
        self.controller.disconnect_instrument("GPIB::2::INSTR")
        self.mock_instrument.close.assert_called_once()
        self.assertNotIn("GPIB::2::INSTR", self.controller.instruments)

    def test_send_command_not_connected(self):
        with self.assertRaises(ValueError):
            self.controller.send_command("GPIB::2::INSTR", "*RST")

    def test_query_instrument_not_connected(self):
        with self.assertRaises(ValueError):
            self.controller.query_instrument("GPIB::2::INSTR", "*IDN?")

    def test_disconnect_instrument_not_connected(self):
        with self.assertRaises(ValueError):
            self.controller.disconnect_instrument("GPIB::2::INSTR")


if __name__ == "__main__":
    unittest.main()
