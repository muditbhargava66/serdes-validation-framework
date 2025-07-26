# tests/test_scope_224g.py

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.instrument_control.scope_224g import HighBandwidthScope, ScopeConfig, WaveformData


class TestHighBandwidthScope(unittest.TestCase):
    """Test cases for high-bandwidth scope control"""

    @patch("src.serdes_validation_framework.instrument_control.controller.InstrumentController")
    def setUp(self, MockInstrumentController) -> None:
        """Set up test fixtures"""
        self.mock_controller = MockInstrumentController.return_value

        # Set up mock responses
        self.mock_responses = {
            ":WAVeform:DATA?": "0.0,0.1,0.2,0.3,0.4,0.5",
            ":WAVeform:XINCrement?": "1.23456e-12",
            ":WAVeform:XORigin?": "0.0",
            ":MEASure:EYE:HEIGht?": "0.4",
            ":MEASure:EYE:WIDTh?": "0.6",
            ":MEASure:EYE:JITTer?": "0.1",
            ":TEST:QUERY?": "1.23456",
        }

        def mock_query(resource: str, query: str) -> str:
            return self.mock_responses.get(query, "0.0")

        self.mock_controller.query_instrument.side_effect = mock_query

        # Create scope instance
        self.scope_address = "GPIB0::7::INSTR"
        self.scope = HighBandwidthScope(self.scope_address, self.mock_controller)

    def test_initialization(self) -> None:
        """Test scope initialization"""
        # Check default configuration
        self.assertEqual(self.scope.default_config.sampling_rate, 256e9)
        self.assertEqual(self.scope.default_config.bandwidth, 120e9)
        self.assertEqual(self.scope.default_config.timebase, 5e-12)
        self.assertEqual(self.scope.default_config.voltage_range, 0.8)

        # Test with invalid configuration
        with self.assertRaises(AssertionError):
            ScopeConfig(
                sampling_rate="256e9",  # String instead of float
                bandwidth=120e9,
                timebase=5e-12,
                voltage_range=0.8,
            )

    def test_configure_for_224g(self) -> None:
        """Test 224G configuration"""
        # Test with default config
        self.scope.configure_for_224g()

        # Update expected commands with correct scientific notation format
        expected_commands = [
            ":ACQuire:SRATe 2.560000e+11",  # 256 GSa/s
            ":CHANnel1:BANDwidth 1.200000e+11",  # 120 GHz
            ":TIMebase:SCALe 5.000000e-12",  # 5 ps/div
            ":CHANnel1:RANGe 8.000000e-01",  # 0.8V
            ":ACQuire:MODE HRES",
            ":TRIGger:MODE AUTO",
            ":WAVeform:FORMat REAL",
        ]

        for cmd in expected_commands:
            self.mock_controller.send_command.assert_any_call(self.scope_address, cmd)

        # Test with custom config
        custom_config = ScopeConfig(sampling_rate=240e9, bandwidth=100e9, timebase=10e-12, voltage_range=1.0)

        self.scope.configure_for_224g(custom_config)

        # Verify custom config commands
        custom_commands = [
            ":ACQuire:SRATe 2.400000e+11",
            ":CHANnel1:BANDwidth 1.000000e+11",
            ":TIMebase:SCALe 1.000000e-11",
            ":CHANnel1:RANGe 1.000000e+00",
        ]

        for cmd in custom_commands:
            self.mock_controller.send_command.assert_any_call(self.scope_address, cmd)

    def test_capture_eye_diagram(self) -> None:
        """Test eye diagram capture"""
        results = self.scope.capture_eye_diagram(duration_seconds=1.0, num_ui=1000)

        # Check result structure
        self.assertIsInstance(results, dict)
        self.assertIn("waveform", results)
        self.assertIn("eye_height", results)
        self.assertIn("eye_width", results)
        self.assertIn("jitter", results)

        # Check data types
        self.assertIsInstance(results["waveform"], WaveformData)
        self.assertIsInstance(results["eye_height"], float)
        self.assertIsInstance(results["eye_width"], float)
        self.assertIsInstance(results["jitter"], float)

        # Test with invalid parameters
        with self.assertRaises(AssertionError):
            self.scope.capture_eye_diagram(duration_seconds=-1.0)
        with self.assertRaises(AssertionError):
            self.scope.capture_eye_diagram(num_ui=0)

    def test_measure_jitter(self) -> None:
        """Test jitter measurements"""
        # Configure mock responses
        self.mock_controller.query_instrument.side_effect = [
            "1e-12",  # TJ
            "0.5e-12",  # RJ
            "0.8e-12",  # DJ
            "0.3e-12",  # PJ
        ]

        results = self.scope.measure_jitter()

        # Check result structure
        self.assertIn("tj", results)
        self.assertIn("rj", results)
        self.assertIn("dj", results)
        self.assertIn("pj", results)

        # Check value types
        self.assertTrue(all(isinstance(v, float) for v in results.values()))

        # Verify measurement commands
        self.mock_controller.send_command.assert_any_call(self.scope_address, ":MEASure:JITTer:ENABle")

    def test_get_waveform_data(self) -> None:
        """Test waveform data acquisition"""
        waveform = self.scope._get_waveform_data()

        # Check return type
        self.assertIsInstance(waveform, np.ndarray)
        self.assertTrue(np.issubdtype(waveform.dtype, np.floating))

        # Verify query
        self.mock_controller.query_instrument.assert_called_with(self.scope_address, ":WAVeform:DATA?")

    def test_process_waveform(self) -> None:
        """Test waveform processing"""
        # Create test data
        raw_data = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

        # Configure mock responses for time base info
        self.mock_controller.query_instrument.side_effect = [
            "1e-12",  # X increment
            "0.0",  # X origin
        ]

        waveform = self.scope._process_waveform(raw_data)

        # Check return type
        self.assertIsInstance(waveform, WaveformData)

        # Check data properties
        self.assertEqual(len(waveform.time), len(waveform.voltage))
        self.assertTrue(np.issubdtype(waveform.time.dtype, np.floating))
        self.assertTrue(np.issubdtype(waveform.voltage.dtype, np.floating))
        self.assertIsInstance(waveform.sample_rate, float)
        self.assertIsInstance(waveform.time_scale, float)

        # Test with invalid input
        with self.assertRaises(AssertionError):
            self.scope._process_waveform(np.array([1, 2, 3]))  # Integer array

    def test_query_float(self) -> None:
        """Test float query helper"""
        self.mock_controller.query_instrument.return_value = "1.23456"

        result = self.scope._query_float(":TEST:QUERY?")

        # Check return type
        self.assertIsInstance(result, float)

        # Check value
        self.assertAlmostEqual(result, 1.23456)

        # Test with invalid response
        self.mock_responses[":TEST:QUERY?"] = "invalid"
        with self.assertRaises(ValueError):
            self.scope._query_float(":TEST:QUERY?")

    def test_cleanup(self) -> None:
        """Test cleanup procedure"""
        self.scope.cleanup()

        # Verify disconnect call
        self.mock_controller.disconnect_instrument.assert_called_with(self.scope_address)


if __name__ == "__main__":
    unittest.main()
