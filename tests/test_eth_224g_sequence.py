# tests/test_eth_224g_sequence.py

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.data_analysis.pam4_analyzer import EVMResults, EyeResults, PAM4Levels
from src.serdes_validation_framework.test_sequence.eth_224g_sequence import (
    ComplianceResults,
    Ethernet224GTestSequence,
    TrainingResults,
)


class TestEthernet224GTestSequence(unittest.TestCase):
    """Test cases for 224G Ethernet test sequences"""

    @patch('src.serdes_validation_framework.data_collection.data_collector.DataCollector')
    @patch('src.serdes_validation_framework.instrument_control.controller.InstrumentController')
    def setUp(self, MockInstrumentController, MockDataCollector) -> None:
        """Set up test fixtures"""
        self.mock_data_collector = MockDataCollector.return_value
        self.mock_instrument_controller = MockInstrumentController.return_value

        # Configure mock responses
        def mock_query(resource: str, query: str) -> str:
            if query == ":WAVeform:DATA?":
                # Generate realistic PAM4 test data
                base_levels = [-3, -1, 1, 3]
                num_points = 1000
                symbols = np.random.choice(base_levels, num_points)
                noise = np.random.normal(0, 0.1, num_points)
                test_data = symbols + noise
                return ','.join(map(str, test_data))

            responses = {
                '*IDN?': 'Mock Instrument',
                ':MEASure:EYE:HEIGht?': '0.4',
                ':MEASure:EYE:WIDTh?': '0.6',
                ':MEASure:JITTer:TJ?': '1e-12',
                ':MEASure:JITTer:RJ?': '0.5e-12',
                ':MEASure:JITTer:DJ?': '0.8e-12'
            }
            return responses.get(query, '0.0')

        self.mock_instrument_controller.query_instrument.side_effect = mock_query

        # Initialize test sequence
        self.sequence = Ethernet224GTestSequence()
        self.sequence.instrument_controller = self.mock_instrument_controller
        self.sequence.data_collector = self.mock_data_collector

        # Test resources
        self.scope = 'GPIB0::7::INSTR'
        self.pattern_gen = 'GPIB0::10::INSTR'

    def test_initialize_test_configs(self) -> None:
        """Test test configuration initialization"""
        configs = self.sequence._initialize_test_configs()

        # Check required configurations
        self.assertIn('sampling_rate', configs)
        self.assertIn('bandwidth', configs)
        self.assertIn('ui_period', configs)
        self.assertIn('voltage_range', configs)

        # Check value types
        self.assertIsInstance(configs['sampling_rate'], float)
        self.assertIsInstance(configs['bandwidth'], float)
        self.assertIsInstance(configs['ui_period'], float)
        self.assertIsInstance(configs['voltage_range'], float)

        # Check reasonable values
        self.assertEqual(configs['sampling_rate'], 256e9)
        self.assertEqual(configs['bandwidth'], 120e9)
        self.assertAlmostEqual(configs['ui_period'], 8.9e-12)

    def test_configure_scope_for_224g(self) -> None:
        """Test scope configuration with complete config"""
        config = {
            'sampling_rate': 256e9,
            'bandwidth': 120e9,
            'timebase': 5e-12,
            'voltage_range': 0.8
        }

        self.sequence.configure_scope_for_224g(self.scope, config)

        # Verify all commands
        expected_commands = [
            ":ACQuire:SRATe 256000000000.0",
            ":CHANnel1:BANDwidth 120000000000.0",
            ":TIMebase:SCALe 5E-12",
            ":CHANnel1:RANGe 0.8"
        ]

        for cmd in expected_commands:
            self.mock_instrument_controller.send_command.assert_any_call(
                self.scope,
                cmd
            )

        # Test with invalid config
        with self.assertRaises(AssertionError):
            self.sequence.configure_scope_for_224g(
                self.scope,
                {'sampling_rate': '256e9'}  # String instead of float
            )

    def test_run_link_training_test(self) -> None:
        """Test link training sequence"""
        results = self.sequence.run_link_training_test(
            self.scope,
            self.pattern_gen,
            timeout_seconds=5.0
        )

        # Check result type
        self.assertIsInstance(results, TrainingResults)

        # Verify pattern generator configuration
        self.mock_instrument_controller.send_command.assert_any_call(
            self.pattern_gen,
            ':PATTern:TYPE PRBS31'
        )

        # Verify scope configuration
        self.mock_instrument_controller.send_command.assert_any_call(
            self.scope,
            ':ACQuire:POINts 1000000'
        )

        # Check results structure
        self.assertIsInstance(results.training_time, float)
        self.assertIsInstance(results.convergence_status, str)
        self.assertTrue(all(isinstance(x, float) for x in results.final_eq_settings))
        self.assertIsInstance(results.adaptation_error, float)

        # Test with invalid timeout
        with self.assertRaises(AssertionError):
            self.sequence.run_link_training_test(
                self.scope,
                self.pattern_gen,
                timeout_seconds=-1.0
            )

    def test_measure_pam4_levels_basic(self) -> None:
        """Test PAM4 level measurements"""
        results = self.sequence.measure_pam4_levels(self.scope)

        # Check result type
        self.assertIsInstance(results, PAM4Levels)

        # Verify scope configuration
        self.mock_instrument_controller.send_command.assert_any_call(
            self.scope,
            ":ACQuire:SRATe 256000000000.0"
        )

        # Check measurements
        self.assertEqual(len(results.level_means), 4)
        self.assertEqual(len(results.level_separations), 3)
        self.assertIsInstance(results.uniformity, float)

    def test_run_compliance_test_suite(self) -> None:
        """Test full compliance test suite"""
        results = self.sequence.run_compliance_test_suite(
            self.scope,
            self.pattern_gen
        )

        # Check result type
        self.assertIsInstance(results, ComplianceResults)

        # Check all components
        self.assertIsInstance(results.pam4_levels, PAM4Levels)
        self.assertIsInstance(results.evm_results, EVMResults)
        self.assertIsInstance(results.eye_results, EyeResults)
        self.assertIsInstance(results.jitter_results, dict)
        self.assertIsInstance(results.test_status, str)

        # Check test status
        self.assertIn(results.test_status, ['PASS', 'FAIL'])

    def test_measure_pam4_levels_realistic(self) -> None:
        """Test PAM4 level measurements"""
        # Create more realistic PAM4 data
        mock_data = np.concatenate([
            np.random.normal(-3, 0.2, 500),  # More samples, larger variance
            np.random.normal(-1, 0.2, 500),
            np.random.normal(1, 0.2, 500),
            np.random.normal(3, 0.2, 500)
        ])

        # Add some noise and jitter
        noise = np.random.normal(0, 0.1, len(mock_data))
        mock_data = mock_data + noise

        # Convert to string format that matches scope output
        mock_response = ','.join(map(str, mock_data))
        self.mock_instrument_controller.query_instrument.return_value = mock_response

        # Run measurement
        results = self.sequence.measure_pam4_levels(self.scope)

        # Verify results
        self.assertIsInstance(results, PAM4Levels)
        self.assertEqual(len(results.level_means), 4)
        self.assertEqual(len(results.level_separations), 3)
        self.assertIsInstance(results.uniformity, float)

        # Check level ordering
        self.assertTrue(np.all(np.diff(results.level_means) > 0))

    def test_calculate_equalizer_taps(self) -> None:
        """Test equalizer tap calculation"""
        # Create test waveform
        test_waveform = np.array([0.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float64)

        taps = self.sequence._calculate_equalizer_taps(test_waveform)

        # Check tap array properties
        self.assertIsInstance(taps, list)
        self.assertTrue(all(isinstance(tap, float) for tap in taps))

        # Test with invalid input
        with self.assertRaises(AssertionError):
            self.sequence._calculate_equalizer_taps(
                np.array([1, 2, 3])  # Integer array instead of float
            )

    def test_capture_training_sequence(self) -> None:
        """Test training sequence capture"""
        # Test normal case
        waveform = self.sequence._capture_training_sequence(
            self.scope,
            timeout_seconds=1.0
        )

        # Check waveform properties
        self.assertIsInstance(waveform, np.ndarray)
        self.assertTrue(np.issubdtype(waveform.dtype, np.floating))

        # Test with invalid timeout
        with self.assertRaises(AssertionError):
            self.sequence._capture_training_sequence(
                self.scope,
                timeout_seconds=-1.0  # Invalid timeout should raise AssertionError
            )

    def test_analyze_training_results(self) -> None:
        """Test training results analysis"""
        # Create test waveform
        test_waveform = np.random.normal(0, 1, 1000).astype(np.float64)

        results = self.sequence._analyze_training_results(test_waveform)

        # Check result type
        self.assertIsInstance(results, TrainingResults)

        # Check result properties
        self.assertIsInstance(results.training_time, float)
        self.assertIsInstance(results.convergence_status, str)
        self.assertIsInstance(results.final_eq_settings, list)
        self.assertIsInstance(results.adaptation_error, float)

        # Test with invalid input
        with self.assertRaises(AssertionError):
            self.sequence._analyze_training_results(
                np.array([1, 2, 3])  # Integer array
            )

    def test_error_handling(self) -> None:
        """Test error handling in various scenarios"""
        # Test scope configuration error
        self.mock_instrument_controller.send_command.side_effect = Exception("Mock error")
        with self.assertRaises(Exception):
            self.sequence.configure_scope_for_224g(self.scope)

        # Reset mock
        self.mock_instrument_controller.send_command.side_effect = None

        # Test data collection error
        self.mock_instrument_controller.query_instrument.side_effect = Exception("Mock error")
        with self.assertRaises(Exception):
            self.sequence._capture_training_sequence(self.scope, 1.0)

        # Reset mock
        self.mock_instrument_controller.query_instrument.side_effect = None

    def test_compliance_criteria(self) -> None:
        """Test compliance criteria checking"""
        # Create mock compliance results
        mock_results = ComplianceResults(
            pam4_levels=PAM4Levels(
                level_means=np.array([-3.0, -1.0, 1.0, 3.0]),
                level_separations=np.array([2.0, 2.0, 2.0]),
                uniformity=0.1
            ),
            evm_results=EVMResults(
                rms_evm_percent=3.0,
                peak_evm_percent=6.0
            ),
            eye_results=EyeResults(
                eye_heights=[0.3, 0.4, 0.3],
                eye_widths=[0.5, 0.5, 0.5]
            ),
            jitter_results={
                'tj': 0.2e-12,
                'rj': 0.1e-12,
                'dj': 0.1e-12
            },
            test_status='PASS'
        )

        # Test with passing criteria
        self.assertEqual(mock_results.test_status, 'PASS')

        # Modify results to fail
        mock_results.eye_results.worst_eye_height = 0.1  # Below minimum
        self.assertEqual(mock_results.test_status, 'FAIL')

if __name__ == '__main__':
    unittest.main()
