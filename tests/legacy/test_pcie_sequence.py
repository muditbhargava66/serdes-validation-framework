#!/usr/bin/env python3
"""
Test cases for PCIe test sequence functionality.

This module contains comprehensive tests for the PCIe test sequence
implementation, including configuration validation, test execution,
and result processing.
"""

import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from src.serdes_validation_framework.protocols.pcie.constants import SignalMode
from src.serdes_validation_framework.test_sequence.pcie_sequence import (
    LaneConfig,
    PCIeTestPhase,
    PCIeTestResult,
    PCIeTestSequence,
    PCIeTestSequenceConfig,
    create_multi_lane_pam4_test,
    create_single_lane_nrz_test,
)

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
logger = logging.getLogger(__name__)


class TestPCIeSequence(unittest.TestCase):
    """Test cases for PCIe test sequence functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create test lane configuration
        self.lane_config = LaneConfig(lane_id=0, mode=SignalMode.NRZ, sample_rate=100e9, bandwidth=50e9, voltage_range=1.0)

        # Create test sequence configuration
        self.sequence_config = PCIeTestSequenceConfig(
            test_name="Test PCIe Sequence",
            lanes=[self.lane_config],
            test_phases=[PCIeTestPhase.INITIALIZATION, PCIeTestPhase.VALIDATION],
            stress_duration=1.0,  # Short duration for tests
            target_ber=1e-9,
        )

        # Generate test signal data
        self.test_signal_data = self._generate_test_signal()

    def _generate_test_signal(self, num_samples: int = 1000) -> dict:
        """Generate test signal data for testing."""
        time = np.linspace(0, num_samples / 100e9, num_samples, dtype=np.float64)
        voltage = np.random.choice([-1.0, 1.0], size=num_samples).astype(np.float64)
        noise = np.random.normal(0, 0.1, num_samples)
        voltage += noise

        return {"time": time, "voltage": voltage}

    def test_lane_config_validation(self) -> None:
        """Test lane configuration validation."""
        # Test valid configuration
        config = LaneConfig(lane_id=0, mode=SignalMode.NRZ, sample_rate=100e9, bandwidth=50e9, voltage_range=1.0)
        self.assertEqual(config.lane_id, 0)
        self.assertEqual(config.mode, SignalMode.NRZ)

        # Test invalid lane ID
        with self.assertRaises(AssertionError):
            LaneConfig(
                lane_id=-1,  # Invalid negative ID
                mode=SignalMode.NRZ,
                sample_rate=100e9,
                bandwidth=50e9,
                voltage_range=1.0,
            )

        # Test invalid sample rate
        with self.assertRaises(AssertionError):
            LaneConfig(
                lane_id=0,
                mode=SignalMode.NRZ,
                sample_rate=-100e9,  # Invalid negative rate
                bandwidth=50e9,
                voltage_range=1.0,
            )

    def test_sequence_config_validation(self) -> None:
        """Test test sequence configuration validation."""
        # Test valid configuration
        config = PCIeTestSequenceConfig(
            test_name="Valid Test", lanes=[self.lane_config], test_phases=[PCIeTestPhase.INITIALIZATION], target_ber=1e-9
        )
        self.assertEqual(config.test_name, "Valid Test")
        self.assertEqual(len(config.lanes), 1)

        # Test empty lanes
        with self.assertRaises(AssertionError):
            PCIeTestSequenceConfig(
                test_name="Invalid Test",
                lanes=[],  # Empty lanes list
                test_phases=[PCIeTestPhase.INITIALIZATION],
                target_ber=1e-9,
            )

        # Test invalid BER
        with self.assertRaises(AssertionError):
            PCIeTestSequenceConfig(
                test_name="Invalid Test",
                lanes=[self.lane_config],
                test_phases=[PCIeTestPhase.INITIALIZATION],
                target_ber=2.0,  # Invalid BER > 1
            )

    def test_pcie_test_sequence_initialization(self) -> None:
        """Test PCIe test sequence initialization."""
        # Test valid initialization
        sequence = PCIeTestSequence(self.sequence_config)
        self.assertIsInstance(sequence, PCIeTestSequence)
        self.assertEqual(sequence.config, self.sequence_config)
        self.assertEqual(len(sequence.analyzers), 1)  # One analyzer per lane

        # Test with invalid configuration
        with self.assertRaises(AssertionError):
            PCIeTestSequence("invalid_config")  # type: ignore

    @patch("src.serdes_validation_framework.instrument_control.pcie_analyzer.PCIeAnalyzer")
    def test_run_complete_sequence(self, mock_analyzer_class) -> None:
        """Test running complete test sequence."""
        # Setup mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_signal.return_value = {"snr_db": 25.0, "level_separation": 2.0}
        mock_analyzer_class.return_value = mock_analyzer

        # Create test sequence
        sequence = PCIeTestSequence(self.sequence_config)

        # Prepare signal data
        signal_data = {0: self.test_signal_data}

        # Run sequence
        result = sequence.run_complete_sequence(signal_data)

        # Validate results
        self.assertIsNotNone(result)
        self.assertIn(result.overall_status, [PCIeTestResult.PASS, PCIeTestResult.FAIL, PCIeTestResult.WARNING])
        self.assertGreaterEqual(result.total_duration, 0)
        self.assertGreater(len(result.phase_results), 0)
        self.assertEqual(len(result.lane_results), 1)

    def test_signal_data_validation(self) -> None:
        """Test signal data validation."""
        sequence = PCIeTestSequence(self.sequence_config)

        # Test valid signal data
        valid_data = {0: self.test_signal_data}
        try:
            sequence._validate_signal_data(valid_data)
        except Exception:
            self.fail("Valid signal data should not raise exception")

        # Test missing lane data
        with self.assertRaises(AssertionError):
            sequence._validate_signal_data({})  # Missing lane 0

        # Test invalid data structure
        with self.assertRaises(AssertionError):
            sequence._validate_signal_data({0: "invalid_data"})  # type: ignore

    def test_factory_functions(self) -> None:
        """Test factory functions for creating test sequences."""
        # Test single lane NRZ test
        nrz_test = create_single_lane_nrz_test(lane_id=0, sample_rate=100e9, bandwidth=50e9)
        self.assertIsInstance(nrz_test, PCIeTestSequence)
        self.assertEqual(len(nrz_test.config.lanes), 1)
        self.assertEqual(nrz_test.config.lanes[0].mode, SignalMode.NRZ)

        # Test multi-lane PAM4 test
        pam4_test = create_multi_lane_pam4_test(num_lanes=2, sample_rate=200e9, bandwidth=100e9)
        self.assertIsInstance(pam4_test, PCIeTestSequence)
        self.assertEqual(len(pam4_test.config.lanes), 2)
        self.assertEqual(pam4_test.config.lanes[0].mode, SignalMode.PAM4)
        self.assertEqual(pam4_test.config.lanes[1].mode, SignalMode.PAM4)

    def test_factory_function_validation(self) -> None:
        """Test factory function parameter validation."""
        # Test invalid lane count for multi-lane test
        with self.assertRaises(AssertionError):
            create_multi_lane_pam4_test(
                num_lanes=0,  # Invalid lane count
                sample_rate=200e9,
                bandwidth=100e9,
            )

        # Test invalid sample rate
        with self.assertRaises(AssertionError):
            create_single_lane_nrz_test(
                lane_id=0,
                sample_rate=-100e9,  # Invalid negative rate
                bandwidth=50e9,
            )

    @patch("src.serdes_validation_framework.instrument_control.pcie_analyzer.PCIeAnalyzer")
    def test_lane_performance_analysis(self, mock_analyzer_class) -> None:
        """Test lane performance analysis."""
        # Setup mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_signal.return_value = {"snr_db": 20.0, "level_separation": 1.8}
        mock_analyzer_class.return_value = mock_analyzer

        # Create test sequence
        sequence = PCIeTestSequence(self.sequence_config)

        # Test lane performance analysis
        performance = sequence._analyze_lane_performance(0, self.test_signal_data)

        # Validate performance metrics
        self.assertIsInstance(performance, dict)
        self.assertIn("snr_db", performance)
        self.assertIn("performance_score", performance)
        self.assertIsInstance(performance["performance_score"], float)
        self.assertGreaterEqual(performance["performance_score"], 0)
        self.assertLessEqual(performance["performance_score"], 100)

    def test_enum_values(self) -> None:
        """Test enum value definitions."""
        # Test PCIeTestPhase enum
        self.assertIn(PCIeTestPhase.INITIALIZATION, PCIeTestPhase)
        self.assertIn(PCIeTestPhase.LINK_TRAINING, PCIeTestPhase)
        self.assertIn(PCIeTestPhase.COMPLIANCE, PCIeTestPhase)
        self.assertIn(PCIeTestPhase.STRESS_TEST, PCIeTestPhase)
        self.assertIn(PCIeTestPhase.VALIDATION, PCIeTestPhase)

        # Test PCIeTestResult enum
        self.assertIn(PCIeTestResult.PASS, PCIeTestResult)
        self.assertIn(PCIeTestResult.FAIL, PCIeTestResult)
        self.assertIn(PCIeTestResult.WARNING, PCIeTestResult)
        self.assertIn(PCIeTestResult.SKIP, PCIeTestResult)

        # Test SignalMode enum
        self.assertIn(SignalMode.NRZ, SignalMode)
        self.assertIn(SignalMode.PAM4, SignalMode)

    def test_error_handling(self) -> None:
        """Test error handling in test sequences."""
        sequence = PCIeTestSequence(self.sequence_config)

        # Test with invalid signal data - should raise ValueError
        invalid_data = {0: {"invalid": "data"}}

        with self.assertRaises(ValueError):
            sequence.run_complete_sequence(invalid_data)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
