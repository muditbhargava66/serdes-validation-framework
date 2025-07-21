#!/usr/bin/env python3
"""
Test Suite for PCIe Analyzer

This module provides comprehensive testing for the PCIe analyzer functionality.
It includes tests for:
- Analyzer initialization and configuration
- Signal mode switching (NRZ/PAM4)
- Signal analysis with type validation
- Link training operations
- Error handling and edge cases

The test suite uses realistic signal parameters and proper type checking throughout.
"""

import logging
import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.instrument_control.pcie_analyzer import (
    PCIeAnalyzer,
    PCIeConfig,
    SignalMode,
    TrainingConfig,
    TrainingResults,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPCIeAnalyzer(unittest.TestCase):
    """
    Test cases for PCIe signal analyzer functionality.
    
    This test suite verifies the correctness of PCIe signal analysis including:
    - Initialization and configuration
    - Signal mode switching
    - Signal analysis
    - Link training
    - Type checking and validation
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class-level test fixtures."""
        # Define test signal parameters for PCIe analysis
        cls.SIGNAL_PARAMS = {
            'NUM_SAMPLES': 1000,
            'SAMPLE_RATE': 256e9,      # 256 GSa/s
            'BANDWIDTH': 120e9,        # 120 GHz
            'LINK_SPEED': 32e9,        # 32 Gbps
            'VOLTAGE_RANGE': 0.8,      # 0.8V
            'LANE_COUNT': 1,
            'UI_PERIOD': 1 / 32e9      # UI period
        }

    def setUp(self) -> None:
        """Set up test fixtures for each test case."""
        # Create mock ModeSwitcher
        mock_mode_switcher = MagicMock()
        mock_switch_result = MagicMock()
        mock_switch_result.success = True
        mock_mode_switcher.switch_mode.return_value = mock_switch_result

        # Create PCIe configuration with default NRZ mode
        self.config = PCIeConfig(
            mode=SignalMode.NRZ,
            sample_rate=float(self.SIGNAL_PARAMS['SAMPLE_RATE']),
            bandwidth=float(self.SIGNAL_PARAMS['BANDWIDTH']),
            voltage_range=float(self.SIGNAL_PARAMS['VOLTAGE_RANGE']),
            link_speed=float(self.SIGNAL_PARAMS['LINK_SPEED']),
            lane_count=int(self.SIGNAL_PARAMS['LANE_COUNT'])
        )

        # Create test data with proper floating-point types
        self.time_data = np.linspace(
            0.0,
            self.SIGNAL_PARAMS['NUM_SAMPLES'] / self.SIGNAL_PARAMS['SAMPLE_RATE'],
            self.SIGNAL_PARAMS['NUM_SAMPLES'],
            dtype=np.float64
        )

        # Create NRZ test signal
        self.nrz_signal = self._generate_nrz_signal()

        # Create PAM4 test signal
        self.pam4_signal = self._generate_pam4_signal()

        # Initialize the analyzer with mock mode switcher
        self.analyzer = PCIeAnalyzer(
            config=self.config,
            mode_switcher=mock_mode_switcher
        )

    def _generate_nrz_signal(self) -> npt.NDArray[np.float64]:
        """
        Generate a realistic NRZ test signal with proper typing.
        
        Returns:
            Numpy array with NRZ signal values
        """
        # Define NRZ levels with exact floating-point values
        high_level = 0.4
        low_level = -0.4
        
        # Validate input types
        assert isinstance(high_level, float), "high_level must be a floating-point number"
        assert isinstance(low_level, float), "low_level must be a floating-point number"

        # Calculate samples per bit
        samples_per_bit = int(self.SIGNAL_PARAMS['SAMPLE_RATE'] / self.SIGNAL_PARAMS['LINK_SPEED'])
        
        # Generate random bits
        num_bits = self.SIGNAL_PARAMS['NUM_SAMPLES'] // samples_per_bit
        bits = np.random.choice([0, 1], size=num_bits)
        
        # Create unfiltered signal
        raw_signal = np.repeat(bits, samples_per_bit)
        if len(raw_signal) < self.SIGNAL_PARAMS['NUM_SAMPLES']:
            padding = self.SIGNAL_PARAMS['NUM_SAMPLES'] - len(raw_signal)
            raw_signal = np.pad(raw_signal, (0, padding), 'edge')
        
        # Convert to voltage levels
        voltage_signal = np.where(
            raw_signal == 1,
            high_level,
            low_level
        ).astype(np.float64)
        
        # Apply filtering to smooth transitions
        from scipy import signal as sig
        # Design a lowpass filter
        b, a = sig.butter(4, 0.4)
        # Apply the filter to smooth transitions
        filtered_signal = sig.filtfilt(b, a, voltage_signal)
        
        # Add noise with specific amplitude
        noise_amplitude = 0.05
        assert isinstance(noise_amplitude, float), "noise_amplitude must be a floating-point number"
        
        noise = np.random.normal(0, noise_amplitude, len(filtered_signal))
        noisy_signal = filtered_signal + noise
        
        # Verify output is float64
        result = noisy_signal.astype(np.float64)
        assert np.issubdtype(result.dtype, np.floating), "Result must be floating-point type"
        
        return result

    def _generate_pam4_signal(self) -> npt.NDArray[np.float64]:
        """
        Generate a realistic PAM4 test signal with proper typing.
        
        Returns:
            Numpy array with PAM4 signal values
        """
        # Define PAM4 levels with exact floating-point values
        levels = np.array([-0.6, -0.2, 0.2, 0.6], dtype=np.float64)
        
        # Validate input types
        assert isinstance(levels, np.ndarray), "levels must be a numpy array"
        assert np.issubdtype(levels.dtype, np.floating), "levels must be floating-point type"

        # Calculate samples per symbol
        samples_per_symbol = int(self.SIGNAL_PARAMS['SAMPLE_RATE'] / self.SIGNAL_PARAMS['LINK_SPEED'])
        
        # Generate random symbols
        num_symbols = self.SIGNAL_PARAMS['NUM_SAMPLES'] // samples_per_symbol
        symbols = np.random.choice(levels, size=num_symbols)
        
        # Create unfiltered signal
        raw_signal = np.repeat(symbols, samples_per_symbol)
        if len(raw_signal) < self.SIGNAL_PARAMS['NUM_SAMPLES']:
            padding = self.SIGNAL_PARAMS['NUM_SAMPLES'] - len(raw_signal)
            raw_signal = np.pad(raw_signal, (0, padding), 'edge')
        
        # Apply filtering to smooth transitions
        from scipy import signal as sig
        # Design a lowpass filter
        b, a = sig.butter(4, 0.4)
        # Apply the filter to smooth transitions
        filtered_signal = sig.filtfilt(b, a, raw_signal)
        
        # Add noise with specific amplitude
        noise_amplitude = 0.03
        assert isinstance(noise_amplitude, float), "noise_amplitude must be a floating-point number"
        
        noise = np.random.normal(0, noise_amplitude, len(filtered_signal))
        noisy_signal = filtered_signal + noise
        
        # Verify output is float64
        result = noisy_signal.astype(np.float64)
        assert np.issubdtype(result.dtype, np.floating), "Result must be floating-point type"
        
        return result

    def test_initialization(self) -> None:
        """Test analyzer initialization and configuration validation."""
        # Test valid initialization
        analyzer = PCIeAnalyzer(config=self.config)
        self.assertIsInstance(analyzer, PCIeAnalyzer)
        self.assertEqual(analyzer.config.mode, SignalMode.NRZ)

        # Test initialization with invalid config type
        invalid_config = {"mode": "NRZ"}  # Wrong type
        with self.assertRaises(AssertionError):
            PCIeAnalyzer(config=invalid_config)  # type: ignore

        # Test initialization with invalid config values
        with self.assertRaises(AssertionError):
            invalid_config = PCIeConfig(
                mode=SignalMode.NRZ,
                sample_rate=-1.0,  # Invalid negative value
                bandwidth=self.SIGNAL_PARAMS['BANDWIDTH'],
                voltage_range=self.SIGNAL_PARAMS['VOLTAGE_RANGE'],
                link_speed=self.SIGNAL_PARAMS['LINK_SPEED'],
                lane_count=self.SIGNAL_PARAMS['LANE_COUNT']
            )

        # Test with invalid sample rate type
        with self.assertRaises(AssertionError):
            PCIeConfig(
                mode=SignalMode.NRZ,
                sample_rate="256e9",  # String instead of float
                bandwidth=float(self.SIGNAL_PARAMS['BANDWIDTH']),
                voltage_range=float(self.SIGNAL_PARAMS['VOLTAGE_RANGE']),
                link_speed=float(self.SIGNAL_PARAMS['LINK_SPEED']),
                lane_count=int(self.SIGNAL_PARAMS['LANE_COUNT'])
            )

    def test_mode_switching(self) -> None:
        """Test signal mode switching with validation."""
        # Test valid mode switch
        result = self.analyzer.configure_mode(SignalMode.PAM4)
        self.assertTrue(result)
        self.assertEqual(self.analyzer.config.mode, SignalMode.PAM4)

        # Test with invalid mode type
        with self.assertRaises(AssertionError):
            self.analyzer.configure_mode("PAM4")  # type: ignore

    def test_analyze_signal_nrz(self) -> None:
        """Test NRZ signal analysis with type validation."""
        # Create test data
        test_data = {
            'time': self.time_data,
            'voltage': self.nrz_signal
        }

        # Verify both arrays are float64
        self.assertTrue(np.issubdtype(test_data['time'].dtype, np.floating))
        self.assertTrue(np.issubdtype(test_data['voltage'].dtype, np.floating))

        # Analyze signal
        results = self.analyzer.analyze_signal(test_data)

        # Check results structure and types
        self.assertIsInstance(results, dict)
        self.assertIn('level_separation', results)
        self.assertIn('snr_db', results)
        self.assertIn('jitter_ps', results)

        # Verify result types
        for _, value in results.items():
            self.assertIsInstance(value, float)

        # Test with mismatched array lengths
        invalid_data = {
            'time': self.time_data[:-10],  # Shortened array
            'voltage': self.nrz_signal
        }
        with self.assertRaises(ValueError):
            self.analyzer.analyze_signal(invalid_data)

        # Test with integer array (should fail type check)
        invalid_data = {
            'time': self.time_data,
            'voltage': np.array([1, 2, 3])  # Integer array
        }
        with self.assertRaises(ValueError):
            self.analyzer.analyze_signal(invalid_data)

        # Test with empty arrays
        empty_data = {
            'time': np.array([], dtype=np.float64),
            'voltage': np.array([], dtype=np.float64)
        }
        with self.assertRaises(ValueError):
            self.analyzer.analyze_signal(empty_data)

    def test_analyze_signal_pam4(self) -> None:
        """Test PAM4 signal analysis with type validation."""
        # Switch mode to PAM4
        self.analyzer.configure_mode(SignalMode.PAM4)

        # Create test data
        test_data = {
            'time': self.time_data,
            'voltage': self.pam4_signal
        }

        # Verify both arrays are float64
        self.assertTrue(np.issubdtype(test_data['time'].dtype, np.floating))
        self.assertTrue(np.issubdtype(test_data['voltage'].dtype, np.floating))

        # Analyze signal
        results = self.analyzer.analyze_signal(test_data)

        # Check results structure and types
        self.assertIsInstance(results, dict)
        self.assertIn('min_level_separation', results)
        self.assertIn('rms_evm_percent', results)
        self.assertIn('snr_db', results)

        # Verify result types
        for _, value in results.items():
            self.assertIsInstance(value, float)

    def test_run_link_training(self) -> None:
        """Test link training with proper configuration validation."""
        # Create valid training config
        config = TrainingConfig(
            preset_index=5,
            adaptation_mode='fast',
            target_snr=20.0,
            max_iterations=50
        )

        # Create test data
        test_data = {
            'time': self.time_data,
            'voltage': self.nrz_signal
        }
        
        # Set current data
        self.analyzer.current_data = test_data

        # Run training
        results = self.analyzer.run_link_training(config)

        # Check results structure and types
        self.assertIsInstance(results, TrainingResults)
        self.assertIsInstance(results.success, bool)
        self.assertIsInstance(results.iterations, int)
        self.assertIsInstance(results.final_snr, float)
        self.assertTrue(all(isinstance(x, float) for x in results.tap_weights))
        self.assertTrue(all(isinstance(x, float) for x in results.error_history))

        # Test with invalid config type
        invalid_config = {"preset_index": 5}  # Wrong type
        with self.assertRaises(ValueError):
            self.analyzer.run_link_training(invalid_config)  # type: ignore

        # Test with invalid config values
        with self.assertRaises(AssertionError):
            invalid_config = TrainingConfig(
                preset_index=20,  # Out of range
                adaptation_mode='fast',
                target_snr=20.0,
                max_iterations=50
            )

        # Test with invalid adaptation mode
        with self.assertRaises(AssertionError):
            invalid_config = TrainingConfig(
                preset_index=5,
                adaptation_mode='invalid',  # Invalid mode
                target_snr=20.0,
                max_iterations=50
            )

        # Test with non-float target_snr
        with self.assertRaises(AssertionError):
            TrainingConfig(
                preset_index=5,
                adaptation_mode='fast',
                target_snr="20.0",  # String instead of float
                max_iterations=50
            )

    def test_signal_validation(self) -> None:
        """Test signal validation with various edge cases."""
        # Test with NaN values
        invalid_data = {
            'time': self.time_data,
            'voltage': np.full_like(self.nrz_signal, np.nan)
        }
        with self.assertRaises(ValueError):
            self.analyzer.analyze_signal(invalid_data)

        # Test with infinite values
        invalid_data = {
            'time': self.time_data,
            'voltage': np.full_like(self.nrz_signal, np.inf)
        }
        with self.assertRaises(ValueError):
            self.analyzer.analyze_signal(invalid_data)

        # Test with missing keys
        invalid_data = {
            'voltage': self.nrz_signal
            # Missing 'time'
        }
        with self.assertRaises(ValueError):
            self.analyzer.analyze_signal(invalid_data)

    def test_find_nrz_levels(self) -> None:
        """Test NRZ level finding with proper type validation."""
        # Create histogram data
        voltage = self.nrz_signal
        hist, bins = np.histogram(voltage, bins=100)
        
        # Convert to float64 to ensure proper types
        hist = hist.astype(np.float64)
        bins = bins.astype(np.float64)
        
        # Find levels
        levels = self.analyzer._find_nrz_levels(hist, bins)
        
        # Check result type and properties
        self.assertIsInstance(levels, np.ndarray)
        self.assertTrue(np.issubdtype(levels.dtype, np.floating))
        self.assertEqual(levels.shape[0], 2)  # Should find exactly 2 levels for NRZ
        
        # Test with integer histogram (should fail)
        with self.assertRaises(ValueError):
            self.analyzer._find_nrz_levels(hist.astype(int), bins)

    def test_find_pam4_levels(self) -> None:
        """Test PAM4 level finding with proper type validation."""
        # Create histogram data
        voltage = self.pam4_signal
        hist, bins = np.histogram(voltage, bins=100)
        
        # Convert to float64 to ensure proper types
        hist = hist.astype(np.float64)
        bins = bins.astype(np.float64)
        
        # Find levels
        levels = self.analyzer._find_pam4_levels(hist, bins)
        
        # Check result type and properties
        self.assertIsInstance(levels, np.ndarray)
        self.assertTrue(np.issubdtype(levels.dtype, np.floating))
        self.assertEqual(levels.shape[0], 4)  # Should find exactly 4 levels for PAM4
        
        # Test with integer histogram (should fail)
        with self.assertRaises(ValueError):
            self.analyzer._find_pam4_levels(hist.astype(int), bins)

    def test_calculate_snr(self) -> None:
        """Test SNR calculation with proper type validation."""
        # Create test data
        test_data = {
            'time': self.time_data,
            'voltage': self.nrz_signal
        }
        
        # Set current data
        self.analyzer.current_data = test_data
        
        # Calculate SNR
        snr = self.analyzer._calculate_snr()
        
        # Check result type
        self.assertIsInstance(snr, float)
        self.assertGreater(snr, 0)  # SNR should be positive
        
        # Test with zero variance (should handle division by zero)
        test_data = {
            'time': self.time_data,
            'voltage': np.full_like(self.time_data, 1.0)  # Constant signal = infinite SNR
        }
        self.analyzer.current_data = test_data
        snr = self.analyzer._calculate_snr()
        self.assertEqual(snr, float('inf'))

    def test_update_equalizer(self) -> None:
        """Test equalizer update with type validation."""
        # Create test data
        test_data = {
            'time': self.time_data,
            'voltage': self.nrz_signal
        }
        
        # Set current data
        self.analyzer.current_data = test_data
        
        # Create tap weights
        tap_weights = [0.0, 0.0, 1.0, 0.0, 0.0]  # 5-tap equalizer
        
        # Update equalizer
        error = self.analyzer._update_equalizer(tap_weights)
        
        # Check result type
        self.assertIsInstance(error, float)
        
        # Test with non-float tap weights
        invalid_taps = [0, 0, 1, 0, 0]  # Integer taps
        with self.assertRaises(ValueError):
            self.analyzer._update_equalizer(invalid_taps)


if __name__ == '__main__':
    unittest.main()
