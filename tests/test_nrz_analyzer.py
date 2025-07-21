#!/usr/bin/env python3
"""
Test Suite for NRZ Signal Analysis

This module provides comprehensive testing for the NRZ signal analysis functionality.
It includes tests for:
- Signal level analysis
- Eye diagram measurement
- Jitter analysis
- Signal quality metrics
- Type checking and validation

The test suite uses realistic signal parameters and enforces proper data types throughout.
"""

import logging
import os
import sys
import unittest
from typing import Tuple

import numpy as np
import numpy.typing as npt

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.data_analysis.nrz_analyzer import EyeResults, JitterResults, NRZAnalyzer, NRZLevels

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestNRZAnalyzer(unittest.TestCase):
    """
    Test cases for NRZ signal analysis functionality.
    
    This test suite verifies the correctness of NRZ signal analysis including:
    - Initialization and parameter validation
    - Level separation analysis
    - Eye diagram measurements
    - Jitter analysis
    - Signal quality metrics
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class-level test fixtures."""
        # Define test signal parameters
        cls.SIGNAL_PARAMS = {
            'NUM_SAMPLES': 1000,      # Number of signal samples
            'SAMPLE_RATE': 256e9,     # 256 GSa/s
            'BIT_RATE': 112e9,        # 112 Gbps
            'NOISE_AMPLITUDE': 0.05,  # Noise amplitude
            'RISE_TIME': 5e-12,       # 5 ps rise time
            'SIGNAL_AMPLITUDE': 0.4,  # 400 mV amplitude
            'FILTER_ORDER': 4,        # Filter order
            'BANDWIDTH': 80e9,        # 80 GHz bandwidth
        }

    def setUp(self) -> None:
        """Set up test fixtures for each test case."""
        # Generate test signal with proper floating-point types
        self.time_data, self.nrz_signal = self._generate_test_signal()
        
        # Create test data dictionary
        self.test_data = {
            'time': self.time_data,
            'voltage': self.nrz_signal
        }
        
        # Initialize analyzer with proper parameters
        self.analyzer = NRZAnalyzer(
            data=self.test_data,
            sample_rate=float(self.SIGNAL_PARAMS['SAMPLE_RATE']),
            bit_rate=float(self.SIGNAL_PARAMS['BIT_RATE'])
        )

    def _generate_test_signal(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Generate a realistic NRZ test signal with proper typing.
        
        Returns:
            Tuple containing time array and NRZ signal array
        """
        # Create time array with proper float64 type
        time = np.linspace(
            0.0,
            self.SIGNAL_PARAMS['NUM_SAMPLES'] / self.SIGNAL_PARAMS['SAMPLE_RATE'],
            self.SIGNAL_PARAMS['NUM_SAMPLES'],
            dtype=np.float64
        )
        
        # Create bit pattern
        bit_period = 1.0 / self.SIGNAL_PARAMS['BIT_RATE']
        assert isinstance(bit_period, float), "bit_period must be a floating-point number"
        
        samples_per_bit = int(self.SIGNAL_PARAMS['SAMPLE_RATE'] / self.SIGNAL_PARAMS['BIT_RATE'])
        num_bits = self.SIGNAL_PARAMS['NUM_SAMPLES'] // samples_per_bit
        
        # Generate random bits with controlled transitions for better eye
        bits = np.zeros(num_bits, dtype=int)
        for i in range(1, num_bits):
            # 50% chance of transition for realistic pattern
            if np.random.random() > 0.5:
                bits[i] = 1 - bits[i-1]  # Transition
            else:
                bits[i] = bits[i-1]      # No transition
        
        # Convert to voltage levels
        signal_amplitude = float(self.SIGNAL_PARAMS['SIGNAL_AMPLITUDE'])
        assert isinstance(signal_amplitude, float), "signal_amplitude must be a floating-point number"
        
        raw_signal = np.repeat(bits, samples_per_bit)
        if len(raw_signal) < self.SIGNAL_PARAMS['NUM_SAMPLES']:
            padding = self.SIGNAL_PARAMS['NUM_SAMPLES'] - len(raw_signal)
            raw_signal = np.pad(raw_signal, (0, padding), 'edge')
        
        voltage_signal = np.where(
            raw_signal == 1,
            signal_amplitude,
            -signal_amplitude
        ).astype(np.float64)
        
        # Apply filtering for realistic rise/fall times
        from scipy import signal as sig
        
        # Design a lowpass filter
        nyquist = self.SIGNAL_PARAMS['SAMPLE_RATE'] / 2.0
        assert isinstance(nyquist, float), "nyquist must be a floating-point number"
        
        filter_order = int(self.SIGNAL_PARAMS['FILTER_ORDER'])
        cutoff = float(self.SIGNAL_PARAMS['BANDWIDTH'] / nyquist)
        assert isinstance(cutoff, float), "cutoff must be a floating-point number"
        
        b, a = sig.butter(filter_order, cutoff)
        
        # Apply the filter
        filtered_signal = sig.filtfilt(b, a, voltage_signal)
        
        # Add noise
        noise_amplitude = float(self.SIGNAL_PARAMS['NOISE_AMPLITUDE'])
        assert isinstance(noise_amplitude, float), "noise_amplitude must be a floating-point number"
        
        noise = np.random.normal(0, noise_amplitude, len(filtered_signal))
        noisy_signal = filtered_signal + noise
        
        # Ensure both arrays are float64
        time_result = time.astype(np.float64)
        signal_result = noisy_signal.astype(np.float64)
        
        # Verify types
        assert np.issubdtype(time_result.dtype, np.floating), "time array must be floating-point type"
        assert np.issubdtype(signal_result.dtype, np.floating), "signal array must be floating-point type"
        
        return time_result, signal_result

    def test_initialization(self) -> None:
        """Test analyzer initialization with parameter validation."""
        # Test valid initialization
        analyzer = NRZAnalyzer(
            data=self.test_data,
            sample_rate=float(self.SIGNAL_PARAMS['SAMPLE_RATE']),
            bit_rate=float(self.SIGNAL_PARAMS['BIT_RATE'])
        )
        self.assertIsInstance(analyzer, NRZAnalyzer)
        
        # Test with invalid data type
        with self.assertRaises(AssertionError):
            NRZAnalyzer(
                data="invalid_data",  # type: ignore
                sample_rate=float(self.SIGNAL_PARAMS['SAMPLE_RATE']),
                bit_rate=float(self.SIGNAL_PARAMS['BIT_RATE'])
            )
        
        # Test with invalid sample rate type
        with self.assertRaises(AssertionError):
            NRZAnalyzer(
                data=self.test_data,
                sample_rate="256e9",  # type: ignore
                bit_rate=float(self.SIGNAL_PARAMS['BIT_RATE'])
            )
        
        # Test with invalid bit rate type
        with self.assertRaises(AssertionError):
            NRZAnalyzer(
                data=self.test_data,
                sample_rate=float(self.SIGNAL_PARAMS['SAMPLE_RATE']),
                bit_rate="112e9"  # type: ignore
            )
        
        # Test with missing required data
        invalid_data = {'voltage': self.nrz_signal}  # Missing 'time'
        with self.assertRaises(AssertionError):
            NRZAnalyzer(
                data=invalid_data,
                sample_rate=float(self.SIGNAL_PARAMS['SAMPLE_RATE']),
                bit_rate=float(self.SIGNAL_PARAMS['BIT_RATE'])
            )
        
        # Test with invalid array types
        invalid_data = {
            'time': self.time_data,
            'voltage': self.nrz_signal.astype(int)  # Integer instead of float
        }
        with self.assertRaises(AssertionError):
            NRZAnalyzer(
                data=invalid_data,
                sample_rate=float(self.SIGNAL_PARAMS['SAMPLE_RATE']),
                bit_rate=float(self.SIGNAL_PARAMS['BIT_RATE'])
            )
        
        # Test with mismatched array lengths
        invalid_data = {
            'time': self.time_data[:-10],  # Shortened array
            'voltage': self.nrz_signal
        }
        with self.assertRaises(AssertionError):
            NRZAnalyzer(
                data=invalid_data,
                sample_rate=float(self.SIGNAL_PARAMS['SAMPLE_RATE']),
                bit_rate=float(self.SIGNAL_PARAMS['BIT_RATE'])
            )

    def test_validate_signal_arrays(self) -> None:
        """Test signal array validation with various edge cases."""
        # Test valid arrays
        self.analyzer._validate_signal_arrays(self.time_data, self.nrz_signal)
        
        # Test with invalid types
        with self.assertRaises(AssertionError):
            self.analyzer._validate_signal_arrays("invalid", self.nrz_signal)  # type: ignore
        
        with self.assertRaises(AssertionError):
            self.analyzer._validate_signal_arrays(self.time_data, "invalid")  # type: ignore
        
        # Test with integer arrays
        with self.assertRaises(AssertionError):
            self.analyzer._validate_signal_arrays(
                self.time_data.astype(int),
                self.nrz_signal
            )
        
        with self.assertRaises(AssertionError):
            self.analyzer._validate_signal_arrays(
                self.time_data,
                self.nrz_signal.astype(int)
            )
        
        # Test with mismatched lengths
        with self.assertRaises(AssertionError):
            self.analyzer._validate_signal_arrays(
                self.time_data[:-10],
                self.nrz_signal
            )
        
        # Test with empty arrays
        with self.assertRaises(AssertionError):
            self.analyzer._validate_signal_arrays(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64)
            )
        
        # Test with NaN values
        nan_array = self.nrz_signal.copy()
        nan_array[10] = np.nan
        with self.assertRaises(AssertionError):
            self.analyzer._validate_signal_arrays(
                self.time_data,
                nan_array
            )
        
        # Test with infinite values
        inf_array = self.nrz_signal.copy()
        inf_array[10] = np.inf
        with self.assertRaises(AssertionError):
            self.analyzer._validate_signal_arrays(
                self.time_data,
                inf_array
            )
        
        # Test with non-monotonic time
        non_monotonic = self.time_data.copy()
        non_monotonic[10] = non_monotonic[20]  # Create duplicate times
        with self.assertRaises(AssertionError):
            self.analyzer._validate_signal_arrays(
                non_monotonic,
                self.nrz_signal
            )

    def test_analyze_level_separation(self) -> None:
        """Test NRZ level separation analysis with type validation."""
        # Test with default parameters
        results = self.analyzer.analyze_level_separation()
        
        # Check result type
        self.assertIsInstance(results, NRZLevels)
        
        # Check level_means array
        self.assertIsInstance(results.level_means, np.ndarray)
        self.assertTrue(np.issubdtype(results.level_means.dtype, np.floating))
        self.assertEqual(len(results.level_means), 2)  # NRZ has 2 levels
        
        # Check numeric properties
        self.assertIsInstance(results.level_separation, float)
        self.assertIsInstance(results.uniformity, float)
        
        # Check value ranges
        self.assertGreater(results.level_separation, 0.0)
        self.assertGreaterEqual(results.uniformity, 0.0)
        self.assertLessEqual(results.uniformity, 1.0)
        
        # Test with custom column name
        results = self.analyzer.analyze_level_separation('voltage')
        self.assertIsInstance(results, NRZLevels)
        
        # Test with invalid column
        with self.assertRaises(ValueError):
            self.analyzer.analyze_level_separation('invalid_column')
        
        # Test with invalid threshold type
        with self.assertRaises(AssertionError):
            self.analyzer.analyze_level_separation(threshold="0.1")  # type: ignore
        
        # Test with invalid threshold value
        with self.assertRaises(AssertionError):
            self.analyzer.analyze_level_separation(threshold=1.5)  # Should be between 0 and 1

    def test_find_voltage_levels(self) -> None:
        """Test voltage level detection with type validation."""
        # Generate histogram data
        hist, bins = np.histogram(self.nrz_signal, bins=100)
        
        # Convert to float64 to ensure proper types
        hist = hist.astype(np.float64)
        bins = bins.astype(np.float64)
        
        # Find levels
        levels = self.analyzer._find_voltage_levels(hist, bins)
        
        # Check result type and properties
        self.assertIsInstance(levels, np.ndarray)
        self.assertTrue(np.issubdtype(levels.dtype, np.floating))
        self.assertEqual(levels.shape[0], 2)  # Should find exactly 2 levels for NRZ
        
        # Test with invalid histogram type
        with self.assertRaises(AssertionError):
            self.analyzer._find_voltage_levels(hist.astype(int), bins)
        
        # Test with invalid bin edges type
        with self.assertRaises(AssertionError):
            self.analyzer._find_voltage_levels(hist, bins.astype(int))
        
        # Test with insufficient data
        with self.assertRaises(ValueError):
            self.analyzer._find_voltage_levels(hist[:3], bins[:4])

    def test_analyze_eye_diagram(self) -> None:
        """Test eye diagram analysis with type validation."""
        # Test with default parameters
        results = self.analyzer.analyze_eye_diagram()
        
        # Check result type
        self.assertIsInstance(results, EyeResults)
        
        # Check numeric properties
        self.assertIsInstance(results.eye_height, float)
        self.assertIsInstance(results.eye_width, float)
        self.assertIsInstance(results.eye_amplitude, float)
        
        # Check value ranges
        self.assertGreaterEqual(results.eye_height, 0.0)
        self.assertGreaterEqual(results.eye_width, 0.0)
        self.assertGreaterEqual(results.eye_amplitude, 0.0)
        
        # Test with custom column names
        results = self.analyzer.analyze_eye_diagram('voltage', 'time')
        self.assertIsInstance(results, EyeResults)
        
        # Test with invalid column
        with self.assertRaises(ValueError):
            self.analyzer.analyze_eye_diagram('invalid_column')
        
        # Test with invalid UI period type
        with self.assertRaises(AssertionError):
            self.analyzer.analyze_eye_diagram(ui_period="8.9e-12")  # type: ignore
        
        # Test with invalid UI period value
        with self.assertRaises(AssertionError):
            self.analyzer.analyze_eye_diagram(ui_period=-1.0)  # Should be positive

    def test_calculate_eye_height(self) -> None:
        """Test eye height calculation with type validation."""
        # Calculate eye height
        eye_height = self.analyzer._calculate_eye_height(self.nrz_signal)
        
        # Check result type and value
        self.assertIsInstance(eye_height, float)
        self.assertGreaterEqual(eye_height, 0.0)
        
        # Test with invalid input type
        with self.assertRaises(ValueError):
            self.analyzer._calculate_eye_height(self.nrz_signal.astype(int))

    def test_calculate_eye_width(self) -> None:
        """Test eye width calculation with type validation."""
        # Calculate eye width
        eye_width = self.analyzer._calculate_eye_width(
            self.nrz_signal,
            self.time_data,
            ui_period=1.0 / self.SIGNAL_PARAMS['BIT_RATE']
        )
        
        # Check result type and value
        self.assertIsInstance(eye_width, float)
        self.assertGreaterEqual(eye_width, 0.0)
        self.assertLessEqual(eye_width, 1.0)  # Width in UI units should be <= 1
        
        # Test with invalid voltage data type
        with self.assertRaises(ValueError):
            self.analyzer._calculate_eye_width(
                self.nrz_signal.astype(int),
                self.time_data,
                ui_period=1.0 / self.SIGNAL_PARAMS['BIT_RATE']
            )
        
        # Test with invalid time data type
        with self.assertRaises(ValueError):
            self.analyzer._calculate_eye_width(
                self.nrz_signal,
                self.time_data.astype(int),
                ui_period=1.0 / self.SIGNAL_PARAMS['BIT_RATE']
            )
        
        # Test with invalid UI period type
        with self.assertRaises(AssertionError):
            self.analyzer._calculate_eye_width(
                self.nrz_signal,
                self.time_data,
                ui_period="1.0"  # type: ignore
            )
        
        # Test with invalid UI period value
        with self.assertRaises(AssertionError):
            self.analyzer._calculate_eye_width(
                self.nrz_signal,
                self.time_data,
                ui_period=-1.0
            )

    def test_analyze_jitter(self) -> None:
        """Test jitter analysis with type validation."""
        # Test normal operation
        results = self.analyzer.analyze_jitter()
        
        # Check result type
        self.assertIsInstance(results, JitterResults)
        
        # Check numeric properties
        self.assertIsInstance(results.total_jitter, float)
        self.assertIsInstance(results.random_jitter, float)
        self.assertIsInstance(results.deterministic_jitter, float)
        self.assertIsInstance(results.jitter_peak_to_peak, float)
        
        # Check value ranges
        self.assertGreaterEqual(results.total_jitter, 0.0)
        self.assertGreaterEqual(results.random_jitter, 0.0)
        self.assertGreaterEqual(results.deterministic_jitter, 0.0)
        self.assertGreaterEqual(results.jitter_peak_to_peak, 0.0)

    def test_calculate_total_jitter(self) -> None:
        """Test total jitter calculation with type validation."""
        # Calculate total jitter
        jitter = self.analyzer._calculate_total_jitter()
        
        # Check result type and value
        self.assertIsInstance(jitter, float)
        self.assertGreaterEqual(jitter, 0.0)
        
        # Create data with insufficient crossings
        flat_data = {
            'time': self.time_data,
            'voltage': np.ones_like(self.time_data)  # Constant signal, no crossings
        }
        
        flat_analyzer = NRZAnalyzer(
            data=flat_data,
            sample_rate=self.SIGNAL_PARAMS['SAMPLE_RATE'],
            bit_rate=self.SIGNAL_PARAMS['BIT_RATE']
        )
        
        # Test with insufficient crossings
        with self.assertRaises(ValueError):
            flat_analyzer._calculate_total_jitter()

    def test_calculate_random_jitter(self) -> None:
        """Test random jitter calculation with type validation."""
        # Calculate random jitter
        jitter = self.analyzer._calculate_random_jitter()
        
        # Check result type and value
        self.assertIsInstance(jitter, float)
        self.assertGreaterEqual(jitter, 0.0)
        
        # Create data with insufficient crossings
        flat_data = {
            'time': self.time_data,
            'voltage': np.ones_like(self.time_data)  # Constant signal, no crossings
        }
        
        flat_analyzer = NRZAnalyzer(
            data=flat_data,
            sample_rate=self.SIGNAL_PARAMS['SAMPLE_RATE'],
            bit_rate=self.SIGNAL_PARAMS['BIT_RATE']
        )
        
        # Test with insufficient crossings
        with self.assertRaises(ValueError):
            flat_analyzer._calculate_random_jitter()

    def test_calculate_deterministic_jitter(self) -> None:
        """Test deterministic jitter calculation with type validation."""
        # Calculate deterministic jitter
        jitter = self.analyzer._calculate_deterministic_jitter()
        
        # Check result type and value
        self.assertIsInstance(jitter, float)
        self.assertGreaterEqual(jitter, 0.0)
        
        # Create corrupted analyzer to test error handling
        corrupted_analyzer = NRZAnalyzer(
            data=self.test_data,
            sample_rate=self.SIGNAL_PARAMS['SAMPLE_RATE'],
            bit_rate=self.SIGNAL_PARAMS['BIT_RATE']
        )
        
        # Force error in total jitter calculation
        corrupted_analyzer._calculate_total_jitter = lambda: 1.0
        corrupted_analyzer._calculate_random_jitter = lambda: 2.0
        
        # Test with total_jitter < random_jitter case
        deterministic_jitter = corrupted_analyzer._calculate_deterministic_jitter()
        self.assertEqual(deterministic_jitter, 0.0)  # Should return 0 instead of negative

    def test_analyze_signal_quality(self) -> None:
        """Test comprehensive signal quality analysis with type validation."""
        # Test with default parameters
        results = self.analyzer.analyze_signal_quality()
        
        # Check result type
        self.assertIsInstance(results, dict)
        
        # Check required metrics
        required_metrics = [
            'level_separation',
            'level_uniformity',
            'eye_height',
            'eye_width',
            'total_jitter',
            'random_jitter',
            'snr_db'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, results)
            self.assertIsInstance(results[metric], float)
        
        # Test with invalid measurement time type
        with self.assertRaises(AssertionError):
            self.analyzer.analyze_signal_quality(measurement_time="1.0")  # type: ignore
        
        # Test with invalid measurement time value
        with self.assertRaises(AssertionError):
            self.analyzer.analyze_signal_quality(measurement_time=-1.0)

    def test_normalize_signal(self) -> None:
        """Test signal normalization with type validation."""
        # Test with valid signal
        normalized = self.analyzer._normalize_signal(self.nrz_signal)
        
        # Check result type
        self.assertIsInstance(normalized, np.ndarray)
        self.assertTrue(np.issubdtype(normalized.dtype, np.floating))
        
        # Check normalization properties
        self.assertAlmostEqual(np.mean(normalized), 0.0, delta=0.1)  # Mean close to 0
        max_abs = np.max(np.abs(normalized))
        self.assertLessEqual(max_abs, 1.0 + 1e-10)  # Max absolute value ≤ 1
        
        # Test with invalid signal type
        with self.assertRaises(AssertionError):
            self.analyzer._normalize_signal(self.nrz_signal.astype(int))
        
        # Test with empty signal
        with self.assertRaises(AssertionError):
            self.analyzer._normalize_signal(np.array([], dtype=np.float64))
        
        # Test with constant signal
        constant_signal = np.full_like(self.nrz_signal, 5.0)
        normalized_constant = self.analyzer._normalize_signal(constant_signal)
        self.assertTrue(np.all(normalized_constant == 0.0))  # Should be all zeros
        
        # Test with tiny signal
        tiny_signal = np.random.normal(0, 1e-10, len(self.nrz_signal)).astype(np.float64)
        normalized_tiny = self.analyzer._normalize_signal(tiny_signal)
        self.assertFalse(np.any(np.isnan(normalized_tiny)))  # Should handle tiny signals
        self.assertFalse(np.any(np.isinf(normalized_tiny)))  # No infinities

    def test_analyze_bit_error_rate(self) -> None:
        """Test BER analysis with type validation."""
        # Test with default parameters
        ber = self.analyzer.analyze_bit_error_rate()
        
        # Check result type and range
        self.assertIsInstance(ber, float)
        self.assertGreaterEqual(ber, 0.0)
        self.assertLessEqual(ber, 1.0)
        
        # Test with custom threshold
        ber = self.analyzer.analyze_bit_error_rate(threshold=0.1)
        self.assertIsInstance(ber, float)
        
        # Test with custom sample offset
        ber = self.analyzer.analyze_bit_error_rate(sample_offset=0.75)
        self.assertIsInstance(ber, float)
        
        # Test with invalid threshold type
        with self.assertRaises(AssertionError):
            self.analyzer.analyze_bit_error_rate(threshold="0.1")  # type: ignore
        
        # Test with invalid sample offset type
        with self.assertRaises(AssertionError):
            self.analyzer.analyze_bit_error_rate(sample_offset="0.5")  # type: ignore
        
        # Test with invalid sample offset range
        with self.assertRaises(AssertionError):
            self.analyzer.analyze_bit_error_rate(sample_offset=1.5)  # Out of range 0-1
        
        # Create data with insufficient transitions
        flat_data = {
            'time': self.time_data,
            'voltage': np.ones_like(self.time_data)  # Constant signal, no transitions
        }
        
        flat_analyzer = NRZAnalyzer(
            data=flat_data,
            sample_rate=self.SIGNAL_PARAMS['SAMPLE_RATE'],
            bit_rate=self.SIGNAL_PARAMS['BIT_RATE']
        )
        
        # Test with insufficient transitions
        with self.assertRaises(ValueError):
            flat_analyzer.analyze_bit_error_rate()


if __name__ == '__main__':
    unittest.main()
