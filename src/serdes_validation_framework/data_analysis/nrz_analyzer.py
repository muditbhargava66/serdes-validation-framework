"""
NRZ Signal Analysis Module

This module provides analysis capabilities for NRZ (Non-Return-to-Zero) signals
with comprehensive type checking and validation.

Features:
- Level separation analysis
- Eye diagram measurements
- Jitter analysis
- Signal quality metrics
"""

import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NRZLevels:
    """Data class for NRZ voltage levels with validation"""

    level_means: npt.NDArray[np.float64]
    level_separation: float
    uniformity: float

    def __post_init__(self) -> None:
        """Validate NRZ level measurements"""
        # Type validation
        assert isinstance(self.level_means, np.ndarray), "Level means must be numpy array"
        assert isinstance(self.level_separation, float), "Level separation must be float"
        assert isinstance(self.uniformity, float), "Uniformity must be float"

        # Array validation
        assert np.issubdtype(
            self.level_means.dtype, np.floating
        ), f"Level means must be floating-point, got {self.level_means.dtype}"
        assert len(self.level_means) == 2, f"Expected 2 NRZ levels, got {len(self.level_means)}"

        # Value validation
        assert self.level_separation > 0, f"Level separation must be positive, got {self.level_separation}"
        assert 0 <= self.uniformity <= 1, f"Uniformity must be between 0 and 1, got {self.uniformity}"


@dataclass
class EyeResults:
    """Eye diagram measurement results"""

    eye_height: float
    eye_width: float
    eye_amplitude: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate eye measurements"""
        # Type validation
        assert isinstance(self.eye_height, float), "Eye height must be float"
        assert isinstance(self.eye_width, float), "Eye width must be float"

        # Value validation
        assert self.eye_height >= 0, f"Eye height must be non-negative, got {self.eye_height}"
        assert self.eye_width >= 0, f"Eye width must be non-negative, got {self.eye_width}"

        # Calculate amplitude
        self.eye_amplitude = float(self.eye_height * 2)


@dataclass
class JitterResults:
    """Jitter measurement results"""

    total_jitter: float
    random_jitter: float
    deterministic_jitter: float
    jitter_peak_to_peak: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate jitter measurements"""
        # Type validation
        assert isinstance(self.total_jitter, float), "Total jitter must be float"
        assert isinstance(self.random_jitter, float), "Random jitter must be float"
        assert isinstance(self.deterministic_jitter, float), "Deterministic jitter must be float"

        # Value validation
        assert self.total_jitter >= 0, "Total jitter must be non-negative"
        assert self.random_jitter >= 0, "Random jitter must be non-negative"
        assert self.deterministic_jitter >= 0, "Deterministic jitter must be non-negative"

        # Calculate peak-to-peak
        self.jitter_peak_to_peak = float(self.random_jitter + self.deterministic_jitter)


class NRZAnalyzer:
    """NRZ signal analyzer with strict type checking"""

    def __init__(self, data: Dict[str, npt.NDArray[np.float64]], sample_rate: float = 256e9, bit_rate: float = 112e9) -> None:
        """
        Initialize NRZ analyzer

        Args:
            data: Dictionary with 'time' and 'voltage' arrays
            sample_rate: Sampling rate in Hz
            bit_rate: Bit rate in Hz

        Raises:
            AssertionError: If parameters are invalid
        """
        # Validate input types
        assert isinstance(data, dict), "Data must be dictionary"
        assert isinstance(sample_rate, float), "Sample rate must be float"
        assert isinstance(bit_rate, float), "Bit rate must be float"

        # Validate required data
        assert "time" in data and "voltage" in data, "Data must contain 'time' and 'voltage' arrays"

        # Validate array types
        self._validate_signal_arrays(data["time"], data["voltage"])

        # Store validated data
        self.data = {"time": data["time"].astype(np.float64), "voltage": data["voltage"].astype(np.float64)}
        self.sample_rate = float(sample_rate)
        self.bit_rate = float(bit_rate)

        logger.info("NRZ analyzer initialized")

    def _validate_signal_arrays(self, time_data: npt.NDArray[np.float64], voltage_data: npt.NDArray[np.float64]) -> None:
        """
        Validate signal array properties

        Args:
            time_data: Time points array
            voltage_data: Voltage measurements array

        Raises:
            AssertionError: If validation fails
        """
        # Type validation
        assert isinstance(time_data, np.ndarray), f"Time data must be numpy array, got {type(time_data)}"
        assert isinstance(voltage_data, np.ndarray), f"Voltage data must be numpy array, got {type(voltage_data)}"

        # Data type validation
        assert np.issubdtype(time_data.dtype, np.floating), f"Time data must be floating-point, got {time_data.dtype}"
        assert np.issubdtype(voltage_data.dtype, np.floating), f"Voltage data must be floating-point, got {voltage_data.dtype}"

        # Array validation
        assert len(time_data) == len(voltage_data), f"Array length mismatch: {len(time_data)} != {len(voltage_data)}"
        assert len(time_data) > 0, "Arrays cannot be empty"

        # Value validation
        assert not np.any(np.isnan(time_data)), "Time data contains NaN values"
        assert not np.any(np.isnan(voltage_data)), "Voltage data contains NaN values"
        assert not np.any(np.isinf(time_data)), "Time data contains infinite values"
        assert not np.any(np.isinf(voltage_data)), "Voltage data contains infinite values"

        # Time array validation
        assert np.all(np.diff(time_data) > 0), "Time data must be monotonically increasing"

    def analyze_level_separation(self, voltage_column: str = "voltage", threshold: float = 0.1) -> NRZLevels:
        """
        Analyze NRZ voltage level separation

        Args:
            voltage_column: Name of voltage data column
            threshold: Detection threshold

        Returns:
            NRZLevels object with analysis results

        Raises:
            ValueError: If analysis fails
        """
        # Validate inputs
        assert isinstance(voltage_column, str), "Voltage column must be string"
        assert isinstance(threshold, float), "Threshold must be float"
        assert 0 < threshold < 1, "Threshold must be between 0 and 1"

        try:
            # Get voltage data
            voltage_data = self.data[voltage_column]

            # Find levels using histogram (ensure floating-point)
            hist, bins = np.histogram(voltage_data, bins=100)
            hist = hist.astype(np.float64)
            level_means = self._find_voltage_levels(hist, bins)

            # Calculate separation
            level_separation = float(np.abs(level_means[1] - level_means[0]))

            # Calculate uniformity (coefficient of variation, clamped to [0,1])
            mean_abs = np.mean(np.abs(voltage_data))
            if mean_abs > 0:
                uniformity = float(min(1.0, np.std(voltage_data) / mean_abs))
            else:
                uniformity = 0.0

            return NRZLevels(level_means=level_means, level_separation=level_separation, uniformity=uniformity)

        except Exception as e:
            raise ValueError(f"Level analysis failed: {e}")

    def _find_voltage_levels(
        self, histogram: npt.NDArray[np.float64], bin_edges: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Find NRZ voltage levels from histogram

        Args:
            histogram: Signal histogram
            bin_edges: Histogram bin edges

        Returns:
            Array of voltage levels

        Raises:
            ValueError: If level detection fails
        """
        # Validate inputs
        assert isinstance(histogram, np.ndarray), "Histogram must be numpy array"
        assert isinstance(bin_edges, np.ndarray), "Bin edges must be numpy array"
        assert np.issubdtype(histogram.dtype, np.floating), "Histogram must be floating-point"
        assert np.issubdtype(bin_edges.dtype, np.floating), "Bin edges must be floating-point"

        try:
            # Find peaks with more restrictive parameters for NRZ
            from scipy.signal import find_peaks

            # Use higher threshold and minimum distance between peaks
            min_height = np.max(histogram) * 0.3  # Higher threshold
            min_distance = len(histogram) // 10  # Minimum distance between peaks

            peaks, properties = find_peaks(
                histogram, height=min_height, distance=min_distance, prominence=np.max(histogram) * 0.2
            )

            # If we still have too many peaks, take the two highest ones
            if len(peaks) > 2:
                # Sort by peak height and take top 2
                peak_heights = histogram[peaks]
                top_indices = np.argsort(peak_heights)[-2:]
                peaks = peaks[top_indices]
            elif len(peaks) < 2:
                # Fallback: use simple bimodal detection
                # Find the two most separated peaks in the histogram
                smoothed = np.convolve(histogram, np.ones(3) / 3, mode="same")
                all_peaks, _ = find_peaks(smoothed, height=np.max(smoothed) / 10)
                if len(all_peaks) >= 2:
                    # Take the first and last significant peaks
                    peaks = np.array([all_peaks[0], all_peaks[-1]])
                else:
                    # Final fallback: assume bimodal distribution
                    center = len(histogram) // 2
                    left_peak = np.argmax(histogram[:center])
                    right_peak = center + np.argmax(histogram[center:])
                    peaks = np.array([left_peak, right_peak])

            # Get voltage levels
            levels = bin_edges[peaks]
            return np.sort(levels).astype(np.float64)

        except Exception as e:
            raise ValueError(f"Level detection failed: {e}")

    def analyze_eye_diagram(
        self, voltage_column: str = "voltage", time_column: str = "time", ui_period: float = 8.9e-12
    ) -> EyeResults:
        """
        Analyze eye diagram parameters

        Args:
            voltage_column: Name of voltage column
            time_column: Name of time column
            ui_period: Unit interval in seconds

        Returns:
            EyeResults object with measurements

        Raises:
            ValueError: If analysis fails
        """
        # Validate inputs
        assert isinstance(voltage_column, str), "Voltage column must be string"
        assert isinstance(time_column, str), "Time column must be string"
        assert isinstance(ui_period, float), "UI period must be float"
        assert ui_period > 0, "UI period must be positive"

        try:
            # Get signal data
            voltage = self.data[voltage_column]
            time = self.data[time_column]

            # Calculate eye parameters
            eye_height = self._calculate_eye_height(voltage)
            eye_width = self._calculate_eye_width(voltage, time, ui_period)

            return EyeResults(eye_height=float(eye_height), eye_width=float(eye_width))

        except Exception as e:
            raise ValueError(f"Eye analysis failed: {e}")

    def _calculate_eye_height(self, voltage_data: npt.NDArray[np.float64]) -> float:
        """
        Calculate eye height

        Args:
            voltage_data: Voltage measurements array

        Returns:
            Eye height measurement

        Raises:
            ValueError: If calculation fails
        """
        # Validate inputs
        assert isinstance(voltage_data, np.ndarray), "Voltage data must be numpy array"
        if not np.issubdtype(voltage_data.dtype, np.floating):
            raise ValueError("Voltage data must be floating-point")

        try:
            # Find level separation
            levels = self.analyze_level_separation()

            # Calculate eye height
            mean_separation = float(levels.level_separation)
            noise_std = float(np.std(voltage_data))

            eye_height = mean_separation - 6 * noise_std  # 6-sigma
            return max(0.0, eye_height)  # Cannot be negative

        except Exception as e:
            raise ValueError(f"Eye height calculation failed: {e}")

    def _calculate_eye_width(
        self, voltage_data: npt.NDArray[np.float64], time_data: npt.NDArray[np.float64], ui_period: float
    ) -> float:
        """
        Calculate eye width

        Args:
            voltage_data: Voltage measurements array
            time_data: Time points array
            ui_period: Unit interval in seconds

        Returns:
            Eye width measurement

        Raises:
            ValueError: If calculation fails
        """
        # Validate inputs
        assert isinstance(voltage_data, np.ndarray), "Voltage data must be numpy array"
        assert isinstance(time_data, np.ndarray), "Time data must be numpy array"
        assert isinstance(ui_period, float), "UI period must be float"
        assert ui_period > 0, "UI period must be positive"

        # Type validation
        if not np.issubdtype(voltage_data.dtype, np.floating):
            raise ValueError("Voltage data must be floating-point")
        if not np.issubdtype(time_data.dtype, np.floating):
            raise ValueError("Time data must be floating-point")

        try:
            # Find zero crossings
            zero_crossings = np.where(np.diff(np.signbit(voltage_data)))[0]

            if len(zero_crossings) < 2:
                raise ValueError("Insufficient zero crossings")

            # Calculate crossing times
            crossing_times = time_data[zero_crossings]
            time_diffs = np.diff(crossing_times)

            # Calculate eye width
            eye_width = float(np.median(time_diffs))
            return min(eye_width, ui_period)  # Cannot exceed UI

        except Exception as e:
            raise ValueError(f"Eye width calculation failed: {e}")

    def analyze_jitter(self) -> JitterResults:
        """
        Analyze signal jitter components

        Returns:
            JitterResults object with measurements

        Raises:
            ValueError: If analysis fails
        """
        try:
            # Calculate jitter components
            total_jitter = self._calculate_total_jitter()
            random_jitter = self._calculate_random_jitter()
            deterministic_jitter = self._calculate_deterministic_jitter()

            return JitterResults(
                total_jitter=float(total_jitter),
                random_jitter=float(random_jitter),
                deterministic_jitter=float(deterministic_jitter),
            )

        except Exception as e:
            raise ValueError(f"Jitter analysis failed: {e}")

    def _calculate_total_jitter(self) -> float:
        """
        Calculate total jitter

        Returns:
            Total jitter measurement

        Raises:
            ValueError: If calculation fails
        """
        try:
            # Find crossing points
            voltage = self.data["voltage"]
            time = self.data["time"]
            crossings = np.where(np.diff(np.signbit(voltage)))[0]

            if len(crossings) < 2:
                raise ValueError("Insufficient crossing points")

            # Calculate crossing time differences
            crossing_times = time[crossings]
            time_diffs = np.diff(crossing_times)

            # Total jitter is peak-to-peak
            return float(np.max(time_diffs) - np.min(time_diffs))

        except Exception as e:
            raise ValueError(f"Total jitter calculation failed: {e}")

    def _calculate_random_jitter(self) -> float:
        """
        Calculate random jitter component

        Returns:
            Random jitter measurement

        Raises:
            ValueError: If calculation fails
        """
        try:
            # Find crossing points
            voltage = self.data["voltage"]
            time = self.data["time"]
            crossings = np.where(np.diff(np.signbit(voltage)))[0]

            if len(crossings) < 2:
                raise ValueError("Insufficient crossing points")

            # Calculate crossing time differences
            crossing_times = time[crossings]
            time_diffs = np.diff(crossing_times)

            # Random jitter is RMS value
            return float(np.std(time_diffs))

        except Exception as e:
            raise ValueError(f"Random jitter calculation failed: {e}")

    def _calculate_deterministic_jitter(self) -> float:
        """
        Calculate deterministic jitter component

        Returns:
            Deterministic jitter measurement

        Raises:
            ValueError: If calculation fails
        """
        try:
            # Total jitter = Random jitter + Deterministic jitter
            total_jitter = self._calculate_total_jitter()
            random_jitter = self._calculate_random_jitter()

            # Deterministic component is remainder
            deterministic_jitter = total_jitter - random_jitter
            return float(max(0.0, deterministic_jitter))  # Cannot be negative

        except Exception as e:
            raise ValueError(f"Deterministic jitter calculation failed: {e}")

    def analyze_signal_quality(self, measurement_time: float = 1e-6) -> Dict[str, float]:
        """
        Comprehensive signal quality analysis

        Args:
            measurement_time: Analysis duration in seconds

        Returns:
            Dictionary of quality metrics

        Raises:
            ValueError: If analysis fails
        """
        # Validate input
        assert isinstance(measurement_time, float), "Measurement time must be float"
        assert measurement_time > 0, "Measurement time must be positive"

        try:
            # Run analysis components
            level_results = self.analyze_level_separation()
            eye_results = self.analyze_eye_diagram()
            jitter_results = self.analyze_jitter()

            # Calculate SNR
            signal_power = float(np.mean(self.data["voltage"] ** 2))
            noise_power = float(np.var(self.data["voltage"]))
            snr = float(10 * np.log10(signal_power / noise_power))

            # Compile results
            quality_metrics = {
                "level_separation": float(level_results.level_separation),
                "level_uniformity": float(level_results.uniformity),
                "eye_height": float(eye_results.eye_height),
                "eye_width": float(eye_results.eye_width),
                "total_jitter": float(jitter_results.total_jitter),
                "random_jitter": float(jitter_results.random_jitter),
                "snr_db": float(snr),
            }

            return quality_metrics

        except Exception as e:
            raise ValueError(f"Signal quality analysis failed: {e}")

    def _normalize_signal(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Normalize signal amplitude

        Args:
            signal: Input signal array

        Returns:
            Normalized signal array

        Raises:
            ValueError: If normalization fails
        """
        # Validate input
        assert isinstance(signal, np.ndarray), "Signal must be numpy array"
        assert np.issubdtype(signal.dtype, np.floating), "Signal must be floating-point"
        assert len(signal) > 0, "Signal cannot be empty"

        try:
            # Remove DC offset
            signal = signal - np.mean(signal)

            # Scale to unit amplitude
            scale = float(np.max(np.abs(signal)))
            if scale > 0:
                signal = signal / scale

            return signal.astype(np.float64)

        except Exception as e:
            raise ValueError(f"Signal normalization failed: {e}")

    def analyze_bit_error_rate(self, threshold: float = 0.0, sample_offset: float = 0.5) -> float:
        """
        Estimate bit error rate (BER)

        Args:
            threshold: Decision threshold
            sample_offset: Sampling point offset (0-1)

        Returns:
            Estimated BER

        Raises:
            ValueError: If analysis fails
        """
        # Validate inputs
        assert isinstance(threshold, float), "Threshold must be float"
        assert isinstance(sample_offset, float), "Sample offset must be float"
        assert 0 <= sample_offset <= 1, "Sample offset must be between 0 and 1"

        try:
            # Get normalized signal
            voltage = self._normalize_signal(self.data["voltage"])

            # Find bit transitions
            transitions = np.where(np.diff(np.signbit(voltage)))[0]

            if len(transitions) < 2:
                raise ValueError("Insufficient transitions for BER analysis")

            # Calculate samples per bit
            samples_per_bit = int(self.sample_rate / self.bit_rate)

            # Sample at offset points
            sample_points = transitions + int(samples_per_bit * sample_offset)
            sample_points = sample_points[sample_points < len(voltage)]

            # Count errors
            decisions = voltage[sample_points] > threshold
            transitions = np.diff(decisions)
            error_count = np.sum(np.abs(transitions))

            # Calculate BER
            total_bits = len(sample_points)
            ber = float(error_count / total_bits)

            return float(max(1e-15, ber))  # Minimum measurable BER

        except Exception as e:
            raise ValueError(f"BER analysis failed: {e}")
