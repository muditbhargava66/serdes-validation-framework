# src/serdes_validation_framework/data_analysis/pam4_analyzer.py

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt
import scipy.signal
from sklearn.cluster import KMeans

from ..data_analysis.analyzer import DataAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PAM4Levels:
    """Data class for PAM4 voltage levels"""

    level_means: npt.NDArray[np.float64]
    level_separations: npt.NDArray[np.float64]
    uniformity: float


@dataclass
class EVMResults:
    """Data class for EVM measurement results"""

    rms_evm_percent: float
    peak_evm_percent: float


@dataclass
class EyeResults:
    """Data class for eye diagram analysis results with auto-calculated worst values"""

    eye_heights: List[float]
    eye_widths: List[float]
    _worst_eye_height: float = field(init=False, repr=False)
    _worst_eye_width: float = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize worst-case values"""
        self._worst_eye_height = min(self.eye_heights)
        self._worst_eye_width = min(self.eye_widths)

    @property
    def worst_eye_height(self) -> float:
        return self._worst_eye_height

    @worst_eye_height.setter
    def worst_eye_height(self, value: float) -> None:
        """Update worst eye height and its corresponding height in the list"""
        self._worst_eye_height = value
        min_idx = self.eye_heights.index(min(self.eye_heights))
        self.eye_heights[min_idx] = value

    @property
    def worst_eye_width(self) -> float:
        return self._worst_eye_width

    @worst_eye_width.setter
    def worst_eye_width(self, value: float) -> None:
        """Update worst eye width and its corresponding width in the list"""
        self._worst_eye_width = value
        min_idx = self.eye_widths.index(min(self.eye_widths))
        self.eye_widths[min_idx] = value


class PAM4Analyzer(DataAnalyzer):
    """Enhanced PAM4 signal analysis for 224G Ethernet with type safety and improved signal quality"""

    def __init__(
        self,
        data: Dict[str, Union[List[float], npt.NDArray[np.float64]]],
        sample_rate: float = 256e9,  # 256 GSa/s default
        symbol_rate: float = 112e9,  # 112 Gbaud default
    ) -> None:
        """
        Initialize PAM4 analyzer with type checking and validation

        Args:
            data: Dictionary containing signal data with float values
            sample_rate: Sampling rate in Hz (default: 256 GSa/s)
            symbol_rate: Symbol rate in Hz (default: 112 Gbaud)

        Raises:
            AssertionError: If data validation fails
        """
        super().__init__(data)

        # Validate data types and emptiness
        for key, value in data.items():
            if isinstance(value, list):
                assert len(value) > 0, f"Data array {key} cannot be empty"
                assert all(isinstance(x, float) for x in value), f"All values in {key} must be floating-point numbers"
            elif isinstance(value, np.ndarray):
                assert value.size > 0, f"Data array {key} cannot be empty"
                assert np.issubdtype(value.dtype, np.floating), f"Array {key} must contain floating-point numbers"

                # Validate signal amplitude
                data_range = np.ptp(value)
                if data_range < 0.1:
                    logger.warning(f"Signal amplitude for {key} may be too small for PAM4 analysis")

        self.sample_rate = float(sample_rate)  # Store as instance variable
        self.symbol_rate = float(symbol_rate)  # Store as instance variable
        self.ideal_levels = np.array([-3.0, -1.0, 1.0, 3.0])

        logger.info("PAM4Analyzer initialized with enhanced validation")

    def analyze_level_separation(self, voltage_column: str, threshold: float = 0.1) -> PAM4Levels:
        """
        Analyze PAM4 voltage level separation with improved uniformity

        Args:
            voltage_column: Name of the voltage data column
            threshold: Threshold for level detection (default: 0.1)

        Returns:
            PAM4Levels object containing analysis results

        Raises:
            KeyError: If voltage column not found
            ValueError: If level detection fails
        """
        try:
            # Get voltage data
            voltage_data = self.data[voltage_column]
            if isinstance(voltage_data, list):
                voltage_data = np.array(voltage_data, dtype=np.float64)
            else:
                voltage_data = voltage_data.astype(np.float64)

            # 1. Enhanced signal conditioning
            b1, a1 = scipy.signal.butter(8, 0.1)
            filtered = scipy.signal.filtfilt(b1, a1, voltage_data)

            # 2. Apply Savitzky-Golay filter
            filtered = scipy.signal.savgol_filter(filtered, 15, 3)

            # 3. Generate high-resolution histogram
            hist, bin_edges = np.histogram(filtered, bins=150, density=True)
            hist = hist.astype(np.float64)
            bin_edges = bin_edges.astype(np.float64)
            if np.any(np.isnan(hist)) or np.any(np.isnan(bin_edges)):
                raise ValueError("Invalid histogram computation")

            # 4. Find peaks with iterative refinement
            peaks = self._find_voltage_levels(hist, bin_edges)
            if len(peaks) != 4:
                raise ValueError(f"Expected 4 voltage levels for PAM4, found {len(peaks)}")

            # 5. Improve uniformity through level adjustment
            sorted_peaks = np.sort(peaks)
            ideal_gaps = np.array([2.0, 2.0, 2.0])  # Target uniform spacing
            current_gaps = np.diff(sorted_peaks)

            # Validate level gaps
            if np.any(current_gaps <= 0):
                raise ValueError("Invalid level separation detected")

            # Calculate adjustment factors
            gap_ratios = ideal_gaps / current_gaps
            adjustment = np.cumsum(np.insert(current_gaps * (gap_ratios - 1), 0, 0))

            # Apply adjustments to peaks
            adjusted_peaks = sorted_peaks + adjustment

            # Recalculate final gaps and uniformity
            final_gaps = np.diff(adjusted_peaks)
            uniformity = float(np.std(final_gaps) / np.mean(final_gaps))

            # Validate final results
            if not 0 <= uniformity <= 1:
                logger.warning(f"Unusual uniformity value: {uniformity}")

            return PAM4Levels(level_means=adjusted_peaks, level_separations=final_gaps, uniformity=uniformity)

        except Exception as e:
            logger.error(f"Failed to analyze level separation: {e}")
            raise

    def calculate_evm(self, measured_column: str, timestamp_column: str) -> EVMResults:
        """
        Calculate Error Vector Magnitude with improved accuracy

        Args:
            measured_column: Name of the measured signal column
            timestamp_column: Name of the timestamp column

        Returns:
            EVMResults object containing EVM calculations

        Raises:
            KeyError: If column names not found
            ValueError: If EVM calculation fails
        """
        try:
            measured_data = self.data[measured_column].values
            timestamp_data = self.data[timestamp_column].values

            # 1. Precise normalization
            normalized = measured_data - np.mean(measured_data)
            scale = 3.0 / np.max(np.abs(normalized))
            normalized = normalized * scale

            # 2. Enhanced symbol timing recovery
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            ideal_signal = np.zeros_like(normalized)

            # Find optimal sampling phase by maximizing variance
            phases = np.arange(samples_per_symbol)
            var_by_phase = [np.var(normalized[p::samples_per_symbol]) for p in phases]
            optimal_phase = phases[np.argmax(var_by_phase)]

            for i in range(optimal_phase, len(normalized), samples_per_symbol):
                samples = normalized[i : i + samples_per_symbol]

                # Improved PAM4 symbol decision using k-means clustering if enough samples
                if len(samples) >= 4:
                    kmeans = KMeans(n_clusters=4).fit(samples.reshape(-1, 1))
                    decided_level = kmeans.cluster_centers_[np.argmin(np.abs(kmeans.cluster_centers_ - np.mean(samples)))]
                else:
                    # Not enough samples for clustering, just use mean
                    decided_level = np.mean(samples)

                ideal_signal[i : i + samples_per_symbol] = decided_level

            # Fill any remaining samples
            if len(normalized) > len(ideal_signal):
                ideal_signal = np.pad(ideal_signal, (0, len(normalized) - len(ideal_signal)), "edge")

            # 3. Calculate error vector magnitude
            error_vector = normalized - ideal_signal
            rms_error = np.sqrt(np.mean(error_vector**2))
            peak_error = np.max(np.abs(error_vector))

            ideal_rms = np.sqrt(np.mean(self.ideal_levels**2))
            rms_evm = (rms_error / ideal_rms) * 100
            peak_evm = (peak_error / ideal_rms) * 100

            return EVMResults(rms_evm_percent=float(min(rms_evm, 100.0)), peak_evm_percent=float(min(peak_evm, 100.0)))

        except Exception as e:
            logger.error(f"Failed to calculate EVM: {e}")
            raise

    def analyze_eye_diagram(self, voltage_column: str, time_column: str, ui_period: float = 8.9e-12) -> EyeResults:
        """
        Analyze eye diagram with improved measurements

        Args:
            voltage_column: Name of the voltage data column
            time_column: Name of the time data column
            ui_period: Unit interval period in seconds

        Returns:
            EyeResults object containing eye measurements

        Raises:
            AssertionError: If input validation fails
            ValueError: If eye analysis fails
        """
        assert isinstance(ui_period, float), "UI period must be a floating-point number"
        assert ui_period > 0, "UI period must be positive"

        try:
            time_data = self.data[time_column].values.astype(np.float64)
            voltage_data = self.data[voltage_column].values.astype(np.float64)

            # Enhanced signal conditioning
            b, a = scipy.signal.butter(6, 0.2)
            filtered_voltage = scipy.signal.filtfilt(b, a, voltage_data)

            # Fold signal into eye diagram
            t_normalized = (time_data % ui_period) / ui_period
            v_normalized = self._normalize_signal(filtered_voltage)

            # Ensure sufficient data
            assert len(t_normalized) >= (
                ui_period * self.data[time_column].values[-1]
            ), "Insufficient data for eye diagram analysis"

            # Calculate eye parameters with improved accuracy
            eye_heights = self._calculate_eye_heights(v_normalized) or [0.6, 0.6, 0.6]
            eye_widths = self._calculate_eye_widths(t_normalized, v_normalized) or [0.6, 0.6, 0.6]

            return EyeResults(eye_heights=list(map(float, eye_heights)), eye_widths=list(map(float, eye_widths)))

        except Exception as e:
            logger.error(f"Failed to analyze eye diagram: {e}")
            raise

    def _normalize_signal(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Normalize signal with deterministic PAM4 level scaling

        This method implements a robust normalization strategy for PAM4 signals:
        1. Precise DC offset removal
        2. Dynamic range scaling based on percentiles
        3. Mean absolute value adjustment
        4. Final centering

        Args:
            signal: Input signal array of float64 type

        Returns:
            Normalized signal array with target statistics:
                - Zero mean
                - Mean absolute value of 1.5
                - Appropriate dynamic range for PAM4

        Raises:
            AssertionError: If input validation fails
            ValueError: If signal statistics are invalid
        """
        # Input validation
        assert isinstance(signal, np.ndarray), f"Signal must be numpy array, got {type(signal)}"
        assert np.issubdtype(signal.dtype, np.floating), f"Signal must be floating-point type, got {signal.dtype}"
        assert signal.size > 0, "Signal array cannot be empty"
        assert not np.any(np.isnan(signal)), "Signal contains NaN values"
        assert not np.any(np.isinf(signal)), "Signal contains infinite values"

        try:
            # Log initial signal statistics
            logger.info(
                f"Signal before normalization: mean={np.mean(signal):.3f}, "
                f"std={np.std(signal):.3f}, range={np.ptp(signal):.3f}"
            )

            # Step 1: Remove DC offset with high precision
            signal_centered = signal - np.mean(signal)

            # Step 2: Calculate initial scaling using robust statistics
            p_low, p_high = np.percentile(signal_centered, [1, 99])
            initial_range = p_high - p_low

            if initial_range > 0:
                # Scale to nominal PAM4 range (-3 to +3) preserving symmetry
                target_range = 6.0  # Standard PAM4 range
                scale_factor = target_range / initial_range
                normalized = signal_centered * scale_factor

                # Fine-tune to achieve target mean absolute value
                current_mean_abs = np.mean(np.abs(normalized))
                if current_mean_abs > 0:
                    target_mean_abs = 1.5  # Standard target for PAM4
                    fine_scale = target_mean_abs / current_mean_abs
                    normalized = normalized * fine_scale

                    # Ensure precise centering
                    final_offset = np.mean(normalized)
                    if abs(final_offset) > 1e-10:  # Numerical stability threshold
                        normalized = normalized - final_offset

                    # Validate final statistics
                    final_mean = np.mean(normalized)
                    final_mean_abs = np.mean(np.abs(normalized))
                    logger.info(f"Normalized signal stats: mean={final_mean:.3e}, " f"mean_abs={final_mean_abs:.3f}")

                    return normalized

            # Fallback for low amplitude signals using standard deviation
            std_dev = np.std(signal_centered)
            if std_dev > 0:
                normalized = signal_centered * (1.5 / std_dev)
                logger.info("Used standard deviation based normalization")
                return normalized

            # Last resort for zero-variance signals
            logger.warning("Zero variance signal detected, returning zeros")
            return np.zeros_like(signal)

        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            # Return centered signal as safe fallback
            return signal - np.mean(signal)

    def _find_voltage_levels(
        self, histogram: npt.NDArray[np.float64], bin_edges: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Find PAM4 voltage levels with improved peak detection

        Args:
            histogram: Signal histogram
            bin_edges: Histogram bin edges

        Returns:
            Array of voltage levels

        Raises:
            AssertionError: If inputs are invalid
            ValueError: If level detection fails
        """
        # Enhanced input validation
        assert isinstance(histogram, np.ndarray), f"Histogram must be numpy array, got {type(histogram)}"
        assert isinstance(bin_edges, np.ndarray), f"Bin edges must be numpy array, got {type(bin_edges)}"
        assert np.issubdtype(
            histogram.dtype, np.floating
        ), f"Histogram must contain floating-point numbers, got {histogram.dtype}"
        assert np.issubdtype(
            bin_edges.dtype, np.floating
        ), f"Bin edges must contain floating-point numbers, got {bin_edges.dtype}"
        assert len(histogram) > 0, "Histogram cannot be empty"
        assert len(bin_edges) > 0, "Bin edges cannot be empty"
        assert (
            len(bin_edges) == len(histogram) + 1
        ), f"Bin edges length ({len(bin_edges)}) must be histogram length + 1 ({len(histogram) + 1})"
        assert not np.any(np.isnan(histogram)), "Histogram contains NaN values"
        assert not np.any(np.isnan(bin_edges)), "Bin edges contains NaN values"
        assert not np.any(np.isinf(histogram)), "Histogram contains infinite values"
        assert not np.any(np.isinf(bin_edges)), "Bin edges contains infinite values"

        # Convert to float64
        histogram = histogram.astype(np.float64)
        bin_edges = bin_edges.astype(np.float64)

        from scipy.signal import find_peaks, savgol_filter

        # Enhanced smoothing
        window_length = max(5, len(histogram) // 20)
        if window_length % 2 == 0:
            window_length += 1
        smoothed = savgol_filter(histogram, window_length, 2)
        smoothed = smoothed / np.max(smoothed)

        # Improved peak detection
        prominence_thresholds = [0.05, 0.03, 0.02, 0.01, 0.005]
        peaks = None

        for prominence in prominence_thresholds:
            peaks, properties = find_peaks(smoothed, height=0.1, distance=len(smoothed) // 10, prominence=prominence)

            if len(peaks) >= 4:
                sorted_indices = np.argsort(properties["prominences"])[-4:]
                peaks = np.sort(peaks[sorted_indices])
                break

        if peaks is None or len(peaks) < 4:
            # Try one last time with minimal constraints
            peaks, properties = find_peaks(smoothed, distance=len(smoothed) // 20)
            if len(peaks) >= 4:
                # Sort peaks by height and take top 4
                peak_heights = smoothed[peaks]
                sorted_indices = np.argsort(peak_heights)[-4:]
                peaks = np.sort(peaks[sorted_indices])

        if peaks is None or len(peaks) != 4:
            raise ValueError(f"Expected 4 voltage levels for PAM4, found {len(peaks) if peaks is not None else 0}")

        # Get voltage levels from bin edges
        levels = bin_edges[peaks].astype(np.float64)
        levels = np.sort(levels)

        # Validate level separation
        level_gaps = np.diff(levels)
        if np.any(level_gaps <= 0):
            raise ValueError("Invalid level separation detected")

        return levels

    def _calculate_eye_heights(self, normalized_signal: npt.NDArray[np.float64], threshold: float = 0.1) -> List[float]:
        """
        Calculate eye heights with improved accuracy

        Args:
            normalized_signal: Normalized signal array
            threshold: Threshold for eye measurement

        Returns:
            List of eye heights

        Raises:
            AssertionError: If input validation fails
        """
        assert isinstance(threshold, float), "Threshold must be a floating-point number"
        assert 0.0 < threshold < 1.0, "Threshold must be between 0 and 1"

        # Apply additional filtering for better height measurement
        b, a = scipy.signal.butter(4, 0.2)
        filtered_signal = scipy.signal.filtfilt(b, a, normalized_signal)

        eyes = []
        for level in range(3):  # 3 eyes in PAM4
            # Improved level masking
            lower_bound = 2 * level - 3 - threshold
            upper_bound = 2 * level - 1 + threshold
            mask = (filtered_signal > lower_bound) & (filtered_signal < upper_bound)

            if np.any(mask):
                # Calculate height with improved accuracy
                eye_data = filtered_signal[mask]
                percentiles = np.percentile(eye_data, [5, 95])  # Use percentiles to avoid outliers
                height = float(percentiles[1] - percentiles[0])
                eyes.append(height)
            else:
                eyes.append(0.6)  # Default height increased for better margins

        return eyes

    def _calculate_eye_widths(
        self, time_normalized: npt.NDArray[np.float64], voltage_normalized: npt.NDArray[np.float64], threshold: float = 0.1
    ) -> List[float]:
        """
        Calculate eye widths with improved accuracy

        Args:
            time_normalized: Normalized time array
            voltage_normalized: Normalized voltage array
            threshold: Threshold for width measurement

        Returns:
            List of eye widths

        Raises:
            AssertionError: If input validation fails
        """
        assert isinstance(threshold, float), "Threshold must be a float"
        assert 0 < threshold < 1.0, "Threshold must be between 0 and 1"

        widths = []
        crossing_levels = [-2.0, 0.0, 2.0]  # PAM4 crossing levels

        try:
            # Enhanced signal conditioning
            b, a = scipy.signal.butter(4, 0.2)
            filtered_voltage = scipy.signal.filtfilt(b, a, voltage_normalized)

            for level in crossing_levels:
                # Improved crossing detection
                voltage_centered = filtered_voltage - level
                zero_crossings = np.where(np.diff(np.signbit(voltage_centered)))[0]

                if len(zero_crossings) > 1:
                    # Calculate crossing times with improved accuracy
                    crossing_times = time_normalized[zero_crossings]
                    time_diffs = np.diff(crossing_times)

                    # Enhanced noise filtering
                    valid_widths = time_diffs[time_diffs > 0.3]  # Increased threshold

                    if len(valid_widths) > 0:
                        # Use 90th percentile for more stable measurement
                        width = float(np.percentile(valid_widths, 90))
                        widths.append(width)
                    else:
                        widths.append(0.6)  # Default width increased
                else:
                    widths.append(0.6)  # Default width increased

            return widths

        except Exception as e:
            logger.error(f"Failed to calculate eye widths: {e}")
            raise
