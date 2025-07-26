"""
USB4 Signal Analysis Module

This module provides comprehensive USB4 signal analysis capabilities including:
- Dual-lane signal processing and analysis
- Lane skew measurement and compensation
- Signal quality assessment for both lanes
- USB4-specific signal integrity validation

Features:
- Dual-lane simultaneous analysis
- Lane skew detection and compensation
- Signal quality metrics
- USB4 compliance validation
- Spread spectrum clocking analysis
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy.signal

from ...data_analysis.analyzer import DataAnalyzer
from .constants import USB4SignalMode, USB4SignalSpecs, USB4Specs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LaneSignalResults:
    """Results for individual lane signal analysis"""

    lane_id: int
    signal_quality: float  # Overall signal quality score (0-1)
    amplitude: float  # Signal amplitude in volts
    noise_floor: float  # Noise floor level
    snr_db: float  # Signal-to-noise ratio in dB
    rise_time: float  # Rise time in seconds
    fall_time: float  # Fall time in seconds
    jitter_rms: float  # RMS jitter in UI
    eye_height: float  # Eye height (normalized)
    eye_width: float  # Eye width (normalized)
    compliance_status: bool  # USB4 compliance status


@dataclass
class SSCResults:
    """Spread Spectrum Clocking analysis results"""

    modulation_depth: float  # SSC modulation depth (%)
    modulation_frequency: float  # SSC frequency in Hz
    profile_type: str  # SSC profile (down_spread, center_spread)
    compliance_status: bool  # SSC compliance status
    frequency_deviation: float  # Frequency deviation from nominal


@dataclass
class USB4SignalResults:
    """Comprehensive USB4 dual-lane signal analysis results"""

    lane0_results: LaneSignalResults
    lane1_results: LaneSignalResults
    lane_skew: float  # Lane skew in seconds
    skew_compensation: float  # Applied skew compensation
    overall_quality: float  # Overall signal quality (0-1)
    ssc_results: Optional[SSCResults]  # SSC analysis results
    compliance_status: bool  # Overall USB4 compliance
    recommendations: List[str] = field(default_factory=list)


@dataclass
class USB4AnalyzerConfig:
    """Configuration for USB4 signal analyzer"""

    sample_rate: float = 256e9  # Default 256 GSa/s
    symbol_rate: float = 20e9  # Default 20 Gbaud per lane
    mode: USB4SignalMode = USB4SignalMode.GEN2X2
    enable_ssc_analysis: bool = True
    skew_compensation_enabled: bool = True
    noise_bandwidth: float = 1e9  # Noise measurement bandwidth

    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.sample_rate > 0, "Sample rate must be positive"
        assert self.symbol_rate > 0, "Symbol rate must be positive"
        assert self.sample_rate >= 2 * self.symbol_rate, "Sample rate must satisfy Nyquist criterion"
        assert self.noise_bandwidth > 0, "Noise bandwidth must be positive"


class USB4SignalAnalyzer(DataAnalyzer):
    """
    USB4 dual-lane signal analyzer with comprehensive signal integrity analysis

    This analyzer provides:
    - Simultaneous dual-lane signal processing
    - Lane skew measurement and compensation
    - USB4-specific signal quality metrics
    - Spread spectrum clocking analysis
    - Compliance validation against USB4 specifications
    """

    def __init__(self, config: USB4AnalyzerConfig):
        """
        Initialize USB4 signal analyzer

        Args:
            config: USB4 analyzer configuration
        """
        # Initialize with empty data - will be populated during analysis
        super().__init__({})

        self.config = config
        self.usb4_specs = USB4Specs()
        self.signal_specs = USB4SignalSpecs()

        # Calculate derived parameters
        self.samples_per_symbol = int(self.config.sample_rate / self.config.symbol_rate)
        self.ui_period = 1.0 / self.config.symbol_rate

        logger.info(f"USB4SignalAnalyzer initialized for {self.config.mode.name} mode")
        logger.info(f"Sample rate: {self.config.sample_rate/1e9:.1f} GSa/s")
        logger.info(f"Symbol rate: {self.config.symbol_rate/1e9:.1f} Gbaud")
        logger.info(f"Samples per symbol: {self.samples_per_symbol}")

    def analyze_dual_lane_signal(
        self,
        lane0_data: npt.NDArray[np.float64],
        lane1_data: npt.NDArray[np.float64],
        time_data: Optional[npt.NDArray[np.float64]] = None,
    ) -> USB4SignalResults:
        """
        Analyze both USB4 lanes simultaneously

        Args:
            lane0_data: Lane 0 signal data
            lane1_data: Lane 1 signal data
            time_data: Optional time base data

        Returns:
            USB4SignalResults containing comprehensive analysis

        Raises:
            ValueError: If input data is invalid
            AssertionError: If data validation fails
        """
        # Input validation
        self._validate_signal_data(lane0_data, lane1_data, time_data)

        try:
            logger.info("Starting dual-lane USB4 signal analysis")

            # Generate time base if not provided
            if time_data is None:
                time_data = np.arange(len(lane0_data)) / self.config.sample_rate

            # Measure lane skew first
            lane_skew = self.measure_lane_skew(lane0_data, lane1_data)
            logger.info(f"Measured lane skew: {lane_skew*1e12:.2f} ps")

            # Apply skew compensation if enabled
            skew_compensation = 0.0
            if self.config.skew_compensation_enabled and abs(lane_skew) > self.signal_specs.LANE_SKEW_TOLERANCE:
                lane0_compensated, lane1_compensated, skew_compensation = self._apply_skew_compensation(
                    lane0_data, lane1_data, lane_skew
                )
                logger.info(f"Applied skew compensation: {skew_compensation*1e12:.2f} ps")
            else:
                lane0_compensated, lane1_compensated = lane0_data, lane1_data

            # Analyze individual lanes
            lane0_results = self._analyze_single_lane(lane0_compensated, time_data, lane_id=0)
            lane1_results = self._analyze_single_lane(lane1_compensated, time_data, lane_id=1)

            # Analyze spread spectrum clocking if enabled
            ssc_results = None
            if self.config.enable_ssc_analysis:
                ssc_results = self._analyze_ssc(lane0_compensated, lane1_compensated)

            # Calculate overall quality and compliance
            overall_quality = self._calculate_overall_quality(lane0_results, lane1_results, lane_skew)
            compliance_status = self._check_overall_compliance(lane0_results, lane1_results, lane_skew, ssc_results)

            # Generate recommendations
            recommendations = self._generate_recommendations(lane0_results, lane1_results, lane_skew, ssc_results)

            results = USB4SignalResults(
                lane0_results=lane0_results,
                lane1_results=lane1_results,
                lane_skew=lane_skew,
                skew_compensation=skew_compensation,
                overall_quality=overall_quality,
                ssc_results=ssc_results,
                compliance_status=compliance_status,
                recommendations=recommendations,
            )

            logger.info(f"Dual-lane analysis complete. Overall quality: {overall_quality:.3f}")
            logger.info(f"Compliance status: {'PASS' if compliance_status else 'FAIL'}")

            return results

        except Exception as e:
            logger.error(f"Dual-lane signal analysis failed: {e}")
            raise

    def measure_lane_skew(self, lane0_data: npt.NDArray[np.float64], lane1_data: npt.NDArray[np.float64]) -> float:
        """
        Measure skew between USB4 lanes using cross-correlation

        Args:
            lane0_data: Lane 0 signal data
            lane1_data: Lane 1 signal data

        Returns:
            Lane skew in seconds (positive means lane1 leads lane0)

        Raises:
            ValueError: If skew measurement fails
        """
        try:
            logger.debug("Measuring lane skew using cross-correlation")

            # Ensure equal length arrays
            min_length = min(len(lane0_data), len(lane1_data))
            lane0_trimmed = lane0_data[:min_length]
            lane1_trimmed = lane1_data[:min_length]

            # Apply bandpass filtering to improve correlation
            nyquist = self.config.sample_rate / 2
            low_freq = self.config.symbol_rate * 0.1
            high_freq = min(self.config.symbol_rate * 2, nyquist * 0.8)

            sos = scipy.signal.butter(6, [low_freq / nyquist, high_freq / nyquist], btype="band", output="sos")
            lane0_filtered = scipy.signal.sosfilt(sos, lane0_trimmed)
            lane1_filtered = scipy.signal.sosfilt(sos, lane1_trimmed)

            # Compute cross-correlation
            correlation = scipy.signal.correlate(lane1_filtered, lane0_filtered, mode="full")

            # Find peak correlation
            peak_index = np.argmax(np.abs(correlation))
            center_index = len(correlation) // 2

            # Convert to time skew
            sample_skew = peak_index - center_index
            time_skew = sample_skew / self.config.sample_rate

            # Validate skew measurement
            max_expected_skew = self.signal_specs.SKEW_COMPENSATION_RANGE
            if abs(time_skew) > max_expected_skew:
                logger.warning(f"Measured skew ({time_skew*1e12:.2f} ps) exceeds expected range")

            return float(time_skew)

        except Exception as e:
            logger.error(f"Lane skew measurement failed: {e}")
            raise ValueError(f"Failed to measure lane skew: {e}")

    def _validate_signal_data(
        self,
        lane0_data: npt.NDArray[np.float64],
        lane1_data: npt.NDArray[np.float64],
        time_data: Optional[npt.NDArray[np.float64]],
    ) -> None:
        """Validate input signal data"""
        # Type validation
        assert isinstance(lane0_data, np.ndarray), "Lane 0 data must be numpy array"
        assert isinstance(lane1_data, np.ndarray), "Lane 1 data must be numpy array"
        assert np.issubdtype(lane0_data.dtype, np.floating), "Lane 0 data must be floating-point"
        assert np.issubdtype(lane1_data.dtype, np.floating), "Lane 1 data must be floating-point"

        # Size validation
        assert lane0_data.size > 0, "Lane 0 data cannot be empty"
        assert lane1_data.size > 0, "Lane 1 data cannot be empty"
        assert lane0_data.size >= 100, "Lane 0 data must have at least 100 samples"
        assert lane1_data.size >= 100, "Lane 1 data must have at least 100 samples"
        assert abs(len(lane0_data) - len(lane1_data)) <= 1, "Lane data lengths must be nearly equal"

        # Content validation
        assert not np.any(np.isnan(lane0_data)), "Lane 0 data contains NaN values"
        assert not np.any(np.isnan(lane1_data)), "Lane 1 data contains NaN values"
        assert not np.any(np.isinf(lane0_data)), "Lane 0 data contains infinite values"
        assert not np.any(np.isinf(lane1_data)), "Lane 1 data contains infinite values"

        # Signal amplitude validation
        lane0_range = np.ptp(lane0_data)
        lane1_range = np.ptp(lane1_data)
        assert lane0_range > 0.01, "Lane 0 signal amplitude too small"
        assert lane1_range > 0.01, "Lane 1 signal amplitude too small"

        # Time data validation if provided
        if time_data is not None:
            assert isinstance(time_data, np.ndarray), "Time data must be numpy array"
            assert np.issubdtype(time_data.dtype, np.floating), "Time data must be floating-point"
            assert len(time_data) >= min(len(lane0_data), len(lane1_data)), "Time data length insufficient"
            assert not np.any(np.isnan(time_data)), "Time data contains NaN values"
            assert not np.any(np.isinf(time_data)), "Time data contains infinite values"
            assert np.all(np.diff(time_data) > 0), "Time data must be monotonically increasing"

    def _analyze_single_lane(
        self, signal_data: npt.NDArray[np.float64], time_data: npt.NDArray[np.float64], lane_id: int
    ) -> LaneSignalResults:
        """
        Analyze individual lane signal quality

        Args:
            signal_data: Lane signal data
            time_data: Time base data
            lane_id: Lane identifier (0 or 1)

        Returns:
            LaneSignalResults containing analysis results
        """
        try:
            logger.debug(f"Analyzing lane {lane_id} signal quality")

            # Basic signal measurements
            amplitude = self._measure_signal_amplitude(signal_data)
            noise_floor = self._measure_noise_floor(signal_data)
            snr_db = self._calculate_snr(signal_data, noise_floor)

            # Timing measurements
            rise_time, fall_time = self._measure_edge_times(signal_data, time_data)

            # Jitter analysis
            jitter_rms = self._measure_rms_jitter(signal_data)

            # Eye diagram analysis
            eye_height, eye_width = self._analyze_eye_diagram(signal_data, time_data)

            # Calculate overall signal quality
            signal_quality = self._calculate_lane_quality(amplitude, noise_floor, snr_db, jitter_rms, eye_height, eye_width)

            # Check compliance
            compliance_status = self._check_lane_compliance(amplitude, rise_time, fall_time, jitter_rms, eye_height, eye_width)

            return LaneSignalResults(
                lane_id=lane_id,
                signal_quality=signal_quality,
                amplitude=amplitude,
                noise_floor=noise_floor,
                snr_db=snr_db,
                rise_time=rise_time,
                fall_time=fall_time,
                jitter_rms=jitter_rms,
                eye_height=eye_height,
                eye_width=eye_width,
                compliance_status=compliance_status,
            )

        except Exception as e:
            logger.error(f"Lane {lane_id} analysis failed: {e}")
            raise

    def _apply_skew_compensation(
        self, lane0_data: npt.NDArray[np.float64], lane1_data: npt.NDArray[np.float64], measured_skew: float
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """
        Apply skew compensation to align lanes

        Args:
            lane0_data: Lane 0 signal data
            lane1_data: Lane 1 signal data
            measured_skew: Measured skew in seconds

        Returns:
            Tuple of (compensated_lane0, compensated_lane1, applied_compensation)
        """
        try:
            # Calculate compensation in samples
            compensation_samples = int(abs(measured_skew) * self.config.sample_rate)

            # Limit compensation to available range
            max_compensation_samples = min(len(lane0_data), len(lane1_data)) // 10
            compensation_samples = min(compensation_samples, max_compensation_samples)

            if measured_skew > 0:
                # Lane 1 leads, delay lane 1
                lane0_compensated = lane0_data[:-compensation_samples] if compensation_samples > 0 else lane0_data
                lane1_compensated = lane1_data[compensation_samples:]
            else:
                # Lane 0 leads, delay lane 0
                lane0_compensated = lane0_data[compensation_samples:]
                lane1_compensated = lane1_data[:-compensation_samples] if compensation_samples > 0 else lane1_data

            # Ensure equal lengths
            min_length = min(len(lane0_compensated), len(lane1_compensated))
            lane0_compensated = lane0_compensated[:min_length]
            lane1_compensated = lane1_compensated[:min_length]

            applied_compensation = compensation_samples / self.config.sample_rate

            return lane0_compensated, lane1_compensated, applied_compensation

        except Exception as e:
            logger.error(f"Skew compensation failed: {e}")
            return lane0_data, lane1_data, 0.0

    def _analyze_ssc(self, lane0_data: npt.NDArray[np.float64], lane1_data: npt.NDArray[np.float64]) -> SSCResults:
        """
        Analyze spread spectrum clocking characteristics

        Args:
            lane0_data: Lane 0 signal data
            lane1_data: Lane 1 signal data

        Returns:
            SSCResults containing SSC analysis
        """
        try:
            logger.debug("Analyzing spread spectrum clocking")

            # Combine lanes for better SSC detection
            combined_signal = (lane0_data + lane1_data) / 2

            # Extract clock component using PLL-like approach
            clock_signal = self._extract_clock_component(combined_signal)

            # Analyze frequency modulation
            instantaneous_freq = self._calculate_instantaneous_frequency(clock_signal)

            # Detect modulation characteristics
            modulation_depth = self._calculate_ssc_modulation_depth(instantaneous_freq)
            modulation_frequency = self._calculate_ssc_frequency(instantaneous_freq)

            # Determine profile type
            profile_type = self._determine_ssc_profile(instantaneous_freq)

            # Calculate frequency deviation
            nominal_freq = self.config.symbol_rate
            frequency_deviation = np.std(instantaneous_freq - nominal_freq) / nominal_freq

            # Check SSC compliance
            compliance_status = self._check_ssc_compliance(modulation_depth, modulation_frequency, profile_type)

            return SSCResults(
                modulation_depth=modulation_depth,
                modulation_frequency=modulation_frequency,
                profile_type=profile_type,
                compliance_status=compliance_status,
                frequency_deviation=frequency_deviation,
            )

        except Exception as e:
            logger.error(f"SSC analysis failed: {e}")
            # Return default results on failure
            return SSCResults(
                modulation_depth=0.0,
                modulation_frequency=0.0,
                profile_type="unknown",
                compliance_status=False,
                frequency_deviation=0.0,
            )

    def _measure_signal_amplitude(self, signal_data: npt.NDArray[np.float64]) -> float:
        """Measure signal amplitude using robust statistics"""
        # Use percentiles to avoid outliers
        p5, p95 = np.percentile(signal_data, [5, 95])
        return float(p95 - p5)

    def _measure_noise_floor(self, signal_data: npt.NDArray[np.float64]) -> float:
        """Measure noise floor using high-frequency content"""
        # High-pass filter to isolate noise
        nyquist = self.config.sample_rate / 2
        cutoff = min(self.config.symbol_rate * 3, nyquist * 0.9)
        sos = scipy.signal.butter(4, cutoff / nyquist, btype="high", output="sos")
        noise_signal = scipy.signal.sosfilt(sos, signal_data)
        return float(np.std(noise_signal))

    def _calculate_snr(self, signal_data: npt.NDArray[np.float64], noise_floor: float) -> float:
        """Calculate signal-to-noise ratio in dB"""
        signal_power = np.var(signal_data)
        noise_power = noise_floor**2
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            return float(10 * np.log10(snr_linear))
        else:
            return float(60.0)  # Very high SNR if no measurable noise

    def _measure_edge_times(
        self, signal_data: npt.NDArray[np.float64], time_data: npt.NDArray[np.float64]
    ) -> Tuple[float, float]:
        """Measure rise and fall times"""
        # Find edges using derivative
        diff_signal = np.diff(signal_data)

        # Find rising and falling edges
        rising_edges = np.where(diff_signal > np.std(diff_signal) * 2)[0]
        falling_edges = np.where(diff_signal < -np.std(diff_signal) * 2)[0]

        # Measure edge times (10% to 90% transition)
        rise_time = self._measure_transition_time(signal_data, time_data, rising_edges, rising=True)
        fall_time = self._measure_transition_time(signal_data, time_data, falling_edges, rising=False)

        return rise_time, fall_time

    def _measure_transition_time(
        self,
        signal_data: npt.NDArray[np.float64],
        time_data: npt.NDArray[np.float64],
        edge_indices: npt.NDArray[np.int_],
        rising: bool,
    ) -> float:
        """Measure 10%-90% transition time for edges"""
        if len(edge_indices) == 0:
            return float(self.signal_specs.RISE_TIME_MAX)  # Default if no edges found

        transition_times = []

        for edge_idx in edge_indices[:10]:  # Analyze first 10 edges
            # Define window around edge
            window_size = self.samples_per_symbol // 4
            start_idx = max(0, edge_idx - window_size)
            end_idx = min(len(signal_data), edge_idx + window_size)

            if end_idx - start_idx < window_size:
                continue

            window_signal = signal_data[start_idx:end_idx]
            window_time = time_data[start_idx:end_idx]

            # Find 10% and 90% levels
            if rising:
                min_val, max_val = np.min(window_signal), np.max(window_signal)
                level_10 = min_val + 0.1 * (max_val - min_val)
                level_90 = min_val + 0.9 * (max_val - min_val)
            else:
                min_val, max_val = np.min(window_signal), np.max(window_signal)
                level_10 = max_val - 0.1 * (max_val - min_val)
                level_90 = max_val - 0.9 * (max_val - min_val)

            # Find crossing times
            try:
                idx_10 = np.where(np.diff(np.sign(window_signal - level_10)))[0][0]
                idx_90 = np.where(np.diff(np.sign(window_signal - level_90)))[0][0]

                if abs(idx_90 - idx_10) > 0:
                    transition_time = abs(window_time[idx_90] - window_time[idx_10])
                    transition_times.append(transition_time)
            except (IndexError, ValueError):
                continue

        if transition_times:
            return float(np.median(transition_times))
        else:
            return float(self.signal_specs.RISE_TIME_MAX)

    def _measure_rms_jitter(self, signal_data: npt.NDArray[np.float64]) -> float:
        """Measure RMS jitter using zero-crossing analysis"""
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]

        if len(zero_crossings) < 10:
            return float(0.1)  # Default jitter if insufficient crossings

        # Calculate crossing intervals
        crossing_intervals = np.diff(zero_crossings) / self.config.sample_rate

        # Expected interval (half UI for differential signal)
        expected_interval = self.ui_period / 2

        # Calculate jitter as deviation from expected
        jitter_seconds = np.std(crossing_intervals - expected_interval)
        jitter_ui = jitter_seconds / self.ui_period

        return float(jitter_ui)

    def _analyze_eye_diagram(
        self, signal_data: npt.NDArray[np.float64], time_data: npt.NDArray[np.float64]
    ) -> Tuple[float, float]:
        """Analyze eye diagram for height and width"""
        # Fold signal into eye diagram
        samples_per_ui = int(self.config.sample_rate * self.ui_period)

        if samples_per_ui < 10:
            return 0.6, 0.6  # Default values if insufficient resolution

        # Create eye diagram matrix
        num_eyes = len(signal_data) // samples_per_ui
        if num_eyes < 10:
            return 0.6, 0.6  # Need sufficient eyes for analysis

        eye_matrix = signal_data[: num_eyes * samples_per_ui].reshape(num_eyes, samples_per_ui)

        # Calculate eye height (vertical opening)
        eye_center_idx = samples_per_ui // 2
        center_samples = eye_matrix[:, eye_center_idx - 2 : eye_center_idx + 3]  # Small window around center

        # Use percentiles for robust measurement
        p5, p95 = np.percentile(center_samples, [5, 95])
        signal_range = np.ptp(signal_data)
        eye_height = (p95 - p5) / signal_range if signal_range > 0 else 0.6

        # Calculate eye width (horizontal opening)
        # Find crossing points at mid-level
        mid_level = np.median(signal_data)
        eye_width = 0.6  # Default width

        for eye_trace in eye_matrix[:10]:  # Analyze first 10 eye traces
            crossings = np.where(np.diff(np.sign(eye_trace - mid_level)))[0]
            if len(crossings) >= 2:
                crossing_span = (crossings[-1] - crossings[0]) / samples_per_ui
                eye_width = max(eye_width, crossing_span)

        return float(min(eye_height, 1.0)), float(min(eye_width, 1.0))

    def _calculate_lane_quality(
        self, amplitude: float, noise_floor: float, snr_db: float, jitter_rms: float, eye_height: float, eye_width: float
    ) -> float:
        """Calculate overall lane signal quality score (0-1)"""
        # Normalize individual metrics
        amplitude_score = min(amplitude / self.signal_specs.VOD_MIN, 1.0)
        snr_score = min(snr_db / 20.0, 1.0)  # 20 dB target
        jitter_score = max(0, 1.0 - jitter_rms / self.signal_specs.RANDOM_JITTER_MAX)
        eye_height_score = eye_height
        eye_width_score = eye_width

        # Weighted combination
        quality = 0.2 * amplitude_score + 0.2 * snr_score + 0.2 * jitter_score + 0.2 * eye_height_score + 0.2 * eye_width_score

        return float(max(0.0, min(1.0, quality)))

    def _check_lane_compliance(
        self, amplitude: float, rise_time: float, fall_time: float, jitter_rms: float, eye_height: float, eye_width: float
    ) -> bool:
        """Check lane compliance against USB4 specifications"""
        checks = [
            self.signal_specs.VOD_MIN <= amplitude <= self.signal_specs.VOD_MAX,
            rise_time <= self.signal_specs.RISE_TIME_MAX,
            fall_time <= self.signal_specs.FALL_TIME_MAX,
            jitter_rms <= self.signal_specs.RANDOM_JITTER_MAX,
            eye_height >= self.signal_specs.EYE_HEIGHT_MIN,
            eye_width >= self.signal_specs.EYE_WIDTH_MIN,
        ]

        return all(checks)

    def _calculate_overall_quality(
        self, lane0_results: LaneSignalResults, lane1_results: LaneSignalResults, lane_skew: float
    ) -> float:
        """Calculate overall dual-lane signal quality"""
        # Average lane qualities
        avg_lane_quality = (lane0_results.signal_quality + lane1_results.signal_quality) / 2

        # Skew penalty
        skew_penalty = min(abs(lane_skew) / self.signal_specs.LANE_SKEW_TOLERANCE, 1.0)
        skew_factor = 1.0 - 0.2 * skew_penalty  # Up to 20% penalty for skew

        return float(avg_lane_quality * skew_factor)

    def _check_overall_compliance(
        self,
        lane0_results: LaneSignalResults,
        lane1_results: LaneSignalResults,
        lane_skew: float,
        ssc_results: Optional[SSCResults],
    ) -> bool:
        """Check overall USB4 compliance"""
        checks = [
            lane0_results.compliance_status,
            lane1_results.compliance_status,
            abs(lane_skew) <= self.signal_specs.LANE_SKEW_TOLERANCE,
        ]

        if ssc_results is not None:
            checks.append(ssc_results.compliance_status)

        return all(checks)

    def _generate_recommendations(
        self,
        lane0_results: LaneSignalResults,
        lane1_results: LaneSignalResults,
        lane_skew: float,
        ssc_results: Optional[SSCResults],
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        # Lane-specific recommendations
        for lane_results in [lane0_results, lane1_results]:
            lane_id = lane_results.lane_id

            if lane_results.snr_db < 15:
                recommendations.append(f"Lane {lane_id}: Improve signal-to-noise ratio (current: {lane_results.snr_db:.1f} dB)")

            if lane_results.jitter_rms > self.signal_specs.RANDOM_JITTER_MAX * 0.8:
                recommendations.append(f"Lane {lane_id}: Reduce jitter (current: {lane_results.jitter_rms:.3f} UI)")

            if lane_results.eye_height < self.signal_specs.EYE_HEIGHT_MIN * 1.2:
                recommendations.append(f"Lane {lane_id}: Improve eye height (current: {lane_results.eye_height:.3f})")

            if lane_results.rise_time > self.signal_specs.RISE_TIME_MAX * 0.8:
                recommendations.append(f"Lane {lane_id}: Improve rise time (current: {lane_results.rise_time*1e12:.1f} ps)")

        # Skew recommendations
        if abs(lane_skew) > self.signal_specs.LANE_SKEW_TOLERANCE * 0.5:
            recommendations.append(f"Reduce lane skew (current: {abs(lane_skew)*1e12:.1f} ps)")

        # SSC recommendations
        if ssc_results and not ssc_results.compliance_status:
            recommendations.append("Adjust spread spectrum clocking parameters")

        return recommendations

    def _extract_clock_component(self, signal_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Extract clock component from signal for SSC analysis"""
        # Use bandpass filter around symbol rate
        nyquist = self.config.sample_rate / 2
        center_freq = self.config.symbol_rate
        bandwidth = center_freq * 0.1

        low_freq = max(center_freq - bandwidth / 2, 1e6)
        high_freq = min(center_freq + bandwidth / 2, nyquist * 0.9)

        sos = scipy.signal.butter(6, [low_freq / nyquist, high_freq / nyquist], btype="band", output="sos")
        return scipy.signal.sosfilt(sos, signal_data)

    def _calculate_instantaneous_frequency(self, clock_signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate instantaneous frequency using Hilbert transform"""
        analytic_signal = scipy.signal.hilbert(clock_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_freq = np.diff(instantaneous_phase) * self.config.sample_rate / (2 * np.pi)

        # Pad to maintain length
        return np.concatenate([[instantaneous_freq[0]], instantaneous_freq])

    def _calculate_ssc_modulation_depth(self, instantaneous_freq: npt.NDArray[np.float64]) -> float:
        """Calculate SSC modulation depth"""
        # Remove DC component and calculate peak-to-peak deviation
        freq_mean = np.mean(instantaneous_freq)
        freq_deviation = instantaneous_freq - freq_mean

        # Use robust statistics to avoid outliers
        p5, p95 = np.percentile(freq_deviation, [5, 95])
        peak_to_peak_deviation = p95 - p5

        # Calculate modulation depth as percentage
        if freq_mean > 0:
            modulation_depth = (peak_to_peak_deviation / freq_mean) * 100
        else:
            modulation_depth = 0.0

        return float(modulation_depth)

    def _calculate_ssc_frequency(self, instantaneous_freq: npt.NDArray[np.float64]) -> float:
        """Calculate SSC modulation frequency"""
        # Detrend and find dominant frequency component
        detrended = instantaneous_freq - np.mean(instantaneous_freq)

        # Use FFT to find modulation frequency
        fft_result = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended), 1 / self.config.sample_rate)

        # Find peak in expected SSC range (20-50 kHz)
        valid_range = (freqs > 20e3) & (freqs < 50e3)
        if np.any(valid_range):
            peak_idx = np.argmax(np.abs(fft_result[valid_range]))
            ssc_freq = freqs[valid_range][peak_idx]
            return float(abs(ssc_freq))
        else:
            return float(33e3)  # Default SSC frequency

    def _determine_ssc_profile(self, instantaneous_freq: npt.NDArray[np.float64]) -> str:
        """Determine SSC profile type (down_spread or center_spread)"""
        freq_mean = np.mean(instantaneous_freq)
        freq_deviation = instantaneous_freq - freq_mean

        # Use more robust thresholds based on signal statistics
        threshold = np.std(freq_deviation) * 0.5

        # Check if frequency goes significantly above and below nominal
        has_positive = np.any(freq_deviation > threshold)
        has_negative = np.any(freq_deviation < -threshold)

        # Additional check: look at the distribution of deviations
        positive_fraction = np.sum(freq_deviation > 0) / len(freq_deviation)

        # If mostly negative deviations or asymmetric, it's down-spread
        if not has_positive or positive_fraction < 0.3:
            return "down_spread"
        elif has_positive and has_negative and 0.3 <= positive_fraction <= 0.7:
            return "center_spread"
        else:
            return "down_spread"

    def _check_ssc_compliance(self, modulation_depth: float, modulation_frequency: float, profile_type: str) -> bool:
        """Check SSC compliance against USB4 specifications"""
        depth_ok = modulation_depth <= self.signal_specs.SSC_MODULATION_DEPTH
        freq_ok = self.signal_specs.SSC_FREQUENCY_MIN <= modulation_frequency <= self.signal_specs.SSC_FREQUENCY_MAX
        profile_ok = profile_type == self.signal_specs.SSC_PROFILE

        return depth_ok and freq_ok and profile_ok


__all__ = ["USB4SignalAnalyzer", "USB4AnalyzerConfig", "USB4SignalResults", "LaneSignalResults", "SSCResults"]
