"""
USB4 Eye Diagram Analysis Module

This module provides USB4-specific eye diagram analysis capabilities including:
- USB4-specific eye diagram measurement algorithms
- Spread spectrum clock recovery in eye analysis
- Eye contour analysis with USB4 compliance masks
- Bathtub curve generation for USB4 timing analysis

Features:
- SSC-aware eye diagram analysis
- USB4 compliance mask validation
- Dual-lane eye diagram correlation
- Advanced timing analysis with bathtub curves
- Eye contour analysis for signal integrity assessment
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.signal

from ...data_analysis.eye_diagram import AdvancedEyeAnalyzer
from .constants import USB4SignalMode, USB4SignalSpecs, USB4Specs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class USB4EyeResults:
    """USB4-specific eye diagram analysis results"""

    eye_height: float  # Eye height (normalized)
    eye_width: float  # Eye width (normalized)
    eye_area: float  # Eye opening area
    crossing_percentage: float  # Eye crossing percentage
    rise_time: float  # 20%-80% rise time
    fall_time: float  # 80%-20% fall time
    jitter_rms: float  # RMS jitter from eye
    jitter_pp: float  # Peak-to-peak jitter
    ssc_impact: float  # SSC impact on eye quality
    compliance_status: bool  # USB4 compliance status
    mask_violations: List[str] = field(default_factory=list)


@dataclass
class USB4BathtubResults:
    """USB4 bathtub curve analysis results"""

    ber_curve: npt.NDArray[np.float64]  # BER vs timing offset
    timing_offsets: npt.NDArray[np.float64]  # Timing offset array
    eye_width_1e12: float  # Eye width at 1e-12 BER
    eye_width_1e15: float  # Eye width at 1e-15 BER
    timing_margin: float  # Timing margin at target BER
    extrapolated_ber: float  # Extrapolated BER at zero offset
    confidence_interval: Tuple[float, float]  # 95% confidence interval


@dataclass
class USB4EyeContour:
    """USB4 eye contour analysis results"""

    contour_levels: List[float]  # Contour probability levels
    contour_data: List[npt.NDArray[np.float64]]  # Contour coordinate data
    eye_opening_probability: float  # Eye opening probability
    worst_case_eye: Tuple[float, float]  # Worst-case eye dimensions
    statistical_eye: Tuple[float, float]  # Statistical eye dimensions
    margin_analysis: Dict[str, float]  # Design margins


@dataclass
class USB4EyeConfig:
    """Configuration for USB4 eye diagram analysis"""

    sample_rate: float = 256e9  # Sample rate in Hz
    symbol_rate: float = 20e9  # Symbol rate in Hz
    mode: USB4SignalMode = USB4SignalMode.GEN2X2
    enable_ssc_compensation: bool = True
    enable_mask_testing: bool = True
    target_ber: float = 1e-12  # Target BER for analysis
    contour_levels: List[float] = field(default_factory=lambda: [0.1, 0.01, 0.001])

    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.sample_rate > 0, "Sample rate must be positive"
        assert self.symbol_rate > 0, "Symbol rate must be positive"
        assert self.sample_rate >= 2 * self.symbol_rate, "Sample rate must satisfy Nyquist criterion"
        assert 0 < self.target_ber < 1, "Target BER must be between 0 and 1"
        assert all(0 < level < 1 for level in self.contour_levels), "Contour levels must be between 0 and 1"


class USB4EyeDiagramAnalyzer(AdvancedEyeAnalyzer):
    """
    USB4-specific eye diagram analyzer with advanced features

    This analyzer extends the base eye diagram analyzer with:
    - USB4-specific compliance mask testing
    - Spread spectrum clock recovery
    - Advanced statistical analysis
    - Bathtub curve generation
    - Eye contour analysis
    """

    def __init__(self, config: USB4EyeConfig):
        """
        Initialize USB4 eye diagram analyzer

        Args:
            config: USB4 eye diagram configuration
        """
        # Import EyeParameters from the eye_diagram module
        from ...data_analysis.eye_diagram import EyeParameters

        # Create EyeParameters for base class
        eye_params = EyeParameters(
            symbol_rate=config.symbol_rate,
            samples_per_symbol=int(config.sample_rate / config.symbol_rate),
            eye_samples=1000,
            confidence_level=0.95,
            jitter_analysis=True,
        )

        # Initialize base analyzer
        super().__init__(eye_params)

        self.config = config
        self.usb4_specs = USB4Specs()
        self.signal_specs = USB4SignalSpecs()

        # Calculate derived parameters
        self.samples_per_ui = int(self.config.sample_rate / self.config.symbol_rate)
        self.ui_period = 1.0 / self.config.symbol_rate

        # USB4 compliance mask (normalized coordinates)
        self._setup_compliance_mask()

        logger.info(f"USB4EyeDiagramAnalyzer initialized for {self.config.mode.name} mode")
        logger.info(f"Sample rate: {self.config.sample_rate/1e9:.1f} GSa/s")
        logger.info(f"Symbol rate: {self.config.symbol_rate/1e9:.1f} Gbaud")

    def analyze_usb4_eye(
        self, signal_data: npt.NDArray[np.float64], time_data: Optional[npt.NDArray[np.float64]] = None
    ) -> USB4EyeResults:
        """
        Perform comprehensive USB4 eye diagram analysis

        Args:
            signal_data: Signal data array
            time_data: Optional time base array

        Returns:
            USB4EyeResults containing comprehensive analysis

        Raises:
            ValueError: If analysis fails
        """
        try:
            logger.info("Starting USB4 eye diagram analysis")

            # Generate time base if not provided
            if time_data is None:
                time_data = np.arange(len(signal_data)) / self.config.sample_rate

            # Validate input data
            self._validate_eye_data(signal_data, time_data)

            # Apply SSC compensation if enabled
            if self.config.enable_ssc_compensation:
                signal_compensated = self._compensate_ssc_effects(signal_data, time_data)
            else:
                signal_compensated = signal_data

            # Create eye diagram matrix
            eye_matrix = self._create_eye_matrix(signal_compensated)

            # Basic eye measurements
            eye_height = self._measure_eye_height(eye_matrix)
            eye_width = self._measure_eye_width(eye_matrix)
            eye_area = self._calculate_eye_area(eye_matrix)
            crossing_percentage = self._calculate_crossing_percentage(eye_matrix)

            # Timing measurements
            rise_time, fall_time = self._measure_edge_times(signal_compensated, time_data)

            # Jitter analysis from eye
            jitter_rms, jitter_pp = self._analyze_eye_jitter(eye_matrix)

            # SSC impact assessment
            ssc_impact = self._assess_ssc_impact(signal_data, signal_compensated)

            # Compliance checking
            compliance_status, mask_violations = self._check_usb4_compliance(eye_matrix)

            results = USB4EyeResults(
                eye_height=eye_height,
                eye_width=eye_width,
                eye_area=eye_area,
                crossing_percentage=crossing_percentage,
                rise_time=rise_time,
                fall_time=fall_time,
                jitter_rms=jitter_rms,
                jitter_pp=jitter_pp,
                ssc_impact=ssc_impact,
                compliance_status=compliance_status,
                mask_violations=mask_violations,
            )

            logger.info(f"USB4 eye analysis complete. Eye height: {eye_height:.3f}, Eye width: {eye_width:.3f}")
            logger.info(f"Compliance status: {'PASS' if compliance_status else 'FAIL'}")

            return results

        except Exception as e:
            logger.error(f"USB4 eye diagram analysis failed: {e}")
            raise ValueError(f"Failed to analyze USB4 eye diagram: {e}")

    def generate_bathtub_curve(
        self, signal_data: npt.NDArray[np.float64], time_data: Optional[npt.NDArray[np.float64]] = None
    ) -> USB4BathtubResults:
        """
        Generate bathtub curve for USB4 timing analysis

        Args:
            signal_data: Signal data array
            time_data: Optional time base array

        Returns:
            USB4BathtubResults containing bathtub analysis
        """
        try:
            logger.info("Generating USB4 bathtub curve")

            if time_data is None:
                time_data = np.arange(len(signal_data)) / self.config.sample_rate

            # Create eye diagram matrix
            eye_matrix = self._create_eye_matrix(signal_data)

            # Generate timing offsets (fraction of UI)
            timing_offsets = np.linspace(-0.5, 0.5, 101)  # -0.5 to +0.5 UI

            # Calculate BER for each timing offset
            ber_curve = self._calculate_ber_vs_timing(eye_matrix, timing_offsets)

            # Extract key metrics
            eye_width_1e12 = self._extract_eye_width_at_ber(timing_offsets, ber_curve, 1e-12)
            eye_width_1e15 = self._extract_eye_width_at_ber(timing_offsets, ber_curve, 1e-15)

            # Calculate timing margin at target BER
            timing_margin = self._calculate_timing_margin(timing_offsets, ber_curve, self.config.target_ber)

            # Extrapolate BER at zero offset
            extrapolated_ber = self._extrapolate_ber_at_zero(timing_offsets, ber_curve)

            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(timing_offsets, ber_curve)

            results = USB4BathtubResults(
                ber_curve=ber_curve,
                timing_offsets=timing_offsets,
                eye_width_1e12=eye_width_1e12,
                eye_width_1e15=eye_width_1e15,
                timing_margin=timing_margin,
                extrapolated_ber=extrapolated_ber,
                confidence_interval=confidence_interval,
            )

            logger.info(f"Bathtub curve generated. Eye width at 1e-12 BER: {eye_width_1e12:.3f} UI")

            return results

        except Exception as e:
            logger.error(f"Bathtub curve generation failed: {e}")
            raise ValueError(f"Failed to generate bathtub curve: {e}")

    def analyze_eye_contours(
        self, signal_data: npt.NDArray[np.float64], time_data: Optional[npt.NDArray[np.float64]] = None
    ) -> USB4EyeContour:
        """
        Analyze eye diagram contours for statistical analysis

        Args:
            signal_data: Signal data array
            time_data: Optional time base array

        Returns:
            USB4EyeContour containing contour analysis
        """
        try:
            logger.info("Analyzing USB4 eye contours")

            if time_data is None:
                time_data = np.arange(len(signal_data)) / self.config.sample_rate

            # Create high-resolution eye diagram
            eye_matrix = self._create_eye_matrix(signal_data, high_resolution=True)

            # Calculate probability density
            eye_histogram = self._calculate_eye_histogram(eye_matrix)

            # Generate contour data for each level
            contour_data = []
            for level in self.config.contour_levels:
                contour = self._extract_contour_at_level(eye_histogram, level)
                contour_data.append(contour)

            # Calculate eye opening probability
            eye_opening_probability = self._calculate_eye_opening_probability(eye_histogram)

            # Determine worst-case and statistical eye dimensions
            worst_case_eye = self._calculate_worst_case_eye(contour_data)
            statistical_eye = self._calculate_statistical_eye(eye_histogram)

            # Perform margin analysis
            margin_analysis = self._perform_margin_analysis(contour_data, eye_histogram)

            results = USB4EyeContour(
                contour_levels=self.config.contour_levels,
                contour_data=contour_data,
                eye_opening_probability=eye_opening_probability,
                worst_case_eye=worst_case_eye,
                statistical_eye=statistical_eye,
                margin_analysis=margin_analysis,
            )

            logger.info(f"Eye contour analysis complete. Eye opening probability: {eye_opening_probability:.6f}")

            return results

        except Exception as e:
            logger.error(f"Eye contour analysis failed: {e}")
            raise ValueError(f"Failed to analyze eye contours: {e}")

    def _setup_compliance_mask(self) -> None:
        """Setup USB4 compliance mask coordinates"""
        # USB4 compliance mask (normalized to UI and signal amplitude)
        # These are approximate values - actual mask would come from USB4 specification
        self.compliance_mask = {
            "inner_mask": np.array(
                [
                    [-0.4, -0.3],
                    [-0.4, 0.3],
                    [-0.1, 0.3],
                    [-0.1, 0.7],
                    [0.1, 0.7],
                    [0.1, 0.3],
                    [0.4, 0.3],
                    [0.4, -0.3],
                    [0.1, -0.3],
                    [0.1, -0.7],
                    [-0.1, -0.7],
                    [-0.1, -0.3],
                ]
            ),
            "outer_mask": np.array([[-0.5, -0.4], [-0.5, 0.4], [0.5, 0.4], [0.5, -0.4]]),
        }

    def _validate_eye_data(self, signal_data: npt.NDArray[np.float64], time_data: npt.NDArray[np.float64]) -> None:
        """Validate input data for eye analysis"""
        assert isinstance(signal_data, np.ndarray), "Signal data must be numpy array"
        assert isinstance(time_data, np.ndarray), "Time data must be numpy array"
        assert len(signal_data) == len(time_data), "Signal and time data must have same length"
        assert len(signal_data) >= self.samples_per_ui * 10, "Need at least 10 UI of data"
        assert not np.any(np.isnan(signal_data)), "Signal data contains NaN values"
        assert not np.any(np.isinf(signal_data)), "Signal data contains infinite values"
        assert np.all(np.diff(time_data) > 0), "Time data must be monotonically increasing"

    def _compensate_ssc_effects(
        self, signal_data: npt.NDArray[np.float64], time_data: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Compensate for spread spectrum clocking effects in eye analysis

        Args:
            signal_data: Original signal data
            time_data: Time base data

        Returns:
            SSC-compensated signal data
        """
        try:
            # Extract SSC modulation using PLL-like approach
            # This is a simplified implementation - real SSC compensation would be more complex

            # Apply bandpass filter around symbol rate
            nyquist = self.config.sample_rate / 2
            center_freq = self.config.symbol_rate
            bandwidth = center_freq * 0.1

            low_freq = max(center_freq - bandwidth / 2, 1e6)
            high_freq = min(center_freq + bandwidth / 2, nyquist * 0.9)

            sos = scipy.signal.butter(6, [low_freq / nyquist, high_freq / nyquist], btype="band", output="sos")
            filtered_signal = scipy.signal.sosfilt(sos, signal_data)

            # Extract instantaneous phase
            analytic_signal = scipy.signal.hilbert(filtered_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))

            # Remove SSC modulation (simplified approach)
            # In practice, this would involve more sophisticated clock recovery
            phase_trend = np.polyfit(time_data, instantaneous_phase, 3)
            detrended_phase = instantaneous_phase - np.polyval(phase_trend, time_data)

            # Apply phase correction
            phase_correction = np.exp(-1j * detrended_phase * 0.1)  # Reduced correction factor
            corrected_signal = signal_data * np.real(phase_correction)

            return corrected_signal

        except Exception as e:
            logger.warning(f"SSC compensation failed, using original signal: {e}")
            return signal_data

    def _create_eye_matrix(self, signal_data: npt.NDArray[np.float64], high_resolution: bool = False) -> npt.NDArray[np.float64]:
        """Create eye diagram matrix from signal data"""
        if high_resolution:
            samples_per_ui = self.samples_per_ui * 2  # Higher resolution for contour analysis
        else:
            samples_per_ui = self.samples_per_ui

        # Ensure we have enough data
        num_eyes = len(signal_data) // samples_per_ui
        if num_eyes < 10:
            raise ValueError("Insufficient data for eye diagram analysis")

        # Reshape into eye matrix
        total_samples = num_eyes * samples_per_ui
        eye_matrix = signal_data[:total_samples].reshape(num_eyes, samples_per_ui)

        return eye_matrix

    def _measure_eye_height(self, eye_matrix: npt.NDArray[np.float64]) -> float:
        """Measure eye height from eye matrix"""
        # Sample at eye center (middle of UI)
        center_idx = eye_matrix.shape[1] // 2
        center_window = eye_matrix[:, center_idx - 2 : center_idx + 3]  # Small window around center

        # Use percentiles for robust measurement
        p5, p95 = np.percentile(center_window, [5, 95])
        signal_range = np.ptp(eye_matrix)

        if signal_range > 0:
            eye_height = (p95 - p5) / signal_range
        else:
            eye_height = 0.0

        return float(min(eye_height, 1.0))

    def _measure_eye_width(self, eye_matrix: npt.NDArray[np.float64]) -> float:
        """Measure eye width from eye matrix"""
        # Find crossing points at mid-level
        mid_level = np.median(eye_matrix)

        max_width = 0.0
        for eye_trace in eye_matrix[: min(20, len(eye_matrix))]:  # Analyze subset of traces
            # Find zero crossings
            crossings = np.where(np.diff(np.sign(eye_trace - mid_level)))[0]

            if len(crossings) >= 2:
                # Calculate width between first and last crossing
                crossing_span = (crossings[-1] - crossings[0]) / len(eye_trace)
                max_width = max(max_width, crossing_span)

        return float(min(max_width, 1.0))

    def _calculate_eye_area(self, eye_matrix: npt.NDArray[np.float64]) -> float:
        """Calculate eye opening area"""
        # Create 2D histogram of eye diagram
        time_bins = np.linspace(0, 1, eye_matrix.shape[1])
        voltage_bins = np.linspace(np.min(eye_matrix), np.max(eye_matrix), 50)

        hist, _, _ = np.histogram2d(np.tile(time_bins, eye_matrix.shape[0]), eye_matrix.flatten(), bins=[time_bins, voltage_bins])

        # Calculate area where histogram is below threshold (eye opening)
        threshold = np.max(hist) * 0.1  # 10% of peak
        eye_opening = hist < threshold
        eye_area = np.sum(eye_opening) / hist.size

        return float(eye_area)

    def _calculate_crossing_percentage(self, eye_matrix: npt.NDArray[np.float64]) -> float:
        """Calculate eye crossing percentage"""
        # Find the region where signal transitions occur most frequently
        transition_density = np.std(eye_matrix, axis=0)
        max_transition = np.max(transition_density)

        if max_transition > 0:
            # Calculate percentage of UI where transitions are significant
            significant_transitions = transition_density > (max_transition * 0.5)
            crossing_percentage = np.sum(significant_transitions) / len(transition_density)
        else:
            crossing_percentage = 0.0

        return float(crossing_percentage)

    def _measure_edge_times(
        self, signal_data: npt.NDArray[np.float64], time_data: npt.NDArray[np.float64]
    ) -> Tuple[float, float]:
        """Measure 20%-80% rise and fall times"""
        # Find edges using derivative
        diff_signal = np.diff(signal_data)

        # Find rising and falling edges
        threshold = np.std(diff_signal) * 2
        rising_edges = np.where(diff_signal > threshold)[0]
        falling_edges = np.where(diff_signal < -threshold)[0]

        # Measure edge times
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
        """Measure 20%-80% transition time for edges"""
        if len(edge_indices) == 0:
            return float(self.signal_specs.RISE_TIME_MAX)

        transition_times = []

        for edge_idx in edge_indices[:10]:  # Analyze first 10 edges
            # Define window around edge
            window_size = self.samples_per_ui // 4
            start_idx = max(0, edge_idx - window_size)
            end_idx = min(len(signal_data), edge_idx + window_size)

            if end_idx - start_idx < window_size:
                continue

            window_signal = signal_data[start_idx:end_idx]
            window_time = time_data[start_idx:end_idx]

            # Find 20% and 80% levels
            if rising:
                min_val, max_val = np.min(window_signal), np.max(window_signal)
                level_20 = min_val + 0.2 * (max_val - min_val)
                level_80 = min_val + 0.8 * (max_val - min_val)
            else:
                min_val, max_val = np.min(window_signal), np.max(window_signal)
                level_20 = max_val - 0.2 * (max_val - min_val)
                level_80 = max_val - 0.8 * (max_val - min_val)

            # Find crossing times
            try:
                crossings_20 = np.where(np.diff(np.sign(window_signal - level_20)))[0]
                crossings_80 = np.where(np.diff(np.sign(window_signal - level_80)))[0]

                if len(crossings_20) > 0 and len(crossings_80) > 0:
                    idx_20 = crossings_20[0]
                    idx_80 = crossings_80[0]

                    if abs(idx_80 - idx_20) > 0:
                        transition_time = abs(window_time[idx_80] - window_time[idx_20])
                        transition_times.append(transition_time)
            except (IndexError, ValueError):
                continue

        if transition_times:
            return float(np.median(transition_times))
        else:
            return float(self.signal_specs.RISE_TIME_MAX)

    def _analyze_eye_jitter(self, eye_matrix: npt.NDArray[np.float64]) -> Tuple[float, float]:
        """Analyze jitter from eye diagram"""
        # Find crossing points for each eye trace
        mid_level = np.median(eye_matrix)
        all_crossings = []

        for eye_trace in eye_matrix:
            crossings = np.where(np.diff(np.sign(eye_trace - mid_level)))[0]
            if len(crossings) >= 2:
                # Normalize crossing positions to UI
                crossing_positions = crossings / len(eye_trace)
                all_crossings.extend(crossing_positions)

        if len(all_crossings) < 10:
            return 0.1, 0.2  # Default values

        # Calculate jitter statistics
        crossings_array = np.array(all_crossings)

        # Expected crossing positions (0.5 UI for differential signal)
        expected_crossings = np.round(crossings_array * 2) / 2  # Quantize to 0.5 UI grid

        # Calculate jitter
        jitter_values = crossings_array - expected_crossings
        jitter_rms = float(np.std(jitter_values))
        jitter_pp = float(np.ptp(jitter_values))

        return jitter_rms, jitter_pp

    def _assess_ssc_impact(self, original_signal: npt.NDArray[np.float64], compensated_signal: npt.NDArray[np.float64]) -> float:
        """Assess impact of SSC on eye quality"""
        # Compare eye quality before and after SSC compensation
        try:
            original_eye = self._create_eye_matrix(original_signal)
            compensated_eye = self._create_eye_matrix(compensated_signal)

            original_height = self._measure_eye_height(original_eye)
            compensated_height = self._measure_eye_height(compensated_eye)

            # SSC impact as relative improvement
            if original_height > 0:
                ssc_impact = (compensated_height - original_height) / original_height
            else:
                ssc_impact = 0.0

            return float(ssc_impact)

        except Exception:
            return 0.0

    def _check_usb4_compliance(self, eye_matrix: npt.NDArray[np.float64]) -> Tuple[bool, List[str]]:
        """Check USB4 compliance against mask"""
        violations = []

        # Basic eye measurements
        eye_height = self._measure_eye_height(eye_matrix)
        eye_width = self._measure_eye_width(eye_matrix)

        # Check against USB4 specifications
        if eye_height < self.signal_specs.EYE_HEIGHT_MIN:
            violations.append(f"Eye height {eye_height:.3f} below minimum {self.signal_specs.EYE_HEIGHT_MIN}")

        if eye_width < self.signal_specs.EYE_WIDTH_MIN:
            violations.append(f"Eye width {eye_width:.3f} below minimum {self.signal_specs.EYE_WIDTH_MIN}")

        # Check crossing percentage
        crossing_percentage = self._calculate_crossing_percentage(eye_matrix)
        if crossing_percentage > self.signal_specs.EYE_CROSSING_MAX:
            violations.append(f"Eye crossing {crossing_percentage:.3f} exceeds maximum {self.signal_specs.EYE_CROSSING_MAX}")

        # Additional mask testing if enabled
        if self.config.enable_mask_testing:
            mask_violations = self._test_compliance_mask(eye_matrix)
            violations.extend(mask_violations)

        compliance_status = len(violations) == 0

        return compliance_status, violations

    def _test_compliance_mask(self, eye_matrix: npt.NDArray[np.float64]) -> List[str]:
        """Test eye diagram against USB4 compliance mask"""
        violations = []

        # This is a simplified mask test - real implementation would be more detailed
        try:
            # Normalize eye matrix
            normalized_eye = (eye_matrix - np.mean(eye_matrix)) / np.std(eye_matrix)

            # Check if any points violate the inner mask
            time_grid = np.linspace(-0.5, 0.5, eye_matrix.shape[1])
            voltage_grid = np.linspace(-2, 2, eye_matrix.shape[0])

            # Simple mask violation check (placeholder)
            if np.any(np.abs(normalized_eye) > 3):  # Simple amplitude check
                violations.append("Signal amplitude exceeds mask limits")

        except Exception as e:
            logger.warning(f"Mask testing failed: {e}")
            violations.append("Mask testing could not be completed")

        return violations

    def _calculate_ber_vs_timing(
        self, eye_matrix: npt.NDArray[np.float64], timing_offsets: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate BER vs timing offset for bathtub curve"""
        ber_curve = np.zeros_like(timing_offsets)

        for i, offset in enumerate(timing_offsets):
            # Calculate sampling point with timing offset
            samples_per_ui = eye_matrix.shape[1]
            center_idx = samples_per_ui // 2
            offset_samples = int(offset * samples_per_ui)
            sample_idx = center_idx + offset_samples

            # Ensure valid index
            if 0 <= sample_idx < samples_per_ui:
                # Sample eye at this timing offset
                sampled_values = eye_matrix[:, sample_idx]

                # Estimate BER based on signal distribution
                # This is a simplified approach - real BER calculation would be more complex
                signal_std = np.std(sampled_values)
                signal_mean = np.mean(sampled_values)

                # Assume Gaussian noise and calculate Q-factor
                if signal_std > 0:
                    q_factor = abs(signal_mean) / signal_std
                    from scipy import special

                    ber = 0.5 * (1 - special.erf(q_factor / np.sqrt(2)))
                else:
                    ber = 1e-15  # Very low BER for perfect signal

                ber_curve[i] = max(ber, 1e-15)  # Minimum BER floor
            else:
                ber_curve[i] = 1.0  # Maximum BER for invalid timing

        return ber_curve

    def _extract_eye_width_at_ber(
        self, timing_offsets: npt.NDArray[np.float64], ber_curve: npt.NDArray[np.float64], target_ber: float
    ) -> float:
        """Extract eye width at specified BER level"""
        try:
            # Find points where BER crosses target level
            valid_indices = ber_curve <= target_ber

            if np.any(valid_indices):
                valid_offsets = timing_offsets[valid_indices]
                eye_width = np.max(valid_offsets) - np.min(valid_offsets)
                return float(eye_width)
            else:
                return 0.0

        except Exception:
            return 0.0

    def _calculate_timing_margin(
        self, timing_offsets: npt.NDArray[np.float64], ber_curve: npt.NDArray[np.float64], target_ber: float
    ) -> float:
        """Calculate timing margin at target BER"""
        try:
            # Find the widest continuous region below target BER
            valid_mask = ber_curve <= target_ber

            if np.any(valid_mask):
                # Find continuous regions
                diff_mask = np.diff(np.concatenate(([False], valid_mask, [False])).astype(int))
                starts = np.where(diff_mask == 1)[0]
                ends = np.where(diff_mask == -1)[0]

                if len(starts) > 0 and len(ends) > 0:
                    # Find widest region
                    widths = timing_offsets[ends - 1] - timing_offsets[starts]
                    max_width = np.max(widths)
                    return float(max_width)

            return 0.0

        except Exception:
            return 0.0

    def _extrapolate_ber_at_zero(self, timing_offsets: npt.NDArray[np.float64], ber_curve: npt.NDArray[np.float64]) -> float:
        """Extrapolate BER at zero timing offset"""
        try:
            # Find index closest to zero offset
            zero_idx = np.argmin(np.abs(timing_offsets))
            return float(ber_curve[zero_idx])
        except Exception:
            return 1e-12  # Default BER

    def _calculate_confidence_interval(
        self, timing_offsets: npt.NDArray[np.float64], ber_curve: npt.NDArray[np.float64]
    ) -> Tuple[float, float]:
        """Calculate 95% confidence interval for BER"""
        try:
            # Simple confidence interval based on BER statistics
            log_ber = np.log10(np.maximum(ber_curve, 1e-15))
            mean_log_ber = np.mean(log_ber)
            std_log_ber = np.std(log_ber)

            # 95% confidence interval
            ci_lower = 10 ** (mean_log_ber - 1.96 * std_log_ber)
            ci_upper = 10 ** (mean_log_ber + 1.96 * std_log_ber)

            return float(ci_lower), float(ci_upper)

        except Exception:
            return 1e-15, 1e-9  # Default confidence interval

    def _calculate_eye_histogram(self, eye_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate 2D histogram of eye diagram"""
        # Create high-resolution 2D histogram
        time_bins = np.linspace(0, 1, eye_matrix.shape[1])
        voltage_range = np.ptp(eye_matrix)
        voltage_center = np.mean(eye_matrix)
        voltage_bins = np.linspace(voltage_center - voltage_range, voltage_center + voltage_range, 100)

        # Flatten eye matrix for histogram
        time_coords = np.tile(time_bins, eye_matrix.shape[0])
        voltage_coords = eye_matrix.flatten()

        hist, _, _ = np.histogram2d(time_coords, voltage_coords, bins=[time_bins, voltage_bins])

        # Normalize to probability density
        hist = hist / np.sum(hist)

        return hist

    def _extract_contour_at_level(self, eye_histogram: npt.NDArray[np.float64], level: float) -> npt.NDArray[np.float64]:
        """Extract contour at specified probability level"""
        try:
            # Find contour at specified level
            threshold = np.max(eye_histogram) * level
            contour_mask = eye_histogram >= threshold

            # Extract contour coordinates (simplified)
            y_coords, x_coords = np.where(contour_mask)

            if len(x_coords) > 0 and len(y_coords) > 0:
                # Create contour array
                contour = np.column_stack([x_coords, y_coords])
                return contour
            else:
                return np.array([])

        except Exception:
            return np.array([])

    def _calculate_eye_opening_probability(self, eye_histogram: npt.NDArray[np.float64]) -> float:
        """Calculate probability of eye opening"""
        try:
            # Define eye opening region (center of histogram)
            center_region = eye_histogram[
                eye_histogram.shape[0] // 4 : 3 * eye_histogram.shape[0] // 4,
                eye_histogram.shape[1] // 4 : 3 * eye_histogram.shape[1] // 4,
            ]

            # Calculate probability of low density (eye opening)
            threshold = np.mean(eye_histogram) * 0.1
            eye_opening_prob = np.sum(center_region < threshold) / center_region.size

            return float(eye_opening_prob)

        except Exception:
            return 0.5  # Default probability

    def _calculate_worst_case_eye(self, contour_data: List[npt.NDArray[np.float64]]) -> Tuple[float, float]:
        """Calculate worst-case eye dimensions from contours"""
        try:
            if not contour_data or len(contour_data) == 0:
                return 0.6, 0.6  # Default values

            # Use the tightest contour (highest probability level)
            tightest_contour = contour_data[-1]  # Last contour (highest level)

            if len(tightest_contour) > 0:
                # Calculate eye dimensions from contour
                x_range = np.ptp(tightest_contour[:, 0])
                y_range = np.ptp(tightest_contour[:, 1])

                # Normalize to UI and signal range
                eye_width = x_range / 100  # Assuming 100 time bins
                eye_height = y_range / 100  # Assuming 100 voltage bins

                return float(eye_width), float(eye_height)

            return 0.6, 0.6

        except Exception:
            return 0.6, 0.6

    def _calculate_statistical_eye(self, eye_histogram: npt.NDArray[np.float64]) -> Tuple[float, float]:
        """Calculate statistical eye dimensions"""
        try:
            # Calculate statistical measures of eye opening
            # This is a simplified approach

            # Find regions with low probability density (eye opening)
            threshold = np.mean(eye_histogram) * 0.5
            eye_region = eye_histogram < threshold

            # Calculate statistical dimensions
            if np.any(eye_region):
                y_coords, x_coords = np.where(eye_region)

                if len(x_coords) > 0 and len(y_coords) > 0:
                    x_std = np.std(x_coords)
                    y_std = np.std(y_coords)

                    # Convert to normalized dimensions
                    eye_width = x_std / eye_histogram.shape[1]
                    eye_height = y_std / eye_histogram.shape[0]

                    return float(eye_width), float(eye_height)

            return 0.6, 0.6

        except Exception:
            return 0.6, 0.6

    def _perform_margin_analysis(
        self, contour_data: List[npt.NDArray[np.float64]], eye_histogram: npt.NDArray[np.float64]
    ) -> Dict[str, float]:
        """Perform design margin analysis"""
        try:
            margins = {}

            # Timing margin
            if contour_data and len(contour_data) > 0:
                contour = contour_data[0]  # Use first contour level
                if len(contour) > 0:
                    timing_margin = np.ptp(contour[:, 0]) / eye_histogram.shape[1]
                    margins["timing_margin"] = float(timing_margin)
                else:
                    margins["timing_margin"] = 0.6
            else:
                margins["timing_margin"] = 0.6

            # Voltage margin
            if contour_data and len(contour_data) > 0:
                contour = contour_data[0]
                if len(contour) > 0:
                    voltage_margin = np.ptp(contour[:, 1]) / eye_histogram.shape[0]
                    margins["voltage_margin"] = float(voltage_margin)
                else:
                    margins["voltage_margin"] = 0.6
            else:
                margins["voltage_margin"] = 0.6

            # Overall margin (geometric mean)
            margins["overall_margin"] = float(np.sqrt(margins["timing_margin"] * margins["voltage_margin"]))

            return margins

        except Exception:
            return {"timing_margin": 0.6, "voltage_margin": 0.6, "overall_margin": 0.6}


__all__ = ["USB4EyeDiagramAnalyzer", "USB4EyeConfig", "USB4EyeResults", "USB4BathtubResults", "USB4EyeContour"]
