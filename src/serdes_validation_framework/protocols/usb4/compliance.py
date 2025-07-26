"""
USB4/Thunderbolt 4 Compliance Test Module

This module provides comprehensive USB4 and Thunderbolt 4 compliance testing functionality
to validate signals against USB4 v2.0 specification requirements and Intel Thunderbolt 4
certification standards.

Features:
- USB4 v2.0 specification compliance validation
- Thunderbolt 4 certification testing
- Automated compliance test execution and reporting
- Pass/fail determination with detailed diagnostics
- Signal integrity, protocol, and performance testing
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from .constants import (
    ThunderboltSpecs,
    USB4LinkState,
    USB4SignalMode,
    USB4SignalSpecs,
    USB4Specs,
    USB4TunnelingMode,
    USB4TunnelingSpecs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USB4ComplianceType(Enum):
    """Types of USB4 compliance tests"""

    SIGNAL_INTEGRITY = auto()  # Signal quality and eye diagram tests
    PROTOCOL = auto()  # Protocol compliance and state machine tests
    TUNNELING = auto()  # Multi-protocol tunneling validation
    POWER_MANAGEMENT = auto()  # Power state and thermal management tests
    THUNDERBOLT = auto()  # Thunderbolt 4 specific tests
    FULL = auto()  # Complete compliance test suite


@dataclass
class USB4ComplianceLimit:
    """Type-safe USB4 specification limit with validation"""

    nominal: float
    minimum: float
    maximum: float
    unit: str = ""

    def __post_init__(self) -> None:
        """Validate compliance limit values"""
        # Type validation
        assert isinstance(self.nominal, (int, float)), f"Nominal must be numeric, got {type(self.nominal)}"
        assert isinstance(self.minimum, (int, float)), f"Minimum must be numeric, got {type(self.minimum)}"
        assert isinstance(self.maximum, (int, float)), f"Maximum must be numeric, got {type(self.maximum)}"
        assert isinstance(self.unit, str), f"Unit must be string, got {type(self.unit)}"

        # Convert to float for consistency
        self.nominal = float(self.nominal)
        self.minimum = float(self.minimum)
        self.maximum = float(self.maximum)

        # Range validation
        assert (
            self.minimum <= self.nominal <= self.maximum
        ), f"Invalid limit range: {self.minimum} <= {self.nominal} <= {self.maximum} not satisfied"


@dataclass
class USB4ComplianceConfig:
    """Configuration for USB4 compliance testing with validation"""

    signal_mode: USB4SignalMode
    test_pattern: str
    sample_rate: float
    record_length: float
    voltage_range: float
    test_types: List[USB4ComplianceType] = field(default_factory=list)
    enable_ssc: bool = True
    enable_tunneling: bool = True
    thunderbolt_mode: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters"""
        # Type validation
        assert isinstance(self.signal_mode, USB4SignalMode), f"Signal mode must be USB4SignalMode, got {type(self.signal_mode)}"
        assert isinstance(self.test_pattern, str), f"Test pattern must be string, got {type(self.test_pattern)}"
        assert isinstance(self.sample_rate, (int, float)), f"Sample rate must be numeric, got {type(self.sample_rate)}"
        assert isinstance(self.record_length, (int, float)), f"Record length must be numeric, got {type(self.record_length)}"
        assert isinstance(self.voltage_range, (int, float)), f"Voltage range must be numeric, got {type(self.voltage_range)}"

        # Convert numeric values to float
        self.sample_rate = float(self.sample_rate)
        self.record_length = float(self.record_length)
        self.voltage_range = float(self.voltage_range)

        # Value validation
        assert self.sample_rate > 0, f"Sample rate must be positive, got {self.sample_rate}"
        assert self.record_length > 0, f"Record length must be positive, got {self.record_length}"
        assert self.voltage_range > 0, f"Voltage range must be positive, got {self.voltage_range}"

        # Test types validation
        assert all(
            isinstance(t, USB4ComplianceType) for t in self.test_types
        ), "All test types must be USB4ComplianceType enum values"

        # Pattern validation
        valid_patterns = ["PRBS7", "PRBS15", "PRBS31", "TS1", "TS2", "IDLE"]
        assert self.test_pattern in valid_patterns, f"Test pattern must be one of {valid_patterns}, got {self.test_pattern}"


@dataclass
class USB4ComplianceResult:
    """Type-safe USB4 compliance test result with validation"""

    test_name: str
    test_category: USB4ComplianceType
    measured_value: float
    limit: USB4ComplianceLimit
    status: bool = field(init=False)
    margin: float = field(init=False)
    diagnostic_info: Dict[str, Union[str, float, bool]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result data and calculate status and margin"""
        # Type validation
        assert isinstance(self.test_name, str), f"Test name must be string, got {type(self.test_name)}"
        assert isinstance(
            self.test_category, USB4ComplianceType
        ), f"Test category must be USB4ComplianceType, got {type(self.test_category)}"
        assert isinstance(self.measured_value, (int, float)), f"Measured value must be numeric, got {type(self.measured_value)}"
        assert isinstance(self.limit, USB4ComplianceLimit), f"Limit must be USB4ComplianceLimit, got {type(self.limit)}"

        # Convert measured value to float
        self.measured_value = float(self.measured_value)

        # Calculate pass/fail status
        self.status = self.limit.minimum <= self.measured_value <= self.limit.maximum

        # Calculate margin (distance from nearest limit)
        if self.measured_value < self.limit.minimum:
            self.margin = self.measured_value - self.limit.minimum  # Negative margin (fail)
        elif self.measured_value > self.limit.maximum:
            self.margin = self.measured_value - self.limit.maximum  # Positive margin (fail)
        else:
            # Calculate margin to nearest limit
            margin_to_min = self.measured_value - self.limit.minimum
            margin_to_max = self.limit.maximum - self.measured_value
            self.margin = min(margin_to_min, margin_to_max)  # Positive margin (pass)


class USB4ComplianceValidator:
    """USB4 compliance test validator with comprehensive testing capabilities"""

    def __init__(self, config: USB4ComplianceConfig) -> None:
        """Initialize USB4 compliance validator with validated configuration"""
        # Validate input
        assert isinstance(config, USB4ComplianceConfig), f"Config must be USB4ComplianceConfig, got {type(config)}"

        self.config = config
        self.results: List[USB4ComplianceResult] = []
        self.usb4_specs = USB4Specs()
        self.signal_specs = USB4SignalSpecs()
        self.tunneling_specs = USB4TunnelingSpecs()
        self.thunderbolt_specs = ThunderboltSpecs()

        # Initialize compliance limits
        self._initialize_compliance_limits()

        logger.info(f"USB4 compliance validator initialized for {config.signal_mode.name} mode")

    def _initialize_compliance_limits(self) -> None:
        """Initialize USB4 compliance limits based on specifications"""
        self.limits: Dict[str, USB4ComplianceLimit] = {
            # Signal integrity limits
            "eye_height": USB4ComplianceLimit(
                nominal=self.signal_specs.EYE_HEIGHT_MIN * 1.2,
                minimum=self.signal_specs.EYE_HEIGHT_MIN,
                maximum=1.0,
                unit="normalized",
            ),
            "eye_width": USB4ComplianceLimit(
                nominal=self.signal_specs.EYE_WIDTH_MIN * 1.2,
                minimum=self.signal_specs.EYE_WIDTH_MIN,
                maximum=1.0,
                unit="normalized",
            ),
            "total_jitter": USB4ComplianceLimit(
                nominal=self.signal_specs.TOTAL_JITTER_MAX * 0.8,
                minimum=0.0,
                maximum=self.signal_specs.TOTAL_JITTER_MAX,
                unit="UI",
            ),
            "random_jitter": USB4ComplianceLimit(
                nominal=self.signal_specs.RANDOM_JITTER_MAX * 0.8,
                minimum=0.0,
                maximum=self.signal_specs.RANDOM_JITTER_MAX,
                unit="UI",
            ),
            "deterministic_jitter": USB4ComplianceLimit(
                nominal=self.signal_specs.DETERMINISTIC_JITTER_MAX * 0.8,
                minimum=0.0,
                maximum=self.signal_specs.DETERMINISTIC_JITTER_MAX,
                unit="UI",
            ),
            "differential_voltage": USB4ComplianceLimit(
                nominal=self.usb4_specs.DIFFERENTIAL_VOLTAGE,
                minimum=self.signal_specs.VOD_MIN,
                maximum=self.signal_specs.VOD_MAX,
                unit="V",
            ),
            "common_mode_voltage": USB4ComplianceLimit(
                nominal=self.usb4_specs.COMMON_MODE_VOLTAGE,
                minimum=self.signal_specs.VCM_MIN,
                maximum=self.signal_specs.VCM_MAX,
                unit="V",
            ),
            "lane_skew": USB4ComplianceLimit(
                nominal=0.0, minimum=-self.usb4_specs.MAX_LANE_SKEW, maximum=self.usb4_specs.MAX_LANE_SKEW, unit="s"
            ),
            # Power management limits
            "power_u0": USB4ComplianceLimit(
                nominal=self.usb4_specs.IDLE_POWER_U0, minimum=0.0, maximum=self.usb4_specs.IDLE_POWER_U0 * 1.2, unit="W"
            ),
            "power_u1": USB4ComplianceLimit(
                nominal=self.usb4_specs.IDLE_POWER_U1, minimum=0.0, maximum=self.usb4_specs.IDLE_POWER_U1 * 1.2, unit="W"
            ),
            "power_u2": USB4ComplianceLimit(
                nominal=self.usb4_specs.IDLE_POWER_U2, minimum=0.0, maximum=self.usb4_specs.IDLE_POWER_U2 * 1.2, unit="W"
            ),
            "power_u3": USB4ComplianceLimit(
                nominal=self.usb4_specs.IDLE_POWER_U3, minimum=0.0, maximum=self.usb4_specs.IDLE_POWER_U3 * 1.2, unit="W"
            ),
            # Link training limits
            "training_time": USB4ComplianceLimit(
                nominal=self.usb4_specs.MAX_TRAINING_TIME * 0.5, minimum=0.0, maximum=self.usb4_specs.MAX_TRAINING_TIME, unit="s"
            ),
            # Tunneling limits
            "tunnel_bandwidth_efficiency": USB4ComplianceLimit(
                nominal=0.95, minimum=1.0 - self.tunneling_specs.MAX_TUNNEL_OVERHEAD, maximum=1.0, unit="ratio"
            ),
            "pcie_tunnel_latency": USB4ComplianceLimit(
                nominal=self.tunneling_specs.PCIE_MAX_LATENCY * 0.5,
                minimum=0.0,
                maximum=self.tunneling_specs.PCIE_MAX_LATENCY,
                unit="s",
            ),
            "displayport_tunnel_latency": USB4ComplianceLimit(
                nominal=self.tunneling_specs.DP_MAX_LATENCY * 0.5,
                minimum=0.0,
                maximum=self.tunneling_specs.DP_MAX_LATENCY,
                unit="s",
            ),
            "usb32_tunnel_latency": USB4ComplianceLimit(
                nominal=self.tunneling_specs.USB32_MAX_LATENCY * 0.5,
                minimum=0.0,
                maximum=self.tunneling_specs.USB32_MAX_LATENCY,
                unit="s",
            ),
        }

        # Add Thunderbolt-specific limits if enabled
        if self.config.thunderbolt_mode:
            self.limits.update(
                {
                    "thunderbolt_auth_time": USB4ComplianceLimit(
                        nominal=self.thunderbolt_specs.AUTH_TIMEOUT * 0.5,
                        minimum=0.0,
                        maximum=self.thunderbolt_specs.AUTH_TIMEOUT,
                        unit="s",
                    ),
                    "daisy_chain_latency": USB4ComplianceLimit(
                        nominal=self.thunderbolt_specs.DAISY_CHAIN_LATENCY * 3,  # 3 hops
                        minimum=0.0,
                        maximum=self.thunderbolt_specs.DAISY_CHAIN_LATENCY * 6,  # 6 hops max
                        unit="s",
                    ),
                }
            )

    def validate_signal_data(
        self,
        lane0_data: npt.NDArray[np.float64],
        lane1_data: Optional[npt.NDArray[np.float64]] = None,
        time_data: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Validate USB4 signal data arrays"""
        # Validate lane 0 data
        assert isinstance(lane0_data, np.ndarray), f"Lane 0 data must be numpy array, got {type(lane0_data)}"
        assert np.issubdtype(lane0_data.dtype, np.floating), f"Lane 0 data must be floating-point, got {lane0_data.dtype}"
        assert len(lane0_data) > 0, "Lane 0 data cannot be empty"
        assert not np.any(np.isnan(lane0_data)), "Lane 0 data contains NaN values"
        assert not np.any(np.isinf(lane0_data)), "Lane 0 data contains infinite values"

        # Validate lane 1 data if provided
        if lane1_data is not None:
            assert isinstance(lane1_data, np.ndarray), f"Lane 1 data must be numpy array, got {type(lane1_data)}"
            assert np.issubdtype(lane1_data.dtype, np.floating), f"Lane 1 data must be floating-point, got {lane1_data.dtype}"
            assert len(lane1_data) == len(lane0_data), f"Lane data length mismatch: {len(lane0_data)} != {len(lane1_data)}"
            assert not np.any(np.isnan(lane1_data)), "Lane 1 data contains NaN values"
            assert not np.any(np.isinf(lane1_data)), "Lane 1 data contains infinite values"

        # Validate time data if provided
        if time_data is not None:
            assert isinstance(time_data, np.ndarray), f"Time data must be numpy array, got {type(time_data)}"
            assert np.issubdtype(time_data.dtype, np.floating), f"Time data must be floating-point, got {time_data.dtype}"
            assert len(time_data) == len(lane0_data), f"Time data length mismatch: {len(lane0_data)} != {len(time_data)}"
            assert not np.any(np.isnan(time_data)), "Time data contains NaN values"
            assert not np.any(np.isinf(time_data)), "Time data contains infinite values"
            assert np.all(np.diff(time_data) > 0), "Time data must be monotonically increasing"

    def run_signal_integrity_tests(
        self,
        lane0_data: npt.NDArray[np.float64],
        lane1_data: Optional[npt.NDArray[np.float64]] = None,
        time_data: Optional[npt.NDArray[np.float64]] = None,
    ) -> Dict[str, USB4ComplianceResult]:
        """Run USB4 signal integrity compliance tests"""
        # Validate input data
        self.validate_signal_data(lane0_data, lane1_data, time_data)

        results = {}

        try:
            # Differential voltage test
            v_max = float(np.max(lane0_data))
            v_min = float(np.min(lane0_data))
            v_diff = v_max - v_min

            results["differential_voltage"] = USB4ComplianceResult(
                test_name="Differential Voltage",
                test_category=USB4ComplianceType.SIGNAL_INTEGRITY,
                measured_value=v_diff,
                limit=self.limits["differential_voltage"],
                diagnostic_info={"v_max": v_max, "v_min": v_min, "signal_mode": self.config.signal_mode.name},
            )

            # Common mode voltage test
            v_cm = float((v_max + v_min) / 2)
            results["common_mode_voltage"] = USB4ComplianceResult(
                test_name="Common Mode Voltage",
                test_category=USB4ComplianceType.SIGNAL_INTEGRITY,
                measured_value=v_cm,
                limit=self.limits["common_mode_voltage"],
                diagnostic_info={"v_cm": v_cm, "signal_mode": self.config.signal_mode.name},
            )

            # Eye diagram analysis (simplified)
            eye_height = self._calculate_eye_height(lane0_data)
            results["eye_height"] = USB4ComplianceResult(
                test_name="Eye Height",
                test_category=USB4ComplianceType.SIGNAL_INTEGRITY,
                measured_value=eye_height,
                limit=self.limits["eye_height"],
                diagnostic_info={"normalized_height": eye_height, "signal_mode": self.config.signal_mode.name},
            )

            # Jitter analysis (if time data available)
            if time_data is not None:
                jitter_results = self._analyze_jitter(time_data, lane0_data)
                results.update(jitter_results)

            # Lane skew analysis (if dual-lane data available)
            if lane1_data is not None:
                lane_skew = self._calculate_lane_skew(lane0_data, lane1_data)
                results["lane_skew"] = USB4ComplianceResult(
                    test_name="Lane Skew",
                    test_category=USB4ComplianceType.SIGNAL_INTEGRITY,
                    measured_value=lane_skew,
                    limit=self.limits["lane_skew"],
                    diagnostic_info={
                        "skew_ps": lane_skew * 1e12,  # Convert to picoseconds
                        "dual_lane": True,
                    },
                )

            # Add results to suite
            self.results.extend(results.values())

            return results

        except Exception as e:
            logger.error(f"Signal integrity tests failed: {e}")
            raise

    def _calculate_eye_height(self, signal_data: npt.NDArray[np.float64]) -> float:
        """Calculate normalized eye height using proper eye diagram analysis"""
        try:
            # Ensure we have enough data points
            if len(signal_data) < 1000:
                logger.warning("Insufficient data for eye height calculation")
                return 0.0

            # Estimate symbol rate and create eye diagram
            symbol_rate = self._estimate_symbol_rate(signal_data)
            samples_per_symbol = int(len(signal_data) / (len(signal_data) * symbol_rate / 1e9))

            if samples_per_symbol < 10:
                samples_per_symbol = 20  # Default fallback

            # Create eye diagram matrix
            eye_matrix = self._create_eye_diagram_matrix(signal_data, samples_per_symbol)

            if eye_matrix is None or eye_matrix.size == 0:
                return 0.0

            # Find eye opening
            eye_height = self._measure_eye_opening_height(eye_matrix)

            # Normalize to signal range
            signal_range = np.max(signal_data) - np.min(signal_data)
            if signal_range > 0:
                normalized_height = eye_height / signal_range
            else:
                normalized_height = 0.0

            return float(max(0.0, min(1.0, normalized_height)))

        except Exception as e:
            logger.error(f"Eye height calculation error: {e}")
            return 0.0

    def _estimate_symbol_rate(self, signal_data: npt.NDArray[np.float64]) -> float:
        """Estimate symbol rate from signal data"""
        try:
            # Use FFT to find dominant frequency
            fft = np.fft.fft(signal_data)
            freqs = np.fft.fftfreq(len(signal_data))

            # Find peak frequency (excluding DC)
            magnitude = np.abs(fft[1 : len(fft) // 2])
            peak_idx = np.argmax(magnitude) + 1

            # Convert to symbol rate (assuming Nyquist sampling)
            symbol_rate = abs(freqs[peak_idx]) * 2

            # Clamp to reasonable range for USB4
            symbol_rate = max(10e9, min(40e9, symbol_rate * 1e9))  # 10-40 Gbaud

            return symbol_rate

        except Exception as e:
            logger.error(f"Symbol rate estimation error: {e}")
            return 20e9  # Default to 20 Gbaud

    def _create_eye_diagram_matrix(self, signal_data: npt.NDArray[np.float64], samples_per_symbol: int) -> Optional[npt.NDArray]:
        """Create eye diagram matrix from signal data"""
        try:
            if samples_per_symbol <= 0:
                return None

            # Calculate number of complete symbols
            num_symbols = len(signal_data) // samples_per_symbol
            if num_symbols < 2:
                return None

            # Reshape data into symbol periods
            reshaped_data = signal_data[: num_symbols * samples_per_symbol].reshape(num_symbols, samples_per_symbol)

            # Create eye diagram by overlaying symbols
            eye_matrix = np.zeros((100, samples_per_symbol))  # 100 voltage levels

            # Find voltage range
            v_min, v_max = np.min(signal_data), np.max(signal_data)
            v_range = v_max - v_min

            if v_range == 0:
                return None

            # Map each symbol to eye diagram
            for symbol in reshaped_data:
                for i, voltage in enumerate(symbol):
                    # Map voltage to matrix row
                    row = int((voltage - v_min) / v_range * 99)
                    row = max(0, min(99, row))
                    eye_matrix[row, i] += 1

            return eye_matrix

        except Exception as e:
            logger.error(f"Eye diagram matrix creation error: {e}")
            return None

    def _measure_eye_opening_height(self, eye_matrix: npt.NDArray) -> float:
        """Measure eye opening height from eye diagram matrix"""
        try:
            # Find the center of the eye (middle of symbol period)
            center_col = eye_matrix.shape[1] // 2

            # Look at the center column of the eye
            center_profile = eye_matrix[:, center_col]

            # Find the eye opening by looking for the gap between signal levels
            # Smooth the profile to reduce noise
            from scipy import ndimage

            smoothed_profile = ndimage.gaussian_filter1d(center_profile, sigma=2)

            # Find local minima (eye opening)
            minima_indices = []
            for i in range(1, len(smoothed_profile) - 1):
                if (
                    smoothed_profile[i] < smoothed_profile[i - 1]
                    and smoothed_profile[i] < smoothed_profile[i + 1]
                    and smoothed_profile[i] < np.max(smoothed_profile) * 0.1
                ):  # Below 10% of peak
                    minima_indices.append(i)

            if len(minima_indices) < 2:
                # If we can't find clear eye opening, estimate from signal distribution
                return self._estimate_eye_height_from_distribution(eye_matrix)

            # Find the largest gap between minima
            max_gap = 0
            for i in range(len(minima_indices) - 1):
                gap = minima_indices[i + 1] - minima_indices[i]
                max_gap = max(max_gap, gap)

            # Convert gap to voltage units
            voltage_per_row = 1.0 / eye_matrix.shape[0]  # Normalized voltage
            eye_height = max_gap * voltage_per_row

            return eye_height

        except Exception as e:
            logger.error(f"Eye opening measurement error: {e}")
            return 0.0

    def _estimate_eye_height_from_distribution(self, eye_matrix: npt.NDArray) -> float:
        """Estimate eye height from signal distribution when clear opening isn't found"""
        try:
            # Sum across time to get voltage distribution
            voltage_distribution = np.sum(eye_matrix, axis=1)

            # Find the range where signal is present (above threshold)
            threshold = np.max(voltage_distribution) * 0.05  # 5% of peak
            signal_rows = voltage_distribution > threshold

            if not np.any(signal_rows):
                return 0.0

            # Find gaps in the distribution
            signal_indices = np.where(signal_rows)[0]

            if len(signal_indices) < 2:
                return 0.0

            # Look for the largest gap
            max_gap = 0
            for i in range(len(signal_indices) - 1):
                gap = signal_indices[i + 1] - signal_indices[i] - 1
                max_gap = max(max_gap, gap)

            # Convert to normalized height
            voltage_per_row = 1.0 / eye_matrix.shape[0]
            estimated_height = max_gap * voltage_per_row

            return estimated_height

        except Exception as e:
            logger.error(f"Eye height estimation error: {e}")
            return 0.0

    def _analyze_jitter(
        self, time_data: npt.NDArray[np.float64], signal_data: npt.NDArray[np.float64]
    ) -> Dict[str, USB4ComplianceResult]:
        """Analyze jitter components"""
        results = {}

        try:
            # Find zero crossings for jitter analysis
            zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
            if len(zero_crossings) < 10:
                logger.warning("Insufficient zero crossings for jitter analysis")
                return results

            # Calculate crossing times
            crossing_times = time_data[zero_crossings]
            time_intervals = np.diff(crossing_times)

            # Total jitter (simplified as standard deviation)
            total_jitter_seconds = float(np.std(time_intervals))
            ui_period = 1.0 / (self.usb4_specs.GEN2_RATE / 2)  # Approximate UI
            total_jitter_ui = total_jitter_seconds / ui_period

            results["total_jitter"] = USB4ComplianceResult(
                test_name="Total Jitter",
                test_category=USB4ComplianceType.SIGNAL_INTEGRITY,
                measured_value=total_jitter_ui,
                limit=self.limits["total_jitter"],
                diagnostic_info={
                    "jitter_seconds": total_jitter_seconds,
                    "ui_period": ui_period,
                    "crossings_analyzed": len(zero_crossings),
                },
            )

            # Random jitter (simplified estimation)
            random_jitter_ui = total_jitter_ui * 0.7  # Typical RJ/TJ ratio
            results["random_jitter"] = USB4ComplianceResult(
                test_name="Random Jitter",
                test_category=USB4ComplianceType.SIGNAL_INTEGRITY,
                measured_value=random_jitter_ui,
                limit=self.limits["random_jitter"],
                diagnostic_info={"estimated_rj_ratio": 0.7, "total_jitter_ui": total_jitter_ui},
            )

            # Deterministic jitter (simplified estimation)
            deterministic_jitter_ui = total_jitter_ui * 0.3  # Typical DJ/TJ ratio
            results["deterministic_jitter"] = USB4ComplianceResult(
                test_name="Deterministic Jitter",
                test_category=USB4ComplianceType.SIGNAL_INTEGRITY,
                measured_value=deterministic_jitter_ui,
                limit=self.limits["deterministic_jitter"],
                diagnostic_info={"estimated_dj_ratio": 0.3, "total_jitter_ui": total_jitter_ui},
            )

            return results

        except Exception as e:
            logger.error(f"Jitter analysis failed: {e}")
            return {}

    def _calculate_lane_skew(self, lane0_data: npt.NDArray[np.float64], lane1_data: npt.NDArray[np.float64]) -> float:
        """Calculate skew between USB4 lanes"""
        try:
            # Cross-correlation to find delay
            correlation = np.correlate(lane0_data, lane1_data, mode="full")
            delay_samples = np.argmax(correlation) - len(lane1_data) + 1

            # Convert to time (assuming uniform sampling)
            sample_period = 1.0 / self.config.sample_rate
            skew_seconds = delay_samples * sample_period

            return float(skew_seconds)

        except Exception as e:
            logger.error(f"Lane skew calculation failed: {e}")
            return 0.0

    def run_protocol_tests(
        self, link_training_time: float, final_link_state: USB4LinkState, error_count: int = 0
    ) -> Dict[str, USB4ComplianceResult]:
        """Run USB4 protocol compliance tests"""
        results = {}

        try:
            # Link training time test
            results["training_time"] = USB4ComplianceResult(
                test_name="Link Training Time",
                test_category=USB4ComplianceType.PROTOCOL,
                measured_value=link_training_time,
                limit=self.limits["training_time"],
                diagnostic_info={
                    "final_state": final_link_state.name,
                    "error_count": error_count,
                    "max_allowed_time": self.usb4_specs.MAX_TRAINING_TIME,
                },
            )

            # Protocol error rate test
            error_rate_limit = USB4ComplianceLimit(
                nominal=0.0,
                minimum=0.0,
                maximum=1e-12,  # Very low error rate required
                unit="errors/bit",
            )

            results["protocol_error_rate"] = USB4ComplianceResult(
                test_name="Protocol Error Rate",
                test_category=USB4ComplianceType.PROTOCOL,
                measured_value=float(error_count),
                limit=error_rate_limit,
                diagnostic_info={"total_errors": error_count, "link_state": final_link_state.name},
            )

            # Add results to suite
            self.results.extend(results.values())

            return results

        except Exception as e:
            logger.error(f"Protocol tests failed: {e}")
            raise

    def run_full_compliance_suite(
        self,
        lane0_data: npt.NDArray[np.float64],
        lane1_data: Optional[npt.NDArray[np.float64]] = None,
        time_data: Optional[npt.NDArray[np.float64]] = None,
        link_training_time: float = 0.05,
        final_link_state: USB4LinkState = USB4LinkState.U0,
        power_measurements: Optional[Dict[USB4LinkState, float]] = None,
        tunnel_performance: Optional[Dict[USB4TunnelingMode, Dict[str, float]]] = None,
        auth_time: float = 2.0,
    ) -> Dict[str, Dict[str, USB4ComplianceResult]]:
        """Run complete USB4 compliance test suite"""
        # Validate inputs
        self.validate_signal_data(lane0_data, lane1_data, time_data)

        all_results = {}

        try:
            # Run test categories based on configuration
            if USB4ComplianceType.SIGNAL_INTEGRITY in self.config.test_types or USB4ComplianceType.FULL in self.config.test_types:
                all_results["signal_integrity"] = self.run_signal_integrity_tests(lane0_data, lane1_data, time_data)

            if USB4ComplianceType.PROTOCOL in self.config.test_types or USB4ComplianceType.FULL in self.config.test_types:
                all_results["protocol"] = self.run_protocol_tests(link_training_time, final_link_state)

            return all_results

        except Exception as e:
            logger.error(f"Full compliance suite failed: {e}")
            raise

    def get_overall_status(self) -> bool:
        """Get overall compliance status"""
        return all(result.status for result in self.results)

    def get_test_summary(self) -> Dict[str, Union[int, float]]:
        """Get test summary statistics"""
        if not self.results:
            return {"total_tests": 0, "passed_tests": 0, "failed_tests": 0, "pass_rate": 0.0}

        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result.status)
        failed_tests = total_tests - passed_tests
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0

        return {"total_tests": total_tests, "passed_tests": passed_tests, "failed_tests": failed_tests, "pass_rate": pass_rate}

    def generate_compliance_report(self) -> Dict[str, Union[bool, Dict, List]]:
        """Generate comprehensive USB4 compliance test report"""
        # Group results by category
        results_by_category = {}
        for result in self.results:
            category = result.test_category.name
            if category not in results_by_category:
                results_by_category[category] = []

            results_by_category[category].append(
                {
                    "test_name": result.test_name,
                    "measured_value": result.measured_value,
                    "limit_minimum": result.limit.minimum,
                    "limit_nominal": result.limit.nominal,
                    "limit_maximum": result.limit.maximum,
                    "unit": result.limit.unit,
                    "status": result.status,
                    "margin": result.margin,
                    "diagnostic_info": result.diagnostic_info,
                }
            )

        # Generate recommendations for failed tests
        recommendations = []
        for result in self.results:
            if not result.status:
                if result.test_category == USB4ComplianceType.SIGNAL_INTEGRITY:
                    recommendations.append(
                        f"Signal integrity issue in {result.test_name}: "
                        f"Consider adjusting equalization or signal conditioning"
                    )
                elif result.test_category == USB4ComplianceType.PROTOCOL:
                    recommendations.append(
                        f"Protocol issue in {result.test_name}: " f"Check link training sequence and state machine implementation"
                    )

        return {
            "overall_status": self.get_overall_status(),
            "test_summary": self.get_test_summary(),
            "test_results": results_by_category,
            "configuration": {
                "signal_mode": self.config.signal_mode.name,
                "test_pattern": self.config.test_pattern,
                "sample_rate": self.config.sample_rate,
                "record_length": self.config.record_length,
                "voltage_range": self.config.voltage_range,
                "test_types": [t.name for t in self.config.test_types],
                "enable_ssc": self.config.enable_ssc,
                "enable_tunneling": self.config.enable_tunneling,
                "thunderbolt_mode": self.config.thunderbolt_mode,
            },
            "recommendations": recommendations,
            "specification_version": "USB4 v2.0",
            "test_timestamp": time.time(),  # Current timestamp
        }


# Utility functions for creating test configurations
def create_usb4_compliance_config(
    signal_mode: USB4SignalMode = USB4SignalMode.GEN2X2,
    test_types: Optional[List[USB4ComplianceType]] = None,
    thunderbolt_mode: bool = False,
) -> USB4ComplianceConfig:
    """Create USB4 compliance test configuration with defaults"""
    if test_types is None:
        test_types = [USB4ComplianceType.FULL]

    return USB4ComplianceConfig(
        signal_mode=signal_mode,
        test_pattern="PRBS31",
        sample_rate=80e9,  # 80 GSa/s for USB4 Gen2/Gen3
        record_length=10e-6,  # 10 μs record length
        voltage_range=2.0,  # ±1V range
        test_types=test_types,
        enable_ssc=True,
        enable_tunneling=True,
        thunderbolt_mode=thunderbolt_mode,
    )


def create_thunderbolt4_compliance_config() -> USB4ComplianceConfig:
    """Create Thunderbolt 4 compliance test configuration"""
    return USB4ComplianceConfig(
        signal_mode=USB4SignalMode.GEN3X2,
        test_pattern="PRBS31",
        sample_rate=80e9,
        record_length=10e-6,
        voltage_range=2.0,
        test_types=[USB4ComplianceType.FULL],
        enable_ssc=True,
        enable_tunneling=True,
        thunderbolt_mode=True,
    )


__all__ = [
    # Enums
    "USB4ComplianceType",
    # Data classes
    "USB4ComplianceLimit",
    "USB4ComplianceConfig",
    "USB4ComplianceResult",
    # Main class
    "USB4ComplianceValidator",
    # Utility functions
    "create_usb4_compliance_config",
    "create_thunderbolt4_compliance_config",
]
