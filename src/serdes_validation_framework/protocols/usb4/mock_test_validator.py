"""
USB4 Mock Test Validator Module

This module provides comprehensive validation for USB4 mock testing to ensure
that simulated results are realistic and match expected USB4 behavior patterns.

Features:
- Mock data validation
- Signal quality verification
- Performance bounds checking
- Compliance validation
- Test result verification
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt

from ..instrument_control.usb4_mock_data_generator import USB4MockConfig, USB4MockSignalData
from .compliance import USB4ComplianceConfig, USB4ComplianceType, USB4ComplianceValidator
from .constants import ThunderboltSpecs, USB4SignalSpecs, USB4Specs, USB4TunnelingMode
from .mock_tunneling_simulator import USB4TunnelConfig, USB4TunnelingSimulator
from .thunderbolt.mock_device_simulator import ThunderboltMockDevice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USB4MockValidationLevel(Enum):
    """USB4 mock validation levels"""

    BASIC = auto()
    STANDARD = auto()
    COMPREHENSIVE = auto()
    CERTIFICATION = auto()


class USB4MockTestResult(Enum):
    """USB4 mock test results"""

    PASS = auto()
    FAIL = auto()
    WARNING = auto()
    INVALID = auto()


@dataclass
class USB4MockValidationConfig:
    """USB4 mock validation configuration"""

    validation_level: USB4MockValidationLevel
    signal_tolerance: float = 0.1  # 10% tolerance
    timing_tolerance: float = 0.05  # 5% tolerance
    power_tolerance: float = 0.15  # 15% tolerance
    enable_bounds_checking: bool = True
    enable_compliance_validation: bool = True
    enable_performance_validation: bool = True

    def __post_init__(self) -> None:
        """Validate configuration"""
        assert isinstance(
            self.validation_level, USB4MockValidationLevel
        ), f"Validation level must be USB4MockValidationLevel, got {type(self.validation_level)}"
        assert isinstance(self.signal_tolerance, float), f"Signal tolerance must be float, got {type(self.signal_tolerance)}"
        assert isinstance(self.timing_tolerance, float), f"Timing tolerance must be float, got {type(self.timing_tolerance)}"
        assert isinstance(self.power_tolerance, float), f"Power tolerance must be float, got {type(self.power_tolerance)}"

        assert 0 < self.signal_tolerance < 1, f"Signal tolerance must be between 0 and 1, got {self.signal_tolerance}"
        assert 0 < self.timing_tolerance < 1, f"Timing tolerance must be between 0 and 1, got {self.timing_tolerance}"
        assert 0 < self.power_tolerance < 1, f"Power tolerance must be between 0 and 1, got {self.power_tolerance}"


@dataclass
class USB4MockValidationResult:
    """USB4 mock validation result"""

    test_name: str
    validation_level: USB4MockValidationLevel
    overall_result: USB4MockTestResult
    signal_validation: Dict[str, Union[bool, float, str]]
    timing_validation: Dict[str, Union[bool, float, str]]
    power_validation: Dict[str, Union[bool, float, str]]
    compliance_validation: Dict[str, Union[bool, float, str]]
    performance_validation: Dict[str, Union[bool, float, str]]
    validation_time: float
    recommendations: List[str]

    def __post_init__(self) -> None:
        """Validate result"""
        assert isinstance(self.test_name, str), f"Test name must be string, got {type(self.test_name)}"
        assert isinstance(
            self.validation_level, USB4MockValidationLevel
        ), f"Validation level must be USB4MockValidationLevel, got {type(self.validation_level)}"
        assert isinstance(
            self.overall_result, USB4MockTestResult
        ), f"Overall result must be USB4MockTestResult, got {type(self.overall_result)}"
        assert isinstance(self.validation_time, float), f"Validation time must be float, got {type(self.validation_time)}"
        assert isinstance(self.recommendations, list), f"Recommendations must be list, got {type(self.recommendations)}"

        assert self.validation_time >= 0, f"Validation time must be non-negative, got {self.validation_time}"


class USB4MockTestValidator:
    """Comprehensive USB4 mock test validator"""

    def __init__(self, config: USB4MockValidationConfig) -> None:
        """
        Initialize USB4 mock test validator

        Args:
            config: Validation configuration

        Raises:
            ValueError: If initialization fails
        """
        assert isinstance(config, USB4MockValidationConfig), f"Config must be USB4MockValidationConfig, got {type(config)}"

        self.config = config
        self.usb4_specs = USB4Specs()
        self.signal_specs = USB4SignalSpecs()
        self.thunderbolt_specs = ThunderboltSpecs()

        # Validation state
        self.validation_history: List[USB4MockValidationResult] = []

        logger.info(f"USB4 mock test validator initialized with {config.validation_level.name} level")

    def validate_mock_signal_data(self, signal_data: USB4MockSignalData, mock_config: USB4MockConfig) -> USB4MockValidationResult:
        """
        Validate USB4 mock signal data

        Args:
            signal_data: Mock signal data to validate
            mock_config: Mock configuration used to generate data

        Returns:
            Validation result

        Raises:
            ValueError: If validation fails
        """
        assert isinstance(signal_data, USB4MockSignalData), f"Signal data must be USB4MockSignalData, got {type(signal_data)}"
        assert isinstance(mock_config, USB4MockConfig), f"Mock config must be USB4MockConfig, got {type(mock_config)}"

        try:
            start_time = time.time()

            # Initialize validation results
            signal_validation = {}
            timing_validation = {}
            power_validation = {}
            compliance_validation = {}
            performance_validation = {}
            recommendations = []

            # Validate signal characteristics
            if self.config.enable_bounds_checking:
                signal_validation = self._validate_signal_bounds(signal_data, mock_config)
                if not signal_validation.get("bounds_valid", True):
                    recommendations.append("Signal bounds validation failed - check amplitude and noise settings")

            # Validate timing characteristics
            timing_validation = self._validate_timing_characteristics(signal_data, mock_config)
            if not timing_validation.get("timing_valid", True):
                recommendations.append("Timing validation failed - check sample rate and jitter settings")

            # Validate power characteristics
            power_validation = self._validate_power_characteristics(signal_data, mock_config)
            if not power_validation.get("power_valid", True):
                recommendations.append("Power validation failed - check signal power levels")

            # Validate compliance if enabled
            if self.config.enable_compliance_validation:
                compliance_validation = self._validate_compliance_characteristics(signal_data, mock_config)
                if not compliance_validation.get("compliance_valid", True):
                    recommendations.append("Compliance validation failed - check USB4 specification adherence")

            # Validate performance if enabled
            if self.config.enable_performance_validation:
                performance_validation = self._validate_performance_characteristics(signal_data, mock_config)
                if not performance_validation.get("performance_valid", True):
                    recommendations.append("Performance validation failed - check signal quality metrics")

            # Determine overall result
            overall_result = self._determine_overall_result(
                signal_validation, timing_validation, power_validation, compliance_validation, performance_validation
            )

            validation_time = time.time() - start_time

            # Create validation result
            result = USB4MockValidationResult(
                test_name=f"Mock Signal Validation - {mock_config.signal_mode.name}",
                validation_level=self.config.validation_level,
                overall_result=overall_result,
                signal_validation=signal_validation,
                timing_validation=timing_validation,
                power_validation=power_validation,
                compliance_validation=compliance_validation,
                performance_validation=performance_validation,
                validation_time=validation_time,
                recommendations=recommendations,
            )

            # Store in history
            self.validation_history.append(result)

            logger.info(f"Mock signal validation completed: {overall_result.name}")
            return result

        except Exception as e:
            logger.error(f"Mock signal validation failed: {e}")
            raise ValueError(f"Mock signal validation failed: {e}")

    def validate_tunneling_simulation(
        self, simulator: USB4TunnelingSimulator, tunnel_config: USB4TunnelConfig, test_data: npt.NDArray[np.uint8]
    ) -> USB4MockValidationResult:
        """
        Validate USB4 tunneling simulation

        Args:
            simulator: Tunneling simulator to validate
            tunnel_config: Tunnel configuration
            test_data: Test data used in simulation

        Returns:
            Validation result

        Raises:
            ValueError: If validation fails
        """
        assert isinstance(simulator, USB4TunnelingSimulator), f"Simulator must be USB4TunnelingSimulator, got {type(simulator)}"
        assert isinstance(tunnel_config, USB4TunnelConfig), f"Tunnel config must be USB4TunnelConfig, got {type(tunnel_config)}"
        assert isinstance(test_data, np.ndarray), f"Test data must be numpy array, got {type(test_data)}"

        try:
            start_time = time.time()

            # Get tunnel statistics
            tunnel_stats = simulator.get_tunnel_statistics()

            # Initialize validation results
            signal_validation = {"tunnel_established": simulator.tunnel_state.name == "ACTIVE"}
            timing_validation = {}
            power_validation = {}
            compliance_validation = {}
            performance_validation = {}
            recommendations = []

            # Validate tunnel performance
            performance_validation = self._validate_tunnel_performance(tunnel_stats, tunnel_config)
            if not performance_validation.get("performance_valid", True):
                recommendations.append("Tunnel performance validation failed - check bandwidth and latency")

            # Validate tunnel compliance
            compliance_validation = self._validate_tunnel_compliance(tunnel_stats, tunnel_config)
            if not compliance_validation.get("compliance_valid", True):
                recommendations.append("Tunnel compliance validation failed - check protocol specifications")

            # Validate protocol-specific behavior
            protocol_validation = self._validate_protocol_behavior(simulator, tunnel_config, test_data)
            performance_validation.update(protocol_validation)

            # Determine overall result
            overall_result = self._determine_overall_result(
                signal_validation, timing_validation, power_validation, compliance_validation, performance_validation
            )

            validation_time = time.time() - start_time

            # Create validation result
            result = USB4MockValidationResult(
                test_name=f"Tunneling Simulation Validation - {tunnel_config.tunnel_mode.name}",
                validation_level=self.config.validation_level,
                overall_result=overall_result,
                signal_validation=signal_validation,
                timing_validation=timing_validation,
                power_validation=power_validation,
                compliance_validation=compliance_validation,
                performance_validation=performance_validation,
                validation_time=validation_time,
                recommendations=recommendations,
            )

            # Store in history
            self.validation_history.append(result)

            logger.info(f"Tunneling simulation validation completed: {overall_result.name}")
            return result

        except Exception as e:
            logger.error(f"Tunneling simulation validation failed: {e}")
            raise ValueError(f"Tunneling simulation validation failed: {e}")

    def validate_thunderbolt_device(self, device: ThunderboltMockDevice) -> USB4MockValidationResult:
        """
        Validate Thunderbolt mock device

        Args:
            device: Thunderbolt device to validate

        Returns:
            Validation result

        Raises:
            ValueError: If validation fails
        """
        assert isinstance(device, ThunderboltMockDevice), f"Device must be ThunderboltMockDevice, got {type(device)}"

        try:
            start_time = time.time()

            # Get device status
            device_status = device.get_device_status()

            # Initialize validation results
            signal_validation = {}
            timing_validation = {}
            power_validation = {}
            compliance_validation = {}
            performance_validation = {}
            recommendations = []

            # Validate device state
            signal_validation = self._validate_device_state(device, device_status)
            if not signal_validation.get("state_valid", True):
                recommendations.append("Device state validation failed - check connection and authentication")

            # Validate authentication if available
            if device.last_auth_result:
                compliance_validation = self._validate_device_authentication(device.last_auth_result)
                if not compliance_validation.get("auth_valid", True):
                    recommendations.append("Device authentication validation failed - check security configuration")

            # Validate power negotiation if available
            if device.power_negotiation:
                power_validation = self._validate_device_power(device.power_negotiation)
                if not power_validation.get("power_valid", True):
                    recommendations.append("Device power validation failed - check power delivery settings")

            # Validate performance characteristics
            performance_validation = self._validate_device_performance(device, device_status)
            if not performance_validation.get("performance_valid", True):
                recommendations.append("Device performance validation failed - check bandwidth and latency")

            # Determine overall result
            overall_result = self._determine_overall_result(
                signal_validation, timing_validation, power_validation, compliance_validation, performance_validation
            )

            validation_time = time.time() - start_time

            # Create validation result
            result = USB4MockValidationResult(
                test_name=f"Thunderbolt Device Validation - {device.device_info.device_name}",
                validation_level=self.config.validation_level,
                overall_result=overall_result,
                signal_validation=signal_validation,
                timing_validation=timing_validation,
                power_validation=power_validation,
                compliance_validation=compliance_validation,
                performance_validation=performance_validation,
                validation_time=validation_time,
                recommendations=recommendations,
            )

            # Store in history
            self.validation_history.append(result)

            logger.info(f"Thunderbolt device validation completed: {overall_result.name}")
            return result

        except Exception as e:
            logger.error(f"Thunderbolt device validation failed: {e}")
            raise ValueError(f"Thunderbolt device validation failed: {e}")

    def _validate_signal_bounds(
        self, signal_data: USB4MockSignalData, mock_config: USB4MockConfig
    ) -> Dict[str, Union[bool, float, str]]:
        """Validate signal bounds and characteristics"""
        try:
            validation = {}

            # Check voltage bounds
            lane0_max = float(np.max(np.abs(signal_data.lane0_voltage)))
            expected_max = mock_config.amplitude * (1 + mock_config.noise_level)
            voltage_bounds_valid = lane0_max <= expected_max * (1 + self.config.signal_tolerance)

            validation["voltage_bounds_valid"] = voltage_bounds_valid
            validation["lane0_max_voltage"] = lane0_max
            validation["expected_max_voltage"] = expected_max

            # Check dual-lane consistency if applicable
            if signal_data.lane1_voltage is not None:
                lane1_max = float(np.max(np.abs(signal_data.lane1_voltage)))
                lane_consistency = abs(lane0_max - lane1_max) / max(lane0_max, lane1_max) < self.config.signal_tolerance

                validation["lane_consistency_valid"] = lane_consistency
                validation["lane1_max_voltage"] = lane1_max
                validation["lane_voltage_mismatch"] = abs(lane0_max - lane1_max) / max(lane0_max, lane1_max)

            # Check signal quality metrics
            signal_quality = signal_data.signal_quality
            snr_valid = signal_quality.get("lane0_snr_db", 0) >= 15.0  # Minimum SNR

            validation["snr_valid"] = snr_valid
            validation["lane0_snr_db"] = signal_quality.get("lane0_snr_db", 0)

            # Overall bounds validation
            validation["bounds_valid"] = voltage_bounds_valid and snr_valid
            if signal_data.lane1_voltage is not None:
                validation["bounds_valid"] = validation["bounds_valid"] and lane_consistency

            return validation

        except Exception as e:
            logger.error(f"Signal bounds validation failed: {e}")
            return {"bounds_valid": False, "error": str(e)}

    def _validate_timing_characteristics(
        self, signal_data: USB4MockSignalData, mock_config: USB4MockConfig
    ) -> Dict[str, Union[bool, float, str]]:
        """Validate timing characteristics"""
        try:
            validation = {}

            # Check sample rate consistency
            expected_sample_period = 1.0 / mock_config.sample_rate
            actual_sample_period = float(np.mean(np.diff(signal_data.time_base)))
            sample_rate_error = abs(actual_sample_period - expected_sample_period) / expected_sample_period
            sample_rate_valid = sample_rate_error < self.config.timing_tolerance

            validation["sample_rate_valid"] = sample_rate_valid
            validation["expected_sample_period"] = expected_sample_period
            validation["actual_sample_period"] = actual_sample_period
            validation["sample_rate_error"] = sample_rate_error

            # Check bit rate consistency
            bit_period = 1.0 / mock_config.bit_rate
            samples_per_bit = int(mock_config.sample_rate / mock_config.bit_rate)
            bit_rate_consistent = samples_per_bit > 0

            validation["bit_rate_consistent"] = bit_rate_consistent
            validation["samples_per_bit"] = samples_per_bit
            validation["bit_period"] = bit_period

            # Check jitter characteristics if specified
            if mock_config.jitter_rms > 0:
                # Estimate jitter from signal
                zero_crossings = np.where(np.diff(np.signbit(signal_data.lane0_voltage)))[0]
                if len(zero_crossings) > 1:
                    crossing_times = signal_data.time_base[zero_crossings]
                    time_intervals = np.diff(crossing_times)
                    estimated_jitter = float(np.std(time_intervals))

                    jitter_reasonable = estimated_jitter <= mock_config.jitter_rms * 2  # Allow 2x margin

                    validation["jitter_reasonable"] = jitter_reasonable
                    validation["estimated_jitter"] = estimated_jitter
                    validation["expected_jitter"] = mock_config.jitter_rms

            # Overall timing validation
            validation["timing_valid"] = sample_rate_valid and bit_rate_consistent

            return validation

        except Exception as e:
            logger.error(f"Timing validation failed: {e}")
            return {"timing_valid": False, "error": str(e)}

    def _validate_power_characteristics(
        self, signal_data: USB4MockSignalData, mock_config: USB4MockConfig
    ) -> Dict[str, Union[bool, float, str]]:
        """Validate power characteristics"""
        try:
            validation = {}

            # Calculate signal power
            lane0_power = float(np.mean(signal_data.lane0_voltage**2))
            expected_power = mock_config.amplitude**2
            power_error = abs(lane0_power - expected_power) / expected_power
            power_valid = power_error < self.config.power_tolerance

            validation["power_valid"] = power_valid
            validation["lane0_power"] = lane0_power
            validation["expected_power"] = expected_power
            validation["power_error"] = power_error

            # Check power balance between lanes if dual-lane
            if signal_data.lane1_voltage is not None:
                lane1_power = float(np.mean(signal_data.lane1_voltage**2))
                power_balance = abs(lane0_power - lane1_power) / max(lane0_power, lane1_power)
                power_balance_valid = power_balance < self.config.power_tolerance

                validation["power_balance_valid"] = power_balance_valid
                validation["lane1_power"] = lane1_power
                validation["power_balance"] = power_balance
                validation["power_valid"] = validation["power_valid"] and power_balance_valid

            return validation

        except Exception as e:
            logger.error(f"Power validation failed: {e}")
            return {"power_valid": False, "error": str(e)}

    def _validate_compliance_characteristics(
        self, signal_data: USB4MockSignalData, mock_config: USB4MockConfig
    ) -> Dict[str, Union[bool, float, str]]:
        """Validate compliance characteristics"""
        try:
            validation = {}

            # Create compliance validator
            compliance_config = USB4ComplianceConfig(
                signal_mode=mock_config.signal_mode,
                test_pattern="PRBS31",
                sample_rate=mock_config.sample_rate,
                record_length=100e-6,
                voltage_range=mock_config.amplitude * 2,
                test_types=[USB4ComplianceType.SIGNAL_INTEGRITY],
            )

            validator = USB4ComplianceValidator(compliance_config)

            # Run compliance tests
            lane1_data = signal_data.lane1_voltage if signal_data.lane1_voltage is not None else None
            compliance_results = validator.run_signal_integrity_tests(
                signal_data.lane0_voltage, lane1_data, signal_data.time_base
            )

            # Check compliance results
            compliance_passed = all(result.status for result in compliance_results.values())

            validation["compliance_valid"] = compliance_passed
            validation["compliance_test_count"] = len(compliance_results)
            validation["compliance_pass_count"] = sum(1 for r in compliance_results.values() if r.status)

            # Add specific compliance metrics
            for test_name, result in compliance_results.items():
                validation[f"compliance_{test_name}_status"] = result.status
                validation[f"compliance_{test_name}_value"] = result.measured_value
                validation[f"compliance_{test_name}_margin"] = result.margin

            return validation

        except Exception as e:
            logger.error(f"Compliance validation failed: {e}")
            return {"compliance_valid": False, "error": str(e)}

    def _validate_performance_characteristics(
        self, signal_data: USB4MockSignalData, mock_config: USB4MockConfig
    ) -> Dict[str, Union[bool, float, str]]:
        """Validate performance characteristics"""
        try:
            validation = {}

            # Check signal quality score
            signal_quality = signal_data.signal_quality
            overall_quality = signal_quality.get("overall_quality_score", 0)
            quality_threshold = 70.0 if self.config.validation_level == USB4MockValidationLevel.BASIC else 80.0
            quality_valid = overall_quality >= quality_threshold

            validation["quality_valid"] = quality_valid
            validation["overall_quality_score"] = overall_quality
            validation["quality_threshold"] = quality_threshold

            # Check eye diagram metrics if available
            if "lane0_eye_height" in signal_quality:
                eye_height = signal_quality["lane0_eye_height"]
                eye_height_valid = eye_height >= self.signal_specs.EYE_HEIGHT_MIN

                validation["eye_height_valid"] = eye_height_valid
                validation["eye_height"] = eye_height
                validation["eye_height_min"] = self.signal_specs.EYE_HEIGHT_MIN

            # Check jitter metrics if available
            if "lane0_total_jitter_ui" in signal_quality:
                total_jitter = signal_quality["lane0_total_jitter_ui"]
                jitter_valid = total_jitter <= self.signal_specs.TOTAL_JITTER_MAX

                validation["jitter_valid"] = jitter_valid
                validation["total_jitter_ui"] = total_jitter
                validation["jitter_max"] = self.signal_specs.TOTAL_JITTER_MAX

            # Overall performance validation
            validation["performance_valid"] = quality_valid
            if "eye_height_valid" in validation:
                validation["performance_valid"] = validation["performance_valid"] and validation["eye_height_valid"]
            if "jitter_valid" in validation:
                validation["performance_valid"] = validation["performance_valid"] and validation["jitter_valid"]

            return validation

        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return {"performance_valid": False, "error": str(e)}

    def _validate_tunnel_performance(self, tunnel_stats, tunnel_config: USB4TunnelConfig) -> Dict[str, Union[bool, float, str]]:
        """Validate tunnel performance"""
        try:
            validation = {}

            # Check bandwidth utilization
            bandwidth_reasonable = 0.0 <= tunnel_stats.bandwidth_utilization <= 1.0

            validation["bandwidth_utilization_valid"] = bandwidth_reasonable
            validation["bandwidth_utilization"] = tunnel_stats.bandwidth_utilization

            # Check latency
            latency_valid = tunnel_stats.average_latency <= tunnel_config.latency_target * 2  # Allow 2x margin

            validation["latency_valid"] = latency_valid
            validation["average_latency"] = tunnel_stats.average_latency
            validation["latency_target"] = tunnel_config.latency_target

            # Check packet loss rate
            loss_rate_valid = tunnel_stats.packet_loss_rate < 1e-6  # Very low loss rate

            validation["loss_rate_valid"] = loss_rate_valid
            validation["packet_loss_rate"] = tunnel_stats.packet_loss_rate

            # Overall performance validation
            validation["performance_valid"] = bandwidth_reasonable and latency_valid and loss_rate_valid

            return validation

        except Exception as e:
            logger.error(f"Tunnel performance validation failed: {e}")
            return {"performance_valid": False, "error": str(e)}

    def _validate_tunnel_compliance(self, tunnel_stats, tunnel_config: USB4TunnelConfig) -> Dict[str, Union[bool, float, str]]:
        """Validate tunnel compliance"""
        try:
            validation = {}

            # Check if tunnel mode is supported
            supported_modes = [
                USB4TunnelingMode.PCIE,
                USB4TunnelingMode.DISPLAYPORT,
                USB4TunnelingMode.USB32,
                USB4TunnelingMode.NATIVE,
            ]
            mode_supported = tunnel_config.tunnel_mode in supported_modes

            validation["mode_supported"] = mode_supported
            validation["tunnel_mode"] = tunnel_config.tunnel_mode.name

            # Check bandwidth allocation
            bandwidth_valid = tunnel_config.bandwidth_allocation <= 40e9  # USB4 max bandwidth

            validation["bandwidth_allocation_valid"] = bandwidth_valid
            validation["bandwidth_allocation_gbps"] = tunnel_config.bandwidth_allocation / 1e9

            # Check packet size limits
            packet_size_valid = tunnel_config.max_packet_size <= 4096  # Reasonable limit

            validation["packet_size_valid"] = packet_size_valid
            validation["max_packet_size"] = tunnel_config.max_packet_size

            # Overall compliance validation
            validation["compliance_valid"] = mode_supported and bandwidth_valid and packet_size_valid

            return validation

        except Exception as e:
            logger.error(f"Tunnel compliance validation failed: {e}")
            return {"compliance_valid": False, "error": str(e)}

    def _validate_protocol_behavior(
        self, simulator: USB4TunnelingSimulator, tunnel_config: USB4TunnelConfig, test_data: npt.NDArray[np.uint8]
    ) -> Dict[str, Union[bool, float, str]]:
        """Validate protocol-specific behavior"""
        try:
            validation = {}

            # Run protocol-specific simulation
            protocol_results = simulator.simulate_protocol_specific_behavior(test_data, 1.0)

            # Validate protocol results
            if tunnel_config.tunnel_mode == USB4TunnelingMode.PCIE:
                # PCIe-specific validation
                efficiency_valid = protocol_results.get("efficiency", 0) >= 0.8
                latency_valid = protocol_results.get("total_latency", float("inf")) <= 10e-6  # 10 μs

                validation["pcie_efficiency_valid"] = efficiency_valid
                validation["pcie_latency_valid"] = latency_valid
                validation["pcie_efficiency"] = protocol_results.get("efficiency", 0)
                validation["pcie_latency"] = protocol_results.get("total_latency", 0)

            elif tunnel_config.tunnel_mode == USB4TunnelingMode.DISPLAYPORT:
                # DisplayPort-specific validation
                efficiency_valid = protocol_results.get("efficiency", 0) >= 0.85
                latency_valid = protocol_results.get("total_latency", float("inf")) <= 100e-6  # 100 μs

                validation["dp_efficiency_valid"] = efficiency_valid
                validation["dp_latency_valid"] = latency_valid
                validation["dp_efficiency"] = protocol_results.get("efficiency", 0)
                validation["dp_latency"] = protocol_results.get("total_latency", 0)

            elif tunnel_config.tunnel_mode == USB4TunnelingMode.USB32:
                # USB 3.2-specific validation
                efficiency_valid = protocol_results.get("efficiency", 0) >= 0.9
                latency_valid = protocol_results.get("total_latency", float("inf")) <= 10e-6  # 10 μs

                validation["usb32_efficiency_valid"] = efficiency_valid
                validation["usb32_latency_valid"] = latency_valid
                validation["usb32_efficiency"] = protocol_results.get("efficiency", 0)
                validation["usb32_latency"] = protocol_results.get("total_latency", 0)

            # Check compliance
            compliance_check = protocol_results.get("compliance_check", False)
            validation["protocol_compliance_valid"] = compliance_check

            return validation

        except Exception as e:
            logger.error(f"Protocol behavior validation failed: {e}")
            return {"protocol_behavior_valid": False, "error": str(e)}

    def _validate_device_state(
        self, device: ThunderboltMockDevice, device_status: Dict[str, Union[str, float, bool, int]]
    ) -> Dict[str, Union[bool, float, str]]:
        """Validate device state"""
        try:
            validation = {}

            # Check device connection state
            connected = device_status.get("connected", False)
            state_valid = isinstance(connected, bool)

            validation["state_valid"] = state_valid
            validation["connected"] = connected
            validation["device_state"] = device_status.get("state", "UNKNOWN")

            # Check device type consistency
            device_type = device_status.get("device_type", "")
            type_valid = device_type in ["HOST", "DEVICE", "HUB", "DOCK", "STORAGE", "DISPLAY", "EGPU"]

            validation["device_type_valid"] = type_valid
            validation["device_type"] = device_type

            # Check chain position if applicable
            chain_position = device_status.get("chain_position", 0)
            chain_valid = 0 <= chain_position <= self.thunderbolt_specs.MAX_DAISY_DEVICES

            validation["chain_position_valid"] = chain_valid
            validation["chain_position"] = chain_position

            return validation

        except Exception as e:
            logger.error(f"Device state validation failed: {e}")
            return {"state_valid": False, "error": str(e)}

    def _validate_device_authentication(self, auth_result) -> Dict[str, Union[bool, float, str]]:
        """Validate device authentication"""
        try:
            validation = {}

            # Check authentication time
            auth_time_valid = auth_result.auth_time <= self.thunderbolt_specs.AUTH_TIMEOUT

            validation["auth_time_valid"] = auth_time_valid
            validation["auth_time"] = auth_result.auth_time
            validation["auth_timeout"] = self.thunderbolt_specs.AUTH_TIMEOUT

            # Check authentication success
            auth_success_valid = isinstance(auth_result.success, bool)

            validation["auth_success_valid"] = auth_success_valid
            validation["auth_success"] = auth_result.success

            # Check certificate validity
            cert_valid = isinstance(auth_result.certificate_valid, bool)

            validation["cert_validity_valid"] = cert_valid
            validation["certificate_valid"] = auth_result.certificate_valid

            # Overall authentication validation
            validation["auth_valid"] = auth_time_valid and auth_success_valid and cert_valid

            return validation

        except Exception as e:
            logger.error(f"Device authentication validation failed: {e}")
            return {"auth_valid": False, "error": str(e)}

    def _validate_device_power(self, power_negotiation) -> Dict[str, Union[bool, float, str]]:
        """Validate device power negotiation"""
        try:
            validation = {}

            # Check power levels
            power_reasonable = 0 <= power_negotiation.allocated_power <= self.thunderbolt_specs.TB4_POWER_DELIVERY

            validation["power_level_valid"] = power_reasonable
            validation["allocated_power"] = power_negotiation.allocated_power
            validation["max_power"] = self.thunderbolt_specs.TB4_POWER_DELIVERY

            # Check voltage levels
            voltage_reasonable = 5.0 <= power_negotiation.voltage <= 28.0  # USB-PD voltage range

            validation["voltage_valid"] = voltage_reasonable
            validation["voltage"] = power_negotiation.voltage

            # Check current levels
            current_reasonable = power_negotiation.current >= 0 and power_negotiation.current <= 5.0  # 5A max

            validation["current_valid"] = current_reasonable
            validation["current"] = power_negotiation.current

            # Check negotiation time
            negotiation_time_valid = power_negotiation.negotiation_time <= self.thunderbolt_specs.POWER_NEGOTIATION_TIME

            validation["negotiation_time_valid"] = negotiation_time_valid
            validation["negotiation_time"] = power_negotiation.negotiation_time

            # Overall power validation
            validation["power_valid"] = power_reasonable and voltage_reasonable and current_reasonable and negotiation_time_valid

            return validation

        except Exception as e:
            logger.error(f"Device power validation failed: {e}")
            return {"power_valid": False, "error": str(e)}

    def _validate_device_performance(
        self, device: ThunderboltMockDevice, device_status: Dict[str, Union[str, float, bool, int]]
    ) -> Dict[str, Union[bool, float, str]]:
        """Validate device performance"""
        try:
            validation = {}

            # Check bandwidth usage
            bandwidth_usage = device_status.get("bandwidth_usage", 0)
            bandwidth_valid = 0 <= bandwidth_usage <= 1.0

            validation["bandwidth_usage_valid"] = bandwidth_valid
            validation["bandwidth_usage"] = bandwidth_usage

            # Check latency if available
            if "average_latency_us" in device_status:
                avg_latency = device_status["average_latency_us"]
                latency_valid = avg_latency <= 100.0  # 100 μs reasonable limit

                validation["latency_valid"] = latency_valid
                validation["average_latency_us"] = avg_latency

            # Check device-specific performance
            device_type = device.device_info.device_type.name
            if device_type == "EGPU":
                # eGPU should have high bandwidth usage when active
                performance_valid = bandwidth_usage >= 0.1 or bandwidth_usage == 0  # Either active or idle
            elif device_type == "STORAGE":
                # Storage should have reasonable bandwidth patterns
                performance_valid = bandwidth_usage <= 0.8  # Not constantly at max
            else:
                performance_valid = True  # Default validation

            validation["device_performance_valid"] = performance_valid
            validation["device_type"] = device_type

            # Overall performance validation
            validation["performance_valid"] = bandwidth_valid and performance_valid
            if "latency_valid" in validation:
                validation["performance_valid"] = validation["performance_valid"] and validation["latency_valid"]

            return validation

        except Exception as e:
            logger.error(f"Device performance validation failed: {e}")
            return {"performance_valid": False, "error": str(e)}

    def _determine_overall_result(
        self,
        signal_validation: Dict[str, Union[bool, float, str]],
        timing_validation: Dict[str, Union[bool, float, str]],
        power_validation: Dict[str, Union[bool, float, str]],
        compliance_validation: Dict[str, Union[bool, float, str]],
        performance_validation: Dict[str, Union[bool, float, str]],
    ) -> USB4MockTestResult:
        """Determine overall validation result"""
        try:
            # Collect all validation results
            validations = [
                signal_validation.get("bounds_valid", True),
                timing_validation.get("timing_valid", True),
                power_validation.get("power_valid", True),
                compliance_validation.get("compliance_valid", True),
                performance_validation.get("performance_valid", True),
            ]

            # Count failures
            failures = sum(1 for v in validations if not v)

            # Determine result based on validation level
            if self.config.validation_level == USB4MockValidationLevel.BASIC:
                if failures == 0:
                    return USB4MockTestResult.PASS
                elif failures <= 1:
                    return USB4MockTestResult.WARNING
                else:
                    return USB4MockTestResult.FAIL
            elif self.config.validation_level == USB4MockValidationLevel.STANDARD:
                if failures == 0:
                    return USB4MockTestResult.PASS
                elif failures <= 1:
                    return USB4MockTestResult.WARNING
                else:
                    return USB4MockTestResult.FAIL
            elif self.config.validation_level == USB4MockValidationLevel.COMPREHENSIVE:
                if failures == 0:
                    return USB4MockTestResult.PASS
                else:
                    return USB4MockTestResult.FAIL
            else:  # CERTIFICATION
                if failures == 0:
                    return USB4MockTestResult.PASS
                else:
                    return USB4MockTestResult.FAIL

        except Exception as e:
            logger.error(f"Overall result determination failed: {e}")
            return USB4MockTestResult.INVALID

    def get_validation_summary(self) -> Dict[str, Union[int, float, List[str]]]:
        """
        Get validation summary statistics

        Returns:
            Dictionary with validation summary
        """
        try:
            if not self.validation_history:
                return {
                    "total_validations": 0,
                    "pass_count": 0,
                    "fail_count": 0,
                    "warning_count": 0,
                    "invalid_count": 0,
                    "pass_rate": 0.0,
                    "average_validation_time": 0.0,
                    "recent_recommendations": [],
                }

            # Count results
            total_validations = len(self.validation_history)
            pass_count = sum(1 for r in self.validation_history if r.overall_result == USB4MockTestResult.PASS)
            fail_count = sum(1 for r in self.validation_history if r.overall_result == USB4MockTestResult.FAIL)
            warning_count = sum(1 for r in self.validation_history if r.overall_result == USB4MockTestResult.WARNING)
            invalid_count = sum(1 for r in self.validation_history if r.overall_result == USB4MockTestResult.INVALID)

            # Calculate statistics
            pass_rate = pass_count / total_validations if total_validations > 0 else 0.0
            average_validation_time = np.mean([r.validation_time for r in self.validation_history])

            # Get recent recommendations
            recent_recommendations = []
            for result in self.validation_history[-5:]:  # Last 5 validations
                recent_recommendations.extend(result.recommendations)

            return {
                "total_validations": total_validations,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "warning_count": warning_count,
                "invalid_count": invalid_count,
                "pass_rate": pass_rate,
                "average_validation_time": float(average_validation_time),
                "recent_recommendations": recent_recommendations[-10:],  # Last 10 recommendations
            }

        except Exception as e:
            logger.error(f"Validation summary generation failed: {e}")
            return {"error": str(e)}


__all__ = [
    "USB4MockValidationLevel",
    "USB4MockTestResult",
    "USB4MockValidationConfig",
    "USB4MockValidationResult",
    "USB4MockTestValidator",
]
