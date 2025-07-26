"""
USB4 Power Meter Module

This module provides power measurement capabilities specifically optimized for USB4 and
Thunderbolt 4 power analysis with support for USB-PD measurements and power state analysis.

Features:
- USB4 v2.0 power measurements
- USB-PD protocol analysis
- Power state transition monitoring
- Thunderbolt 4 power delivery testing
- Real-time power monitoring
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..protocols.usb4.constants import ThunderboltSpecs, USB4LinkState, USB4Specs
from .controller import InstrumentController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USB4PowerMeasurementType(Enum):
    """USB4 power measurement types"""

    VOLTAGE = auto()
    CURRENT = auto()
    POWER = auto()
    ENERGY = auto()
    EFFICIENCY = auto()


class USB4PowerRange(Enum):
    """USB4 power measurement ranges"""

    LOW_POWER = auto()  # < 15W
    MEDIUM_POWER = auto()  # 15W - 60W
    HIGH_POWER = auto()  # 60W - 100W
    EXTENDED_POWER = auto()  # > 100W (Thunderbolt)


@dataclass
class USB4PowerConfig:
    """USB4 power meter configuration"""

    measurement_type: USB4PowerMeasurementType
    power_range: USB4PowerRange
    sample_rate: float
    averaging_count: int
    voltage_range: float
    current_range: float
    enable_usb_pd: bool = True

    def __post_init__(self) -> None:
        """Validate power meter configuration"""
        assert isinstance(
            self.measurement_type, USB4PowerMeasurementType
        ), f"Measurement type must be USB4PowerMeasurementType, got {type(self.measurement_type)}"
        assert isinstance(self.power_range, USB4PowerRange), f"Power range must be USB4PowerRange, got {type(self.power_range)}"
        assert isinstance(self.sample_rate, float), f"Sample rate must be float, got {type(self.sample_rate)}"
        assert isinstance(self.averaging_count, int), f"Averaging count must be int, got {type(self.averaging_count)}"
        assert isinstance(self.voltage_range, float), f"Voltage range must be float, got {type(self.voltage_range)}"
        assert isinstance(self.current_range, float), f"Current range must be float, got {type(self.current_range)}"

        assert self.sample_rate > 0, f"Sample rate must be positive, got {self.sample_rate}"
        assert self.averaging_count > 0, f"Averaging count must be positive, got {self.averaging_count}"
        assert self.voltage_range > 0, f"Voltage range must be positive, got {self.voltage_range}"
        assert self.current_range > 0, f"Current range must be positive, got {self.current_range}"


@dataclass
class USB4PowerMeasurement:
    """USB4 power measurement result"""

    voltage: float
    current: float
    power: float
    timestamp: float
    link_state: Optional[USB4LinkState] = None
    usb_pd_voltage: Optional[float] = None
    usb_pd_current: Optional[float] = None
    metadata: Dict[str, Union[str, float, bool]] = None

    def __post_init__(self) -> None:
        """Validate power measurement"""
        assert isinstance(self.voltage, float), f"Voltage must be float, got {type(self.voltage)}"
        assert isinstance(self.current, float), f"Current must be float, got {type(self.current)}"
        assert isinstance(self.power, float), f"Power must be float, got {type(self.power)}"
        assert isinstance(self.timestamp, float), f"Timestamp must be float, got {type(self.timestamp)}"

        assert self.voltage >= 0, f"Voltage must be non-negative, got {self.voltage}"
        assert self.current >= 0, f"Current must be non-negative, got {self.current}"
        assert self.power >= 0, f"Power must be non-negative, got {self.power}"
        assert self.timestamp > 0, f"Timestamp must be positive, got {self.timestamp}"

        if self.metadata is None:
            self.metadata = {}


@dataclass
class USB4PowerProfile:
    """USB4 power consumption profile"""

    measurements: List[USB4PowerMeasurement]
    average_power: float
    peak_power: float
    minimum_power: float
    power_efficiency: float
    state_transitions: List[Tuple[float, USB4LinkState, USB4LinkState]]

    def __post_init__(self) -> None:
        """Validate power profile"""
        assert isinstance(self.measurements, list), f"Measurements must be list, got {type(self.measurements)}"
        assert all(
            isinstance(m, USB4PowerMeasurement) for m in self.measurements
        ), "All measurements must be USB4PowerMeasurement instances"
        assert isinstance(self.average_power, float), f"Average power must be float, got {type(self.average_power)}"
        assert isinstance(self.peak_power, float), f"Peak power must be float, got {type(self.peak_power)}"
        assert isinstance(self.minimum_power, float), f"Minimum power must be float, got {type(self.minimum_power)}"
        assert isinstance(self.power_efficiency, float), f"Power efficiency must be float, got {type(self.power_efficiency)}"
        assert isinstance(self.state_transitions, list), f"State transitions must be list, got {type(self.state_transitions)}"

        assert len(self.measurements) > 0, "Measurements cannot be empty"
        assert self.average_power >= 0, f"Average power must be non-negative, got {self.average_power}"
        assert self.peak_power >= self.average_power, "Peak power must be >= average power"
        assert self.minimum_power <= self.average_power, "Minimum power must be <= average power"
        assert 0 <= self.power_efficiency <= 1, f"Power efficiency must be between 0 and 1, got {self.power_efficiency}"


class USB4PowerMeter:
    """Power meter for USB4 power analysis"""

    def __init__(self, resource_name: str, config: USB4PowerConfig) -> None:
        """
        Initialize USB4 power meter

        Args:
            resource_name: VISA resource identifier
            config: Power meter configuration

        Raises:
            ValueError: If initialization fails
        """
        assert isinstance(resource_name, str), f"Resource name must be string, got {type(resource_name)}"
        assert isinstance(config, USB4PowerConfig), f"Config must be USB4PowerConfig, got {type(config)}"

        self.resource_name = resource_name
        self.config = config
        self.usb4_specs = USB4Specs()
        self.thunderbolt_specs = ThunderboltSpecs()

        # Initialize instrument controller
        self.controller = InstrumentController()
        self.connected = False

        # Power measurement state
        self.current_measurements: List[USB4PowerMeasurement] = []
        self.monitoring_active = False

        logger.info(f"USB4 power meter initialized for {resource_name}")

    def connect(self) -> bool:
        """
        Connect to power meter

        Returns:
            True if connection successful

        Raises:
            ValueError: If connection fails
        """
        try:
            self.controller.connect_instrument(self.resource_name)

            # Verify instrument identity
            idn = self.controller.query_instrument(self.resource_name, "*IDN?")
            logger.info(f"Connected to: {idn.strip()}")

            # Initialize power meter for USB4
            self._initialize_power_meter()

            self.connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to power meter: {e}")
            raise ValueError(f"Power meter connection failed: {e}")

    def disconnect(self) -> None:
        """Disconnect from power meter"""
        try:
            if self.connected:
                # Stop any ongoing measurements
                if self.monitoring_active:
                    self.stop_monitoring()

                self.controller.disconnect_instrument(self.resource_name)
                self.connected = False
                logger.info("Disconnected from power meter")
        except Exception as e:
            logger.error(f"Failed to disconnect from power meter: {e}")

    def _initialize_power_meter(self) -> None:
        """Initialize power meter for USB4 measurements"""
        try:
            # Reset instrument
            self.controller.send_command(self.resource_name, "*RST")
            time.sleep(1.0)  # Wait for reset

            # Configure measurement type
            if self.config.measurement_type == USB4PowerMeasurementType.VOLTAGE:
                self.controller.send_command(self.resource_name, ":CONFigure:VOLTage:DC")
            elif self.config.measurement_type == USB4PowerMeasurementType.CURRENT:
                self.controller.send_command(self.resource_name, ":CONFigure:CURRent:DC")
            elif self.config.measurement_type == USB4PowerMeasurementType.POWER:
                self.controller.send_command(self.resource_name, ":CONFigure:POWer:DC")

            # Set measurement ranges
            self._configure_measurement_ranges()

            # Configure averaging
            self.controller.send_command(self.resource_name, f":AVERage:COUNt {self.config.averaging_count}")
            self.controller.send_command(self.resource_name, ":AVERage:STATe ON")

            # Configure sample rate
            sample_period = 1.0 / self.config.sample_rate
            self.controller.send_command(self.resource_name, f":TRIGger:TIMer {sample_period}")

            # Configure USB-PD if enabled
            if self.config.enable_usb_pd:
                self._configure_usb_pd()

            logger.info("Power meter initialized for USB4 measurements")

        except Exception as e:
            raise ValueError(f"Power meter initialization failed: {e}")

    def _configure_measurement_ranges(self) -> None:
        """Configure measurement ranges based on power range"""
        try:
            if self.config.power_range == USB4PowerRange.LOW_POWER:
                # < 15W range
                voltage_range = 20.0  # 20V max
                current_range = 1.0  # 1A max
            elif self.config.power_range == USB4PowerRange.MEDIUM_POWER:
                # 15W - 60W range
                voltage_range = 20.0  # 20V max
                current_range = 3.0  # 3A max
            elif self.config.power_range == USB4PowerRange.HIGH_POWER:
                # 60W - 100W range
                voltage_range = 20.0  # 20V max
                current_range = 5.0  # 5A max
            else:  # EXTENDED_POWER
                # > 100W range (Thunderbolt)
                voltage_range = 28.0  # 28V max
                current_range = 5.0  # 5A max

            # Set voltage range
            self.controller.send_command(self.resource_name, f":VOLTage:RANGe {voltage_range}")

            # Set current range
            self.controller.send_command(self.resource_name, f":CURRent:RANGe {current_range}")

            # Set resolution for high accuracy
            self.controller.send_command(self.resource_name, ":VOLTage:RESOlution MAX")
            self.controller.send_command(self.resource_name, ":CURRent:RESOlution MAX")

        except Exception as e:
            raise ValueError(f"Measurement range configuration failed: {e}")

    def _configure_usb_pd(self) -> None:
        """Configure USB Power Delivery measurements"""
        try:
            # Enable USB-PD protocol analysis
            self.controller.send_command(self.resource_name, ":USB:PD:STATe ON")

            # Configure USB-PD voltage monitoring
            self.controller.send_command(self.resource_name, ":USB:PD:VOLTage:MONitor ON")

            # Configure USB-PD current monitoring
            self.controller.send_command(self.resource_name, ":USB:PD:CURRent:MONitor ON")

            # Set USB-PD negotiation timeout
            self.controller.send_command(self.resource_name, ":USB:PD:TIMeout 5.0")  # 5 seconds

        except Exception as e:
            logger.warning(f"USB-PD configuration failed: {e}")

    def measure_instantaneous_power(self, link_state: Optional[USB4LinkState] = None) -> USB4PowerMeasurement:
        """
        Measure instantaneous power

        Args:
            link_state: Optional USB4 link state

        Returns:
            Power measurement result

        Raises:
            ValueError: If measurement fails
        """
        if not self.connected:
            raise ValueError("Power meter not connected")

        try:
            # Trigger measurement
            self.controller.send_command(self.resource_name, ":INITiate")

            # Wait for measurement completion
            self.controller.send_command(self.resource_name, "*WAI")

            # Read voltage
            voltage_str = self.controller.query_instrument(self.resource_name, ":FETCh:VOLTage?")
            voltage = float(voltage_str)

            # Read current
            current_str = self.controller.query_instrument(self.resource_name, ":FETCh:CURRent?")
            current = float(current_str)

            # Calculate power
            power = voltage * current

            # Get USB-PD measurements if enabled
            usb_pd_voltage = None
            usb_pd_current = None

            if self.config.enable_usb_pd:
                try:
                    pd_voltage_str = self.controller.query_instrument(self.resource_name, ":USB:PD:VOLTage?")
                    usb_pd_voltage = float(pd_voltage_str)

                    pd_current_str = self.controller.query_instrument(self.resource_name, ":USB:PD:CURRent?")
                    usb_pd_current = float(pd_current_str)
                except Exception as e:
                    logger.warning(f"USB-PD measurement failed: {e}")

            # Create metadata
            metadata = {
                "measurement_type": self.config.measurement_type.name,
                "power_range": self.config.power_range.name,
                "averaging_count": self.config.averaging_count,
                "usb_pd_enabled": self.config.enable_usb_pd,
            }

            return USB4PowerMeasurement(
                voltage=voltage,
                current=current,
                power=power,
                timestamp=time.time(),
                link_state=link_state,
                usb_pd_voltage=usb_pd_voltage,
                usb_pd_current=usb_pd_current,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Power measurement failed: {e}")
            raise ValueError(f"Power measurement failed: {e}")

    def start_monitoring(self, duration: float, link_state_callback: Optional[callable] = None) -> bool:
        """
        Start continuous power monitoring

        Args:
            duration: Monitoring duration in seconds
            link_state_callback: Optional callback to get current link state

        Returns:
            True if monitoring started successfully

        Raises:
            ValueError: If monitoring start fails
        """
        assert isinstance(duration, float), f"Duration must be float, got {type(duration)}"
        assert duration > 0, f"Duration must be positive, got {duration}"

        if not self.connected:
            raise ValueError("Power meter not connected")

        if self.monitoring_active:
            raise ValueError("Monitoring already active")

        try:
            # Clear previous measurements
            self.current_measurements = []

            # Configure continuous measurement
            self.controller.send_command(self.resource_name, ":TRIGger:SOURce TIMer")
            self.controller.send_command(self.resource_name, ":TRIGger:COUNt INFinity")

            # Start monitoring
            self.controller.send_command(self.resource_name, ":INITiate")

            self.monitoring_active = True

            # Monitor for specified duration
            start_time = time.time()
            sample_interval = 1.0 / self.config.sample_rate

            while time.time() - start_time < duration and self.monitoring_active:
                # Get current link state if callback provided
                current_link_state = None
                if link_state_callback:
                    try:
                        current_link_state = link_state_callback()
                    except Exception as e:
                        logger.warning(f"Link state callback failed: {e}")

                # Take measurement
                measurement = self.measure_instantaneous_power(current_link_state)
                self.current_measurements.append(measurement)

                # Wait for next sample
                time.sleep(sample_interval)

            logger.info(f"Power monitoring completed: {len(self.current_measurements)} measurements")
            return True

        except Exception as e:
            logger.error(f"Power monitoring failed: {e}")
            self.monitoring_active = False
            raise ValueError(f"Power monitoring failed: {e}")

    def stop_monitoring(self) -> bool:
        """
        Stop continuous power monitoring

        Returns:
            True if monitoring stopped successfully

        Raises:
            ValueError: If monitoring stop fails
        """
        if not self.connected:
            raise ValueError("Power meter not connected")

        try:
            if self.monitoring_active:
                # Stop measurement
                self.controller.send_command(self.resource_name, ":ABORt")

                self.monitoring_active = False
                logger.info("Power monitoring stopped")

            return True

        except Exception as e:
            logger.error(f"Failed to stop power monitoring: {e}")
            raise ValueError(f"Power monitoring stop failed: {e}")

    def analyze_power_profile(self, measurements: Optional[List[USB4PowerMeasurement]] = None) -> USB4PowerProfile:
        """
        Analyze power consumption profile

        Args:
            measurements: Optional list of measurements (uses current if None)

        Returns:
            Power profile analysis

        Raises:
            ValueError: If analysis fails
        """
        if measurements is None:
            measurements = self.current_measurements

        if not measurements:
            raise ValueError("No measurements available for analysis")

        try:
            # Calculate power statistics
            powers = [m.power for m in measurements]
            average_power = float(np.mean(powers))
            peak_power = float(np.max(powers))
            minimum_power = float(np.min(powers))

            # Calculate power efficiency (simplified)
            # In real implementation, this would compare to theoretical minimum
            theoretical_minimum = self.usb4_specs.IDLE_POWER_U3  # Use U3 as baseline
            power_efficiency = theoretical_minimum / average_power if average_power > 0 else 0.0
            power_efficiency = min(power_efficiency, 1.0)  # Cap at 100%

            # Detect state transitions
            state_transitions = []
            if len(measurements) > 1:
                for i in range(1, len(measurements)):
                    prev_state = measurements[i - 1].link_state
                    curr_state = measurements[i].link_state

                    if prev_state and curr_state and prev_state != curr_state:
                        transition_time = measurements[i].timestamp
                        state_transitions.append((transition_time, prev_state, curr_state))

            return USB4PowerProfile(
                measurements=measurements,
                average_power=average_power,
                peak_power=peak_power,
                minimum_power=minimum_power,
                power_efficiency=power_efficiency,
                state_transitions=state_transitions,
            )

        except Exception as e:
            logger.error(f"Power profile analysis failed: {e}")
            raise ValueError(f"Power profile analysis failed: {e}")

    def measure_power_state_consumption(self, target_state: USB4LinkState, measurement_duration: float = 5.0) -> Dict[str, float]:
        """
        Measure power consumption for specific USB4 link state

        Args:
            target_state: USB4 link state to measure
            measurement_duration: Duration of measurement in seconds

        Returns:
            Dictionary with power consumption metrics

        Raises:
            ValueError: If measurement fails
        """
        assert isinstance(target_state, USB4LinkState), f"Target state must be USB4LinkState, got {type(target_state)}"
        assert isinstance(measurement_duration, float), f"Measurement duration must be float, got {type(measurement_duration)}"
        assert measurement_duration > 0, f"Measurement duration must be positive, got {measurement_duration}"

        if not self.connected:
            raise ValueError("Power meter not connected")

        try:
            # Start monitoring for the specified duration
            state_measurements = []
            start_time = time.time()
            sample_interval = 1.0 / self.config.sample_rate

            while time.time() - start_time < measurement_duration:
                measurement = self.measure_instantaneous_power(target_state)
                state_measurements.append(measurement)
                time.sleep(sample_interval)

            # Analyze measurements
            if not state_measurements:
                raise ValueError("No measurements collected")

            powers = [m.power for m in state_measurements]
            voltages = [m.voltage for m in state_measurements]
            currents = [m.current for m in state_measurements]

            results = {
                "state": target_state.name,
                "average_power": float(np.mean(powers)),
                "peak_power": float(np.max(powers)),
                "minimum_power": float(np.min(powers)),
                "power_std": float(np.std(powers)),
                "average_voltage": float(np.mean(voltages)),
                "average_current": float(np.mean(currents)),
                "measurement_count": len(state_measurements),
                "measurement_duration": measurement_duration,
            }

            # Compare to USB4 specifications
            expected_power = self._get_expected_power_for_state(target_state)
            results["expected_power"] = expected_power
            results["power_deviation"] = abs(results["average_power"] - expected_power)
            results["power_compliance"] = results["power_deviation"] < expected_power * 0.2  # 20% tolerance

            return results

        except Exception as e:
            logger.error(f"Power state measurement failed: {e}")
            raise ValueError(f"Power state measurement failed: {e}")

    def _get_expected_power_for_state(self, state: USB4LinkState) -> float:
        """Get expected power consumption for USB4 link state"""
        power_map = {
            USB4LinkState.U0: self.usb4_specs.IDLE_POWER_U0,
            USB4LinkState.U1: self.usb4_specs.IDLE_POWER_U1,
            USB4LinkState.U2: self.usb4_specs.IDLE_POWER_U2,
            USB4LinkState.U3: self.usb4_specs.IDLE_POWER_U3,
        }
        return power_map.get(state, 0.0)

    def measure_usb_pd_negotiation(self, timeout: float = 10.0) -> Dict[str, Union[float, bool, str]]:
        """
        Measure USB Power Delivery negotiation

        Args:
            timeout: Negotiation timeout in seconds

        Returns:
            Dictionary with USB-PD negotiation results

        Raises:
            ValueError: If measurement fails
        """
        assert isinstance(timeout, float), f"Timeout must be float, got {type(timeout)}"
        assert timeout > 0, f"Timeout must be positive, got {timeout}"

        if not self.connected:
            raise ValueError("Power meter not connected")

        if not self.config.enable_usb_pd:
            raise ValueError("USB-PD not enabled in configuration")

        try:
            # Start USB-PD negotiation monitoring
            self.controller.send_command(self.resource_name, ":USB:PD:NEGotiation:STARt")

            # Wait for negotiation completion or timeout
            start_time = time.time()
            negotiation_complete = False

            while time.time() - start_time < timeout and not negotiation_complete:
                status = self.controller.query_instrument(self.resource_name, ":USB:PD:NEGotiation:STATus?")
                if "COMPLETE" in status.upper():
                    negotiation_complete = True
                    break
                time.sleep(0.1)

            if not negotiation_complete:
                raise ValueError("USB-PD negotiation timeout")

            # Get negotiation results
            negotiated_voltage = float(self.controller.query_instrument(self.resource_name, ":USB:PD:VOLTage:NEGotiated?"))
            negotiated_current = float(self.controller.query_instrument(self.resource_name, ":USB:PD:CURRent:NEGotiated?"))
            negotiated_power = negotiated_voltage * negotiated_current

            # Get negotiation time
            negotiation_time = float(self.controller.query_instrument(self.resource_name, ":USB:PD:NEGotiation:TIME?"))

            # Get power delivery profile
            pd_profile = self.controller.query_instrument(self.resource_name, ":USB:PD:PROFile?")

            results = {
                "negotiation_successful": negotiation_complete,
                "negotiated_voltage": negotiated_voltage,
                "negotiated_current": negotiated_current,
                "negotiated_power": negotiated_power,
                "negotiation_time": negotiation_time,
                "pd_profile": pd_profile.strip(),
                "max_power_compliant": negotiated_power <= self.usb4_specs.MAX_POWER_DELIVERY,
                "thunderbolt_compliant": negotiated_power <= self.thunderbolt_specs.TB4_POWER_DELIVERY,
            }

            return results

        except Exception as e:
            logger.error(f"USB-PD negotiation measurement failed: {e}")
            raise ValueError(f"USB-PD negotiation measurement failed: {e}")

    def get_power_meter_status(self) -> Dict[str, Union[str, float, bool]]:
        """
        Get power meter status

        Returns:
            Dictionary with power meter status information

        Raises:
            ValueError: If status query fails
        """
        if not self.connected:
            raise ValueError("Power meter not connected")

        try:
            status = {}

            # Measurement configuration
            config_str = self.controller.query_instrument(self.resource_name, ":CONFigure?")
            status["measurement_configuration"] = config_str.strip()

            # Voltage range
            voltage_range = self.controller.query_instrument(self.resource_name, ":VOLTage:RANGe?")
            status["voltage_range"] = float(voltage_range)

            # Current range
            current_range = self.controller.query_instrument(self.resource_name, ":CURRent:RANGe?")
            status["current_range"] = float(current_range)

            # Averaging settings
            avg_count = self.controller.query_instrument(self.resource_name, ":AVERage:COUNt?")
            status["averaging_count"] = int(avg_count)

            avg_state = self.controller.query_instrument(self.resource_name, ":AVERage:STATe?")
            status["averaging_enabled"] = "ON" in avg_state.upper()

            # Trigger settings
            trigger_source = self.controller.query_instrument(self.resource_name, ":TRIGger:SOURce?")
            status["trigger_source"] = trigger_source.strip()

            # USB-PD status
            if self.config.enable_usb_pd:
                pd_state = self.controller.query_instrument(self.resource_name, ":USB:PD:STATe?")
                status["usb_pd_enabled"] = "ON" in pd_state.upper()

                if status["usb_pd_enabled"]:
                    pd_voltage = self.controller.query_instrument(self.resource_name, ":USB:PD:VOLTage?")
                    status["usb_pd_voltage"] = float(pd_voltage)

                    pd_current = self.controller.query_instrument(self.resource_name, ":USB:PD:CURRent?")
                    status["usb_pd_current"] = float(pd_current)

            # Monitoring status
            status["monitoring_active"] = self.monitoring_active
            status["measurement_count"] = len(self.current_measurements)

            return status

        except Exception as e:
            logger.error(f"Power meter status query failed: {e}")
            raise ValueError(f"Status query failed: {e}")


__all__ = [
    "USB4PowerMeasurementType",
    "USB4PowerRange",
    "USB4PowerConfig",
    "USB4PowerMeasurement",
    "USB4PowerProfile",
    "USB4PowerMeter",
]
