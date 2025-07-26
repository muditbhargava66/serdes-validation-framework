"""
USB4 Power State Management

This module implements USB4 power state management, monitoring, and validation
according to the USB4 v2.0 specification and power delivery requirements.

Features:
- USB4 power state transition monitoring (U0-U3)
- Power consumption measurement and validation
- Wake event validation and timing analysis
- Thermal management and throttling validation
- Power delivery protocol compliance
- USB-PD negotiation and validation
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import (
    TestResult,
    USB4Component,
    USB4Config,
    USB4TestResult,
)
from .constants import (
    USB4LinkState,
    USB4Specs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USB4PowerEvent(Enum):
    """USB4 power management events"""

    STATE_TRANSITION = auto()  # Power state transition
    WAKE_EVENT = auto()  # Wake from low power state
    THERMAL_EVENT = auto()  # Thermal management event
    POWER_DELIVERY = auto()  # USB-PD event
    THROTTLING = auto()  # Performance throttling
    VOLTAGE_CHANGE = auto()  # Supply voltage change
    CURRENT_LIMIT = auto()  # Current limiting event


class USB4ThermalState(Enum):
    """USB4 thermal management states"""

    NORMAL = auto()  # Normal operating temperature
    WARNING = auto()  # Temperature warning threshold
    THROTTLING = auto()  # Performance throttling active
    CRITICAL = auto()  # Critical temperature reached
    SHUTDOWN = auto()  # Thermal shutdown required


class USB4WakeSource(Enum):
    """USB4 wake event sources"""

    REMOTE_WAKE = auto()  # Remote wake signaling
    LOCAL_WAKE = auto()  # Local wake event
    TIMER_WAKE = auto()  # Timer-based wake
    POWER_BUTTON = auto()  # Power button wake
    USB_DEVICE = auto()  # USB device wake
    SYSTEM_WAKE = auto()  # System-initiated wake


@dataclass
class USB4PowerConfig(USB4Config):
    """USB4 power management configuration"""

    monitor_duration: float = 10.0  # Power monitoring duration (s)
    power_measurement_interval: float = 0.1  # Power measurement interval (s)
    thermal_monitoring: bool = True  # Enable thermal monitoring
    wake_event_monitoring: bool = True  # Enable wake event monitoring
    power_delivery_testing: bool = True  # Enable USB-PD testing
    throttling_testing: bool = True  # Enable throttling testing
    temperature_range: Tuple[float, float] = (-40.0, 85.0)  # Operating temp range (°C)
    voltage_tolerance: float = 0.05  # Voltage tolerance (5%)
    current_limit: float = 5.0  # Current limit (A)


@dataclass
class USB4PowerTransition:
    """USB4 power state transition record"""

    from_state: USB4LinkState
    to_state: USB4LinkState
    event: USB4PowerEvent
    timestamp: float
    transition_time: float
    power_before: float
    power_after: float
    success: bool
    wake_source: Optional[USB4WakeSource] = None
    error_info: Optional[str] = None


@dataclass
class USB4PowerMeasurement:
    """USB4 power measurement data"""

    timestamp: float
    link_state: USB4LinkState
    voltage: float  # Supply voltage (V)
    current: float  # Supply current (A)
    power: float  # Power consumption (W)
    temperature: float  # Junction temperature (°C)
    thermal_state: USB4ThermalState
    throttling_active: bool


@dataclass
class USB4WakeEvent:
    """USB4 wake event record"""

    timestamp: float
    wake_source: USB4WakeSource
    from_state: USB4LinkState
    to_state: USB4LinkState
    wake_time: float  # Time to complete wake (s)
    signal_integrity: float  # Signal quality after wake
    success: bool
    error_info: Optional[str] = None


@dataclass
class USB4ThermalEvent:
    """USB4 thermal management event"""

    timestamp: float
    temperature: float
    thermal_state: USB4ThermalState
    throttling_level: float  # Throttling percentage (0-1)
    performance_impact: float  # Performance reduction (0-1)
    recovery_time: float  # Time to recover from thermal event
    mitigation_action: str


@dataclass
class USB4PowerResults:
    """USB4 power management test results"""

    test_duration: float
    total_transitions: int
    successful_transitions: int
    power_measurements: List[USB4PowerMeasurement] = field(default_factory=list)
    state_transitions: List[USB4PowerTransition] = field(default_factory=list)
    wake_events: List[USB4WakeEvent] = field(default_factory=list)
    thermal_events: List[USB4ThermalEvent] = field(default_factory=list)
    average_power_by_state: Dict[USB4LinkState, float] = field(default_factory=dict)
    transition_times: Dict[Tuple[USB4LinkState, USB4LinkState], float] = field(default_factory=dict)
    wake_success_rate: float = 0.0
    thermal_compliance: bool = True
    power_delivery_compliance: bool = True


class USB4PowerManager(USB4Component):
    """USB4 power state management and monitoring"""

    def __init__(self, config: USB4PowerConfig):
        """
        Initialize USB4 power manager

        Args:
            config: Power management configuration
        """
        super().__init__(config)
        self.config: USB4PowerConfig = config
        self.specs = USB4Specs()

        # Current state
        self.current_state = USB4LinkState.U3
        self.current_temperature = 25.0  # Room temperature
        self.current_thermal_state = USB4ThermalState.NORMAL

        # Monitoring data
        self.power_measurements: List[USB4PowerMeasurement] = []
        self.state_transitions: List[USB4PowerTransition] = []
        self.wake_events: List[USB4WakeEvent] = []
        self.thermal_events: List[USB4ThermalEvent] = []

        # Statistics
        self.total_transitions = 0
        self.successful_transitions = 0
        self.wake_attempts = 0
        self.successful_wakes = 0

    def initialize(self) -> bool:
        """
        Initialize power management component

        Returns:
            True if initialization successful
        """
        try:
            self.validate_config()
            self.specs.validate_all()
            self._reset_monitoring_data()
            self._initialized = True
            self.logger.info("USB4 power manager initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize USB4 power manager: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up power management resources"""
        self._reset_monitoring_data()
        self._initialized = False
        self.logger.info("USB4 power manager cleaned up")

    def monitor_power_states(self, duration: float) -> USB4PowerResults:
        """
        Monitor USB4 power state transitions for specified duration

        Args:
            duration: Monitoring duration in seconds

        Returns:
            Power monitoring results
        """
        if not self.is_initialized:
            raise RuntimeError("Power manager not initialized")

        self.logger.info(f"Starting USB4 power state monitoring for {duration:.1f}s")

        start_time = time.time()
        self._reset_monitoring_data()

        while (time.time() - start_time) < duration:
            # Take power measurement
            measurement = self._take_power_measurement()
            self.power_measurements.append(measurement)

            # Check for state transitions
            self._check_state_transitions()

            # Monitor thermal conditions
            if self.config.thermal_monitoring:
                self._monitor_thermal_conditions()

            # Check for wake events
            if self.config.wake_event_monitoring:
                self._monitor_wake_events()

            time.sleep(self.config.power_measurement_interval)

        # Calculate results
        results = self._calculate_power_results(time.time() - start_time)

        self.logger.info(
            f"Power monitoring completed: {self.total_transitions} transitions, " f"{len(self.wake_events)} wake events"
        )
        return results

    def validate_power_consumption(self, target_state: USB4LinkState) -> USB4TestResult:
        """
        Validate power consumption for specific USB4 state

        Args:
            target_state: USB4 link state to validate

        Returns:
            Power consumption test result
        """
        self.logger.info(f"Validating power consumption for state {target_state.name}")

        # Transition to target state
        success = self._transition_to_state(target_state)
        if not success:
            return USB4TestResult(
                test_name=f"Power Consumption {target_state.name}",
                result=TestResult.FAIL,
                message="Failed to transition to target state",
            )

        # Measure power consumption
        time.sleep(0.1)  # Allow state to stabilize
        measurement = self._take_power_measurement()

        # Get expected power consumption
        expected_power = self._get_expected_power(target_state)
        tolerance = expected_power * 0.1  # 10% tolerance

        # Validate measurement
        power_valid = abs(measurement.power - expected_power) <= tolerance

        return USB4TestResult(
            test_name=f"Power Consumption {target_state.name}",
            result=TestResult.PASS if power_valid else TestResult.FAIL,
            measured_value=measurement.power,
            limit_value=expected_power,
            units="W",
            message=f"Power consumption {'within' if power_valid else 'outside'} specification",
        )

    def test_wake_events(self, wake_source: USB4WakeSource, count: int = 5) -> List[USB4WakeEvent]:
        """
        Test wake event validation and timing

        Args:
            wake_source: Source of wake events to test
            count: Number of wake events to test

        Returns:
            List of wake event results
        """
        self.logger.info(f"Testing {count} wake events from {wake_source.name}")

        wake_events = []

        for _i in range(count):
            # Ensure we're in a sleep state
            self._transition_to_state(USB4LinkState.U2)
            time.sleep(0.1)  # Allow state to stabilize

            # Generate wake event
            wake_event = self._generate_wake_event(wake_source)
            wake_events.append(wake_event)

            # Brief delay between tests
            time.sleep(0.5)

        self.wake_events.extend(wake_events)
        self.logger.info(f"Wake event testing completed: {len([e for e in wake_events if e.success])}/{count} successful")

        return wake_events

    def validate_thermal_management(self, temperature_profile: List[float]) -> List[USB4ThermalEvent]:
        """
        Validate thermal management and throttling

        Args:
            temperature_profile: List of temperatures to test (°C)

        Returns:
            List of thermal events
        """
        self.logger.info(f"Testing thermal management with {len(temperature_profile)} temperature points")

        thermal_events = []

        for temperature in temperature_profile:
            # Simulate temperature change
            self.current_temperature = temperature

            # Check thermal state
            thermal_state = self._determine_thermal_state(temperature)

            # Generate thermal event if state changed
            if thermal_state != self.current_thermal_state:
                thermal_event = self._handle_thermal_transition(thermal_state)
                thermal_events.append(thermal_event)
                self.current_thermal_state = thermal_state

            time.sleep(0.1)  # Brief delay for thermal response

        self.thermal_events.extend(thermal_events)
        self.logger.info(f"Thermal management testing completed: {len(thermal_events)} thermal events")

        return thermal_events

    def test_power_delivery(self, power_levels: List[float]) -> List[USB4TestResult]:
        """
        Test USB-PD power delivery compliance

        Args:
            power_levels: List of power levels to test (W)

        Returns:
            List of power delivery test results
        """
        if not self.config.power_delivery_testing:
            self.logger.warning("Power delivery testing disabled")
            return []

        self.logger.info(f"Testing USB-PD compliance with {len(power_levels)} power levels")

        results = []

        for power_level in power_levels:
            result = self._test_power_delivery_level(power_level)
            results.append(result)

        self.logger.info(
            f"Power delivery testing completed: "
            f"{len([r for r in results if r.result == TestResult.PASS])}/{len(results)} passed"
        )

        return results

    def measure_transition_times(
        self, transitions: List[Tuple[USB4LinkState, USB4LinkState]]
    ) -> Dict[Tuple[USB4LinkState, USB4LinkState], float]:
        """
        Measure power state transition times

        Args:
            transitions: List of state transitions to measure

        Returns:
            Dictionary of transition times
        """
        self.logger.info(f"Measuring transition times for {len(transitions)} transitions")

        transition_times = {}

        for from_state, to_state in transitions:
            # Ensure we're in the source state
            self._transition_to_state(from_state)
            time.sleep(0.1)  # Allow state to stabilize

            # Measure transition time
            start_time = time.time()
            success = self._transition_to_state(to_state)
            transition_time = time.time() - start_time

            if success:
                transition_times[(from_state, to_state)] = transition_time
                self.logger.debug(f"Transition {from_state.name}->{to_state.name}: {transition_time*1000:.1f}ms")
            else:
                self.logger.warning(f"Failed transition {from_state.name}->{to_state.name}")

        return transition_times

    def _reset_monitoring_data(self) -> None:
        """Reset all monitoring data"""
        self.power_measurements.clear()
        self.state_transitions.clear()
        self.wake_events.clear()
        self.thermal_events.clear()
        self.total_transitions = 0
        self.successful_transitions = 0
        self.wake_attempts = 0
        self.successful_wakes = 0

    def _take_power_measurement(self) -> USB4PowerMeasurement:
        """
        Take a power measurement

        Returns:
            Power measurement data
        """
        # Simulate power measurement (in real implementation, this would interface with power meter)
        base_power = self._get_expected_power(self.current_state)

        # Add measurement noise and thermal effects
        noise = np.random.normal(0, 0.02 * base_power)  # 2% noise
        thermal_factor = 1.0 + (self.current_temperature - 25.0) * 0.001  # 0.1%/°C

        measured_power = base_power * thermal_factor + noise
        measured_voltage = 5.0 + np.random.normal(0, 0.05)  # 5V ± 50mV
        measured_current = measured_power / measured_voltage

        return USB4PowerMeasurement(
            timestamp=time.time(),
            link_state=self.current_state,
            voltage=measured_voltage,
            current=measured_current,
            power=max(0.0, measured_power),
            temperature=self.current_temperature,
            thermal_state=self.current_thermal_state,
            throttling_active=self.current_thermal_state in [USB4ThermalState.THROTTLING, USB4ThermalState.CRITICAL],
        )

    def _get_expected_power(self, state: USB4LinkState) -> float:
        """
        Get expected power consumption for USB4 state

        Args:
            state: USB4 link state

        Returns:
            Expected power consumption in watts
        """
        power_map = {
            USB4LinkState.U0: self.specs.IDLE_POWER_U0,
            USB4LinkState.U1: self.specs.IDLE_POWER_U1,
            USB4LinkState.U2: self.specs.IDLE_POWER_U2,
            USB4LinkState.U3: self.specs.IDLE_POWER_U3,
        }
        return power_map[state]

    def _check_state_transitions(self) -> None:
        """Check for and handle power state transitions"""
        # Simulate occasional state transitions
        if np.random.random() < 0.02:  # 2% chance per measurement
            new_state = np.random.choice(list(USB4LinkState))
            if new_state != self.current_state:
                self._execute_state_transition(new_state, USB4PowerEvent.STATE_TRANSITION)

    def _execute_state_transition(self, new_state: USB4LinkState, event: USB4PowerEvent) -> None:
        """
        Execute power state transition

        Args:
            new_state: Target power state
            event: Power event causing transition
        """
        old_state = self.current_state
        start_time = time.time()

        # Measure power before transition
        power_before = self._take_power_measurement().power

        # Simulate transition time
        transition_time = self._get_transition_time(old_state, new_state)
        time.sleep(transition_time)

        # Update state
        self.current_state = new_state

        # Measure power after transition
        power_after = self._take_power_measurement().power

        # Record transition
        transition = USB4PowerTransition(
            from_state=old_state,
            to_state=new_state,
            event=event,
            timestamp=start_time,
            transition_time=transition_time,
            power_before=power_before,
            power_after=power_after,
            success=True,
        )

        self.state_transitions.append(transition)
        self.total_transitions += 1
        self.successful_transitions += 1

        self.logger.debug(f"State transition: {old_state.name} -> {new_state.name} " f"({transition_time*1000:.1f}ms)")

    def _get_transition_time(self, from_state: USB4LinkState, to_state: USB4LinkState) -> float:
        """
        Get expected transition time between states

        Args:
            from_state: Source state
            to_state: Target state

        Returns:
            Transition time in seconds
        """
        # Transition time matrix (simplified)
        transition_times = {
            (USB4LinkState.U0, USB4LinkState.U1): 0.001,  # 1ms
            (USB4LinkState.U0, USB4LinkState.U2): 0.010,  # 10ms
            (USB4LinkState.U0, USB4LinkState.U3): 0.100,  # 100ms
            (USB4LinkState.U1, USB4LinkState.U0): 0.001,  # 1ms
            (USB4LinkState.U1, USB4LinkState.U2): 0.005,  # 5ms
            (USB4LinkState.U1, USB4LinkState.U3): 0.050,  # 50ms
            (USB4LinkState.U2, USB4LinkState.U0): 0.010,  # 10ms
            (USB4LinkState.U2, USB4LinkState.U1): 0.005,  # 5ms
            (USB4LinkState.U2, USB4LinkState.U3): 0.020,  # 20ms
            (USB4LinkState.U3, USB4LinkState.U0): 0.100,  # 100ms
            (USB4LinkState.U3, USB4LinkState.U1): 0.050,  # 50ms
            (USB4LinkState.U3, USB4LinkState.U2): 0.020,  # 20ms
        }

        return transition_times.get((from_state, to_state), 0.001)

    def _transition_to_state(self, target_state: USB4LinkState) -> bool:
        """
        Transition to specific power state

        Args:
            target_state: Target power state

        Returns:
            True if transition successful
        """
        if self.current_state == target_state:
            return True

        try:
            self._execute_state_transition(target_state, USB4PowerEvent.STATE_TRANSITION)
            return True
        except Exception as e:
            self.logger.error(f"Failed to transition to {target_state.name}: {e}")
            return False

    def _monitor_thermal_conditions(self) -> None:
        """Monitor thermal conditions and handle thermal events"""
        # Simulate temperature variation
        temp_change = np.random.normal(0, 0.5)  # ±0.5°C variation
        self.current_temperature += temp_change

        # Determine thermal state
        new_thermal_state = self._determine_thermal_state(self.current_temperature)

        # Handle thermal state change
        if new_thermal_state != self.current_thermal_state:
            thermal_event = self._handle_thermal_transition(new_thermal_state)
            self.thermal_events.append(thermal_event)
            self.current_thermal_state = new_thermal_state

    def _determine_thermal_state(self, temperature: float) -> USB4ThermalState:
        """
        Determine thermal state based on temperature

        Args:
            temperature: Junction temperature in °C

        Returns:
            Thermal state
        """
        if temperature < 70.0:
            return USB4ThermalState.NORMAL
        elif temperature < 80.0:
            return USB4ThermalState.WARNING
        elif temperature < 90.0:
            return USB4ThermalState.THROTTLING
        elif temperature < 100.0:
            return USB4ThermalState.CRITICAL
        else:
            return USB4ThermalState.SHUTDOWN

    def _handle_thermal_transition(self, new_thermal_state: USB4ThermalState) -> USB4ThermalEvent:
        """
        Handle thermal state transition

        Args:
            new_thermal_state: New thermal state

        Returns:
            Thermal event record
        """
        throttling_level = 0.0
        performance_impact = 0.0
        mitigation_action = "None"

        if new_thermal_state == USB4ThermalState.WARNING:
            mitigation_action = "Temperature monitoring increased"
        elif new_thermal_state == USB4ThermalState.THROTTLING:
            throttling_level = 0.25  # 25% throttling
            performance_impact = 0.15  # 15% performance reduction
            mitigation_action = "Performance throttling activated"
        elif new_thermal_state == USB4ThermalState.CRITICAL:
            throttling_level = 0.50  # 50% throttling
            performance_impact = 0.35  # 35% performance reduction
            mitigation_action = "Critical throttling activated"
        elif new_thermal_state == USB4ThermalState.SHUTDOWN:
            throttling_level = 1.0  # Complete shutdown
            performance_impact = 1.0  # 100% performance loss
            mitigation_action = "Thermal shutdown initiated"

        recovery_time = throttling_level * 5.0  # Recovery time proportional to throttling

        return USB4ThermalEvent(
            timestamp=time.time(),
            temperature=self.current_temperature,
            thermal_state=new_thermal_state,
            throttling_level=throttling_level,
            performance_impact=performance_impact,
            recovery_time=recovery_time,
            mitigation_action=mitigation_action,
        )

    def _monitor_wake_events(self) -> None:
        """Monitor for wake events"""
        # Simulate occasional wake events
        if self.current_state in [USB4LinkState.U2, USB4LinkState.U3] and np.random.random() < 0.01:
            wake_source = np.random.choice(list(USB4WakeSource))
            wake_event = self._generate_wake_event(wake_source)
            self.wake_events.append(wake_event)

    def _generate_wake_event(self, wake_source: USB4WakeSource) -> USB4WakeEvent:
        """
        Generate and process wake event

        Args:
            wake_source: Source of wake event

        Returns:
            Wake event record
        """
        start_time = time.time()
        from_state = self.current_state

        self.wake_attempts += 1

        # Simulate wake time based on source state
        wake_time = self._get_wake_time(from_state, wake_source)
        time.sleep(wake_time)

        # Transition to active state
        to_state = USB4LinkState.U0
        success = self._transition_to_state(to_state)

        if success:
            self.successful_wakes += 1

        # Measure signal integrity after wake
        signal_integrity = np.random.uniform(0.8, 1.0) if success else np.random.uniform(0.3, 0.7)

        return USB4WakeEvent(
            timestamp=start_time,
            wake_source=wake_source,
            from_state=from_state,
            to_state=to_state,
            wake_time=wake_time,
            signal_integrity=signal_integrity,
            success=success,
            error_info=None if success else "Wake event failed",
        )

    def _get_wake_time(self, from_state: USB4LinkState, wake_source: USB4WakeSource) -> float:
        """
        Get expected wake time

        Args:
            from_state: State being woken from
            wake_source: Source of wake event

        Returns:
            Wake time in seconds
        """
        base_times = {
            USB4LinkState.U0: 0.0,  # Already awake
            USB4LinkState.U1: 0.001,  # 1ms
            USB4LinkState.U2: 0.010,  # 10ms
            USB4LinkState.U3: 0.100,  # 100ms
        }

        # Wake source multipliers
        source_multipliers = {
            USB4WakeSource.REMOTE_WAKE: 1.0,
            USB4WakeSource.LOCAL_WAKE: 0.8,
            USB4WakeSource.TIMER_WAKE: 1.2,
            USB4WakeSource.POWER_BUTTON: 0.5,
            USB4WakeSource.USB_DEVICE: 1.1,
            USB4WakeSource.SYSTEM_WAKE: 0.9,
        }

        base_time = base_times[from_state]
        multiplier = source_multipliers[wake_source]

        return base_time * multiplier

    def _test_power_delivery_level(self, power_level: float) -> USB4TestResult:
        """
        Test specific USB-PD power level

        Args:
            power_level: Power level to test (W)

        Returns:
            Test result
        """
        test_name = f"USB-PD {power_level}W"

        # Check if power level is within USB4 specifications
        if power_level > self.specs.MAX_POWER_DELIVERY:
            return USB4TestResult(
                test_name=test_name,
                result=TestResult.FAIL,
                measured_value=power_level,
                limit_value=self.specs.MAX_POWER_DELIVERY,
                units="W",
                message="Power level exceeds USB4 specification",
            )

        # Simulate power delivery negotiation
        negotiation_success = np.random.random() < 0.95  # 95% success rate

        if not negotiation_success:
            return USB4TestResult(test_name=test_name, result=TestResult.FAIL, message="Power delivery negotiation failed")

        # Simulate power delivery measurement
        measured_power = power_level + np.random.normal(0, 0.02 * power_level)  # 2% accuracy
        tolerance = power_level * self.config.voltage_tolerance

        power_accurate = abs(measured_power - power_level) <= tolerance

        return USB4TestResult(
            test_name=test_name,
            result=TestResult.PASS if power_accurate else TestResult.FAIL,
            measured_value=measured_power,
            limit_value=power_level,
            units="W",
            message=f"Power delivery {'accurate' if power_accurate else 'inaccurate'}",
        )

    def _calculate_power_results(self, duration: float) -> USB4PowerResults:
        """
        Calculate power monitoring results

        Args:
            duration: Monitoring duration

        Returns:
            Power monitoring results
        """
        # Calculate average power by state
        average_power_by_state = {}
        for state in USB4LinkState:
            state_measurements = [m for m in self.power_measurements if m.link_state == state]
            if state_measurements:
                average_power_by_state[state] = np.mean([m.power for m in state_measurements])

        # Calculate transition times
        transition_times = {}
        for transition in self.state_transitions:
            key = (transition.from_state, transition.to_state)
            if key not in transition_times:
                transition_times[key] = []
            transition_times[key].append(transition.transition_time)

        # Average transition times
        avg_transition_times = {k: np.mean(v) for k, v in transition_times.items()}

        # Calculate wake success rate
        wake_success_rate = self.successful_wakes / max(1, self.wake_attempts)

        # Check thermal compliance
        thermal_compliance = all(event.thermal_state != USB4ThermalState.SHUTDOWN for event in self.thermal_events)

        # Check power delivery compliance (simplified)
        power_delivery_compliance = all(
            measurement.power <= self.specs.MAX_POWER_DELIVERY for measurement in self.power_measurements
        )

        return USB4PowerResults(
            test_duration=duration,
            total_transitions=self.total_transitions,
            successful_transitions=self.successful_transitions,
            power_measurements=self.power_measurements.copy(),
            state_transitions=self.state_transitions.copy(),
            wake_events=self.wake_events.copy(),
            thermal_events=self.thermal_events.copy(),
            average_power_by_state=average_power_by_state,
            transition_times=avg_transition_times,
            wake_success_rate=wake_success_rate,
            thermal_compliance=thermal_compliance,
            power_delivery_compliance=power_delivery_compliance,
        )


__all__ = [
    # Enums
    "USB4PowerEvent",
    "USB4ThermalState",
    "USB4WakeSource",
    # Data structures
    "USB4PowerConfig",
    "USB4PowerTransition",
    "USB4PowerMeasurement",
    "USB4WakeEvent",
    "USB4ThermalEvent",
    "USB4PowerResults",
    # Main class
    "USB4PowerManager",
]
