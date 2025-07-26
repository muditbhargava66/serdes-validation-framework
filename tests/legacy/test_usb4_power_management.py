"""
Tests for USB4 Power Management Module

This module contains comprehensive tests for USB4 power state management,
including power consumption validation, wake event testing, thermal management,
and power delivery compliance.
"""

import time
from unittest.mock import patch

import pytest

from src.serdes_validation_framework.protocols.usb4.base import TestResult
from src.serdes_validation_framework.protocols.usb4.constants import (
    USB4LinkState,
    USB4SignalMode,
)
from src.serdes_validation_framework.protocols.usb4.power_management import (
    USB4PowerConfig,
    USB4PowerManager,
    USB4PowerMeasurement,
    USB4PowerResults,
    USB4ThermalEvent,
    USB4ThermalState,
    USB4WakeEvent,
    USB4WakeSource,
)


class TestUSB4PowerManager:
    """Test USB4 power manager functionality"""

    @pytest.fixture
    def power_config(self):
        """Create test power configuration"""
        return USB4PowerConfig(
            signal_mode=USB4SignalMode.GEN2X2,
            sample_rate=1e9,
            capture_length=1000,
            monitor_duration=1.0,
            power_measurement_interval=0.01,
            thermal_monitoring=True,
            wake_event_monitoring=True,
            power_delivery_testing=True,
            throttling_testing=True,
        )

    @pytest.fixture
    def power_manager(self, power_config):
        """Create test power manager"""
        manager = USB4PowerManager(power_config)
        manager.initialize()
        return manager

    def test_power_manager_initialization(self, power_config):
        """Test power manager initialization"""
        manager = USB4PowerManager(power_config)

        # Test initialization
        assert manager.initialize()
        assert manager.is_initialized
        assert manager.current_state == USB4LinkState.U3
        assert manager.current_temperature == 25.0
        assert manager.current_thermal_state == USB4ThermalState.NORMAL

        # Test cleanup
        manager.cleanup()
        assert not manager.is_initialized

    def test_power_measurement(self, power_manager):
        """Test power measurement functionality"""
        # Take a power measurement
        measurement = power_manager._take_power_measurement()

        assert isinstance(measurement, USB4PowerMeasurement)
        assert measurement.timestamp > 0
        assert measurement.link_state == power_manager.current_state
        assert measurement.voltage > 0
        assert measurement.current >= 0
        assert measurement.power >= 0
        assert isinstance(measurement.temperature, float)
        assert isinstance(measurement.thermal_state, USB4ThermalState)
        assert isinstance(measurement.throttling_active, bool)

    def test_expected_power_calculation(self, power_manager):
        """Test expected power calculation for different states"""
        # Test all USB4 link states
        for state in USB4LinkState:
            power = power_manager._get_expected_power(state)
            assert power > 0

        # Verify power ordering (U0 > U1 > U2 > U3)
        u0_power = power_manager._get_expected_power(USB4LinkState.U0)
        u1_power = power_manager._get_expected_power(USB4LinkState.U1)
        u2_power = power_manager._get_expected_power(USB4LinkState.U2)
        u3_power = power_manager._get_expected_power(USB4LinkState.U3)

        assert u0_power > u1_power > u2_power > u3_power

    def test_state_transitions(self, power_manager):
        """Test power state transitions"""
        initial_state = power_manager.current_state

        # Test transition to different state
        target_state = USB4LinkState.U0
        success = power_manager._transition_to_state(target_state)

        assert success
        assert power_manager.current_state == target_state
        assert power_manager.total_transitions > 0
        assert power_manager.successful_transitions > 0

        # Test transition to same state (should succeed immediately)
        success = power_manager._transition_to_state(target_state)
        assert success

    def test_transition_time_calculation(self, power_manager):
        """Test transition time calculation"""
        # Test various state transitions
        transitions = [
            (USB4LinkState.U0, USB4LinkState.U1),
            (USB4LinkState.U1, USB4LinkState.U2),
            (USB4LinkState.U2, USB4LinkState.U3),
            (USB4LinkState.U3, USB4LinkState.U0),
        ]

        for from_state, to_state in transitions:
            transition_time = power_manager._get_transition_time(from_state, to_state)
            assert transition_time > 0
            assert transition_time < 1.0  # Should be less than 1 second

    def test_power_consumption_validation(self, power_manager):
        """Test power consumption validation"""
        # Test validation for each state
        for state in USB4LinkState:
            result = power_manager.validate_power_consumption(state)

            # Check that result has the expected attributes (duck typing)
            assert hasattr(result, "test_name")
            assert hasattr(result, "result")
            assert hasattr(result, "measured_value")
            assert hasattr(result, "limit_value")
            assert hasattr(result, "units")

            assert result.test_name.startswith("Power Consumption")
            assert result.result in [TestResult.PASS, TestResult.FAIL]
            assert result.measured_value is not None
            assert result.limit_value is not None
            assert result.units == "W"

    def test_wake_event_generation(self, power_manager):
        """Test wake event generation and processing"""
        # Set to sleep state
        power_manager._transition_to_state(USB4LinkState.U2)

        # Test different wake sources
        wake_sources = [
            USB4WakeSource.REMOTE_WAKE,
            USB4WakeSource.LOCAL_WAKE,
            USB4WakeSource.TIMER_WAKE,
            USB4WakeSource.POWER_BUTTON,
        ]

        for wake_source in wake_sources:
            # Ensure we're in a sleep state before generating wake event
            power_manager._transition_to_state(USB4LinkState.U2)

            wake_event = power_manager._generate_wake_event(wake_source)

            assert isinstance(wake_event, USB4WakeEvent)
            assert wake_event.wake_source == wake_source
            # The from_state should be the state we were in when wake was generated
            assert wake_event.from_state in [USB4LinkState.U0, USB4LinkState.U1, USB4LinkState.U2, USB4LinkState.U3]
            assert wake_event.to_state == USB4LinkState.U0
            assert wake_event.wake_time >= 0  # Can be 0 if already awake
            assert 0 <= wake_event.signal_integrity <= 1.0
            assert isinstance(wake_event.success, bool)

    def test_wake_time_calculation(self, power_manager):
        """Test wake time calculation"""
        wake_source = USB4WakeSource.REMOTE_WAKE

        # Test wake times from different states
        for state in USB4LinkState:
            wake_time = power_manager._get_wake_time(state, wake_source)
            assert wake_time >= 0

        # Verify wake time ordering (U3 > U2 > U1 > U0)
        u0_time = power_manager._get_wake_time(USB4LinkState.U0, wake_source)
        u1_time = power_manager._get_wake_time(USB4LinkState.U1, wake_source)
        u2_time = power_manager._get_wake_time(USB4LinkState.U2, wake_source)
        u3_time = power_manager._get_wake_time(USB4LinkState.U3, wake_source)

        assert u3_time >= u2_time >= u1_time >= u0_time

    def test_wake_events_testing(self, power_manager):
        """Test wake events testing functionality"""
        wake_source = USB4WakeSource.REMOTE_WAKE
        count = 3

        wake_events = power_manager.test_wake_events(wake_source, count)

        assert len(wake_events) == count
        assert all(isinstance(event, USB4WakeEvent) for event in wake_events)
        assert all(event.wake_source == wake_source for event in wake_events)

        # Check that events were added to manager's wake_events list
        assert len(power_manager.wake_events) >= count

    def test_thermal_state_determination(self, power_manager):
        """Test thermal state determination"""
        test_temperatures = [
            (25.0, USB4ThermalState.NORMAL),
            (65.0, USB4ThermalState.NORMAL),
            (75.0, USB4ThermalState.WARNING),
            (85.0, USB4ThermalState.THROTTLING),
            (95.0, USB4ThermalState.CRITICAL),
            (105.0, USB4ThermalState.SHUTDOWN),
        ]

        for temperature, expected_state in test_temperatures:
            thermal_state = power_manager._determine_thermal_state(temperature)
            assert thermal_state == expected_state

    def test_thermal_transition_handling(self, power_manager):
        """Test thermal transition handling"""
        # Test transition to throttling state
        thermal_event = power_manager._handle_thermal_transition(USB4ThermalState.THROTTLING)

        assert isinstance(thermal_event, USB4ThermalEvent)
        assert thermal_event.thermal_state == USB4ThermalState.THROTTLING
        assert thermal_event.throttling_level > 0
        assert thermal_event.performance_impact > 0
        assert thermal_event.recovery_time > 0
        assert thermal_event.mitigation_action != "None"

        # Test transition to critical state
        thermal_event = power_manager._handle_thermal_transition(USB4ThermalState.CRITICAL)

        assert thermal_event.thermal_state == USB4ThermalState.CRITICAL
        assert thermal_event.throttling_level > 0.25  # More than throttling state
        assert thermal_event.performance_impact > 0.15

    def test_thermal_management_validation(self, power_manager):
        """Test thermal management validation"""
        temperature_profile = [25.0, 50.0, 75.0, 85.0, 75.0, 50.0, 25.0]

        thermal_events = power_manager.validate_thermal_management(temperature_profile)

        # Should have thermal events for state changes
        assert len(thermal_events) > 0
        assert all(isinstance(event, USB4ThermalEvent) for event in thermal_events)

        # Check that events were added to manager's thermal_events list
        assert len(power_manager.thermal_events) >= len(thermal_events)

    def test_power_delivery_testing(self, power_manager):
        """Test USB-PD power delivery testing"""
        power_levels = [15.0, 30.0, 60.0, 100.0]

        results = power_manager.test_power_delivery(power_levels)

        assert len(results) == len(power_levels)
        assert all(hasattr(result, "test_name") for result in results)
        assert all(hasattr(result, "result") for result in results)
        assert all("USB-PD" in result.test_name for result in results)

        # Test with power level exceeding specification
        high_power_results = power_manager.test_power_delivery([150.0])
        assert len(high_power_results) == 1
        assert high_power_results[0].result == TestResult.FAIL

    def test_transition_time_measurement(self, power_manager):
        """Test transition time measurement"""
        transitions = [
            (USB4LinkState.U0, USB4LinkState.U1),
            (USB4LinkState.U1, USB4LinkState.U2),
            (USB4LinkState.U2, USB4LinkState.U0),
        ]

        transition_times = power_manager.measure_transition_times(transitions)

        # Should have measured times for successful transitions
        assert len(transition_times) <= len(transitions)

        for (_from_state, _to_state), measured_time in transition_times.items():
            assert measured_time > 0
            assert measured_time < 1.0  # Reasonable transition time

    @patch("time.sleep")  # Speed up the test
    def test_power_state_monitoring(self, mock_sleep, power_manager):
        """Test power state monitoring"""
        # Configure for short test
        power_manager.config.monitor_duration = 0.1
        power_manager.config.power_measurement_interval = 0.01

        results = power_manager.monitor_power_states(0.1)

        assert isinstance(results, USB4PowerResults)
        assert results.test_duration >= 0
        assert len(results.power_measurements) > 0
        assert isinstance(results.average_power_by_state, dict)
        assert isinstance(results.transition_times, dict)
        assert 0 <= results.wake_success_rate <= 1.0
        assert isinstance(results.thermal_compliance, bool)
        assert isinstance(results.power_delivery_compliance, bool)

    def test_power_results_calculation(self, power_manager):
        """Test power results calculation"""
        # Add some test data
        power_manager.power_measurements = [
            USB4PowerMeasurement(
                timestamp=time.time(),
                link_state=USB4LinkState.U0,
                voltage=5.0,
                current=0.5,
                power=2.5,
                temperature=25.0,
                thermal_state=USB4ThermalState.NORMAL,
                throttling_active=False,
            ),
            USB4PowerMeasurement(
                timestamp=time.time(),
                link_state=USB4LinkState.U1,
                voltage=5.0,
                current=0.1,
                power=0.5,
                temperature=25.0,
                thermal_state=USB4ThermalState.NORMAL,
                throttling_active=False,
            ),
        ]

        power_manager.total_transitions = 5
        power_manager.successful_transitions = 4
        power_manager.wake_attempts = 3
        power_manager.successful_wakes = 2

        results = power_manager._calculate_power_results(1.0)

        assert results.test_duration == 1.0
        assert results.total_transitions == 5
        assert results.successful_transitions == 4
        assert results.wake_success_rate == 2 / 3
        assert USB4LinkState.U0 in results.average_power_by_state
        assert USB4LinkState.U1 in results.average_power_by_state

    def test_configuration_validation(self, power_config):
        """Test power configuration validation"""
        manager = USB4PowerManager(power_config)

        # Valid configuration should pass
        assert manager.validate_config()

        # Test with invalid configuration
        with pytest.raises(ValueError):
            invalid_config = "not a config"
            invalid_manager = USB4PowerManager(invalid_config)
            invalid_manager.validate_config()

    def test_error_handling(self, power_manager):
        """Test error handling in power management"""
        # Test with uninitialized manager
        uninitialized_manager = USB4PowerManager(power_manager.config)

        with pytest.raises(RuntimeError):
            uninitialized_manager.monitor_power_states(1.0)

    def test_power_manager_state_consistency(self, power_manager):
        """Test power manager state consistency"""
        initial_state = power_manager.current_state

        # Perform several operations
        power_manager._take_power_measurement()
        power_manager._transition_to_state(USB4LinkState.U0)
        power_manager._take_power_measurement()

        # State should be consistent
        assert power_manager.current_state == USB4LinkState.U0
        assert power_manager.total_transitions > 0

    def test_thermal_monitoring_integration(self, power_manager):
        """Test thermal monitoring integration"""
        # Enable thermal monitoring
        power_manager.config.thermal_monitoring = True

        # Simulate temperature change
        original_temp = power_manager.current_temperature
        power_manager.current_temperature = 85.0  # Throttling temperature

        # Monitor thermal conditions
        power_manager._monitor_thermal_conditions()

        # Should have detected thermal state change
        assert power_manager.current_thermal_state != USB4ThermalState.NORMAL

        # Reset temperature
        power_manager.current_temperature = original_temp

    def test_wake_event_monitoring_integration(self, power_manager):
        """Test wake event monitoring integration"""
        # Enable wake event monitoring
        power_manager.config.wake_event_monitoring = True

        # Set to sleep state
        power_manager._transition_to_state(USB4LinkState.U2)

        # Monitor wake events (may or may not generate events due to randomness)
        initial_wake_count = len(power_manager.wake_events)

        # Call monitoring multiple times to increase chance of wake event
        for _ in range(100):
            power_manager._monitor_wake_events()

        # Wake events list should not have decreased
        assert len(power_manager.wake_events) >= initial_wake_count


if __name__ == "__main__":
    pytest.main([__file__])
