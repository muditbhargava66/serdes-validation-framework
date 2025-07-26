"""
Tests for USB4 Link Recovery Testing

This module contains comprehensive tests for USB4 link recovery functionality,
including error injection, recovery timing validation, error logging and analysis,
and recovery mechanism effectiveness measurement.
"""

import time
from unittest.mock import patch

import pytest

from src.serdes_validation_framework.protocols.usb4.constants import (
    USB4ErrorType,
    USB4LinkState,
    USB4SignalMode,
)
from src.serdes_validation_framework.protocols.usb4.link_recovery import (
    USB4ErrorEvent,
    USB4ErrorInjectionConfig,
    USB4ErrorSeverity,
    USB4LinkRecoveryTester,
    USB4RecoveryAttempt,
    USB4RecoveryConfig,
    USB4RecoveryMethod,
    USB4RecoveryStatistics,
    USB4RecoveryStatus,
    USB4RecoveryTestResults,
)
from src.serdes_validation_framework.protocols.usb4.link_training import (
    USB4TrainingState,
)


class TestUSB4LinkRecoveryTester:
    """Test USB4 link recovery tester functionality"""

    @pytest.fixture
    def recovery_config(self):
        """Create test recovery configuration"""
        return USB4RecoveryConfig(
            signal_mode=USB4SignalMode.GEN2X2,
            sample_rate=50.0e9,
            capture_length=1000000,
            max_recovery_time=5.0,
            recovery_timeout=10.0,
            max_recovery_attempts=3,
            enable_statistics=True,
            enable_detailed_logging=True,
            test_all_error_types=True,
            measure_timing=True,
            validate_signal_quality=True,
            stress_test_duration=1.0,  # Short duration for testing
            error_injection_rate=1.0,
        )

    @pytest.fixture
    def recovery_tester(self, recovery_config):
        """Create test recovery tester"""
        tester = USB4LinkRecoveryTester(recovery_config)
        assert tester.initialize()
        return tester

    @pytest.fixture
    def error_injection_config(self):
        """Create test error injection configuration"""
        return USB4ErrorInjectionConfig(
            error_type=USB4ErrorType.LINK_TRAINING,
            severity=USB4ErrorSeverity.MEDIUM,
            duration=0.1,
            frequency=1.0,
            burst_count=1,
            recovery_timeout=5.0,
            enable_logging=True,
        )

    def test_recovery_tester_initialization(self, recovery_config):
        """Test recovery tester initialization"""
        tester = USB4LinkRecoveryTester(recovery_config)

        # Test initialization
        assert tester.initialize()
        assert tester.is_initialized
        assert tester.link_trainer.is_initialized

        # Test cleanup
        tester.cleanup()
        assert not tester.is_initialized

    def test_error_injection_and_recovery(self, recovery_tester, error_injection_config):
        """Test error injection and recovery process"""
        # Test error injection and recovery
        results = recovery_tester.test_error_injection_and_recovery(error_injection_config)

        # Verify results structure
        assert isinstance(results, USB4RecoveryTestResults)
        assert results.total_errors_tested == 1
        assert len(results.error_events) == 1
        assert len(results.recovery_attempts) >= 1
        assert results.test_duration > 0

        # Verify error event
        error_event = results.error_events[0]
        assert error_event.error_type == USB4ErrorType.LINK_TRAINING
        assert error_event.severity == USB4ErrorSeverity.MEDIUM
        assert error_event.recovery_triggered

        # Verify recovery attempts
        for attempt in results.recovery_attempts:
            assert isinstance(attempt, USB4RecoveryAttempt)
            assert attempt.duration > 0
            assert attempt.error_type == USB4ErrorType.LINK_TRAINING
            assert attempt.method in USB4RecoveryMethod
            assert attempt.status in USB4RecoveryStatus

    def test_comprehensive_recovery_testing(self, recovery_tester):
        """Test comprehensive recovery testing across all error types"""
        # Reduce stress test duration for faster testing
        recovery_tester.config.stress_test_duration = 0.1

        results = recovery_tester.test_comprehensive_recovery()

        # Verify comprehensive results
        assert isinstance(results, USB4RecoveryTestResults)
        assert results.total_errors_tested > 1  # Should test multiple error types
        assert len(results.error_events) > 1
        assert len(results.recovery_attempts) >= len(results.error_events)

        # Verify error type coverage
        error_types_tested = set(event.error_type for event in results.error_events)
        assert len(error_types_tested) > 1  # Should test multiple error types

        # Verify statistics
        stats = results.recovery_statistics
        assert stats.total_errors_injected > 0
        assert stats.total_recovery_attempts > 0
        assert 0 <= stats.recovery_success_rate <= 1.0

    def test_recovery_timing_validation(self, recovery_tester):
        """Test recovery timing validation"""
        # Create mock recovery attempts with different timings
        recovery_attempts = [
            USB4RecoveryAttempt(
                attempt_number=1,
                start_time=time.time(),
                end_time=time.time() + 1.0,
                method=USB4RecoveryMethod.LINK_RETRAIN,
                status=USB4RecoveryStatus.SUCCESS,
                duration=1.0,
                error_type=USB4ErrorType.LINK_TRAINING,
                initial_state=USB4TrainingState.RECOVERY,
                final_state=USB4TrainingState.L0,
            ),
            USB4RecoveryAttempt(
                attempt_number=1,
                start_time=time.time(),
                end_time=time.time() + 2.0,
                method=USB4RecoveryMethod.SPEED_DOWNGRADE,
                status=USB4RecoveryStatus.SUCCESS,
                duration=2.0,
                error_type=USB4ErrorType.SIGNAL_INTEGRITY,
                initial_state=USB4TrainingState.RECOVERY,
                final_state=USB4TrainingState.L0,
            ),
            USB4RecoveryAttempt(
                attempt_number=1,
                start_time=time.time(),
                end_time=time.time() + 6.0,  # Exceeds max recovery time
                method=USB4RecoveryMethod.POWER_CYCLE,
                status=USB4RecoveryStatus.TIMEOUT,
                duration=6.0,
                error_type=USB4ErrorType.PROTOCOL,
                initial_state=USB4TrainingState.RECOVERY,
                final_state=USB4TrainingState.RECOVERY,
            ),
        ]

        timing_results = recovery_tester.validate_recovery_timing(recovery_attempts)

        # Verify timing validation results
        assert "average_recovery_time" in timing_results
        assert "min_recovery_time" in timing_results
        assert "max_recovery_time" in timing_results
        assert "timing_compliance" in timing_results
        assert "timing_violations" in timing_results
        assert "timing_violation_rate" in timing_results

        # Verify calculations
        assert timing_results["average_recovery_time"] == 3.0  # (1+2+6)/3
        assert timing_results["min_recovery_time"] == 1.0
        assert timing_results["max_recovery_time"] == 6.0
        assert not timing_results["timing_compliance"]  # Should fail due to 6s attempt
        assert timing_results["timing_violations"] == 1
        assert timing_results["timing_violation_rate"] == 1 / 3

    def test_recovery_effectiveness_measurement(self, recovery_tester):
        """Test recovery effectiveness measurement"""
        # Create mock recovery attempts with different outcomes
        recovery_attempts = [
            USB4RecoveryAttempt(
                attempt_number=1,
                start_time=time.time(),
                end_time=time.time() + 1.0,
                method=USB4RecoveryMethod.LINK_RETRAIN,
                status=USB4RecoveryStatus.SUCCESS,
                duration=1.0,
                error_type=USB4ErrorType.LINK_TRAINING,
                initial_state=USB4TrainingState.RECOVERY,
                final_state=USB4TrainingState.L0,
                signal_quality_improvement=0.2,
                bandwidth_after_recovery=40.0e9,
                power_consumption_change=-0.1,
            ),
            USB4RecoveryAttempt(
                attempt_number=1,
                start_time=time.time(),
                end_time=time.time() + 2.0,
                method=USB4RecoveryMethod.SPEED_DOWNGRADE,
                status=USB4RecoveryStatus.SUCCESS,
                duration=2.0,
                error_type=USB4ErrorType.SIGNAL_INTEGRITY,
                initial_state=USB4TrainingState.RECOVERY,
                final_state=USB4TrainingState.L0,
                signal_quality_improvement=0.1,
                bandwidth_after_recovery=30.0e9,
                power_consumption_change=0.0,
            ),
            USB4RecoveryAttempt(
                attempt_number=1,
                start_time=time.time(),
                end_time=time.time() + 3.0,
                method=USB4RecoveryMethod.POWER_CYCLE,
                status=USB4RecoveryStatus.FAILURE,
                duration=3.0,
                error_type=USB4ErrorType.PROTOCOL,
                initial_state=USB4TrainingState.RECOVERY,
                final_state=USB4TrainingState.RECOVERY,
                signal_quality_improvement=0.0,
                bandwidth_after_recovery=0.0,
                power_consumption_change=0.2,
            ),
        ]

        effectiveness_metrics = recovery_tester.measure_recovery_effectiveness(recovery_attempts)

        # Verify effectiveness metrics
        assert "overall_success_rate" in effectiveness_metrics
        assert "success_by_error_type" in effectiveness_metrics
        assert "success_by_method" in effectiveness_metrics
        assert "first_attempt_success_rate" in effectiveness_metrics

        # Verify calculations
        assert effectiveness_metrics["overall_success_rate"] == 2 / 3  # 2 successes out of 3
        assert effectiveness_metrics["first_attempt_success_rate"] == 2 / 3  # All are first attempts

        # Verify quality improvement metrics
        assert "average_quality_improvement" in effectiveness_metrics
        assert effectiveness_metrics["average_quality_improvement"] == 0.15  # (0.2+0.1)/2

    def test_error_injection_configuration(self, recovery_tester):
        """Test different error injection configurations"""
        # Test different error types
        for error_type in USB4ErrorType:
            config = USB4ErrorInjectionConfig(
                error_type=error_type, severity=USB4ErrorSeverity.LOW, duration=0.05, recovery_timeout=2.0
            )

            results = recovery_tester.test_error_injection_and_recovery(config)
            assert results.error_events[0].error_type == error_type

    def test_recovery_method_selection(self, recovery_tester):
        """Test recovery method selection logic"""
        # Test method selection for different error types
        test_cases = [
            (USB4ErrorType.SIGNAL_INTEGRITY, USB4RecoveryMethod.EQUALIZATION_ADJUST),
            (USB4ErrorType.LINK_TRAINING, USB4RecoveryMethod.LINK_RETRAIN),
            (USB4ErrorType.PROTOCOL, USB4RecoveryMethod.PROTOCOL_RESET),
            (USB4ErrorType.POWER_MANAGEMENT, USB4RecoveryMethod.POWER_CYCLE),
        ]

        for error_type, expected_first_method in test_cases:
            selected_method = recovery_tester._select_recovery_method(error_type, 1)
            assert selected_method == expected_first_method

    def test_recovery_statistics_calculation(self, recovery_tester):
        """Test recovery statistics calculation"""
        # Inject some errors and perform recoveries
        for error_type in [USB4ErrorType.LINK_TRAINING, USB4ErrorType.SIGNAL_INTEGRITY]:
            config = USB4ErrorInjectionConfig(
                error_type=error_type, severity=USB4ErrorSeverity.MEDIUM, duration=0.05, recovery_timeout=2.0
            )
            recovery_tester.test_error_injection_and_recovery(config)

        # Calculate statistics
        stats = recovery_tester._calculate_current_statistics()

        # Verify statistics
        assert isinstance(stats, USB4RecoveryStatistics)
        assert stats.total_errors_injected >= 2
        assert stats.total_recovery_attempts >= 2
        assert 0 <= stats.recovery_success_rate <= 1.0
        assert stats.average_recovery_time > 0

    def test_signal_quality_measurement(self, recovery_tester):
        """Test signal quality measurement"""
        # Test signal quality measurement
        quality = recovery_tester._measure_signal_quality()
        assert 0.0 <= quality <= 1.0

        # Test multiple measurements for consistency
        qualities = [recovery_tester._measure_signal_quality() for _ in range(10)]
        assert all(0.0 <= q <= 1.0 for q in qualities)
        assert len(set(qualities)) > 1  # Should have some variation

    def test_bandwidth_measurement(self, recovery_tester):
        """Test bandwidth measurement after recovery"""
        # Test bandwidth measurement
        bandwidth = recovery_tester._measure_bandwidth_after_recovery()
        assert bandwidth > 0

        # Test with different link states
        original_state = recovery_tester.link_trainer.current_link_state

        # Test U0 state (should have full bandwidth)
        recovery_tester.link_trainer.current_link_state = USB4LinkState.U0
        bandwidth_u0 = recovery_tester._measure_bandwidth_after_recovery()
        assert bandwidth_u0 > 0

        # Test U3 state (should have reduced bandwidth)
        recovery_tester.link_trainer.current_link_state = USB4LinkState.U3
        bandwidth_u3 = recovery_tester._measure_bandwidth_after_recovery()
        assert bandwidth_u3 < bandwidth_u0

        # Restore original state
        recovery_tester.link_trainer.current_link_state = original_state

    def test_power_consumption_measurement(self, recovery_tester):
        """Test power consumption change measurement"""
        # Test power consumption measurement
        power_change = recovery_tester._measure_power_consumption_change()
        assert isinstance(power_change, float)

        # Test with different link states
        original_state = recovery_tester.link_trainer.current_link_state

        for state in USB4LinkState:
            recovery_tester.link_trainer.current_link_state = state
            power_change = recovery_tester._measure_power_consumption_change()
            assert isinstance(power_change, float)

        # Restore original state
        recovery_tester.link_trainer.current_link_state = original_state

    def test_recovery_report_generation(self, recovery_tester, error_injection_config):
        """Test recovery analysis report generation"""
        # Perform a test to generate results
        results = recovery_tester.test_error_injection_and_recovery(error_injection_config)

        # Generate report
        report = recovery_tester.generate_recovery_analysis_report(results)

        # Verify report content
        assert isinstance(report, str)
        assert len(report) > 0
        assert "USB4 Link Recovery Analysis Report" in report
        assert "Test Summary:" in report
        assert "Recovery Statistics:" in report
        assert f"Total Errors Tested: {results.total_errors_tested}" in report

    def test_error_severity_handling(self, recovery_tester):
        """Test handling of different error severities"""
        for severity in USB4ErrorSeverity:
            config = USB4ErrorInjectionConfig(
                error_type=USB4ErrorType.LINK_TRAINING, severity=severity, duration=0.05, recovery_timeout=2.0
            )

            results = recovery_tester.test_error_injection_and_recovery(config)
            assert results.error_events[0].severity == severity

    def test_recovery_timeout_handling(self, recovery_tester):
        """Test recovery timeout handling"""
        # Create config with very short timeout
        config = USB4ErrorInjectionConfig(
            error_type=USB4ErrorType.LINK_TRAINING,
            severity=USB4ErrorSeverity.HIGH,
            duration=0.05,
            recovery_timeout=0.01,  # Very short timeout
        )

        # Mock recovery methods to take longer than timeout
        with patch.object(recovery_tester, "_execute_recovery_method") as mock_recovery:
            mock_recovery.return_value = USB4RecoveryStatus.TIMEOUT

            results = recovery_tester.test_error_injection_and_recovery(config)

            # Should have timeout status
            timeout_attempts = [a for a in results.recovery_attempts if a.status == USB4RecoveryStatus.TIMEOUT]
            assert len(timeout_attempts) > 0

    def test_multiple_recovery_attempts(self, recovery_tester):
        """Test multiple recovery attempts for persistent errors"""
        config = USB4ErrorInjectionConfig(
            error_type=USB4ErrorType.PROTOCOL, severity=USB4ErrorSeverity.HIGH, duration=0.05, recovery_timeout=5.0
        )

        # Mock first few recovery attempts to fail
        original_method = recovery_tester._execute_recovery_method
        call_count = 0

        def mock_recovery_method(method, error_event):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # First two attempts fail
                return USB4RecoveryStatus.FAILURE
            else:  # Third attempt succeeds
                return USB4RecoveryStatus.SUCCESS

        with patch.object(recovery_tester, "_execute_recovery_method", side_effect=mock_recovery_method):
            results = recovery_tester.test_error_injection_and_recovery(config)

            # Should have multiple attempts
            assert len(results.recovery_attempts) >= 2

            # Last attempt should be successful
            assert results.recovery_attempts[-1].status == USB4RecoveryStatus.SUCCESS

    def test_stress_testing(self, recovery_tester):
        """Test stress testing functionality"""
        # Set very short stress test duration
        recovery_tester.config.stress_test_duration = 0.1
        recovery_tester.config.error_injection_rate = 10.0  # High rate for quick testing

        stress_results = recovery_tester._perform_stress_testing()

        # Verify stress test results
        assert "error_events" in stress_results
        assert "recovery_attempts" in stress_results
        assert len(stress_results["error_events"]) > 0
        assert len(stress_results["recovery_attempts"]) >= len(stress_results["error_events"])

    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        valid_config = USB4RecoveryConfig(
            signal_mode=USB4SignalMode.GEN2X2,
            sample_rate=50.0e9,
            capture_length=1000000,
            max_recovery_time=5.0,
            recovery_timeout=10.0,
            max_recovery_attempts=3,
        )
        tester = USB4LinkRecoveryTester(valid_config)
        assert tester.initialize()
        tester.cleanup()

        # Test configuration with invalid values
        invalid_config = USB4RecoveryConfig(
            signal_mode=USB4SignalMode.GEN2X2,
            sample_rate=50.0e9,
            capture_length=1000000,
            max_recovery_time=-1.0,  # Invalid negative time
            recovery_timeout=10.0,
            max_recovery_attempts=0,  # Invalid zero attempts
        )
        tester = USB4LinkRecoveryTester(invalid_config)
        # Should still initialize but may have issues during operation
        assert tester.initialize()
        tester.cleanup()


class TestUSB4RecoveryDataStructures:
    """Test USB4 recovery data structures"""

    def test_error_injection_config(self):
        """Test error injection configuration"""
        config = USB4ErrorInjectionConfig(
            error_type=USB4ErrorType.LINK_TRAINING,
            severity=USB4ErrorSeverity.MEDIUM,
            duration=0.1,
            frequency=1.0,
            burst_count=1,
            recovery_timeout=5.0,
        )

        assert config.error_type == USB4ErrorType.LINK_TRAINING
        assert config.severity == USB4ErrorSeverity.MEDIUM
        assert config.duration == 0.1
        assert config.recovery_timeout == 5.0

    def test_recovery_attempt(self):
        """Test recovery attempt data structure"""
        attempt = USB4RecoveryAttempt(
            attempt_number=1,
            start_time=time.time(),
            end_time=time.time() + 1.0,
            method=USB4RecoveryMethod.LINK_RETRAIN,
            status=USB4RecoveryStatus.SUCCESS,
            duration=1.0,
            error_type=USB4ErrorType.LINK_TRAINING,
            initial_state=USB4TrainingState.RECOVERY,
            final_state=USB4TrainingState.L0,
        )

        assert attempt.attempt_number == 1
        assert attempt.method == USB4RecoveryMethod.LINK_RETRAIN
        assert attempt.status == USB4RecoveryStatus.SUCCESS
        assert attempt.duration == 1.0
        assert attempt.error_type == USB4ErrorType.LINK_TRAINING

    def test_recovery_statistics(self):
        """Test recovery statistics data structure"""
        stats = USB4RecoveryStatistics()

        # Test default values
        assert stats.total_errors_injected == 0
        assert stats.total_recovery_attempts == 0
        assert stats.successful_recoveries == 0
        assert stats.recovery_success_rate == 0.0
        assert stats.min_recovery_time == float("inf")
        assert stats.max_recovery_time == 0.0

    def test_error_event(self):
        """Test error event data structure"""
        event = USB4ErrorEvent(
            timestamp=time.time(),
            error_type=USB4ErrorType.SIGNAL_INTEGRITY,
            severity=USB4ErrorSeverity.HIGH,
            description="Test error event",
            recovery_triggered=True,
        )

        assert event.error_type == USB4ErrorType.SIGNAL_INTEGRITY
        assert event.severity == USB4ErrorSeverity.HIGH
        assert event.recovery_triggered
        assert "Test error event" in event.description


if __name__ == "__main__":
    pytest.main([__file__])
