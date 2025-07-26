"""
USB4 Link Recovery Testing

This module implements comprehensive USB4 link recovery testing capabilities,
including error injection, recovery timing validation, error logging and analysis,
and recovery mechanism effectiveness measurement.

Features:
- Error injection and recovery testing methods
- Link recovery timing validation
- Comprehensive error logging and analysis
- Recovery mechanism effectiveness measurement
- Statistical analysis of recovery performance
- Detailed recovery diagnostics and reporting
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np

from .base import USB4Component, USB4Config
from .constants import (
    USB4ErrorType,
    USB4LinkState,
    USB4SignalMode,
    USB4Specs,
)
from .link_training import (
    USB4LinkTraining,
    USB4TrainingConfig,
    USB4TrainingState,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USB4RecoveryMethod(Enum):
    """USB4 link recovery methods"""

    LINK_RETRAIN = auto()  # Full link retraining
    SPEED_DOWNGRADE = auto()  # Downgrade to lower speed
    LANE_DISABLE = auto()  # Disable faulty lane
    POWER_CYCLE = auto()  # Power cycle recovery
    PROTOCOL_RESET = auto()  # Protocol layer reset
    EQUALIZATION_ADJUST = auto()  # Adjust signal equalization
    CLOCK_RECOVERY = auto()  # Clock and data recovery reset


class USB4ErrorSeverity(Enum):
    """USB4 error severity levels"""

    LOW = auto()  # Minor errors, quick recovery expected
    MEDIUM = auto()  # Moderate errors, may require retries
    HIGH = auto()  # Severe errors, extensive recovery needed
    CRITICAL = auto()  # Critical errors, may require power cycle


class USB4RecoveryStatus(Enum):
    """USB4 recovery attempt status"""

    SUCCESS = auto()  # Recovery successful
    PARTIAL_SUCCESS = auto()  # Partial recovery achieved
    FAILURE = auto()  # Recovery failed
    TIMEOUT = auto()  # Recovery timed out
    DEGRADED = auto()  # Recovered with degraded performance


@dataclass
class USB4ErrorInjectionConfig:
    """Configuration for USB4 error injection"""

    error_type: USB4ErrorType
    severity: USB4ErrorSeverity = USB4ErrorSeverity.MEDIUM
    duration: float = 0.1  # Error duration (s)
    frequency: float = 1.0  # Error injection frequency (Hz)
    burst_count: int = 1  # Number of errors in burst
    burst_interval: float = 0.01  # Interval between burst errors (s)
    target_lane: Optional[int] = None  # Target lane for injection (0 or 1)
    recovery_timeout: float = 5.0  # Maximum recovery time (s)
    enable_logging: bool = True  # Enable detailed error logging


@dataclass
class USB4RecoveryConfig(USB4Config):
    """USB4 link recovery testing configuration"""

    max_recovery_time: float = 10.0  # Maximum recovery time (s)
    recovery_timeout: float = 30.0  # Overall recovery timeout (s)
    max_recovery_attempts: int = 5  # Maximum recovery attempts per error
    enable_statistics: bool = True  # Enable recovery statistics
    enable_detailed_logging: bool = True  # Enable detailed recovery logging
    test_all_error_types: bool = True  # Test all error types
    measure_timing: bool = True  # Measure recovery timing
    validate_signal_quality: bool = True  # Validate signal quality after recovery
    stress_test_duration: float = 60.0  # Stress test duration (s)
    error_injection_rate: float = 0.1  # Errors per second for stress testing

    def __post_init__(self):
        """Set default values for base USB4Config if not provided"""
        if not hasattr(self, "signal_mode") or self.signal_mode is None:
            self.signal_mode = USB4SignalMode.GEN2X2
        if not hasattr(self, "sample_rate") or self.sample_rate is None:
            self.sample_rate = 50.0e9  # 50 GSa/s
        if not hasattr(self, "capture_length") or self.capture_length is None:
            self.capture_length = 1000000  # 1M samples


@dataclass
class USB4ErrorEvent:
    """USB4 error event record"""

    timestamp: float
    error_type: USB4ErrorType
    severity: USB4ErrorSeverity
    lane: Optional[int] = None
    description: str = ""
    signal_quality_before: Optional[float] = None
    signal_quality_after: Optional[float] = None
    recovery_triggered: bool = False


@dataclass
class USB4RecoveryAttempt:
    """USB4 recovery attempt record"""

    attempt_number: int
    start_time: float
    end_time: float
    method: USB4RecoveryMethod
    status: USB4RecoveryStatus
    duration: float
    error_type: USB4ErrorType
    initial_state: USB4TrainingState
    final_state: USB4TrainingState
    signal_quality_improvement: float = 0.0
    bandwidth_after_recovery: float = 0.0
    power_consumption_change: float = 0.0
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class USB4RecoveryStatistics:
    """USB4 recovery performance statistics"""

    total_errors_injected: int = 0
    total_recovery_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    timeout_recoveries: int = 0
    average_recovery_time: float = 0.0
    min_recovery_time: float = float("inf")
    max_recovery_time: float = 0.0
    recovery_success_rate: float = 0.0
    recovery_times_by_error_type: Dict[USB4ErrorType, List[float]] = field(default_factory=dict)
    recovery_success_by_method: Dict[USB4RecoveryMethod, int] = field(default_factory=dict)
    recovery_attempts_by_method: Dict[USB4RecoveryMethod, int] = field(default_factory=dict)
    signal_quality_improvements: List[float] = field(default_factory=list)
    bandwidth_degradation_events: int = 0
    power_consumption_increases: int = 0


@dataclass
class USB4RecoveryTestResults:
    """USB4 link recovery test results"""

    test_duration: float
    total_errors_tested: int
    recovery_statistics: USB4RecoveryStatistics
    error_events: List[USB4ErrorEvent] = field(default_factory=list)
    recovery_attempts: List[USB4RecoveryAttempt] = field(default_factory=list)
    timing_validation_results: Dict[str, float] = field(default_factory=dict)
    effectiveness_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    test_passed: bool = True
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)


class USB4LinkRecoveryTester(USB4Component):
    """USB4 link recovery testing implementation"""

    def __init__(self, config: USB4RecoveryConfig):
        """
        Initialize USB4 link recovery tester

        Args:
            config: Recovery testing configuration
        """
        super().__init__(config)
        self.config: USB4RecoveryConfig = config
        self.specs = USB4Specs()

        # Initialize link training component
        training_config = USB4TrainingConfig(
            signal_mode=config.signal_mode,
            sample_rate=config.sample_rate,
            capture_length=config.capture_length,
            max_training_time=self.specs.MAX_TRAINING_TIME,
            training_timeout=self.specs.TRAINING_TIMEOUT,
            max_retries=self.specs.MAX_RETRIES,
            enable_recovery=True,
        )
        self.link_trainer = USB4LinkTraining(training_config)

        # Recovery state
        self.error_events: List[USB4ErrorEvent] = []
        self.recovery_attempts: List[USB4RecoveryAttempt] = []
        self.statistics = USB4RecoveryStatistics()

        # Timing measurements
        self.recovery_timings: Dict[USB4ErrorType, List[float]] = {}
        self.method_effectiveness: Dict[USB4RecoveryMethod, float] = {}

        # Test state
        self.test_start_time = 0.0
        self.current_error_count = 0

    def initialize(self) -> bool:
        """
        Initialize link recovery tester

        Returns:
            True if initialization successful
        """
        try:
            self.validate_config()
            self.specs.validate_all()

            # Initialize link trainer
            if not self.link_trainer.initialize():
                raise RuntimeError("Failed to initialize link trainer")

            self._reset_statistics()
            self._initialized = True
            self.logger.info("USB4 link recovery tester initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize USB4 link recovery tester: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up recovery tester resources"""
        if hasattr(self, "link_trainer"):
            self.link_trainer.cleanup()
        self.error_events.clear()
        self.recovery_attempts.clear()
        self._reset_statistics()
        self._initialized = False
        self.logger.info("USB4 link recovery tester cleaned up")

    def test_error_injection_and_recovery(self, injection_config: USB4ErrorInjectionConfig) -> USB4RecoveryTestResults:
        """
        Test error injection and recovery for specific error type

        Args:
            injection_config: Error injection configuration

        Returns:
            Recovery test results
        """
        if not self.is_initialized:
            raise RuntimeError("Recovery tester not initialized")

        self.logger.info(f"Testing error injection and recovery for {injection_config.error_type.name}")
        test_start_time = time.time()

        try:
            # Inject error and measure recovery
            error_event = self._inject_error(injection_config)
            recovery_attempts = self._perform_recovery(error_event, injection_config)

            # Validate recovery timing
            timing_results = self._validate_recovery_timing(recovery_attempts)

            # Calculate effectiveness metrics
            effectiveness_metrics = self._calculate_effectiveness_metrics(recovery_attempts)

            # Generate recommendations
            recommendations = self._generate_recovery_recommendations(recovery_attempts)

            test_duration = time.time() - test_start_time

            results = USB4RecoveryTestResults(
                test_duration=test_duration,
                total_errors_tested=1,
                recovery_statistics=self._calculate_current_statistics(),
                error_events=[error_event],
                recovery_attempts=recovery_attempts,
                timing_validation_results=timing_results,
                effectiveness_metrics=effectiveness_metrics,
                recommendations=recommendations,
                test_passed=self._evaluate_test_success(recovery_attempts),
                detailed_analysis=self._generate_detailed_analysis(recovery_attempts),
            )

            self.logger.info(f"Error injection and recovery test completed in {test_duration:.3f}s")
            return results

        except Exception as e:
            self.logger.error(f"Error injection and recovery test failed: {e}")
            raise

    def test_comprehensive_recovery(self) -> USB4RecoveryTestResults:
        """
        Test comprehensive recovery across all error types

        Returns:
            Comprehensive recovery test results
        """
        if not self.is_initialized:
            raise RuntimeError("Recovery tester not initialized")

        self.logger.info("Starting comprehensive USB4 recovery testing")
        test_start_time = time.time()

        try:
            all_error_events = []
            all_recovery_attempts = []

            # Test each error type
            error_types = list(USB4ErrorType) if self.config.test_all_error_types else [USB4ErrorType.LINK_TRAINING]

            for error_type in error_types:
                self.logger.info(f"Testing recovery for error type: {error_type.name}")

                # Configure error injection
                injection_config = USB4ErrorInjectionConfig(
                    error_type=error_type, severity=USB4ErrorSeverity.MEDIUM, recovery_timeout=self.config.max_recovery_time
                )

                # Test error injection and recovery
                single_test_results = self.test_error_injection_and_recovery(injection_config)
                all_error_events.extend(single_test_results.error_events)
                all_recovery_attempts.extend(single_test_results.recovery_attempts)

            # Perform stress testing if configured
            if self.config.stress_test_duration > 0:
                stress_results = self._perform_stress_testing()
                all_error_events.extend(stress_results["error_events"])
                all_recovery_attempts.extend(stress_results["recovery_attempts"])

            # Calculate comprehensive statistics
            comprehensive_stats = self._calculate_comprehensive_statistics(all_recovery_attempts)

            # Validate overall timing performance
            timing_results = self._validate_comprehensive_timing(all_recovery_attempts)

            # Calculate overall effectiveness
            effectiveness_metrics = self._calculate_comprehensive_effectiveness(all_recovery_attempts)

            # Generate comprehensive recommendations
            recommendations = self._generate_comprehensive_recommendations(all_recovery_attempts)

            test_duration = time.time() - test_start_time

            results = USB4RecoveryTestResults(
                test_duration=test_duration,
                total_errors_tested=len(all_error_events),
                recovery_statistics=comprehensive_stats,
                error_events=all_error_events,
                recovery_attempts=all_recovery_attempts,
                timing_validation_results=timing_results,
                effectiveness_metrics=effectiveness_metrics,
                recommendations=recommendations,
                test_passed=self._evaluate_comprehensive_test_success(all_recovery_attempts),
                detailed_analysis=self._generate_comprehensive_analysis(all_recovery_attempts),
            )

            self.logger.info(
                f"Comprehensive recovery testing completed in {test_duration:.3f}s, "
                f"tested {len(all_error_events)} errors with {len(all_recovery_attempts)} recovery attempts"
            )
            return results

        except Exception as e:
            self.logger.error(f"Comprehensive recovery testing failed: {e}")
            raise

    def validate_recovery_timing(self, recovery_attempts: List[USB4RecoveryAttempt]) -> Dict[str, float]:
        """
        Validate link recovery timing against specifications

        Args:
            recovery_attempts: List of recovery attempts to validate

        Returns:
            Timing validation results
        """
        if not recovery_attempts:
            return {}

        self.logger.info(f"Validating recovery timing for {len(recovery_attempts)} attempts")

        timing_results = {}

        # Calculate timing statistics
        recovery_times = [attempt.duration for attempt in recovery_attempts]
        timing_results["average_recovery_time"] = np.mean(recovery_times)
        timing_results["min_recovery_time"] = np.min(recovery_times)
        timing_results["max_recovery_time"] = np.max(recovery_times)
        timing_results["std_recovery_time"] = np.std(recovery_times)

        # Validate against specifications
        max_allowed_time = self.config.max_recovery_time
        timing_results["max_allowed_time"] = max_allowed_time
        timing_results["timing_compliance"] = np.max(recovery_times) <= max_allowed_time
        timing_results["timing_violations"] = sum(1 for t in recovery_times if t > max_allowed_time)
        timing_results["timing_violation_rate"] = timing_results["timing_violations"] / len(recovery_times)

        # Analyze timing by error type
        timing_by_error_type = {}
        for error_type in USB4ErrorType:
            error_attempts = [a for a in recovery_attempts if a.error_type == error_type]
            if error_attempts:
                error_times = [a.duration for a in error_attempts]
                timing_by_error_type[error_type.name] = {
                    "average": np.mean(error_times),
                    "max": np.max(error_times),
                    "count": len(error_times),
                }
        timing_results["timing_by_error_type"] = timing_by_error_type

        # Analyze timing by recovery method
        timing_by_method = {}
        for method in USB4RecoveryMethod:
            method_attempts = [a for a in recovery_attempts if a.method == method]
            if method_attempts:
                method_times = [a.duration for a in method_attempts]
                timing_by_method[method.name] = {
                    "average": np.mean(method_times),
                    "max": np.max(method_times),
                    "count": len(method_times),
                }
        timing_results["timing_by_method"] = timing_by_method

        self.logger.info(
            f"Recovery timing validation completed: "
            f"avg={timing_results['average_recovery_time']:.3f}s, "
            f"max={timing_results['max_recovery_time']:.3f}s, "
            f"compliance={timing_results['timing_compliance']}"
        )

        return timing_results

    def measure_recovery_effectiveness(self, recovery_attempts: List[USB4RecoveryAttempt]) -> Dict[str, float]:
        """
        Measure recovery mechanism effectiveness

        Args:
            recovery_attempts: List of recovery attempts to analyze

        Returns:
            Effectiveness metrics
        """
        if not recovery_attempts:
            return {}

        self.logger.info(f"Measuring recovery effectiveness for {len(recovery_attempts)} attempts")

        effectiveness_metrics = {}

        # Overall success rate
        successful_attempts = [a for a in recovery_attempts if a.status == USB4RecoveryStatus.SUCCESS]
        effectiveness_metrics["overall_success_rate"] = len(successful_attempts) / len(recovery_attempts)

        # Success rate by error type
        success_by_error_type = {}
        for error_type in USB4ErrorType:
            error_attempts = [a for a in recovery_attempts if a.error_type == error_type]
            if error_attempts:
                error_successes = [a for a in error_attempts if a.status == USB4RecoveryStatus.SUCCESS]
                success_by_error_type[error_type.name] = len(error_successes) / len(error_attempts)
        effectiveness_metrics["success_by_error_type"] = success_by_error_type

        # Success rate by recovery method
        success_by_method = {}
        for method in USB4RecoveryMethod:
            method_attempts = [a for a in recovery_attempts if a.method == method]
            if method_attempts:
                method_successes = [a for a in method_attempts if a.status == USB4RecoveryStatus.SUCCESS]
                success_by_method[method.name] = len(method_successes) / len(method_attempts)
        effectiveness_metrics["success_by_method"] = success_by_method

        # Signal quality improvement
        quality_improvements = [a.signal_quality_improvement for a in successful_attempts if a.signal_quality_improvement > 0]
        if quality_improvements:
            effectiveness_metrics["average_quality_improvement"] = np.mean(quality_improvements)
            effectiveness_metrics["max_quality_improvement"] = np.max(quality_improvements)

        # Bandwidth preservation
        bandwidth_preserved = [a for a in successful_attempts if a.bandwidth_after_recovery > 0]
        if bandwidth_preserved:
            bandwidth_ratios = [a.bandwidth_after_recovery / self.specs.TOTAL_BANDWIDTH for a in bandwidth_preserved]
            effectiveness_metrics["average_bandwidth_preservation"] = np.mean(bandwidth_ratios)
            effectiveness_metrics["min_bandwidth_preservation"] = np.min(bandwidth_ratios)

        # Power efficiency
        power_efficient_recoveries = [a for a in successful_attempts if a.power_consumption_change <= 0]
        effectiveness_metrics["power_efficient_recovery_rate"] = (
            len(power_efficient_recoveries) / len(successful_attempts) if successful_attempts else 0
        )

        # Recovery attempt efficiency
        single_attempt_successes = [a for a in successful_attempts if a.attempt_number == 1]
        effectiveness_metrics["first_attempt_success_rate"] = len(single_attempt_successes) / len(recovery_attempts)

        self.logger.info(
            f"Recovery effectiveness measurement completed: "
            f"success_rate={effectiveness_metrics['overall_success_rate']:.3f}, "
            f"first_attempt_rate={effectiveness_metrics['first_attempt_success_rate']:.3f}"
        )

        return effectiveness_metrics

    def generate_recovery_analysis_report(self, test_results: USB4RecoveryTestResults) -> str:
        """
        Generate comprehensive recovery analysis report

        Args:
            test_results: Recovery test results to analyze

        Returns:
            Formatted analysis report
        """
        report_lines = []
        report_lines.append("USB4 Link Recovery Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Test summary
        report_lines.append("Test Summary:")
        report_lines.append(f"  Test Duration: {test_results.test_duration:.3f} seconds")
        report_lines.append(f"  Total Errors Tested: {test_results.total_errors_tested}")
        report_lines.append(f"  Total Recovery Attempts: {len(test_results.recovery_attempts)}")
        report_lines.append(f"  Test Result: {'PASSED' if test_results.test_passed else 'FAILED'}")
        report_lines.append("")

        # Recovery statistics
        stats = test_results.recovery_statistics
        report_lines.append("Recovery Statistics:")
        report_lines.append(f"  Success Rate: {stats.recovery_success_rate:.3f}")
        report_lines.append(f"  Average Recovery Time: {stats.average_recovery_time:.3f}s")
        report_lines.append(f"  Min Recovery Time: {stats.min_recovery_time:.3f}s")
        report_lines.append(f"  Max Recovery Time: {stats.max_recovery_time:.3f}s")
        report_lines.append(f"  Successful Recoveries: {stats.successful_recoveries}")
        report_lines.append(f"  Failed Recoveries: {stats.failed_recoveries}")
        report_lines.append(f"  Timeout Recoveries: {stats.timeout_recoveries}")
        report_lines.append("")

        # Timing validation
        if test_results.timing_validation_results:
            timing = test_results.timing_validation_results
            report_lines.append("Timing Validation:")
            report_lines.append(f"  Timing Compliance: {'PASS' if timing.get('timing_compliance', False) else 'FAIL'}")
            report_lines.append(f"  Timing Violations: {timing.get('timing_violations', 0)}")
            report_lines.append(f"  Violation Rate: {timing.get('timing_violation_rate', 0):.3f}")
            report_lines.append("")

        # Effectiveness metrics
        if test_results.effectiveness_metrics:
            effectiveness = test_results.effectiveness_metrics
            report_lines.append("Effectiveness Metrics:")
            report_lines.append(f"  Overall Success Rate: {effectiveness.get('overall_success_rate', 0):.3f}")
            report_lines.append(f"  First Attempt Success Rate: {effectiveness.get('first_attempt_success_rate', 0):.3f}")
            if "average_quality_improvement" in effectiveness:
                report_lines.append(f"  Average Quality Improvement: {effectiveness['average_quality_improvement']:.3f}")
            if "average_bandwidth_preservation" in effectiveness:
                report_lines.append(f"  Average Bandwidth Preservation: {effectiveness['average_bandwidth_preservation']:.3f}")
            report_lines.append("")

        # Recommendations
        if test_results.recommendations:
            report_lines.append("Recommendations:")
            for i, recommendation in enumerate(test_results.recommendations, 1):
                report_lines.append(f"  {i}. {recommendation}")
            report_lines.append("")

        # Error analysis
        if test_results.error_events:
            error_types = {}
            for event in test_results.error_events:
                error_types[event.error_type] = error_types.get(event.error_type, 0) + 1

            report_lines.append("Error Type Distribution:")
            for error_type, count in error_types.items():
                report_lines.append(f"  {error_type.name}: {count}")
            report_lines.append("")

        return "\n".join(report_lines)

    def _inject_error(self, config: USB4ErrorInjectionConfig) -> USB4ErrorEvent:
        """
        Inject error for testing

        Args:
            config: Error injection configuration

        Returns:
            Error event record
        """
        self.logger.info(f"Injecting {config.error_type.name} error with {config.severity.name} severity")

        # Measure signal quality before error
        signal_quality_before = self._measure_signal_quality()

        # Create error event
        error_event = USB4ErrorEvent(
            timestamp=time.time(),
            error_type=config.error_type,
            severity=config.severity,
            lane=config.target_lane,
            description=f"Injected {config.error_type.name} error for testing",
            signal_quality_before=signal_quality_before,
            recovery_triggered=True,
        )

        # Simulate error injection effects
        self._simulate_error_effects(config)

        # Measure signal quality after error
        error_event.signal_quality_after = self._measure_signal_quality()

        self.error_events.append(error_event)
        self.current_error_count += 1

        if config.enable_logging:
            self.logger.info(
                f"Error injected: {config.error_type.name}, "
                f"quality degradation: {signal_quality_before - error_event.signal_quality_after:.3f}"
            )

        return error_event

    def _perform_recovery(self, error_event: USB4ErrorEvent, config: USB4ErrorInjectionConfig) -> List[USB4RecoveryAttempt]:
        """
        Perform recovery from injected error

        Args:
            error_event: Error event to recover from
            config: Error injection configuration

        Returns:
            List of recovery attempts
        """
        recovery_attempts = []
        attempt_number = 1
        recovery_successful = False

        while (
            attempt_number <= self.config.max_recovery_attempts
            and not recovery_successful
            and (time.time() - error_event.timestamp) < config.recovery_timeout
        ):
            # Select recovery method based on error type and attempt number
            recovery_method = self._select_recovery_method(error_event.error_type, attempt_number)

            self.logger.info(f"Recovery attempt {attempt_number} using {recovery_method.name}")

            # Perform recovery attempt
            attempt = self._execute_recovery_attempt(attempt_number, error_event, recovery_method)

            recovery_attempts.append(attempt)

            # Check if recovery was successful
            if attempt.status == USB4RecoveryStatus.SUCCESS:
                recovery_successful = True
                self.logger.info(f"Recovery successful after {attempt_number} attempts")
            else:
                self.logger.warning(f"Recovery attempt {attempt_number} failed: {attempt.status.name}")

            attempt_number += 1

        if not recovery_successful:
            self.logger.error(f"Recovery failed after {attempt_number - 1} attempts")

        return recovery_attempts

    def _execute_recovery_attempt(
        self, attempt_number: int, error_event: USB4ErrorEvent, method: USB4RecoveryMethod
    ) -> USB4RecoveryAttempt:
        """
        Execute single recovery attempt

        Args:
            attempt_number: Recovery attempt number
            error_event: Error event being recovered from
            method: Recovery method to use

        Returns:
            Recovery attempt record
        """
        start_time = time.time()
        initial_state = self.link_trainer.current_state

        # Execute recovery method
        recovery_status = self._execute_recovery_method(method, error_event)

        end_time = time.time()
        duration = end_time - start_time
        final_state = self.link_trainer.current_state

        # Measure recovery effectiveness
        signal_quality_improvement = self._measure_recovery_effectiveness(error_event)
        bandwidth_after_recovery = self._measure_bandwidth_after_recovery()
        power_consumption_change = self._measure_power_consumption_change()

        attempt = USB4RecoveryAttempt(
            attempt_number=attempt_number,
            start_time=start_time,
            end_time=end_time,
            method=method,
            status=recovery_status,
            duration=duration,
            error_type=error_event.error_type,
            initial_state=initial_state,
            final_state=final_state,
            signal_quality_improvement=signal_quality_improvement,
            bandwidth_after_recovery=bandwidth_after_recovery,
            power_consumption_change=power_consumption_change,
            additional_info={
                "error_timestamp": error_event.timestamp,
                "recovery_delay": start_time - error_event.timestamp,
                "method_specific_data": self._get_method_specific_data(method),
            },
        )

        self.recovery_attempts.append(attempt)
        return attempt

    def _select_recovery_method(self, error_type: USB4ErrorType, attempt_number: int) -> USB4RecoveryMethod:
        """
        Select appropriate recovery method based on error type and attempt number

        Args:
            error_type: Type of error to recover from
            attempt_number: Current attempt number

        Returns:
            Selected recovery method
        """
        # Recovery method selection strategy
        method_strategies = {
            USB4ErrorType.SIGNAL_INTEGRITY: [
                USB4RecoveryMethod.EQUALIZATION_ADJUST,
                USB4RecoveryMethod.SPEED_DOWNGRADE,
                USB4RecoveryMethod.LINK_RETRAIN,
            ],
            USB4ErrorType.LINK_TRAINING: [
                USB4RecoveryMethod.LINK_RETRAIN,
                USB4RecoveryMethod.PROTOCOL_RESET,
                USB4RecoveryMethod.POWER_CYCLE,
            ],
            USB4ErrorType.PROTOCOL: [
                USB4RecoveryMethod.PROTOCOL_RESET,
                USB4RecoveryMethod.LINK_RETRAIN,
                USB4RecoveryMethod.POWER_CYCLE,
            ],
            USB4ErrorType.POWER_MANAGEMENT: [
                USB4RecoveryMethod.POWER_CYCLE,
                USB4RecoveryMethod.PROTOCOL_RESET,
                USB4RecoveryMethod.LINK_RETRAIN,
            ],
            USB4ErrorType.TUNNELING: [
                USB4RecoveryMethod.PROTOCOL_RESET,
                USB4RecoveryMethod.LINK_RETRAIN,
                USB4RecoveryMethod.POWER_CYCLE,
            ],
        }

        methods = method_strategies.get(error_type, [USB4RecoveryMethod.LINK_RETRAIN])
        method_index = min(attempt_number - 1, len(methods) - 1)
        return methods[method_index]

    def _execute_recovery_method(self, method: USB4RecoveryMethod, error_event: USB4ErrorEvent) -> USB4RecoveryStatus:
        """
        Execute specific recovery method

        Args:
            method: Recovery method to execute
            error_event: Error event being recovered from

        Returns:
            Recovery status
        """
        self.logger.debug(f"Executing recovery method: {method.name}")

        try:
            if method == USB4RecoveryMethod.LINK_RETRAIN:
                return self._execute_link_retrain_recovery()
            elif method == USB4RecoveryMethod.SPEED_DOWNGRADE:
                return self._execute_speed_downgrade_recovery()
            elif method == USB4RecoveryMethod.LANE_DISABLE:
                return self._execute_lane_disable_recovery(error_event.lane)
            elif method == USB4RecoveryMethod.POWER_CYCLE:
                return self._execute_power_cycle_recovery()
            elif method == USB4RecoveryMethod.PROTOCOL_RESET:
                return self._execute_protocol_reset_recovery()
            elif method == USB4RecoveryMethod.EQUALIZATION_ADJUST:
                return self._execute_equalization_adjust_recovery()
            elif method == USB4RecoveryMethod.CLOCK_RECOVERY:
                return self._execute_clock_recovery_recovery()
            else:
                self.logger.warning(f"Unknown recovery method: {method.name}")
                return USB4RecoveryStatus.FAILURE

        except Exception as e:
            self.logger.error(f"Recovery method {method.name} failed: {e}")
            return USB4RecoveryStatus.FAILURE

    def _execute_link_retrain_recovery(self) -> USB4RecoveryStatus:
        """Execute link retraining recovery"""
        try:
            # Simulate link retraining
            time.sleep(0.1)  # Simulate retraining time
            training_result = self.link_trainer.execute_training()

            if training_result.convergence_status:
                return USB4RecoveryStatus.SUCCESS
            else:
                return USB4RecoveryStatus.FAILURE
        except Exception:
            return USB4RecoveryStatus.FAILURE

    def _execute_speed_downgrade_recovery(self) -> USB4RecoveryStatus:
        """Execute speed downgrade recovery"""
        try:
            # Simulate speed downgrade
            time.sleep(0.05)  # Simulate downgrade time

            # Attempt negotiation to lower speed
            current_mode = self.link_trainer.negotiated_mode
            if current_mode == USB4SignalMode.GEN3X2:
                # Downgrade to Gen2
                negotiation_result = self.link_trainer.validate_speed_negotiation(USB4SignalMode.GEN2X2)
                if negotiation_result.negotiated_mode == USB4SignalMode.GEN2X2:
                    return USB4RecoveryStatus.SUCCESS
                else:
                    return USB4RecoveryStatus.PARTIAL_SUCCESS
            else:
                # Already at lowest speed
                return USB4RecoveryStatus.DEGRADED
        except Exception:
            return USB4RecoveryStatus.FAILURE

    def _execute_lane_disable_recovery(self, target_lane: Optional[int]) -> USB4RecoveryStatus:
        """Execute lane disable recovery"""
        try:
            # Simulate lane disable (not applicable for USB4 which requires 2 lanes)
            time.sleep(0.02)
            # USB4 requires both lanes, so this would result in degraded performance
            return USB4RecoveryStatus.DEGRADED
        except Exception:
            return USB4RecoveryStatus.FAILURE

    def _execute_power_cycle_recovery(self) -> USB4RecoveryStatus:
        """Execute power cycle recovery"""
        try:
            # Simulate power cycle
            time.sleep(0.5)  # Simulate power cycle time

            # Reinitialize link trainer
            if self.link_trainer.initialize():
                training_result = self.link_trainer.execute_training()
                if training_result.convergence_status:
                    return USB4RecoveryStatus.SUCCESS
                else:
                    return USB4RecoveryStatus.PARTIAL_SUCCESS
            else:
                return USB4RecoveryStatus.FAILURE
        except Exception:
            return USB4RecoveryStatus.FAILURE

    def _execute_protocol_reset_recovery(self) -> USB4RecoveryStatus:
        """Execute protocol reset recovery"""
        try:
            # Simulate protocol reset
            time.sleep(0.1)  # Simulate reset time

            # Reset protocol state machine
            success_rate = 0.8  # 80% success rate for protocol reset
            if np.random.random() < success_rate:
                return USB4RecoveryStatus.SUCCESS
            else:
                return USB4RecoveryStatus.PARTIAL_SUCCESS
        except Exception:
            return USB4RecoveryStatus.FAILURE

    def _execute_equalization_adjust_recovery(self) -> USB4RecoveryStatus:
        """Execute equalization adjustment recovery"""
        try:
            # Simulate equalization adjustment
            time.sleep(0.05)  # Simulate adjustment time

            # Simulate equalization effectiveness
            success_rate = 0.7  # 70% success rate for equalization adjustment
            if np.random.random() < success_rate:
                return USB4RecoveryStatus.SUCCESS
            else:
                return USB4RecoveryStatus.PARTIAL_SUCCESS
        except Exception:
            return USB4RecoveryStatus.FAILURE

    def _execute_clock_recovery_recovery(self) -> USB4RecoveryStatus:
        """Execute clock recovery reset"""
        try:
            # Simulate clock recovery reset
            time.sleep(0.03)  # Simulate clock recovery time

            # Simulate clock recovery effectiveness
            success_rate = 0.85  # 85% success rate for clock recovery
            if np.random.random() < success_rate:
                return USB4RecoveryStatus.SUCCESS
            else:
                return USB4RecoveryStatus.PARTIAL_SUCCESS
        except Exception:
            return USB4RecoveryStatus.FAILURE

    def _simulate_error_effects(self, config: USB4ErrorInjectionConfig) -> None:
        """
        Simulate error effects on the link

        Args:
            config: Error injection configuration
        """
        # Simulate error effects based on type and severity
        if config.error_type == USB4ErrorType.SIGNAL_INTEGRITY:
            # Simulate signal degradation
            pass
        elif config.error_type == USB4ErrorType.LINK_TRAINING:
            # Force link training failure
            self.link_trainer.current_state = USB4TrainingState.RECOVERY
        elif config.error_type == USB4ErrorType.PROTOCOL:
            # Simulate protocol violation
            pass
        elif config.error_type == USB4ErrorType.POWER_MANAGEMENT:
            # Force power state error
            self.link_trainer.current_link_state = USB4LinkState.U3
        elif config.error_type == USB4ErrorType.TUNNELING:
            # Simulate tunneling error
            pass

        # Simulate error duration
        time.sleep(config.duration)

    def _measure_signal_quality(self) -> float:
        """
        Measure current signal quality

        Returns:
            Signal quality metric (0.0 to 1.0)
        """
        # Simulate signal quality measurement
        base_quality = 0.8
        noise = np.random.normal(0, 0.1)
        return max(0.0, min(1.0, base_quality + noise))

    def _measure_recovery_effectiveness(self, error_event: USB4ErrorEvent) -> float:
        """
        Measure recovery effectiveness

        Args:
            error_event: Original error event

        Returns:
            Signal quality improvement
        """
        current_quality = self._measure_signal_quality()
        if error_event.signal_quality_after is not None:
            return max(0.0, current_quality - error_event.signal_quality_after)
        return 0.0

    def _measure_bandwidth_after_recovery(self) -> float:
        """
        Measure bandwidth after recovery

        Returns:
            Available bandwidth in bps
        """
        # Simulate bandwidth measurement based on current link state
        if self.link_trainer.current_link_state == USB4LinkState.U0:
            # Full bandwidth available
            mode_bandwidth = {
                USB4SignalMode.GEN2X2: self.specs.TOTAL_BANDWIDTH,
                USB4SignalMode.GEN3X2: self.specs.TOTAL_BANDWIDTH,
                USB4SignalMode.ASYMMETRIC: self.specs.TOTAL_BANDWIDTH * 0.8,
            }
            return mode_bandwidth.get(self.link_trainer.negotiated_mode, self.specs.TOTAL_BANDWIDTH)
        else:
            # Reduced bandwidth in power saving states
            return self.specs.TOTAL_BANDWIDTH * 0.1

    def _measure_power_consumption_change(self) -> float:
        """
        Measure power consumption change after recovery

        Returns:
            Power consumption change in watts (positive = increase)
        """
        # Simulate power consumption measurement
        current_power = self.link_trainer._measure_power_consumption(self.link_trainer.current_link_state)
        baseline_power = self.specs.IDLE_POWER_U0
        return current_power - baseline_power

    def _get_method_specific_data(self, method: USB4RecoveryMethod) -> Dict[str, Any]:
        """
        Get method-specific recovery data

        Args:
            method: Recovery method

        Returns:
            Method-specific data
        """
        method_data = {"method_name": method.name, "timestamp": time.time()}

        if method == USB4RecoveryMethod.SPEED_DOWNGRADE:
            method_data["original_mode"] = self.link_trainer.negotiated_mode.name
        elif method == USB4RecoveryMethod.EQUALIZATION_ADJUST:
            method_data["equalization_settings"] = "adaptive"
        elif method == USB4RecoveryMethod.POWER_CYCLE:
            method_data["power_cycle_duration"] = 0.5

        return method_data

    def _validate_recovery_timing(self, recovery_attempts: List[USB4RecoveryAttempt]) -> Dict[str, float]:
        """
        Validate recovery timing for single test

        Args:
            recovery_attempts: Recovery attempts to validate

        Returns:
            Timing validation results
        """
        if not recovery_attempts:
            return {}

        recovery_times = [attempt.duration for attempt in recovery_attempts]

        return {
            "total_recovery_time": sum(recovery_times),
            "average_attempt_time": np.mean(recovery_times),
            "max_attempt_time": np.max(recovery_times),
            "timing_compliance": np.max(recovery_times) <= self.config.max_recovery_time,
            "attempt_count": len(recovery_attempts),
        }

    def _calculate_effectiveness_metrics(self, recovery_attempts: List[USB4RecoveryAttempt]) -> Dict[str, float]:
        """
        Calculate effectiveness metrics for single test

        Args:
            recovery_attempts: Recovery attempts to analyze

        Returns:
            Effectiveness metrics
        """
        if not recovery_attempts:
            return {}

        successful_attempts = [a for a in recovery_attempts if a.status == USB4RecoveryStatus.SUCCESS]

        metrics = {
            "success_rate": len(successful_attempts) / len(recovery_attempts),
            "attempts_required": len(recovery_attempts),
            "first_attempt_success": recovery_attempts[0].status == USB4RecoveryStatus.SUCCESS if recovery_attempts else False,
        }

        if successful_attempts:
            metrics["average_quality_improvement"] = np.mean([a.signal_quality_improvement for a in successful_attempts])
            metrics["bandwidth_preservation"] = np.mean(
                [a.bandwidth_after_recovery / self.specs.TOTAL_BANDWIDTH for a in successful_attempts]
            )

        return metrics

    def _generate_recovery_recommendations(self, recovery_attempts: List[USB4RecoveryAttempt]) -> List[str]:
        """
        Generate recovery recommendations based on test results

        Args:
            recovery_attempts: Recovery attempts to analyze

        Returns:
            List of recommendations
        """
        recommendations = []

        if not recovery_attempts:
            return ["No recovery attempts to analyze"]

        # Analyze success rates by method
        method_success = {}
        for attempt in recovery_attempts:
            if attempt.method not in method_success:
                method_success[attempt.method] = {"success": 0, "total": 0}
            method_success[attempt.method]["total"] += 1
            if attempt.status == USB4RecoveryStatus.SUCCESS:
                method_success[attempt.method]["success"] += 1

        # Find most effective method
        best_method = None
        best_rate = 0.0
        for method, stats in method_success.items():
            rate = stats["success"] / stats["total"]
            if rate > best_rate:
                best_rate = rate
                best_method = method

        if best_method:
            recommendations.append(f"Most effective recovery method: {best_method.name} (success rate: {best_rate:.3f})")

        # Timing recommendations
        long_recoveries = [a for a in recovery_attempts if a.duration > self.config.max_recovery_time]
        if long_recoveries:
            recommendations.append(f"Consider optimizing recovery timing - {len(long_recoveries)} attempts exceeded maximum time")

        # Multiple attempt recommendations
        if len(recovery_attempts) > 1:
            recommendations.append("Consider improving first-attempt success rate to reduce recovery overhead")

        return recommendations

    def _evaluate_test_success(self, recovery_attempts: List[USB4RecoveryAttempt]) -> bool:
        """
        Evaluate if test was successful

        Args:
            recovery_attempts: Recovery attempts to evaluate

        Returns:
            True if test passed
        """
        if not recovery_attempts:
            return False

        # Test passes if at least one recovery was successful and within time limits
        successful_attempts = [
            a for a in recovery_attempts if a.status == USB4RecoveryStatus.SUCCESS and a.duration <= self.config.max_recovery_time
        ]

        return len(successful_attempts) > 0

    def _generate_detailed_analysis(self, recovery_attempts: List[USB4RecoveryAttempt]) -> Dict[str, Any]:
        """
        Generate detailed analysis of recovery attempts

        Args:
            recovery_attempts: Recovery attempts to analyze

        Returns:
            Detailed analysis data
        """
        if not recovery_attempts:
            return {}

        analysis = {
            "total_attempts": len(recovery_attempts),
            "unique_methods_used": len(set(a.method for a in recovery_attempts)),
            "state_transitions": len(set((a.initial_state, a.final_state) for a in recovery_attempts)),
            "timing_analysis": {
                "fastest_recovery": min(a.duration for a in recovery_attempts),
                "slowest_recovery": max(a.duration for a in recovery_attempts),
                "timing_variance": np.var([a.duration for a in recovery_attempts]),
            },
            "effectiveness_analysis": {
                "quality_improvements": [a.signal_quality_improvement for a in recovery_attempts],
                "bandwidth_preservation": [a.bandwidth_after_recovery for a in recovery_attempts],
                "power_impact": [a.power_consumption_change for a in recovery_attempts],
            },
        }

        return analysis

    def _perform_stress_testing(self) -> Dict[str, List]:
        """
        Perform stress testing with multiple errors

        Returns:
            Stress test results
        """
        self.logger.info(f"Starting stress testing for {self.config.stress_test_duration}s")

        stress_error_events = []
        stress_recovery_attempts = []

        start_time = time.time()
        error_interval = 1.0 / self.config.error_injection_rate
        next_error_time = start_time + error_interval

        while (time.time() - start_time) < self.config.stress_test_duration:
            if time.time() >= next_error_time:
                # Inject random error
                error_type = np.random.choice(list(USB4ErrorType))
                injection_config = USB4ErrorInjectionConfig(
                    error_type=error_type,
                    severity=USB4ErrorSeverity.MEDIUM,
                    duration=0.05,
                    recovery_timeout=self.config.max_recovery_time,
                )

                # Inject error and perform recovery
                error_event = self._inject_error(injection_config)
                recovery_attempts = self._perform_recovery(error_event, injection_config)

                stress_error_events.append(error_event)
                stress_recovery_attempts.extend(recovery_attempts)

                next_error_time = time.time() + error_interval

            time.sleep(0.01)  # Small delay to prevent busy waiting

        self.logger.info(
            f"Stress testing completed: {len(stress_error_events)} errors, " f"{len(stress_recovery_attempts)} recovery attempts"
        )

        return {"error_events": stress_error_events, "recovery_attempts": stress_recovery_attempts}

    def _calculate_current_statistics(self) -> USB4RecoveryStatistics:
        """Calculate current recovery statistics"""
        stats = USB4RecoveryStatistics()

        if not self.recovery_attempts:
            return stats

        stats.total_errors_injected = len(self.error_events)
        stats.total_recovery_attempts = len(self.recovery_attempts)

        successful_attempts = [a for a in self.recovery_attempts if a.status == USB4RecoveryStatus.SUCCESS]
        failed_attempts = [a for a in self.recovery_attempts if a.status == USB4RecoveryStatus.FAILURE]
        timeout_attempts = [a for a in self.recovery_attempts if a.status == USB4RecoveryStatus.TIMEOUT]

        stats.successful_recoveries = len(successful_attempts)
        stats.failed_recoveries = len(failed_attempts)
        stats.timeout_recoveries = len(timeout_attempts)

        if self.recovery_attempts:
            recovery_times = [a.duration for a in self.recovery_attempts]
            stats.average_recovery_time = np.mean(recovery_times)
            stats.min_recovery_time = np.min(recovery_times)
            stats.max_recovery_time = np.max(recovery_times)
            stats.recovery_success_rate = len(successful_attempts) / len(self.recovery_attempts)

        # Statistics by error type
        for error_type in USB4ErrorType:
            error_attempts = [a for a in self.recovery_attempts if a.error_type == error_type]
            if error_attempts:
                stats.recovery_times_by_error_type[error_type] = [a.duration for a in error_attempts]

        # Statistics by method
        for method in USB4RecoveryMethod:
            method_attempts = [a for a in self.recovery_attempts if a.method == method]
            method_successes = [a for a in method_attempts if a.status == USB4RecoveryStatus.SUCCESS]

            stats.recovery_attempts_by_method[method] = len(method_attempts)
            stats.recovery_success_by_method[method] = len(method_successes)

        # Signal quality improvements
        stats.signal_quality_improvements = [a.signal_quality_improvement for a in successful_attempts]

        # Bandwidth and power statistics
        stats.bandwidth_degradation_events = len(
            [a for a in self.recovery_attempts if a.bandwidth_after_recovery < self.specs.TOTAL_BANDWIDTH * 0.9]
        )
        stats.power_consumption_increases = len([a for a in self.recovery_attempts if a.power_consumption_change > 0])

        return stats

    def _calculate_comprehensive_statistics(self, all_attempts: List[USB4RecoveryAttempt]) -> USB4RecoveryStatistics:
        """Calculate comprehensive statistics for all recovery attempts"""
        # Temporarily store all attempts and calculate statistics
        original_attempts = self.recovery_attempts.copy()
        self.recovery_attempts = all_attempts
        stats = self._calculate_current_statistics()
        self.recovery_attempts = original_attempts
        return stats

    def _validate_comprehensive_timing(self, all_attempts: List[USB4RecoveryAttempt]) -> Dict[str, float]:
        """Validate timing for comprehensive test"""
        return self.validate_recovery_timing(all_attempts)

    def _calculate_comprehensive_effectiveness(self, all_attempts: List[USB4RecoveryAttempt]) -> Dict[str, float]:
        """Calculate comprehensive effectiveness metrics"""
        return self.measure_recovery_effectiveness(all_attempts)

    def _generate_comprehensive_recommendations(self, all_attempts: List[USB4RecoveryAttempt]) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = self._generate_recovery_recommendations(all_attempts)

        # Add comprehensive-specific recommendations
        if len(all_attempts) > 10:
            error_types_tested = set(a.error_type for a in all_attempts)
            if len(error_types_tested) == len(USB4ErrorType):
                recommendations.append("Comprehensive testing completed for all error types")
            else:
                missing_types = set(USB4ErrorType) - error_types_tested
                recommendations.append(f"Consider testing additional error types: {[t.name for t in missing_types]}")

        return recommendations

    def _evaluate_comprehensive_test_success(self, all_attempts: List[USB4RecoveryAttempt]) -> bool:
        """Evaluate comprehensive test success"""
        if not all_attempts:
            return False

        # Test passes if recovery success rate is above threshold
        successful_attempts = [a for a in all_attempts if a.status == USB4RecoveryStatus.SUCCESS]
        success_rate = len(successful_attempts) / len(all_attempts)

        # Require at least 70% success rate for comprehensive test to pass
        return success_rate >= 0.7

    def _generate_comprehensive_analysis(self, all_attempts: List[USB4RecoveryAttempt]) -> Dict[str, Any]:
        """Generate comprehensive analysis"""
        analysis = self._generate_detailed_analysis(all_attempts)

        # Add comprehensive-specific analysis
        error_types_tested = list(set(a.error_type for a in all_attempts))
        methods_used = list(set(a.method for a in all_attempts))

        analysis["comprehensive_coverage"] = {
            "error_types_tested": [t.name for t in error_types_tested],
            "recovery_methods_used": [m.name for m in methods_used],
            "coverage_percentage": len(error_types_tested) / len(USB4ErrorType),
        }

        return analysis

    def _reset_statistics(self) -> None:
        """Reset recovery statistics"""
        self.statistics = USB4RecoveryStatistics()
        self.error_events.clear()
        self.recovery_attempts.clear()
        self.recovery_timings.clear()
        self.method_effectiveness.clear()
        self.current_error_count = 0


__all__ = [
    # Enums
    "USB4RecoveryMethod",
    "USB4ErrorSeverity",
    "USB4RecoveryStatus",
    # Data structures
    "USB4ErrorInjectionConfig",
    "USB4RecoveryConfig",
    "USB4ErrorEvent",
    "USB4RecoveryAttempt",
    "USB4RecoveryStatistics",
    "USB4RecoveryTestResults",
    # Main class
    "USB4LinkRecoveryTester",
]
