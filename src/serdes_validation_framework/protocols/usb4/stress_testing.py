"""
USB4 Stress Testing Module

This module provides comprehensive USB4 stress testing capabilities including
long-duration stability testing, thermal stress testing with temperature monitoring,
and error injection and recovery stress testing.

Features:
- Long-duration stability testing methods
- Thermal stress testing with temperature monitoring
- Error injection and recovery stress testing
- Comprehensive stress test reporting
- Real-time monitoring and alerting
- Automated test execution and analysis
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .constants import (
    ThunderboltSpecs,
    USB4SignalMode,
    USB4SignalSpecs,
    USB4Specs,
    USB4TunnelingSpecs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USB4StressTestType(Enum):
    """Types of USB4 stress tests"""

    STABILITY = auto()  # Long-duration stability testing
    THERMAL = auto()  # Temperature stress testing
    ERROR_INJECTION = auto()  # Error injection and recovery
    POWER_CYCLING = auto()  # Power state cycling stress
    BANDWIDTH_SATURATION = auto()  # Maximum bandwidth stress
    PROTOCOL_STRESS = auto()  # Protocol state machine stress
    MULTI_DEVICE = auto()  # Multiple device stress testing


class USB4StressTestStatus(Enum):
    """Status of stress test execution"""

    NOT_STARTED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABORTED = auto()


@dataclass
class USB4StressTestConfig:
    """Configuration for USB4 stress testing"""

    test_type: USB4StressTestType
    duration: float
    signal_mode: USB4SignalMode
    target_temperature: Optional[float] = None
    temperature_range: Optional[Tuple[float, float]] = None
    error_injection_rate: float = 0.0
    power_cycle_interval: float = 60.0
    bandwidth_utilization: float = 0.95
    monitoring_interval: float = 1.0
    failure_threshold: int = 10
    enable_recovery_testing: bool = True
    enable_real_time_monitoring: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters"""
        # Type validation
        assert isinstance(self.test_type, USB4StressTestType), f"Test type must be USB4StressTestType, got {type(self.test_type)}"
        assert isinstance(self.signal_mode, USB4SignalMode), f"Signal mode must be USB4SignalMode, got {type(self.signal_mode)}"
        assert isinstance(self.duration, (int, float)), f"Duration must be numeric, got {type(self.duration)}"

        # Convert numeric values
        self.duration = float(self.duration)
        self.error_injection_rate = float(self.error_injection_rate)
        self.power_cycle_interval = float(self.power_cycle_interval)
        self.bandwidth_utilization = float(self.bandwidth_utilization)
        self.monitoring_interval = float(self.monitoring_interval)

        # Value validation
        assert self.duration > 0, f"Duration must be positive, got {self.duration}"
        assert (
            0 <= self.error_injection_rate <= 1
        ), f"Error injection rate must be between 0 and 1, got {self.error_injection_rate}"
        assert self.power_cycle_interval > 0, f"Power cycle interval must be positive, got {self.power_cycle_interval}"
        assert (
            0 < self.bandwidth_utilization <= 1
        ), f"Bandwidth utilization must be between 0 and 1, got {self.bandwidth_utilization}"
        assert self.monitoring_interval > 0, f"Monitoring interval must be positive, got {self.monitoring_interval}"
        assert self.failure_threshold > 0, f"Failure threshold must be positive, got {self.failure_threshold}"

        # Temperature validation
        if self.target_temperature is not None:
            assert self.target_temperature > -273.15, "Temperature must be above absolute zero"

        if self.temperature_range is not None:
            assert len(self.temperature_range) == 2, "Temperature range must be (min, max) tuple"
            assert self.temperature_range[0] < self.temperature_range[1], "Temperature range minimum must be less than maximum"


@dataclass
class USB4StressTestResult:
    """Result from USB4 stress test execution"""

    test_type: USB4StressTestType
    status: USB4StressTestStatus
    start_time: float
    end_time: Optional[float]
    duration: float
    error_count: int
    recovery_count: int
    temperature_data: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, temp)
    performance_data: List[Tuple[float, Dict[str, float]]] = field(default_factory=list)  # (timestamp, metrics)
    failure_events: List[Tuple[float, str, str]] = field(default_factory=list)  # (timestamp, type, description)
    metadata: Dict[str, Union[str, float, bool, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result data"""
        assert isinstance(self.test_type, USB4StressTestType), f"Test type must be USB4StressTestType, got {type(self.test_type)}"
        assert isinstance(self.status, USB4StressTestStatus), f"Status must be USB4StressTestStatus, got {type(self.status)}"
        assert self.start_time > 0, "Start time must be positive"
        assert self.duration >= 0, "Duration must be non-negative"
        assert self.error_count >= 0, "Error count must be non-negative"
        assert self.recovery_count >= 0, "Recovery count must be non-negative"


@dataclass
class USB4StressTestSummary:
    """Summary of USB4 stress test results"""

    total_tests: int
    passed_tests: int
    failed_tests: int
    total_duration: float
    total_errors: int
    total_recoveries: int
    temperature_stats: Dict[str, float]
    performance_stats: Dict[str, float]
    reliability_metrics: Dict[str, float]

    def __post_init__(self) -> None:
        """Validate summary data"""
        assert self.total_tests >= 0, "Total tests must be non-negative"
        assert self.passed_tests >= 0, "Passed tests must be non-negative"
        assert self.failed_tests >= 0, "Failed tests must be non-negative"
        assert self.passed_tests + self.failed_tests <= self.total_tests, "Passed + failed tests cannot exceed total tests"
        assert self.total_duration >= 0, "Total duration must be non-negative"
        assert self.total_errors >= 0, "Total errors must be non-negative"
        assert self.total_recoveries >= 0, "Total recoveries must be non-negative"


class USB4StressTester:
    """USB4 stress testing class with comprehensive testing capabilities"""

    def __init__(self, config: USB4StressTestConfig) -> None:
        """Initialize USB4 stress tester with validated configuration"""
        # Validate input
        assert isinstance(config, USB4StressTestConfig), f"Config must be USB4StressTestConfig, got {type(config)}"

        self.config = config
        self.results: List[USB4StressTestResult] = []
        self.current_test: Optional[USB4StressTestResult] = None
        self.is_running = False
        self.should_stop = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Initialize specifications
        self.usb4_specs = USB4Specs()
        self.signal_specs = USB4SignalSpecs()
        self.tunneling_specs = USB4TunnelingSpecs()
        self.thunderbolt_specs = ThunderboltSpecs()

        # Initialize stress test parameters
        self._initialize_stress_parameters()

        logger.info(f"USB4 stress tester initialized for {config.test_type.name} testing")

    def _initialize_stress_parameters(self) -> None:
        """Initialize stress test parameters based on configuration"""
        # Set temperature limits based on Thunderbolt specs if available
        if self.config.temperature_range is None:
            self.config.temperature_range = self.thunderbolt_specs.CERT_TEMPERATURE_RANGE

        # Set bandwidth targets based on signal mode
        if self.config.signal_mode == USB4SignalMode.GEN2X2:
            self.max_bandwidth = self.usb4_specs.GEN2_RATE * self.usb4_specs.MAX_LANES
        elif self.config.signal_mode == USB4SignalMode.GEN3X2:
            self.max_bandwidth = self.usb4_specs.GEN3_RATE * self.usb4_specs.MAX_LANES
        else:  # ASYMMETRIC
            self.max_bandwidth = self.usb4_specs.TOTAL_BANDWIDTH

        # Set stress test thresholds
        self.error_rate_threshold = 1e-9  # 1 error per billion bits
        self.temperature_alert_threshold = 80.0  # °C
        self.recovery_time_threshold = 1.0  # seconds

    def run_stability_test(
        self,
        data_generator: Callable[[], bytes],
        data_validator: Callable[[bytes], bool],
        temperature_monitor: Optional[Callable[[], float]] = None,
    ) -> USB4StressTestResult:
        """
        Run long-duration stability stress test

        Args:
            data_generator: Function to generate test data
            data_validator: Function to validate received data
            temperature_monitor: Optional function to monitor temperature

        Returns:
            USB4StressTestResult with stability test results
        """
        # Validate inputs
        assert callable(data_generator), "Data generator must be callable"
        assert callable(data_validator), "Data validator must be callable"
        if temperature_monitor is not None:
            assert callable(temperature_monitor), "Temperature monitor must be callable"

        logger.info(f"Starting stability test for {self.config.duration} seconds")

        # Initialize test result
        result = USB4StressTestResult(
            test_type=USB4StressTestType.STABILITY,
            status=USB4StressTestStatus.RUNNING,
            start_time=time.time(),
            end_time=None,
            duration=0.0,
            error_count=0,
            recovery_count=0,
            metadata={
                "signal_mode": self.config.signal_mode.name,
                "target_duration": self.config.duration,
                "bandwidth_utilization": self.config.bandwidth_utilization,
            },
        )

        self.current_test = result
        self.is_running = True
        self.should_stop = False

        try:
            # Start monitoring thread if enabled
            if self.config.enable_real_time_monitoring:
                self._start_monitoring_thread(temperature_monitor)

            start_time = time.time()
            last_monitor_time = start_time
            bytes_transmitted = 0

            while (time.time() - start_time) < self.config.duration and not self.should_stop:
                try:
                    # Generate and transmit test data
                    test_data = data_generator()
                    bytes_transmitted += len(test_data)

                    # Simulate data transmission and validation
                    # In a real implementation, this would involve actual USB4 communication
                    time.sleep(0.001)  # Simulate transmission time

                    # Validate received data
                    if not data_validator(test_data):
                        result.error_count += 1
                        result.failure_events.append((time.time(), "DATA_VALIDATION", "Data validation failed"))

                        # Attempt recovery if enabled
                        if self.config.enable_recovery_testing:
                            recovery_success = self._attempt_recovery()
                            if recovery_success:
                                result.recovery_count += 1

                    # Periodic monitoring
                    current_time = time.time()
                    if current_time - last_monitor_time >= self.config.monitoring_interval:
                        self._record_performance_data(result, current_time, bytes_transmitted)
                        last_monitor_time = current_time

                    # Check failure threshold
                    if result.error_count >= self.config.failure_threshold:
                        logger.error(f"Failure threshold exceeded: {result.error_count} errors")
                        result.status = USB4StressTestStatus.FAILED
                        break

                except Exception as e:
                    logger.error(f"Error during stability test: {e}")
                    result.error_count += 1
                    result.failure_events.append((time.time(), "EXCEPTION", str(e)))

            # Finalize test result
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time

            if result.status == USB4StressTestStatus.RUNNING:
                if self.should_stop:
                    result.status = USB4StressTestStatus.ABORTED
                else:
                    result.status = USB4StressTestStatus.COMPLETED

            result.metadata.update(
                {
                    "bytes_transmitted": bytes_transmitted,
                    "average_throughput": bytes_transmitted * 8 / result.duration if result.duration > 0 else 0,
                    "error_rate": result.error_count / bytes_transmitted if bytes_transmitted > 0 else 0,
                    "recovery_rate": result.recovery_count / result.error_count if result.error_count > 0 else 0,
                }
            )

            self.results.append(result)
            logger.info(f"Stability test completed: {result.status.name}")

            return result

        except Exception as e:
            logger.error(f"Stability test failed: {e}")
            result.status = USB4StressTestStatus.FAILED
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
            result.failure_events.append((time.time(), "CRITICAL_ERROR", str(e)))
            self.results.append(result)
            raise
        finally:
            self.is_running = False
            self._stop_monitoring_thread()

    def run_thermal_stress_test(
        self,
        temperature_controller: Callable[[float], bool],
        performance_monitor: Callable[[], Dict[str, float]],
        temperature_monitor: Callable[[], float],
    ) -> USB4StressTestResult:
        """
        Run thermal stress test with temperature monitoring

        Args:
            temperature_controller: Function to control temperature
            performance_monitor: Function to monitor performance metrics
            temperature_monitor: Function to monitor current temperature

        Returns:
            USB4StressTestResult with thermal test results
        """
        # Validate inputs
        assert callable(temperature_controller), "Temperature controller must be callable"
        assert callable(performance_monitor), "Performance monitor must be callable"
        assert callable(temperature_monitor), "Temperature monitor must be callable"

        logger.info(f"Starting thermal stress test for {self.config.duration} seconds")

        # Initialize test result
        result = USB4StressTestResult(
            test_type=USB4StressTestType.THERMAL,
            status=USB4StressTestStatus.RUNNING,
            start_time=time.time(),
            end_time=None,
            duration=0.0,
            error_count=0,
            recovery_count=0,
            metadata={
                "signal_mode": self.config.signal_mode.name,
                "target_temperature": self.config.target_temperature,
                "temperature_range": self.config.temperature_range,
            },
        )

        self.current_test = result
        self.is_running = True
        self.should_stop = False

        try:
            start_time = time.time()
            last_monitor_time = start_time

            # Temperature cycling parameters
            temp_min, temp_max = self.config.temperature_range
            temp_cycle_duration = self.config.duration / 10  # 10 temperature cycles
            temp_step_time = temp_cycle_duration / 20  # 20 steps per cycle

            cycle_count = 0

            while (time.time() - start_time) < self.config.duration and not self.should_stop:
                try:
                    current_time = time.time()
                    elapsed_time = current_time - start_time

                    # Calculate target temperature for current cycle
                    cycle_progress = (elapsed_time % temp_cycle_duration) / temp_cycle_duration
                    if cycle_progress < 0.5:
                        # Heating phase
                        target_temp = temp_min + (temp_max - temp_min) * (cycle_progress * 2)
                    else:
                        # Cooling phase
                        target_temp = temp_max - (temp_max - temp_min) * ((cycle_progress - 0.5) * 2)

                    # Control temperature
                    temp_control_success = temperature_controller(target_temp)
                    if not temp_control_success:
                        result.error_count += 1
                        result.failure_events.append(
                            (current_time, "TEMPERATURE_CONTROL", f"Failed to reach target temperature {target_temp:.1f}°C")
                        )

                    # Monitor current temperature
                    current_temp = temperature_monitor()
                    result.temperature_data.append((current_time, current_temp))

                    # Check temperature alerts
                    if current_temp > self.temperature_alert_threshold:
                        result.failure_events.append(
                            (current_time, "TEMPERATURE_ALERT", f"Temperature exceeded alert threshold: {current_temp:.1f}°C")
                        )

                    # Monitor performance during thermal stress
                    if current_time - last_monitor_time >= self.config.monitoring_interval:
                        performance_metrics = performance_monitor()
                        result.performance_data.append((current_time, performance_metrics))

                        # Check for performance degradation
                        if "throughput" in performance_metrics:
                            expected_throughput = self.max_bandwidth * self.config.bandwidth_utilization
                            if performance_metrics["throughput"] < expected_throughput * 0.8:  # 20% degradation
                                result.error_count += 1
                                result.failure_events.append(
                                    (
                                        current_time,
                                        "PERFORMANCE_DEGRADATION",
                                        f"Throughput dropped to {performance_metrics['throughput']:.0f} bps",
                                    )
                                )

                        last_monitor_time = current_time

                    # Check if we completed a temperature cycle
                    new_cycle_count = int(elapsed_time / temp_cycle_duration)
                    if new_cycle_count > cycle_count:
                        cycle_count = new_cycle_count
                        logger.info(f"Completed thermal cycle {cycle_count}")

                    time.sleep(temp_step_time)

                except Exception as e:
                    logger.error(f"Error during thermal stress test: {e}")
                    result.error_count += 1
                    result.failure_events.append((time.time(), "EXCEPTION", str(e)))

            # Finalize test result
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time

            if result.status == USB4StressTestStatus.RUNNING:
                if self.should_stop:
                    result.status = USB4StressTestStatus.ABORTED
                else:
                    result.status = USB4StressTestStatus.COMPLETED

            # Calculate thermal statistics
            if result.temperature_data:
                temperatures = [temp for _, temp in result.temperature_data]
                result.metadata.update(
                    {
                        "min_temperature": min(temperatures),
                        "max_temperature": max(temperatures),
                        "avg_temperature": sum(temperatures) / len(temperatures),
                        "temperature_cycles": cycle_count,
                        "temperature_samples": len(temperatures),
                    }
                )

            self.results.append(result)
            logger.info(f"Thermal stress test completed: {result.status.name}")

            return result

        except Exception as e:
            logger.error(f"Thermal stress test failed: {e}")
            result.status = USB4StressTestStatus.FAILED
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
            self.results.append(result)
            raise
        finally:
            self.is_running = False

    def run_error_injection_test(
        self,
        error_injector: Callable[[float], bool],
        recovery_monitor: Callable[[], bool],
        performance_monitor: Callable[[], Dict[str, float]],
    ) -> USB4StressTestResult:
        """
        Run error injection and recovery stress test

        Args:
            error_injector: Function to inject errors at specified rate
            recovery_monitor: Function to monitor recovery status
            performance_monitor: Function to monitor performance during recovery

        Returns:
            USB4StressTestResult with error injection test results
        """
        # Validate inputs
        assert callable(error_injector), "Error injector must be callable"
        assert callable(recovery_monitor), "Recovery monitor must be callable"
        assert callable(performance_monitor), "Performance monitor must be callable"

        logger.info(f"Starting error injection test for {self.config.duration} seconds")

        # Initialize test result
        result = USB4StressTestResult(
            test_type=USB4StressTestType.ERROR_INJECTION,
            status=USB4StressTestStatus.RUNNING,
            start_time=time.time(),
            end_time=None,
            duration=0.0,
            error_count=0,
            recovery_count=0,
            metadata={
                "signal_mode": self.config.signal_mode.name,
                "error_injection_rate": self.config.error_injection_rate,
                "recovery_enabled": self.config.enable_recovery_testing,
            },
        )

        self.current_test = result
        self.is_running = True
        self.should_stop = False

        try:
            start_time = time.time()
            last_monitor_time = start_time
            last_error_injection = start_time

            # Calculate error injection interval
            if self.config.error_injection_rate > 0:
                error_injection_interval = 1.0 / self.config.error_injection_rate
            else:
                error_injection_interval = float("inf")  # No error injection

            while (time.time() - start_time) < self.config.duration and not self.should_stop:
                try:
                    current_time = time.time()

                    # Inject errors at specified rate
                    if (current_time - last_error_injection) >= error_injection_interval:
                        error_injected = error_injector(self.config.error_injection_rate)
                        if error_injected:
                            result.error_count += 1
                            result.failure_events.append((current_time, "ERROR_INJECTION", "Error injected into USB4 link"))

                            # Monitor recovery if enabled
                            if self.config.enable_recovery_testing:
                                recovery_start_time = current_time
                                recovery_success = False
                                recovery_timeout = recovery_start_time + self.recovery_time_threshold

                                while current_time < recovery_timeout and not self.should_stop:
                                    if recovery_monitor():
                                        recovery_success = True
                                        recovery_time = current_time - recovery_start_time
                                        result.recovery_count += 1
                                        result.failure_events.append(
                                            (current_time, "ERROR_RECOVERY", f"Recovery successful in {recovery_time:.3f}s")
                                        )
                                        break
                                    time.sleep(0.01)  # 10ms polling
                                    current_time = time.time()

                                if not recovery_success:
                                    result.failure_events.append(
                                        (
                                            current_time,
                                            "RECOVERY_TIMEOUT",
                                            f"Recovery failed within {self.recovery_time_threshold}s",
                                        )
                                    )

                        last_error_injection = current_time

                    # Monitor performance during error injection
                    if current_time - last_monitor_time >= self.config.monitoring_interval:
                        performance_metrics = performance_monitor()
                        result.performance_data.append((current_time, performance_metrics))
                        last_monitor_time = current_time

                    time.sleep(0.01)  # 10ms loop interval

                except Exception as e:
                    logger.error(f"Error during error injection test: {e}")
                    result.failure_events.append((time.time(), "EXCEPTION", str(e)))

            # Finalize test result
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time

            if result.status == USB4StressTestStatus.RUNNING:
                if self.should_stop:
                    result.status = USB4StressTestStatus.ABORTED
                else:
                    result.status = USB4StressTestStatus.COMPLETED

            # Calculate error injection statistics
            result.metadata.update(
                {
                    "errors_injected": result.error_count,
                    "successful_recoveries": result.recovery_count,
                    "recovery_success_rate": result.recovery_count / result.error_count if result.error_count > 0 else 0,
                    "average_recovery_time": self._calculate_average_recovery_time(result.failure_events),
                }
            )

            self.results.append(result)
            logger.info(f"Error injection test completed: {result.status.name}")

            return result

        except Exception as e:
            logger.error(f"Error injection test failed: {e}")
            result.status = USB4StressTestStatus.FAILED
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
            self.results.append(result)
            raise
        finally:
            self.is_running = False

    def _attempt_recovery(self) -> bool:
        """Attempt to recover from an error condition"""
        try:
            # Simulate recovery procedure
            # In a real implementation, this would involve:
            # - Link reset
            # - Re-training
            # - State machine reset
            # - Buffer flush
            time.sleep(0.1)  # Simulate recovery time

            # Simulate 90% recovery success rate
            return np.random.random() < 0.9

        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return False

    def _start_monitoring_thread(self, temperature_monitor: Optional[Callable[[], float]]) -> None:
        """Start background monitoring thread"""

        def monitoring_loop():
            while self.is_running and not self.should_stop:
                try:
                    if self.current_test and temperature_monitor:
                        current_temp = temperature_monitor()
                        self.current_test.temperature_data.append((time.time(), current_temp))

                    time.sleep(self.config.monitoring_interval)

                except Exception as e:
                    logger.error(f"Monitoring thread error: {e}")

        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def _stop_monitoring_thread(self) -> None:
        """Stop background monitoring thread"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)

    def _record_performance_data(self, result: USB4StressTestResult, timestamp: float, bytes_transmitted: int) -> None:
        """Record performance data during test execution"""
        elapsed_time = timestamp - result.start_time
        if elapsed_time > 0:
            current_throughput = (bytes_transmitted * 8) / elapsed_time

            performance_data = {
                "throughput_bps": current_throughput,
                "utilization": current_throughput / self.max_bandwidth,
                "error_rate": result.error_count / bytes_transmitted if bytes_transmitted > 0 else 0,
                "elapsed_time": elapsed_time,
            }

            result.performance_data.append((timestamp, performance_data))

    def _calculate_average_recovery_time(self, failure_events: List[Tuple[float, str, str]]) -> float:
        """Calculate average recovery time from failure events"""
        recovery_times = []

        for _i, (_timestamp, event_type, description) in enumerate(failure_events):
            if event_type == "ERROR_RECOVERY" and "successful in" in description:
                try:
                    # Extract recovery time from description
                    time_str = description.split("successful in ")[1].split("s")[0]
                    recovery_time = float(time_str)
                    recovery_times.append(recovery_time)
                except (IndexError, ValueError):
                    continue

        return float(np.mean(recovery_times)) if recovery_times else 0.0

    def stop_test(self) -> None:
        """Stop currently running stress test"""
        logger.info("Stopping stress test")
        self.should_stop = True

    def pause_test(self) -> None:
        """Pause currently running stress test"""
        if self.current_test and self.current_test.status == USB4StressTestStatus.RUNNING:
            self.current_test.status = USB4StressTestStatus.PAUSED
            logger.info("Stress test paused")

    def resume_test(self) -> None:
        """Resume paused stress test"""
        if self.current_test and self.current_test.status == USB4StressTestStatus.PAUSED:
            self.current_test.status = USB4StressTestStatus.RUNNING
            logger.info("Stress test resumed")

    def get_test_status(self) -> Optional[USB4StressTestStatus]:
        """Get current test status"""
        return self.current_test.status if self.current_test else None

    def get_real_time_metrics(self) -> Dict[str, Union[str, float, int]]:
        """Get real-time test metrics"""
        if not self.current_test:
            return {}

        current_time = time.time()
        elapsed_time = current_time - self.current_test.start_time

        metrics = {
            "test_type": self.current_test.test_type.name,
            "status": self.current_test.status.name,
            "elapsed_time": elapsed_time,
            "remaining_time": max(0, self.config.duration - elapsed_time),
            "progress_percent": min(100, (elapsed_time / self.config.duration) * 100),
            "error_count": self.current_test.error_count,
            "recovery_count": self.current_test.recovery_count,
            "failure_events": len(self.current_test.failure_events),
        }

        # Add latest temperature if available
        if self.current_test.temperature_data:
            latest_temp = self.current_test.temperature_data[-1][1]
            metrics["current_temperature"] = latest_temp

        # Add latest performance data if available
        if self.current_test.performance_data:
            latest_performance = self.current_test.performance_data[-1][1]
            metrics.update(latest_performance)

        return metrics

    def generate_stress_test_summary(self) -> USB4StressTestSummary:
        """Generate comprehensive stress test summary"""
        if not self.results:
            return USB4StressTestSummary(
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                total_duration=0.0,
                total_errors=0,
                total_recoveries=0,
                temperature_stats={},
                performance_stats={},
                reliability_metrics={},
            )

        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == USB4StressTestStatus.COMPLETED)
        failed_tests = sum(1 for r in self.results if r.status == USB4StressTestStatus.FAILED)
        total_duration = sum(r.duration for r in self.results)
        total_errors = sum(r.error_count for r in self.results)
        total_recoveries = sum(r.recovery_count for r in self.results)

        # Calculate temperature statistics
        all_temperatures = []
        for result in self.results:
            all_temperatures.extend([temp for _, temp in result.temperature_data])

        temperature_stats = {}
        if all_temperatures:
            temperature_stats = {
                "min_temperature": float(np.min(all_temperatures)),
                "max_temperature": float(np.max(all_temperatures)),
                "avg_temperature": float(np.mean(all_temperatures)),
                "std_temperature": float(np.std(all_temperatures)),
            }

        # Calculate performance statistics
        all_throughputs = []
        all_utilizations = []
        for result in self.results:
            for _, perf_data in result.performance_data:
                if "throughput_bps" in perf_data:
                    all_throughputs.append(perf_data["throughput_bps"])
                if "utilization" in perf_data:
                    all_utilizations.append(perf_data["utilization"])

        performance_stats = {}
        if all_throughputs:
            performance_stats.update(
                {
                    "avg_throughput_bps": float(np.mean(all_throughputs)),
                    "min_throughput_bps": float(np.min(all_throughputs)),
                    "max_throughput_bps": float(np.max(all_throughputs)),
                }
            )

        if all_utilizations:
            performance_stats.update(
                {
                    "avg_utilization": float(np.mean(all_utilizations)),
                    "min_utilization": float(np.min(all_utilizations)),
                    "max_utilization": float(np.max(all_utilizations)),
                }
            )

        # Calculate reliability metrics
        reliability_metrics = {
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "error_rate": total_errors / total_duration if total_duration > 0 else 0.0,
            "recovery_rate": total_recoveries / total_errors if total_errors > 0 else 0.0,
            "mtbf_hours": total_duration / (3600 * total_errors) if total_errors > 0 else float("inf"),
            "availability": (total_duration - sum(len(r.failure_events) * 0.1 for r in self.results)) / total_duration
            if total_duration > 0
            else 1.0,
        }

        return USB4StressTestSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            total_duration=total_duration,
            total_errors=total_errors,
            total_recoveries=total_recoveries,
            temperature_stats=temperature_stats,
            performance_stats=performance_stats,
            reliability_metrics=reliability_metrics,
        )

    def generate_stress_test_report(self) -> Dict[str, Union[str, Dict, List, bool]]:
        """Generate comprehensive stress test report"""
        summary = self.generate_stress_test_summary()

        # Generate recommendations based on results
        recommendations = []

        if summary.reliability_metrics.get("pass_rate", 0) < 0.9:
            recommendations.append(
                "Low pass rate detected. Investigate system stability and " "consider improving error handling mechanisms."
            )

        if summary.reliability_metrics.get("error_rate", 0) > self.error_rate_threshold:
            recommendations.append(
                "High error rate detected. Check signal integrity, power supply quality, " "and thermal management."
            )

        if summary.reliability_metrics.get("recovery_rate", 1.0) < 0.8:
            recommendations.append("Low recovery rate detected. Improve error recovery algorithms " "and reduce recovery time.")

        if summary.temperature_stats.get("max_temperature", 0) > self.temperature_alert_threshold:
            recommendations.append(
                f"High temperatures detected (max: {summary.temperature_stats.get('max_temperature', 0):.1f}°C). "
                "Improve thermal management and cooling."
            )

        if not recommendations:
            recommendations.append("All stress tests passed successfully. System demonstrates good reliability.")

        return {
            "test_configuration": {
                "test_type": self.config.test_type.name,
                "signal_mode": self.config.signal_mode.name,
                "duration": self.config.duration,
                "error_injection_rate": self.config.error_injection_rate,
                "bandwidth_utilization": self.config.bandwidth_utilization,
                "temperature_range": self.config.temperature_range,
            },
            "summary_statistics": {
                "total_tests": summary.total_tests,
                "passed_tests": summary.passed_tests,
                "failed_tests": summary.failed_tests,
                "pass_rate_percent": summary.reliability_metrics.get("pass_rate", 0) * 100,
                "total_duration_hours": summary.total_duration / 3600,
                "total_errors": summary.total_errors,
                "total_recoveries": summary.total_recoveries,
            },
            "reliability_metrics": summary.reliability_metrics,
            "temperature_analysis": summary.temperature_stats,
            "performance_analysis": summary.performance_stats,
            "detailed_results": [
                {
                    "test_type": result.test_type.name,
                    "status": result.status.name,
                    "duration": result.duration,
                    "error_count": result.error_count,
                    "recovery_count": result.recovery_count,
                    "failure_events": len(result.failure_events),
                    "metadata": result.metadata,
                }
                for result in self.results
            ],
            "recommendations": recommendations,
            "specification_version": "USB4 v2.0",
            "report_timestamp": time.time(),
        }


# Utility functions for creating stress test configurations
def create_stability_test_config(
    signal_mode: USB4SignalMode = USB4SignalMode.GEN2X2,
    duration: float = 3600.0,  # 1 hour
) -> USB4StressTestConfig:
    """Create USB4 stability stress test configuration"""
    return USB4StressTestConfig(
        test_type=USB4StressTestType.STABILITY,
        duration=duration,
        signal_mode=signal_mode,
        bandwidth_utilization=0.95,
        monitoring_interval=10.0,  # 10 second intervals
        failure_threshold=100,
        enable_recovery_testing=True,
        enable_real_time_monitoring=True,
    )


def create_thermal_stress_config(
    signal_mode: USB4SignalMode = USB4SignalMode.GEN2X2,
    duration: float = 7200.0,  # 2 hours
    temperature_range: Tuple[float, float] = (-40.0, 85.0),
) -> USB4StressTestConfig:
    """Create USB4 thermal stress test configuration"""
    return USB4StressTestConfig(
        test_type=USB4StressTestType.THERMAL,
        duration=duration,
        signal_mode=signal_mode,
        temperature_range=temperature_range,
        bandwidth_utilization=0.90,
        monitoring_interval=5.0,  # 5 second intervals
        failure_threshold=50,
        enable_recovery_testing=True,
        enable_real_time_monitoring=True,
    )


def create_error_injection_config(
    signal_mode: USB4SignalMode = USB4SignalMode.GEN2X2,
    duration: float = 1800.0,  # 30 minutes
    error_injection_rate: float = 0.1,  # 0.1 errors per second
) -> USB4StressTestConfig:
    """Create USB4 error injection stress test configuration"""
    return USB4StressTestConfig(
        test_type=USB4StressTestType.ERROR_INJECTION,
        duration=duration,
        signal_mode=signal_mode,
        error_injection_rate=error_injection_rate,
        bandwidth_utilization=0.80,
        monitoring_interval=1.0,  # 1 second intervals
        failure_threshold=1000,  # Allow many errors for injection testing
        enable_recovery_testing=True,
        enable_real_time_monitoring=True,
    )


__all__ = [
    # Enums
    "USB4StressTestType",
    "USB4StressTestStatus",
    # Data classes
    "USB4StressTestConfig",
    "USB4StressTestResult",
    "USB4StressTestSummary",
    # Main class
    "USB4StressTester",
    # Utility functions
    "create_stability_test_config",
    "create_thermal_stress_config",
    "create_error_injection_config",
]
