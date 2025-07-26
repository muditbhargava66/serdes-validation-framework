"""
PCIe Test Sequence Module

This module provides comprehensive PCIe test sequences with advanced features
including multi-lane support, stress testing, and automated validation workflows.

Features:
- Complete PCIe 6.0 test sequences
- Multi-lane validation
- Stress testing capabilities
- Automated compliance workflows
- Advanced signal integrity analysis
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

from ..instrument_control.pcie_analyzer import PCIeAnalyzer, PCIeConfig
from ..protocols.pcie.compliance import ComplianceConfig, ComplianceTestSuite, ComplianceType
from ..protocols.pcie.constants import SignalMode
from ..protocols.pcie.link_training import create_nrz_trainer, create_pam4_trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCIeTestPhase(Enum):
    """PCIe test phases"""

    INITIALIZATION = auto()
    LINK_TRAINING = auto()
    COMPLIANCE = auto()
    STRESS_TEST = auto()
    VALIDATION = auto()


class PCIeTestResult(Enum):
    """Test result status"""

    PASS = auto()
    FAIL = auto()
    WARNING = auto()
    SKIP = auto()


@dataclass
class LaneConfig:
    """Configuration for individual PCIe lane"""

    lane_id: int
    mode: SignalMode
    sample_rate: float
    bandwidth: float
    voltage_range: float

    def __post_init__(self) -> None:
        """Validate lane configuration"""
        assert isinstance(self.lane_id, int), f"Lane ID must be int, got {type(self.lane_id)}"
        assert isinstance(self.mode, SignalMode), f"Mode must be SignalMode, got {type(self.mode)}"
        assert isinstance(self.sample_rate, float), f"Sample rate must be float, got {type(self.sample_rate)}"
        assert isinstance(self.bandwidth, float), f"Bandwidth must be float, got {type(self.bandwidth)}"
        assert isinstance(self.voltage_range, float), f"Voltage range must be float, got {type(self.voltage_range)}"

        assert self.lane_id >= 0, f"Lane ID must be non-negative, got {self.lane_id}"
        assert self.sample_rate > 0, f"Sample rate must be positive, got {self.sample_rate}"
        assert self.bandwidth > 0, f"Bandwidth must be positive, got {self.bandwidth}"
        assert self.voltage_range > 0, f"Voltage range must be positive, got {self.voltage_range}"


@dataclass
class PCIeTestSequenceConfig:
    """Configuration for PCIe test sequence"""

    test_name: str
    lanes: List[LaneConfig]
    test_phases: List[PCIeTestPhase]
    stress_duration: float = 60.0  # seconds
    compliance_patterns: List[str] = field(default_factory=lambda: ["PRBS31", "PRBS15"])
    target_ber: float = 1e-12

    def __post_init__(self) -> None:
        """Validate test sequence configuration"""
        assert isinstance(self.test_name, str), f"Test name must be string, got {type(self.test_name)}"
        assert isinstance(self.lanes, list), f"Lanes must be list, got {type(self.lanes)}"
        assert all(isinstance(lane, LaneConfig) for lane in self.lanes), "All lanes must be LaneConfig instances"
        assert isinstance(self.test_phases, list), f"Test phases must be list, got {type(self.test_phases)}"
        assert all(
            isinstance(phase, PCIeTestPhase) for phase in self.test_phases
        ), "All test phases must be PCIeTestPhase enum values"
        assert isinstance(self.stress_duration, float), f"Stress duration must be float, got {type(self.stress_duration)}"
        assert isinstance(self.target_ber, float), f"Target BER must be float, got {type(self.target_ber)}"

        assert len(self.lanes) > 0, "Must have at least one lane"
        assert self.stress_duration > 0, f"Stress duration must be positive, got {self.stress_duration}"
        assert 0 < self.target_ber < 1, f"Target BER must be between 0 and 1, got {self.target_ber}"


@dataclass
class PhaseResult:
    """Result of a test phase"""

    phase: PCIeTestPhase
    status: PCIeTestResult
    duration: float
    metrics: Dict[str, float]
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate phase result"""
        assert isinstance(self.phase, PCIeTestPhase), f"Phase must be PCIeTestPhase, got {type(self.phase)}"
        assert isinstance(self.status, PCIeTestResult), f"Status must be PCIeTestResult, got {type(self.status)}"
        assert isinstance(self.duration, float), f"Duration must be float, got {type(self.duration)}"
        assert isinstance(self.metrics, dict), f"Metrics must be dict, got {type(self.metrics)}"

        assert self.duration >= 0, f"Duration must be non-negative, got {self.duration}"
        assert all(isinstance(k, str) for k in self.metrics.keys()), "Metric keys must be strings"
        assert all(isinstance(v, float) for v in self.metrics.values()), "Metric values must be floats"


@dataclass
class SequenceResult:
    """Complete test sequence result"""

    config: PCIeTestSequenceConfig
    overall_status: PCIeTestResult
    total_duration: float
    phase_results: List[PhaseResult]
    lane_results: Dict[int, Dict[str, float]]

    def __post_init__(self) -> None:
        """Validate sequence result"""
        assert isinstance(self.config, PCIeTestSequenceConfig), f"Config must be PCIeTestSequenceConfig, got {type(self.config)}"
        assert isinstance(
            self.overall_status, PCIeTestResult
        ), f"Overall status must be PCIeTestResult, got {type(self.overall_status)}"
        assert isinstance(self.total_duration, float), f"Total duration must be float, got {type(self.total_duration)}"
        assert isinstance(self.phase_results, list), f"Phase results must be list, got {type(self.phase_results)}"
        assert isinstance(self.lane_results, dict), f"Lane results must be dict, got {type(self.lane_results)}"

        assert self.total_duration >= 0, f"Total duration must be non-negative, got {self.total_duration}"
        assert all(
            isinstance(result, PhaseResult) for result in self.phase_results
        ), "All phase results must be PhaseResult instances"


class PCIeTestSequence:
    """Advanced PCIe test sequence with comprehensive validation"""

    def __init__(self, config: PCIeTestSequenceConfig) -> None:
        """
        Initialize PCIe test sequence

        Args:
            config: Test sequence configuration

        Raises:
            AssertionError: If configuration is invalid
        """
        assert isinstance(config, PCIeTestSequenceConfig), f"Config must be PCIeTestSequenceConfig, got {type(config)}"

        self.config = config
        self.analyzers: Dict[int, PCIeAnalyzer] = {}
        self.phase_results: List[PhaseResult] = []

        # Initialize analyzers for each lane
        for lane in config.lanes:
            pcie_config = PCIeConfig(
                mode=lane.mode,
                sample_rate=lane.sample_rate,
                bandwidth=lane.bandwidth,
                voltage_range=lane.voltage_range,
                link_speed=32e9,  # PCIe 6.0 speed
                lane_count=1,
            )
            self.analyzers[lane.lane_id] = PCIeAnalyzer(pcie_config)

        logger.info(f"PCIe test sequence '{config.test_name}' initialized with {len(config.lanes)} lanes")

    def run_complete_sequence(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> SequenceResult:
        """
        Run complete PCIe test sequence

        Args:
            signal_data: Dictionary mapping lane IDs to signal data

        Returns:
            Complete sequence result

        Raises:
            ValueError: If sequence execution fails
        """
        try:
            start_time = time.time()
            self.phase_results = []
            lane_results: Dict[int, Dict[str, float]] = {}

            # Validate signal data
            self._validate_signal_data(signal_data)

            # Run each test phase
            overall_status = PCIeTestResult.PASS
            for phase in self.config.test_phases:
                logger.info(f"Running test phase: {phase.name}")

                phase_result = self._run_test_phase(phase, signal_data)
                self.phase_results.append(phase_result)

                if phase_result.status == PCIeTestResult.FAIL:
                    overall_status = PCIeTestResult.FAIL
                elif phase_result.status == PCIeTestResult.WARNING and overall_status == PCIeTestResult.PASS:
                    overall_status = PCIeTestResult.WARNING

            # Collect lane-specific results
            for lane_id in signal_data.keys():
                lane_results[lane_id] = self._analyze_lane_performance(lane_id, signal_data[lane_id])

            total_duration = time.time() - start_time

            return SequenceResult(
                config=self.config,
                overall_status=overall_status,
                total_duration=total_duration,
                phase_results=self.phase_results,
                lane_results=lane_results,
            )

        except Exception as e:
            logger.error(f"Test sequence failed: {e}")
            raise ValueError(f"PCIe test sequence execution failed: {e}")

    def _validate_signal_data(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> None:
        """Validate signal data for all lanes"""
        assert isinstance(signal_data, dict), f"Signal data must be dict, got {type(signal_data)}"

        # Check that all configured lanes have data
        configured_lanes = {lane.lane_id for lane in self.config.lanes}
        provided_lanes = set(signal_data.keys())

        missing_lanes = configured_lanes - provided_lanes
        assert not missing_lanes, f"Missing signal data for lanes: {missing_lanes}"

        # Validate each lane's signal data
        for lane_id, data in signal_data.items():
            assert isinstance(data, dict), f"Lane {lane_id} data must be dict, got {type(data)}"
            assert "time" in data and "voltage" in data, f"Lane {lane_id} must have 'time' and 'voltage' arrays"

            time_data = data["time"]
            voltage_data = data["voltage"]

            assert isinstance(time_data, np.ndarray), f"Lane {lane_id} time data must be numpy array"
            assert isinstance(voltage_data, np.ndarray), f"Lane {lane_id} voltage data must be numpy array"
            assert len(time_data) == len(voltage_data), f"Lane {lane_id} time and voltage arrays must have same length"

    def _run_test_phase(self, phase: PCIeTestPhase, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> PhaseResult:
        """Run specific test phase"""
        start_time = time.time()

        try:
            if phase == PCIeTestPhase.INITIALIZATION:
                return self._run_initialization_phase(signal_data)
            elif phase == PCIeTestPhase.LINK_TRAINING:
                return self._run_link_training_phase(signal_data)
            elif phase == PCIeTestPhase.COMPLIANCE:
                return self._run_compliance_phase(signal_data)
            elif phase == PCIeTestPhase.STRESS_TEST:
                return self._run_stress_test_phase(signal_data)
            elif phase == PCIeTestPhase.VALIDATION:
                return self._run_validation_phase(signal_data)
            else:
                raise ValueError(f"Unknown test phase: {phase}")

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Phase {phase.name} failed: {e}")
            return PhaseResult(phase=phase, status=PCIeTestResult.FAIL, duration=duration, metrics={}, error_message=str(e))

    def _run_initialization_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> PhaseResult:
        """Run initialization phase"""
        start_time = time.time()

        try:
            metrics = {}

            # Check signal quality for each lane
            for lane_id, data in signal_data.items():
                analyzer = self.analyzers[lane_id]
                results = analyzer.analyze_signal(data)

                # Store key metrics
                for metric, value in results.items():
                    metrics[f"lane_{lane_id}_{metric}"] = value

            duration = time.time() - start_time

            # Determine status based on signal quality
            status = PCIeTestResult.PASS
            for lane_id in signal_data.keys():
                snr_key = f"lane_{lane_id}_snr_db"
                if snr_key in metrics and metrics[snr_key] < 15.0:  # Minimum SNR threshold
                    status = PCIeTestResult.WARNING
                    break

            return PhaseResult(phase=PCIeTestPhase.INITIALIZATION, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"Initialization phase failed: {e}")

    def _run_link_training_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> PhaseResult:
        """Run link training phase"""
        start_time = time.time()

        try:
            metrics = {}
            all_lanes_trained = True

            for lane_id, data in signal_data.items():
                lane_config = next(lane for lane in self.config.lanes if lane.lane_id == lane_id)

                # Create trainer based on mode
                if lane_config.mode == SignalMode.NRZ:
                    trainer = create_nrz_trainer(target_ber=self.config.target_ber)
                else:
                    trainer = create_pam4_trainer(target_ber=self.config.target_ber)

                # Run training
                result = trainer.run_training(data)

                # Store metrics
                metrics[f"lane_{lane_id}_training_success"] = float(result.success)
                metrics[f"lane_{lane_id}_final_ber"] = result.final_ber
                metrics[f"lane_{lane_id}_training_iterations"] = float(result.iterations)
                if result.snr_history:
                    metrics[f"lane_{lane_id}_final_snr"] = result.snr_history[-1]

                if not result.success:
                    all_lanes_trained = False

            duration = time.time() - start_time

            status = PCIeTestResult.PASS if all_lanes_trained else PCIeTestResult.FAIL

            return PhaseResult(phase=PCIeTestPhase.LINK_TRAINING, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"Link training phase failed: {e}")

    def _run_compliance_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> PhaseResult:
        """Run compliance testing phase"""
        start_time = time.time()

        try:
            metrics = {}
            all_lanes_compliant = True

            for lane_id, data in signal_data.items():
                # Create compliance test suite
                config = ComplianceConfig(
                    test_pattern="PRBS31",
                    sample_rate=100e9,
                    record_length=100e-6,
                    voltage_range=2.0,
                    test_types=[ComplianceType.ELECTRICAL, ComplianceType.TIMING],
                )

                test_suite = ComplianceTestSuite(config)

                # Run compliance tests
                results = test_suite.run_compliance_tests(data["time"], data["voltage"])

                # Process results
                lane_compliant = True
                for category, tests in results.items():
                    for test_name, result in tests.items():
                        metric_name = f"lane_{lane_id}_{category}_{test_name}_status"
                        metrics[metric_name] = float(result.status)

                        metric_value = f"lane_{lane_id}_{category}_{test_name}_value"
                        metrics[metric_value] = result.measured_value

                        if not result.status:
                            lane_compliant = False

                if not lane_compliant:
                    all_lanes_compliant = False

            duration = time.time() - start_time

            status = PCIeTestResult.PASS if all_lanes_compliant else PCIeTestResult.FAIL

            return PhaseResult(phase=PCIeTestPhase.COMPLIANCE, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"Compliance phase failed: {e}")

    def _run_stress_test_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> PhaseResult:
        """Run stress testing phase"""
        start_time = time.time()

        try:
            metrics = {}

            # Simulate stress testing by analyzing signal under various conditions
            stress_conditions = [
                ("nominal", 1.0),
                ("high_temp", 0.95),  # Reduced performance at high temperature
                ("low_voltage", 0.90),  # Reduced performance at low voltage
                ("interference", 0.85),  # Reduced performance with interference
            ]

            for condition_name, performance_factor in stress_conditions:
                logger.info(f"Testing under {condition_name} conditions")

                for lane_id, data in signal_data.items():
                    # Simulate degraded signal
                    degraded_voltage = data["voltage"] * performance_factor
                    degraded_data = {"time": data["time"], "voltage": degraded_voltage}

                    # Analyze degraded signal
                    analyzer = self.analyzers[lane_id]
                    results = analyzer.analyze_signal(degraded_data)

                    # Store stress test metrics
                    for metric, value in results.items():
                        stress_metric = f"lane_{lane_id}_{condition_name}_{metric}"
                        metrics[stress_metric] = value

                # Simulate test duration
                time.sleep(min(self.config.stress_duration / len(stress_conditions), 5.0))

            duration = time.time() - start_time

            # Determine status based on stress test results
            status = PCIeTestResult.PASS
            for lane_id in signal_data.keys():
                interference_snr_key = f"lane_{lane_id}_interference_snr_db"
                if interference_snr_key in metrics and metrics[interference_snr_key] < 10.0:
                    status = PCIeTestResult.WARNING
                    break

            return PhaseResult(phase=PCIeTestPhase.STRESS_TEST, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"Stress test phase failed: {e}")

    def _run_validation_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> PhaseResult:
        """Run final validation phase"""
        start_time = time.time()

        try:
            metrics = {}

            # Final validation checks
            for lane_id, data in signal_data.items():
                analyzer = self.analyzers[lane_id]
                results = analyzer.analyze_signal(data)

                # Store final validation metrics
                for metric, value in results.items():
                    metrics[f"lane_{lane_id}_final_{metric}"] = value

            # Cross-lane analysis for multi-lane configurations
            if len(signal_data) > 1:
                lane_ids = list(signal_data.keys())

                # Calculate lane-to-lane skew
                skew_measurements = []
                for i in range(len(lane_ids) - 1):
                    lane1_data = signal_data[lane_ids[i]]["voltage"]
                    lane2_data = signal_data[lane_ids[i + 1]]["voltage"]

                    # Simple cross-correlation for skew estimation
                    correlation = np.correlate(lane1_data[:1000], lane2_data[:1000], mode="full")
                    skew_samples = np.argmax(correlation) - len(lane2_data[:1000]) + 1
                    skew_time = float(skew_samples / 100e9)  # Assuming 100 GSa/s
                    skew_measurements.append(skew_time)

                metrics["max_lane_skew_ps"] = float(max(np.abs(skew_measurements)) * 1e12)
                metrics["avg_lane_skew_ps"] = float(np.mean(np.abs(skew_measurements)) * 1e12)

            duration = time.time() - start_time

            # Determine final validation status
            status = PCIeTestResult.PASS
            if len(signal_data) > 1 and "max_lane_skew_ps" in metrics:
                if metrics["max_lane_skew_ps"] > 50.0:  # 50 ps skew limit
                    status = PCIeTestResult.WARNING

            return PhaseResult(phase=PCIeTestPhase.VALIDATION, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"Validation phase failed: {e}")

    def _analyze_lane_performance(self, lane_id: int, data: Dict[str, npt.NDArray[np.float64]]) -> Dict[str, float]:
        """Analyze overall performance for a specific lane"""
        try:
            analyzer = self.analyzers[lane_id]
            results = analyzer.analyze_signal(data)

            # Add performance score
            if "snr_db" in results:
                snr = results["snr_db"]
                if snr >= 20:
                    performance_score = 100.0
                elif snr >= 15:
                    performance_score = 80.0
                elif snr >= 10:
                    performance_score = 60.0
                else:
                    performance_score = 40.0

                results["performance_score"] = performance_score

            return results

        except Exception as e:
            logger.error(f"Lane {lane_id} performance analysis failed: {e}")
            return {"error": 1.0}


# Factory functions for common test configurations
def create_single_lane_nrz_test(lane_id: int = 0, sample_rate: float = 100e9, bandwidth: float = 50e9) -> PCIeTestSequence:
    """Create single-lane NRZ test sequence"""
    assert isinstance(lane_id, int), f"Lane ID must be int, got {type(lane_id)}"
    assert isinstance(sample_rate, float), f"Sample rate must be float, got {type(sample_rate)}"
    assert isinstance(bandwidth, float), f"Bandwidth must be float, got {type(bandwidth)}"

    lane_config = LaneConfig(
        lane_id=lane_id, mode=SignalMode.NRZ, sample_rate=sample_rate, bandwidth=bandwidth, voltage_range=1.0
    )

    config = PCIeTestSequenceConfig(
        test_name="Single Lane NRZ Test",
        lanes=[lane_config],
        test_phases=[
            PCIeTestPhase.INITIALIZATION,
            PCIeTestPhase.LINK_TRAINING,
            PCIeTestPhase.COMPLIANCE,
            PCIeTestPhase.VALIDATION,
        ],
    )

    return PCIeTestSequence(config)


def create_multi_lane_pam4_test(num_lanes: int = 4, sample_rate: float = 200e9, bandwidth: float = 100e9) -> PCIeTestSequence:
    """Create multi-lane PAM4 test sequence"""
    assert isinstance(num_lanes, int), f"Num lanes must be int, got {type(num_lanes)}"
    assert isinstance(sample_rate, float), f"Sample rate must be float, got {type(sample_rate)}"
    assert isinstance(bandwidth, float), f"Bandwidth must be float, got {type(bandwidth)}"

    assert 1 <= num_lanes <= 16, f"Number of lanes must be between 1 and 16, got {num_lanes}"

    lanes = []
    for i in range(num_lanes):
        lane_config = LaneConfig(lane_id=i, mode=SignalMode.PAM4, sample_rate=sample_rate, bandwidth=bandwidth, voltage_range=1.2)
        lanes.append(lane_config)

    config = PCIeTestSequenceConfig(
        test_name=f"{num_lanes}-Lane PAM4 Test",
        lanes=lanes,
        test_phases=[
            PCIeTestPhase.INITIALIZATION,
            PCIeTestPhase.LINK_TRAINING,
            PCIeTestPhase.COMPLIANCE,
            PCIeTestPhase.STRESS_TEST,
            PCIeTestPhase.VALIDATION,
        ],
        stress_duration=30.0,  # Shorter for multi-lane
    )

    return PCIeTestSequence(config)


def create_comprehensive_pcie_test(mixed_modes: bool = True) -> PCIeTestSequence:
    """Create comprehensive PCIe test with mixed modes"""
    assert isinstance(mixed_modes, bool), f"Mixed modes must be bool, got {type(mixed_modes)}"

    if mixed_modes:
        # Mix of NRZ and PAM4 lanes
        lanes = [
            LaneConfig(0, SignalMode.NRZ, 100e9, 50e9, 1.0),
            LaneConfig(1, SignalMode.PAM4, 200e9, 100e9, 1.2),
            LaneConfig(2, SignalMode.NRZ, 100e9, 50e9, 1.0),
            LaneConfig(3, SignalMode.PAM4, 200e9, 100e9, 1.2),
        ]
        test_name = "Mixed Mode PCIe Test"
    else:
        # All PAM4 lanes
        lanes = [LaneConfig(i, SignalMode.PAM4, 200e9, 100e9, 1.2) for i in range(4)]
        test_name = "4-Lane PAM4 PCIe Test"

    config = PCIeTestSequenceConfig(
        test_name=test_name,
        lanes=lanes,
        test_phases=list(PCIeTestPhase),  # All phases
        stress_duration=60.0,
        target_ber=1e-15,  # Stricter requirement
    )

    return PCIeTestSequence(config)
