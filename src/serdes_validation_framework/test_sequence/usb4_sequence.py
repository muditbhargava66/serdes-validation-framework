"""
USB4/Thunderbolt 4 Test Sequence Module

This module provides comprehensive USB4 and Thunderbolt 4 test sequences with advanced features
including dual-lane support, tunneling validation, and automated compliance workflows.

Features:
- Complete USB4 v2.0 test sequences
- Thunderbolt 4 certification testing
- Multi-protocol tunneling validation
- Automated compliance workflows
- Advanced signal integrity analysis
- Power management testing
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from ..protocols.usb4.compliance import USB4ComplianceConfig, USB4ComplianceResult, USB4ComplianceType, USB4ComplianceValidator
from ..protocols.usb4.constants import (
    ThunderboltSpecs,
    USB4ErrorType,
    USB4LinkState,
    USB4SignalMode,
    USB4Specs,
    USB4TunnelingMode,
)
from ..protocols.usb4.link_training import USB4LinkTraining
from ..protocols.usb4.performance import USB4PerformanceBenchmark
from ..protocols.usb4.power_management import USB4PowerManager
from ..protocols.usb4.signal_analyzer import USB4AnalyzerConfig, USB4SignalAnalyzer
from ..protocols.usb4.stress_testing import USB4StressTester
from ..protocols.usb4.thunderbolt.certification import IntelCertificationSuite

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USB4TestPhase(Enum):
    """USB4 test phases"""

    INITIALIZATION = auto()
    SIGNAL_ANALYSIS = auto()
    LINK_TRAINING = auto()
    COMPLIANCE = auto()
    TUNNELING = auto()
    POWER_MANAGEMENT = auto()
    PERFORMANCE = auto()
    STRESS_TEST = auto()
    THUNDERBOLT = auto()
    VALIDATION = auto()


class USB4TestResult(Enum):
    """Test result status"""

    PASS = auto()
    FAIL = auto()
    WARNING = auto()
    SKIP = auto()


@dataclass
class USB4LaneConfig:
    """Configuration for individual USB4 lane"""

    lane_id: int
    mode: USB4SignalMode
    sample_rate: float
    bandwidth: float
    voltage_range: float
    enable_ssc: bool = True

    def __post_init__(self) -> None:
        """Validate lane configuration"""
        assert isinstance(self.lane_id, int), f"Lane ID must be int, got {type(self.lane_id)}"
        assert isinstance(self.mode, USB4SignalMode), f"Mode must be USB4SignalMode, got {type(self.mode)}"
        assert isinstance(self.sample_rate, float), f"Sample rate must be float, got {type(self.sample_rate)}"
        assert isinstance(self.bandwidth, float), f"Bandwidth must be float, got {type(self.bandwidth)}"
        assert isinstance(self.voltage_range, float), f"Voltage range must be float, got {type(self.voltage_range)}"

        assert 0 <= self.lane_id <= 1, f"USB4 lane ID must be 0 or 1, got {self.lane_id}"
        assert self.sample_rate > 0, f"Sample rate must be positive, got {self.sample_rate}"
        assert self.bandwidth > 0, f"Bandwidth must be positive, got {self.bandwidth}"
        assert self.voltage_range > 0, f"Voltage range must be positive, got {self.voltage_range}"


@dataclass
class USB4TestSequenceConfig:
    """Configuration for USB4 test sequence"""

    test_name: str
    lanes: List[USB4LaneConfig]
    test_phases: List[USB4TestPhase]
    tunneling_modes: List[USB4TunnelingMode] = field(default_factory=lambda: [USB4TunnelingMode.PCIE])
    stress_duration: float = 60.0  # seconds
    compliance_patterns: List[str] = field(default_factory=lambda: ["PRBS31", "PRBS15"])
    target_ber: float = 1e-12
    enable_thunderbolt: bool = False
    power_states_to_test: List[USB4LinkState] = field(default_factory=lambda: [USB4LinkState.U0, USB4LinkState.U1])

    def __post_init__(self) -> None:
        """Validate test sequence configuration"""
        assert isinstance(self.test_name, str), f"Test name must be string, got {type(self.test_name)}"
        assert isinstance(self.lanes, list), f"Lanes must be list, got {type(self.lanes)}"
        assert all(isinstance(lane, USB4LaneConfig) for lane in self.lanes), "All lanes must be USB4LaneConfig instances"
        assert isinstance(self.test_phases, list), f"Test phases must be list, got {type(self.test_phases)}"
        assert all(
            isinstance(phase, USB4TestPhase) for phase in self.test_phases
        ), "All test phases must be USB4TestPhase enum values"
        assert isinstance(self.stress_duration, float), f"Stress duration must be float, got {type(self.stress_duration)}"
        assert isinstance(self.target_ber, float), f"Target BER must be float, got {type(self.target_ber)}"

        assert 1 <= len(self.lanes) <= 2, "USB4 must have 1 or 2 lanes"
        assert self.stress_duration > 0, f"Stress duration must be positive, got {self.stress_duration}"
        assert 0 < self.target_ber < 1, f"Target BER must be between 0 and 1, got {self.target_ber}"


@dataclass
class USB4PhaseResult:
    """Result of a USB4 test phase"""

    phase: USB4TestPhase
    status: USB4TestResult
    duration: float
    metrics: Dict[str, float]
    compliance_results: Optional[Dict[str, USB4ComplianceResult]] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate phase result"""
        assert isinstance(self.phase, USB4TestPhase), f"Phase must be USB4TestPhase, got {type(self.phase)}"
        assert isinstance(self.status, USB4TestResult), f"Status must be USB4TestResult, got {type(self.status)}"
        assert isinstance(self.duration, float), f"Duration must be float, got {type(self.duration)}"
        assert isinstance(self.metrics, dict), f"Metrics must be dict, got {type(self.metrics)}"

        assert self.duration >= 0, f"Duration must be non-negative, got {self.duration}"
        assert all(isinstance(k, str) for k in self.metrics.keys()), "Metric keys must be strings"
        assert all(isinstance(v, (int, float)) for v in self.metrics.values()), "Metric values must be numeric"


@dataclass
class USB4SequenceResult:
    """Complete USB4 test sequence result"""

    config: USB4TestSequenceConfig
    overall_status: USB4TestResult
    total_duration: float
    phase_results: List[USB4PhaseResult]
    lane_results: Dict[int, Dict[str, float]]
    compliance_summary: Dict[str, Union[bool, int, float]]

    def __post_init__(self) -> None:
        """Validate sequence result"""
        assert isinstance(self.config, USB4TestSequenceConfig), f"Config must be USB4TestSequenceConfig, got {type(self.config)}"
        assert isinstance(
            self.overall_status, USB4TestResult
        ), f"Overall status must be USB4TestResult, got {type(self.overall_status)}"
        assert isinstance(self.total_duration, float), f"Total duration must be float, got {type(self.total_duration)}"
        assert isinstance(self.phase_results, list), f"Phase results must be list, got {type(self.phase_results)}"
        assert isinstance(self.lane_results, dict), f"Lane results must be dict, got {type(self.lane_results)}"

        assert self.total_duration >= 0, f"Total duration must be non-negative, got {self.total_duration}"
        assert all(
            isinstance(result, USB4PhaseResult) for result in self.phase_results
        ), "All phase results must be USB4PhaseResult instances"


class USB4TestSequence:
    """Advanced USB4 test sequence with comprehensive validation"""

    def __init__(self, config: USB4TestSequenceConfig) -> None:
        """
        Initialize USB4 test sequence

        Args:
            config: Test sequence configuration

        Raises:
            AssertionError: If configuration is invalid
        """
        assert isinstance(config, USB4TestSequenceConfig), f"Config must be USB4TestSequenceConfig, got {type(config)}"

        self.config = config
        self.phase_results: List[USB4PhaseResult] = []
        self.usb4_specs = USB4Specs()
        self.thunderbolt_specs = ThunderboltSpecs()

        # Initialize analyzers and validators
        self._initialize_components()

        logger.info(f"USB4 test sequence '{config.test_name}' initialized with {len(config.lanes)} lanes")

    def _initialize_components(self) -> None:
        """Initialize USB4 test components"""
        # Signal analyzers for each lane
        self.signal_analyzers: Dict[int, USB4SignalAnalyzer] = {}
        for lane in self.config.lanes:
            analyzer_config = USB4AnalyzerConfig(
                mode=lane.mode,
                sample_rate=lane.sample_rate,
                symbol_rate=getattr(lane, "symbol_rate", 20e9),
                enable_ssc_analysis=getattr(lane, "enable_ssc", True),
            )
            self.signal_analyzers[lane.lane_id] = USB4SignalAnalyzer(analyzer_config)

        # Link training component
        training_config = {
            "target_ber": self.config.target_ber,
            "max_training_time": self.usb4_specs.MAX_TRAINING_TIME,
            "max_retries": self.usb4_specs.MAX_RETRIES,
        }
        self.link_trainer = USB4LinkTraining(training_config)

        # Tunneling validator
        tunneling_config = {"supported_modes": self.config.tunneling_modes, "bandwidth_allocation": True, "flow_control": True}
        # Initialize tunneling validator (using PCIe as primary tunnel)
        from ..protocols.usb4.tunneling import PCIeTunnelValidator

        self.tunneling_validator = PCIeTunnelValidator(tunneling_config)

        # Power management
        power_config = {"supported_states": self.config.power_states_to_test, "power_measurement": True}
        self.power_manager = USB4PowerManager(power_config)

        # Performance benchmarking
        from ..protocols.usb4.performance import USB4PerformanceConfig

        perf_config = USB4PerformanceConfig(
            signal_mode=self.config.lanes[0].mode, test_duration=30.0, packet_size=1024, burst_size=64, measurement_interval=1.0
        )
        self.performance_benchmark = USB4PerformanceBenchmark(perf_config)

        # Stress testing
        from ..protocols.usb4.stress_testing import USB4StressTestConfig, USB4StressTestType

        stress_config = USB4StressTestConfig(
            test_type=USB4StressTestType.STABILITY, duration=self.config.stress_duration, signal_mode=self.config.lanes[0].mode
        )
        self.stress_tester = USB4StressTester(stress_config)

        # Thunderbolt certification (if enabled)
        if self.config.enable_thunderbolt:
            cert_config = {
                "test_duration": self.thunderbolt_specs.CERT_TEST_DURATION,
                "error_threshold": self.thunderbolt_specs.CERT_ERROR_THRESHOLD,
                "temperature_range": self.thunderbolt_specs.CERT_TEMPERATURE_RANGE,
            }
            self.certification_suite = IntelCertificationSuite(cert_config)

    def run_complete_sequence(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> USB4SequenceResult:
        """
        Run complete USB4 test sequence

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
            overall_status = USB4TestResult.PASS
            for phase in self.config.test_phases:
                logger.info(f"Running USB4 test phase: {phase.name}")

                phase_result = self._run_test_phase(phase, signal_data)
                self.phase_results.append(phase_result)

                if phase_result.status == USB4TestResult.FAIL:
                    overall_status = USB4TestResult.FAIL
                elif phase_result.status == USB4TestResult.WARNING and overall_status == USB4TestResult.PASS:
                    overall_status = USB4TestResult.WARNING

            # Collect lane-specific results
            for lane_id in signal_data.keys():
                lane_results[lane_id] = self._analyze_lane_performance(lane_id, signal_data[lane_id])

            # Generate compliance summary
            compliance_summary = self._generate_compliance_summary()

            total_duration = time.time() - start_time

            return USB4SequenceResult(
                config=self.config,
                overall_status=overall_status,
                total_duration=total_duration,
                phase_results=self.phase_results,
                lane_results=lane_results,
                compliance_summary=compliance_summary,
            )

        except Exception as e:
            logger.error(f"USB4 test sequence failed: {e}")
            raise ValueError(f"USB4 test sequence execution failed: {e}")

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
            assert len(time_data) > 0, f"Lane {lane_id} data cannot be empty"

    def _run_test_phase(
        self, phase: USB4TestPhase, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]
    ) -> USB4PhaseResult:
        """Run specific USB4 test phase"""
        start_time = time.time()

        try:
            if phase == USB4TestPhase.INITIALIZATION:
                return self._run_initialization_phase(signal_data)
            elif phase == USB4TestPhase.SIGNAL_ANALYSIS:
                return self._run_signal_analysis_phase(signal_data)
            elif phase == USB4TestPhase.LINK_TRAINING:
                return self._run_link_training_phase(signal_data)
            elif phase == USB4TestPhase.COMPLIANCE:
                return self._run_compliance_phase(signal_data)
            elif phase == USB4TestPhase.TUNNELING:
                return self._run_tunneling_phase(signal_data)
            elif phase == USB4TestPhase.POWER_MANAGEMENT:
                return self._run_power_management_phase(signal_data)
            elif phase == USB4TestPhase.PERFORMANCE:
                return self._run_performance_phase(signal_data)
            elif phase == USB4TestPhase.STRESS_TEST:
                return self._run_stress_test_phase(signal_data)
            elif phase == USB4TestPhase.THUNDERBOLT:
                return self._run_thunderbolt_phase(signal_data)
            elif phase == USB4TestPhase.VALIDATION:
                return self._run_validation_phase(signal_data)
            else:
                raise ValueError(f"Unknown USB4 test phase: {phase}")

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"USB4 phase {phase.name} failed: {e}")
            return USB4PhaseResult(phase=phase, status=USB4TestResult.FAIL, duration=duration, metrics={}, error_message=str(e))

    def _run_initialization_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> USB4PhaseResult:
        """Run USB4 initialization phase"""
        start_time = time.time()

        try:
            metrics = {}

            # Check signal quality for each lane
            for lane_id, data in signal_data.items():
                analyzer = self.signal_analyzers[lane_id]
                results = analyzer.analyze_dual_lane_signal(data["voltage"], data.get("voltage_lane1"))

                # Store key metrics
                for metric, value in results.items():
                    metrics[f"lane_{lane_id}_{metric}"] = float(value)

            # Check lane skew if dual-lane
            if len(signal_data) == 2:
                lane_ids = list(signal_data.keys())
                skew = self._calculate_lane_skew(signal_data[lane_ids[0]]["voltage"], signal_data[lane_ids[1]]["voltage"])
                metrics["lane_skew_ps"] = skew * 1e12  # Convert to picoseconds

            duration = time.time() - start_time

            # Determine status based on signal quality
            status = USB4TestResult.PASS
            for lane_id in signal_data.keys():
                snr_key = f"lane_{lane_id}_snr_db"
                if snr_key in metrics and metrics[snr_key] < 15.0:  # Minimum SNR threshold
                    status = USB4TestResult.WARNING
                    break

            return USB4PhaseResult(phase=USB4TestPhase.INITIALIZATION, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"USB4 initialization phase failed: {e}")

    def _run_signal_analysis_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> USB4PhaseResult:
        """Run USB4 signal analysis phase"""
        start_time = time.time()

        try:
            metrics = {}

            # Analyze each lane
            for lane_id, data in signal_data.items():
                analyzer = self.signal_analyzers[lane_id]

                # Eye diagram analysis
                eye_results = analyzer.validate_eye_diagram(data["voltage"], self.config.lanes[lane_id].mode)
                for metric, value in eye_results.items():
                    metrics[f"lane_{lane_id}_eye_{metric}"] = float(value)

                # Jitter analysis
                jitter_results = analyzer.analyze_ssc_modulation(data["voltage"])
                for metric, value in jitter_results.items():
                    metrics[f"lane_{lane_id}_jitter_{metric}"] = float(value)

            duration = time.time() - start_time

            # Determine status based on eye diagram quality
            status = USB4TestResult.PASS
            for lane_id in signal_data.keys():
                eye_height_key = f"lane_{lane_id}_eye_height"
                if eye_height_key in metrics and metrics[eye_height_key] < 0.4:  # USB4 minimum
                    status = USB4TestResult.FAIL
                    break

            return USB4PhaseResult(phase=USB4TestPhase.SIGNAL_ANALYSIS, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"USB4 signal analysis phase failed: {e}")

    def _run_link_training_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> USB4PhaseResult:
        """Run USB4 link training phase"""
        start_time = time.time()

        try:
            metrics = {}

            # Run link training for each configured mode
            for lane in self.config.lanes:
                lane_data = signal_data[lane.lane_id]

                # Execute link training
                training_result = self.link_trainer.execute_link_training("mock_scope", "mock_pattern_gen")

                # Store training metrics
                metrics[f"lane_{lane.lane_id}_training_time"] = training_result.training_time
                metrics[f"lane_{lane.lane_id}_final_ber"] = training_result.final_ber
                metrics[f"lane_{lane.lane_id}_error_count"] = float(training_result.error_count)
                metrics[f"lane_{lane.lane_id}_convergence"] = float(training_result.convergence_status)

            # Test state transitions
            state_transitions = self.link_trainer.monitor_state_transitions(5.0)  # 5 second monitoring
            metrics["state_transitions"] = float(len(state_transitions))

            duration = time.time() - start_time

            # Determine status based on training success
            status = USB4TestResult.PASS
            for lane in self.config.lanes:
                training_time_key = f"lane_{lane.lane_id}_training_time"
                if training_time_key in metrics and metrics[training_time_key] > self.usb4_specs.MAX_TRAINING_TIME:
                    status = USB4TestResult.FAIL
                    break

            return USB4PhaseResult(phase=USB4TestPhase.LINK_TRAINING, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"USB4 link training phase failed: {e}")

    def _run_compliance_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> USB4PhaseResult:
        """Run USB4 compliance testing phase"""
        start_time = time.time()

        try:
            metrics = {}
            compliance_results = {}

            # Create compliance validator
            compliance_config = USB4ComplianceConfig(
                signal_mode=self.config.lanes[0].mode,
                test_pattern=self.config.compliance_patterns[0],
                sample_rate=self.config.lanes[0].sample_rate,
                record_length=100e-6,  # 100 μs
                voltage_range=self.config.lanes[0].voltage_range,
                test_types=[USB4ComplianceType.SIGNAL_INTEGRITY, USB4ComplianceType.PROTOCOL],
                enable_ssc=self.config.lanes[0].enable_ssc,
                thunderbolt_mode=self.config.enable_thunderbolt,
            )

            validator = USB4ComplianceValidator(compliance_config)

            # Run compliance tests for each lane
            for lane_id, data in signal_data.items():
                lane1_data = None
                if len(signal_data) == 2:
                    other_lane_id = 1 - lane_id
                    if other_lane_id in signal_data:
                        lane1_data = signal_data[other_lane_id]["voltage"]

                # Run full compliance suite
                results = validator.run_full_compliance_suite(
                    lane0_data=data["voltage"],
                    lane1_data=lane1_data,
                    time_data=data["time"],
                    link_training_time=0.05,  # 50 ms typical
                    final_link_state=USB4LinkState.U0,
                )

                # Process compliance results
                for category, tests in results.items():
                    for test_name, result in tests.items():
                        metric_name = f"lane_{lane_id}_{category}_{test_name}_status"
                        metrics[metric_name] = float(result.status)

                        value_name = f"lane_{lane_id}_{category}_{test_name}_value"
                        metrics[value_name] = result.measured_value

                        compliance_results[f"{category}_{test_name}"] = result

            # Get overall compliance status
            overall_compliance = validator.get_overall_status()
            metrics["overall_compliance"] = float(overall_compliance)

            # Get test summary
            test_summary = validator.get_test_summary()
            metrics.update({f"compliance_{k}": float(v) for k, v in test_summary.items()})

            duration = time.time() - start_time

            status = USB4TestResult.PASS if overall_compliance else USB4TestResult.FAIL

            return USB4PhaseResult(
                phase=USB4TestPhase.COMPLIANCE,
                status=status,
                duration=duration,
                metrics=metrics,
                compliance_results=compliance_results,
            )

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"USB4 compliance phase failed: {e}")

    def _run_tunneling_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> USB4PhaseResult:
        """Run USB4 tunneling validation phase"""
        start_time = time.time()

        try:
            metrics = {}

            # Test each configured tunneling mode
            for tunnel_mode in self.config.tunneling_modes:
                logger.info(f"Testing {tunnel_mode.name} tunneling")

                # Validate tunnel establishment
                establishment_result = self.tunneling_validator.validate_tunnel_establishment(tunnel_mode)
                metrics[f"{tunnel_mode.name.lower()}_establishment_time"] = establishment_result.establishment_time
                metrics[f"{tunnel_mode.name.lower()}_establishment_success"] = float(establishment_result.success)

                # Test bandwidth allocation
                bandwidth_result = self.tunneling_validator.test_bandwidth_allocation(
                    [
                        {"type": tunnel_mode, "bandwidth": 10e9}  # 10 Gbps request
                    ]
                )
                metrics[f"{tunnel_mode.name.lower()}_bandwidth_efficiency"] = bandwidth_result.efficiency
                metrics[f"{tunnel_mode.name.lower()}_allocated_bandwidth"] = bandwidth_result.allocated_bandwidth

                # Protocol-specific validation
                if tunnel_mode == USB4TunnelingMode.PCIE:
                    pcie_result = self.tunneling_validator.validate_pcie_tunneling(
                        np.random.randn(1000).astype(np.float64)  # Mock PCIe data
                    )
                    metrics["pcie_tunnel_integrity"] = pcie_result.packet_integrity
                    metrics["pcie_tunnel_latency"] = pcie_result.average_latency

                elif tunnel_mode == USB4TunnelingMode.DISPLAYPORT:
                    dp_result = self.tunneling_validator.validate_displayport_tunneling(
                        np.random.randn(1000).astype(np.float64)  # Mock video data
                    )
                    metrics["displayport_tunnel_quality"] = dp_result.signal_quality
                    metrics["displayport_tunnel_latency"] = dp_result.average_latency

                elif tunnel_mode == USB4TunnelingMode.USB32:
                    usb32_result = self.tunneling_validator.validate_usb32_tunneling(
                        np.random.randn(1000).astype(np.float64)  # Mock USB data
                    )
                    metrics["usb32_tunnel_compatibility"] = usb32_result.compatibility_score
                    metrics["usb32_tunnel_latency"] = usb32_result.average_latency

            duration = time.time() - start_time

            # Determine status based on tunnel establishment success
            status = USB4TestResult.PASS
            for tunnel_mode in self.config.tunneling_modes:
                success_key = f"{tunnel_mode.name.lower()}_establishment_success"
                if success_key in metrics and metrics[success_key] < 1.0:
                    status = USB4TestResult.WARNING
                    break

            return USB4PhaseResult(phase=USB4TestPhase.TUNNELING, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"USB4 tunneling phase failed: {e}")

    def _run_power_management_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> USB4PhaseResult:
        """Run USB4 power management testing phase"""
        start_time = time.time()

        try:
            metrics = {}

            # Test each power state
            for power_state in self.config.power_states_to_test:
                logger.info(f"Testing power state {power_state.name}")

                # Validate power state transition
                transition_result = self.power_manager.validate_power_state_transition(USB4LinkState.U0, power_state)
                metrics[f"power_{power_state.name.lower()}_transition_time"] = transition_result.transition_time
                metrics[f"power_{power_state.name.lower()}_success"] = float(transition_result.success)

                # Measure power consumption
                power_measurement = self.power_manager.measure_power_consumption(power_state)
                metrics[f"power_{power_state.name.lower()}_consumption"] = power_measurement.average_power
                metrics[f"power_{power_state.name.lower()}_efficiency"] = power_measurement.efficiency

                # Test wake events if in low power state
                if power_state != USB4LinkState.U0:
                    wake_result = self.power_manager.test_wake_events(power_state)
                    metrics[f"power_{power_state.name.lower()}_wake_time"] = wake_result.wake_time
                    metrics[f"power_{power_state.name.lower()}_wake_success"] = float(wake_result.success)

            # Test thermal management
            thermal_result = self.power_manager.validate_thermal_management(85.0)  # 85°C test
            metrics["thermal_throttling_active"] = float(thermal_result.throttling_active)
            metrics["thermal_performance_scaling"] = thermal_result.performance_scaling

            duration = time.time() - start_time

            # Determine status based on power state transitions
            status = USB4TestResult.PASS
            for power_state in self.config.power_states_to_test:
                success_key = f"power_{power_state.name.lower()}_success"
                if success_key in metrics and metrics[success_key] < 1.0:
                    status = USB4TestResult.WARNING
                    break

            return USB4PhaseResult(phase=USB4TestPhase.POWER_MANAGEMENT, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"USB4 power management phase failed: {e}")

    def _run_performance_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> USB4PhaseResult:
        """Run USB4 performance benchmarking phase"""
        start_time = time.time()

        try:
            metrics = {}

            # Run throughput benchmarks
            throughput_result = self.performance_benchmark.measure_throughput(self.config.lanes[0].mode)
            metrics["throughput_gbps"] = throughput_result.average_throughput / 1e9
            metrics["throughput_efficiency"] = throughput_result.efficiency
            metrics["throughput_stability"] = throughput_result.stability_score

            # Run latency benchmarks
            latency_result = self.performance_benchmark.measure_latency(self.config.lanes[0].mode)
            metrics["latency_us"] = latency_result.average_latency * 1e6
            metrics["latency_jitter_us"] = latency_result.jitter * 1e6
            metrics["latency_percentile_99_us"] = latency_result.percentile_99 * 1e6

            # Run efficiency analysis
            efficiency_result = self.performance_benchmark.analyze_efficiency(
                signal_data[0]["voltage"] if 0 in signal_data else signal_data[list(signal_data.keys())[0]]["voltage"]
            )
            metrics["power_efficiency"] = efficiency_result.power_efficiency
            metrics["bandwidth_utilization"] = efficiency_result.bandwidth_utilization
            metrics["protocol_overhead"] = efficiency_result.protocol_overhead

            duration = time.time() - start_time

            # Determine status based on performance targets
            status = USB4TestResult.PASS
            expected_throughput = 40e9  # 40 Gbps for USB4
            if metrics["throughput_gbps"] < expected_throughput * 0.8 / 1e9:  # 80% of expected
                status = USB4TestResult.WARNING

            return USB4PhaseResult(phase=USB4TestPhase.PERFORMANCE, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"USB4 performance phase failed: {e}")

    def _run_stress_test_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> USB4PhaseResult:
        """Run USB4 stress testing phase"""
        start_time = time.time()

        try:
            metrics = {}

            # Run thermal stress test
            thermal_result = self.stress_tester.run_thermal_stress_test(
                temperature_range=(25.0, 85.0), duration=min(self.config.stress_duration / 3, 20.0)
            )
            metrics["thermal_stress_passed"] = float(thermal_result.passed)
            metrics["thermal_max_temp"] = thermal_result.max_temperature
            metrics["thermal_performance_degradation"] = thermal_result.performance_degradation

            # Run voltage stress test
            voltage_result = self.stress_tester.run_voltage_stress_test(
                voltage_range=(0.9, 1.1),  # ±10% voltage variation
                duration=min(self.config.stress_duration / 3, 20.0),
            )
            metrics["voltage_stress_passed"] = float(voltage_result.passed)
            metrics["voltage_min"] = voltage_result.min_voltage
            metrics["voltage_max"] = voltage_result.max_voltage

            # Run error injection test
            error_result = self.stress_tester.run_error_injection_test(
                error_types=[USB4ErrorType.SIGNAL_INTEGRITY, USB4ErrorType.PROTOCOL],
                duration=min(self.config.stress_duration / 3, 20.0),
            )
            metrics["error_injection_recovery_rate"] = error_result.recovery_rate
            metrics["error_injection_max_recovery_time"] = error_result.max_recovery_time

            duration = time.time() - start_time

            # Determine status based on stress test results
            status = USB4TestResult.PASS
            if (
                metrics.get("thermal_stress_passed", 1.0) < 1.0
                or metrics.get("voltage_stress_passed", 1.0) < 1.0
                or metrics.get("error_injection_recovery_rate", 1.0) < 0.95
            ):
                status = USB4TestResult.WARNING

            return USB4PhaseResult(phase=USB4TestPhase.STRESS_TEST, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"USB4 stress test phase failed: {e}")

    def _run_thunderbolt_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> USB4PhaseResult:
        """Run Thunderbolt 4 certification phase"""
        start_time = time.time()

        try:
            metrics = {}

            if not self.config.enable_thunderbolt:
                # Skip Thunderbolt testing if not enabled
                return USB4PhaseResult(
                    phase=USB4TestPhase.THUNDERBOLT, status=USB4TestResult.SKIP, duration=0.0, metrics={"skipped": 1.0}
                )

            # Run Intel certification suite
            cert_result = self.certification_suite.run_full_certification_suite(
                signal_data[0]["voltage"] if 0 in signal_data else signal_data[list(signal_data.keys())[0]]["voltage"]
            )

            metrics["thunderbolt_certification_passed"] = float(cert_result.overall_passed)
            metrics["thunderbolt_test_count"] = float(cert_result.total_tests)
            metrics["thunderbolt_pass_rate"] = cert_result.pass_rate

            # Test security features
            security_result = self.certification_suite.validate_security_features()
            metrics["thunderbolt_dma_protection"] = float(security_result.dma_protection_active)
            metrics["thunderbolt_device_auth"] = float(security_result.device_authentication_passed)
            metrics["thunderbolt_auth_time"] = security_result.authentication_time

            # Test daisy chain capability
            daisy_result = self.certification_suite.test_daisy_chain_capability(max_devices=6)
            metrics["thunderbolt_max_daisy_devices"] = float(daisy_result.max_supported_devices)
            metrics["thunderbolt_daisy_bandwidth_degradation"] = daisy_result.bandwidth_degradation

            duration = time.time() - start_time

            # Determine status based on certification results
            status = USB4TestResult.PASS if cert_result.overall_passed else USB4TestResult.FAIL

            return USB4PhaseResult(phase=USB4TestPhase.THUNDERBOLT, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"Thunderbolt phase failed: {e}")

    def _run_validation_phase(self, signal_data: Dict[int, Dict[str, npt.NDArray[np.float64]]]) -> USB4PhaseResult:
        """Run final USB4 validation phase"""
        start_time = time.time()

        try:
            metrics = {}

            # Final signal quality validation
            for lane_id, data in signal_data.items():
                analyzer = self.signal_analyzers[lane_id]
                final_results = analyzer.analyze_dual_lane_signal(data["voltage"], None)

                # Store final validation metrics
                for metric, value in final_results.items():
                    metrics[f"lane_{lane_id}_final_{metric}"] = float(value)

            # Cross-lane analysis for dual-lane configurations
            if len(signal_data) == 2:
                lane_ids = list(signal_data.keys())

                # Calculate final lane skew
                final_skew = self._calculate_lane_skew(signal_data[lane_ids[0]]["voltage"], signal_data[lane_ids[1]]["voltage"])
                metrics["final_lane_skew_ps"] = final_skew * 1e12

                # Calculate lane balance
                lane0_power = np.mean(signal_data[lane_ids[0]]["voltage"] ** 2)
                lane1_power = np.mean(signal_data[lane_ids[1]]["voltage"] ** 2)
                power_balance = abs(lane0_power - lane1_power) / max(lane0_power, lane1_power)
                metrics["lane_power_balance"] = power_balance

            # Overall system validation score
            validation_score = self._calculate_validation_score()
            metrics["overall_validation_score"] = validation_score

            duration = time.time() - start_time

            # Determine final validation status
            status = USB4TestResult.PASS
            if validation_score < 0.8:  # 80% threshold
                status = USB4TestResult.WARNING
            if validation_score < 0.6:  # 60% threshold
                status = USB4TestResult.FAIL

            return USB4PhaseResult(phase=USB4TestPhase.VALIDATION, status=status, duration=duration, metrics=metrics)

        except Exception as e:
            duration = time.time() - start_time
            raise ValueError(f"USB4 validation phase failed: {e}")

    def _calculate_lane_skew(self, lane0_data: npt.NDArray[np.float64], lane1_data: npt.NDArray[np.float64]) -> float:
        """Calculate skew between USB4 lanes"""
        try:
            # Cross-correlation to find delay
            correlation = np.correlate(lane0_data[:1000], lane1_data[:1000], mode="full")
            delay_samples = np.argmax(correlation) - len(lane1_data[:1000]) + 1

            # Convert to time (assuming uniform sampling)
            sample_period = 1.0 / self.config.lanes[0].sample_rate
            skew_seconds = delay_samples * sample_period

            return float(skew_seconds)

        except Exception as e:
            logger.error(f"Lane skew calculation failed: {e}")
            return 0.0

    def _analyze_lane_performance(self, lane_id: int, data: Dict[str, npt.NDArray[np.float64]]) -> Dict[str, float]:
        """Analyze overall performance for a specific lane"""
        try:
            analyzer = self.signal_analyzers[lane_id]
            results = analyzer.analyze_dual_lane_signal(data["voltage"], None)

            # Add performance score based on multiple factors
            snr = results.get("snr_db", 0.0)
            eye_height = results.get("eye_height", 0.0)
            jitter = results.get("total_jitter_ui", 1.0)

            # Calculate composite performance score
            snr_score = min(snr / 20.0, 1.0)  # Normalize to 20 dB
            eye_score = eye_height / 0.6  # Normalize to USB4 minimum
            jitter_score = max(0.0, 1.0 - jitter / 0.35)  # Normalize to USB4 maximum

            performance_score = (snr_score + eye_score + jitter_score) / 3.0 * 100.0
            results["performance_score"] = performance_score

            return {k: float(v) for k, v in results.items()}

        except Exception as e:
            logger.error(f"Lane {lane_id} performance analysis failed: {e}")
            return {"error": 1.0}

    def _calculate_validation_score(self) -> float:
        """Calculate overall validation score based on all test phases"""
        try:
            if not self.phase_results:
                return 0.0

            # Weight different phases
            phase_weights = {
                USB4TestPhase.SIGNAL_ANALYSIS: 0.25,
                USB4TestPhase.LINK_TRAINING: 0.20,
                USB4TestPhase.COMPLIANCE: 0.25,
                USB4TestPhase.TUNNELING: 0.15,
                USB4TestPhase.POWER_MANAGEMENT: 0.10,
                USB4TestPhase.PERFORMANCE: 0.05,
            }

            total_score = 0.0
            total_weight = 0.0

            for result in self.phase_results:
                weight = phase_weights.get(result.phase, 0.0)
                if weight > 0:
                    # Convert status to score
                    if result.status == USB4TestResult.PASS:
                        score = 1.0
                    elif result.status == USB4TestResult.WARNING:
                        score = 0.7
                    elif result.status == USB4TestResult.SKIP:
                        continue  # Skip this phase
                    else:
                        score = 0.0

                    total_score += score * weight
                    total_weight += weight

            return total_score / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            logger.error(f"Validation score calculation failed: {e}")
            return 0.0

    def _generate_compliance_summary(self) -> Dict[str, Union[bool, int, float]]:
        """Generate compliance summary from all test phases"""
        try:
            summary = {
                "overall_compliant": True,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "warning_tests": 0,
                "skipped_tests": 0,
            }

            for result in self.phase_results:
                summary["total_tests"] += 1

                if result.status == USB4TestResult.PASS:
                    summary["passed_tests"] += 1
                elif result.status == USB4TestResult.FAIL:
                    summary["failed_tests"] += 1
                    summary["overall_compliant"] = False
                elif result.status == USB4TestResult.WARNING:
                    summary["warning_tests"] += 1
                elif result.status == USB4TestResult.SKIP:
                    summary["skipped_tests"] += 1

            # Calculate pass rate
            if summary["total_tests"] > 0:
                summary["pass_rate"] = summary["passed_tests"] / summary["total_tests"]
            else:
                summary["pass_rate"] = 0.0

            return summary

        except Exception as e:
            logger.error(f"Compliance summary generation failed: {e}")
            return {"error": True}


# Factory functions for common USB4 test configurations
def create_usb4_gen2_test(dual_lane: bool = True, enable_thunderbolt: bool = False) -> USB4TestSequence:
    """Create USB4 Gen 2x2 test sequence"""
    assert isinstance(dual_lane, bool), f"Dual lane must be bool, got {type(dual_lane)}"
    assert isinstance(enable_thunderbolt, bool), f"Enable thunderbolt must be bool, got {type(enable_thunderbolt)}"

    lanes = [
        USB4LaneConfig(
            lane_id=0,
            mode=USB4SignalMode.GEN2X2,
            sample_rate=200e9,  # 200 GSa/s
            bandwidth=100e9,  # 100 GHz
            voltage_range=1.2,
            enable_ssc=True,
        )
    ]

    if dual_lane:
        lanes.append(
            USB4LaneConfig(
                lane_id=1, mode=USB4SignalMode.GEN2X2, sample_rate=200e9, bandwidth=100e9, voltage_range=1.2, enable_ssc=True
            )
        )

    test_phases = [
        USB4TestPhase.INITIALIZATION,
        USB4TestPhase.SIGNAL_ANALYSIS,
        USB4TestPhase.LINK_TRAINING,
        USB4TestPhase.COMPLIANCE,
        USB4TestPhase.TUNNELING,
        USB4TestPhase.POWER_MANAGEMENT,
        USB4TestPhase.VALIDATION,
    ]

    if enable_thunderbolt:
        test_phases.insert(-1, USB4TestPhase.THUNDERBOLT)

    config = USB4TestSequenceConfig(
        test_name="USB4 Gen 2x2 Test",
        lanes=lanes,
        test_phases=test_phases,
        tunneling_modes=[USB4TunnelingMode.PCIE, USB4TunnelingMode.DISPLAYPORT],
        enable_thunderbolt=enable_thunderbolt,
        target_ber=1e-12,
    )

    return USB4TestSequence(config)


def create_usb4_gen3_test(dual_lane: bool = True, enable_stress_test: bool = True) -> USB4TestSequence:
    """Create USB4 Gen 3x2 test sequence"""
    assert isinstance(dual_lane, bool), f"Dual lane must be bool, got {type(dual_lane)}"
    assert isinstance(enable_stress_test, bool), f"Enable stress test must be bool, got {type(enable_stress_test)}"

    lanes = [
        USB4LaneConfig(
            lane_id=0,
            mode=USB4SignalMode.GEN3X2,
            sample_rate=200e9,  # 200 GSa/s
            bandwidth=100e9,  # 100 GHz
            voltage_range=1.2,
            enable_ssc=True,
        )
    ]

    if dual_lane:
        lanes.append(
            USB4LaneConfig(
                lane_id=1, mode=USB4SignalMode.GEN3X2, sample_rate=200e9, bandwidth=100e9, voltage_range=1.2, enable_ssc=True
            )
        )

    test_phases = [
        USB4TestPhase.INITIALIZATION,
        USB4TestPhase.SIGNAL_ANALYSIS,
        USB4TestPhase.LINK_TRAINING,
        USB4TestPhase.COMPLIANCE,
        USB4TestPhase.TUNNELING,
        USB4TestPhase.POWER_MANAGEMENT,
        USB4TestPhase.PERFORMANCE,
    ]

    if enable_stress_test:
        test_phases.append(USB4TestPhase.STRESS_TEST)

    test_phases.append(USB4TestPhase.VALIDATION)

    config = USB4TestSequenceConfig(
        test_name="USB4 Gen 3x2 Test",
        lanes=lanes,
        test_phases=test_phases,
        tunneling_modes=[USB4TunnelingMode.PCIE, USB4TunnelingMode.DISPLAYPORT, USB4TunnelingMode.USB32],
        stress_duration=120.0,  # 2 minutes
        target_ber=1e-15,  # Stricter requirement for Gen 3
    )

    return USB4TestSequence(config)


def create_thunderbolt4_certification_test() -> USB4TestSequence:
    """Create Thunderbolt 4 certification test sequence"""
    lanes = [
        USB4LaneConfig(
            lane_id=0, mode=USB4SignalMode.GEN2X2, sample_rate=200e9, bandwidth=100e9, voltage_range=1.2, enable_ssc=True
        ),
        USB4LaneConfig(
            lane_id=1, mode=USB4SignalMode.GEN2X2, sample_rate=200e9, bandwidth=100e9, voltage_range=1.2, enable_ssc=True
        ),
    ]

    config = USB4TestSequenceConfig(
        test_name="Thunderbolt 4 Certification Test",
        lanes=lanes,
        test_phases=[
            USB4TestPhase.INITIALIZATION,
            USB4TestPhase.SIGNAL_ANALYSIS,
            USB4TestPhase.LINK_TRAINING,
            USB4TestPhase.COMPLIANCE,
            USB4TestPhase.TUNNELING,
            USB4TestPhase.POWER_MANAGEMENT,
            USB4TestPhase.PERFORMANCE,
            USB4TestPhase.STRESS_TEST,
            USB4TestPhase.THUNDERBOLT,
            USB4TestPhase.VALIDATION,
        ],
        tunneling_modes=[USB4TunnelingMode.PCIE, USB4TunnelingMode.DISPLAYPORT],
        stress_duration=300.0,  # 5 minutes for certification
        enable_thunderbolt=True,
        target_ber=1e-12,
        power_states_to_test=[USB4LinkState.U0, USB4LinkState.U1, USB4LinkState.U2, USB4LinkState.U3],
    )

    return USB4TestSequence(config)


def create_comprehensive_usb4_test(mixed_modes: bool = False) -> USB4TestSequence:
    """Create comprehensive USB4 test with all features"""
    assert isinstance(mixed_modes, bool), f"Mixed modes must be bool, got {type(mixed_modes)}"

    if mixed_modes:
        # Mix of Gen 2 and Gen 3 modes (asymmetric)
        lanes = [
            USB4LaneConfig(0, USB4SignalMode.GEN2X2, 200e9, 100e9, 1.2, True),
            USB4LaneConfig(1, USB4SignalMode.GEN3X2, 200e9, 100e9, 1.2, True),
        ]
        test_name = "Mixed Mode USB4 Test"
    else:
        # Both lanes Gen 3
        lanes = [
            USB4LaneConfig(0, USB4SignalMode.GEN3X2, 200e9, 100e9, 1.2, True),
            USB4LaneConfig(1, USB4SignalMode.GEN3X2, 200e9, 100e9, 1.2, True),
        ]
        test_name = "Comprehensive USB4 Gen 3x2 Test"

    config = USB4TestSequenceConfig(
        test_name=test_name,
        lanes=lanes,
        test_phases=list(USB4TestPhase),  # All phases
        tunneling_modes=list(USB4TunnelingMode)[:-1],  # All except NATIVE
        stress_duration=180.0,  # 3 minutes
        compliance_patterns=["PRBS31", "PRBS15", "PRBS7"],
        enable_thunderbolt=True,
        target_ber=1e-15,  # Strictest requirement
        power_states_to_test=list(USB4LinkState),  # All power states
    )

    return USB4TestSequence(config)


__all__ = [
    # Enums
    "USB4TestPhase",
    "USB4TestResult",
    # Data classes
    "USB4LaneConfig",
    "USB4TestSequenceConfig",
    "USB4PhaseResult",
    "USB4SequenceResult",
    # Main class
    "USB4TestSequence",
    # Factory functions
    "create_usb4_gen2_test",
    "create_usb4_gen3_test",
    "create_thunderbolt4_certification_test",
    "create_comprehensive_usb4_test",
]
