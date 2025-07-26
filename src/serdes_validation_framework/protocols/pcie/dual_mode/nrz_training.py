"""
PCIe NRZ Training Module
This module provides functionality for training and optimizing NRZ signal mode
in PCIe dual-mode operation with robust type checking and validation.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class TrainingPhase(Enum):
    """Enumeration of NRZ training phases."""

    DETECT = auto()
    POLLING = auto()
    CONFIG = auto()
    RECOVERY = auto()
    LOOPBACK = auto()
    COMPLETE = auto()


@dataclass
class NRZTrainingConfig:
    """
    Configuration parameters for NRZ training.

    Attributes:
        voltage_swing (float): Target voltage swing in volts.
        pre_emphasis (float): Pre-emphasis level (0.0-1.0).
        de_emphasis (float): De-emphasis level (0.0-1.0).
        eq_boost (float): Equalization boost factor.
        training_pattern (int): Training pattern number to use.
        max_iterations (int): Maximum number of training iterations.
        convergence_threshold (float): Threshold for determining convergence.
        timeout_seconds (float): Timeout in seconds for training operations.
    """

    voltage_swing: float
    pre_emphasis: float
    de_emphasis: float
    eq_boost: float
    training_pattern: int
    max_iterations: int
    convergence_threshold: float
    timeout_seconds: float

    def __post_init__(self) -> None:
        """
        Validate the configuration parameters.

        Raises:
            AssertionError: If any parameter has an invalid type or value.
        """
        # Type validation
        assert isinstance(self.voltage_swing, float), "voltage_swing must be a floating-point number"
        assert isinstance(self.pre_emphasis, float), "pre_emphasis must be a floating-point number"
        assert isinstance(self.de_emphasis, float), "de_emphasis must be a floating-point number"
        assert isinstance(self.eq_boost, float), "eq_boost must be a floating-point number"
        assert isinstance(self.training_pattern, int), "training_pattern must be an integer"
        assert isinstance(self.max_iterations, int), "max_iterations must be an integer"
        assert isinstance(self.convergence_threshold, float), "convergence_threshold must be a floating-point number"
        assert isinstance(self.timeout_seconds, float), "timeout_seconds must be a floating-point number"

        # Value range validation
        assert self.voltage_swing > 0.0, "voltage_swing must be positive"
        assert 0.0 <= self.pre_emphasis <= 1.0, "pre_emphasis must be between 0.0 and 1.0"
        assert 0.0 <= self.de_emphasis <= 1.0, "de_emphasis must be between 0.0 and 1.0"
        assert self.eq_boost >= 0.0, "eq_boost must be non-negative"
        assert self.training_pattern in [1, 2, 3, 4, 11], "training_pattern must be a valid PCIe training pattern"
        assert self.max_iterations > 0, "max_iterations must be positive"
        assert self.convergence_threshold > 0.0, "convergence_threshold must be positive"
        assert self.timeout_seconds > 0.0, "timeout_seconds must be positive"

    @classmethod
    def default_config(cls) -> "NRZTrainingConfig":
        """
        Create a default NRZ training configuration.

        Returns:
            NRZTrainingConfig: Default configuration instance.
        """
        return cls(
            voltage_swing=0.8,
            pre_emphasis=0.2,
            de_emphasis=0.1,
            eq_boost=1.0,
            training_pattern=2,
            max_iterations=50,
            convergence_threshold=0.01,
            timeout_seconds=60.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the configuration.
        """
        return {
            "voltage_swing": self.voltage_swing,
            "pre_emphasis": self.pre_emphasis,
            "de_emphasis": self.de_emphasis,
            "eq_boost": self.eq_boost,
            "training_pattern": self.training_pattern,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NRZTrainingConfig":
        """
        Create an NRZTrainingConfig instance from a dictionary.

        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration values.

        Returns:
            NRZTrainingConfig: A new NRZTrainingConfig instance.

        Raises:
            AssertionError: If any required key is missing or has an invalid type.
        """
        # Ensure all required keys are present
        required_keys = [
            "voltage_swing",
            "pre_emphasis",
            "de_emphasis",
            "eq_boost",
            "training_pattern",
            "max_iterations",
            "convergence_threshold",
            "timeout_seconds",
        ]
        for key in required_keys:
            assert key in config_dict, f"{key} key is missing in config_dict"

        # Convert numeric values to correct types
        voltage_swing = float(config_dict["voltage_swing"])
        pre_emphasis = float(config_dict["pre_emphasis"])
        de_emphasis = float(config_dict["de_emphasis"])
        eq_boost = float(config_dict["eq_boost"])
        training_pattern = int(config_dict["training_pattern"])
        max_iterations = int(config_dict["max_iterations"])
        convergence_threshold = float(config_dict["convergence_threshold"])
        timeout_seconds = float(config_dict["timeout_seconds"])

        return cls(
            voltage_swing=voltage_swing,
            pre_emphasis=pre_emphasis,
            de_emphasis=de_emphasis,
            eq_boost=eq_boost,
            training_pattern=training_pattern,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            timeout_seconds=timeout_seconds,
        )


@dataclass
class NRZTrainingResults:
    """
    Results of NRZ training process.

    Attributes:
        success (bool): Whether training was successful.
        phase_reached (TrainingPhase): The final training phase reached.
        iterations (int): Number of iterations performed.
        final_eye_height (float): Final eye height in volts.
        final_eye_width (float): Final eye width in unit intervals.
        ber_estimate (float): Estimated bit error rate.
        training_time (float): Total training time in seconds.
        voltage_adjustments (List[float]): History of voltage adjustments.
        emphasis_adjustments (List[float]): History of emphasis adjustments.
    """

    success: bool = False
    phase_reached: TrainingPhase = TrainingPhase.DETECT
    iterations: int = 0
    final_eye_height: float = 0.0
    final_eye_width: float = 0.0
    ber_estimate: float = 1.0
    training_time: float = 0.0
    voltage_adjustments: List[float] = field(default_factory=list)
    emphasis_adjustments: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Validate the results parameters.

        Raises:
            AssertionError: If any parameter has an invalid type or value.
        """
        # Type validation
        assert isinstance(self.success, bool), "success must be a boolean"
        assert isinstance(self.phase_reached, TrainingPhase), "phase_reached must be a TrainingPhase enum value"
        assert isinstance(self.iterations, int), "iterations must be an integer"
        assert isinstance(self.final_eye_height, float), "final_eye_height must be a floating-point number"
        assert isinstance(self.final_eye_width, float), "final_eye_width must be a floating-point number"
        assert isinstance(self.ber_estimate, float), "ber_estimate must be a floating-point number"
        assert isinstance(self.training_time, float), "training_time must be a floating-point number"
        assert isinstance(self.voltage_adjustments, list), "voltage_adjustments must be a list"
        assert isinstance(self.emphasis_adjustments, list), "emphasis_adjustments must be a list"

        # Value validation
        assert self.iterations >= 0, "iterations must be non-negative"
        assert self.final_eye_height >= 0.0, "final_eye_height must be non-negative"
        assert self.final_eye_width >= 0.0, "final_eye_width must be non-negative"
        assert 0.0 <= self.ber_estimate <= 1.0, "ber_estimate must be between 0.0 and 1.0"
        assert self.training_time >= 0.0, "training_time must be non-negative"

        # Validate list elements
        for i, v in enumerate(self.voltage_adjustments):
            assert isinstance(v, float), f"voltage_adjustments[{i}] must be a floating-point number"

        for i, e in enumerate(self.emphasis_adjustments):
            assert isinstance(e, float), f"emphasis_adjustments[{i}] must be a floating-point number"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the results to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the results.
        """
        return {
            "success": self.success,
            "phase_reached": self.phase_reached.name,
            "iterations": self.iterations,
            "final_eye_height": self.final_eye_height,
            "final_eye_width": self.final_eye_width,
            "ber_estimate": self.ber_estimate,
            "training_time": self.training_time,
            "voltage_adjustments": self.voltage_adjustments,
            "emphasis_adjustments": self.emphasis_adjustments,
        }

    @classmethod
    def from_dict(cls, results_dict: Dict[str, Any]) -> "NRZTrainingResults":
        """
        Create an NRZTrainingResults instance from a dictionary.

        Args:
            results_dict (Dict[str, Any]): Dictionary containing results values.

        Returns:
            NRZTrainingResults: A new NRZTrainingResults instance.

        Raises:
            AssertionError: If any required key is missing or has an invalid type.
        """
        # Ensure all required keys are present
        required_keys = [
            "success",
            "phase_reached",
            "iterations",
            "final_eye_height",
            "final_eye_width",
            "ber_estimate",
            "training_time",
            "voltage_adjustments",
            "emphasis_adjustments",
        ]
        for key in required_keys:
            assert key in results_dict, f"{key} key is missing in results_dict"

        # Convert phase string to enum
        phase_str = results_dict["phase_reached"]
        assert phase_str in [p.name for p in TrainingPhase], f"Invalid phase: {phase_str}"
        phase = TrainingPhase[phase_str]

        # Ensure numeric lists contain floats
        voltage_adjustments = [float(v) for v in results_dict["voltage_adjustments"]]
        emphasis_adjustments = [float(e) for e in results_dict["emphasis_adjustments"]]

        return cls(
            success=bool(results_dict["success"]),
            phase_reached=phase,
            iterations=int(results_dict["iterations"]),
            final_eye_height=float(results_dict["final_eye_height"]),
            final_eye_width=float(results_dict["final_eye_width"]),
            ber_estimate=float(results_dict["ber_estimate"]),
            training_time=float(results_dict["training_time"]),
            voltage_adjustments=voltage_adjustments,
            emphasis_adjustments=emphasis_adjustments,
        )


class NRZTrainer:
    """
    Trainer class for PCIe NRZ signal mode optimization.

    This class handles the training sequence for optimizing NRZ signal
    parameters to achieve optimal signal integrity and minimal bit error rate.
    """

    def __init__(self, config: Optional[NRZTrainingConfig] = None) -> None:
        """
        Initialize a new NRZTrainer with the specified configuration.

        Args:
            config (Optional[NRZTrainingConfig]): Training configuration.
                If None, the default configuration is used.

        Raises:
            AssertionError: If config is provided but is not an NRZTrainingConfig instance.
        """
        if config is None:
            self.config = NRZTrainingConfig.default_config()
        else:
            assert isinstance(config, NRZTrainingConfig), "config must be an NRZTrainingConfig instance"
            self.config = config

        self._current_phase = TrainingPhase.DETECT
        self._iterations = 0
        self._start_time = 0.0
        self._voltage_history: List[float] = []
        self._emphasis_history: List[float] = []

    def train(self) -> NRZTrainingResults:
        """
        Execute the NRZ training sequence.

        Returns:
            NRZTrainingResults: Results of the training process.

        Raises:
            RuntimeError: If an error occurs during training.
        """
        self._start_time = time.time()
        self._current_phase = TrainingPhase.DETECT
        self._iterations = 0
        self._voltage_history = []
        self._emphasis_history = []

        try:
            # Execute training phases in sequence
            success = (
                self._detect_phase()
                and self._polling_phase()
                and self._config_phase()
                and self._recovery_phase()
                and self._loopback_phase()
            )

            if success:
                self._current_phase = TrainingPhase.COMPLETE

            # Gather final measurements
            eye_height, eye_width = self._measure_eye_diagram()
            ber = self._estimate_ber(eye_height, eye_width)

            # Create and return results
            training_time = time.time() - self._start_time
            return NRZTrainingResults(
                success=success,
                phase_reached=self._current_phase,
                iterations=self._iterations,
                final_eye_height=eye_height,
                final_eye_width=eye_width,
                ber_estimate=ber,
                training_time=training_time,
                voltage_adjustments=self._voltage_history.copy(),
                emphasis_adjustments=self._emphasis_history.copy(),
            )

        except Exception as e:
            # Create failure results
            training_time = time.time() - self._start_time
            return NRZTrainingResults(
                success=False,
                phase_reached=self._current_phase,
                iterations=self._iterations,
                training_time=training_time,
                voltage_adjustments=self._voltage_history.copy(),
                emphasis_adjustments=self._emphasis_history.copy(),
            )

    def _detect_phase(self) -> bool:
        """
        Execute the Detect phase of training.

        Returns:
            bool: True if the phase completed successfully, False otherwise.
        """
        self._current_phase = TrainingPhase.DETECT

        # Simulate training logic with type-checked values
        voltage = self.config.voltage_swing
        assert isinstance(voltage, float), "voltage must be a floating-point number"

        # Record initial values
        self._voltage_history.append(voltage)
        self._emphasis_history.append(self.config.pre_emphasis)

        # Successful detection simulation
        return True

    def _polling_phase(self) -> bool:
        """
        Execute the Polling phase of training.

        Returns:
            bool: True if the phase completed successfully, False otherwise.
        """
        self._current_phase = TrainingPhase.POLLING

        # Simulate training iterations with convergence check
        convergence = False
        for _ in range(self.config.max_iterations):
            if self._check_timeout():
                return False

            self._iterations += 1

            # Adjust voltage and emphasis
            new_voltage = self._adjust_voltage()
            new_emphasis = self._adjust_emphasis()

            # Record adjustments
            self._voltage_history.append(new_voltage)
            self._emphasis_history.append(new_emphasis)

            # Check for convergence
            if self._check_convergence():
                convergence = True
                break

        return convergence

    def _config_phase(self) -> bool:
        """
        Execute the Config phase of training.

        Returns:
            bool: True if the phase completed successfully, False otherwise.
        """
        self._current_phase = TrainingPhase.CONFIG

        # Apply config phase adjustments
        for _ in range(5):  # Limited iterations for config phase
            if self._check_timeout():
                return False

            self._iterations += 1

            # Adjust voltage and emphasis
            new_voltage = self._adjust_voltage()
            new_emphasis = self._adjust_emphasis()

            # Record adjustments
            self._voltage_history.append(new_voltage)
            self._emphasis_history.append(new_emphasis)

        return True

    def _recovery_phase(self) -> bool:
        """
        Execute the Recovery phase of training.

        Returns:
            bool: True if the phase completed successfully, False otherwise.
        """
        self._current_phase = TrainingPhase.RECOVERY

        # Apply recovery phase logic
        for _ in range(3):  # Limited iterations for recovery phase
            if self._check_timeout():
                return False

            self._iterations += 1

            # Fine-tune voltage and emphasis
            new_voltage = self._fine_tune_voltage()
            new_emphasis = self._fine_tune_emphasis()

            # Record adjustments
            self._voltage_history.append(new_voltage)
            self._emphasis_history.append(new_emphasis)

        return True

    def _loopback_phase(self) -> bool:
        """
        Execute the Loopback phase of training.

        Returns:
            bool: True if the phase completed successfully, False otherwise.
        """
        self._current_phase = TrainingPhase.LOOPBACK

        # Apply loopback test
        if self._check_timeout():
            return False

        # Simulate loopback test with BER measurement
        eye_height, eye_width = self._measure_eye_diagram()
        ber = self._estimate_ber(eye_height, eye_width)

        # Success criteria based on BER threshold
        return ber < 1e-12

    def _adjust_voltage(self) -> float:
        """
        Adjust voltage based on current training state.

        Returns:
            float: New voltage value.
        """
        # Get the last voltage value
        last_voltage = self._voltage_history[-1]
        assert isinstance(last_voltage, float), "last_voltage must be a floating-point number"

        # Simulate voltage adjustment algorithm
        # In a real implementation, this would use feedback from measurements
        adjustment = math.sin(self._iterations / 5.0) * 0.05

        # Ensure adjustment is a float
        assert isinstance(adjustment, float), "adjustment must be a floating-point number"

        # Apply adjustment with bounds checking
        new_voltage = max(0.1, min(1.2, last_voltage + adjustment))
        return new_voltage

    def _adjust_emphasis(self) -> float:
        """
        Adjust pre-emphasis based on current training state.

        Returns:
            float: New pre-emphasis value.
        """
        # Get the last emphasis value
        last_emphasis = self._emphasis_history[-1]
        assert isinstance(last_emphasis, float), "last_emphasis must be a floating-point number"

        # Simulate emphasis adjustment algorithm
        adjustment = math.cos(self._iterations / 4.0) * 0.02
        assert isinstance(adjustment, float), "adjustment must be a floating-point number"

        # Apply adjustment with bounds checking
        new_emphasis = max(0.0, min(0.5, last_emphasis + adjustment))
        return new_emphasis

    def _fine_tune_voltage(self) -> float:
        """
        Fine-tune voltage with smaller adjustments.

        Returns:
            float: New voltage value.
        """
        # Get the last voltage value
        last_voltage = self._voltage_history[-1]
        assert isinstance(last_voltage, float), "last_voltage must be a floating-point number"

        # Simulate fine-tuning algorithm (smaller adjustments)
        adjustment = math.sin(self._iterations / 10.0) * 0.01
        assert isinstance(adjustment, float), "adjustment must be a floating-point number"

        # Apply adjustment with bounds checking
        new_voltage = max(0.1, min(1.2, last_voltage + adjustment))
        return new_voltage

    def _fine_tune_emphasis(self) -> float:
        """
        Fine-tune pre-emphasis with smaller adjustments.

        Returns:
            float: New pre-emphasis value.
        """
        # Get the last emphasis value
        last_emphasis = self._emphasis_history[-1]
        assert isinstance(last_emphasis, float), "last_emphasis must be a floating-point number"

        # Simulate fine-tuning algorithm (smaller adjustments)
        adjustment = math.cos(self._iterations / 8.0) * 0.01
        assert isinstance(adjustment, float), "adjustment must be a floating-point number"

        # Apply adjustment with bounds checking
        new_emphasis = max(0.0, min(0.5, last_emphasis + adjustment))
        return new_emphasis

    def _check_convergence(self) -> bool:
        """
        Check if training has converged based on recent adjustments.

        Returns:
            bool: True if training has converged, False otherwise.
        """
        # Need at least 3 measurements to check convergence
        if len(self._voltage_history) < 3 or len(self._emphasis_history) < 3:
            return False

        # Calculate recent voltage changes
        v_changes = [abs(self._voltage_history[-i] - self._voltage_history[-i - 1]) for i in range(1, 3)]

        # Calculate recent emphasis changes
        e_changes = [abs(self._emphasis_history[-i] - self._emphasis_history[-i - 1]) for i in range(1, 3)]

        # Check if all changes are below threshold
        threshold = self.config.convergence_threshold
        assert isinstance(threshold, float), "convergence_threshold must be a floating-point number"

        return all(v < threshold for v in v_changes) and all(e < threshold for e in e_changes)

    def _check_timeout(self) -> bool:
        """
        Check if the training operation has timed out.

        Returns:
            bool: True if timeout occurred, False otherwise.
        """
        elapsed_time = time.time() - self._start_time
        assert isinstance(elapsed_time, float), "elapsed_time must be a floating-point number"

        timeout = self.config.timeout_seconds
        assert isinstance(timeout, float), "timeout_seconds must be a floating-point number"

        return elapsed_time > timeout

    def _measure_eye_diagram(self) -> Tuple[float, float]:
        """
        Measure eye diagram parameters.

        Returns:
            Tuple[float, float]: Eye height (volts) and eye width (UI).
        """
        # Simulate eye measurement based on current parameters
        # In real implementation, this would use actual hardware measurements

        # Get latest parameters
        latest_voltage = self._voltage_history[-1] if self._voltage_history else self.config.voltage_swing
        latest_emphasis = self._emphasis_history[-1] if self._emphasis_history else self.config.pre_emphasis

        assert isinstance(latest_voltage, float), "latest_voltage must be a floating-point number"
        assert isinstance(latest_emphasis, float), "latest_emphasis must be a floating-point number"

        # Simulated eye height calculation
        eye_height = latest_voltage * (0.7 + latest_emphasis * 0.3) * (0.9 + 0.1 * math.sin(self._iterations / 10.0))

        # Simulated eye width calculation (in unit intervals)
        eye_width = 0.6 + 0.3 * latest_emphasis + 0.1 * math.cos(self._iterations / 7.0)

        # Ensure values are floats
        assert isinstance(eye_height, float), "eye_height must be a floating-point number"
        assert isinstance(eye_width, float), "eye_width must be a floating-point number"

        # Apply bounds
        eye_height = max(0.0, min(1.0, eye_height))
        eye_width = max(0.0, min(1.0, eye_width))

        return eye_height, eye_width

    def _estimate_ber(self, eye_height: float, eye_width: float) -> float:
        """
        Estimate bit error rate based on eye diagram parameters.

        Args:
            eye_height (float): Eye height in volts.
            eye_width (float): Eye width in unit intervals.

        Returns:
            float: Estimated bit error rate.

        Raises:
            AssertionError: If eye_height or eye_width is not a float.
        """
        assert isinstance(eye_height, float), "eye_height must be a floating-point number"
        assert isinstance(eye_width, float), "eye_width must be a floating-point number"

        # Simple BER estimation model
        # Real implementations would use more sophisticated models or measurements
        if eye_height <= 0 or eye_width <= 0:
            return 1.0

        # Q-factor based on eye parameters (simplified model)
        q_factor = 20.0 * eye_height * eye_width

        # BER estimation using Q-factor approximation
        if q_factor > 50.0:  # Prevent underflow for very small BER
            ber = 1e-15
        else:
            ber = 0.5 * math.erfc(q_factor / math.sqrt(8.0))

        # Ensure BER is a float
        assert isinstance(ber, float), "ber must be a floating-point number"

        # Apply bounds
        ber = max(1e-15, min(1.0, ber))

        return ber
