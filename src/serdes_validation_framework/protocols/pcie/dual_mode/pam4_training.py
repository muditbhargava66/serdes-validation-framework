"""
PCIe PAM4 Training Module
This module provides functionality for training and optimizing PAM4 signal mode
in PCIe dual-mode operation with robust type checking and validation.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class PAM4TrainingPhase(Enum):
    """Enumeration of PAM4 training phases."""

    DETECT = auto()
    COARSE_TUNING = auto()
    FINE_TUNING = auto()
    LEVEL_ADJUSTMENT = auto()
    SIGNAL_INTEGRITY = auto()
    COMPLETE = auto()


@dataclass
class PAM4TrainingConfig:
    """
    Configuration parameters for PAM4 training.

    Attributes:
        voltage_levels (List[float]): Target voltage levels for PAM4 symbols (3 thresholds).
        pre_cursor_taps (List[float]): Pre-cursor tap weights for FFE.
        post_cursor_taps (List[float]): Post-cursor tap weights for FFE.
        level_separation_mse_threshold (float): Threshold for level separation MSE convergence.
        eye_height_threshold (float): Minimum acceptable eye height in volts.
        training_pattern (int): Training pattern number to use.
        max_iterations (int): Maximum number of training iterations.
        timeout_seconds (float): Timeout in seconds for training operations.
    """

    voltage_levels: List[float]
    pre_cursor_taps: List[float]
    post_cursor_taps: List[float]
    level_separation_mse_threshold: float
    eye_height_threshold: float
    training_pattern: int
    max_iterations: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        """
        Validate the configuration parameters.

        Raises:
            AssertionError: If any parameter has an invalid type or value.
        """
        # Type validation
        assert isinstance(self.voltage_levels, list), "voltage_levels must be a list"
        assert isinstance(self.pre_cursor_taps, list), "pre_cursor_taps must be a list"
        assert isinstance(self.post_cursor_taps, list), "post_cursor_taps must be a list"
        assert isinstance(
            self.level_separation_mse_threshold, float
        ), "level_separation_mse_threshold must be a floating-point number"
        assert isinstance(self.eye_height_threshold, float), "eye_height_threshold must be a floating-point number"
        assert isinstance(self.training_pattern, int), "training_pattern must be an integer"
        assert isinstance(self.max_iterations, int), "max_iterations must be an integer"
        assert isinstance(self.timeout_seconds, float), "timeout_seconds must be a floating-point number"

        # Validate list elements
        assert len(self.voltage_levels) == 3, "voltage_levels must contain exactly 3 elements (PAM4 thresholds)"
        for i, level in enumerate(self.voltage_levels):
            assert isinstance(level, float), f"voltage_levels[{i}] must be a floating-point number"

        for i, tap in enumerate(self.pre_cursor_taps):
            assert isinstance(tap, float), f"pre_cursor_taps[{i}] must be a floating-point number"

        for i, tap in enumerate(self.post_cursor_taps):
            assert isinstance(tap, float), f"post_cursor_taps[{i}] must be a floating-point number"

        # Value range validation
        for level in self.voltage_levels:
            assert level > 0.0, "all voltage levels must be positive"

        assert self.level_separation_mse_threshold > 0.0, "level_separation_mse_threshold must be positive"
        assert self.eye_height_threshold > 0.0, "eye_height_threshold must be positive"
        assert self.training_pattern in [1, 2, 3, 4, 11], "training_pattern must be a valid PCIe training pattern"
        assert self.max_iterations > 0, "max_iterations must be positive"
        assert self.timeout_seconds > 0.0, "timeout_seconds must be positive"

        # Check voltage level ordering (should be increasing)
        for i in range(1, len(self.voltage_levels)):
            assert self.voltage_levels[i] > self.voltage_levels[i - 1], "voltage levels must be in increasing order"

    @classmethod
    def default_config(cls) -> "PAM4TrainingConfig":
        """
        Create a default PAM4 training configuration.

        Returns:
            PAM4TrainingConfig: Default configuration instance.
        """
        return cls(
            voltage_levels=[0.2, 0.4, 0.6],
            pre_cursor_taps=[-0.1, 0.0],
            post_cursor_taps=[-0.2, -0.1, -0.05],
            level_separation_mse_threshold=0.005,
            eye_height_threshold=0.05,
            training_pattern=4,
            max_iterations=100,
            timeout_seconds=120.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the configuration.
        """
        return {
            "voltage_levels": self.voltage_levels,
            "pre_cursor_taps": self.pre_cursor_taps,
            "post_cursor_taps": self.post_cursor_taps,
            "level_separation_mse_threshold": self.level_separation_mse_threshold,
            "eye_height_threshold": self.eye_height_threshold,
            "training_pattern": self.training_pattern,
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PAM4TrainingConfig":
        """
        Create a PAM4TrainingConfig instance from a dictionary.

        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration values.

        Returns:
            PAM4TrainingConfig: A new PAM4TrainingConfig instance.

        Raises:
            AssertionError: If any required key is missing or has an invalid type.
        """
        # Ensure all required keys are present
        required_keys = [
            "voltage_levels",
            "pre_cursor_taps",
            "post_cursor_taps",
            "level_separation_mse_threshold",
            "eye_height_threshold",
            "training_pattern",
            "max_iterations",
            "timeout_seconds",
        ]
        for key in required_keys:
            assert key in config_dict, f"{key} key is missing in config_dict"

        # Convert numeric values to correct types and ensure lists contain floats
        voltage_levels = [float(v) for v in config_dict["voltage_levels"]]
        pre_cursor_taps = [float(t) for t in config_dict["pre_cursor_taps"]]
        post_cursor_taps = [float(t) for t in config_dict["post_cursor_taps"]]

        return cls(
            voltage_levels=voltage_levels,
            pre_cursor_taps=pre_cursor_taps,
            post_cursor_taps=post_cursor_taps,
            level_separation_mse_threshold=float(config_dict["level_separation_mse_threshold"]),
            eye_height_threshold=float(config_dict["eye_height_threshold"]),
            training_pattern=int(config_dict["training_pattern"]),
            max_iterations=int(config_dict["max_iterations"]),
            timeout_seconds=float(config_dict["timeout_seconds"]),
        )


@dataclass
class PAM4TrainingResults:
    """
    Results of PAM4 training process.

    Attributes:
        success (bool): Whether training was successful.
        phase_reached (PAM4TrainingPhase): The final training phase reached.
        iterations (int): Number of iterations performed.
        final_voltage_levels (List[float]): Final voltage levels for PAM4 symbols.
        final_eye_heights (List[float]): Eye heights for PAM4 eyes (3 eyes).
        final_snr (float): Signal-to-noise ratio in dB.
        level_mse (float): Mean squared error of level separation.
        training_time (float): Total training time in seconds.
        voltage_adjustments_history (List[List[float]]): History of voltage level adjustments.
        tap_adjustments_history (List[List[float]]): History of FFE tap adjustments.
    """

    success: bool = False
    phase_reached: PAM4TrainingPhase = PAM4TrainingPhase.DETECT
    iterations: int = 0
    final_voltage_levels: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    final_eye_heights: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    final_snr: float = 0.0
    level_mse: float = 1.0
    training_time: float = 0.0
    voltage_adjustments_history: List[List[float]] = field(default_factory=list)
    tap_adjustments_history: List[List[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Validate the results parameters.

        Raises:
            AssertionError: If any parameter has an invalid type or value.
        """
        # Type validation
        assert isinstance(self.success, bool), "success must be a boolean"
        assert isinstance(self.phase_reached, PAM4TrainingPhase), "phase_reached must be a PAM4TrainingPhase enum value"
        assert isinstance(self.iterations, int), "iterations must be an integer"
        assert isinstance(self.final_voltage_levels, list), "final_voltage_levels must be a list"
        assert isinstance(self.final_eye_heights, list), "final_eye_heights must be a list"
        assert isinstance(self.final_snr, float), "final_snr must be a floating-point number"
        assert isinstance(self.level_mse, float), "level_mse must be a floating-point number"
        assert isinstance(self.training_time, float), "training_time must be a floating-point number"
        assert isinstance(self.voltage_adjustments_history, list), "voltage_adjustments_history must be a list"
        assert isinstance(self.tap_adjustments_history, list), "tap_adjustments_history must be a list"

        # Validate list elements and lengths
        assert len(self.final_voltage_levels) == 3, "final_voltage_levels must contain exactly 3 elements"
        assert len(self.final_eye_heights) == 3, "final_eye_heights must contain exactly 3 elements"

        for i, level in enumerate(self.final_voltage_levels):
            assert isinstance(level, float), f"final_voltage_levels[{i}] must be a floating-point number"

        for i, height in enumerate(self.final_eye_heights):
            assert isinstance(height, float), f"final_eye_heights[{i}] must be a floating-point number"

        # Validate voltage_adjustments_history
        for i, adjustment in enumerate(self.voltage_adjustments_history):
            assert isinstance(adjustment, list), f"voltage_adjustments_history[{i}] must be a list"
            assert len(adjustment) == 3, f"voltage_adjustments_history[{i}] must contain exactly 3 elements"
            for j, val in enumerate(adjustment):
                assert isinstance(val, float), f"voltage_adjustments_history[{i}][{j}] must be a floating-point number"

        # Validate tap_adjustments_history (flexible length)
        for i, tap_set in enumerate(self.tap_adjustments_history):
            assert isinstance(tap_set, list), f"tap_adjustments_history[{i}] must be a list"
            for j, tap in enumerate(tap_set):
                assert isinstance(tap, float), f"tap_adjustments_history[{i}][{j}] must be a floating-point number"

        # Value validation
        assert self.iterations >= 0, "iterations must be non-negative"
        assert self.level_mse >= 0.0, "level_mse must be non-negative"
        assert self.training_time >= 0.0, "training_time must be non-negative"

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
            "final_voltage_levels": self.final_voltage_levels,
            "final_eye_heights": self.final_eye_heights,
            "final_snr": self.final_snr,
            "level_mse": self.level_mse,
            "training_time": self.training_time,
            "voltage_adjustments_history": self.voltage_adjustments_history,
            "tap_adjustments_history": self.tap_adjustments_history,
        }

    @classmethod
    def from_dict(cls, results_dict: Dict[str, Any]) -> "PAM4TrainingResults":
        """
        Create a PAM4TrainingResults instance from a dictionary.

        Args:
            results_dict (Dict[str, Any]): Dictionary containing results values.

        Returns:
            PAM4TrainingResults: A new PAM4TrainingResults instance.

        Raises:
            AssertionError: If any required key is missing or has an invalid type.
        """
        # Ensure all required keys are present
        required_keys = [
            "success",
            "phase_reached",
            "iterations",
            "final_voltage_levels",
            "final_eye_heights",
            "final_snr",
            "level_mse",
            "training_time",
            "voltage_adjustments_history",
            "tap_adjustments_history",
        ]
        for key in required_keys:
            assert key in results_dict, f"{key} key is missing in results_dict"

        # Convert phase string to enum
        phase_str = results_dict["phase_reached"]
        assert phase_str in [p.name for p in PAM4TrainingPhase], f"Invalid phase: {phase_str}"
        phase = PAM4TrainingPhase[phase_str]

        # Convert numeric values to correct types
        final_voltage_levels = [float(v) for v in results_dict["final_voltage_levels"]]
        final_eye_heights = [float(h) for h in results_dict["final_eye_heights"]]

        # Convert history lists
        voltage_adjustments_history = []
        for adj_set in results_dict["voltage_adjustments_history"]:
            voltage_adjustments_history.append([float(v) for v in adj_set])

        tap_adjustments_history = []
        for tap_set in results_dict["tap_adjustments_history"]:
            tap_adjustments_history.append([float(t) for t in tap_set])

        return cls(
            success=bool(results_dict["success"]),
            phase_reached=phase,
            iterations=int(results_dict["iterations"]),
            final_voltage_levels=final_voltage_levels,
            final_eye_heights=final_eye_heights,
            final_snr=float(results_dict["final_snr"]),
            level_mse=float(results_dict["level_mse"]),
            training_time=float(results_dict["training_time"]),
            voltage_adjustments_history=voltage_adjustments_history,
            tap_adjustments_history=tap_adjustments_history,
        )


class PAM4Trainer:
    """
    Trainer class for PCIe PAM4 signal mode optimization.

    This class handles the training sequence for optimizing PAM4 signal
    parameters to achieve optimal signal integrity and minimal bit error rate.
    """

    def __init__(self, config: Optional[PAM4TrainingConfig] = None) -> None:
        """
        Initialize a new PAM4Trainer with the specified configuration.

        Args:
            config (Optional[PAM4TrainingConfig]): Training configuration.
                If None, the default configuration is used.

        Raises:
            AssertionError: If config is provided but is not a PAM4TrainingConfig instance.
        """
        if config is None:
            self.config = PAM4TrainingConfig.default_config()
        else:
            assert isinstance(config, PAM4TrainingConfig), "config must be a PAM4TrainingConfig instance"
            self.config = config

        self._current_phase = PAM4TrainingPhase.DETECT
        self._iterations = 0
        self._start_time = 0.0
        self._voltage_levels_history: List[List[float]] = []
        self._tap_weights_history: List[List[float]] = []

    def train(self) -> PAM4TrainingResults:
        """
        Execute the PAM4 training sequence.

        Returns:
            PAM4TrainingResults: Results of the training process.

        Raises:
            RuntimeError: If an error occurs during training.
        """
        self._start_time = time.time()
        self._current_phase = PAM4TrainingPhase.DETECT
        self._iterations = 0
        self._voltage_levels_history = []
        self._tap_weights_history = []

        # Initialize voltage levels and tap weights history with initial values
        self._voltage_levels_history.append(self.config.voltage_levels.copy())

        # Combine pre and post cursor taps for history tracking
        initial_taps = self.config.pre_cursor_taps + [1.0] + self.config.post_cursor_taps
        self._tap_weights_history.append(initial_taps)

        try:
            # Execute training phases in sequence
            success = (
                self._detect_phase()
                and self._coarse_tuning_phase()
                and self._fine_tuning_phase()
                and self._level_adjustment_phase()
                and self._signal_integrity_phase()
            )

            if success:
                self._current_phase = PAM4TrainingPhase.COMPLETE

            # Gather final measurements
            final_voltage_levels = self._voltage_levels_history[-1]
            eye_heights = self._measure_eye_heights()
            snr = self._calculate_snr()
            level_mse = self._calculate_level_separation_mse()

            # Create and return results
            training_time = time.time() - self._start_time
            return PAM4TrainingResults(
                success=success,
                phase_reached=self._current_phase,
                iterations=self._iterations,
                final_voltage_levels=final_voltage_levels,
                final_eye_heights=eye_heights,
                final_snr=snr,
                level_mse=level_mse,
                training_time=training_time,
                voltage_adjustments_history=self._voltage_levels_history.copy(),
                tap_adjustments_history=self._tap_weights_history.copy(),
            )

        except Exception as e:
            # Create failure results
            training_time = time.time() - self._start_time
            return PAM4TrainingResults(
                success=False,
                phase_reached=self._current_phase,
                iterations=self._iterations,
                training_time=training_time,
                voltage_adjustments_history=self._voltage_levels_history.copy(),
                tap_adjustments_history=self._tap_weights_history.copy(),
            )

    def _detect_phase(self) -> bool:
        """
        Execute the Detect phase of PAM4 training.

        Returns:
            bool: True if the phase completed successfully, False otherwise.
        """
        self._current_phase = PAM4TrainingPhase.DETECT

        # Simulate detection logic
        # In a real implementation, this would detect the presence of PAM4 signal

        # Check for timeout
        if self._check_timeout():
            return False

        # Detection is simulated as always successful for this example
        return True

    def _coarse_tuning_phase(self) -> bool:
        """
        Execute the Coarse Tuning phase of PAM4 training.

        Returns:
            bool: True if the phase completed successfully, False otherwise.
        """
        self._current_phase = PAM4TrainingPhase.COARSE_TUNING

        # Perform coarse tuning iterations
        for _ in range(min(20, self.config.max_iterations // 3)):
            if self._check_timeout():
                return False

            self._iterations += 1

            # Adjust voltage levels with larger step sizes
            new_levels = self._adjust_voltage_levels(step_scale=1.0)
            self._voltage_levels_history.append(new_levels)

            # Adjust FFE taps with larger step sizes
            new_taps = self._adjust_ffe_taps(step_scale=1.0)
            self._tap_weights_history.append(new_taps)

            # Check if coarse tuning has reached sufficient quality
            if self._check_coarse_convergence():
                break

        return True

    def _fine_tuning_phase(self) -> bool:
        """
        Execute the Fine Tuning phase of PAM4 training.

        Returns:
            bool: True if the phase completed successfully, False otherwise.
        """
        self._current_phase = PAM4TrainingPhase.FINE_TUNING

        # Perform fine tuning iterations
        for _ in range(min(30, self.config.max_iterations // 2)):
            if self._check_timeout():
                return False

            self._iterations += 1

            # Adjust voltage levels with smaller step sizes
            new_levels = self._adjust_voltage_levels(step_scale=0.3)
            self._voltage_levels_history.append(new_levels)

            # Adjust FFE taps with smaller step sizes
            new_taps = self._adjust_ffe_taps(step_scale=0.3)
            self._tap_weights_history.append(new_taps)

            # Check for convergence
            if self._check_fine_convergence():
                break

        return True

    def _level_adjustment_phase(self) -> bool:
        """
        Execute the Level Adjustment phase of PAM4 training.

        Returns:
            bool: True if the phase completed successfully, False otherwise.
        """
        self._current_phase = PAM4TrainingPhase.LEVEL_ADJUSTMENT

        # Final adjustment of voltage levels for optimal spacing
        for _ in range(min(10, self.config.max_iterations // 5)):
            if self._check_timeout():
                return False

            self._iterations += 1

            # Only adjust voltage levels with very small steps
            new_levels = self._adjust_voltage_levels(step_scale=0.1)
            self._voltage_levels_history.append(new_levels)

            # Check level separation MSE against threshold
            if self._calculate_level_separation_mse() < self.config.level_separation_mse_threshold:
                break

        return True

    def _signal_integrity_phase(self) -> bool:
        """
        Execute the Signal Integrity phase of PAM4 training.

        Returns:
            bool: True if the phase completed successfully, False otherwise.
        """
        self._current_phase = PAM4TrainingPhase.SIGNAL_INTEGRITY

        # Check eye heights against threshold
        eye_heights = self._measure_eye_heights()
        assert isinstance(eye_heights, list), "eye_heights must be a list"
        for i, height in enumerate(eye_heights):
            assert isinstance(height, float), f"eye_heights[{i}] must be a floating-point number"

        # All eyes must meet minimum height requirement
        threshold = self.config.eye_height_threshold
        assert isinstance(threshold, float), "eye_height_threshold must be a floating-point number"

        return all(height >= threshold for height in eye_heights)

    def _adjust_voltage_levels(self, step_scale: float = 1.0) -> List[float]:
        """
        Adjust PAM4 voltage levels based on current training state.

        Args:
            step_scale (float): Scale factor for adjustment step size.

        Returns:
            List[float]: New voltage levels.

        Raises:
            AssertionError: If step_scale is not a float.
        """
        assert isinstance(step_scale, float), "step_scale must be a floating-point number"

        # Get the most recent voltage levels
        last_levels = self._voltage_levels_history[-1]

        # Generate adjustments for each level
        # In a real implementation, this would use feedback from measurements
        adjustments = [
            math.sin(self._iterations / 10.0) * 0.02 * step_scale,
            math.sin(self._iterations / 7.0) * 0.015 * step_scale,
            math.sin(self._iterations / 5.0) * 0.01 * step_scale,
        ]

        # Apply adjustments with bounds checking
        new_levels = []
        min_level = 0.1

        for i, (level, adj) in enumerate(zip(last_levels, adjustments, strict=False)):
            assert isinstance(level, float), f"last_levels[{i}] must be a floating-point number"
            assert isinstance(adj, float), f"adjustments[{i}] must be a floating-point number"

            # Apply adjustment with minimum value constraint
            new_level = max(min_level, level + adj)
            new_levels.append(new_level)

            # Update minimum for next level to ensure proper ordering
            min_level = new_level + 0.05

        return new_levels

    def _adjust_ffe_taps(self, step_scale: float = 1.0) -> List[float]:
        """
        Adjust FFE tap weights based on current training state.

        Args:
            step_scale (float): Scale factor for adjustment step size.

        Returns:
            List[float]: New tap weights.

        Raises:
            AssertionError: If step_scale is not a float.
        """
        assert isinstance(step_scale, float), "step_scale must be a floating-point number"

        # Get the most recent tap weights
        last_taps = self._tap_weights_history[-1]

        # Generate adjustments for each tap
        # In a real implementation, this would use feedback from measurements
        adjustments = []
        for i in range(len(last_taps)):
            if i == len(self.config.pre_cursor_taps):  # Main tap
                adj = 0.0  # Don't adjust the main tap
            else:
                # Different frequency for each tap position
                freq = 5.0 + i * 2.0
                adj = math.cos(self._iterations / freq) * 0.01 * step_scale

            adjustments.append(adj)

        # Apply adjustments with constraints
        new_taps = []
        main_tap_idx = len(self.config.pre_cursor_taps)

        for i, (tap, adj) in enumerate(zip(last_taps, adjustments, strict=False)):
            assert isinstance(tap, float), f"last_taps[{i}] must be a floating-point number"
            assert isinstance(adj, float), f"adjustments[{i}] must be a floating-point number"

            if i == main_tap_idx:
                # Main tap is always 1.0
                new_taps.append(1.0)
            else:
                # Limit tap values between -0.5 and 0.5
                new_tap = max(-0.5, min(0.5, tap + adj))
                new_taps.append(new_tap)

        return new_taps

    def _check_coarse_convergence(self) -> bool:
        """
        Check if coarse tuning has converged sufficiently.

        Returns:
            bool: True if coarse tuning has converged, False otherwise.
        """
        # Need at least 3 iterations to check convergence
        if len(self._voltage_levels_history) < 3:
            return False

        # Calculate average change in voltage levels
        level_changes = []
        for i in range(3):  # For each level
            changes = [
                abs(self._voltage_levels_history[-1][i] - self._voltage_levels_history[-2][i]),
                abs(self._voltage_levels_history[-2][i] - self._voltage_levels_history[-3][i]),
            ]
            level_changes.extend(changes)

        # Calculate average change in tap weights
        tap_changes = []
        for i in range(len(self._tap_weights_history[-1])):
            changes = [
                abs(self._tap_weights_history[-1][i] - self._tap_weights_history[-2][i]),
                abs(self._tap_weights_history[-2][i] - self._tap_weights_history[-3][i]),
            ]
            tap_changes.extend(changes)

        # Use larger threshold for coarse convergence
        coarse_threshold = self.config.level_separation_mse_threshold * 5.0
        assert isinstance(coarse_threshold, float), "coarse_threshold must be a floating-point number"

        # Check if average changes are below threshold
        avg_level_change = sum(level_changes) / len(level_changes)
        avg_tap_change = sum(tap_changes) / len(tap_changes)

        return avg_level_change < coarse_threshold and avg_tap_change < coarse_threshold

    def _check_fine_convergence(self) -> bool:
        """
        Check if fine tuning has converged sufficiently.

        Returns:
            bool: True if fine tuning has converged, False otherwise.
        """
        # Need at least 3 iterations to check convergence
        if len(self._voltage_levels_history) < 3:
            return False

        # Calculate average change in voltage levels
        level_changes = []
        for i in range(3):  # For each level
            changes = [
                abs(self._voltage_levels_history[-1][i] - self._voltage_levels_history[-2][i]),
                abs(self._voltage_levels_history[-2][i] - self._voltage_levels_history[-3][i]),
            ]
            level_changes.extend(changes)

        # Calculate average change in tap weights
        tap_changes = []
        for i in range(len(self._tap_weights_history[-1])):
            changes = [
                abs(self._tap_weights_history[-1][i] - self._tap_weights_history[-2][i]),
                abs(self._tap_weights_history[-2][i] - self._tap_weights_history[-3][i]),
            ]
            tap_changes.extend(changes)

        # Use threshold from config for fine convergence
        fine_threshold = self.config.level_separation_mse_threshold * 2.0
        assert isinstance(fine_threshold, float), "fine_threshold must be a floating-point number"

        # Check if average changes are below threshold
        avg_level_change = sum(level_changes) / len(level_changes)
        avg_tap_change = sum(tap_changes) / len(tap_changes)

        return avg_level_change < fine_threshold and avg_tap_change < fine_threshold

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

    def _measure_eye_heights(self) -> List[float]:
        """
        Measure the eye heights for each of the three PAM4 eyes.

        Returns:
            List[float]: Eye heights in volts for the three eyes.
        """
        # Get latest voltage levels
        latest_levels = self._voltage_levels_history[-1]
        assert isinstance(latest_levels, list), "latest_levels must be a list"
        assert len(latest_levels) == 3, "latest_levels must contain exactly 3 elements"

        # In a real implementation, this would use actual hardware measurements
        # Here we simulate based on voltage level separation and tap weights

        # Calculate level separations
        level_separations = [
            latest_levels[0],  # 0 to first level
            latest_levels[1] - latest_levels[0],  # first to second level
            latest_levels[2] - latest_levels[1],  # second to third level
            1.0 - latest_levels[2],  # third level to 1.0
        ]

        # Get the current tap weights
        latest_taps = self._tap_weights_history[-1]

        # Calculate tap weight effect on eye height
        # Simplistic model: more extreme tap weights reduce eye height
        tap_penalty = 0.0
        for tap in latest_taps:
            tap_penalty += abs(tap) * 0.1
        tap_penalty = min(0.5, tap_penalty)  # Cap the penalty

        # Calculate eye heights (based on level separations)
        # Add some noise with iteration-dependent pattern
        eye_heights = [
            max(0.0, level_separations[0] * 0.7 - tap_penalty + 0.02 * math.sin(self._iterations / 5.0)),
            max(0.0, level_separations[1] * 0.7 - tap_penalty + 0.015 * math.sin(self._iterations / 6.0)),
            max(0.0, level_separations[2] * 0.7 - tap_penalty + 0.01 * math.sin(self._iterations / 7.0)),
        ]

        # Validate types
        for i, height in enumerate(eye_heights):
            assert isinstance(height, float), f"eye_heights[{i}] must be a floating-point number"

        return eye_heights

    def _calculate_snr(self) -> float:
        """
        Calculate the signal-to-noise ratio (SNR) in dB.

        Returns:
            float: SNR in dB.
        """
        # Get latest voltage levels
        latest_levels = self._voltage_levels_history[-1]

        # In a real implementation, this would use actual hardware measurements
        # Here we simulate based on voltage levels and iterations

        # Calculate peak-to-peak signal level
        signal_pp = latest_levels[2]  # Assuming 0 to level[2] is the range
        assert isinstance(signal_pp, float), "signal_pp must be a floating-point number"

        # Simulate noise level (decreasing with iterations as training improves)
        noise_rms = 0.05 * math.exp(-self._iterations / 50.0)
        assert isinstance(noise_rms, float), "noise_rms must be a floating-point number"

        # Calculate SNR in dB
        # SNR = 20 * log10(signal_amplitude / noise_rms)
        # For peak-to-peak to RMS conversion, divide by 2*sqrt(2)
        if noise_rms > 0.0:
            signal_rms = signal_pp / (2.0 * math.sqrt(2.0))
            snr_db = 20.0 * math.log10(signal_rms / noise_rms)
        else:
            snr_db = 100.0  # High value for near-zero noise

        assert isinstance(snr_db, float), "snr_db must be a floating-point number"

        # Ensure non-negative value (should be impossible with this calculation)
        snr_db = max(0.0, snr_db)

        return snr_db

    def _calculate_level_separation_mse(self) -> float:
        """
        Calculate the mean squared error (MSE) of level separation.

        Returns:
            float: MSE of level separation.
        """
        # Get latest voltage levels
        latest_levels = self._voltage_levels_history[-1]
        assert isinstance(latest_levels, list), "latest_levels must be a list"
        assert len(latest_levels) == 3, "latest_levels must contain exactly 3 elements"

        # Calculate ideal uniform separation
        ideal_step = 1.0 / 3.0  # For 4 levels (0, 1/3, 2/3, 1)
        ideal_levels = [ideal_step, 2.0 * ideal_step, 3.0 * ideal_step]

        # Calculate squared errors
        squared_errors = [(latest_levels[i] - ideal_levels[i]) ** 2 for i in range(3)]

        # Calculate mean squared error
        mse = sum(squared_errors) / 3.0
        assert isinstance(mse, float), "mse must be a floating-point number"

        return mse
