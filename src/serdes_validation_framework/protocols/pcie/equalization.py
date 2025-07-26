"""
PCIe Equalization Module

This module provides comprehensive equalization algorithms for PCIe signals
with type-safe implementation supporting both NRZ and PAM4 modes.

Features:
- Adaptive equalization algorithms
- Multiple equalizer types (FFE, CTLE, DFE)
- Convergence optimization
- Performance monitoring
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EqualizationType(Enum):
    """Types of equalization algorithms"""

    LMS = auto()  # Least Mean Squares
    RLS = auto()  # Recursive Least Squares
    CMA = auto()  # Constant Modulus Algorithm
    DECISION_DIRECTED = auto()  # Decision-directed adaptation


class EqualizerStructure(Enum):
    """Equalizer structures"""

    LINEAR = auto()  # Linear equalizer
    DFE = auto()  # Decision feedback equalizer
    VOLTERRA = auto()  # Volterra series equalizer


@dataclass
class EqualizationConfig:
    """Equalization configuration with validation"""

    algorithm: EqualizationType
    structure: EqualizerStructure
    num_forward_taps: int
    num_feedback_taps: int
    step_size: float
    forgetting_factor: float

    def __post_init__(self) -> None:
        """
        Validate equalization configuration

        Raises:
            AssertionError: If configuration is invalid
        """
        # Type validation
        assert isinstance(self.algorithm, EqualizationType), f"Algorithm must be EqualizationType, got {type(self.algorithm)}"
        assert isinstance(self.structure, EqualizerStructure), f"Structure must be EqualizerStructure, got {type(self.structure)}"
        assert isinstance(self.num_forward_taps, int), f"Forward taps must be int, got {type(self.num_forward_taps)}"
        assert isinstance(self.num_feedback_taps, int), f"Feedback taps must be int, got {type(self.num_feedback_taps)}"
        assert isinstance(self.step_size, float), f"Step size must be float, got {type(self.step_size)}"
        assert isinstance(self.forgetting_factor, float), f"Forgetting factor must be float, got {type(self.forgetting_factor)}"

        # Value validation
        assert self.num_forward_taps > 0, f"Forward taps must be positive, got {self.num_forward_taps}"
        assert self.num_feedback_taps >= 0, f"Feedback taps must be non-negative, got {self.num_feedback_taps}"
        assert 0 < self.step_size < 1, f"Step size must be between 0 and 1, got {self.step_size}"
        assert 0 < self.forgetting_factor <= 1, f"Forgetting factor must be between 0 and 1, got {self.forgetting_factor}"


@dataclass
class EqualizationResult:
    """Equalization result with validation"""

    converged: bool
    final_mse: float
    iterations: int
    forward_coeffs: List[float]
    feedback_coeffs: List[float]
    mse_history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Validate equalization result

        Raises:
            AssertionError: If result is invalid
        """
        # Type validation
        assert isinstance(self.converged, bool), f"Converged must be bool, got {type(self.converged)}"
        assert isinstance(self.final_mse, float), f"Final MSE must be float, got {type(self.final_mse)}"
        assert isinstance(self.iterations, int), f"Iterations must be int, got {type(self.iterations)}"
        assert isinstance(self.forward_coeffs, list), f"Forward coeffs must be list, got {type(self.forward_coeffs)}"
        assert isinstance(self.feedback_coeffs, list), f"Feedback coeffs must be list, got {type(self.feedback_coeffs)}"

        # Value validation
        assert self.final_mse >= 0, f"Final MSE must be non-negative, got {self.final_mse}"
        assert self.iterations >= 0, f"Iterations must be non-negative, got {self.iterations}"

        # Coefficient validation
        assert all(isinstance(c, float) for c in self.forward_coeffs), "All forward coefficients must be floats"
        assert all(isinstance(c, float) for c in self.feedback_coeffs), "All feedback coefficients must be floats"
        assert all(isinstance(mse, float) for mse in self.mse_history), "All MSE values must be floats"


class AdaptiveEqualizer:
    """Adaptive equalizer with type safety"""

    def __init__(self, config: EqualizationConfig) -> None:
        """
        Initialize adaptive equalizer

        Args:
            config: Equalization configuration

        Raises:
            AssertionError: If configuration is invalid
        """
        # Validate input
        assert isinstance(config, EqualizationConfig), f"Config must be EqualizationConfig, got {type(config)}"

        self.config = config

        # Initialize coefficients
        self.forward_coeffs = np.zeros(config.num_forward_taps, dtype=np.float64)
        self.feedback_coeffs = np.zeros(config.num_feedback_taps, dtype=np.float64)

        # Set center tap to 1 for initialization
        center_tap = config.num_forward_taps // 2
        self.forward_coeffs[center_tap] = 1.0

        # Initialize buffers
        self.input_buffer = np.zeros(config.num_forward_taps, dtype=np.float64)
        self.output_buffer = np.zeros(config.num_feedback_taps, dtype=np.float64)

        # Algorithm-specific initialization
        if config.algorithm == EqualizationType.RLS:
            self.P_matrix = np.eye(config.num_forward_taps + config.num_feedback_taps) / 0.01

        self.mse_history: List[float] = []

        logger.info(f"Adaptive equalizer initialized: {config.algorithm.name}")

    def equalize_signal(
        self, input_signal: npt.NDArray[np.float64], reference_signal: Optional[npt.NDArray[np.float64]] = None
    ) -> EqualizationResult:
        """
        Equalize input signal

        Args:
            input_signal: Input signal to equalize
            reference_signal: Optional reference for supervised adaptation

        Returns:
            Equalization result

        Raises:
            ValueError: If equalization fails
        """
        try:
            # Validate input
            self._validate_signal(input_signal)
            if reference_signal is not None:
                self._validate_signal(reference_signal)
                assert len(input_signal) == len(reference_signal), "Input and reference signals must have same length"

            # Run equalization algorithm
            if self.config.algorithm == EqualizationType.LMS:
                return self._run_lms(input_signal, reference_signal)
            elif self.config.algorithm == EqualizationType.RLS:
                return self._run_rls(input_signal, reference_signal)
            elif self.config.algorithm == EqualizationType.CMA:
                return self._run_cma(input_signal)
            elif self.config.algorithm == EqualizationType.DECISION_DIRECTED:
                return self._run_decision_directed(input_signal)
            else:
                raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

        except Exception as e:
            logger.error(f"Equalization failed: {e}")
            raise ValueError(f"Signal equalization failed: {e}")

    def _validate_signal(self, signal: npt.NDArray[np.float64]) -> None:
        """
        Validate signal array

        Args:
            signal: Signal to validate

        Raises:
            AssertionError: If signal is invalid
        """
        # Type validation
        assert isinstance(signal, np.ndarray), f"Signal must be numpy array, got {type(signal)}"

        # Data type validation
        assert np.issubdtype(signal.dtype, np.floating), f"Signal must be floating-point, got {signal.dtype}"

        # Array validation
        assert len(signal) > 0, "Signal cannot be empty"
        assert len(signal) >= self.config.num_forward_taps, f"Signal too short for {self.config.num_forward_taps} taps"

        # Value validation
        assert not np.any(np.isnan(signal)), "Signal contains NaN values"
        assert not np.any(np.isinf(signal)), "Signal contains infinite values"

    def _run_lms(
        self, input_signal: npt.NDArray[np.float64], reference_signal: Optional[npt.NDArray[np.float64]] = None
    ) -> EqualizationResult:
        """
        Run LMS equalization algorithm

        Args:
            input_signal: Input signal
            reference_signal: Reference signal

        Returns:
            Equalization result
        """
        try:
            mse_history = []
            converged = False

            for n in range(len(input_signal)):
                # Update input buffer
                self.input_buffer = np.roll(self.input_buffer, 1)
                self.input_buffer[0] = input_signal[n]

                # Forward filtering
                forward_output = float(np.dot(self.forward_coeffs, self.input_buffer))

                # Feedback filtering (for DFE structure)
                if self.config.structure == EqualizerStructure.DFE and self.config.num_feedback_taps > 0:
                    feedback_output = float(np.dot(self.feedback_coeffs, self.output_buffer))
                    equalizer_output = forward_output - feedback_output
                else:
                    equalizer_output = forward_output

                # Calculate error
                if reference_signal is not None:
                    error = float(reference_signal[n] - equalizer_output)
                else:
                    # Blind adaptation using decision-directed approach
                    decision = self._make_decision(equalizer_output)
                    error = float(decision - equalizer_output)

                # Update coefficients
                self.forward_coeffs += self.config.step_size * error * self.input_buffer

                if self.config.structure == EqualizerStructure.DFE and self.config.num_feedback_taps > 0:
                    self.feedback_coeffs += self.config.step_size * error * self.output_buffer

                    # Update output buffer
                    self.output_buffer = np.roll(self.output_buffer, 1)
                    if reference_signal is not None:
                        self.output_buffer[0] = reference_signal[n]
                    else:
                        self.output_buffer[0] = self._make_decision(equalizer_output)

                # Track MSE
                mse = float(error**2)
                mse_history.append(mse)

                # Check convergence
                if n > 100 and len(mse_history) > 50:
                    recent_mse = np.mean(mse_history[-50:])
                    if recent_mse < 1e-6:
                        converged = True
                        break

            return EqualizationResult(
                converged=converged,
                final_mse=mse_history[-1] if mse_history else float("inf"),
                iterations=len(mse_history),
                forward_coeffs=self.forward_coeffs.tolist(),
                feedback_coeffs=self.feedback_coeffs.tolist(),
                mse_history=mse_history,
            )

        except Exception as e:
            logger.error(f"LMS algorithm failed: {e}")
            raise

    def _run_rls(
        self, input_signal: npt.NDArray[np.float64], reference_signal: Optional[npt.NDArray[np.float64]] = None
    ) -> EqualizationResult:
        """
        Run RLS equalization algorithm

        Args:
            input_signal: Input signal
            reference_signal: Reference signal

        Returns:
            Equalization result
        """
        try:
            mse_history = []
            converged = False

            # Combined coefficient vector
            total_taps = self.config.num_forward_taps + self.config.num_feedback_taps
            coeffs = np.concatenate([self.forward_coeffs, self.feedback_coeffs])

            for n in range(len(input_signal)):
                # Update input buffer
                self.input_buffer = np.roll(self.input_buffer, 1)
                self.input_buffer[0] = input_signal[n]

                # Create input vector
                if self.config.num_feedback_taps > 0:
                    input_vector = np.concatenate([self.input_buffer, self.output_buffer])
                else:
                    input_vector = self.input_buffer

                # Calculate output
                equalizer_output = float(np.dot(coeffs, input_vector))

                # Calculate error
                if reference_signal is not None:
                    error = float(reference_signal[n] - equalizer_output)
                else:
                    decision = self._make_decision(equalizer_output)
                    error = float(decision - equalizer_output)

                # RLS update
                k = self.P_matrix @ input_vector / (self.config.forgetting_factor + input_vector.T @ self.P_matrix @ input_vector)

                coeffs += error * k

                self.P_matrix = (self.P_matrix - np.outer(k, input_vector.T @ self.P_matrix)) / self.config.forgetting_factor

                # Update output buffer for DFE
                if self.config.structure == EqualizerStructure.DFE and self.config.num_feedback_taps > 0:
                    self.output_buffer = np.roll(self.output_buffer, 1)
                    if reference_signal is not None:
                        self.output_buffer[0] = reference_signal[n]
                    else:
                        self.output_buffer[0] = self._make_decision(equalizer_output)

                # Track MSE
                mse = float(error**2)
                mse_history.append(mse)

                # Check convergence
                if n > 100 and len(mse_history) > 50:
                    recent_mse = np.mean(mse_history[-50:])
                    if recent_mse < 1e-6:
                        converged = True
                        break

            # Update coefficient arrays
            self.forward_coeffs = coeffs[: self.config.num_forward_taps]
            if self.config.num_feedback_taps > 0:
                self.feedback_coeffs = coeffs[self.config.num_forward_taps :]

            return EqualizationResult(
                converged=converged,
                final_mse=mse_history[-1] if mse_history else float("inf"),
                iterations=len(mse_history),
                forward_coeffs=self.forward_coeffs.tolist(),
                feedback_coeffs=self.feedback_coeffs.tolist(),
                mse_history=mse_history,
            )

        except Exception as e:
            logger.error(f"RLS algorithm failed: {e}")
            raise

    def _run_cma(self, input_signal: npt.NDArray[np.float64]) -> EqualizationResult:
        """
        Run Constant Modulus Algorithm

        Args:
            input_signal: Input signal

        Returns:
            Equalization result
        """
        try:
            mse_history = []
            converged = False

            # CMA constant (for unit power signals)
            R = 1.0

            for n in range(len(input_signal)):
                # Update input buffer
                self.input_buffer = np.roll(self.input_buffer, 1)
                self.input_buffer[0] = input_signal[n]

                # Forward filtering
                equalizer_output = float(np.dot(self.forward_coeffs, self.input_buffer))

                # CMA error
                error = float(equalizer_output * (R - equalizer_output**2))

                # Update coefficients
                self.forward_coeffs += self.config.step_size * error * self.input_buffer

                # Track cost function
                cost = float((equalizer_output**2 - R) ** 2)
                mse_history.append(cost)

                # Check convergence
                if n > 100 and len(mse_history) > 50:
                    recent_cost = np.mean(mse_history[-50:])
                    if recent_cost < 1e-4:
                        converged = True
                        break

            return EqualizationResult(
                converged=converged,
                final_mse=mse_history[-1] if mse_history else float("inf"),
                iterations=len(mse_history),
                forward_coeffs=self.forward_coeffs.tolist(),
                feedback_coeffs=self.feedback_coeffs.tolist(),
                mse_history=mse_history,
            )

        except Exception as e:
            logger.error(f"CMA algorithm failed: {e}")
            raise

    def _run_decision_directed(self, input_signal: npt.NDArray[np.float64]) -> EqualizationResult:
        """
        Run decision-directed equalization

        Args:
            input_signal: Input signal

        Returns:
            Equalization result
        """
        try:
            mse_history = []
            converged = False

            for n in range(len(input_signal)):
                # Update input buffer
                self.input_buffer = np.roll(self.input_buffer, 1)
                self.input_buffer[0] = input_signal[n]

                # Forward filtering
                forward_output = float(np.dot(self.forward_coeffs, self.input_buffer))

                # Feedback filtering (for DFE)
                if self.config.structure == EqualizerStructure.DFE and self.config.num_feedback_taps > 0:
                    feedback_output = float(np.dot(self.feedback_coeffs, self.output_buffer))
                    equalizer_output = forward_output - feedback_output
                else:
                    equalizer_output = forward_output

                # Make decision
                decision = self._make_decision(equalizer_output)
                error = float(decision - equalizer_output)

                # Update coefficients
                self.forward_coeffs += self.config.step_size * error * self.input_buffer

                if self.config.structure == EqualizerStructure.DFE and self.config.num_feedback_taps > 0:
                    self.feedback_coeffs += self.config.step_size * error * self.output_buffer

                    # Update output buffer with decision
                    self.output_buffer = np.roll(self.output_buffer, 1)
                    self.output_buffer[0] = decision

                # Track MSE
                mse = float(error**2)
                mse_history.append(mse)

                # Check convergence
                if n > 100 and len(mse_history) > 50:
                    recent_mse = np.mean(mse_history[-50:])
                    if recent_mse < 1e-6:
                        converged = True
                        break

            return EqualizationResult(
                converged=converged,
                final_mse=mse_history[-1] if mse_history else float("inf"),
                iterations=len(mse_history),
                forward_coeffs=self.forward_coeffs.tolist(),
                feedback_coeffs=self.feedback_coeffs.tolist(),
                mse_history=mse_history,
            )

        except Exception as e:
            logger.error(f"Decision-directed algorithm failed: {e}")
            raise

    def _make_decision(self, value: float) -> float:
        """
        Make symbol decision

        Args:
            value: Input value

        Returns:
            Decision value
        """
        # Simple hard decision for binary/PAM4
        if value > 0.5:
            return 1.0
        elif value < -0.5:
            return -1.0
        else:
            return 0.0

    def apply_equalization(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Apply current equalization to signal

        Args:
            signal: Input signal

        Returns:
            Equalized signal

        Raises:
            ValueError: If application fails
        """
        try:
            self._validate_signal(signal)

            equalized = np.zeros_like(signal, dtype=np.float64)
            input_buffer = np.zeros(self.config.num_forward_taps, dtype=np.float64)
            output_buffer = np.zeros(self.config.num_feedback_taps, dtype=np.float64)

            for n in range(len(signal)):
                # Update input buffer
                input_buffer = np.roll(input_buffer, 1)
                input_buffer[0] = signal[n]

                # Forward filtering
                forward_output = float(np.dot(self.forward_coeffs, input_buffer))

                # Feedback filtering (for DFE)
                if self.config.structure == EqualizerStructure.DFE and self.config.num_feedback_taps > 0:
                    feedback_output = float(np.dot(self.feedback_coeffs, output_buffer))
                    equalizer_output = forward_output - feedback_output

                    # Update output buffer
                    output_buffer = np.roll(output_buffer, 1)
                    output_buffer[0] = self._make_decision(equalizer_output)
                else:
                    equalizer_output = forward_output

                equalized[n] = equalizer_output

            return equalized

        except Exception as e:
            logger.error(f"Equalization application failed: {e}")
            raise ValueError(f"Failed to apply equalization: {e}")


# Factory functions
def create_lms_equalizer(num_forward_taps: int = 11, num_feedback_taps: int = 0, step_size: float = 0.01) -> AdaptiveEqualizer:
    """
    Create LMS adaptive equalizer

    Args:
        num_forward_taps: Number of forward taps
        num_feedback_taps: Number of feedback taps
        step_size: LMS step size

    Returns:
        Configured LMS equalizer

    Raises:
        AssertionError: If parameters are invalid
    """
    # Validate inputs
    assert isinstance(num_forward_taps, int), f"Forward taps must be int, got {type(num_forward_taps)}"
    assert isinstance(num_feedback_taps, int), f"Feedback taps must be int, got {type(num_feedback_taps)}"
    assert isinstance(step_size, float), f"Step size must be float, got {type(step_size)}"

    assert num_forward_taps > 0, f"Forward taps must be positive, got {num_forward_taps}"
    assert num_feedback_taps >= 0, f"Feedback taps must be non-negative, got {num_feedback_taps}"
    assert 0 < step_size < 1, f"Step size must be between 0 and 1, got {step_size}"

    config = EqualizationConfig(
        algorithm=EqualizationType.LMS,
        structure=EqualizerStructure.DFE if num_feedback_taps > 0 else EqualizerStructure.LINEAR,
        num_forward_taps=num_forward_taps,
        num_feedback_taps=num_feedback_taps,
        step_size=step_size,
        forgetting_factor=0.99,
    )

    return AdaptiveEqualizer(config)


def create_rls_equalizer(
    num_forward_taps: int = 11, num_feedback_taps: int = 0, forgetting_factor: float = 0.99
) -> AdaptiveEqualizer:
    """
    Create RLS adaptive equalizer

    Args:
        num_forward_taps: Number of forward taps
        num_feedback_taps: Number of feedback taps
        forgetting_factor: RLS forgetting factor

    Returns:
        Configured RLS equalizer

    Raises:
        AssertionError: If parameters are invalid
    """
    # Validate inputs
    assert isinstance(num_forward_taps, int), f"Forward taps must be int, got {type(num_forward_taps)}"
    assert isinstance(num_feedback_taps, int), f"Feedback taps must be int, got {type(num_feedback_taps)}"
    assert isinstance(forgetting_factor, float), f"Forgetting factor must be float, got {type(forgetting_factor)}"

    assert num_forward_taps > 0, f"Forward taps must be positive, got {num_forward_taps}"
    assert num_feedback_taps >= 0, f"Feedback taps must be non-negative, got {num_feedback_taps}"
    assert 0 < forgetting_factor <= 1, f"Forgetting factor must be between 0 and 1, got {forgetting_factor}"

    config = EqualizationConfig(
        algorithm=EqualizationType.RLS,
        structure=EqualizerStructure.DFE if num_feedback_taps > 0 else EqualizerStructure.LINEAR,
        num_forward_taps=num_forward_taps,
        num_feedback_taps=num_feedback_taps,
        step_size=0.01,  # Not used in RLS but required
        forgetting_factor=forgetting_factor,
    )

    return AdaptiveEqualizer(config)
