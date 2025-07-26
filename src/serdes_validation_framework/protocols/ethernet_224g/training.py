# src/serdes_validation_framework/protocols/ethernet_224g/training.py

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EqualizerConfig:
    """Enhanced equalizer configuration"""

    fir_taps: int  # Number of feed-forward taps
    dfe_taps: int  # Number of feedback taps
    adaptation_rate: float  # Learning rate
    max_iterations: int  # Maximum training iterations
    convergence_threshold: float  # Convergence criteria
    min_snr: float  # Minimum required SNR
    algorithm: Literal["lms", "rls", "cma"]  # Adaptation algorithm

    def __post_init__(self) -> None:
        """Validate configuration parameters"""
        assert isinstance(self.fir_taps, int) and self.fir_taps > 0, "FIR taps must be a positive integer"
        assert isinstance(self.dfe_taps, int) and self.dfe_taps >= 0, "DFE taps must be a non-negative integer"
        assert (
            isinstance(self.adaptation_rate, float) and 0 < self.adaptation_rate < 1
        ), "Adaptation rate must be a float between 0 and 1"
        assert isinstance(self.max_iterations, int) and self.max_iterations > 0, "Max iterations must be a positive integer"
        assert (
            isinstance(self.convergence_threshold, float) and self.convergence_threshold > 0
        ), "Convergence threshold must be a positive float"
        assert isinstance(self.min_snr, float) and self.min_snr > 0, "Minimum SNR must be a positive float"
        assert self.algorithm in ["lms", "rls", "cma"], "Algorithm must be 'lms', 'rls', or 'cma'"


@dataclass
class EqualizerState:
    """State of the adaptive equalizer"""

    fir_taps: npt.NDArray[np.float64]  # Feed-forward tap weights
    dfe_taps: npt.NDArray[np.float64]  # Feedback tap weights
    error_history: List[float]  # Training error history
    correlation_matrix: Optional[npt.NDArray[np.float64]] = None  # For RLS

    def __post_init__(self) -> None:
        """Validate equalizer state"""
        assert isinstance(self.fir_taps, np.ndarray) and np.issubdtype(
            self.fir_taps.dtype, np.floating
        ), "FIR taps must be a float array"
        assert isinstance(self.dfe_taps, np.ndarray) and np.issubdtype(
            self.dfe_taps.dtype, np.floating
        ), "DFE taps must be a float array"
        assert isinstance(self.error_history, list) and all(
            isinstance(x, float) for x in self.error_history
        ), "Error history must be a list of floats"
        if self.correlation_matrix is not None:
            assert isinstance(self.correlation_matrix, np.ndarray) and np.issubdtype(
                self.correlation_matrix.dtype, np.floating
            ), "Correlation matrix must be a float array"


class EnhancedEqualizer:
    """Advanced adaptive equalizer implementation"""

    def __init__(self, config: Optional[EqualizerConfig] = None) -> None:
        """
        Initialize enhanced equalizer

        Args:
            config: Optional equalizer configuration
        """
        self.config = config or EqualizerConfig(
            fir_taps=11,  # Odd number for symmetric FIR
            dfe_taps=3,  # Typical for 224G
            adaptation_rate=0.01,
            max_iterations=1000,
            convergence_threshold=1e-6,
            min_snr=20.0,
            algorithm="lms",
        )
        self._initialize_state()
        logger.info(f"Enhanced equalizer initialized with {self.config.algorithm} algorithm")

    def _initialize_state(self) -> None:
        """Initialize equalizer state"""
        self.state = EqualizerState(
            fir_taps=np.zeros(self.config.fir_taps, dtype=np.float64),
            dfe_taps=np.zeros(self.config.dfe_taps, dtype=np.float64),
            error_history=[],
            correlation_matrix=None if self.config.algorithm != "rls" else np.eye(self.config.fir_taps, dtype=np.float64) * 100.0,
        )
        # Center tap initialization for FIR
        self.state.fir_taps[self.config.fir_taps // 2] = 1.0

    def train(
        self, received_signal: npt.NDArray[np.float64], target_signal: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], EqualizerState]:
        """
        Train equalizer using selected algorithm

        Args:
            received_signal: Received signal samples
            target_signal: Target signal samples

        Returns:
            Tuple of (equalized signal, final state)
        """
        assert np.issubdtype(received_signal.dtype, np.floating), "Received signal must contain floating-point numbers"
        assert np.issubdtype(target_signal.dtype, np.floating), "Target signal must contain floating-point numbers"
        assert len(received_signal) == len(target_signal), "Signal lengths must match"

        try:
            if self.config.algorithm == "lms":
                return self._train_lms(received_signal, target_signal)
            elif self.config.algorithm == "rls":
                return self._train_rls(received_signal, target_signal)
            else:  # CMA
                return self._train_cma(received_signal)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _train_lms(
        self, received_signal: npt.NDArray[np.float64], target_signal: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], EqualizerState]:
        """
        Train using LMS algorithm with both FIR and DFE
        """
        try:
            output = np.zeros_like(received_signal)
            dfe_memory = np.zeros(self.config.dfe_taps)

            for i in range(self.config.fir_taps, len(received_signal)):
                # FIR filtering
                fir_input = received_signal[i - self.config.fir_taps : i]
                fir_output = np.dot(self.state.fir_taps, fir_input)

                # DFE correction
                dfe_correction = np.dot(self.state.dfe_taps, dfe_memory)
                output[i] = fir_output + dfe_correction

                # Error calculation
                error = target_signal[i] - output[i]
                self.state.error_history.append(float(error))

                # Update FIR taps
                self.state.fir_taps += self.config.adaptation_rate * error * fir_input

                # Update DFE taps
                self.state.dfe_taps += self.config.adaptation_rate * error * dfe_memory

                # Update DFE memory
                dfe_memory[1:] = dfe_memory[:-1]
                dfe_memory[0] = output[i]

                # Check convergence
                if len(self.state.error_history) > 100:
                    recent_error = np.mean(np.abs(self.state.error_history[-100:]))
                    if recent_error < self.config.convergence_threshold:
                        break

            return output, self.state

        except Exception as e:
            logger.error(f"LMS training failed: {e}")
            raise

    def _train_rls(
        self, received_signal: npt.NDArray[np.float64], target_signal: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], EqualizerState]:
        """
        Train using RLS algorithm
        """
        try:
            output = np.zeros_like(received_signal)
            forgetting_factor = 0.99  # Typical RLS forgetting factor

            for i in range(self.config.fir_taps, len(received_signal)):
                # Input vector
                input_vector = received_signal[i - self.config.fir_taps : i]

                # Kalman gain calculation
                P = self.state.correlation_matrix
                k = P @ input_vector / (forgetting_factor + input_vector @ P @ input_vector)

                # Output and error
                output[i] = np.dot(self.state.fir_taps, input_vector)
                error = target_signal[i] - output[i]
                self.state.error_history.append(float(error))

                # Update taps
                self.state.fir_taps += k * error

                # Update correlation matrix
                self.state.correlation_matrix = (P - np.outer(k, input_vector @ P)) / forgetting_factor

                # Check convergence
                if len(self.state.error_history) > 100:
                    recent_error = np.mean(np.abs(self.state.error_history[-100:]))
                    if recent_error < self.config.convergence_threshold:
                        break

            return output, self.state

        except Exception as e:
            logger.error(f"RLS training failed: {e}")
            raise

    def _train_cma(self, received_signal: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], EqualizerState]:
        """
        Train using Constant Modulus Algorithm
        """
        try:
            output = np.zeros_like(received_signal)
            target_modulus = 1.0  # Normalized target amplitude

            for i in range(self.config.fir_taps, len(received_signal)):
                input_vector = received_signal[i - self.config.fir_taps : i]
                output[i] = np.dot(self.state.fir_taps, input_vector)

                # CMA error
                error = target_modulus - output[i] ** 2
                self.state.error_history.append(float(error))

                # Update taps
                self.state.fir_taps += self.config.adaptation_rate * error * output[i] * input_vector

                # Check convergence
                if len(self.state.error_history) > 100:
                    recent_error = np.mean(np.abs(self.state.error_history[-100:]))
                    if recent_error < self.config.convergence_threshold:
                        break

            return output, self.state

        except Exception as e:
            logger.error(f"CMA training failed: {e}")
            raise

    def equalize(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Apply trained equalizer to new signal

        Args:
            signal: Input signal to equalize

        Returns:
            Equalized signal
        """
        assert np.issubdtype(signal.dtype, np.floating), "Signal must contain floating-point numbers"

        try:
            output = np.zeros_like(signal)
            dfe_memory = np.zeros(self.config.dfe_taps)

            for i in range(self.config.fir_taps, len(signal)):
                # FIR filtering
                fir_input = signal[i - self.config.fir_taps : i]
                fir_output = np.dot(self.state.fir_taps, fir_input)

                # DFE correction
                dfe_correction = np.dot(self.state.dfe_taps, dfe_memory)
                output[i] = fir_output + dfe_correction

                # Update DFE memory
                dfe_memory[1:] = dfe_memory[:-1]
                dfe_memory[0] = output[i]

            return output

        except Exception as e:
            logger.error(f"Equalization failed: {e}")
            raise

    def analyze_performance(
        self, original_signal: npt.NDArray[np.float64], equalized_signal: npt.NDArray[np.float64]
    ) -> Dict[str, float]:
        """
        Analyze equalizer performance

        Args:
            original_signal: Original received signal
            equalized_signal: Equalized signal

        Returns:
            Dictionary of performance metrics
        """
        try:
            # Calculate SNR improvement
            original_snr = self._calculate_snr(original_signal)
            equalized_snr = self._calculate_snr(equalized_signal)

            # Calculate MSE
            mse = np.mean((equalized_signal - np.mean(equalized_signal)) ** 2)

            # Calculate convergence time
            convergence_time = len(self.state.error_history) / len(original_signal)

            return {
                "original_snr_db": float(original_snr),
                "equalized_snr_db": float(equalized_snr),
                "snr_improvement_db": float(equalized_snr - original_snr),
                "mse": float(mse),
                "convergence_time": float(convergence_time),
            }

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            raise

    def _calculate_snr(self, signal: npt.NDArray[np.float64]) -> float:
        """Calculate signal-to-noise ratio"""
        try:
            signal_power = np.mean(signal**2)
            noise_power = np.var(signal)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")
            return float(snr)
        except Exception as e:
            logger.error(f"SNR calculation failed: {e}")
            raise
