"""
PCIe Link Training Module

This module provides comprehensive PCIe link training functionality with
type-safe implementation for both NRZ and PAM4 modes.

Features:
- Adaptive equalization
- Preset-based training
- SNR optimization
- Convergence detection
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.optimize import minimize

from .constants import SignalMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """PCIe link training phases"""
    PHASE0 = auto()  # Initial preset
    PHASE1 = auto()  # Coefficient optimization
    PHASE2 = auto()  # Fine tuning
    PHASE3 = auto()  # Final verification


class EqualizerType(Enum):
    """Types of equalizers"""
    TX_FFE = auto()    # Transmit Feed-Forward Equalizer
    RX_CTLE = auto()   # Receive Continuous Time Linear Equalizer
    RX_DFE = auto()    # Receive Decision Feedback Equalizer


@dataclass
class EqualizerConfig:
    """Equalizer configuration with validation"""
    eq_type: EqualizerType
    num_taps: int
    tap_range: Tuple[float, float]
    step_size: float
    
    def __post_init__(self) -> None:
        """
        Validate equalizer configuration
        
        Raises:
            AssertionError: If configuration is invalid
        """
        # Type validation
        assert isinstance(self.eq_type, EqualizerType), \
            f"Equalizer type must be EqualizerType, got {type(self.eq_type)}"
        assert isinstance(self.num_taps, int), \
            f"Number of taps must be int, got {type(self.num_taps)}"
        assert isinstance(self.tap_range, tuple), \
            f"Tap range must be tuple, got {type(self.tap_range)}"
        assert len(self.tap_range) == 2, \
            f"Tap range must have 2 elements, got {len(self.tap_range)}"
        assert all(isinstance(x, float) for x in self.tap_range), \
            "Tap range values must be floats"
        assert isinstance(self.step_size, float), \
            f"Step size must be float, got {type(self.step_size)}"
        
        # Value validation
        assert self.num_taps > 0, \
            f"Number of taps must be positive, got {self.num_taps}"
        assert self.tap_range[0] < self.tap_range[1], \
            f"Invalid tap range: {self.tap_range[0]} >= {self.tap_range[1]}"
        assert self.step_size > 0, \
            f"Step size must be positive, got {self.step_size}"


@dataclass
class TrainingConfig:
    """Link training configuration with validation"""
    mode: SignalMode
    target_ber: float
    max_iterations: int
    convergence_threshold: float
    equalizers: List[EqualizerConfig] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """
        Validate training configuration
        
        Raises:
            AssertionError: If configuration is invalid
        """
        # Type validation
        assert isinstance(self.mode, SignalMode), \
            f"Mode must be SignalMode, got {type(self.mode)}"
        assert isinstance(self.target_ber, float), \
            f"Target BER must be float, got {type(self.target_ber)}"
        assert isinstance(self.max_iterations, int), \
            f"Max iterations must be int, got {type(self.max_iterations)}"
        assert isinstance(self.convergence_threshold, float), \
            f"Convergence threshold must be float, got {type(self.convergence_threshold)}"
        
        # Value validation
        assert 0 < self.target_ber < 1, \
            f"Target BER must be between 0 and 1, got {self.target_ber}"
        assert self.max_iterations > 0, \
            f"Max iterations must be positive, got {self.max_iterations}"
        assert 0 < self.convergence_threshold < 1, \
            f"Convergence threshold must be between 0 and 1, got {self.convergence_threshold}"
        
        # Equalizer validation
        assert all(isinstance(eq, EqualizerConfig) for eq in self.equalizers), \
            "All equalizers must be EqualizerConfig instances"


@dataclass
class TrainingResult:
    """Link training result with validation"""
    success: bool
    final_ber: float
    iterations: int
    phase: TrainingPhase
    equalizer_coeffs: Dict[EqualizerType, List[float]]
    snr_history: List[float] = field(default_factory=list)
    error_message: Optional[str] = None
    
    def __post_init__(self) -> None:
        """
        Validate training result
        
        Raises:
            AssertionError: If result is invalid
        """
        # Type validation
        assert isinstance(self.success, bool), \
            f"Success must be bool, got {type(self.success)}"
        assert isinstance(self.final_ber, float), \
            f"Final BER must be float, got {type(self.final_ber)}"
        assert isinstance(self.iterations, int), \
            f"Iterations must be int, got {type(self.iterations)}"
        assert isinstance(self.phase, TrainingPhase), \
            f"Phase must be TrainingPhase, got {type(self.phase)}"
        assert isinstance(self.equalizer_coeffs, dict), \
            f"Equalizer coeffs must be dict, got {type(self.equalizer_coeffs)}"
        
        # Value validation
        assert self.final_ber >= 0, \
            f"Final BER must be non-negative, got {self.final_ber}"
        assert self.iterations >= 0, \
            f"Iterations must be non-negative, got {self.iterations}"
        
        # Coefficient validation
        for eq_type, coeffs in self.equalizer_coeffs.items():
            assert isinstance(eq_type, EqualizerType), \
                f"Equalizer type must be EqualizerType, got {type(eq_type)}"
            assert isinstance(coeffs, list), \
                f"Coefficients must be list, got {type(coeffs)}"
            assert all(isinstance(c, float) for c in coeffs), \
                "All coefficients must be floats"
        
        # SNR history validation
        assert all(isinstance(snr, float) for snr in self.snr_history), \
            "All SNR values must be floats"


class LinkTrainer:
    """PCIe link training engine with type safety"""
    
    def __init__(self, config: TrainingConfig) -> None:
        """
        Initialize link trainer
        
        Args:
            config: Training configuration
            
        Raises:
            AssertionError: If configuration is invalid
        """
        # Validate input
        assert isinstance(config, TrainingConfig), \
            f"Config must be TrainingConfig, got {type(config)}"
        
        self.config = config
        self.current_phase = TrainingPhase.PHASE0
        self.equalizer_coeffs: Dict[EqualizerType, npt.NDArray[np.float64]] = {}
        self.snr_history: List[float] = []
        
        # Initialize equalizers
        self._initialize_equalizers()
        
        logger.info(f"Link trainer initialized for {config.mode.name} mode")
    
    def _initialize_equalizers(self) -> None:
        """Initialize equalizer coefficients"""
        for eq_config in self.config.equalizers:
            # Initialize with center tap = 1, others = 0
            coeffs = np.zeros(eq_config.num_taps, dtype=np.float64)
            center_tap = eq_config.num_taps // 2
            coeffs[center_tap] = 1.0
            
            self.equalizer_coeffs[eq_config.eq_type] = coeffs
    
    def run_training(
        self,
        signal_data: Dict[str, npt.NDArray[np.float64]]
    ) -> TrainingResult:
        """
        Run complete link training sequence
        
        Args:
            signal_data: Dictionary with time and voltage arrays
            
        Returns:
            Training result
            
        Raises:
            ValueError: If training fails
        """
        try:
            # Validate input data
            self._validate_signal_data(signal_data)
            
            # Training phases
            for phase in TrainingPhase:
                self.current_phase = phase
                logger.info(f"Starting training {phase.name}")
                
                success = self._run_training_phase(signal_data, phase)
                if not success:
                    return TrainingResult(
                        success=False,
                        final_ber=1.0,
                        iterations=0,
                        phase=phase,
                        equalizer_coeffs=self._get_coeffs_dict(),
                        snr_history=self.snr_history,
                        error_message=f"Training failed in {phase.name}"
                    )
            
            # Calculate final BER
            final_ber = self._calculate_ber(signal_data)
            
            return TrainingResult(
                success=final_ber <= self.config.target_ber,
                final_ber=final_ber,
                iterations=len(self.snr_history),
                phase=TrainingPhase.PHASE3,
                equalizer_coeffs=self._get_coeffs_dict(),
                snr_history=self.snr_history
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise ValueError(f"Link training failed: {e}")
    
    def _validate_signal_data(
        self,
        signal_data: Dict[str, npt.NDArray[np.float64]]
    ) -> None:
        """
        Validate signal data
        
        Args:
            signal_data: Signal data dictionary
            
        Raises:
            AssertionError: If data is invalid
        """
        # Check required keys
        assert 'time' in signal_data, "Signal data must contain 'time'"
        assert 'voltage' in signal_data, "Signal data must contain 'voltage'"
        
        time_data = signal_data['time']
        voltage_data = signal_data['voltage']
        
        # Type validation
        assert isinstance(time_data, np.ndarray), \
            f"Time data must be numpy array, got {type(time_data)}"
        assert isinstance(voltage_data, np.ndarray), \
            f"Voltage data must be numpy array, got {type(voltage_data)}"
        
        # Data type validation
        assert np.issubdtype(time_data.dtype, np.floating), \
            f"Time data must be floating-point, got {time_data.dtype}"
        assert np.issubdtype(voltage_data.dtype, np.floating), \
            f"Voltage data must be floating-point, got {voltage_data.dtype}"
        
        # Array validation
        assert len(time_data) == len(voltage_data), \
            f"Array length mismatch: {len(time_data)} != {len(voltage_data)}"
        assert len(time_data) > 0, "Arrays cannot be empty"
        
        # Value validation
        assert not np.any(np.isnan(time_data)), "Time data contains NaN"
        assert not np.any(np.isnan(voltage_data)), "Voltage data contains NaN"
        assert not np.any(np.isinf(time_data)), "Time data contains infinity"
        assert not np.any(np.isinf(voltage_data)), "Voltage data contains infinity"
    
    def _run_training_phase(
        self,
        signal_data: Dict[str, npt.NDArray[np.float64]],
        phase: TrainingPhase
    ) -> bool:
        """
        Run specific training phase
        
        Args:
            signal_data: Signal data
            phase: Training phase
            
        Returns:
            True if phase successful
        """
        try:
            if phase == TrainingPhase.PHASE0:
                return self._run_preset_phase(signal_data)
            elif phase == TrainingPhase.PHASE1:
                return self._run_optimization_phase(signal_data)
            elif phase == TrainingPhase.PHASE2:
                return self._run_fine_tuning_phase(signal_data)
            elif phase == TrainingPhase.PHASE3:
                return self._run_verification_phase(signal_data)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Phase {phase.name} failed: {e}")
            return False
    
    def _run_preset_phase(
        self,
        signal_data: Dict[str, npt.NDArray[np.float64]]
    ) -> bool:
        """
        Run preset initialization phase
        
        Args:
            signal_data: Signal data
            
        Returns:
            True if successful
        """
        try:
            # Apply initial presets based on mode
            if self.config.mode == SignalMode.NRZ:
                preset_coeffs = [0.0, 0.0, 1.0, 0.0, 0.0]  # Simple preset
            else:  # PAM4
                preset_coeffs = [0.1, -0.2, 1.0, -0.2, 0.1]  # More complex preset
            
            # Apply to TX FFE
            for eq_config in self.config.equalizers:
                if eq_config.eq_type == EqualizerType.TX_FFE:
                    coeffs = np.array(preset_coeffs[:eq_config.num_taps], dtype=np.float64)
                    self.equalizer_coeffs[eq_config.eq_type] = coeffs
            
            # Calculate initial SNR
            snr = self._calculate_snr(signal_data)
            self.snr_history.append(snr)
            
            return True
            
        except Exception as e:
            logger.error(f"Preset phase failed: {e}")
            return False
    
    def _run_optimization_phase(
        self,
        signal_data: Dict[str, npt.NDArray[np.float64]]
    ) -> bool:
        """
        Run coefficient optimization phase
        
        Args:
            signal_data: Signal data
            
        Returns:
            True if successful
        """
        try:
            # Optimize each equalizer
            for eq_config in self.config.equalizers:
                success = self._optimize_equalizer(signal_data, eq_config)
                if not success:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Optimization phase failed: {e}")
            return False
    
    def _optimize_equalizer(
        self,
        signal_data: Dict[str, npt.NDArray[np.float64]],
        eq_config: EqualizerConfig
    ) -> bool:
        """
        Optimize specific equalizer
        
        Args:
            signal_data: Signal data
            eq_config: Equalizer configuration
            
        Returns:
            True if successful
        """
        try:
            current_coeffs = self.equalizer_coeffs[eq_config.eq_type]
            
            # Define optimization objective
            def objective(coeffs: npt.NDArray[np.float64]) -> float:
                # Update coefficients
                self.equalizer_coeffs[eq_config.eq_type] = coeffs.astype(np.float64)
                
                # Calculate BER (negative for minimization)
                ber = self._calculate_ber(signal_data)
                return float(ber)
            
            # Set bounds
            bounds = [(eq_config.tap_range[0], eq_config.tap_range[1]) 
                     for _ in range(len(current_coeffs))]
            
            # Run optimization
            result = minimize(
                objective,
                current_coeffs,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            if result.success:
                self.equalizer_coeffs[eq_config.eq_type] = result.x.astype(np.float64)
                
                # Update SNR history
                snr = self._calculate_snr(signal_data)
                self.snr_history.append(snr)
                
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Equalizer optimization failed: {e}")
            return False
    
    def _run_fine_tuning_phase(
        self,
        signal_data: Dict[str, npt.NDArray[np.float64]]
    ) -> bool:
        """
        Run fine tuning phase
        
        Args:
            signal_data: Signal data
            
        Returns:
            True if successful
        """
        try:
            # Fine-tune with smaller step sizes
            for eq_config in self.config.equalizers:
                current_coeffs = self.equalizer_coeffs[eq_config.eq_type]
                
                # Small perturbations
                for i in range(eq_config.num_taps):
                    # Try positive perturbation
                    test_coeffs = current_coeffs.copy()
                    test_coeffs[i] += eq_config.step_size * 0.1
                    
                    # Check if within bounds
                    if eq_config.tap_range[0] <= test_coeffs[i] <= eq_config.tap_range[1]:
                        self.equalizer_coeffs[eq_config.eq_type] = test_coeffs
                        new_ber = self._calculate_ber(signal_data)
                        
                        # Keep if better
                        current_ber = self._calculate_ber(signal_data)
                        if new_ber < current_ber:
                            current_coeffs = test_coeffs
                        else:
                            # Restore original
                            self.equalizer_coeffs[eq_config.eq_type] = current_coeffs
            
            # Update SNR
            snr = self._calculate_snr(signal_data)
            self.snr_history.append(snr)
            
            return True
            
        except Exception as e:
            logger.error(f"Fine tuning phase failed: {e}")
            return False
    
    def _run_verification_phase(
        self,
        signal_data: Dict[str, npt.NDArray[np.float64]]
    ) -> bool:
        """
        Run verification phase
        
        Args:
            signal_data: Signal data
            
        Returns:
            True if successful
        """
        try:
            # Final BER check
            final_ber = self._calculate_ber(signal_data)
            final_snr = self._calculate_snr(signal_data)
            
            self.snr_history.append(final_snr)
            
            # Check convergence
            success = final_ber <= self.config.target_ber
            
            if success:
                logger.info(f"Training converged: BER={final_ber:.2e}, SNR={final_snr:.1f}dB")
            else:
                logger.warning(f"Training did not converge: BER={final_ber:.2e}")
            
            return success
            
        except Exception as e:
            logger.error(f"Verification phase failed: {e}")
            return False
    
    def _calculate_ber(
        self,
        signal_data: Dict[str, npt.NDArray[np.float64]]
    ) -> float:
        """
        Calculate bit error rate
        
        Args:
            signal_data: Signal data
            
        Returns:
            BER value
        """
        try:
            voltage = signal_data['voltage']
            
            # Simple BER estimation based on eye opening
            if self.config.mode == SignalMode.NRZ:
                # NRZ BER estimation
                levels = self._find_signal_levels(voltage, 2)
                eye_opening = float(np.abs(levels[1] - levels[0]))
                noise_std = float(np.std(voltage))
                
                # Q-factor approximation
                q_factor = eye_opening / (2 * noise_std)
                ber = 0.5 * np.exp(-q_factor**2 / 2)
                
            else:  # PAM4
                # PAM4 BER estimation
                levels = self._find_signal_levels(voltage, 4)
                min_separation = float(np.min(np.diff(np.sort(levels))))
                noise_std = float(np.std(voltage))
                
                # Simplified PAM4 BER
                q_factor = min_separation / (2 * noise_std)
                ber = 1.5 * np.exp(-q_factor**2 / 2)
            
            return float(np.clip(ber, 1e-15, 1.0))
            
        except Exception as e:
            logger.error(f"BER calculation failed: {e}")
            return 1.0  # Worst case
    
    def _calculate_snr(
        self,
        signal_data: Dict[str, npt.NDArray[np.float64]]
    ) -> float:
        """
        Calculate signal-to-noise ratio
        
        Args:
            signal_data: Signal data
            
        Returns:
            SNR in dB
        """
        try:
            voltage = signal_data['voltage']
            
            # Signal power
            signal_power = float(np.mean(voltage**2))
            
            # Noise power estimation
            # Use high-frequency components as noise estimate
            from scipy.signal import butter, filtfilt
            
            # High-pass filter for noise
            nyquist = 0.5 * len(voltage)
            high_cutoff = 0.8  # Normalized frequency
            b, a = butter(4, high_cutoff, btype='high')
            noise_estimate = filtfilt(b, a, voltage)
            noise_power = float(np.mean(noise_estimate**2))
            
            # Calculate SNR
            if noise_power > 0:
                snr_db = float(10 * np.log10(signal_power / noise_power))
            else:
                snr_db = float('inf')
            
            return snr_db
            
        except Exception as e:
            logger.error(f"SNR calculation failed: {e}")
            return 0.0
    
    def _find_signal_levels(
        self,
        voltage: npt.NDArray[np.float64],
        num_levels: int
    ) -> npt.NDArray[np.float64]:
        """
        Find signal levels using clustering
        
        Args:
            voltage: Voltage data
            num_levels: Expected number of levels
            
        Returns:
            Array of voltage levels
        """
        try:
            from sklearn.cluster import KMeans
            
            # Reshape for clustering
            data = voltage.reshape(-1, 1)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=num_levels, random_state=42)
            kmeans.fit(data)
            
            # Get cluster centers
            levels = kmeans.cluster_centers_.flatten()
            
            return np.sort(levels).astype(np.float64)
            
        except Exception as e:
            logger.error(f"Level detection failed: {e}")
            # Fallback: use histogram peaks
            hist, bins = np.histogram(voltage, bins=100)
            peaks = signal.find_peaks(hist, height=np.max(hist)/4)[0]
            
            if len(peaks) >= num_levels:
                return np.sort(bins[peaks[:num_levels]]).astype(np.float64)
            else:
                # Linear spacing fallback
                return np.linspace(np.min(voltage), np.max(voltage), num_levels).astype(np.float64)
    
    def _get_coeffs_dict(self) -> Dict[EqualizerType, List[float]]:
        """
        Get equalizer coefficients as dictionary
        
        Returns:
            Dictionary of coefficients
        """
        return {
            eq_type: coeffs.tolist()
            for eq_type, coeffs in self.equalizer_coeffs.items()
        }


# Factory functions
def create_nrz_trainer(
    target_ber: float = 1e-12,
    max_iterations: int = 1000
) -> LinkTrainer:
    """
    Create NRZ link trainer
    
    Args:
        target_ber: Target bit error rate
        max_iterations: Maximum training iterations
        
    Returns:
        Configured link trainer
        
    Raises:
        AssertionError: If parameters are invalid
    """
    # Validate inputs
    assert isinstance(target_ber, float), \
        f"Target BER must be float, got {type(target_ber)}"
    assert isinstance(max_iterations, int), \
        f"Max iterations must be int, got {type(max_iterations)}"
    assert 0 < target_ber < 1, \
        f"Target BER must be between 0 and 1, got {target_ber}"
    assert max_iterations > 0, \
        f"Max iterations must be positive, got {max_iterations}"
    
    # Create configuration
    config = TrainingConfig(
        mode=SignalMode.NRZ,
        target_ber=target_ber,
        max_iterations=max_iterations,
        convergence_threshold=0.01,
        equalizers=[
            EqualizerConfig(
                eq_type=EqualizerType.TX_FFE,
                num_taps=5,
                tap_range=(-0.5, 0.5),
                step_size=0.01
            )
        ]
    )
    
    return LinkTrainer(config)


def create_pam4_trainer(
    target_ber: float = 1e-12,
    max_iterations: int = 2000
) -> LinkTrainer:
    """
    Create PAM4 link trainer
    
    Args:
        target_ber: Target bit error rate
        max_iterations: Maximum training iterations
        
    Returns:
        Configured link trainer
        
    Raises:
        AssertionError: If parameters are invalid
    """
    # Validate inputs
    assert isinstance(target_ber, float), \
        f"Target BER must be float, got {type(target_ber)}"
    assert isinstance(max_iterations, int), \
        f"Max iterations must be int, got {type(max_iterations)}"
    assert 0 < target_ber < 1, \
        f"Target BER must be between 0 and 1, got {target_ber}"
    assert max_iterations > 0, \
        f"Max iterations must be positive, got {max_iterations}"
    
    # Create configuration
    config = TrainingConfig(
        mode=SignalMode.PAM4,
        target_ber=target_ber,
        max_iterations=max_iterations,
        convergence_threshold=0.005,  # Tighter convergence for PAM4
        equalizers=[
            EqualizerConfig(
                eq_type=EqualizerType.TX_FFE,
                num_taps=7,
                tap_range=(-0.3, 0.3),
                step_size=0.005
            ),
            EqualizerConfig(
                eq_type=EqualizerType.RX_CTLE,
                num_taps=3,
                tap_range=(-0.2, 0.2),
                step_size=0.01
            )
        ]
    )
    
    return LinkTrainer(config)
