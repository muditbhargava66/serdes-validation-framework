"""
PCIe Dual Mode Control Module

This module provides functionality for controlling and switching between
NRZ and PAM4 signaling modes for PCIe with comprehensive type validation.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalMode(Enum):
    """PCIe signaling modes"""
    NRZ = auto()
    PAM4 = auto()


class SwitchResult(Enum):
    """Mode switch result status"""
    SUCCESS = auto()
    FAILURE = auto()
    NOT_SUPPORTED = auto()


@dataclass
class ModeConfig:
    """
    Mode configuration parameters with validation
    """
    mode: SignalMode
    sample_rate: float
    bandwidth: float
    eye_height_threshold: float
    equalization_taps: int

    def __post_init__(self) -> None:
        """
        Validate configuration parameters
        
        Raises:
            AssertionError: If parameters are invalid
        """
        # Type validation
        assert isinstance(self.mode, SignalMode), \
            f"Mode must be SignalMode, got {type(self.mode)}"
        assert isinstance(self.sample_rate, float), \
            f"Sample rate must be float, got {type(self.sample_rate)}"
        assert isinstance(self.bandwidth, float), \
            f"Bandwidth must be float, got {type(self.bandwidth)}"
        assert isinstance(self.eye_height_threshold, float), \
            f"Eye height threshold must be float, got {type(self.eye_height_threshold)}"
        assert isinstance(self.equalization_taps, int), \
            f"Equalization taps must be integer, got {type(self.equalization_taps)}"
        
        # Value validation
        assert self.sample_rate > 0, f"Sample rate must be positive, got {self.sample_rate}"
        assert self.bandwidth > 0, f"Bandwidth must be positive, got {self.bandwidth}"
        assert 0 <= self.eye_height_threshold <= 1, \
            f"Eye height threshold must be between 0 and 1, got {self.eye_height_threshold}"
        assert self.equalization_taps > 0, f"Equalization taps must be positive, got {self.equalization_taps}"


@dataclass
class SwitchStatus:
    """
    Mode switch status with validation
    """
    success: bool
    current_mode: SignalMode
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Validate status parameters
        
        Raises:
            AssertionError: If parameters are invalid
        """
        # Type validation
        assert isinstance(self.success, bool), \
            f"Success must be boolean, got {type(self.success)}"
        assert isinstance(self.current_mode, SignalMode), \
            f"Current mode must be SignalMode, got {type(self.current_mode)}"
        if self.error_message is not None:
            assert isinstance(self.error_message, str), \
                f"Error message must be string, got {type(self.error_message)}"


class ModeController:
    """
    PCIe dual-mode controller with type safety
    """
    
    def __init__(self, default_mode: SignalMode = SignalMode.NRZ) -> None:
        """
        Initialize mode controller
        
        Args:
            default_mode: Initial signal mode
            
        Raises:
            AssertionError: If default mode is invalid
        """
        # Validate input
        assert isinstance(default_mode, SignalMode), \
            f"Default mode must be SignalMode, got {type(default_mode)}"
        
        self.current_mode = default_mode
        self.configs: Dict[SignalMode, ModeConfig] = {}
        logger.info(f"Mode controller initialized in {default_mode.name} mode")

    def register_mode_config(self, config: ModeConfig) -> None:
        """
        Register configuration for a signal mode
        
        Args:
            config: Mode configuration
            
        Raises:
            AssertionError: If configuration is invalid
        """
        # Validate input
        assert isinstance(config, ModeConfig), \
            f"Config must be ModeConfig, got {type(config)}"
        
        self.configs[config.mode] = config
        logger.info(f"Registered configuration for {config.mode.name} mode")

    def switch_mode(
        self, 
        target_mode: SignalMode,
        data: Optional[Dict[str, npt.NDArray[np.float64]]] = None
    ) -> SwitchStatus:
        """
        Switch between signal modes
        
        Args:
            target_mode: Target signal mode
            data: Optional signal data for validation
            
        Returns:
            Switch status result
            
        Raises:
            AssertionError: If parameters are invalid
        """
        # Validate input
        assert isinstance(target_mode, SignalMode), \
            f"Target mode must be SignalMode, got {type(target_mode)}"
        if data is not None:
            assert isinstance(data, dict), f"Data must be dictionary, got {type(data)}"
            if 'voltage' in data:
                assert isinstance(data['voltage'], np.ndarray), \
                    f"Voltage data must be numpy array, got {type(data['voltage'])}"
                assert np.issubdtype(data['voltage'].dtype, np.floating), \
                    f"Voltage data must be floating-point, got {data['voltage'].dtype}"
        
        # Check if already in target mode
        if self.current_mode == target_mode:
            return SwitchStatus(
                success=True,
                current_mode=self.current_mode,
                error_message=None
            )
        
        # Check if configuration exists
        if target_mode not in self.configs:
            return SwitchStatus(
                success=False,
                current_mode=self.current_mode,
                error_message=f"No configuration registered for {target_mode.name} mode"
            )
        
        try:
            # Validate signal quality if data provided
            if data is not None and 'voltage' in data:
                valid = self._validate_signal_quality(
                    data['voltage'],
                    target_mode,
                    self.configs[target_mode].eye_height_threshold
                )
                
                if not valid:
                    return SwitchStatus(
                        success=False,
                        current_mode=self.current_mode,
                        error_message=f"Signal quality insufficient for {target_mode.name} mode"
                    )
            
            # Perform mode switch
            self._perform_mode_switch(target_mode)
            
            # Update current mode
            self.current_mode = target_mode
            
            return SwitchStatus(
                success=True,
                current_mode=self.current_mode,
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Mode switch failed: {str(e)}")
            return SwitchStatus(
                success=False,
                current_mode=self.current_mode,
                error_message=f"Mode switch failed: {str(e)}"
            )

    def _validate_signal_quality(
        self, 
        voltage_data: npt.NDArray[np.float64],
        target_mode: SignalMode,
        threshold: float
    ) -> bool:
        """
        Validate signal quality for mode switch
        
        Args:
            voltage_data: Voltage measurements array
            target_mode: Target signal mode
            threshold: Quality threshold
            
        Returns:
            True if signal quality sufficient, False otherwise
            
        Raises:
            AssertionError: If parameters are invalid
        """
        # Validate inputs
        assert isinstance(voltage_data, np.ndarray), \
            f"Voltage data must be numpy array, got {type(voltage_data)}"
        assert np.issubdtype(voltage_data.dtype, np.floating), \
            f"Voltage data must be floating-point, got {voltage_data.dtype}"
        assert isinstance(target_mode, SignalMode), \
            f"Target mode must be SignalMode, got {type(target_mode)}"
        assert isinstance(threshold, float), \
            f"Threshold must be float, got {type(threshold)}"
        assert 0 <= threshold <= 1, \
            f"Threshold must be between 0 and 1, got {threshold}"
        
        try:
            if target_mode == SignalMode.PAM4:
                # Check PAM4 levels
                hist, bins = np.histogram(voltage_data, bins=100)
                peaks, _ = self._find_peaks(hist)
                
                # Need at least 4 clear peaks for PAM4
                if len(peaks) < 4:
                    logger.warning(f"Found only {len(peaks)} levels, need 4 for PAM4")
                    return False
                
                # Check level separation
                levels = bins[peaks]
                sorted_levels = np.sort(levels)
                level_gaps = np.diff(sorted_levels[:4])  # Use first 4 if more found
                min_gap = float(np.min(level_gaps))
                
                # Calculate peak-to-noise ratio
                noise = float(np.std(voltage_data))
                snr = min_gap / noise
                
                return snr >= threshold
                
            else:  # NRZ mode
                # Check NRZ levels
                hist, bins = np.histogram(voltage_data, bins=100)
                peaks, _ = self._find_peaks(hist)
                
                # Need at least 2 clear peaks for NRZ
                if len(peaks) < 2:
                    logger.warning(f"Found only {len(peaks)} levels, need 2 for NRZ")
                    return False
                
                # For NRZ, just ensure signal amplitude is sufficient
                v_pp = float(np.max(voltage_data) - np.min(voltage_data))
                noise = float(np.std(voltage_data))
                snr = v_pp / noise
                
                return snr >= threshold
                
        except Exception as e:
            logger.error(f"Signal quality validation failed: {str(e)}")
            return False

    def _find_peaks(
        self, 
        histogram: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """
        Find peaks in histogram for level detection
        
        Args:
            histogram: Signal histogram
            
        Returns:
            Tuple of (peak indices, peak heights)
            
        Raises:
            AssertionError: If histogram is invalid
        """
        # Validate input
        assert isinstance(histogram, np.ndarray), \
            f"Histogram must be numpy array, got {type(histogram)}"
        
        # Find local maxima
        peak_indices = []
        peak_heights = []
        
        for i in range(1, len(histogram) - 1):
            if histogram[i] > histogram[i-1] and histogram[i] > histogram[i+1]:
                peak_indices.append(i)
                peak_heights.append(histogram[i])
        
        # Sort by height (descending)
        sorted_indices = np.argsort(peak_heights)[::-1]
        peak_indices = np.array(peak_indices, dtype=np.int64)[sorted_indices]
        peak_heights = np.array(peak_heights, dtype=np.float64)[sorted_indices]
        
        return peak_indices, peak_heights

    def _perform_mode_switch(self, target_mode: SignalMode) -> None:
        """
        Perform actual mode switch operation
        
        Args:
            target_mode: Target signal mode
            
        Raises:
            AssertionError: If target mode is invalid
        """
        # Validate input
        assert isinstance(target_mode, SignalMode), \
            f"Target mode must be SignalMode, got {type(target_mode)}"
        
        # Get configuration
        config = self.configs[target_mode]
        
        # Apply mode-specific settings
        logger.info(f"Switching to {target_mode.name} mode")
        logger.info(f"Setting sample rate to {config.sample_rate:.2e} Hz")
        logger.info(f"Setting bandwidth to {config.bandwidth:.2e} Hz")
        logger.info(f"Setting {config.equalization_taps} equalization taps")
        
        # In a real implementation, this would include hardware settings
        # For now, we just log the changes
        # ToDo: Implement hardware control


def create_mode_controller() -> ModeController:
    """
    Create and configure mode controller
    
    Returns:
        Configured mode controller
    """
    # Create controller
    controller = ModeController(default_mode=SignalMode.NRZ)
    
    # Register NRZ configuration
    nrz_config = ModeConfig(
        mode=SignalMode.NRZ,
        sample_rate=20e9,  # 20 GSa/s
        bandwidth=10e9,    # 10 GHz
        eye_height_threshold=0.5,
        equalization_taps=3
    )
    controller.register_mode_config(nrz_config)
    
    # Register PAM4 configuration
    pam4_config = ModeConfig(
        mode=SignalMode.PAM4,
        sample_rate=40e9,  # 40 GSa/s
        bandwidth=20e9,    # 20 GHz
        eye_height_threshold=0.4,
        equalization_taps=5
    )
    controller.register_mode_config(pam4_config)
    
    return controller
