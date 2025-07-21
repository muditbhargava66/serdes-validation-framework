# src/serdes_validation_framework/protocols/pcie/constants.py

"""
PCIe Protocol Constants

This module defines constants and specifications for PCI Express 6.0 validation,
including NRZ/PAM4 dual-mode operation.

Features:
- Protocol specifications
- Operating modes
- Compliance limits 
- Training parameters
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Final, Tuple

import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalMode(Enum):
    """PCIe signal modes"""
    NRZ = auto()
    PAM4 = auto()

class LinkState(Enum):
    """PCIe link states"""
    DETECT = auto()
    POLLING = auto()
    CONFIGURATION = auto()
    L0 = auto()  # Normal operation
    L0s = auto() # Low power standby
    L1 = auto()  # Low power idle
    L2 = auto()  # Deep sleep
    RECOVERY = auto()
    LOOPBACK = auto()
    DISABLED = auto()
    HOT_RESET = auto()

@dataclass(frozen=True)
class PCIeSpecs:
    """PCIe 6.0 specifications"""
    # Link parameters
    GEN6_RATE: float = 64.0e9        # 64 GT/s
    MAX_LANES: int = 16              # Maximum lanes
    MIN_LANES: int = 1               # Minimum lanes
    
    # Timing parameters
    UI_PERIOD: float = 15.625e-12    # Unit interval (seconds)
    TX_UI_PPM: float = 300.0        # Transmit UI tolerance (ppm)
    RX_UI_PPM: float = 600.0        # Receive UI tolerance (ppm)
    
    def __post_init__(self) -> None:
        """Validate specifications"""
        # Validate link parameters
        assert isinstance(self.GEN6_RATE, float), \
            "GEN6_RATE must be float"
        assert isinstance(self.MAX_LANES, int), \
            "MAX_LANES must be integer"
        assert isinstance(self.MIN_LANES, int), \
            "MIN_LANES must be integer"
            
        # Validate timing parameters
        assert isinstance(self.UI_PERIOD, float), \
            "UI_PERIOD must be float"
        assert isinstance(self.TX_UI_PPM, float), \
            "TX_UI_PPM must be float"
        assert isinstance(self.RX_UI_PPM, float), \
            "RX_UI_PPM must be float"
            
        # Validate values
        assert self.GEN6_RATE > 0, "GEN6_RATE must be positive"
        assert self.MAX_LANES > 0, "MAX_LANES must be positive"
        assert self.MIN_LANES > 0, "MIN_LANES must be positive"
        assert self.UI_PERIOD > 0, "UI_PERIOD must be positive"
        assert self.TX_UI_PPM > 0, "TX_UI_PPM must be positive"
        assert self.RX_UI_PPM > 0, "RX_UI_PPM must be positive"

@dataclass(frozen=True)
class NRZSpecs:
    """NRZ mode specifications"""
    # Level parameters
    NOMINAL_AMPLITUDE: float = 0.8   # Nominal amplitude (V)
    MIN_AMPLITUDE: float = 0.6      # Minimum amplitude (V)
    MAX_AMPLITUDE: float = 1.0      # Maximum amplitude (V)
    
    # Timing parameters
    MIN_EYE_WIDTH: float = 0.4     # Minimum eye width (UI)
    MIN_EYE_HEIGHT: float = 0.2    # Minimum eye height (V)
    MAX_JITTER: float = 0.3e-12    # Maximum total jitter (s)
    
    def __post_init__(self) -> None:
        """Validate specifications"""
        # Validate level parameters
        assert isinstance(self.NOMINAL_AMPLITUDE, float), \
            "NOMINAL_AMPLITUDE must be float"
        assert isinstance(self.MIN_AMPLITUDE, float), \
            "MIN_AMPLITUDE must be float"
        assert isinstance(self.MAX_AMPLITUDE, float), \
            "MAX_AMPLITUDE must be float"
            
        # Validate timing parameters
        assert isinstance(self.MIN_EYE_WIDTH, float), \
            "MIN_EYE_WIDTH must be float"
        assert isinstance(self.MIN_EYE_HEIGHT, float), \
            "MIN_EYE_HEIGHT must be float"
        assert isinstance(self.MAX_JITTER, float), \
            "MAX_JITTER must be float"
            
        # Validate values
        assert self.MIN_AMPLITUDE < self.NOMINAL_AMPLITUDE < self.MAX_AMPLITUDE, \
            "Invalid amplitude range"
        assert 0 < self.MIN_EYE_WIDTH <= 1.0, \
            "MIN_EYE_WIDTH must be between 0 and 1"
        assert self.MIN_EYE_HEIGHT > 0, \
            "MIN_EYE_HEIGHT must be positive"
        assert self.MAX_JITTER > 0, \
            "MAX_JITTER must be positive"

@dataclass(frozen=True)
class PAM4Specs:
    """PAM4 mode specifications"""
    # Level parameters
    LEVEL_AMPLITUDES: Tuple[float, float, float, float] = \
        (-0.9, -0.3, 0.3, 0.9)  # PAM4 levels (V)
    MIN_SEPARATION: float = 0.4  # Minimum level separation
    MAX_RMS_EVM: float = 5.0    # Maximum RMS EVM (%)
    
    # Timing parameters
    MIN_EYE_WIDTH: float = 0.4  # Minimum eye width (UI)
    MIN_EYE_HEIGHT: float = 0.2 # Minimum eye height (V)
    MAX_JITTER: float = 0.3e-12 # Maximum total jitter (s)
    
    def __post_init__(self) -> None:
        """Validate specifications"""
        # Type validation
        assert isinstance(self.LEVEL_AMPLITUDES, tuple), \
            "LEVEL_AMPLITUDES must be tuple"
        assert all(isinstance(x, float) for x in self.LEVEL_AMPLITUDES), \
            "LEVEL_AMPLITUDES must be floats"
        assert len(self.LEVEL_AMPLITUDES) == 4, \
            "LEVEL_AMPLITUDES must have 4 levels"
            
        assert isinstance(self.MIN_SEPARATION, float), \
            "MIN_SEPARATION must be float"
        assert isinstance(self.MAX_RMS_EVM, float), \
            "MAX_RMS_EVM must be float"
        assert isinstance(self.MIN_EYE_WIDTH, float), \
            "MIN_EYE_WIDTH must be float"
        assert isinstance(self.MIN_EYE_HEIGHT, float), \
            "MIN_EYE_HEIGHT must be float"
        assert isinstance(self.MAX_JITTER, float), \
            "MAX_JITTER must be float"
            
        # Value validation
        assert all(x1 < x2 for x1, x2 in zip(self.LEVEL_AMPLITUDES[:-1], 
                                           self.LEVEL_AMPLITUDES[1:], strict=False)), \
            "LEVEL_AMPLITUDES must be monotonically increasing"
        assert self.MIN_SEPARATION > 0, \
            "MIN_SEPARATION must be positive"
        assert self.MAX_RMS_EVM > 0, \
            "MAX_RMS_EVM must be positive"
        assert 0 < self.MIN_EYE_WIDTH <= 1.0, \
            "MIN_EYE_WIDTH must be between 0 and 1"
        assert self.MIN_EYE_HEIGHT > 0, \
            "MIN_EYE_HEIGHT must be positive"
        assert self.MAX_JITTER > 0, \
            "MAX_JITTER must be positive"

@dataclass(frozen=True)
class TrainingSpecs:
    """Link training specifications"""
    # Timing parameters
    MAX_TRAINING_TIME: float = 10.0  # Maximum training time (s)
    MIN_SNR: float = 20.0           # Minimum SNR (dB)
    ADAPTATION_THRESHOLD: float = 0.1 # Convergence threshold
    
    # Equalizer parameters
    MAX_TX_TAPS: int = 5            # Maximum transmit taps
    MAX_RX_TAPS: int = 3            # Maximum receive taps
    MAX_TAP_VALUE: float = 1.0      # Maximum tap value
    
    def __post_init__(self) -> None:
        """Validate specifications"""
        # Type validation
        assert isinstance(self.MAX_TRAINING_TIME, float), \
            "MAX_TRAINING_TIME must be float"
        assert isinstance(self.MIN_SNR, float), \
            "MIN_SNR must be float"
        assert isinstance(self.ADAPTATION_THRESHOLD, float), \
            "ADAPTATION_THRESHOLD must be float"
            
        assert isinstance(self.MAX_TX_TAPS, int), \
            "MAX_TX_TAPS must be integer"
        assert isinstance(self.MAX_RX_TAPS, int), \
            "MAX_RX_TAPS must be integer"
        assert isinstance(self.MAX_TAP_VALUE, float), \
            "MAX_TAP_VALUE must be float"
            
        # Value validation
        assert self.MAX_TRAINING_TIME > 0, \
            "MAX_TRAINING_TIME must be positive"
        assert self.MIN_SNR > 0, \
            "MIN_SNR must be positive"
        assert 0 < self.ADAPTATION_THRESHOLD < 1, \
            "ADAPTATION_THRESHOLD must be between 0 and 1"
            
        assert self.MAX_TX_TAPS > 0, \
            "MAX_TX_TAPS must be positive"
        assert self.MAX_RX_TAPS > 0, \
            "MAX_RX_TAPS must be positive"
        assert self.MAX_TAP_VALUE > 0, \
            "MAX_TAP_VALUE must be positive"

# Protocol constants
PCIE_SPECS: Final[Dict[str, object]] = {
    'base': PCIeSpecs(),
    'nrz': NRZSpecs(),
    'pam4': PAM4Specs(),
    'training': TrainingSpecs()
}

# Training patterns
TRAINING_PATTERNS: Final[Dict[str, npt.NDArray[np.float64]]] = {
    'ts1': np.array([1, -1, 1, -1], dtype=np.float64),  # TS1 ordered set
    'ts2': np.array([1, 1, -1, -1], dtype=np.float64),  # TS2 ordered set
    'eios': np.array([1, 1, 1, -1], dtype=np.float64),  # EIOS pattern
    'eieos': np.array([-1, -1, -1, 1], dtype=np.float64) # EIEOS pattern
}

# Compliance patterns 
COMPLIANCE_PATTERNS: Final[Dict[str, str]] = {
    'loopback': 'CP0',     # Compliance pattern 0
    'jitter': 'CP1',       # Jitter measurement pattern
    'isi': 'CP2',          # ISI measurement pattern
    'levels': 'CP3',       # Level measurement pattern
    'transitions': 'CP9'   # Transition density pattern  
}

def validate_link_width(lanes: int) -> bool:
    """
    Validate PCIe link width
    
    Args:
        lanes: Number of lanes
        
    Returns:
        True if width is valid
        
    Raises:
        AssertionError: If validation fails
    """
    # Validate input
    assert isinstance(lanes, int), "Lanes must be integer"
    
    # Check range
    return PCIE_SPECS['base'].MIN_LANES <= lanes <= PCIE_SPECS['base'].MAX_LANES

def calculate_ui_parameters(
    signaling_rate: float = PCIeSpecs.GEN6_RATE
) -> Tuple[float, float]:
    """
    Calculate unit interval parameters
    
    Args:
        signaling_rate: Signaling rate in Hz
        
    Returns:
        Tuple of (UI period, UI tolerance)
        
    Raises:
        AssertionError: If validation fails
    """
    # Validate input
    assert isinstance(signaling_rate, float), \
        "Signaling rate must be float"
    assert signaling_rate > 0, \
        "Signaling rate must be positive"
    
    # Calculate parameters
    ui_period = 1.0 / signaling_rate
    ui_tolerance = ui_period * PCIE_SPECS['base'].TX_UI_PPM / 1e6
    
    return float(ui_period), float(ui_tolerance)

def check_compliance_limits(
    measurements: Dict[str, float],
    mode: SignalMode
) -> bool:
    """
    Check compliance against limits
    
    Args:
        measurements: Dictionary of measurements
        mode: Current signal mode
        
    Returns:
        True if all measurements pass
        
    Raises:
        AssertionError: If validation fails
    """
    # Validate inputs
    assert isinstance(measurements, dict), \
        "Measurements must be dictionary"
    assert all(isinstance(v, float) for v in measurements.values()), \
        "Measurement values must be floats"
    assert isinstance(mode, SignalMode), \
        "Mode must be SignalMode"
    
    # Get limits based on mode
    limits = PCIE_SPECS['nrz'] if mode == SignalMode.NRZ else PCIE_SPECS['pam4']
    
    # Check limits
    try:
        if 'eye_height' in measurements:
            if measurements['eye_height'] < limits.MIN_EYE_HEIGHT:
                return False
                
        if 'eye_width' in measurements:
            if measurements['eye_width'] < limits.MIN_EYE_WIDTH:
                return False
                
        if 'jitter' in measurements:
            if measurements['jitter'] > limits.MAX_JITTER:
                return False
                
        if mode == SignalMode.PAM4 and 'rms_evm' in measurements:
            if measurements['rms_evm'] > limits.MAX_RMS_EVM:
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        return False
