# src/serdes_validation_framework/protocols/ethernet_224g/constants.py

from typing import Dict, List, Tuple, Final
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass(frozen=True)
class PAM4Specs:
    """PAM4 signaling specifications"""
    SYMBOL_RATE: float = 112e9        # 112 GBaud
    UI_PERIOD: float = 8.9e-12        # Unit interval (seconds)
    VOLTAGE_LEVELS: Tuple[float, ...] = (-3.0, -1.0, 1.0, 3.0)
    MIN_LEVEL_SEPARATION: float = 0.4  # Minimum voltage between levels

@dataclass(frozen=True)
class JitterSpecs:
    """Jitter specifications"""
    MAX_TJ: float = 0.3e-12          # Maximum total jitter (seconds)
    MAX_RJ: float = 0.15e-12         # Maximum random jitter (seconds)
    MAX_DJ: float = 0.15e-12         # Maximum deterministic jitter (seconds)
    COMPLIANCE_BER: float = 1e-15     # Target bit error rate

@dataclass(frozen=True)
class EyeSpecs:
    """Eye diagram specifications"""
    MIN_HEIGHT: float = 0.2           # Minimum eye height (normalized)
    MIN_WIDTH: float = 0.4            # Minimum eye width (normalized)
    MAX_EVM_RMS: float = 5.0         # Maximum RMS EVM percentage
    MAX_EVM_PEAK: float = 10.0       # Maximum peak EVM percentage

@dataclass(frozen=True)
class TrainingSpecs:
    """Link training specifications"""
    MAX_TRAINING_TIME: float = 10.0   # Maximum training time (seconds)
    MIN_SNR: float = 20.0            # Minimum SNR (dB)
    ADAPTATION_THRESHOLD: float = 0.1  # Convergence threshold
    MAX_TAP_VALUE: float = 1.0       # Maximum equalizer tap value

# Protocol constants
ETHERNET_224G_SPECS: Final[Dict[str, object]] = {
    'pam4': PAM4Specs(),
    'jitter': JitterSpecs(),
    'eye': EyeSpecs(),
    'training': TrainingSpecs()
}

# Training patterns
TRAINING_PATTERNS: Final[Dict[str, npt.NDArray[np.float64]]] = {
    'preset': np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float64),
    'adapt': np.array([3.0, 1.0, -1.0, -3.0, 1.0, -1.0], dtype=np.float64),
    'verify': np.array([3.0, -3.0, 1.0, -1.0, -3.0, 3.0], dtype=np.float64)
}

# Compliance test patterns
COMPLIANCE_PATTERNS: Final[Dict[str, str]] = {
    'jitter': 'PRBS31',
    'eye': 'PRBS13',
    'levels': 'PRBS7',
    'isi': 'SQWV'
}

def validate_pam4_levels(levels: npt.NDArray[np.float64]) -> bool:
    """
    Validate PAM4 voltage levels against specifications
    
    Args:
        levels: Array of measured voltage levels
        
    Returns:
        True if levels meet specifications
    """
    assert isinstance(levels, np.ndarray), "Levels must be a numpy array"
    assert np.issubdtype(levels.dtype, np.floating), "Levels must be floating-point"
    assert len(levels) == 4, "Must have exactly 4 PAM4 levels"
    
    specs = ETHERNET_224G_SPECS['pam4']
    level_gaps = np.diff(sorted(levels))
    return all(gap >= specs.MIN_LEVEL_SEPARATION for gap in level_gaps)

def calculate_ui_parameters(
    symbol_rate: float = PAM4Specs.SYMBOL_RATE
) -> Tuple[float, float]:
    """
    Calculate unit interval parameters
    
    Args:
        symbol_rate: Symbol rate in Hz
        
    Returns:
        Tuple of (UI period, UI width tolerance)
    """
    assert isinstance(symbol_rate, float), "Symbol rate must be a float"
    assert symbol_rate > 0, "Symbol rate must be positive"
    
    ui_period = 1.0 / symbol_rate
    ui_tolerance = ui_period * 0.1  # 10% tolerance
    return float(ui_period), float(ui_tolerance)