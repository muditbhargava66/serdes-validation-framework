"""
PCIe Dual Mode Support Module v1.4.0

This module provides support for PCIe dual-mode operation, supporting both
NRZ and PAM4 signal modes with robust type checking and validation.

Features:
- Seamless NRZ/PAM4 mode switching
- Mode-specific training algorithms
- Real-time mode transition capabilities
- Adaptive configuration per signaling mode
"""

from .mode_control import ModeController as DualModeController
from .mode_control import SwitchStatus as ModeTransition
from .nrz_training import NRZTrainer as NRZTrainingController
from .pam4_training import PAM4Trainer as PAM4TrainingController

__version__ = "1.4.0"

__all__ = [
    "DualModeController",
    "ModeTransition",
    "NRZTrainingController",
    "PAM4TrainingController",
    "__version__",
]
