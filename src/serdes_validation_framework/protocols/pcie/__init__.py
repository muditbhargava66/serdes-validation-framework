"""
PCIe Module v1.4.0

This module provides comprehensive support for PCIe 6.0 protocol validation,
including constants, compliance testing, signal analysis, and dual-mode NRZ/PAM4 support.

Features:
- PCIe 6.0 specification compliance (64 GT/s)
- NRZ and PAM4 dual-mode support
- Advanced link training analysis
- Multi-lane validation (1-16 lanes)
- Enhanced equalization algorithms
- Comprehensive compliance testing
"""

from .compliance import ComplianceLimit, ComplianceType
from .constants import SignalMode
from .equalization import EqualizationType
from .link_training import TrainingPhase

__version__ = "1.4.0"

__all__ = [
    "SignalMode",
    "ComplianceType",
    "ComplianceLimit",
    "TrainingPhase",
    "EqualizationType",
    "__version__",
]
