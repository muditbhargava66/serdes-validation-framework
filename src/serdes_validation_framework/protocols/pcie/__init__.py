"""
PCIe Module

This module provides comprehensive support for PCIe 6.0 protocol validation,
including constants, compliance testing, and signal analysis.
"""

from .compliance import ComplianceLimit, ComplianceType
from .constants import SignalMode
from .equalization import EqualizationType
from .link_training import TrainingPhase

__all__ = [
    'SignalMode',
    'ComplianceType',
    'ComplianceLimit',
    'TrainingPhase',
    'EqualizationType',
]
