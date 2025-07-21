"""
PCIe Module

This module provides comprehensive support for PCIe 6.0 protocol validation,
including constants, compliance testing, and signal analysis.
"""

from .constants import SignalMode
from .compliance import ComplianceType, ComplianceLimit
from .link_training import TrainingPhase
from .equalization import EqualizationType

__all__ = [
    'SignalMode',
    'ComplianceType',
    'ComplianceLimit',
    'TrainingPhase',
    'EqualizationType',
]
