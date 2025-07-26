"""
Instrument Control Module v1.4.0

This module contains functions and classes for controlling lab instruments
across multiple high-speed SerDes protocols.

Key Components:
- Universal instrument controller
- Protocol-specific scope controllers
- Mock instrument support for testing
- USB4/Thunderbolt 4 specialized controllers
- Power management and synchronization
"""

from .controller import InstrumentController
from .mock_controller import MockInstrumentController
from .mode_switcher import ModeSwitcher
from .pcie_analyzer import PCIeAnalyzer

__version__ = "1.4.0"

__all__ = [
    "InstrumentController",
    "MockInstrumentController",
    "ModeSwitcher",
    "PCIeAnalyzer",
]
