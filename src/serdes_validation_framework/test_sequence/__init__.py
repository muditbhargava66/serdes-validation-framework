"""
Test Sequence Module v1.4.0

This module contains functions and classes for creating and running comprehensive
test sequences across multiple high-speed SerDes protocols.

Key Components:
- PCIe 6.0 test sequences with NRZ/PAM4 support
- 224G Ethernet validation sequences
- USB4/Thunderbolt 4 comprehensive testing
- Multi-protocol test orchestration
- Automated compliance testing
"""

# Import key classes for easier access
from .eth_224g_sequence import Ethernet224GTestSequence
from .pcie_sequence import (
    LaneConfig,
    PCIeTestPhase,
    PCIeTestResult,
    PCIeTestSequence,
    PCIeTestSequenceConfig,
    create_multi_lane_pam4_test,
    create_single_lane_nrz_test,
)
from .sequencer import PCIeTestSequencer
from .usb4_sequence import (
    USB4LaneConfig,
    USB4PhaseResult,
    USB4SequenceResult,
    USB4TestPhase,
    USB4TestResult,
    USB4TestSequence,
    USB4TestSequenceConfig,
)

__version__ = "1.4.0"

# Define __all__ to control what's imported with wildcard imports
__all__ = [
    "PCIeTestSequencer",
    "PCIeTestPhase",
    "PCIeTestResult",
    "PCIeTestSequenceConfig",
    "PCIeTestSequence",
    "LaneConfig",
    "create_single_lane_nrz_test",
    "create_multi_lane_pam4_test",
    "Ethernet224GTestSequence",
    "USB4TestSequence",
    "USB4TestSequenceConfig",
    "USB4TestPhase",
    "USB4TestResult",
    "USB4LaneConfig",
    "USB4PhaseResult",
    "USB4SequenceResult",
]
