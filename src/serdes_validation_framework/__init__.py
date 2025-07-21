"""
SerDes Validation Framework v1.3.0

This package provides comprehensive tools for validating high-speed SerDes protocols,
with particular focus on PCIe 6.0 and 224G Ethernet standards.

Key Features:
- PCIe 6.0 validation (64 GT/s)
- NRZ/PAM4 dual-mode support
- Advanced link training analysis
- Enhanced equalization algorithms
- Comprehensive compliance testing
- Advanced eye diagram analysis
- Multi-phase test sequences
- Intelligent signal analysis
"""

# Package version
__version__ = "1.3.0"

# Package version
__version__ = "1.3.0"

# Note: We're not importing classes directly at the top level to avoid circular imports
# Users should import from specific modules as needed:
# from serdes_validation_framework.test_sequence import PCIeTestSequence
# from serdes_validation_framework.protocols.pcie import SignalMode

__all__ = [
    '__version__',
]
