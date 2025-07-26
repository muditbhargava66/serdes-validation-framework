"""
SerDes Validation Framework v1.4.0

This package provides comprehensive tools for validating high-speed SerDes protocols,
with support for PCIe 6.0, 224G Ethernet, and USB4/Thunderbolt 4 standards.

Key Features:
- PCIe 6.0 validation (64 GT/s)
- 224G Ethernet validation (112 GBaud PAM4)
- USB4/Thunderbolt 4 validation (40 Gbps)
- NRZ/PAM4 dual-mode support
- Advanced link training analysis
- Enhanced equalization algorithms
- Multi-protocol tunneling validation
- Comprehensive compliance testing
- Advanced eye diagram analysis
- Multi-phase test sequences
- Intelligent signal analysis
- Automatic protocol detection
- Unified validation framework

Quick Start:
    from serdes_validation_framework import create_validation_framework, auto_validate_signal

    # Automatic validation
    results = auto_validate_signal(signal_data, sample_rate=200e9, voltage_range=0.8)

    # Manual framework usage
    framework = create_validation_framework()
    protocol = framework.detect_protocol(signal_data, sample_rate=200e9, voltage_range=0.8)
"""

# Package version
__version__ = "1.4.0"

# Import main framework components for easy access with error handling
try:
    from .framework_integration import (
        ValidationFramework,
        auto_validate_signal,
        create_usb4_test_sequence,
        create_validation_framework,
        detect_signal_protocol,
    )

    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False

try:
    from .protocol_detector import ProtocolType

    PROTOCOL_DETECTOR_AVAILABLE = True
except ImportError:
    PROTOCOL_DETECTOR_AVAILABLE = False

# Conditional exports based on availability
__all__ = ["__version__"]

if FRAMEWORK_AVAILABLE:
    __all__.extend(
        [
            "ValidationFramework",
            "create_validation_framework",
            "auto_validate_signal",
            "detect_signal_protocol",
            "create_usb4_test_sequence",
        ]
    )

if PROTOCOL_DETECTOR_AVAILABLE:
    __all__.append("ProtocolType")

# Add availability flags for testing
__all__.extend(["FRAMEWORK_AVAILABLE", "PROTOCOL_DETECTOR_AVAILABLE"])
