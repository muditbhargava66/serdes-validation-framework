"""
USB4/Thunderbolt 4 Protocol Support

This module provides comprehensive USB4 and Thunderbolt 4 validation capabilities.
"""

# Import core tunneling classes
try:
    from .tunneling import (
        BandwidthAllocationMode,
        BandwidthMonitor,
        DisplayPortTunnelResults,
        DisplayPortTunnelValidator,
        MultiProtocolBandwidthManager,
        PCIeTLPHeader,
        PCIeTunnelResults,
        PCIeTunnelValidator,
        TunnelConfig,
        TunnelResults,
        TunnelState,
        USB32TunnelResults,
        USB32TunnelValidator,
    )

    TUNNELING_AVAILABLE = True
except ImportError:
    TUNNELING_AVAILABLE = False

# Import other USB4 modules conditionally
try:
    from .base import USB4Config, USB4SignalMode

    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False

try:
    from .constants import USB4_PROTOCOL_SPECS, USB4LinkState, USB4TunnelingMode, USB4TunnelingSpecs

    CONSTANTS_AVAILABLE = True
except ImportError:
    CONSTANTS_AVAILABLE = False

# Check for visualization support
try:
    from .visualization import USB4Visualizer

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Import additional USB4 components
try:
    from .jitter_analyzer import USB4JitterAnalyzer
    from .power_management import USB4PowerManager

    ADDITIONAL_COMPONENTS_AVAILABLE = True
except ImportError:
    ADDITIONAL_COMPONENTS_AVAILABLE = False

# Build exports list
__all__ = []

if TUNNELING_AVAILABLE:
    __all__.extend(
        [
            "PCIeTunnelValidator",
            "DisplayPortTunnelValidator",
            "USB32TunnelValidator",
            "MultiProtocolBandwidthManager",
            "PCIeTLPHeader",
            "TunnelConfig",
            "TunnelState",
            "BandwidthAllocationMode",
            "TunnelResults",
            "PCIeTunnelResults",
            "DisplayPortTunnelResults",
            "USB32TunnelResults",
            "BandwidthMonitor",
        ]
    )

if BASE_AVAILABLE:
    __all__.extend(["USB4Config", "USB4SignalMode"])

if CONSTANTS_AVAILABLE:
    __all__.extend(["USB4TunnelingMode", "USB4TunnelingSpecs"])

if VISUALIZATION_AVAILABLE:
    __all__.append("USB4Visualizer")

# Add availability flags for testing
__all__.extend(["TUNNELING_AVAILABLE", "BASE_AVAILABLE", "CONSTANTS_AVAILABLE", "VISUALIZATION_AVAILABLE"])
