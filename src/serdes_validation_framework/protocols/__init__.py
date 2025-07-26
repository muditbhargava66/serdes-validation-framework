"""
Protocol Support Module

This module provides protocol-specific validation capabilities.
"""

# Import available protocols
try:
    from . import usb4

    USB4_AVAILABLE = True
except ImportError:
    USB4_AVAILABLE = False

try:
    from . import pcie

    PCIE_AVAILABLE = True
except ImportError:
    PCIE_AVAILABLE = False

try:
    from . import ethernet_224g

    ETHERNET_AVAILABLE = True
except ImportError:
    ETHERNET_AVAILABLE = False

__all__ = []

if USB4_AVAILABLE:
    __all__.append("usb4")
if PCIE_AVAILABLE:
    __all__.append("pcie")
if ETHERNET_AVAILABLE:
    __all__.append("ethernet_224g")

# Create supported protocols dictionary
SUPPORTED_PROTOCOLS = {}
if USB4_AVAILABLE:
    SUPPORTED_PROTOCOLS["USB4"] = {"module": usb4, "available": True}
if PCIE_AVAILABLE:
    SUPPORTED_PROTOCOLS["PCIE"] = {"module": pcie, "available": True}
if ETHERNET_AVAILABLE:
    SUPPORTED_PROTOCOLS["ETHERNET_224G"] = {"module": ethernet_224g, "available": True}

# Add availability flags for testing
__all__.extend(["USB4_AVAILABLE", "PCIE_AVAILABLE", "ETHERNET_AVAILABLE", "SUPPORTED_PROTOCOLS"])
