"""
Data Collection Module v1.4.0

This module contains functions and classes for collecting data from lab instruments
supporting PCIe 6.0, 224G Ethernet, and USB4/Thunderbolt 4 protocols.

Key Components:
- Multi-protocol data collection
- Instrument abstraction layer
- Real-time data streaming
- Mock data generation for testing
"""

from .data_collector import DataCollector

__version__ = "1.4.0"

__all__ = [
    "DataCollector",
]
