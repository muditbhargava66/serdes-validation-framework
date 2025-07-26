"""
Data Analysis Module v1.4.0

This module contains functions and classes for analyzing collected data from
high-speed SerDes protocols including PCIe 6.0, 224G Ethernet, and USB4/Thunderbolt 4.

Key Components:
- Eye diagram analysis with advanced jitter decomposition
- NRZ and PAM4 signal analysis
- Statistical signal characterization
- Multi-protocol data analysis support
"""

from .analyzer import DataAnalyzer
from .eye_diagram import AdvancedEyeAnalyzer as EyeDiagramAnalyzer
from .eye_diagram import EyeDiagramResult as EyeResults
from .nrz_analyzer import NRZAnalyzer
from .pam4_analyzer import PAM4Analyzer

__version__ = "1.4.0"

__all__ = [
    "DataAnalyzer",
    "EyeDiagramAnalyzer",
    "EyeResults",
    "NRZAnalyzer",
    "PAM4Analyzer",
]
