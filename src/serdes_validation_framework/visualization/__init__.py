"""
SerDes Validation Framework Visualization Module

This module provides comprehensive visualization capabilities for all supported protocols
including USB4, PCIe, and 224G Ethernet validation results.

Features:
- Eye diagram visualization
- Signal quality plots
- Protocol-specific dashboards
- Interactive visualizations
- Multi-protocol comparison charts
- Performance trend analysis
- Compliance test result visualization
"""

try:
    from .base_visualizer import BaseVisualizer
    from .comparison import ProtocolComparison
    from .dashboard import ValidationDashboard
    from .eye_diagram import EyeDiagramVisualizer
    from .protocol_visualizers import EthernetVisualizer, PCIeVisualizer, USB4Visualizer

    VISUALIZATION_CORE_AVAILABLE = True
except ImportError:
    VISUALIZATION_CORE_AVAILABLE = False

__all__ = []

if VISUALIZATION_CORE_AVAILABLE:
    __all__.extend(
        [
            "BaseVisualizer",
            "EyeDiagramVisualizer",
            "USB4Visualizer",
            "PCIeVisualizer",
            "EthernetVisualizer",
            "ValidationDashboard",
            "ProtocolComparison",
        ]
    )

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
