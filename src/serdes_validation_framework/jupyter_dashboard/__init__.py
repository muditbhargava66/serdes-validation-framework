"""
Jupyter Dashboard Module

This module provides Jupyter Notebook-based interactive dashboards for
SerDes validation visualization, including eye diagrams, waveforms,
and pass/fail annotations.
"""

from .eye_diagram_dashboard import DashboardConfig, EyeDiagramDashboard, check_dashboard_dependencies, create_dashboard
from .interactive_plots import InteractivePlotter, check_interactive_dependencies
from .waveform_analyzer import WaveformAnalyzer

__all__ = [
    'EyeDiagramDashboard',
    'create_dashboard',
    'DashboardConfig',
    'WaveformAnalyzer', 
    'InteractivePlotter',
    'check_dashboard_dependencies',
    'check_interactive_dependencies'
]
