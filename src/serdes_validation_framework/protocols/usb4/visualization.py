"""
USB4 Visualization Support

This module provides enhanced visualization capabilities for USB4 validation results.
Uses the comprehensive visualization framework for advanced features.

Note: This is a compatibility wrapper. For full functionality, use:
from serdes_validation_framework.visualization import USB4Visualizer
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class USB4Visualizer:
    """USB4 data visualization class - Enhanced with comprehensive visualization support"""

    def __init__(self):
        """Initialize USB4 visualizer with enhanced capabilities"""
        self.enhanced_visualizer = None

        # Try to use the enhanced visualizer
        try:
            from ...visualization.protocol_visualizers import USB4Visualizer as EnhancedUSB4Visualizer

            self.enhanced_visualizer = EnhancedUSB4Visualizer()
            logger.info("Enhanced USB4 visualizer loaded successfully")
        except ImportError:
            logger.warning("Enhanced visualization not available, using basic implementation")

        # Fallback to basic implementation
        self.matplotlib_available = False
        self.plotly_available = False

        try:
            import matplotlib.pyplot as plt

            self.matplotlib_available = True
            self.plt = plt
        except ImportError:
            logger.warning("Matplotlib not available - some visualizations disabled")

        try:
            import plotly.express as px
            import plotly.graph_objects as go

            self.plotly_available = True
            self.go = go
            self.px = px
        except ImportError:
            logger.warning("Plotly not available - interactive visualizations disabled")

    def plot_eye_diagram(self, data: Union[np.ndarray, Dict], **kwargs) -> Optional[str]:
        """Plot eye diagram for USB4 signal - Enhanced version"""
        # Use enhanced visualizer if available
        if self.enhanced_visualizer:
            return self.enhanced_visualizer.plot_eye_diagram(data, **kwargs)

        # Fallback to basic implementation
        if not self.matplotlib_available:
            logger.error("Matplotlib required for eye diagram plotting")
            return None

        try:
            # Handle different data formats
            if isinstance(data, dict):
                # Multi-lane data - use first lane
                lane_data = next(iter(data.values()))
                if isinstance(lane_data, dict) and "voltage" in lane_data:
                    plot_data = lane_data["voltage"]
                else:
                    plot_data = lane_data
            else:
                plot_data = data

            fig, ax = self.plt.subplots(figsize=(10, 6))
            ax.plot(plot_data[:1000])  # Plot first 1000 samples
            ax.set_title("USB4 Eye Diagram")
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)

            filename = kwargs.get("filename", "usb4_eye_diagram.png")
            self.plt.savefig(filename, dpi=300, bbox_inches="tight")
            self.plt.close()

            return filename
        except Exception as e:
            logger.error(f"Eye diagram plotting failed: {e}")
            return None

    def plot_tunnel_bandwidth(self, bandwidth_data: Dict[str, Any], **kwargs) -> Optional[str]:
        """Plot tunnel bandwidth utilization - Enhanced version"""
        # Use enhanced visualizer if available
        if self.enhanced_visualizer:
            return self.enhanced_visualizer.plot_tunnel_bandwidth(bandwidth_data, **kwargs)

        # Fallback to basic implementation
        if not self.matplotlib_available:
            logger.error("Matplotlib required for bandwidth plotting")
            return None

        try:
            fig, ax = self.plt.subplots(figsize=(12, 8))

            # Extract bandwidth data
            tunnels = list(bandwidth_data.keys())
            utilization = [bandwidth_data[t].get("utilization", 0) for t in tunnels]

            bars = ax.bar(tunnels, utilization)
            ax.set_title("USB4 Tunnel Bandwidth Utilization")
            ax.set_xlabel("Tunnel Type")
            ax.set_ylabel("Utilization (%)")
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)

            # Color bars based on utilization
            for bar, util in zip(bars, utilization):
                if util > 80:
                    bar.set_color("red")
                elif util > 60:
                    bar.set_color("orange")
                else:
                    bar.set_color("green")

            filename = kwargs.get("filename", "usb4_bandwidth.png")
            self.plt.savefig(filename, dpi=300, bbox_inches="tight")
            self.plt.close()

            return filename
        except Exception as e:
            logger.error(f"Bandwidth plotting failed: {e}")
            return None

    def create_interactive_dashboard(self, test_results: Dict[str, Any], **kwargs) -> Optional[str]:
        """Create interactive dashboard for test results - Enhanced version"""
        # Use enhanced visualizer if available
        if self.enhanced_visualizer:
            return self.enhanced_visualizer.create_interactive_dashboard(test_results, **kwargs)

        # Fallback to basic implementation
        if not self.plotly_available:
            logger.error("Plotly required for interactive dashboard")
            return None

        try:
            # Basic dashboard implementation
            fig = self.go.Figure()

            # Add basic test results visualization
            if isinstance(test_results, dict) and test_results:
                # Try to extract meaningful data
                if "test_results" in test_results:
                    results = test_results["test_results"]
                    passed = sum(1 for r in results.values() if r.get("status") == "PASS")
                    failed = sum(1 for r in results.values() if r.get("status") == "FAIL")

                    fig.add_trace(
                        self.go.Pie(labels=["Passed", "Failed"], values=[passed, failed], marker_colors=["green", "red"])
                    )
                else:
                    # Simple scatter plot
                    fig.add_trace(
                        self.go.Scatter(
                            x=list(range(len(test_results))), y=[1] * len(test_results), mode="markers", name="Test Results"
                        )
                    )

            fig.update_layout(title="USB4 Validation Dashboard", xaxis_title="Test Index", yaxis_title="Status")

            filename = kwargs.get("filename", "usb4_dashboard.html")
            fig.write_html(filename)

            return filename
        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            return None

    def plot_signal_quality(self, metrics: Dict[str, Any], **kwargs) -> Optional[str]:
        """Plot signal quality metrics"""
        # Use enhanced visualizer if available
        if self.enhanced_visualizer:
            return self.enhanced_visualizer.plot_signal_quality(metrics, **kwargs)

        # Basic implementation
        if not self.matplotlib_available:
            logger.error("Matplotlib required for signal quality plotting")
            return None

        try:
            fig, ax = self.plt.subplots(figsize=(10, 6))

            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())

            bars = ax.bar(metric_names, metric_values)
            ax.set_title("USB4 Signal Quality Metrics")
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)

            # Color bars based on values
            for bar, value in zip(bars, metric_values):
                if isinstance(value, (int, float)):
                    if value > 0.8:
                        bar.set_color("green")
                    elif value > 0.6:
                        bar.set_color("orange")
                    else:
                        bar.set_color("red")

            self.plt.tight_layout()

            filename = kwargs.get("filename", "usb4_signal_quality.png")
            self.plt.savefig(filename, dpi=300, bbox_inches="tight")
            self.plt.close()

            return filename

        except Exception as e:
            logger.error(f"Signal quality plotting failed: {e}")
            return None
