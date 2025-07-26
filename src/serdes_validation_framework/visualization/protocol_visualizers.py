"""
Protocol-Specific Visualizers

Enhanced visualizers for USB4, PCIe, and Ethernet protocols.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base_visualizer import BaseVisualizer
from .eye_diagram import EyeDiagramVisualizer

logger = logging.getLogger(__name__)


class USB4Visualizer(BaseVisualizer):
    """Enhanced USB4 visualization with tunneling and power management support"""

    def __init__(self):
        """Initialize USB4 visualizer"""
        super().__init__("USB4")
        self.eye_visualizer = EyeDiagramVisualizer("USB4")

    def plot_eye_diagram(self, signal_data: Union[np.ndarray, Dict], **kwargs) -> Optional[str]:
        """Plot USB4 eye diagram with protocol-specific enhancements"""
        # Set USB4-specific defaults
        kwargs.setdefault("symbol_rate", 20e9)  # 20 Gbaud per lane
        kwargs.setdefault("samples_per_symbol", 10)
        kwargs.setdefault("title", "USB4 Eye Diagram (20 Gbaud)")

        return self.eye_visualizer.plot_eye_diagram(signal_data, **kwargs)

    def plot_signal_quality(self, metrics: Dict[str, Any], **kwargs) -> Optional[str]:
        """Plot USB4 signal quality metrics"""
        return self.eye_visualizer.plot_signal_quality(metrics, **kwargs)

    def plot_tunnel_bandwidth(self, bandwidth_data: Dict[str, Any], **kwargs) -> Optional[str]:
        """Plot USB4 tunnel bandwidth utilization"""
        if not self.matplotlib_available:
            logger.error("Matplotlib required for bandwidth plotting")
            return None

        try:
            fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(16, 6))

            # Plot 1: Current utilization
            tunnels = list(bandwidth_data.keys())
            utilization = [bandwidth_data[t].get("utilization", 0) for t in tunnels]
            max_bandwidth = [bandwidth_data[t].get("max_bandwidth", 100) for t in tunnels]

            bars1 = ax1.bar(tunnels, utilization)
            ax1.set_title("USB4 Tunnel Bandwidth Utilization")
            ax1.set_xlabel("Tunnel Type")
            ax1.set_ylabel("Utilization (%)")
            ax1.set_ylim(0, 100)

            # Color bars based on utilization
            for bar, util in zip(bars1, utilization):
                if util > 80:
                    bar.set_color("red")
                elif util > 60:
                    bar.set_color("orange")
                else:
                    bar.set_color("green")

            # Add utilization labels
            for bar, util in zip(bars1, utilization):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{util:.1f}%", ha="center", va="bottom")

            # Plot 2: Bandwidth allocation
            labels = tunnels
            sizes = [bandwidth_data[t].get("allocated_bandwidth", 1) for t in tunnels]
            colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"][: len(labels)]

            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax2.set_title("USB4 Bandwidth Allocation")

            # Enhance pie chart
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")

            self.plt.tight_layout()

            # Save plot
            filename = kwargs.get("filename", "usb4_tunnel_bandwidth.png")
            filepath = self._ensure_output_dir(filename)
            self.plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.plt.close()

            logger.info(f"USB4 bandwidth plot saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"USB4 bandwidth plotting failed: {e}")
            return None

    def plot_power_states(self, power_data: Dict[str, Any], **kwargs) -> Optional[str]:
        """Plot USB4 power state transitions"""
        if not self.matplotlib_available:
            logger.error("Matplotlib required for power state plotting")
            return None

        try:
            fig, (ax1, ax2) = self.plt.subplots(2, 1, figsize=(12, 10))

            # Plot 1: Power state timeline
            if "timeline" in power_data:
                timeline = power_data["timeline"]
                times = timeline.get("times", [])
                states = timeline.get("states", [])

                # Map states to numeric values
                state_map = {"U0": 0, "U1": 1, "U2": 2, "U3": 3}
                numeric_states = [state_map.get(state, 0) for state in states]

                ax1.step(times, numeric_states, where="post", linewidth=2)
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Power State")
                ax1.set_title("USB4 Power State Timeline")
                ax1.set_yticks(list(state_map.values()))
                ax1.set_yticklabels(list(state_map.keys()))
                ax1.grid(True, alpha=0.3)

            # Plot 2: Power consumption by state
            if "consumption" in power_data:
                consumption = power_data["consumption"]
                states = list(consumption.keys())
                power_values = list(consumption.values())

                bars = ax2.bar(states, power_values)
                ax2.set_xlabel("Power State")
                ax2.set_ylabel("Power Consumption (W)")
                ax2.set_title("USB4 Power Consumption by State")

                # Color bars based on power consumption
                max_power = max(power_values) if power_values else 1
                for bar, power in zip(bars, power_values):
                    intensity = power / max_power
                    bar.set_color(self.plt.cm.Reds(intensity))

                # Add power labels
                for bar, power in zip(bars, power_values):
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2.0, height + height * 0.01, f"{power:.2f}W", ha="center", va="bottom"
                    )

            self.plt.tight_layout()

            # Save plot
            filename = kwargs.get("filename", "usb4_power_states.png")
            filepath = self._ensure_output_dir(filename)
            self.plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.plt.close()

            return str(filepath)

        except Exception as e:
            logger.error(f"USB4 power state plotting failed: {e}")
            return None

    def create_interactive_dashboard(self, test_results: Dict[str, Any], **kwargs) -> Optional[str]:
        """Create interactive USB4 dashboard"""
        if not self.plotly_available:
            logger.error("Plotly required for interactive dashboard")
            return None

        try:
            from plotly.subplots import make_subplots

            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("Test Results", "Signal Quality", "Tunnel Bandwidth", "Power States"),
                specs=[[{"type": "pie"}, {"type": "bar"}], [{"type": "bar"}, {"type": "scatter"}]],
            )

            # Plot 1: Test results pie chart
            if "test_results" in test_results:
                results = test_results["test_results"]
                passed = sum(1 for r in results.values() if r.get("status") == "PASS")
                failed = sum(1 for r in results.values() if r.get("status") == "FAIL")

                fig.add_trace(
                    self.go.Pie(labels=["Passed", "Failed"], values=[passed, failed], marker_colors=["green", "red"]),
                    row=1,
                    col=1,
                )

            # Plot 2: Signal quality metrics
            if "signal_quality" in test_results:
                metrics = test_results["signal_quality"]
                fig.add_trace(self.go.Bar(x=list(metrics.keys()), y=list(metrics.values()), marker_color="blue"), row=1, col=2)

            # Plot 3: Tunnel bandwidth
            if "tunnel_bandwidth" in test_results:
                bandwidth = test_results["tunnel_bandwidth"]
                fig.add_trace(
                    self.go.Bar(x=list(bandwidth.keys()), y=list(bandwidth.values()), marker_color="orange"), row=2, col=1
                )

            # Plot 4: Power states
            if "power_timeline" in test_results:
                timeline = test_results["power_timeline"]
                fig.add_trace(
                    self.go.Scatter(
                        x=timeline.get("times", []), y=timeline.get("states", []), mode="lines+markers", name="Power State"
                    ),
                    row=2,
                    col=2,
                )

            # Update layout
            fig.update_layout(title_text="USB4 Validation Dashboard", title_x=0.5, showlegend=False, height=800)

            # Save dashboard
            filename = kwargs.get("filename", "usb4_dashboard.html")
            filepath = self._ensure_output_dir(filename)
            fig.write_html(str(filepath))

            logger.info(f"USB4 dashboard saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"USB4 dashboard creation failed: {e}")
            return None


class PCIeVisualizer(BaseVisualizer):
    """Enhanced PCIe visualization with link training and equalization support"""

    def __init__(self):
        """Initialize PCIe visualizer"""
        super().__init__("PCIe")
        self.eye_visualizer = EyeDiagramVisualizer("PCIe")

    def plot_eye_diagram(self, signal_data: Union[np.ndarray, Dict], **kwargs) -> Optional[str]:
        """Plot PCIe eye diagram with protocol-specific enhancements"""
        # Set PCIe-specific defaults
        kwargs.setdefault("symbol_rate", 32e9)  # 32 GT/s for PCIe 6.0
        kwargs.setdefault("samples_per_symbol", 8)
        kwargs.setdefault("title", "PCIe 6.0 Eye Diagram (32 GT/s)")

        return self.eye_visualizer.plot_eye_diagram(signal_data, **kwargs)

    def plot_signal_quality(self, metrics: Dict[str, Any], **kwargs) -> Optional[str]:
        """Plot PCIe signal quality metrics"""
        return self.eye_visualizer.plot_signal_quality(metrics, **kwargs)

    def plot_link_training(self, training_data: Dict[str, Any], **kwargs) -> Optional[str]:
        """Plot PCIe link training progression"""
        if not self.matplotlib_available:
            logger.error("Matplotlib required for link training plotting")
            return None

        try:
            fig, (ax1, ax2) = self.plt.subplots(2, 1, figsize=(12, 10))

            # Plot 1: Training phases
            if "phases" in training_data:
                phases = training_data["phases"]
                phase_names = list(phases.keys())
                durations = [phases[name].get("duration", 0) for name in phase_names]

                bars = ax1.barh(phase_names, durations)
                ax1.set_xlabel("Duration (ms)")
                ax1.set_title("PCIe Link Training Phases")

                # Color bars based on success
                for bar, name in zip(bars, phase_names):
                    success = phases[name].get("success", True)
                    bar.set_color("green" if success else "red")

            # Plot 2: Equalization coefficients
            if "equalization" in training_data:
                eq_data = training_data["equalization"]
                coefficients = ["C-1", "C0", "C+1"]
                values = [eq_data.get(coef, 0) for coef in coefficients]

                ax2.bar(coefficients, values)
                ax2.set_ylabel("Coefficient Value")
                ax2.set_title("PCIe Equalization Coefficients")
                ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            self.plt.tight_layout()

            # Save plot
            filename = kwargs.get("filename", "pcie_link_training.png")
            filepath = self._ensure_output_dir(filename)
            self.plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.plt.close()

            return str(filepath)

        except Exception as e:
            logger.error(f"PCIe link training plotting failed: {e}")
            return None


class EthernetVisualizer(BaseVisualizer):
    """Enhanced 224G Ethernet visualization with PAM4 support"""

    def __init__(self):
        """Initialize Ethernet visualizer"""
        super().__init__("224G Ethernet")
        self.eye_visualizer = EyeDiagramVisualizer("224G Ethernet")

    def plot_eye_diagram(self, signal_data: Union[np.ndarray, Dict], **kwargs) -> Optional[str]:
        """Plot Ethernet eye diagram with PAM4 support"""
        # Set Ethernet-specific defaults
        kwargs.setdefault("symbol_rate", 112e9)  # 112 GBaud for 224G
        kwargs.setdefault("samples_per_symbol", 4)
        kwargs.setdefault("title", "224G Ethernet PAM4 Eye Diagram (112 GBaud)")

        return self.eye_visualizer.plot_eye_diagram(signal_data, **kwargs)

    def plot_signal_quality(self, metrics: Dict[str, Any], **kwargs) -> Optional[str]:
        """Plot Ethernet signal quality metrics"""
        return self.eye_visualizer.plot_signal_quality(metrics, **kwargs)

    def plot_pam4_levels(self, signal_data: np.ndarray, **kwargs) -> Optional[str]:
        """Plot PAM4 level distribution"""
        if not self.matplotlib_available:
            logger.error("Matplotlib required for PAM4 level plotting")
            return None

        try:
            fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(16, 6))

            # Plot 1: Signal histogram
            ax1.hist(signal_data, bins=50, alpha=0.7, density=True)
            ax1.set_xlabel("Amplitude (V)")
            ax1.set_ylabel("Probability Density")
            ax1.set_title("PAM4 Signal Amplitude Distribution")
            ax1.grid(True, alpha=0.3)

            # Identify PAM4 levels (simplified)
            levels = np.percentile(signal_data, [12.5, 37.5, 62.5, 87.5])
            colors = ["red", "orange", "green", "blue"]
            for level, color in zip(levels, colors):
                ax1.axvline(level, color=color, linestyle="--", alpha=0.8)

            # Plot 2: Level separation
            level_names = ["Level 0", "Level 1", "Level 2", "Level 3"]
            ax2.bar(level_names, levels, color=colors, alpha=0.7)
            ax2.set_ylabel("Amplitude (V)")
            ax2.set_title("PAM4 Level Values")
            ax2.tick_params(axis="x", rotation=45)

            # Add level values as text
            for i, level in enumerate(levels):
                ax2.text(i, level + 0.01, f"{level:.3f}V", ha="center", va="bottom", fontweight="bold")

            self.plt.tight_layout()

            # Save plot
            filename = kwargs.get("filename", "ethernet_pam4_levels.png")
            filepath = self._ensure_output_dir(filename)
            self.plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.plt.close()

            return str(filepath)

        except Exception as e:
            logger.error(f"PAM4 level plotting failed: {e}")
            return None
