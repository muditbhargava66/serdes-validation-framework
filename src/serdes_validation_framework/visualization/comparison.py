"""
Protocol Comparison Visualizations

Provides comparison visualizations across different protocols.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class ProtocolComparison(BaseVisualizer):
    """Multi-protocol comparison visualizer"""

    def __init__(self):
        """Initialize protocol comparison visualizer"""
        super().__init__("Multi-Protocol")

    def plot_eye_diagram(self, signal_data, **kwargs) -> Optional[str]:
        """Plot eye diagram - implemented for compatibility"""
        return self.compare_eye_diagrams(signal_data, **kwargs)

    def plot_signal_quality(self, metrics: Dict[str, Any], **kwargs) -> Optional[str]:
        """Plot signal quality comparison"""
        if not self.matplotlib_available:
            logger.error("Matplotlib required for signal quality comparison")
            return None

        try:
            protocols = list(metrics.keys())
            metric_names = set()

            # Collect all metric names
            for protocol_metrics in metrics.values():
                if isinstance(protocol_metrics, dict):
                    metric_names.update(protocol_metrics.keys())

            metric_names = sorted(list(metric_names))

            # Create comparison plot
            fig, ax = self.plt.subplots(figsize=(12, 8))

            x = np.arange(len(metric_names))
            width = 0.25

            colors = ["blue", "red", "green", "orange", "purple"]

            for i, protocol in enumerate(protocols[:5]):  # Limit to 5 protocols
                protocol_values = []
                for metric in metric_names:
                    value = metrics[protocol].get(metric, 0)
                    protocol_values.append(value if isinstance(value, (int, float)) else 0)

                ax.bar(x + i * width, protocol_values, width, label=protocol, color=colors[i % len(colors)], alpha=0.8)

            ax.set_xlabel("Metrics")
            ax.set_ylabel("Values")
            ax.set_title("Protocol Signal Quality Comparison")
            ax.set_xticks(x + width * (len(protocols) - 1) / 2)
            ax.set_xticklabels(metric_names, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)

            self.plt.tight_layout()

            # Save plot
            filename = kwargs.get("filename", "protocol_signal_quality_comparison.png")
            filepath = self._ensure_output_dir(filename)
            self.plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.plt.close()

            return str(filepath)

        except Exception as e:
            logger.error(f"Signal quality comparison failed: {e}")
            return None

    def compare_eye_diagrams(self, protocol_data: Dict[str, Any], **kwargs) -> Optional[str]:
        """Compare eye diagrams across protocols"""
        if not self.matplotlib_available:
            logger.error("Matplotlib required for eye diagram comparison")
            return None

        try:
            protocols = list(protocol_data.keys())
            num_protocols = len(protocols)

            if num_protocols == 0:
                logger.error("No protocol data provided")
                return None

            # Create subplots
            cols = min(3, num_protocols)
            rows = (num_protocols + cols - 1) // cols

            fig, axes = self.plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            if num_protocols == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
            else:
                axes = axes.flatten()

            for i, protocol in enumerate(protocols):
                ax = axes[i]
                signal_data = protocol_data[protocol]["signal"]
                params = protocol_data[protocol].get("params", {})

                # Extract signal data
                if isinstance(signal_data, dict):
                    # Multi-lane data - use first lane
                    lane_data = next(iter(signal_data.values()))
                    if isinstance(lane_data, dict) and "voltage" in lane_data:
                        voltage_data = lane_data["voltage"]
                    else:
                        voltage_data = lane_data
                else:
                    voltage_data = signal_data

                # Create simple eye pattern
                samples_per_symbol = params.get("samples_per_symbol", 10)
                eye_data = self._create_simple_eye(voltage_data, samples_per_symbol)

                # Plot eye traces
                time_axis = np.linspace(0, 2, eye_data.shape[1])
                alpha = min(0.8, max(0.1, 50.0 / len(eye_data)))

                for trace in eye_data:
                    ax.plot(time_axis, trace, "b-", alpha=alpha, linewidth=0.5)

                ax.set_title(f"{protocol} Eye Diagram")
                ax.set_xlabel("Time (UI)")
                ax.set_ylabel("Amplitude (V)")
                ax.grid(True, alpha=0.3)
                ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7)

            # Hide unused subplots
            for i in range(num_protocols, len(axes)):
                axes[i].set_visible(False)

            self.plt.tight_layout()

            # Save plot
            filename = kwargs.get("filename", "protocol_eye_comparison.png")
            filepath = self._ensure_output_dir(filename)
            self.plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.plt.close()

            return str(filepath)

        except Exception as e:
            logger.error(f"Eye diagram comparison failed: {e}")
            return None

    def _create_simple_eye(self, signal_data: np.ndarray, samples_per_symbol: int) -> np.ndarray:
        """Create simple eye pattern for comparison"""
        try:
            if len(signal_data) < samples_per_symbol * 3:
                samples_per_symbol = max(1, len(signal_data) // 3)

            num_symbols = len(signal_data) // samples_per_symbol
            eye_width = 2 * samples_per_symbol
            eye_traces = []

            for i in range(min(num_symbols - 1, 100)):  # Limit traces for performance
                start_idx = i * samples_per_symbol
                end_idx = start_idx + eye_width
                if end_idx <= len(signal_data):
                    eye_traces.append(signal_data[start_idx:end_idx])

            if not eye_traces:
                eye_traces = [signal_data[:eye_width]] if len(signal_data) >= eye_width else [signal_data]

            return np.array(eye_traces)

        except Exception as e:
            logger.error(f"Simple eye creation failed: {e}")
            return np.array([signal_data[: min(len(signal_data), 2 * samples_per_symbol)]])

    def create_performance_comparison(self, performance_data: Dict[str, Dict], **kwargs) -> Optional[str]:
        """Create performance comparison across protocols"""
        if not self.matplotlib_available:
            logger.error("Matplotlib required for performance comparison")
            return None

        try:
            protocols = list(performance_data.keys())

            fig, ((ax1, ax2), (ax3, ax4)) = self.plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("Protocol Performance Comparison", fontsize=16, fontweight="bold")

            # Plot 1: Throughput comparison
            throughputs = [performance_data[p].get("throughput", 0) for p in protocols]
            bars1 = ax1.bar(protocols, throughputs, color=["blue", "red", "green", "orange"][: len(protocols)])
            ax1.set_title("Throughput Comparison")
            ax1.set_ylabel("Throughput (Gbps)")
            ax1.tick_params(axis="x", rotation=45)

            # Add value labels
            for bar, value in zip(bars1, throughputs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2.0, height + height * 0.01, f"{value:.1f}", ha="center", va="bottom")

            # Plot 2: Latency comparison
            latencies = [performance_data[p].get("latency", 0) for p in protocols]
            bars2 = ax2.bar(protocols, latencies, color=["blue", "red", "green", "orange"][: len(protocols)])
            ax2.set_title("Latency Comparison")
            ax2.set_ylabel("Latency (ns)")
            ax2.tick_params(axis="x", rotation=45)

            # Plot 3: Power consumption
            power = [performance_data[p].get("power", 0) for p in protocols]
            bars3 = ax3.bar(protocols, power, color=["blue", "red", "green", "orange"][: len(protocols)])
            ax3.set_title("Power Consumption")
            ax3.set_ylabel("Power (W)")
            ax3.tick_params(axis="x", rotation=45)

            # Plot 4: Efficiency (Gbps/W)
            efficiency = [t / p if p > 0 else 0 for t, p in zip(throughputs, power)]
            bars4 = ax4.bar(protocols, efficiency, color=["blue", "red", "green", "orange"][: len(protocols)])
            ax4.set_title("Power Efficiency")
            ax4.set_ylabel("Efficiency (Gbps/W)")
            ax4.tick_params(axis="x", rotation=45)

            self.plt.tight_layout()

            # Save plot
            filename = kwargs.get("filename", "protocol_performance_comparison.png")
            filepath = self._ensure_output_dir(filename)
            self.plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.plt.close()

            return str(filepath)

        except Exception as e:
            logger.error(f"Performance comparison failed: {e}")
            return None
