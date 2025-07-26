"""
Base Visualizer Class

Provides common visualization functionality for all protocols.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class BaseVisualizer(ABC):
    """Base class for all protocol visualizers"""

    def __init__(self, protocol_name: str):
        """Initialize base visualizer"""
        self.protocol_name = protocol_name
        self.matplotlib_available = False
        self.plotly_available = False
        self.seaborn_available = False

        # Check for optional dependencies
        try:
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt

            self.plt = plt
            self.patches = patches
            self.matplotlib_available = True
            logger.info(f"Matplotlib available for {protocol_name} visualization")
        except ImportError:
            logger.warning(f"Matplotlib not available for {protocol_name} - static plots disabled")

        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import plotly.subplots as sp

            self.go = go
            self.px = px
            self.sp = sp
            self.plotly_available = True
            logger.info(f"Plotly available for {protocol_name} visualization")
        except ImportError:
            logger.warning(f"Plotly not available for {protocol_name} - interactive plots disabled")

        try:
            import seaborn as sns

            self.sns = sns
            self.seaborn_available = True
            logger.info(f"Seaborn available for {protocol_name} visualization")
        except ImportError:
            logger.warning(f"Seaborn not available for {protocol_name} - enhanced styling disabled")

    def _ensure_output_dir(self, filepath: str) -> Path:
        """Ensure output directory exists"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _validate_signal_data(self, signal_data: Union[np.ndarray, Dict[int, Dict[str, np.ndarray]]]) -> bool:
        """Validate signal data format"""
        if isinstance(signal_data, np.ndarray):
            return len(signal_data) > 0
        elif isinstance(signal_data, dict):
            return all(
                isinstance(lane_data, dict) and "voltage" in lane_data and isinstance(lane_data["voltage"], np.ndarray)
                for lane_data in signal_data.values()
            )
        return False

    @abstractmethod
    def plot_eye_diagram(self, signal_data: Union[np.ndarray, Dict], **kwargs) -> Optional[str]:
        """Plot eye diagram - must be implemented by subclasses"""
        pass

    @abstractmethod
    def plot_signal_quality(self, metrics: Dict[str, Any], **kwargs) -> Optional[str]:
        """Plot signal quality metrics - must be implemented by subclasses"""
        pass

    def plot_spectrum(self, signal_data: np.ndarray, sample_rate: float, **kwargs) -> Optional[str]:
        """Plot frequency spectrum"""
        if not self.matplotlib_available:
            logger.error("Matplotlib required for spectrum plotting")
            return None

        try:
            # Calculate FFT
            fft_data = np.fft.fft(signal_data)
            freqs = np.fft.fftfreq(len(signal_data), 1 / sample_rate)

            # Only plot positive frequencies
            pos_mask = freqs >= 0
            freqs = freqs[pos_mask]
            magnitude = np.abs(fft_data[pos_mask])

            # Create plot
            fig, ax = self.plt.subplots(figsize=(12, 6))
            ax.semilogx(freqs[1:], 20 * np.log10(magnitude[1:]))  # Skip DC component
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_title(f"{self.protocol_name} Signal Spectrum")
            ax.grid(True, alpha=0.3)

            # Save plot
            filename = kwargs.get("filename", f"{self.protocol_name.lower()}_spectrum.png")
            filepath = self._ensure_output_dir(filename)
            self.plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.plt.close()

            return str(filepath)

        except Exception as e:
            logger.error(f"Spectrum plotting failed: {e}")
            return None

    def plot_time_domain(self, signal_data: np.ndarray, time_axis: Optional[np.ndarray] = None, **kwargs) -> Optional[str]:
        """Plot time domain signal"""
        if not self.matplotlib_available:
            logger.error("Matplotlib required for time domain plotting")
            return None

        try:
            if time_axis is None:
                time_axis = np.arange(len(signal_data))

            fig, ax = self.plt.subplots(figsize=(12, 6))
            ax.plot(time_axis, signal_data, linewidth=0.8)
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude (V)")
            ax.set_title(f"{self.protocol_name} Time Domain Signal")
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_val = np.mean(signal_data)
            std_val = np.std(signal_data)
            ax.axhline(mean_val, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_val:.3f}V")
            ax.axhline(mean_val + std_val, color="orange", linestyle="--", alpha=0.7, label=f"+1σ: {mean_val + std_val:.3f}V")
            ax.axhline(mean_val - std_val, color="orange", linestyle="--", alpha=0.7, label=f"-1σ: {mean_val - std_val:.3f}V")
            ax.legend()

            # Save plot
            filename = kwargs.get("filename", f"{self.protocol_name.lower()}_time_domain.png")
            filepath = self._ensure_output_dir(filename)
            self.plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.plt.close()

            return str(filepath)

        except Exception as e:
            logger.error(f"Time domain plotting failed: {e}")
            return None

    def create_summary_report(self, test_results: Dict[str, Any], **kwargs) -> Optional[str]:
        """Create a comprehensive summary report"""
        if not self.matplotlib_available:
            logger.error("Matplotlib required for summary report")
            return None

        try:
            # Create multi-subplot figure
            fig, axes = self.plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"{self.protocol_name} Validation Summary Report", fontsize=16, fontweight="bold")

            # Plot 1: Test Results Overview
            ax1 = axes[0, 0]
            if "test_results" in test_results:
                results = test_results["test_results"]
                passed = sum(1 for r in results.values() if r.get("status") == "PASS")
                failed = sum(1 for r in results.values() if r.get("status") == "FAIL")

                ax1.pie([passed, failed], labels=["Passed", "Failed"], colors=["green", "red"], autopct="%1.1f%%")
                ax1.set_title("Test Results Overview")

            # Plot 2: Signal Quality Metrics
            ax2 = axes[0, 1]
            if "signal_quality" in test_results:
                metrics = test_results["signal_quality"]
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())

                bars = ax2.bar(metric_names, metric_values)
                ax2.set_title("Signal Quality Metrics")
                ax2.set_ylabel("Value")
                ax2.tick_params(axis="x", rotation=45)

                # Color bars based on values
                for bar, value in zip(bars, metric_values):
                    if isinstance(value, (int, float)):
                        if value > 0.8:
                            bar.set_color("green")
                        elif value > 0.6:
                            bar.set_color("orange")
                        else:
                            bar.set_color("red")

            # Plot 3: Performance Metrics
            ax3 = axes[1, 0]
            if "performance" in test_results:
                perf = test_results["performance"]
                if "throughput" in perf and "latency" in perf:
                    ax3.scatter([perf["throughput"]], [perf["latency"]], s=100, c="blue")
                    ax3.set_xlabel("Throughput (Gbps)")
                    ax3.set_ylabel("Latency (ns)")
                    ax3.set_title("Performance Characteristics")
                    ax3.grid(True, alpha=0.3)

            # Plot 4: Compliance Status
            ax4 = axes[1, 1]
            if "compliance" in test_results:
                compliance = test_results["compliance"]
                categories = list(compliance.keys())
                statuses = [1 if compliance[cat] else 0 for cat in categories]

                colors = ["green" if s else "red" for s in statuses]
                ax4.bar(categories, statuses, color=colors)
                ax4.set_title("Compliance Status")
                ax4.set_ylabel("Pass/Fail")
                ax4.set_ylim(0, 1.2)
                ax4.tick_params(axis="x", rotation=45)

            self.plt.tight_layout()

            # Save report
            filename = kwargs.get("filename", f"{self.protocol_name.lower()}_summary_report.png")
            filepath = self._ensure_output_dir(filename)
            self.plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.plt.close()

            return str(filepath)

        except Exception as e:
            logger.error(f"Summary report creation failed: {e}")
            return None
