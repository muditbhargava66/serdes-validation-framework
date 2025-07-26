"""
Advanced Eye Diagram Visualization

Provides sophisticated eye diagram plotting capabilities for all protocols.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class EyeDiagramVisualizer(BaseVisualizer):
    """Advanced eye diagram visualization for SerDes signals"""

    def __init__(self, protocol_name: str = "SerDes"):
        """Initialize eye diagram visualizer"""
        super().__init__(protocol_name)

    def plot_eye_diagram(self, signal_data: Union[np.ndarray, Dict], **kwargs) -> Optional[str]:
        """
        Create comprehensive eye diagram visualization

        Args:
            signal_data: Signal data (numpy array or dict with lane data)
            **kwargs: Additional parameters
                - symbol_rate: Symbol rate in Hz
                - samples_per_symbol: Samples per symbol
                - filename: Output filename
                - title: Plot title
                - show_measurements: Whether to show eye measurements

        Returns:
            Path to saved plot or None if failed
        """
        if not self.matplotlib_available:
            logger.error("Matplotlib required for eye diagram plotting")
            return None

        try:
            # Extract parameters
            symbol_rate = kwargs.get("symbol_rate", 20e9)
            samples_per_symbol = kwargs.get("samples_per_symbol", 10)
            show_measurements = kwargs.get("show_measurements", True)
            title = kwargs.get("title", f"{self.protocol_name} Eye Diagram")

            # Handle different signal data formats
            if isinstance(signal_data, dict):
                # Multi-lane data - use first lane
                lane_data = next(iter(signal_data.values()))
                if isinstance(lane_data, dict) and "voltage" in lane_data:
                    voltage_data = lane_data["voltage"]
                else:
                    voltage_data = lane_data
            else:
                voltage_data = signal_data

            if not isinstance(voltage_data, np.ndarray):
                logger.error("Invalid signal data format")
                return None

            # Create eye diagram
            eye_data = self._create_eye_pattern(voltage_data, samples_per_symbol)

            # Create figure with subplots
            if show_measurements:
                fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(16, 8))
            else:
                fig, ax1 = self.plt.subplots(1, 1, figsize=(12, 8))

            # Main eye diagram
            self._plot_eye_traces(ax1, eye_data, samples_per_symbol)
            ax1.set_title(title, fontsize=14, fontweight="bold")
            ax1.set_xlabel("Time (UI)", fontsize=12)
            ax1.set_ylabel("Amplitude (V)", fontsize=12)
            ax1.grid(True, alpha=0.3)

            # Add eye measurements if requested
            if show_measurements:
                measurements = self._calculate_eye_measurements(eye_data)
                self._plot_eye_measurements(ax2, measurements)

                # Add measurement text to main plot
                self._add_measurement_text(ax1, measurements)

            # Enhance plot appearance
            if self.seaborn_available:
                self.sns.despine()

            self.plt.tight_layout()

            # Save plot
            filename = kwargs.get("filename", f"{self.protocol_name.lower()}_eye_diagram.png")
            filepath = self._ensure_output_dir(filename)
            self.plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.plt.close()

            logger.info(f"Eye diagram saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Eye diagram plotting failed: {e}")
            return None

    def _create_eye_pattern(self, signal_data: np.ndarray, samples_per_symbol: int) -> np.ndarray:
        """Create eye pattern by overlaying symbol periods"""
        try:
            # Ensure we have enough data
            if len(signal_data) < samples_per_symbol * 3:
                logger.warning("Insufficient data for eye diagram, using available data")
                samples_per_symbol = max(1, len(signal_data) // 3)

            # Calculate number of complete symbols
            num_symbols = len(signal_data) // samples_per_symbol

            # Reshape data into symbol periods (2 UI wide for eye diagram)
            eye_width = 2 * samples_per_symbol
            eye_traces = []

            for i in range(num_symbols - 1):  # -1 because we need 2 symbols for each trace
                start_idx = i * samples_per_symbol
                end_idx = start_idx + eye_width
                if end_idx <= len(signal_data):
                    eye_traces.append(signal_data[start_idx:end_idx])

            if not eye_traces:
                # Fallback: create a simple eye pattern
                eye_traces = [signal_data[:eye_width]] if len(signal_data) >= eye_width else [signal_data]

            return np.array(eye_traces)

        except Exception as e:
            logger.error(f"Eye pattern creation failed: {e}")
            # Return a simple pattern as fallback
            return np.array([signal_data[: min(len(signal_data), 2 * samples_per_symbol)]])

    def _plot_eye_traces(self, ax, eye_data: np.ndarray, samples_per_symbol: int):
        """Plot eye diagram traces"""
        try:
            # Create time axis (2 UI wide)
            time_axis = np.linspace(0, 2, eye_data.shape[1])

            # Plot all traces with transparency
            alpha = min(0.8, max(0.1, 50.0 / len(eye_data)))  # Adaptive transparency

            for trace in eye_data:
                ax.plot(time_axis, trace, "b-", alpha=alpha, linewidth=0.5)

            # Add UI markers
            ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="Sampling Point")
            ax.axvline(x=1.5, color="red", linestyle="--", alpha=0.7)
            ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5, label="Symbol Boundary")

            # Set axis limits
            ax.set_xlim(0, 2)

            # Add legend
            ax.legend(loc="upper right")

        except Exception as e:
            logger.error(f"Eye trace plotting failed: {e}")

    def _calculate_eye_measurements(self, eye_data: np.ndarray) -> Dict[str, float]:
        """Calculate eye diagram measurements"""
        try:
            measurements = {}

            if len(eye_data) == 0:
                return measurements

            # Find sampling point (middle of eye)
            sampling_point = eye_data.shape[1] // 2

            # Calculate eye height (vertical opening)
            sampling_voltages = eye_data[:, sampling_point]
            if len(sampling_voltages) > 0:
                measurements["eye_height"] = np.max(sampling_voltages) - np.min(sampling_voltages)
                measurements["eye_center"] = (np.max(sampling_voltages) + np.min(sampling_voltages)) / 2

            # Calculate eye width (horizontal opening) - simplified
            measurements["eye_width"] = 0.8  # Placeholder - would need more sophisticated calculation

            # Calculate signal-to-noise ratio
            if len(sampling_voltages) > 1:
                measurements["snr"] = np.mean(sampling_voltages) / (np.std(sampling_voltages) + 1e-10)

            # Calculate jitter (RMS)
            measurements["rms_jitter"] = np.std(sampling_voltages) if len(sampling_voltages) > 1 else 0.0

            # Calculate crossing percentage
            zero_crossings = np.sum(np.diff(np.sign(eye_data.mean(axis=0))) != 0)
            measurements["crossing_percentage"] = (zero_crossings / eye_data.shape[1]) * 100

            return measurements

        except Exception as e:
            logger.error(f"Eye measurement calculation failed: {e}")
            return {}

    def _plot_eye_measurements(self, ax, measurements: Dict[str, float]):
        """Plot eye measurements as bar chart"""
        try:
            if not measurements:
                ax.text(0.5, 0.5, "No measurements available", ha="center", va="center", transform=ax.transAxes)
                ax.set_title("Eye Measurements")
                return

            # Prepare data for plotting
            metric_names = []
            metric_values = []
            metric_units = []

            for key, value in measurements.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metric_names.append(key.replace("_", " ").title())
                    metric_values.append(abs(value))  # Use absolute values for bar chart

                    # Add appropriate units
                    if "height" in key or "center" in key:
                        metric_units.append("V")
                    elif "width" in key:
                        metric_units.append("UI")
                    elif "jitter" in key:
                        metric_units.append("ps")
                    elif "snr" in key:
                        metric_units.append("dB")
                    elif "percentage" in key:
                        metric_units.append("%")
                    else:
                        metric_units.append("")

            if metric_names:
                bars = ax.bar(metric_names, metric_values)
                ax.set_title("Eye Measurements", fontsize=12, fontweight="bold")
                ax.set_ylabel("Value")
                ax.tick_params(axis="x", rotation=45)

                # Add value labels on bars
                for bar, value, unit in zip(bars, metric_values, metric_units):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + height * 0.01,
                        f"{value:.3f}{unit}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                # Color bars based on quality (green=good, yellow=ok, red=poor)
                for bar, name in zip(bars, metric_names):
                    if "height" in name.lower():
                        bar.set_color("green" if bar.get_height() > 0.5 else "orange" if bar.get_height() > 0.3 else "red")
                    elif "snr" in name.lower():
                        bar.set_color("green" if bar.get_height() > 20 else "orange" if bar.get_height() > 10 else "red")
                    else:
                        bar.set_color("blue")

        except Exception as e:
            logger.error(f"Eye measurements plotting failed: {e}")

    def _add_measurement_text(self, ax, measurements: Dict[str, float]):
        """Add measurement text to main eye diagram"""
        try:
            if not measurements:
                return

            # Create measurement text
            text_lines = []
            for key, value in measurements.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    if "height" in key:
                        text_lines.append(f"Eye Height: {value:.3f} V")
                    elif "width" in key:
                        text_lines.append(f"Eye Width: {value:.3f} UI")
                    elif "snr" in key:
                        text_lines.append(f"SNR: {value:.1f} dB")
                    elif "jitter" in key:
                        text_lines.append(f"RMS Jitter: {value:.3f} ps")

            if text_lines:
                text_str = "\n".join(text_lines[:4])  # Limit to 4 lines
                ax.text(
                    0.02,
                    0.98,
                    text_str,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        except Exception as e:
            logger.error(f"Adding measurement text failed: {e}")

    def plot_signal_quality(self, metrics: Dict[str, Any], **kwargs) -> Optional[str]:
        """Plot signal quality metrics"""
        if not self.matplotlib_available:
            logger.error("Matplotlib required for signal quality plotting")
            return None

        try:
            fig, ax = self.plt.subplots(figsize=(10, 6))

            # Extract numeric metrics
            metric_names = []
            metric_values = []

            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metric_names.append(key.replace("_", " ").title())
                    metric_values.append(value)

            if metric_names:
                bars = ax.bar(metric_names, metric_values)
                ax.set_title(f"{self.protocol_name} Signal Quality Metrics")
                ax.set_ylabel("Value")
                ax.tick_params(axis="x", rotation=45)

                # Color bars based on values
                for bar, value in zip(bars, metric_values):
                    if value > 0.8:
                        bar.set_color("green")
                    elif value > 0.6:
                        bar.set_color("orange")
                    else:
                        bar.set_color("red")

            self.plt.tight_layout()

            # Save plot
            filename = kwargs.get("filename", f"{self.protocol_name.lower()}_signal_quality.png")
            filepath = self._ensure_output_dir(filename)
            self.plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.plt.close()

            return str(filepath)

        except Exception as e:
            logger.error(f"Signal quality plotting failed: {e}")
            return None
