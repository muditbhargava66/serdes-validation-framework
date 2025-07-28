"""
Interactive Plots for Jupyter Dashboard

Provides interactive plotting capabilities optimized for Jupyter notebooks
with widgets and real-time updates.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, Slider
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import iplot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False


@dataclass
class PlotConfig:
    """Configuration for interactive plots"""
    
    width: int = 800
    height: int = 600
    theme: str = 'plotly_white'
    color_scheme: List[str] = None
    
    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = ['blue', 'red', 'green', 'orange', 'purple']


class InteractivePlotter:
    """
    Interactive plotter for Jupyter dashboard
    
    Provides interactive plotting capabilities with widgets for
    real-time parameter adjustment and visualization updates.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self.plots = {}
        self.widgets = {}
        self.callbacks = {}
        
        if not JUPYTER_AVAILABLE:
            print("⚠️  Jupyter widgets not available. Interactive features limited.")
    
    def create_interactive_eye_plot(self, 
                                   signal_data: Dict[str, np.ndarray],
                                   time_data: np.ndarray,
                                   sample_rate: float = 40e9,
                                   protocol: str = "USB4") -> None:
        """
        Create interactive eye diagram plot with controls
        
        Args:
            signal_data: Dictionary of lane signals
            time_data: Time axis data
            sample_rate: Sampling rate
            protocol: Protocol type
        """
        if not JUPYTER_AVAILABLE:
            print("❌ Jupyter widgets required for interactive plots")
            return
        
        # Create control widgets
        lane_options = list(signal_data.keys())
        
        self.widgets['eye_lane_selector'] = widgets.Dropdown(
            options=lane_options,
            value=lane_options[0],
            description='Lane:',
            style={'description_width': 'initial'}
        )
        
        self.widgets['eye_symbols'] = widgets.IntSlider(
            value=100,
            min=10,
            max=500,
            step=10,
            description='Symbols:',
            style={'description_width': 'initial'}
        )
        
        self.widgets['eye_persistence'] = widgets.FloatSlider(
            value=0.1,
            min=0.01,
            max=1.0,
            step=0.01,
            description='Persistence:',
            style={'description_width': 'initial'}
        )
        
        self.widgets['eye_update'] = widgets.Button(
            description='Update Eye Diagram',
            button_style='primary'
        )
        
        # Output widget for plots
        self.widgets['eye_output'] = widgets.Output()
        
        # Bind update function
        def update_eye_plot(change=None):
            with self.widgets['eye_output']:
                clear_output(wait=True)
                self._plot_eye_diagram(
                    signal_data,
                    time_data,
                    self.widgets['eye_lane_selector'].value,
                    self.widgets['eye_symbols'].value,
                    self.widgets['eye_persistence'].value,
                    sample_rate,
                    protocol
                )
        
        self.widgets['eye_update'].on_click(update_eye_plot)
        
        # Display controls
        controls = widgets.VBox([
            widgets.HBox([
                self.widgets['eye_lane_selector'],
                self.widgets['eye_symbols'],
                self.widgets['eye_persistence'],
                self.widgets['eye_update']
            ]),
            self.widgets['eye_output']
        ])
        
        display(controls)
        
        # Initial plot
        update_eye_plot()
    
    def _plot_eye_diagram(self, signal_data: Dict[str, np.ndarray],
                         time_data: np.ndarray,
                         lane: str,
                         num_symbols: int,
                         persistence: float,
                         sample_rate: float,
                         protocol: str):
        """Plot eye diagram with current parameters"""
        if lane not in signal_data:
            print(f"❌ Lane {lane} not found")
            return
        
        signal = signal_data[lane]
        
        # Create eye pattern
        eye_pattern = self._create_eye_pattern(signal, num_symbols)
        
        if PLOTLY_AVAILABLE:
            # Create Plotly eye diagram
            fig = go.Figure()
            
            # Add eye traces with persistence effect
            for i in range(0, len(eye_pattern), len(eye_pattern)//num_symbols):
                if i + len(eye_pattern)//num_symbols < len(eye_pattern):
                    trace_data = eye_pattern[i:i+len(eye_pattern)//num_symbols]
                    trace_time = np.linspace(-0.5, 0.5, len(trace_data))
                    
                    fig.add_trace(go.Scatter(
                        x=trace_time,
                        y=trace_data,
                        mode='lines',
                        line=dict(color='blue', width=1),
                        opacity=persistence,
                        showlegend=False
                    ))
            
            fig.update_layout(
                title=f"Eye Diagram - {protocol} {lane}",
                xaxis_title="Time (UI)",
                yaxis_title="Voltage (V)",
                width=self.config.width,
                height=self.config.height//2,
                template=self.config.theme
            )
            
            fig.show()
            
        elif MATPLOTLIB_AVAILABLE:
            # Create matplotlib eye diagram
            plt.figure(figsize=(10, 6))
            
            for i in range(0, len(eye_pattern), len(eye_pattern)//num_symbols):
                if i + len(eye_pattern)//num_symbols < len(eye_pattern):
                    trace_data = eye_pattern[i:i+len(eye_pattern)//num_symbols]
                    trace_time = np.linspace(-0.5, 0.5, len(trace_data))
                    
                    plt.plot(trace_time, trace_data, 'b-', alpha=persistence, linewidth=0.5)
            
            plt.title(f"Eye Diagram - {protocol} {lane}")
            plt.xlabel("Time (UI)")
            plt.ylabel("Voltage (V)")
            plt.grid(True, alpha=0.3)
            plt.show()
    
    def _create_eye_pattern(self, signal: np.ndarray, num_symbols: int) -> np.ndarray:
        """Create eye pattern from signal"""
        symbol_length = len(signal) // num_symbols
        if symbol_length < 10:
            return signal
        
        eye_pattern = []
        for i in range(0, len(signal) - symbol_length, symbol_length):
            eye_pattern.extend(signal[i:i+symbol_length])
        
        return np.array(eye_pattern)
    
    def create_interactive_spectrum_plot(self,
                                       signal_data: Dict[str, np.ndarray],
                                       sample_rate: float = 40e9,
                                       protocol: str = "USB4") -> None:
        """
        Create interactive spectrum analyzer plot
        
        Args:
            signal_data: Dictionary of lane signals
            sample_rate: Sampling rate
            protocol: Protocol type
        """
        if not JUPYTER_AVAILABLE:
            print("❌ Jupyter widgets required for interactive plots")
            return
        
        # Create control widgets
        lane_options = list(signal_data.keys())
        
        self.widgets['spectrum_lane'] = widgets.Dropdown(
            options=lane_options,
            value=lane_options[0],
            description='Lane:',
            style={'description_width': 'initial'}
        )
        
        self.widgets['spectrum_window'] = widgets.Dropdown(
            options=['hanning', 'hamming', 'blackman', 'rectangular'],
            value='hanning',
            description='Window:',
            style={'description_width': 'initial'}
        )
        
        self.widgets['spectrum_scale'] = widgets.Dropdown(
            options=['linear', 'log'],
            value='log',
            description='Scale:',
            style={'description_width': 'initial'}
        )
        
        self.widgets['spectrum_update'] = widgets.Button(
            description='Update Spectrum',
            button_style='primary'
        )
        
        # Output widget
        self.widgets['spectrum_output'] = widgets.Output()
        
        # Bind update function
        def update_spectrum(change=None):
            with self.widgets['spectrum_output']:
                clear_output(wait=True)
                self._plot_spectrum(
                    signal_data,
                    self.widgets['spectrum_lane'].value,
                    self.widgets['spectrum_window'].value,
                    self.widgets['spectrum_scale'].value,
                    sample_rate,
                    protocol
                )
        
        self.widgets['spectrum_update'].on_click(update_spectrum)
        
        # Display controls
        controls = widgets.VBox([
            widgets.HBox([
                self.widgets['spectrum_lane'],
                self.widgets['spectrum_window'],
                self.widgets['spectrum_scale'],
                self.widgets['spectrum_update']
            ]),
            self.widgets['spectrum_output']
        ])
        
        display(controls)
        
        # Initial plot
        update_spectrum()
    
    def _plot_spectrum(self, signal_data: Dict[str, np.ndarray],
                      lane: str,
                      window: str,
                      scale: str,
                      sample_rate: float,
                      protocol: str):
        """Plot spectrum with current parameters"""
        if lane not in signal_data:
            print(f"❌ Lane {lane} not found")
            return
        
        signal = signal_data[lane]
        
        # Apply window function
        if window == 'hanning':
            windowed_signal = signal * np.hanning(len(signal))
        elif window == 'hamming':
            windowed_signal = signal * np.hamming(len(signal))
        elif window == 'blackman':
            windowed_signal = signal * np.blackman(len(signal))
        else:  # rectangular
            windowed_signal = signal
        
        # Compute FFT
        fft_data = np.fft.fft(windowed_signal)
        freqs = np.fft.fftfreq(len(windowed_signal), 1/sample_rate)
        
        # Take positive frequencies only
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_data[:len(fft_data)//2]
        
        # Convert to power spectrum
        if scale == 'log':
            power_spectrum = 20 * np.log10(np.abs(pos_fft) + 1e-12)
            ylabel = "Power (dB)"
        else:
            power_spectrum = np.abs(pos_fft)
            ylabel = "Magnitude"
        
        if PLOTLY_AVAILABLE:
            # Create Plotly spectrum plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pos_freqs/1e9,
                y=power_spectrum,
                mode='lines',
                name='Spectrum',
                line=dict(color='red', width=1)
            ))
            
            fig.update_layout(
                title=f"Spectrum Analysis - {protocol} {lane} ({window} window)",
                xaxis_title="Frequency (GHz)",
                yaxis_title=ylabel,
                width=self.config.width,
                height=self.config.height//2,
                template=self.config.theme
            )
            
            fig.show()
            
        elif MATPLOTLIB_AVAILABLE:
            # Create matplotlib spectrum plot
            plt.figure(figsize=(10, 6))
            plt.plot(pos_freqs/1e9, power_spectrum, 'r-', linewidth=1)
            plt.title(f"Spectrum Analysis - {protocol} {lane} ({window} window)")
            plt.xlabel("Frequency (GHz)")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.show()
    
    def create_multi_lane_comparison(self,
                                   signal_data: Dict[str, np.ndarray],
                                   time_data: np.ndarray,
                                   protocol: str = "USB4") -> None:
        """
        Create multi-lane comparison plot
        
        Args:
            signal_data: Dictionary of lane signals
            time_data: Time axis data
            protocol: Protocol type
        """
        if not JUPYTER_AVAILABLE:
            print("❌ Jupyter widgets required for interactive plots")
            return
        
        # Create control widgets
        self.widgets['comparison_lanes'] = widgets.SelectMultiple(
            options=list(signal_data.keys()),
            value=list(signal_data.keys())[:min(4, len(signal_data))],
            description='Lanes:',
            style={'description_width': 'initial'}
        )
        
        self.widgets['comparison_plot_type'] = widgets.Dropdown(
            options=['overlay', 'separate'],
            value='separate',
            description='Plot Type:',
            style={'description_width': 'initial'}
        )
        
        self.widgets['comparison_update'] = widgets.Button(
            description='Update Comparison',
            button_style='primary'
        )
        
        # Output widget
        self.widgets['comparison_output'] = widgets.Output()
        
        # Bind update function
        def update_comparison(change=None):
            with self.widgets['comparison_output']:
                clear_output(wait=True)
                self._plot_multi_lane_comparison(
                    signal_data,
                    time_data,
                    list(self.widgets['comparison_lanes'].value),
                    self.widgets['comparison_plot_type'].value,
                    protocol
                )
        
        self.widgets['comparison_update'].on_click(update_comparison)
        
        # Display controls
        controls = widgets.VBox([
            widgets.HBox([
                self.widgets['comparison_lanes'],
                self.widgets['comparison_plot_type'],
                self.widgets['comparison_update']
            ]),
            self.widgets['comparison_output']
        ])
        
        display(controls)
        
        # Initial plot
        update_comparison()
    
    def _plot_multi_lane_comparison(self,
                                   signal_data: Dict[str, np.ndarray],
                                   time_data: np.ndarray,
                                   selected_lanes: List[str],
                                   plot_type: str,
                                   protocol: str):
        """Plot multi-lane comparison"""
        if not selected_lanes:
            print("❌ No lanes selected")
            return
        
        if PLOTLY_AVAILABLE:
            if plot_type == 'overlay':
                # Single plot with all lanes overlaid
                fig = go.Figure()
                
                for i, lane in enumerate(selected_lanes):
                    if lane in signal_data:
                        color = self.config.color_scheme[i % len(self.config.color_scheme)]
                        fig.add_trace(go.Scatter(
                            x=time_data*1e9,
                            y=signal_data[lane],
                            mode='lines',
                            name=lane,
                            line=dict(color=color, width=1)
                        ))
                
                fig.update_layout(
                    title=f"Multi-Lane Comparison - {protocol}",
                    xaxis_title="Time (ns)",
                    yaxis_title="Voltage (V)",
                    width=self.config.width,
                    height=self.config.height,
                    template=self.config.theme
                )
                
                fig.show()
                
            else:  # separate subplots
                fig = sp.make_subplots(
                    rows=len(selected_lanes),
                    cols=1,
                    subplot_titles=selected_lanes,
                    shared_xaxes=True
                )
                
                for i, lane in enumerate(selected_lanes):
                    if lane in signal_data:
                        color = self.config.color_scheme[i % len(self.config.color_scheme)]
                        fig.add_trace(
                            go.Scatter(
                                x=time_data*1e9,
                                y=signal_data[lane],
                                mode='lines',
                                name=lane,
                                line=dict(color=color, width=1)
                            ),
                            row=i+1, col=1
                        )
                
                fig.update_layout(
                    title=f"Multi-Lane Comparison - {protocol}",
                    height=200 * len(selected_lanes),
                    template=self.config.theme
                )
                
                fig.update_xaxes(title_text="Time (ns)", row=len(selected_lanes), col=1)
                
                for i in range(len(selected_lanes)):
                    fig.update_yaxes(title_text="Voltage (V)", row=i+1, col=1)
                
                fig.show()
                
        elif MATPLOTLIB_AVAILABLE:
            if plot_type == 'overlay':
                # Single plot with all lanes overlaid
                plt.figure(figsize=(12, 6))
                
                for i, lane in enumerate(selected_lanes):
                    if lane in signal_data:
                        color = self.config.color_scheme[i % len(self.config.color_scheme)]
                        plt.plot(time_data*1e9, signal_data[lane], 
                                color=color, label=lane, linewidth=1)
                
                plt.title(f"Multi-Lane Comparison - {protocol}")
                plt.xlabel("Time (ns)")
                plt.ylabel("Voltage (V)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
                
            else:  # separate subplots
                fig, axes = plt.subplots(len(selected_lanes), 1, 
                                       figsize=(12, 3*len(selected_lanes)),
                                       sharex=True)
                
                if len(selected_lanes) == 1:
                    axes = [axes]
                
                for i, lane in enumerate(selected_lanes):
                    if lane in signal_data:
                        color = self.config.color_scheme[i % len(self.config.color_scheme)]
                        axes[i].plot(time_data*1e9, signal_data[lane], 
                                   color=color, linewidth=1)
                        axes[i].set_ylabel("Voltage (V)")
                        axes[i].set_title(lane)
                        axes[i].grid(True, alpha=0.3)
                
                axes[-1].set_xlabel("Time (ns)")
                plt.suptitle(f"Multi-Lane Comparison - {protocol}")
                plt.tight_layout()
                plt.show()
    
    def create_measurement_dashboard(self,
                                   analysis_results: Dict[str, Any],
                                   protocol: str = "USB4") -> None:
        """
        Create measurement dashboard with key metrics
        
        Args:
            analysis_results: Dictionary of analysis results per lane
            protocol: Protocol type
        """
        if not analysis_results:
            print("❌ No analysis results provided")
            return
        
        if PLOTLY_AVAILABLE:
            # Create measurement dashboard with Plotly
            lanes = list(analysis_results.keys())
            
            # Extract metrics
            eye_heights = [analysis_results[lane].get('eye_height', 0) for lane in lanes]
            snr_values = [analysis_results[lane].get('snr', 0) for lane in lanes]
            jitter_values = [analysis_results[lane].get('rms_jitter', 0) for lane in lanes]
            pass_status = [analysis_results[lane].get('passed', False) for lane in lanes]
            
            # Create subplots
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=('Eye Height', 'SNR', 'Jitter', 'Pass/Fail Status'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Eye height bar chart
            fig.add_trace(
                go.Bar(x=lanes, y=eye_heights, name='Eye Height', 
                      marker_color='blue'),
                row=1, col=1
            )
            
            # SNR bar chart
            fig.add_trace(
                go.Bar(x=lanes, y=snr_values, name='SNR', 
                      marker_color='green'),
                row=1, col=2
            )
            
            # Jitter bar chart
            fig.add_trace(
                go.Bar(x=lanes, y=jitter_values, name='Jitter', 
                      marker_color='orange'),
                row=2, col=1
            )
            
            # Pass/Fail status
            status_colors = ['green' if status else 'red' for status in pass_status]
            status_values = [1 if status else 0 for status in pass_status]
            
            fig.add_trace(
                go.Bar(x=lanes, y=status_values, name='Status',
                      marker_color=status_colors),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f"Measurement Dashboard - {protocol}",
                height=600,
                showlegend=False,
                template=self.config.theme
            )
            
            # Update axes
            fig.update_yaxes(title_text="Eye Height (V)", row=1, col=1)
            fig.update_yaxes(title_text="SNR (dB)", row=1, col=2)
            fig.update_yaxes(title_text="Jitter (UI)", row=2, col=1)
            fig.update_yaxes(title_text="Pass (1) / Fail (0)", row=2, col=2)
            
            fig.show()
            
        elif MATPLOTLIB_AVAILABLE:
            # Create measurement dashboard with matplotlib
            lanes = list(analysis_results.keys())
            
            # Extract metrics
            eye_heights = [analysis_results[lane].get('eye_height', 0) for lane in lanes]
            snr_values = [analysis_results[lane].get('snr', 0) for lane in lanes]
            jitter_values = [analysis_results[lane].get('rms_jitter', 0) for lane in lanes]
            pass_status = [analysis_results[lane].get('passed', False) for lane in lanes]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Eye height
            ax1.bar(lanes, eye_heights, color='blue', alpha=0.7)
            ax1.set_ylabel('Eye Height (V)')
            ax1.set_title('Eye Height')
            ax1.tick_params(axis='x', rotation=45)
            
            # SNR
            ax2.bar(lanes, snr_values, color='green', alpha=0.7)
            ax2.set_ylabel('SNR (dB)')
            ax2.set_title('SNR')
            ax2.tick_params(axis='x', rotation=45)
            
            # Jitter
            ax3.bar(lanes, jitter_values, color='orange', alpha=0.7)
            ax3.set_ylabel('Jitter (UI)')
            ax3.set_title('Jitter')
            ax3.tick_params(axis='x', rotation=45)
            
            # Pass/Fail status
            status_colors = ['green' if status else 'red' for status in pass_status]
            status_values = [1 if status else 0 for status in pass_status]
            
            ax4.bar(lanes, status_values, color=status_colors, alpha=0.7)
            ax4.set_ylabel('Pass (1) / Fail (0)')
            ax4.set_title('Pass/Fail Status')
            ax4.tick_params(axis='x', rotation=45)
            ax4.set_ylim(0, 1.2)
            
            plt.suptitle(f"Measurement Dashboard - {protocol}")
            plt.tight_layout()
            plt.show()


def check_interactive_dependencies():
    """Check dependencies for interactive plotting"""
    status = {
        'Jupyter': JUPYTER_AVAILABLE,
        'Matplotlib': MATPLOTLIB_AVAILABLE,
        'Plotly': PLOTLY_AVAILABLE
    }
    
    print("🎨 Interactive Plotting Dependencies:")
    for name, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {name}: {'Available' if available else 'Not Available'}")
    
    if not JUPYTER_AVAILABLE:
        print("\n⚠️  Install Jupyter for full interactive features:")
        print("    pip install jupyter ipywidgets")
    
    return status
