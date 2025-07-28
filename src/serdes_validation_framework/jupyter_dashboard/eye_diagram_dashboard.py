"""
Jupyter Eye Diagram Dashboard

Interactive dashboard for eye diagram visualization in Jupyter notebooks
with SVF analyzer integration and pass/fail annotations.
"""

# Set matplotlib backend to non-GUI for testing environments
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

if os.environ.get('SVF_MOCK_MODE') == '1' or os.environ.get('MPLBACKEND') == 'Agg':
    import matplotlib
    matplotlib.use('Agg')

# Jupyter and visualization imports with fallbacks
try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import init_notebook_mode, iplot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from ..data_analysis.eye_masks import EyeMaskAnalyzer, create_eye_mask_analyzer
    EYE_MASKS_AVAILABLE = True
except ImportError:
    EYE_MASKS_AVAILABLE = False

try:
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# Framework imports
try:
    from ..data_analysis.eye_diagram import AdvancedEyeAnalyzer, EyeParameters
    from ..visualization.eye_diagram import EyeDiagramVisualizer
    EYE_ANALYSIS_AVAILABLE = True
except ImportError:
    EYE_ANALYSIS_AVAILABLE = False


@dataclass
class DashboardConfig:
    """Configuration for Jupyter dashboard"""
    
    # Display options
    figure_width: int = 12
    figure_height: int = 8
    dpi: int = 100
    
    # Eye diagram options
    show_measurements: bool = True
    show_mask: bool = True
    show_contours: bool = True
    show_mask_overlay: bool = True
    mask_overlay_alpha: float = 0.3
    mask_overlay_color: str = 'red'    
    
    # Interactive options
    enable_zoom: bool = True
    enable_pan: bool = True
    show_controls: bool = True
    
    # Color scheme
    background_color: str = 'white'
    grid_color: str = 'lightgray'
    trace_color: str = 'blue'
    mask_color: str = 'red'
    pass_color: str = 'green'
    fail_color: str = 'red'


class EyeDiagramDashboard:
    """
    Interactive Jupyter dashboard for eye diagram visualization
    
    Provides comprehensive eye diagram analysis with interactive controls,
    pass/fail annotations, and integration with SVF analyzers.
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.current_data = None
        self.current_results = None
        self.widgets = {}
        
        # Check dependencies
        if not JUPYTER_AVAILABLE:
            warnings.warn("Jupyter widgets not available. Dashboard functionality limited.", stacklevel=2)
        if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
            raise ImportError("Neither matplotlib nor plotly available for visualization")
        
        # Initialize notebook mode for plotly
        if PLOTLY_AVAILABLE and JUPYTER_AVAILABLE:
            init_notebook_mode(connected=True)
    
    def load_waveform_data(self, 
                          signal_data: Union[np.ndarray, Dict[str, np.ndarray]], 
                          time_data: Optional[np.ndarray] = None,
                          sample_rate: float = 40e9,
                          protocol: str = "USB4") -> None:
        """
        Load waveform data for analysis
        
        Args:
            signal_data: Signal voltage data (single array or multi-lane dict)
            time_data: Time axis data (optional, will be generated if not provided)
            sample_rate: Sampling rate in Hz
            protocol: Protocol type for analysis
        """
        # Handle different signal data formats
        if isinstance(signal_data, dict):
            # Multi-lane format (USB4)
            self.signal_data = signal_data
            signal_length = len(next(iter(signal_data.values())))
        else:
            # Single signal array
            self.signal_data = {'lane_0': signal_data}
            signal_length = len(signal_data)
        
        # Generate time data if not provided
        if time_data is None:
            self.time_data = np.linspace(0, signal_length / sample_rate, signal_length)
        else:
            self.time_data = time_data
        
        self.sample_rate = sample_rate
        self.protocol = protocol
        self.current_data = {
            'signal': self.signal_data,
            'time': self.time_data,
            'sample_rate': sample_rate,
            'protocol': protocol
        }
        
        print(f"✅ Loaded {protocol} waveform data: {signal_length} samples at {sample_rate/1e9:.1f} GSa/s")    
   
    def analyze_eye_diagram(self, lane: Union[int, str] = 0) -> Dict[str, Any]:
        """
        Analyze eye diagram for specified lane
        
        Args:
            lane: Lane number or key to analyze
            
        Returns:
            Analysis results dictionary
        """
        if self.current_data is None:
            raise ValueError("No waveform data loaded. Call load_waveform_data() first.")
        
        # Get signal data for specified lane
        if isinstance(lane, int):
            lane_key = f'lane_{lane}' if f'lane_{lane}' in self.signal_data else list(self.signal_data.keys())[lane]
        else:
            lane_key = lane
        
        if lane_key not in self.signal_data:
            raise ValueError(f"Lane {lane} not found in signal data")
        
        signal = self.signal_data[lane_key]
        
        # Perform eye diagram analysis
        if EYE_ANALYSIS_AVAILABLE:
            # Use advanced analyzer
            try:
                # Determine data rate based on protocol
                data_rates = {
                    'USB4': 20e9,
                    'PCIe': 32e9,
                    'Ethernet': 112e9
                }
                data_rate = data_rates.get(self.protocol, 20e9)
                
                required_sample_rate = data_rate * 4  # At least 4 samples per symbol
                actual_sample_rate = max(self.sample_rate, required_sample_rate)
                samples_per_symbol = max(2, int(actual_sample_rate / data_rate))
                
                params = EyeParameters(
                    symbol_rate=data_rate,
                    samples_per_symbol=samples_per_symbol,
                    eye_samples=1000,
                    confidence_level=0.95,
                    jitter_analysis=True
                )
                
                analyzer = AdvancedEyeAnalyzer(params)
                eye_results = analyzer.analyze_eye_diagram(self.time_data, signal)
                
                # Convert to dashboard format
                results = {
                    'eye_height': eye_results.eye_height,
                    'eye_width': eye_results.eye_width,
                    'eye_area': eye_results.eye_area,
                    'q_factor': eye_results.q_factor,
                    'snr': 20 * np.log10(eye_results.q_factor) if eye_results.q_factor > 0 else 0,
                    'jitter_analysis': eye_results.jitter_analysis,
                    'eye_diagram': eye_results.eye_diagram,
                    'time_axis': eye_results.time_axis,
                    'voltage_axis': eye_results.voltage_axis,
                    'passed': eye_results.q_factor > 6.0,  # Basic pass/fail criteria
                    'lane': lane_key,
                    'protocol': self.protocol
                }
                
                # Add eye mask analysis if available
                if EYE_MASKS_AVAILABLE and self.config.show_mask:
                    try:
                        mask_analyzer = create_eye_mask_analyzer(self.protocol, data_rate/1e9)
                        mask_result = mask_analyzer.analyze_eye_against_mask(
                            eye_results.eye_diagram,
                            eye_results.time_axis,
                            eye_results.voltage_axis
                        )
                        results['mask_analysis'] = {
                            'mask_passed': mask_result.mask_passed,
                            'violations': len(mask_result.violations),
                            'margin_percentage': mask_result.margin_percentage,
                            'compliance_level': mask_result.compliance_level,
                            'eye_opening_percentage': mask_result.eye_opening_percentage,
                            'mask_result': mask_result
                        }
                        # Update overall pass status based on mask
                        results['passed'] = results['passed'] and mask_result.mask_passed
                    except Exception as e:
                        print(f"Mask analysis failed: {e}")
                        results['mask_analysis'] = None
                
            except Exception as e:
                print(f"Advanced analysis failed: {e}")
                results = self._simple_eye_analysis(signal, lane_key)
        else:
            # Use simple analysis
            results = self._simple_eye_analysis(signal, lane_key)
        
        self.current_results = results
        return results    

    def _simple_eye_analysis(self, signal: np.ndarray, lane: str) -> Dict[str, Any]:
        """Simple eye diagram analysis fallback"""
        # Basic eye diagram metrics
        signal_range = np.max(signal) - np.min(signal)
        eye_height = signal_range * 0.8  # Estimate
        eye_width = 0.7  # Normalized
        snr = 20 * np.log10(np.mean(signal) / np.std(signal)) if np.std(signal) > 0 else 10
        
        results = {
            'eye_height': eye_height,
            'eye_width': eye_width,
            'eye_area': eye_height * eye_width,
            'q_factor': 10 ** (snr / 20),
            'snr': snr,
            'jitter_analysis': None,
            'eye_diagram': None,
            'time_axis': self.time_data,
            'voltage_axis': signal,
            'passed': snr > 15.0,  # Basic pass/fail
            'lane': lane,
            'protocol': self.protocol
        }
        
        # Add basic mask compliance check
        if EYE_MASKS_AVAILABLE and self.config.show_mask:
            try:
                # Create simple eye diagram for mask analysis
                eye_diagram = np.outer(signal, np.ones(len(self.time_data)))
                time_axis = np.linspace(-0.5, 0.5, len(self.time_data))
                voltage_axis = np.linspace(np.min(signal), np.max(signal), len(signal))
                
                mask_analyzer = create_eye_mask_analyzer(self.protocol)
                mask_result = mask_analyzer.analyze_eye_against_mask(
                    eye_diagram, time_axis, voltage_axis * 1000  # Convert to mV
                )
                
                results['mask_analysis'] = {
                    'mask_passed': mask_result.mask_passed,
                    'violations': len(mask_result.violations),
                    'margin_percentage': mask_result.margin_percentage,
                    'compliance_level': mask_result.compliance_level,
                    'eye_opening_percentage': mask_result.eye_opening_percentage,
                    'mask_result': mask_result
                }
                # Update overall pass status
                results['passed'] = results['passed'] and mask_result.mask_passed
            except Exception as e:
                print(f"Simple mask analysis failed: {e}")
                results['mask_analysis'] = None
        
        return results
    
    def create_interactive_dashboard(self) -> None:
        """Create interactive Jupyter dashboard with controls"""
        if not JUPYTER_AVAILABLE:
            print("❌ Jupyter widgets not available. Use create_static_dashboard() instead.")
            return
        
        if self.current_data is None:
            print("❌ No data loaded. Call load_waveform_data() first.")
            return
        
        # Create control widgets
        self._create_control_widgets()
        
        # Create initial plots
        self._update_dashboard()
        
        print("✅ Interactive dashboard created! Use the controls above to explore the data.")
    
    def _create_control_widgets(self):
        """Create interactive control widgets"""
        # Lane selection
        lane_options = list(self.signal_data.keys())
        self.widgets['lane_selector'] = widgets.Dropdown(
            options=lane_options,
            value=lane_options[0],
            description='Lane:',
            style={'description_width': 'initial'}
        )
        
        # Analysis options
        self.widgets['show_measurements'] = widgets.Checkbox(
            value=self.config.show_measurements,
            description='Show Measurements'
        )
        
        self.widgets['show_mask'] = widgets.Checkbox(
            value=self.config.show_mask,
            description='Show Eye Mask'
        )
        
        # Update button
        self.widgets['update_button'] = widgets.Button(
            description='Update Analysis',
            button_style='primary'
        )
        
        # Bind events
        self.widgets['update_button'].on_click(self._on_update_click)
        
        # Display controls
        controls = widgets.HBox([
            self.widgets['lane_selector'],
            self.widgets['show_measurements'],
            self.widgets['show_mask'],
            self.widgets['update_button']
        ])
        
        display(controls)    

    def _on_update_click(self, button):
        """Handle update button click"""
        with self.widgets.get('output', widgets.Output()):
            clear_output(wait=True)
            self._update_dashboard()
    
    def _update_dashboard(self):
        """Update dashboard with current settings"""
        # Get current lane
        current_lane = self.widgets.get('lane_selector', {}).value or list(self.signal_data.keys())[0]
        
        # Analyze current lane
        results = self.analyze_eye_diagram(current_lane)
        
        # Create plots based on available libraries
        if PLOTLY_AVAILABLE:
            self._create_plotly_dashboard(results)
        elif MATPLOTLIB_AVAILABLE:
            self._create_matplotlib_dashboard(results)
    
    def _create_plotly_dashboard(self, results: Dict[str, Any]):
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('Waveform', 'Eye Diagram', 'Measurements', 'Pass/Fail Status'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Waveform plot
        signal = self.signal_data[results['lane']]
        fig.add_trace(
            go.Scatter(x=self.time_data*1e9, y=signal, name='Signal', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Eye diagram (if available)
        if results['eye_diagram'] is not None:
            fig.add_trace(
                go.Heatmap(
                    z=results['eye_diagram'],
                    x=results['time_axis']*1e12,
                    y=results['voltage_axis'],
                    colorscale='Blues',
                    name='Eye Diagram'
                ),
                row=1, col=2
            )
        else:
            # Simple eye diagram simulation
            eye_signal = self._simulate_eye_pattern(signal)
            fig.add_trace(
                go.Scatter(x=np.linspace(-0.5, 0.5, len(eye_signal)), y=eye_signal, 
                          name='Eye Pattern', line=dict(color='green')),
                row=1, col=2
            )
        
        # Measurements table
        measurements_text = f"""
        Eye Height: {results['eye_height']:.4f} V<br>
        Eye Width: {results['eye_width']:.4f} UI<br>
        SNR: {results['snr']:.2f} dB<br>
        Q-Factor: {results['q_factor']:.2f}<br>
        Protocol: {results['protocol']}<br>
        Lane: {results['lane']}
        """
        
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='text', 
                      text=[measurements_text], 
                      textposition='middle center',
                      showlegend=False),
            row=2, col=1
        )
        
        # Pass/Fail status
        status_color = 'green' if results['passed'] else 'red'
        status_text = 'PASS' if results['passed'] else 'FAIL'
        
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='markers+text',
                      marker=dict(size=100, color=status_color),
                      text=[status_text],
                      textfont=dict(size=20, color='white'),
                      showlegend=False),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text=f"SerDes Eye Diagram Dashboard - {results['protocol']} {results['lane']}",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (ns)", row=1, col=1)
        fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
        fig.update_xaxes(title_text="Time (ps)", row=1, col=2)
        fig.update_yaxes(title_text="Voltage (V)", row=1, col=2)
        
        # Display
        iplot(fig) 
   
    def _create_matplotlib_dashboard(self, results: Dict[str, Any]):
        """Create static matplotlib dashboard"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Waveform plot
        signal = self.signal_data[results['lane']]
        ax1.plot(self.time_data*1e9, signal, 'b-', linewidth=1)
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('Waveform')
        ax1.grid(True, alpha=0.3)
        
        # Eye diagram
        if results['eye_diagram'] is not None:
            im = ax2.imshow(results['eye_diagram'], aspect='auto', 
                           extent=[results['time_axis'][0]*1e12, results['time_axis'][-1]*1e12,
                                  results['voltage_axis'][0], results['voltage_axis'][-1]],
                           cmap='Blues')
            plt.colorbar(im, ax=ax2)
        else:
            # Simple eye pattern
            eye_signal = self._simulate_eye_pattern(signal)
            ax2.plot(np.linspace(-0.5, 0.5, len(eye_signal)), eye_signal, 'g-')
        
        ax2.set_xlabel('Time (ps)')
        ax2.set_ylabel('Voltage (V)')
        ax2.set_title('Eye Diagram')
        ax2.grid(True, alpha=0.3)
        
        # Measurements
        measurements = [
            f"Eye Height: {results['eye_height']:.4f} V",
            f"Eye Width: {results['eye_width']:.4f} UI",
            f"SNR: {results['snr']:.2f} dB",
            f"Q-Factor: {results['q_factor']:.2f}",
            f"Protocol: {results['protocol']}",
            f"Lane: {results['lane']}"
        ]
        
        ax3.text(0.1, 0.9, '\n'.join(measurements), transform=ax3.transAxes,
                verticalalignment='top', fontsize=10, fontfamily='monospace')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_title('Measurements')
        ax3.axis('off')
        
        # Pass/Fail status
        status_color = 'green' if results['passed'] else 'red'
        status_text = 'PASS' if results['passed'] else 'FAIL'
        
        circle = patches.Circle((0.5, 0.5), 0.3, color=status_color, alpha=0.7)
        ax4.add_patch(circle)
        ax4.text(0.5, 0.5, status_text, ha='center', va='center', 
                fontsize=16, fontweight='bold', color='white')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('Status')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _simulate_eye_pattern(self, signal: np.ndarray, symbols_per_eye: int = 100) -> np.ndarray:
        """Simulate eye pattern from signal data"""
        # Simple eye pattern simulation
        symbol_length = len(signal) // symbols_per_eye
        if symbol_length < 10:
            return signal[:200] if len(signal) > 200 else signal
        
        eye_pattern = []
        for i in range(0, len(signal) - symbol_length, symbol_length):
            eye_pattern.extend(signal[i:i+symbol_length])
        
        return np.array(eye_pattern[:symbol_length*2])  # Two symbol periods    

    def create_static_dashboard(self, lane: Union[int, str] = 0) -> None:
        """Create static dashboard without interactive controls"""
        if self.current_data is None:
            print("❌ No data loaded. Call load_waveform_data() first.")
            return
        
        # Analyze specified lane
        results = self.analyze_eye_diagram(lane)
        
        # Create static plots
        if MATPLOTLIB_AVAILABLE:
            self._create_matplotlib_dashboard(results)
        else:
            print("❌ Matplotlib not available for static dashboard")
    
    def export_results(self, filename: str = "eye_analysis_results.json") -> None:
        """Export analysis results to file"""
        if self.current_results is None:
            print("❌ No analysis results to export. Run analyze_eye_diagram() first.")
            return
        
        import json
        
        # Prepare results for JSON export (handle numpy arrays)
        export_data = {}
        for key, value in self.current_results.items():
            if isinstance(value, np.ndarray):
                export_data[key] = value.tolist()
            elif hasattr(value, '__dict__'):  # Handle complex objects
                export_data[key] = str(value)
            else:
                export_data[key] = value
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Results exported to {filename}")


def create_dashboard(signal_data: Union[np.ndarray, Dict[str, np.ndarray]], 
                    sample_rate: float = 40e9,
                    protocol: str = "USB4",
                    config: Optional[DashboardConfig] = None) -> EyeDiagramDashboard:
    """
    Create and initialize eye diagram dashboard
    
    Args:
        signal_data: Signal voltage data
        sample_rate: Sampling rate in Hz
        protocol: Protocol type
        config: Dashboard configuration
        
    Returns:
        Initialized dashboard instance
    """
    dashboard = EyeDiagramDashboard(config)
    dashboard.load_waveform_data(signal_data, sample_rate=sample_rate, protocol=protocol)
    return dashboard


# Utility functions for Jupyter notebook integration
def display_requirements():
    """Display requirements for Jupyter dashboard"""
    requirements = """
    📋 Jupyter Dashboard Requirements:
    
    Required packages:
    - jupyter or jupyterlab
    - matplotlib >= 3.5.0
    - plotly >= 5.0.0 (for interactive plots)
    - ipywidgets >= 7.6.0 (for interactive controls)
    
    Installation:
    pip install jupyter matplotlib plotly ipywidgets
    
    For JupyterLab:
    pip install jupyterlab
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    """
    
    if JUPYTER_AVAILABLE:
        display(HTML(f"<pre>{requirements}</pre>"))
    else:
        print(requirements)


def check_dashboard_dependencies():
    """Check and report dashboard dependencies"""
    status = {
        'Jupyter': JUPYTER_AVAILABLE,
        'Matplotlib': MATPLOTLIB_AVAILABLE, 
        'Plotly': PLOTLY_AVAILABLE,
        'Eye Analysis': EYE_ANALYSIS_AVAILABLE
    }
    
    print("📊 Dashboard Dependencies Status:")
    for name, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {name}: {'Available' if available else 'Not Available'}")
    
    if not any(status.values()):
        print("\n⚠️  No visualization libraries available. Install matplotlib or plotly.")
    elif not JUPYTER_AVAILABLE:
        print("\n⚠️  Jupyter not available. Interactive features will be limited.")
    
    return status
