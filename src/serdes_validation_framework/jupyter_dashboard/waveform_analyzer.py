"""
Waveform Analyzer for Jupyter Dashboard

Provides waveform analysis capabilities specifically designed for
Jupyter notebook integration with interactive features.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class WaveformAnalysisResult:
    """Results from waveform analysis"""
    
    # Basic statistics
    mean_voltage: float
    rms_voltage: float
    peak_to_peak: float
    std_deviation: float
    
    # Signal quality metrics
    snr_db: float
    thd_percent: float
    dynamic_range: float
    
    # Timing metrics
    rise_time: float
    fall_time: float
    pulse_width: float
    
    # Pass/fail status
    passed: bool
    failure_reasons: List[str]
    
    # Raw data
    time_data: np.ndarray
    voltage_data: np.ndarray
    
    # Analysis metadata
    sample_rate: float
    protocol: str
    lane: str


class WaveformAnalyzer:
    """
    Waveform analyzer for Jupyter dashboard integration
    
    Provides comprehensive waveform analysis with visualization
    capabilities optimized for Jupyter notebooks.
    """
    
    def __init__(self, sample_rate: float = 40e9, protocol: str = "USB4"):
        self.sample_rate = sample_rate
        self.protocol = protocol
        self.analysis_results = {}
    
    def analyze_waveform(self, 
                        voltage_data: np.ndarray,
                        time_data: Optional[np.ndarray] = None,
                        lane: str = "lane_0") -> WaveformAnalysisResult:
        """
        Perform comprehensive waveform analysis
        
        Args:
            voltage_data: Voltage samples
            time_data: Time samples (optional)
            lane: Lane identifier
            
        Returns:
            Analysis results
        """
        # Generate time data if not provided
        if time_data is None:
            time_data = np.linspace(0, len(voltage_data) / self.sample_rate, len(voltage_data))
        
        # Basic statistics
        mean_voltage = np.mean(voltage_data)
        rms_voltage = np.sqrt(np.mean(voltage_data**2))
        peak_to_peak = np.max(voltage_data) - np.min(voltage_data)
        std_deviation = np.std(voltage_data)
        
        # Signal quality metrics
        signal_power = mean_voltage**2
        noise_power = std_deviation**2
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 40.0
        
        # THD calculation (simplified)
        thd_percent = self._calculate_thd(voltage_data)
        
        # Dynamic range
        dynamic_range = 20 * np.log10(peak_to_peak / (2 * std_deviation)) if std_deviation > 0 else 60.0
        
        # Timing analysis
        rise_time, fall_time, pulse_width = self._analyze_timing(voltage_data, time_data)
        
        # Pass/fail evaluation
        passed, failure_reasons = self._evaluate_pass_fail(
            snr_db, thd_percent, dynamic_range, peak_to_peak
        )
        
        # Create result object
        result = WaveformAnalysisResult(
            mean_voltage=mean_voltage,
            rms_voltage=rms_voltage,
            peak_to_peak=peak_to_peak,
            std_deviation=std_deviation,
            snr_db=snr_db,
            thd_percent=thd_percent,
            dynamic_range=dynamic_range,
            rise_time=rise_time,
            fall_time=fall_time,
            pulse_width=pulse_width,
            passed=passed,
            failure_reasons=failure_reasons,
            time_data=time_data,
            voltage_data=voltage_data,
            sample_rate=self.sample_rate,
            protocol=self.protocol,
            lane=lane
        )
        
        # Store result
        self.analysis_results[lane] = result
        
        return result
    
    def _calculate_thd(self, voltage_data: np.ndarray) -> float:
        """Calculate Total Harmonic Distortion (simplified)"""
        try:
            # Simple THD estimation using FFT
            fft_data = np.fft.fft(voltage_data)
            power_spectrum = np.abs(fft_data)**2
            
            # Find fundamental frequency (peak in spectrum)
            fundamental_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            fundamental_power = power_spectrum[fundamental_idx]
            
            # Calculate harmonic power (simplified)
            harmonic_power = np.sum(power_spectrum[2*fundamental_idx::fundamental_idx])
            
            thd = np.sqrt(harmonic_power / fundamental_power) * 100
            return min(thd, 50.0)  # Cap at 50%
            
        except Exception:
            return 5.0  # Default reasonable value
    
    def _analyze_timing(self, voltage_data: np.ndarray, time_data: np.ndarray) -> Tuple[float, float, float]:
        """Analyze timing characteristics"""
        try:
            # Find transitions
            diff_data = np.diff(voltage_data)
            
            # Rise time (10% to 90% of peak)
            peak_value = np.max(voltage_data)
            low_threshold = 0.1 * peak_value
            high_threshold = 0.9 * peak_value
            
            # Find rising edges
            rising_edges = np.where(diff_data > np.std(diff_data))[0]
            if len(rising_edges) > 0:
                # Estimate rise time
                dt = time_data[1] - time_data[0]
                rise_time = dt * 10  # Simplified estimation
            else:
                rise_time = 0.0
            
            # Fall time (similar approach)
            falling_edges = np.where(diff_data < -np.std(diff_data))[0]
            if len(falling_edges) > 0:
                fall_time = dt * 10  # Simplified estimation
            else:
                fall_time = 0.0
            
            # Pulse width (time between rising and falling edges)
            if len(rising_edges) > 0 and len(falling_edges) > 0:
                pulse_width = abs(time_data[falling_edges[0]] - time_data[rising_edges[0]])
            else:
                pulse_width = 0.0
            
            return rise_time, fall_time, pulse_width
            
        except Exception:
            return 0.0, 0.0, 0.0
    
    def _evaluate_pass_fail(self, snr_db: float, thd_percent: float, 
                           dynamic_range: float, peak_to_peak: float) -> Tuple[bool, List[str]]:
        """Evaluate pass/fail criteria"""
        failure_reasons = []
        
        # Protocol-specific thresholds
        thresholds = {
            'USB4': {'snr_min': 15.0, 'thd_max': 10.0, 'dr_min': 40.0, 'pp_min': 0.4},
            'PCIe': {'snr_min': 18.0, 'thd_max': 8.0, 'dr_min': 45.0, 'pp_min': 0.6},
            'Ethernet': {'snr_min': 20.0, 'thd_max': 6.0, 'dr_min': 50.0, 'pp_min': 0.5}
        }
        
        thresh = thresholds.get(self.protocol, thresholds['USB4'])
        
        # Check criteria
        if snr_db < thresh['snr_min']:
            failure_reasons.append(f"SNR too low: {snr_db:.1f} dB < {thresh['snr_min']} dB")
        
        if thd_percent > thresh['thd_max']:
            failure_reasons.append(f"THD too high: {thd_percent:.1f}% > {thresh['thd_max']}%")
        
        if dynamic_range < thresh['dr_min']:
            failure_reasons.append(f"Dynamic range too low: {dynamic_range:.1f} dB < {thresh['dr_min']} dB")
        
        if peak_to_peak < thresh['pp_min']:
            failure_reasons.append(f"Peak-to-peak too low: {peak_to_peak:.3f} V < {thresh['pp_min']} V")
        
        passed = len(failure_reasons) == 0
        
        return passed, failure_reasons
    
    def create_analysis_plot(self, lane: str = "lane_0", plot_type: str = "matplotlib") -> None:
        """
        Create analysis plot for specified lane
        
        Args:
            lane: Lane to plot
            plot_type: 'matplotlib' or 'plotly'
        """
        if lane not in self.analysis_results:
            print(f"❌ No analysis results for lane {lane}. Run analyze_waveform() first.")
            return
        
        result = self.analysis_results[lane]
        
        if plot_type == "plotly" and PLOTLY_AVAILABLE:
            self._create_plotly_analysis(result)
        elif plot_type == "matplotlib" and MATPLOTLIB_AVAILABLE:
            self._create_matplotlib_analysis(result)
        else:
            print(f"❌ {plot_type} not available for plotting")
    
    def _create_plotly_analysis(self, result: WaveformAnalysisResult):
        """Create Plotly analysis plot"""
        # Create subplots
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('Waveform', 'FFT Spectrum', 'Statistics', 'Pass/Fail'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Waveform plot
        fig.add_trace(
            go.Scatter(x=result.time_data*1e9, y=result.voltage_data, 
                      name='Waveform', line=dict(color='blue')),
            row=1, col=1
        )
        
        # FFT spectrum
        fft_data = np.fft.fft(result.voltage_data)
        freqs = np.fft.fftfreq(len(result.voltage_data), 1/result.sample_rate)
        power_spectrum = 20 * np.log10(np.abs(fft_data))
        
        # Plot positive frequencies only
        pos_freqs = freqs[:len(freqs)//2]
        pos_spectrum = power_spectrum[:len(power_spectrum)//2]
        
        fig.add_trace(
            go.Scatter(x=pos_freqs/1e9, y=pos_spectrum, 
                      name='Spectrum', line=dict(color='red')),
            row=1, col=2
        )
        
        # Statistics
        stats_text = f"""
        Mean: {result.mean_voltage:.4f} V<br>
        RMS: {result.rms_voltage:.4f} V<br>
        Peak-to-Peak: {result.peak_to_peak:.4f} V<br>
        SNR: {result.snr_db:.2f} dB<br>
        THD: {result.thd_percent:.2f}%<br>
        Dynamic Range: {result.dynamic_range:.2f} dB
        """
        
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='text', 
                      text=[stats_text], 
                      textposition='middle center',
                      showlegend=False),
            row=2, col=1
        )
        
        # Pass/Fail status
        status_color = 'green' if result.passed else 'red'
        status_text = 'PASS' if result.passed else 'FAIL'
        
        if not result.passed:
            status_text += '<br>' + '<br>'.join(result.failure_reasons)
        
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='markers+text',
                      marker=dict(size=100, color=status_color),
                      text=[status_text],
                      textfont=dict(size=16, color='white'),
                      showlegend=False),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text=f"Waveform Analysis - {result.protocol} {result.lane}",
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (ns)", row=1, col=1)
        fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (GHz)", row=1, col=2)
        fig.update_yaxes(title_text="Power (dB)", row=1, col=2)
        
        fig.show()
    
    def _create_matplotlib_analysis(self, result: WaveformAnalysisResult):
        """Create matplotlib analysis plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Waveform plot
        ax1.plot(result.time_data*1e9, result.voltage_data, 'b-', linewidth=1)
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('Waveform')
        ax1.grid(True, alpha=0.3)
        
        # FFT spectrum
        fft_data = np.fft.fft(result.voltage_data)
        freqs = np.fft.fftfreq(len(result.voltage_data), 1/result.sample_rate)
        power_spectrum = 20 * np.log10(np.abs(fft_data))
        
        # Plot positive frequencies only
        pos_freqs = freqs[:len(freqs)//2]
        pos_spectrum = power_spectrum[:len(power_spectrum)//2]
        
        ax2.plot(pos_freqs/1e9, pos_spectrum, 'r-', linewidth=1)
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('Power (dB)')
        ax2.set_title('FFT Spectrum')
        ax2.grid(True, alpha=0.3)
        
        # Statistics
        stats = [
            f"Mean: {result.mean_voltage:.4f} V",
            f"RMS: {result.rms_voltage:.4f} V",
            f"Peak-to-Peak: {result.peak_to_peak:.4f} V",
            f"SNR: {result.snr_db:.2f} dB",
            f"THD: {result.thd_percent:.2f}%",
            f"Dynamic Range: {result.dynamic_range:.2f} dB"
        ]
        
        ax3.text(0.1, 0.9, '\n'.join(stats), transform=ax3.transAxes,
                verticalalignment='top', fontsize=10, fontfamily='monospace')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_title('Statistics')
        ax3.axis('off')
        
        # Pass/Fail status
        status_color = 'green' if result.passed else 'red'
        status_text = 'PASS' if result.passed else 'FAIL'
        
        circle = plt.Circle((0.5, 0.5), 0.3, color=status_color, alpha=0.7)
        ax4.add_patch(circle)
        ax4.text(0.5, 0.5, status_text, ha='center', va='center', 
                fontsize=16, fontweight='bold', color='white')
        
        if not result.passed:
            failure_text = '\n'.join(result.failure_reasons)
            ax4.text(0.5, 0.1, failure_text, ha='center', va='bottom',
                    fontsize=8, transform=ax4.transAxes, wrap=True)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('Status')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_summary_report(self) -> str:
        """Generate summary report for all analyzed lanes"""
        if not self.analysis_results:
            return "❌ No analysis results available"
        
        report = f"📊 Waveform Analysis Summary - {self.protocol}\n"
        report += "=" * 50 + "\n\n"
        
        for lane, result in self.analysis_results.items():
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report += f"{lane}: {status}\n"
            report += f"  SNR: {result.snr_db:.2f} dB\n"
            report += f"  Peak-to-Peak: {result.peak_to_peak:.4f} V\n"
            report += f"  THD: {result.thd_percent:.2f}%\n"
            
            if not result.passed:
                report += "  Failures:\n"
                for reason in result.failure_reasons:
                    report += f"    - {reason}\n"
            
            report += "\n"
        
        return report
