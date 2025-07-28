"""
Loopback Stress Test Module

This module implements loopback stress testing for SerDes validation,
simulating TX → RX → back to TX loops and tracking signal degradation
over multiple cycles.
"""

import csv
import logging

# Set matplotlib backend to non-GUI for testing environments
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if os.environ.get('SVF_MOCK_MODE') == '1' or os.environ.get('MPLBACKEND') == 'Agg':
    import matplotlib
    matplotlib.use('Agg')

try:
    from ..data_analysis.eye_diagram import AdvancedEyeAnalyzer
    EYE_ANALYSIS_AVAILABLE = True
except ImportError:
    EYE_ANALYSIS_AVAILABLE = False

try:
    from ..visualization.eye_diagram import EyeDiagramVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from ..bert_integration.bert_hooks import BERTHook, BERTHookManager, HookTrigger, HookType, create_standard_hooks
    BERT_HOOKS_AVAILABLE = True
except ImportError:
    BERT_HOOKS_AVAILABLE = False
    # Create dummy classes for when BERT hooks are not available
    class HookType:
        PRE_TEST = "PRE_TEST"
        POST_TEST = "POST_TEST"
        ON_ERROR = "ON_ERROR"
        ON_THRESHOLD = "ON_THRESHOLD"
        PERIODIC = "PERIODIC"


@dataclass
class StressTestConfig:
    """Configuration for loopback stress testing"""
    
    # Test parameters
    num_cycles: int = 1000
    cycle_duration: float = 1.0  # seconds per cycle
    sample_rate: float = 40e9  # 40 GSa/s
    signal_length: int = 10000  # samples per test
    
    # Protocol settings
    protocol: str = "USB4"  # USB4, PCIe, Ethernet
    data_rate: float = 20e9  # 20 Gbps for USB4
    voltage_swing: float = 0.8  # 800mV
    
    # Degradation thresholds
    eye_height_threshold: float = 0.1  # 10% degradation threshold
    jitter_threshold: float = 0.05  # 5% jitter increase threshold
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("stress_test_results"))
    save_waveforms: bool = False  # Save waveforms for failed cycles
    generate_plots: bool = True
    
    # Logging
    log_level: str = "INFO"
    
    # BERT hooks settings
    enable_bert_hooks: bool = False
    bert_hooks_config: Optional[str] = None  # Path to hooks config file


@dataclass
class CycleResults:
    """Results from a single stress test cycle"""
    
    cycle_number: int
    timestamp: float
    eye_height: float
    eye_width: float
    rms_jitter: float
    peak_jitter: float
    snr: float
    ber_estimate: float
    passed: bool
    degradation_percent: float = 0.0
    notes: str = ""


@dataclass
class StressTestResults:
    """Complete stress test results"""
    
    config: StressTestConfig
    start_time: float
    end_time: float
    total_cycles: int
    passed_cycles: int
    failed_cycles: int
    cycle_results: List[CycleResults] = field(default_factory=list)
    
    # Degradation tracking
    initial_eye_height: float = 0.0
    final_eye_height: float = 0.0
    max_degradation: float = 0.0
    degradation_rate: float = 0.0  # per cycle
    
    # Summary statistics
    mean_eye_height: float = 0.0
    std_eye_height: float = 0.0
    mean_jitter: float = 0.0
    std_jitter: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate"""
        if self.total_cycles == 0:
            return 0.0
        return self.passed_cycles / self.total_cycles
    
    @property
    def duration(self) -> float:
        """Test duration in seconds"""
        return self.end_time - self.start_time


class SimpleEyeAnalyzer:
    """Simple eye analyzer for stress testing"""
    
    def analyze_eye_diagram(self, signal, sample_rate, symbol_rate=None):
        """Simple eye diagram analysis for stress testing"""
        try:
            # Basic eye diagram metrics calculation
            signal_std = np.std(signal)
            signal_mean = np.mean(signal)
            signal_range = np.max(signal) - np.min(signal)
            
            # Estimate eye height (simplified)
            eye_height = signal_range * 0.8  # Assume 80% of signal range
            
            # Estimate eye width (simplified)
            eye_width = 0.7  # Normalized eye width
            
            # Estimate jitter (simplified)
            rms_jitter = signal_std / signal_range * 0.1  # Simplified jitter calculation
            peak_jitter = rms_jitter * 3  # Peak is ~3x RMS
            
            # Estimate SNR (simplified)
            signal_power = signal_mean ** 2
            noise_power = signal_std ** 2
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 20.0
            
            # Estimate BER (simplified)
            ber_estimate = max(1e-15, min(1e-3, 10 ** (-snr / 10)))
            
            # Return results in a simple object
            class EyeResults:
                def __init__(self):
                    self.eye_height = eye_height
                    self.eye_width = eye_width
                    self.rms_jitter = rms_jitter
                    self.peak_jitter = peak_jitter
                    self.snr = snr
                    self.ber_estimate = ber_estimate
            
            return EyeResults()
            
        except Exception as e:
            # Return default failed results
            class FailedResults:
                def __init__(self):
                    self.eye_height = 0.0
                    self.eye_width = 0.0
                    self.rms_jitter = 1.0
                    self.peak_jitter = 1.0
                    self.snr = 0.0
                    self.ber_estimate = 1.0
            
            return FailedResults()


class LoopbackStressTest:
    """
    Loopback Stress Test Implementation
    
    Simulates SerDes loopback (TX → RX → back to TX) and tracks
    signal degradation over multiple test cycles.
    """
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        
        # Create output directory first
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging after directory creation
        self.logger = self._setup_logging()
        
        # Initialize analyzers
        if EYE_ANALYSIS_AVAILABLE:
            # Use advanced analyzer if available
            from ..data_analysis.eye_diagram import EyeParameters
            
            # Ensure adequate sampling rate for the data rate
            required_sample_rate = config.data_rate * 4  # At least 4 samples per symbol
            actual_sample_rate = max(config.sample_rate, required_sample_rate)
            samples_per_symbol = max(2, int(actual_sample_rate / config.data_rate))
            
            params = EyeParameters(
                symbol_rate=config.data_rate,
                samples_per_symbol=samples_per_symbol,
                eye_samples=1000,
                confidence_level=0.95,
                jitter_analysis=True
            )
            self.eye_analyzer = AdvancedEyeAnalyzer(params)
        else:
            # Use simple analyzer for stress testing
            self.eye_analyzer = SimpleEyeAnalyzer()
        
        self.visualizer = None
        if config.generate_plots and VISUALIZATION_AVAILABLE:
            self.visualizer = EyeDiagramVisualizer()
        
        # Initialize CSV logger
        self.csv_file = self.config.output_dir / "stress_test_results.csv"
        self._init_csv_logger()
        
        # Initialize BERT hooks if enabled
        self.bert_hook_manager = None
        if config.enable_bert_hooks and BERT_HOOKS_AVAILABLE:
            self._init_bert_hooks()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for stress test"""
        logger = logging.getLogger("LoopbackStressTest")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Create file handler
        log_file = self.config.output_dir / "stress_test.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(getattr(logging, self.config.log_level))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _init_csv_logger(self):
        """Initialize CSV file for results logging"""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'cycle_number', 'timestamp', 'eye_height', 'eye_width',
                'rms_jitter', 'peak_jitter', 'snr', 'ber_estimate',
                'passed', 'degradation_percent', 'notes'
            ])
    
    def _log_cycle_result(self, result: CycleResults):
        """Log cycle result to CSV file"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.cycle_number, result.timestamp, result.eye_height,
                result.eye_width, result.rms_jitter, result.peak_jitter,
                result.snr, result.ber_estimate, result.passed,
                result.degradation_percent, result.notes
            ])
    
    def _generate_loopback_signal(self, cycle: int) -> np.ndarray:
        """
        Generate simulated loopback signal with progressive degradation
        
        Args:
            cycle: Current test cycle number
            
        Returns:
            Simulated signal data with degradation effects
        """
        # Base signal generation
        t = np.linspace(0, self.config.signal_length / self.config.sample_rate, 
                       self.config.signal_length)
        
        # Generate base data pattern (PRBS-like)
        np.random.seed(42 + cycle)  # Reproducible but varying
        data_bits = np.random.choice([-1, 1], size=len(t))
        
        # Apply protocol-specific characteristics
        if self.config.protocol == "USB4":
            # NRZ signaling
            signal = data_bits * self.config.voltage_swing / 2
        elif self.config.protocol == "PCIe":
            # PAM4 signaling (simplified)
            pam4_levels = np.random.choice([-3, -1, 1, 3], size=len(t))
            signal = pam4_levels * self.config.voltage_swing / 6
        else:  # Ethernet
            # PAM4 signaling
            pam4_levels = np.random.choice([-3, -1, 1, 3], size=len(t))
            signal = pam4_levels * self.config.voltage_swing / 6
        
        # Add progressive degradation effects
        degradation_factor = cycle / self.config.num_cycles
        
        # 1. Amplitude degradation (signal attenuation)
        amplitude_loss = 1.0 - (degradation_factor * 0.1)  # Up to 10% loss
        signal *= amplitude_loss
        
        # 2. Jitter increase (timing degradation)
        jitter_std = 0.01 * (1 + degradation_factor * 2)  # Jitter increases
        timing_jitter = np.random.normal(0, jitter_std, len(t))
        
        # Apply timing jitter by shifting samples
        jittered_indices = np.clip(
            np.arange(len(t)) + timing_jitter * self.config.sample_rate,
            0, len(t) - 1
        ).astype(int)
        signal = signal[jittered_indices]
        
        # 3. Noise increase
        noise_level = 0.02 * (1 + degradation_factor)  # Noise increases
        noise = np.random.normal(0, noise_level, len(signal))
        signal += noise
        
        # 4. ISI (Inter-Symbol Interference) - simplified
        if cycle > self.config.num_cycles * 0.3:  # ISI kicks in after 30% cycles
            isi_filter = np.array([0.1, 0.8, 0.1])  # Simple ISI filter
            signal = np.convolve(signal, isi_filter, mode='same')
        
        return signal
    
    def _analyze_signal(self, signal: np.ndarray, cycle: int) -> CycleResults:
        """
        Analyze signal and extract eye diagram metrics
        
        Args:
            signal: Signal data to analyze
            cycle: Current cycle number
            
        Returns:
            Analysis results for this cycle
        """
        timestamp = time.time()
        
        try:
            # Perform eye diagram analysis
            if EYE_ANALYSIS_AVAILABLE:
                # Use advanced analyzer
                t = np.linspace(0, len(signal) / self.config.sample_rate, len(signal))
                eye_results = self.eye_analyzer.analyze_eye_diagram(t, signal)
            else:
                # Use simple analyzer
                eye_results = self.eye_analyzer.analyze_eye_diagram(
                    signal, 
                    self.config.sample_rate,
                    symbol_rate=self.config.data_rate
                )
            
            # Extract metrics based on analyzer type
            eye_height = eye_results.eye_height
            eye_width = eye_results.eye_width
            
            if EYE_ANALYSIS_AVAILABLE and hasattr(eye_results, 'jitter_analysis'):
                # Advanced analyzer results
                if eye_results.jitter_analysis:
                    rms_jitter = eye_results.jitter_analysis.random_jitter
                    peak_jitter = eye_results.jitter_analysis.total_jitter
                else:
                    rms_jitter = 0.02
                    peak_jitter = 0.05
                
                # Calculate SNR from Q-factor
                q_factor = eye_results.q_factor
                snr = 20 * np.log10(q_factor) if q_factor > 0 else 10.0
                
                # Estimate BER from Q-factor
                ber_estimate = 0.5 * np.exp(-0.5 * q_factor**2) if q_factor > 0 else 1e-3
            else:
                # Simple analyzer results
                rms_jitter = eye_results.rms_jitter
                peak_jitter = eye_results.peak_jitter
                snr = eye_results.snr
                ber_estimate = eye_results.ber_estimate
            
            # Calculate degradation (compared to first cycle)
            if cycle == 1:
                self.initial_eye_height = eye_height
                degradation_percent = 0.0
            else:
                degradation_percent = (
                    (self.initial_eye_height - eye_height) / self.initial_eye_height * 100
                )
            
            # Determine if cycle passed
            eye_degraded = degradation_percent > (self.config.eye_height_threshold * 100)
            jitter_high = rms_jitter > self.config.jitter_threshold
            passed = not (eye_degraded or jitter_high)
            
            # Generate notes
            notes = []
            if eye_degraded:
                notes.append(f"Eye height degraded by {degradation_percent:.1f}%")
            if jitter_high:
                notes.append(f"High jitter: {rms_jitter:.4f}")
            
            return CycleResults(
                cycle_number=cycle,
                timestamp=timestamp,
                eye_height=eye_height,
                eye_width=eye_width,
                rms_jitter=rms_jitter,
                peak_jitter=peak_jitter,
                snr=snr,
                ber_estimate=ber_estimate,
                passed=passed,
                degradation_percent=degradation_percent,
                notes="; ".join(notes)
            )
            
        except Exception as e:
            self.logger.error(f"Analysis failed for cycle {cycle}: {e}")
            return CycleResults(
                cycle_number=cycle,
                timestamp=timestamp,
                eye_height=0.0,
                eye_width=0.0,
                rms_jitter=1.0,
                peak_jitter=1.0,
                snr=0.0,
                ber_estimate=1.0,
                passed=False,
                notes=f"Analysis error: {str(e)}"
            )
    
    def _save_waveform(self, signal: np.ndarray, cycle: int):
        """Save waveform data for failed cycles"""
        if self.config.save_waveforms:
            waveform_file = self.config.output_dir / f"waveform_cycle_{cycle:04d}.csv"
            np.savetxt(waveform_file, signal, delimiter=',')
    
    def _generate_plots(self, results: StressTestResults):
        """Generate visualization plots for stress test results"""
        if not self.visualizer:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Extract data for plotting
            cycles = [r.cycle_number for r in results.cycle_results]
            eye_heights = [r.eye_height for r in results.cycle_results]
            jitter_values = [r.rms_jitter for r in results.cycle_results]
            degradation = [r.degradation_percent for r in results.cycle_results]
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Eye Height vs Cycle
            ax1.plot(cycles, eye_heights, 'b-', linewidth=2, label='Eye Height')
            ax1.axhline(y=results.initial_eye_height * (1 - self.config.eye_height_threshold), 
                       color='r', linestyle='--', label='Threshold')
            ax1.set_xlabel('Cycle Number')
            ax1.set_ylabel('Eye Height (V)')
            ax1.set_title('Eye Height Degradation Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Jitter vs Cycle
            ax2.plot(cycles, jitter_values, 'g-', linewidth=2, label='RMS Jitter')
            ax2.axhline(y=self.config.jitter_threshold, color='r', linestyle='--', label='Threshold')
            ax2.set_xlabel('Cycle Number')
            ax2.set_ylabel('RMS Jitter (UI)')
            ax2.set_title('Jitter Increase Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Degradation Percentage
            ax3.plot(cycles, degradation, 'r-', linewidth=2, label='Degradation %')
            ax3.axhline(y=self.config.eye_height_threshold * 100, 
                       color='r', linestyle='--', label='Threshold')
            ax3.set_xlabel('Cycle Number')
            ax3.set_ylabel('Degradation (%)')
            ax3.set_title('Signal Degradation Percentage')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Pass/Fail Status
            pass_fail = [1 if r.passed else 0 for r in results.cycle_results]
            ax4.plot(cycles, pass_fail, 'ko-', markersize=3, label='Pass/Fail')
            ax4.set_xlabel('Cycle Number')
            ax4.set_ylabel('Status (1=Pass, 0=Fail)')
            ax4.set_title('Test Status Over Time')
            ax4.set_ylim(-0.1, 1.1)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.config.output_dir / "stress_test_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Analysis plots saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plots: {e}")
    
    def _init_bert_hooks(self):
        """Initialize BERT hooks system"""
        try:
            self.bert_hook_manager = BERTHookManager()
            
            # Load hooks from config file if specified
            if self.config.bert_hooks_config and Path(self.config.bert_hooks_config).exists():
                self.bert_hook_manager.import_hooks_config(self.config.bert_hooks_config)
            else:
                # Create standard hooks
                standard_hooks = create_standard_hooks()
                for hook in standard_hooks:
                    self.bert_hook_manager.register_hook(hook)
            
            self.logger.info("BERT hooks initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BERT hooks: {e}")
            self.bert_hook_manager = None
    
    def _execute_bert_hooks(self, hook_type, context: Dict[str, Any]):
        """Execute BERT hooks of specified type"""
        if self.bert_hook_manager and BERT_HOOKS_AVAILABLE:
            try:
                results = self.bert_hook_manager.execute_hooks(hook_type, context)
                for result in results:
                    if result.success:
                        self.logger.debug(f"Hook {result.hook_name} executed successfully")
                    else:
                        self.logger.warning(f"Hook {result.hook_name} failed: {result.exception}")
            except Exception as e:
                self.logger.error(f"BERT hooks execution failed: {e}")
    
    def run_stress_test(self) -> StressTestResults:
        """
        Run the complete loopback stress test
        
        Returns:
            Complete stress test results
        """
        self.logger.info(f"Starting loopback stress test with {self.config.num_cycles} cycles")
        self.logger.info(f"Protocol: {self.config.protocol}, Data Rate: {self.config.data_rate/1e9:.1f} Gbps")
        
        start_time = time.time()
        results = StressTestResults(
            config=self.config,
            start_time=start_time,
            end_time=0.0,
            total_cycles=self.config.num_cycles,
            passed_cycles=0,
            failed_cycles=0
        )
        
        # Execute pre-test hooks
        if self.bert_hook_manager:
            pre_test_context = {
                'test_type': 'loopback_stress',
                'protocol': self.config.protocol,
                'data_rate': self.config.data_rate,
                'num_cycles': self.config.num_cycles,
                'start_time': start_time
            }
            self._execute_bert_hooks(HookType.PRE_TEST, pre_test_context)
        
        try:
            for cycle in range(1, self.config.num_cycles + 1):
                cycle_start = time.time()
                
                # Generate loopback signal with degradation
                signal = self._generate_loopback_signal(cycle)
                
                # Analyze signal
                cycle_result = self._analyze_signal(signal, cycle)
                
                # Update counters
                if cycle_result.passed:
                    results.passed_cycles += 1
                else:
                    results.failed_cycles += 1
                    # Save waveform for failed cycles
                    if self.config.save_waveforms:
                        self._save_waveform(signal, cycle)
                
                # Store result
                results.cycle_results.append(cycle_result)
                
                # Log to CSV
                self._log_cycle_result(cycle_result)
                
                # Execute BERT hooks based on cycle results
                if self.bert_hook_manager:
                    cycle_context = {
                        'cycle': cycle,
                        'ber': cycle_result.ber_estimate,
                        'error_count': int(cycle_result.ber_estimate * self.config.signal_length),
                        'eye_height': cycle_result.eye_height,
                        'jitter': cycle_result.rms_jitter,
                        'degradation': cycle_result.degradation_percent,
                        'passed': cycle_result.passed,
                        'signal_quality': cycle_result.eye_height / results.initial_eye_height if results.initial_eye_height > 0 else 1.0,
                        'last_hook_execution': getattr(self, '_last_hook_execution', {})
                    }
                    
                    # Execute threshold hooks if BER exceeded
                    if cycle_result.ber_estimate > 1e-9:  # Default threshold
                        self._execute_bert_hooks(HookType.ON_THRESHOLD, cycle_context)
                    
                    # Execute periodic hooks
                    self._execute_bert_hooks(HookType.PERIODIC, cycle_context)
                    
                    # Update last execution time
                    if not hasattr(self, '_last_hook_execution'):
                        self._last_hook_execution = {}
                    for hook_name in self.bert_hook_manager.hooks.keys():
                        self._last_hook_execution[hook_name] = time.time()
                
                # Progress logging
                if cycle % 100 == 0 or cycle == 1:
                    self.logger.info(
                        f"Cycle {cycle}/{self.config.num_cycles}: "
                        f"Eye Height={cycle_result.eye_height:.4f}V, "
                        f"Jitter={cycle_result.rms_jitter:.4f}, "
                        f"Degradation={cycle_result.degradation_percent:.1f}%, "
                        f"Status={'PASS' if cycle_result.passed else 'FAIL'}"
                    )
                
                # Wait for cycle duration
                cycle_elapsed = time.time() - cycle_start
                if cycle_elapsed < self.config.cycle_duration:
                    time.sleep(self.config.cycle_duration - cycle_elapsed)
            
            # Calculate final statistics
            end_time = time.time()
            results.end_time = end_time
            
            if results.cycle_results:
                eye_heights = [r.eye_height for r in results.cycle_results]
                jitter_values = [r.rms_jitter for r in results.cycle_results]
                degradations = [r.degradation_percent for r in results.cycle_results]
                
                results.initial_eye_height = eye_heights[0]
                results.final_eye_height = eye_heights[-1]
                results.max_degradation = max(degradations)
                results.degradation_rate = results.max_degradation / self.config.num_cycles
                
                results.mean_eye_height = np.mean(eye_heights)
                results.std_eye_height = np.std(eye_heights)
                results.mean_jitter = np.mean(jitter_values)
                results.std_jitter = np.std(jitter_values)
            
            # Generate plots
            if self.config.generate_plots:
                self._generate_plots(results)
            
            # Execute post-test hooks
            if self.bert_hook_manager:
                post_test_context = {
                    'test_type': 'loopback_stress',
                    'protocol': self.config.protocol,
                    'total_cycles': results.total_cycles,
                    'passed_cycles': results.passed_cycles,
                    'failed_cycles': results.failed_cycles,
                    'success_rate': results.success_rate,
                    'max_degradation': results.max_degradation,
                    'duration': results.duration,
                    'end_time': results.end_time
                }
                self._execute_bert_hooks(HookType.POST_TEST, post_test_context)
            
            # Final summary
            self.logger.info(f"Stress test completed in {results.duration:.1f} seconds")
            self.logger.info(f"Success rate: {results.success_rate:.1%}")
            self.logger.info(f"Max degradation: {results.max_degradation:.1f}%")
            self.logger.info(f"Final eye height: {results.final_eye_height:.4f}V")
            
            return results
            
        except KeyboardInterrupt:
            self.logger.info("Stress test interrupted by user")
            results.end_time = time.time()
            
            # Execute error hooks for interruption
            if self.bert_hook_manager:
                error_context = {
                    'error_type': 'user_interrupt',
                    'message': 'Test interrupted by user',
                    'cycles_completed': len(results.cycle_results)
                }
                self._execute_bert_hooks(HookType.ON_ERROR, error_context)
            
            return results
        except Exception as e:
            self.logger.error(f"Stress test failed: {e}")
            results.end_time = time.time()
            
            # Execute error hooks for exception
            if self.bert_hook_manager:
                error_context = {
                    'error_type': 'exception',
                    'message': str(e),
                    'cycles_completed': len(results.cycle_results)
                }
                self._execute_bert_hooks(HookType.ON_ERROR, error_context)
            
            raise


def create_stress_test_config(
    protocol: str = "USB4",
    num_cycles: int = 1000,
    output_dir: str = "stress_test_results",
    enable_bert_hooks: bool = False
) -> StressTestConfig:
    """
    Create a stress test configuration with sensible defaults
    
    Args:
        protocol: Protocol to test (USB4, PCIe, Ethernet)
        num_cycles: Number of test cycles to run
        output_dir: Output directory for results
        enable_bert_hooks: Enable BERT script hooks
        
    Returns:
        Configured StressTestConfig object
    """
    # Protocol-specific configurations to fix Ethernet issues
    protocol_configs = {
        "USB4": {
            "data_rate": 20e9,
            "sample_rate": 80e9,  # 4x oversampling
            "voltage_swing": 0.8
        },
        "PCIe": {
            "data_rate": 32e9,
            "sample_rate": 128e9,  # 4x oversampling
            "voltage_swing": 1.2
        },
        "Ethernet": {
            "data_rate": 112e9,
            "sample_rate": 448e9,  # 4x oversampling, prevent division by zero
            "voltage_swing": 1.6
        }
    }
    
    config_params = protocol_configs.get(protocol, protocol_configs["USB4"])
    
    return StressTestConfig(
        num_cycles=num_cycles,
        protocol=protocol,
        data_rate=config_params["data_rate"],
        sample_rate=config_params["sample_rate"],
        voltage_swing=config_params["voltage_swing"],
        output_dir=Path(output_dir),
        generate_plots=True,
        save_waveforms=False,  # Only save on failures
        enable_bert_hooks=enable_bert_hooks
    )


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = create_stress_test_config(
        protocol="USB4",
        num_cycles=500,
        output_dir="usb4_stress_test"
    )
    
    # Run stress test
    stress_test = LoopbackStressTest(config)
    results = stress_test.run_stress_test()
    
    print(f"Test completed: {results.success_rate:.1%} success rate")
    print(f"Max degradation: {results.max_degradation:.1f}%")
