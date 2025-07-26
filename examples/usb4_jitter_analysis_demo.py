#!/usr/bin/env python3
"""
USB4 Jitter Analysis Demonstration

This script demonstrates the USB4 jitter analysis capabilities including:
- Advanced jitter decomposition (RJ, DJ, PJ)
- SSC-aware jitter measurement
- Periodic jitter analysis for spread spectrum effects
- Comprehensive jitter compliance checking
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from serdes_validation_framework.protocols.usb4.constants import USB4SignalMode
from serdes_validation_framework.protocols.usb4.jitter_analyzer import USB4JitterAnalyzer, USB4JitterConfig


def generate_usb4_signal_with_jitter(sample_rate=256e9, symbol_rate=20e9, length=50000):
    """Generate a realistic USB4 signal with various jitter components"""
    
    # Time base
    time_data = np.arange(length) / sample_rate
    ui_period = 1.0 / symbol_rate
    
    # Base USB4 signal
    fundamental = symbol_rate
    signal = np.sin(2 * np.pi * fundamental * time_data)
    
    # Add random jitter (thermal noise, shot noise)
    rj_amplitude = 0.04  # 0.04 UI RMS
    timing_jitter_rj = np.random.normal(0, rj_amplitude * ui_period, length)
    
    # Add deterministic jitter components
    # 1. Pattern-dependent jitter (1 MHz pattern)
    pdj_freq = 1e6
    pdj_amplitude = 0.02  # 0.02 UI
    timing_jitter_pdj = pdj_amplitude * ui_period * np.sin(2 * np.pi * pdj_freq * time_data)
    
    # 2. Duty cycle distortion (systematic offset)
    dcd_amplitude = 0.015  # 0.015 UI
    timing_jitter_dcd = dcd_amplitude * ui_period * np.sign(np.sin(2 * np.pi * fundamental * time_data))
    
    # Add periodic jitter components
    # 1. Spread spectrum clocking (SSC) at 33 kHz
    ssc_freq = 33e3
    ssc_amplitude = 0.025  # 0.025 UI
    timing_jitter_ssc = ssc_amplitude * ui_period * np.sin(2 * np.pi * ssc_freq * time_data)
    
    # 2. Power supply noise at 100 kHz (switching regulator)
    power_freq = 100e3
    power_amplitude = 0.008  # 0.008 UI
    timing_jitter_power = power_amplitude * ui_period * np.sin(2 * np.pi * power_freq * time_data)
    
    # 3. Line frequency interference at 60 Hz
    line_freq = 60
    line_amplitude = 0.003  # 0.003 UI
    timing_jitter_line = line_amplitude * ui_period * np.sin(2 * np.pi * line_freq * time_data)
    
    # Combine all jitter components
    total_timing_jitter = (timing_jitter_rj + timing_jitter_pdj + timing_jitter_dcd + 
                          timing_jitter_ssc + timing_jitter_power + timing_jitter_line)
    
    # Apply jitter to signal
    jittered_time = time_data + total_timing_jitter
    jittered_signal = np.sin(2 * np.pi * fundamental * jittered_time)
    
    # Add amplitude noise
    amplitude_noise = np.random.normal(0, 0.02, length)
    final_signal = jittered_signal + amplitude_noise
    
    return final_signal, time_data


def analyze_usb4_jitter():
    """Demonstrate USB4 jitter analysis"""
    
    print("=" * 80)
    print("USB4 Jitter Analysis Demonstration")
    print("=" * 80)
    
    # Configuration
    config = USB4JitterConfig(
        sample_rate=256e9,
        symbol_rate=20e9,
        mode=USB4SignalMode.GEN2X2,
        enable_ssc_analysis=True,
        target_ber=1e-12,
        analysis_length=50000,
        confidence_level=0.95
    )
    
    print("Configuration:")
    print(f"  Sample Rate: {config.sample_rate/1e9:.1f} GSa/s")
    print(f"  Symbol Rate: {config.symbol_rate/1e9:.1f} Gbaud")
    print(f"  Mode: {config.mode.name}")
    print(f"  Target BER: {config.target_ber}")
    print(f"  Analysis Length: {config.analysis_length:,} samples")
    print()
    
    # Generate test signal
    print("Generating USB4 signal with controlled jitter components...")
    signal_data, time_data = generate_usb4_signal_with_jitter(
        config.sample_rate, config.symbol_rate, config.analysis_length
    )
    print(f"Generated {len(signal_data):,} samples over {time_data[-1]*1e9:.2f} ns")
    print()
    
    # Create analyzer
    analyzer = USB4JitterAnalyzer(config)
    
    # Perform jitter analysis
    print("Performing comprehensive USB4 jitter analysis...")
    results = analyzer.analyze_usb4_jitter(signal_data, time_data)
    print("Analysis complete!")
    print()
    
    # Display results
    print("=" * 60)
    print("JITTER ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"Total Jitter: {results.total_jitter:.4f} UI")
    print(f"Compliance Status: {'PASS' if results.compliance_status else 'FAIL'}")
    print()
    
    # Random Jitter Results
    print("Random Jitter (RJ) Analysis:")
    print(f"  RMS Jitter: {results.random_jitter.rj_rms:.4f} UI")
    print(f"  Peak-to-Peak @ 1e-12 BER: {results.random_jitter.rj_pp_1e12:.4f} UI")
    print(f"  Peak-to-Peak @ 1e-15 BER: {results.random_jitter.rj_pp_1e15:.4f} UI")
    print(f"  Gaussian Fit Quality (R²): {results.random_jitter.gaussian_fit_quality:.3f}")
    print(f"  95% Confidence Interval: ({results.random_jitter.confidence_interval[0]:.4f}, {results.random_jitter.confidence_interval[1]:.4f}) UI")
    print()
    
    # Deterministic Jitter Results
    print("Deterministic Jitter (DJ) Analysis:")
    print(f"  Peak-to-Peak DJ: {results.deterministic_jitter.dj_pp:.4f} UI")
    print(f"  Pattern-Dependent Jitter: {results.deterministic_jitter.pattern_dependent_jitter:.4f} UI")
    print(f"  Duty Cycle Distortion: {results.deterministic_jitter.duty_cycle_distortion:.4f} UI")
    print(f"  Intersymbol Interference: {results.deterministic_jitter.intersymbol_interference:.4f} UI")
    print(f"  DJ Compliance: {'PASS' if results.deterministic_jitter.compliance_status else 'FAIL'}")
    print()
    
    # Periodic Jitter Results
    print("Periodic Jitter (PJ) Analysis:")
    print(f"  RMS PJ: {results.periodic_jitter.pj_rms:.4f} UI")
    print(f"  Peak-to-Peak PJ: {results.periodic_jitter.pj_pp:.4f} UI")
    print(f"  SSC Contribution: {results.periodic_jitter.ssc_contribution:.4f} UI")
    print(f"  Power Supply Noise: {results.periodic_jitter.power_supply_noise:.4f} UI")
    print(f"  Dominant Frequencies: {[f'{f/1e3:.1f} kHz' for f in results.periodic_jitter.dominant_frequencies[:3]]}")
    print(f"  PJ Compliance: {'PASS' if results.periodic_jitter.compliance_status else 'FAIL'}")
    print()
    
    # SSC-Aware Results
    print("SSC-Aware Jitter Analysis:")
    print(f"  Total Jitter (SSC ON): {results.ssc_aware_results.total_jitter_ssc_on:.4f} UI")
    print(f"  Total Jitter (SSC OFF): {results.ssc_aware_results.total_jitter_ssc_off:.4f} UI")
    print(f"  SSC Jitter Contribution: {results.ssc_aware_results.ssc_jitter_contribution:.4f} UI")
    print(f"  Clock Recovery Quality: {results.ssc_aware_results.clock_recovery_quality:.3f}")
    print(f"  SSC Tracking Error: {results.ssc_aware_results.ssc_tracking_error:.4f} UI")
    print()
    
    # Compliance Margins
    print("Compliance Margins:")
    for component, margin in results.compliance_margins.items():
        status = "PASS" if margin >= 0 else "FAIL"
        print(f"  {component.replace('_', ' ').title()}: {margin:+.4f} UI ({status})")
    print()
    
    # Recommendations
    if results.recommendations:
        print("Recommendations for Improvement:")
        for i, rec in enumerate(results.recommendations, 1):
            print(f"  {i}. {rec}")
        print()
    
    # USB4 Specification Limits
    print("USB4 Specification Limits:")
    print(f"  Maximum Total Jitter: {analyzer.signal_specs.TOTAL_JITTER_MAX:.3f} UI")
    print(f"  Maximum Random Jitter: {analyzer.signal_specs.RANDOM_JITTER_MAX:.3f} UI")
    print(f"  Maximum Deterministic Jitter: {analyzer.signal_specs.DETERMINISTIC_JITTER_MAX:.3f} UI")
    print(f"  Maximum Periodic Jitter: {analyzer.signal_specs.PERIODIC_JITTER_MAX:.3f} UI")
    print()
    
    return results, signal_data, time_data


def plot_jitter_analysis(results, signal_data, time_data):
    """Create plots showing jitter analysis results"""
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('USB4 Jitter Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Time domain signal
        ax1 = axes[0, 0]
        time_ns = time_data[:1000] * 1e9  # First 1000 samples in ns
        ax1.plot(time_ns, signal_data[:1000], 'b-', linewidth=0.8)
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Amplitude (V)')
        ax1.set_title('USB4 Signal with Jitter')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Jitter components comparison
        ax2 = axes[0, 1]
        components = ['RJ (RMS)', 'DJ (PP)', 'PJ (RMS)', 'Total']
        values = [
            results.random_jitter.rj_rms,
            results.deterministic_jitter.dj_pp,
            results.periodic_jitter.pj_rms,
            results.total_jitter
        ]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        bars = ax2.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Jitter (UI)')
        ax2.set_title('Jitter Components')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values, strict=False):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Compliance margins
        ax3 = axes[1, 0]
        margin_names = list(results.compliance_margins.keys())
        margin_values = list(results.compliance_margins.values())
        colors = ['green' if m >= 0 else 'red' for m in margin_values]
        
        bars = ax3.barh(margin_names, margin_values, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Margin (UI)')
        ax3.set_title('Compliance Margins')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        # Add value labels
        for bar, value in zip(bars, margin_values, strict=False):
            width = bar.get_width()
            ax3.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2.,
                    f'{value:+.3f}', ha='left' if width >= 0 else 'right', va='center', fontweight='bold')
        
        # Plot 4: Distribution parameters
        ax4 = axes[1, 1]
        dist_params = results.random_jitter.distribution_parameters
        param_names = ['Mean', 'Std Dev', 'Skewness', 'Kurtosis']
        param_values = [
            dist_params['mean'],
            dist_params['std'],
            dist_params['skewness'],
            dist_params['kurtosis']
        ]
        
        bars = ax4.bar(param_names, param_values, color='lightblue', alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Value')
        ax4.set_title('Random Jitter Distribution Parameters')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, param_values, strict=False):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
                    f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(__file__).parent / "usb4_jitter_analysis_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        
        # Show plot if running interactively
        plt.show()
        
    except ImportError:
        print("Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"Error creating plots: {e}")


def main():
    """Main demonstration function"""
    
    # Perform jitter analysis
    results, signal_data, time_data = analyze_usb4_jitter()
    
    # Create plots
    plot_jitter_analysis(results, signal_data, time_data)
    
    print("=" * 80)
    print("USB4 Jitter Analysis Demonstration Complete!")
    print("=" * 80)
    
    # Summary
    print("\nSUMMARY:")
    print("✓ Generated realistic USB4 signal with multiple jitter sources")
    print("✓ Performed comprehensive jitter decomposition (RJ, DJ, PJ)")
    print("✓ Analyzed SSC-aware jitter characteristics")
    print("✓ Validated compliance against USB4 specifications")
    print("✓ Generated improvement recommendations")
    
    compliance_status = "COMPLIANT" if results.compliance_status else "NON-COMPLIANT"
    print(f"\nOverall USB4 Jitter Compliance: {compliance_status}")
    print(f"Total Jitter: {results.total_jitter:.4f} UI (Limit: {results.compliance_margins['total_jitter'] + results.total_jitter:.3f} UI)")


if __name__ == "__main__":
    main()
