#!/usr/bin/env python3
"""
Jupyter Dashboard Example

This example demonstrates the Jupyter-based SVF Eye Diagram Dashboard
capabilities for interactive visualization of SerDes signals.

Usage:
    python jupyter_dashboard_example.py
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from serdes_validation_framework.jupyter_dashboard import (
    DashboardConfig,
    EyeDiagramDashboard,
    WaveformAnalyzer,
    check_dashboard_dependencies,
    create_dashboard,
)


def generate_sample_usb4_data():
    """Generate sample USB4 signal data"""
    # Generate realistic USB4 dual-lane signal
    duration = 2e-6  # 2 microseconds
    sample_rate = 40e9  # 40 GSa/s
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    signal_data = {}
    for lane in range(2):
        # Generate NRZ-like signal
        data_bits = np.random.choice([-1, 1], size=len(t))
        
        # Add some filtering (bandwidth limitation)
        try:
            from scipy import signal as scipy_signal
            b, a = scipy_signal.butter(4, 0.3)
            filtered_signal = scipy_signal.filtfilt(b, a, data_bits)
        except ImportError:
            # Fallback without scipy
            filtered_signal = data_bits
        
        # Scale to USB4 voltage levels (±400mV)
        usb4_signal = filtered_signal * 0.4
        
        # Add noise and jitter
        noise = np.random.normal(0, 0.02, len(usb4_signal))
        jitter = np.random.normal(0, 0.01, len(usb4_signal))
        
        # Apply lane-specific offset
        lane_offset = lane * 0.05
        
        signal_data[f'lane_{lane}'] = usb4_signal + noise + jitter + lane_offset
    
    return signal_data, t


def demonstrate_basic_dashboard():
    """Demonstrate basic dashboard functionality"""
    print("🔄 Generating sample USB4 data...")
    signal_data, time_data = generate_sample_usb4_data()
    
    print(f"✅ Generated USB4 data: {len(signal_data)} lanes, {len(time_data)} samples")
    
    # Create dashboard
    print("\n📊 Creating eye diagram dashboard...")
    config = DashboardConfig(
        figure_width=12,
        figure_height=8,
        show_measurements=True,
        show_mask=True
    )
    
    dashboard = create_dashboard(
        signal_data=signal_data,
        sample_rate=40e9,
        protocol="USB4",
        config=config
    )
    
    # Analyze both lanes
    results = {}
    for lane in signal_data.keys():
        print(f"\n🔍 Analyzing {lane}...")
        result = dashboard.analyze_eye_diagram(lane)
        results[lane] = result
        
        print(f"   Eye Height: {result['eye_height']:.4f} V")
        print(f"   Eye Width: {result['eye_width']:.4f} UI")
        print(f"   SNR: {result['snr']:.2f} dB")
        print(f"   Q-Factor: {result['q_factor']:.2f}")
        print(f"   Status: {'✅ PASS' if result['passed'] else '❌ FAIL'}")
    
    # Create static dashboard (works without Jupyter)
    print("\n📊 Creating static dashboard for lane_0...")
    dashboard.create_static_dashboard(lane='lane_0')
    
    # Export results
    print("\n💾 Exporting results...")
    dashboard.export_results("usb4_dashboard_results.json")
    
    return dashboard, results


def demonstrate_waveform_analysis():
    """Demonstrate waveform analysis capabilities"""
    print("\n🔍 Demonstrating waveform analysis...")
    
    # Generate sample data
    signal_data, time_data = generate_sample_usb4_data()
    
    # Create waveform analyzer
    analyzer = WaveformAnalyzer(sample_rate=40e9, protocol="USB4")
    
    # Analyze all lanes
    for lane_name, signal in signal_data.items():
        print(f"\n📊 Analyzing waveform for {lane_name}...")
        
        result = analyzer.analyze_waveform(
            voltage_data=signal,
            time_data=time_data,
            lane=lane_name
        )
        
        print(f"   Mean Voltage: {result.mean_voltage:.4f} V")
        print(f"   Peak-to-Peak: {result.peak_to_peak:.4f} V")
        print(f"   SNR: {result.snr_db:.2f} dB")
        print(f"   THD: {result.thd_percent:.2f}%")
        print(f"   Dynamic Range: {result.dynamic_range:.2f} dB")
        print(f"   Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
        
        if not result.passed:
            print(f"   Failures: {', '.join(result.failure_reasons)}")
    
    # Generate summary report
    print("\n📋 Waveform Analysis Summary:")
    summary = analyzer.get_summary_report()
    print(summary)
    
    # Create analysis plot (if matplotlib available)
    try:
        print("\n📊 Creating waveform analysis plot...")
        analyzer.create_analysis_plot(lane='lane_0', plot_type='matplotlib')
    except Exception as e:
        print(f"   ⚠️  Plot creation failed: {e}")
    
    return analyzer


def demonstrate_multi_protocol_comparison():
    """Demonstrate multi-protocol comparison"""
    print("\n🔄 Demonstrating multi-protocol comparison...")
    
    protocols = ['USB4', 'PCIe', 'Ethernet']
    results = {}
    
    for protocol in protocols:
        print(f"\n📊 Creating {protocol} dashboard...")
        
        # Generate protocol-specific data
        if protocol == 'USB4':
            signal_data, time_data = generate_sample_usb4_data()
            sample_rate = 40e9
        elif protocol == 'PCIe':
            # Generate PAM4-like signal for PCIe
            duration = 1e-6
            sample_rate = 80e9
            t = np.linspace(0, duration, int(duration * sample_rate))
            pam4_levels = np.random.choice([-3, -1, 1, 3], size=len(t))
            signal = pam4_levels * 0.2 + np.random.normal(0, 0.03, len(t))
            signal_data = {'lane_0': signal}
            time_data = t
        else:  # Ethernet
            # Generate high-speed PAM4 for Ethernet
            duration = 0.5e-6
            sample_rate = 224e9
            t = np.linspace(0, duration, int(duration * sample_rate))
            pam4_levels = np.random.choice([-3, -1, 1, 3], size=len(t))
            signal = pam4_levels * 0.15 + np.random.normal(0, 0.025, len(t))
            signal_data = {'lane_0': signal}
            time_data = t
        
        # Create dashboard and analyze
        dashboard = create_dashboard(
            signal_data=signal_data,
            sample_rate=sample_rate,
            protocol=protocol
        )
        
        result = dashboard.analyze_eye_diagram(lane=0)
        results[protocol] = result
        
        print(f"   ✅ {protocol}: Eye Height={result['eye_height']:.4f}V, SNR={result['snr']:.2f}dB, Status={'PASS' if result['passed'] else 'FAIL'}")
    
    # Create comparison summary
    print("\n📊 Multi-Protocol Comparison Summary:")
    print(f"{'Protocol':<10} {'Eye Height':<12} {'SNR':<8} {'Status':<8}")
    print("-" * 45)
    
    for protocol, result in results.items():
        status = "PASS" if result['passed'] else "FAIL"
        print(f"{protocol:<10} {result['eye_height']:<12.4f} {result['snr']:<8.2f} {status:<8}")
    
    return results


def main():
    """Main example function"""
    print("🚀 SerDes Validation Framework - Jupyter Dashboard Examples")
    print("=" * 65)
    
    # Check dependencies
    print("🔍 Checking dashboard dependencies...")
    deps = check_dashboard_dependencies()
    
    if not any(deps.values()):
        print("\n❌ No visualization libraries available.")
        print("   Install matplotlib and/or plotly for full functionality:")
        print("   pip install matplotlib plotly")
        return
    
    try:
        # Basic dashboard demonstration
        print("\n" + "="*50)
        print("📊 BASIC DASHBOARD DEMONSTRATION")
        print("="*50)
        dashboard, results = demonstrate_basic_dashboard()
        
        # Waveform analysis demonstration
        print("\n" + "="*50)
        print("🔍 WAVEFORM ANALYSIS DEMONSTRATION")
        print("="*50)
        analyzer = demonstrate_waveform_analysis()
        
        # Multi-protocol comparison
        print("\n" + "="*50)
        print("🔄 MULTI-PROTOCOL COMPARISON")
        print("="*50)
        comparison_results = demonstrate_multi_protocol_comparison()
        
        # Final summary
        print("\n" + "="*50)
        print("🎉 DEMONSTRATION COMPLETE")
        print("="*50)
        print("✅ All dashboard features demonstrated successfully!")
        print("\n📋 Generated Files:")
        print("   - usb4_dashboard_results.json (eye diagram analysis)")
        print("   - Various matplotlib plots (if available)")
        
        print("\n💡 Next Steps:")
        print("   1. Install Jupyter: pip install jupyter ipywidgets")
        print("   2. Run the Jupyter notebook: examples/jupyter_eye_diagram_dashboard.ipynb")
        print("   3. Explore interactive features with real-time controls")
        print("   4. Integrate with your own signal data")
        
        print("\n🔗 For more information:")
        print("   - Documentation: docs/guides/")
        print("   - API Reference: docs/api/")
        print("   - Examples: examples/")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
