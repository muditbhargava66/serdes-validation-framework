#!/usr/bin/env python3
"""
Comprehensive Visualization Example

This example demonstrates the advanced visualization capabilities of the
SerDes Validation Framework across all supported protocols.

Features demonstrated:
- Eye diagram visualization for USB4, PCIe, and Ethernet
- Protocol-specific visualizations (tunneling, link training, PAM4)
- Interactive dashboards
- Multi-protocol comparison
- Signal quality analysis
"""

import os
import sys
from pathlib import Path

import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from serdes_validation_framework.visualization import EthernetVisualizer, EyeDiagramVisualizer, PCIeVisualizer, USB4Visualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Visualization modules not available: {e}")
    VISUALIZATION_AVAILABLE = False

def generate_test_signals():
    """Generate test signals for different protocols"""
    np.random.seed(42)  # For reproducible results
    
    signals = {}
    
    # USB4 signal (20 Gbaud, NRZ)
    usb4_samples = 2000
    usb4_time = np.linspace(0, 100e-9, usb4_samples)  # 100 ns
    usb4_data = np.random.choice([-0.4, 0.4], usb4_samples)  # NRZ levels
    usb4_noise = 0.02 * np.random.randn(usb4_samples)
    signals['usb4'] = {
        0: {'time': usb4_time, 'voltage': usb4_data + usb4_noise},
        1: {'time': usb4_time, 'voltage': usb4_data + usb4_noise + 0.01 * np.random.randn(usb4_samples)}
    }
    
    # PCIe signal (32 GT/s, NRZ)
    pcie_samples = 2560
    pcie_time = np.linspace(0, 80e-9, pcie_samples)  # 80 ns
    pcie_data = np.random.choice([-0.5, 0.5], pcie_samples)  # NRZ levels
    pcie_noise = 0.03 * np.random.randn(pcie_samples)
    signals['pcie'] = pcie_data + pcie_noise
    
    # Ethernet signal (112 GBaud, PAM4)
    eth_samples = 1120
    eth_time = np.linspace(0, 10e-9, eth_samples)  # 10 ns
    pam4_levels = [-0.75, -0.25, 0.25, 0.75]
    eth_data = np.random.choice(pam4_levels, eth_samples)  # PAM4 levels
    eth_noise = 0.05 * np.random.randn(eth_samples)
    signals['ethernet'] = eth_data + eth_noise
    
    return signals

def demonstrate_usb4_visualization():
    """Demonstrate USB4-specific visualizations"""
    print("\\n" + "="*60)
    print("USB4 VISUALIZATION DEMONSTRATION")
    print("="*60)
    
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization not available")
        return
    
    # Create output directory
    output_dir = Path("visualization_output/usb4")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    usb4_viz = USB4Visualizer()
    
    # Generate test data
    signals = generate_test_signals()
    usb4_signal = signals['usb4']
    
    print("\\n1. Creating USB4 Eye Diagram...")
    eye_result = usb4_viz.plot_eye_diagram(
        usb4_signal,
        filename=str(output_dir / "usb4_eye_diagram.png"),
        show_measurements=True
    )
    if eye_result:
        print(f"   ✅ Eye diagram saved: {eye_result}")
    else:
        print("   ❌ Eye diagram failed")
    
    print("\\n2. Creating USB4 Signal Quality Plot...")
    signal_metrics = {
        'eye_height': 0.75,
        'eye_width': 0.85,
        'snr': 22.5,
        'jitter_rms': 0.12,
        'ber': 1e-12
    }
    quality_result = usb4_viz.plot_signal_quality(
        signal_metrics,
        filename=str(output_dir / "usb4_signal_quality.png")
    )
    if quality_result:
        print(f"   ✅ Signal quality plot saved: {quality_result}")
    
    print("\\n3. Creating USB4 Tunnel Bandwidth Plot...")
    bandwidth_data = {
        'PCIe': {'utilization': 75, 'max_bandwidth': 32, 'allocated_bandwidth': 24},
        'DisplayPort': {'utilization': 60, 'max_bandwidth': 20, 'allocated_bandwidth': 12},
        'USB3.2': {'utilization': 45, 'max_bandwidth': 10, 'allocated_bandwidth': 4.5}
    }
    bandwidth_result = usb4_viz.plot_tunnel_bandwidth(
        bandwidth_data,
        filename=str(output_dir / "usb4_tunnel_bandwidth.png")
    )
    if bandwidth_result:
        print(f"   ✅ Tunnel bandwidth plot saved: {bandwidth_result}")
    
    print("\\n4. Creating USB4 Power States Plot...")
    power_data = {
        'timeline': {
            'times': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'states': ['U0', 'U0', 'U1', 'U1', 'U0', 'U2', 'U2', 'U0', 'U0', 'U3', 'U0']
        },
        'consumption': {
            'U0': 2.5,  # Active
            'U1': 1.2,  # Standby
            'U2': 0.5,  # Sleep
            'U3': 0.1   # Deep sleep
        }
    }
    power_result = usb4_viz.plot_power_states(
        power_data,
        filename=str(output_dir / "usb4_power_states.png")
    )
    if power_result:
        print(f"   ✅ Power states plot saved: {power_result}")
    
    print("\\n5. Creating USB4 Interactive Dashboard...")
    dashboard_data = {
        'test_results': {
            'eye_diagram': {'status': 'PASS'},
            'jitter': {'status': 'PASS'},
            'power_management': {'status': 'PASS'},
            'tunneling': {'status': 'FAIL'}
        },
        'signal_quality': signal_metrics,
        'tunnel_bandwidth': {tunnel: data['utilization'] for tunnel, data in bandwidth_data.items()},
        'power_timeline': power_data['timeline']
    }
    dashboard_result = usb4_viz.create_interactive_dashboard(
        dashboard_data,
        filename=str(output_dir / "usb4_dashboard.html")
    )
    if dashboard_result:
        print(f"   ✅ Interactive dashboard saved: {dashboard_result}")

def demonstrate_pcie_visualization():
    """Demonstrate PCIe-specific visualizations"""
    print("\\n" + "="*60)
    print("PCIe VISUALIZATION DEMONSTRATION")
    print("="*60)
    
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization not available")
        return
    
    # Create output directory
    output_dir = Path("visualization_output/pcie")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    pcie_viz = PCIeVisualizer()
    
    # Generate test data
    signals = generate_test_signals()
    pcie_signal = signals['pcie']
    
    print("\\n1. Creating PCIe Eye Diagram...")
    eye_result = pcie_viz.plot_eye_diagram(
        pcie_signal,
        filename=str(output_dir / "pcie_eye_diagram.png"),
        show_measurements=True
    )
    if eye_result:
        print(f"   ✅ Eye diagram saved: {eye_result}")
    
    print("\\n2. Creating PCIe Link Training Plot...")
    training_data = {
        'phases': {
            'Detect': {'duration': 12, 'success': True},
            'Polling': {'duration': 24, 'success': True},
            'Configuration': {'duration': 48, 'success': True},
            'L0': {'duration': 1000, 'success': True}
        },
        'equalization': {
            'C-1': -2,
            'C0': 15,
            'C+1': -3
        }
    }
    training_result = pcie_viz.plot_link_training(
        training_data,
        filename=str(output_dir / "pcie_link_training.png")
    )
    if training_result:
        print(f"   ✅ Link training plot saved: {training_result}")

def demonstrate_ethernet_visualization():
    """Demonstrate Ethernet-specific visualizations"""
    print("\\n" + "="*60)
    print("224G ETHERNET VISUALIZATION DEMONSTRATION")
    print("="*60)
    
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization not available")
        return
    
    # Create output directory
    output_dir = Path("visualization_output/ethernet")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    eth_viz = EthernetVisualizer()
    
    # Generate test data
    signals = generate_test_signals()
    eth_signal = signals['ethernet']
    
    print("\\n1. Creating Ethernet PAM4 Eye Diagram...")
    eye_result = eth_viz.plot_eye_diagram(
        eth_signal,
        filename=str(output_dir / "ethernet_eye_diagram.png"),
        show_measurements=True
    )
    if eye_result:
        print(f"   ✅ PAM4 eye diagram saved: {eye_result}")
    
    print("\\n2. Creating PAM4 Level Analysis...")
    pam4_result = eth_viz.plot_pam4_levels(
        eth_signal,
        filename=str(output_dir / "ethernet_pam4_levels.png")
    )
    if pam4_result:
        print(f"   ✅ PAM4 level analysis saved: {pam4_result}")

def demonstrate_advanced_features():
    """Demonstrate advanced visualization features"""
    print("\\n" + "="*60)
    print("ADVANCED VISUALIZATION FEATURES")
    print("="*60)
    
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization not available")
        return
    
    # Create output directory
    output_dir = Path("visualization_output/advanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test data
    signals = generate_test_signals()
    
    print("\\n1. Creating Multi-Protocol Eye Diagram Comparison...")
    eye_viz = EyeDiagramVisualizer("Multi-Protocol")
    
    # Create comparison plots for each protocol
    protocols = [
        ('USB4', signals['usb4'], {'symbol_rate': 20e9, 'samples_per_symbol': 10}),
        ('PCIe', signals['pcie'], {'symbol_rate': 32e9, 'samples_per_symbol': 8}),
        ('Ethernet', signals['ethernet'], {'symbol_rate': 112e9, 'samples_per_symbol': 4})
    ]
    
    for protocol_name, signal_data, params in protocols:
        result = eye_viz.plot_eye_diagram(
            signal_data,
            filename=str(output_dir / f"{protocol_name.lower()}_comparison_eye.png"),
            title=f"{protocol_name} Eye Diagram Comparison",
            **params
        )
        if result:
            print(f"   ✅ {protocol_name} comparison eye diagram saved")
    
    print("\\n2. Creating Signal Spectrum Analysis...")
    usb4_viz = USB4Visualizer()
    
    # Extract single channel data for spectrum analysis
    usb4_data = signals['usb4'][0]['voltage']
    spectrum_result = usb4_viz.plot_spectrum(
        usb4_data,
        sample_rate=200e9,
        filename=str(output_dir / "usb4_spectrum.png")
    )
    if spectrum_result:
        print(f"   ✅ Spectrum analysis saved: {spectrum_result}")
    
    print("\\n3. Creating Comprehensive Summary Report...")
    test_results = {
        'test_results': {
            'eye_diagram': {'status': 'PASS'},
            'jitter': {'status': 'PASS'},
            'power_management': {'status': 'PASS'},
            'compliance': {'status': 'PASS'},
            'tunneling': {'status': 'FAIL'}
        },
        'signal_quality': {
            'eye_height': 0.75,
            'eye_width': 0.85,
            'snr': 22.5,
            'jitter_rms': 0.12
        },
        'performance': {
            'throughput': 38.5,
            'latency': 125
        },
        'compliance': {
            'USB4_v2.0': True,
            'Thunderbolt_4': True,
            'Power_Delivery': False
        }
    }
    
    summary_result = usb4_viz.create_summary_report(
        test_results,
        filename=str(output_dir / "comprehensive_summary.png")
    )
    if summary_result:
        print(f"   ✅ Summary report saved: {summary_result}")

def main():
    """Main function demonstrating comprehensive visualization capabilities"""
    print("🎨 SerDes Validation Framework - Comprehensive Visualization Example")
    print("="*80)
    
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization modules not available. Please install matplotlib and plotly:")
        print("   pip install matplotlib plotly seaborn")
        return
    
    print("✅ Visualization modules available")
    print("📁 Output will be saved to: ./visualization_output/")
    
    # Demonstrate protocol-specific visualizations
    demonstrate_usb4_visualization()
    demonstrate_pcie_visualization()
    demonstrate_ethernet_visualization()
    demonstrate_advanced_features()
    
    print("\\n" + "="*80)
    print("🎉 VISUALIZATION DEMONSTRATION COMPLETE")
    print("="*80)
    print("\\n📊 Generated visualizations:")
    
    # List all generated files
    output_base = Path("visualization_output")
    if output_base.exists():
        for protocol_dir in output_base.iterdir():
            if protocol_dir.is_dir():
                print(f"\\n📁 {protocol_dir.name.upper()}:")
                for file in protocol_dir.iterdir():
                    if file.is_file():
                        print(f"   📄 {file.name}")
    
    print("\\n💡 Tips:")
    print("   • Open .png files with any image viewer")
    print("   • Open .html files in a web browser for interactive dashboards")
    print("   • Use these visualizations in your validation reports")
    print("   • Extend the visualizers for your specific needs")

if __name__ == "__main__":
    main()
