#!/usr/bin/env python3
"""
Loopback Stress Test Example

This example demonstrates how to use the SerDes Validation Framework
to perform loopback stress testing, simulating TX → RX → back to TX
loops and tracking signal degradation over multiple cycles.

Usage:
    python loopback_stress_test_example.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from serdes_validation_framework.stress_testing import LoopbackStressTest, StressTestConfig, create_stress_test_config


def run_usb4_stress_test():
    """Run USB4 loopback stress test"""
    print("🔄 Running USB4 Loopback Stress Test...")
    
    # Create configuration for USB4
    config = create_stress_test_config(
        protocol="USB4",
        num_cycles=200,  # Reduced for demo
        output_dir="results/usb4_loopback_stress"
    )
    
    # Customize configuration
    config.data_rate = 20e9  # 20 Gbps
    config.voltage_swing = 0.8  # 800mV
    config.cycle_duration = 0.5  # 0.5 seconds per cycle
    config.eye_height_threshold = 0.15  # 15% degradation threshold
    config.save_waveforms = True  # Save failed waveforms
    
    # Run stress test
    stress_test = LoopbackStressTest(config)
    results = stress_test.run_stress_test()
    
    # Print summary
    print("\n📊 USB4 Stress Test Results:")
    print(f"   Duration: {results.duration:.1f} seconds")
    print(f"   Success Rate: {results.success_rate:.1%}")
    print(f"   Max Degradation: {results.max_degradation:.1f}%")
    print(f"   Initial Eye Height: {results.initial_eye_height:.4f}V")
    print(f"   Final Eye Height: {results.final_eye_height:.4f}V")
    print(f"   Mean Jitter: {results.mean_jitter:.4f}")
    
    return results


def run_pcie_stress_test():
    """Run PCIe loopback stress test"""
    print("\n🔄 Running PCIe Loopback Stress Test...")
    
    # Create configuration for PCIe
    config = StressTestConfig(
        protocol="PCIe",
        num_cycles=150,
        data_rate=32e9,  # 32 GT/s
        voltage_swing=1.2,  # 1.2V for PCIe
        cycle_duration=0.8,
        eye_height_threshold=0.12,  # 12% threshold
        jitter_threshold=0.04,  # 4% jitter threshold
        output_dir=Path("results/pcie_loopback_stress"),
        generate_plots=True,
        save_waveforms=False
    )
    
    # Run stress test
    stress_test = LoopbackStressTest(config)
    results = stress_test.run_stress_test()
    
    # Print summary
    print("\n📊 PCIe Stress Test Results:")
    print(f"   Duration: {results.duration:.1f} seconds")
    print(f"   Success Rate: {results.success_rate:.1%}")
    print(f"   Max Degradation: {results.max_degradation:.1f}%")
    print(f"   Degradation Rate: {results.degradation_rate:.4f}% per cycle")
    
    return results


def run_ethernet_stress_test():
    """Run Ethernet loopback stress test"""
    print("\n🔄 Running Ethernet 224G Loopback Stress Test...")
    
    # Create configuration for Ethernet
    config = create_stress_test_config(
        protocol="Ethernet",
        num_cycles=100,
        output_dir="results/ethernet_loopback_stress"
    )
    
    # Customize for 224G Ethernet
    config.data_rate = 112e9  # 112 Gbps (PAM4)
    config.voltage_swing = 0.8
    config.cycle_duration = 1.0
    config.eye_height_threshold = 0.10  # 10% threshold
    
    # Run stress test
    stress_test = LoopbackStressTest(config)
    results = stress_test.run_stress_test()
    
    # Print summary
    print("\n📊 Ethernet Stress Test Results:")
    print(f"   Duration: {results.duration:.1f} seconds")
    print(f"   Success Rate: {results.success_rate:.1%}")
    print(f"   Failed Cycles: {results.failed_cycles}")
    
    return results


def compare_protocols():
    """Compare stress test results across protocols"""
    print("\n🔬 Running Multi-Protocol Comparison...")
    
    protocols = ["USB4", "PCIe", "Ethernet"]
    results = {}
    
    for protocol in protocols:
        config = create_stress_test_config(
            protocol=protocol,
            num_cycles=50,  # Quick comparison
            output_dir=f"results/{protocol.lower()}_comparison"
        )
        
        # Protocol-specific settings
        if protocol == "USB4":
            config.data_rate = 20e9
            config.voltage_swing = 0.8
        elif protocol == "PCIe":
            config.data_rate = 32e9
            config.voltage_swing = 1.2
        else:  # Ethernet
            config.data_rate = 112e9
            config.voltage_swing = 0.8
        
        stress_test = LoopbackStressTest(config)
        results[protocol] = stress_test.run_stress_test()
    
    # Print comparison
    print("\n📈 Protocol Comparison Results:")
    print(f"{'Protocol':<10} {'Success Rate':<12} {'Max Degradation':<15} {'Final Eye Height':<16}")
    print("-" * 60)
    
    for protocol, result in results.items():
        print(f"{protocol:<10} {result.success_rate:<12.1%} "
              f"{result.max_degradation:<15.1f}% {result.final_eye_height:<16.4f}V")
    
    return results


def demonstrate_csv_logging():
    """Demonstrate CSV logging and data analysis"""
    print("\n📝 Demonstrating CSV Logging...")
    
    config = create_stress_test_config(
        protocol="USB4",
        num_cycles=20,
        output_dir="results/csv_demo"
    )
    
    stress_test = LoopbackStressTest(config)
    results = stress_test.run_stress_test()
    
    # Show CSV file location
    csv_file = config.output_dir / "stress_test_results.csv"
    print(f"   CSV results saved to: {csv_file}")
    
    # Read and display first few lines
    try:
        with open(csv_file, 'r') as f:
            lines = f.readlines()[:6]  # Header + first 5 data lines
            print("   CSV Preview:")
            for line in lines:
                print(f"     {line.strip()}")
    except Exception as e:
        print(f"   Error reading CSV: {e}")
    
    return results


def main():
    """Main example function"""
    print("🚀 SerDes Validation Framework - Loopback Stress Test Examples")
    print("=" * 65)
    
    try:
        # Run individual protocol tests
        usb4_results = run_usb4_stress_test()
        pcie_results = run_pcie_stress_test()
        ethernet_results = run_ethernet_stress_test()
        
        # Run protocol comparison
        comparison_results = compare_protocols()
        
        # Demonstrate CSV logging
        csv_results = demonstrate_csv_logging()
        
        print("\n✅ All stress tests completed successfully!")
        print("   Check the 'results/' directory for detailed outputs")
        print("   - CSV files with cycle-by-cycle data")
        print("   - PNG plots showing degradation trends")
        print("   - Log files with detailed test information")
        
        # Overall summary
        all_results = [usb4_results, pcie_results, ethernet_results]
        avg_success_rate = sum(r.success_rate for r in all_results) / len(all_results)
        
        print("\n🎯 Overall Summary:")
        print(f"   Average Success Rate: {avg_success_rate:.1%}")
        print(f"   Total Test Cycles: {sum(r.total_cycles for r in all_results)}")
        print(f"   Total Test Duration: {sum(r.duration for r in all_results):.1f} seconds")
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running stress tests: {e}")
        raise


if __name__ == "__main__":
    main()
