#!/usr/bin/env python3
"""
USB4 Quick Start Example

This is the simplest way to get started with USB4 validation using the
SerDes Validation Framework. This example demonstrates:

1. Automatic protocol detection
2. Quick USB4 validation
3. Basic result interpretation

Usage:
    python usb4_quick_start.py
"""

import sys
from pathlib import Path

import numpy as np

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the framework
from serdes_validation_framework import auto_validate_signal, create_usb4_test_sequence, detect_signal_protocol


def generate_simple_usb4_signal():
    """Generate a simple USB4 test signal"""
    print("Generating USB4 test signal...")
    
    # Signal parameters
    duration = 5e-6  # 5 microseconds
    sample_rate = 200e9  # 200 GSa/s
    bit_rate = 20e9  # 20 Gbps per lane
    
    # Create time array
    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples)
    
    # Generate NRZ signal (USB4 uses NRZ modulation)
    bit_period = 1.0 / bit_rate
    num_bits = int(duration / bit_period)
    
    # Random data pattern
    np.random.seed(42)  # For reproducible results
    data_bits = np.random.randint(0, 2, num_bits)
    
    # Create NRZ waveform
    voltage = np.zeros(num_samples)
    for i, bit in enumerate(data_bits):
        start_idx = int(i * bit_period * sample_rate)
        end_idx = min(int((i + 1) * bit_period * sample_rate), num_samples)
        if end_idx > start_idx:
            voltage[start_idx:end_idx] = 0.4 if bit else -0.4  # ±400mV
    
    # Add some realistic noise
    noise = 0.02 * np.random.randn(num_samples)
    voltage += noise
    
    # Create dual-lane data (USB4 uses 2 lanes)
    signal_data = {
        0: {'time': time, 'voltage': voltage},
        1: {'time': time, 'voltage': voltage + 0.01 * np.random.randn(num_samples)}  # Slight variation
    }
    
    print(f"Generated {num_samples} samples over {duration*1e6:.1f} μs")
    return signal_data

def main():
    """Main function demonstrating USB4 quick start"""
    print("=" * 60)
    print("USB4 QUICK START EXAMPLE")
    print("=" * 60)
    
    # Step 1: Generate test signal
    signal_data = generate_simple_usb4_signal()
    
    # Step 2: Automatic protocol detection and validation
    print("\nStep 1: Automatic Protocol Detection and Validation")
    print("-" * 50)
    
    try:
        # Use the first lane for protocol detection
        detection_signal = signal_data[0]['voltage']
        
        # Automatic validation with protocol detection
        results = auto_validate_signal(
            signal_data=signal_data,
            sample_rate=200e9,
            voltage_range=0.8,
            protocol_hint="usb4"  # Optional hint
        )
        
        print(f"✓ Detected Protocol: {results['protocol_type']}")
        print(f"✓ Framework Version: {results['framework_version']}")
        
        # Extract validation results
        validation_results = results['validation_results']
        print(f"✓ Overall Status: {validation_results.overall_status.name}")
        print(f"✓ Test Duration: {validation_results.total_duration:.2f} seconds")
        print(f"✓ Phases Completed: {len(validation_results.phase_results)}")
        
    except Exception as e:
        print(f"✗ Automatic validation failed: {e}")
        print("Trying manual approach...")
        
        # Step 3: Manual protocol detection
        print("\nStep 2: Manual Protocol Detection")
        print("-" * 50)
        
        try:
            protocol_type = detect_signal_protocol(
                signal_data=detection_signal,
                sample_rate=200e9,
                voltage_range=0.8
            )
            print(f"✓ Detected Protocol: {protocol_type.name}")
            
            # Step 4: Create and run USB4 test sequence
            print("\nStep 3: Manual USB4 Validation")
            print("-" * 50)
            
            # Create USB4 test sequence
            test_sequence = create_usb4_test_sequence(enable_thunderbolt=False)
            
            # Run validation
            results = test_sequence.run_complete_sequence(signal_data)
            
            print(f"✓ Overall Status: {results.overall_status.name}")
            print(f"✓ Test Duration: {results.total_duration:.2f} seconds")
            
        except Exception as e:
            print(f"✗ Manual validation failed: {e}")
            return
    
    # Step 5: Display detailed results
    print("\nStep 4: Detailed Results")
    print("-" * 50)
    
    validation_results = results.get('validation_results', results)
    
    # Show phase results
    print("Phase Results:")
    for phase_result in validation_results.phase_results:
        status_symbol = "✓" if phase_result.status.name == "PASS" else "✗"
        print(f"  {status_symbol} {phase_result.phase.name:<20} {phase_result.status.name:<8} ({phase_result.duration:.2f}s)")
    
    # Show key metrics
    print("\nKey Metrics:")
    total_metrics = 0
    for phase_result in validation_results.phase_results:
        total_metrics += len(phase_result.metrics)
        # Show a few key metrics from each phase
        if phase_result.metrics:
            key_metrics = list(phase_result.metrics.items())[:2]  # First 2 metrics
            for metric, value in key_metrics:
                print(f"  {metric}: {value:.3f}")
    
    print(f"\nTotal Metrics Collected: {total_metrics}")
    
    # Show lane performance
    if hasattr(validation_results, 'lane_results') and validation_results.lane_results:
        print("\nLane Performance:")
        for lane_id, metrics in validation_results.lane_results.items():
            print(f"  Lane {lane_id}: {len(metrics)} metrics")
    
    # Show compliance summary
    if hasattr(validation_results, 'compliance_summary') and validation_results.compliance_summary:
        print("\nCompliance Summary:")
        for test, result in validation_results.compliance_summary.items():
            print(f"  {test}: {result}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("QUICK START SUMMARY")
    print("=" * 60)
    
    if validation_results.overall_status.name == "PASS":
        print("🎉 SUCCESS: USB4 signal validation completed successfully!")
        print("\nNext Steps:")
        print("- Try the comprehensive validation example: usb4_basic_validation_example.py")
        print("- Explore Thunderbolt 4 certification: usb4_thunderbolt_certification_example.py")
        print("- Check the USB4 API guide: docs/usb4_api_guide.md")
    else:
        print("⚠️  WARNING: USB4 validation completed with issues")
        print("\nTroubleshooting:")
        print("- Check signal quality and noise levels")
        print("- Verify signal parameters match USB4 specifications")
        print("- Review the USB4 API guide for detailed configuration options")
    
    print(f"\nFramework Version: {results.get('framework_version', '1.4.0')}")
    print("For more examples and documentation, see the examples/ and docs/ directories")

if __name__ == "__main__":
    main()
