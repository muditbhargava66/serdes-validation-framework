#!/usr/bin/env python3
"""
Loopback Stress Test Script

Command-line script for running SerDes loopback stress tests with
customizable parameters and output options.

Usage:
    python run_loopback_stress_test.py --protocol USB4 --cycles 1000
    python run_loopback_stress_test.py --protocol PCIe --cycles 500 --output pcie_test
    python run_loopback_stress_test.py --help
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from serdes_validation_framework.stress_testing import LoopbackStressTest, StressTestConfig


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run SerDes loopback stress test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Test parameters
    parser.add_argument(
        "--protocol", "-p",
        choices=["USB4", "PCIe", "Ethernet"],
        default="USB4",
        help="Protocol to test"
    )
    
    parser.add_argument(
        "--cycles", "-c",
        type=int,
        default=1000,
        help="Number of test cycles to run"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=1.0,
        help="Duration per cycle in seconds"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="stress_test_results",
        help="Output directory for results"
    )
    
    # Protocol-specific parameters
    parser.add_argument(
        "--data-rate",
        type=float,
        help="Data rate in Gbps (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--voltage-swing",
        type=float,
        help="Voltage swing in V (auto-detected if not specified)"
    )
    
    # Thresholds
    parser.add_argument(
        "--eye-threshold",
        type=float,
        default=0.1,
        help="Eye height degradation threshold (0.1 = 10%%)"
    )
    
    parser.add_argument(
        "--jitter-threshold",
        type=float,
        default=0.05,
        help="Jitter increase threshold"
    )
    
    # Output options
    parser.add_argument(
        "--save-waveforms",
        action="store_true",
        help="Save waveforms for failed cycles"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    # Quick test option
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test (100 cycles, 0.1s per cycle)"
    )
    
    return parser.parse_args()


def get_protocol_defaults(protocol: str):
    """Get default parameters for each protocol"""
    defaults = {
        "USB4": {
            "data_rate": 20e9,  # 20 Gbps
            "voltage_swing": 0.8,  # 800mV
        },
        "PCIe": {
            "data_rate": 32e9,  # 32 GT/s
            "voltage_swing": 1.2,  # 1.2V
        },
        "Ethernet": {
            "data_rate": 112e9,  # 112 Gbps (PAM4)
            "voltage_swing": 0.8,  # 800mV
        }
    }
    return defaults.get(protocol, defaults["USB4"])


def create_config_from_args(args) -> StressTestConfig:
    """Create stress test configuration from command line arguments"""
    
    # Get protocol defaults
    defaults = get_protocol_defaults(args.protocol)
    
    # Override with command line values if provided
    data_rate = args.data_rate * 1e9 if args.data_rate else defaults["data_rate"]
    voltage_swing = args.voltage_swing if args.voltage_swing else defaults["voltage_swing"]
    
    # Quick test overrides
    if args.quick:
        cycles = 100
        duration = 0.1
        print("🚀 Running quick test (100 cycles, 0.1s per cycle)")
    else:
        cycles = args.cycles
        duration = args.duration
    
    # Create configuration
    config = StressTestConfig(
        # Test parameters
        num_cycles=cycles,
        cycle_duration=duration,
        
        # Protocol settings
        protocol=args.protocol,
        data_rate=data_rate,
        voltage_swing=voltage_swing,
        
        # Thresholds
        eye_height_threshold=args.eye_threshold,
        jitter_threshold=args.jitter_threshold,
        
        # Output settings
        output_dir=Path(args.output),
        save_waveforms=args.save_waveforms,
        generate_plots=not args.no_plots,
        
        # Logging
        log_level=args.log_level
    )
    
    return config


def print_config_summary(config: StressTestConfig):
    """Print configuration summary"""
    print("🔧 Test Configuration:")
    print(f"   Protocol: {config.protocol}")
    print(f"   Cycles: {config.num_cycles}")
    print(f"   Duration per cycle: {config.cycle_duration}s")
    print(f"   Data rate: {config.data_rate/1e9:.1f} Gbps")
    print(f"   Voltage swing: {config.voltage_swing}V")
    print(f"   Eye threshold: {config.eye_height_threshold*100:.1f}%")
    print(f"   Jitter threshold: {config.jitter_threshold*100:.1f}%")
    print(f"   Output directory: {config.output_dir}")
    print(f"   Save waveforms: {config.save_waveforms}")
    print(f"   Generate plots: {config.generate_plots}")
    print()


def print_results_summary(results):
    """Print test results summary"""
    print("\n📊 Test Results Summary:")
    print(f"   Duration: {results.duration:.1f} seconds")
    print(f"   Total cycles: {results.total_cycles}")
    print(f"   Passed cycles: {results.passed_cycles}")
    print(f"   Failed cycles: {results.failed_cycles}")
    print(f"   Success rate: {results.success_rate:.1%}")
    print()
    
    print("📈 Signal Quality Metrics:")
    print(f"   Initial eye height: {results.initial_eye_height:.4f}V")
    print(f"   Final eye height: {results.final_eye_height:.4f}V")
    print(f"   Max degradation: {results.max_degradation:.1f}%")
    print(f"   Degradation rate: {results.degradation_rate:.4f}% per cycle")
    print(f"   Mean eye height: {results.mean_eye_height:.4f}V ± {results.std_eye_height:.4f}V")
    print(f"   Mean jitter: {results.mean_jitter:.4f} ± {results.std_jitter:.4f}")
    print()
    
    # Status assessment
    if results.success_rate >= 0.95:
        status = "✅ EXCELLENT"
    elif results.success_rate >= 0.90:
        status = "✅ GOOD"
    elif results.success_rate >= 0.80:
        status = "⚠️  MARGINAL"
    else:
        status = "❌ POOR"
    
    print(f"🎯 Overall Assessment: {status}")
    
    # Output files
    print("\n📁 Output Files:")
    print(f"   CSV data: {results.config.output_dir}/stress_test_results.csv")
    print(f"   Log file: {results.config.output_dir}/stress_test.log")
    if results.config.generate_plots:
        print(f"   Analysis plots: {results.config.output_dir}/stress_test_analysis.png")
    if results.config.save_waveforms and results.failed_cycles > 0:
        print(f"   Failed waveforms: {results.config.output_dir}/waveform_cycle_*.csv")


def main():
    """Main script function"""
    print("🚀 SerDes Validation Framework - Loopback Stress Test")
    print("=" * 55)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create configuration
        config = create_config_from_args(args)
        
        # Print configuration summary
        print_config_summary(config)
        
        # Estimate test duration
        estimated_duration = config.num_cycles * config.cycle_duration
        print(f"⏱️  Estimated test duration: {estimated_duration:.1f} seconds ({estimated_duration/60:.1f} minutes)")
        
        # Confirm if long test
        if estimated_duration > 300:  # 5 minutes
            response = input("This is a long test. Continue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Test cancelled.")
                return
        
        print("\n🔄 Starting stress test...")
        
        # Run stress test
        stress_test = LoopbackStressTest(config)
        results = stress_test.run_stress_test()
        
        # Print results
        print_results_summary(results)
        
        # Success message
        print("✅ Stress test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running stress test: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
