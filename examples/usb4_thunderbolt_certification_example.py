#!/usr/bin/env python3
"""
Thunderbolt 4 Certification Example

This example demonstrates Thunderbolt 4 certification testing using the SerDes Validation Framework.
It shows how to:
1. Configure Thunderbolt 4 certification parameters
2. Run Intel certification test suite
3. Validate security features
4. Test daisy chain functionality
5. Generate certification reports

Usage:
    python usb4_thunderbolt_certification_example.py [--device-type TYPE] [--output-dir OUTPUT_DIR]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to the path so we can import the framework
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from serdes_validation_framework.protocols.usb4 import ThunderboltSpecs, USB4LinkState, USB4SignalMode, USB4TunnelingMode
from serdes_validation_framework.protocols.usb4.thunderbolt import ThunderboltDeviceType
from serdes_validation_framework.test_sequence import USB4LaneConfig, USB4TestPhase, USB4TestSequence, USB4TestSequenceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_thunderbolt_certification_config(
    device_type: ThunderboltDeviceType = ThunderboltDeviceType.HOST_CONTROLLER
) -> USB4TestSequenceConfig:
    """
    Create Thunderbolt 4 certification test configuration
    
    Args:
        device_type: Type of Thunderbolt device being tested
        
    Returns:
        USB4 test sequence configuration for Thunderbolt certification
    """
    # Configure dual lanes for Thunderbolt 4
    lane_configs = [
        USB4LaneConfig(
            lane_id=0,
            mode=USB4SignalMode.GEN3X2,  # Thunderbolt 4 uses Gen 3
            sample_rate=200e9,  # 200 GSa/s
            bandwidth=25e9,     # 25 GHz bandwidth
            voltage_range=0.8,  # ±400mV differential
            enable_ssc=True
        ),
        USB4LaneConfig(
            lane_id=1,
            mode=USB4SignalMode.GEN3X2,
            sample_rate=200e9,
            bandwidth=25e9,
            voltage_range=0.8,
            enable_ssc=True
        )
    ]
    
    # Define comprehensive test phases for certification
    test_phases = [
        USB4TestPhase.INITIALIZATION,
        USB4TestPhase.SIGNAL_ANALYSIS,
        USB4TestPhase.LINK_TRAINING,
        USB4TestPhase.COMPLIANCE,
        USB4TestPhase.TUNNELING,
        USB4TestPhase.POWER_MANAGEMENT,
        USB4TestPhase.PERFORMANCE,
        USB4TestPhase.STRESS_TEST,
        USB4TestPhase.THUNDERBOLT,  # Thunderbolt-specific tests
        USB4TestPhase.VALIDATION
    ]
    
    return USB4TestSequenceConfig(
        test_name=f"Thunderbolt 4 Certification - {device_type.name}",
        lanes=lane_configs,
        test_phases=test_phases,
        tunneling_modes=[
            USB4TunnelingMode.PCIE,
            USB4TunnelingMode.DISPLAYPORT,
            USB4TunnelingMode.USB32
        ],
        stress_duration=300.0,  # 5 minutes for certification
        compliance_patterns=["PRBS31", "PRBS15", "PRBS7"],
        target_ber=1e-15,  # Stricter BER for certification
        enable_thunderbolt=True,
        power_states_to_test=[
            USB4LinkState.U0,
            USB4LinkState.U1,
            USB4LinkState.U2,
            USB4LinkState.U3
        ]
    )


def generate_thunderbolt_test_signals(
    config: USB4TestSequenceConfig,
    duration: float = 50e-6,
    include_security_patterns: bool = True
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Generate Thunderbolt 4 test signals with security and certification patterns
    
    Args:
        config: Test configuration
        duration: Signal duration in seconds
        include_security_patterns: Include security validation patterns
        
    Returns:
        Dictionary with signal data for each lane
    """
    logger.info("Generating Thunderbolt 4 certification test signals")
    
    signal_data = {}
    thunderbolt_specs = ThunderboltSpecs()
    
    for lane_config in config.lanes:
        sample_rate = lane_config.sample_rate
        num_samples = int(duration * sample_rate)
        time = np.linspace(0, duration, num_samples)
        
        # Generate high-quality Thunderbolt signal
        bit_rate = 20e9  # 20 Gbps per lane
        bit_period = 1.0 / bit_rate
        
        # Generate certification test pattern
        num_bits = int(duration / bit_period)
        
        # Use PRBS31 for certification (longer pattern for better validation)
        np.random.seed(42 + lane_config.lane_id)  # Reproducible patterns
        data_bits = np.random.randint(0, 2, num_bits)
        
        # Create high-quality NRZ signal
        voltage = np.zeros(num_samples)
        for i, bit in enumerate(data_bits):
            start_idx = int(i * bit_period * sample_rate)
            end_idx = min(int((i + 1) * bit_period * sample_rate), num_samples)
            if end_idx > start_idx:
                # Use precise Thunderbolt voltage levels
                voltage[start_idx:end_idx] = 0.4 if bit else -0.4
        
        # Add spread spectrum clocking with Thunderbolt specifications
        ssc_freq = thunderbolt_specs.SSC_FREQUENCY
        ssc_deviation = thunderbolt_specs.SSC_DEVIATION
        ssc_modulation = ssc_deviation * np.sin(2 * np.pi * ssc_freq * time)
        
        # Apply SSC modulation
        phase_modulation = np.cumsum(ssc_modulation) * 2 * np.pi * duration / num_samples
        voltage = voltage * (1 + 0.05 * np.sin(phase_modulation))
        
        # Add security pattern markers (for demonstration)
        if include_security_patterns and lane_config.lane_id == 0:
            # Add periodic security validation markers
            marker_period = int(sample_rate * 1e-6)  # Every 1 μs
            for i in range(0, num_samples, marker_period):
                if i + 100 < num_samples:
                    voltage[i:i+100] *= 1.1  # Slight amplitude increase as marker
        
        # Add minimal noise for certification-grade signal
        noise_level = 0.01  # 1% noise for high-quality signal
        noise = noise_level * np.random.randn(num_samples) * np.std(voltage)
        voltage += noise
        
        # Apply realistic channel effects
        if len(voltage) > 20:
            # High-frequency roll-off (channel loss)
            channel_filter = np.array([0.05, 0.15, 0.6, 1.0, 0.6, 0.15, 0.05])
            channel_filter = channel_filter / np.sum(channel_filter)
            voltage = np.convolve(voltage, channel_filter, mode='same')
        
        signal_data[lane_config.lane_id] = {
            'time': time,
            'voltage': voltage
        }
    
    logger.info(f"Generated certification signals for {len(signal_data)} lanes")
    return signal_data


def run_thunderbolt_certification(
    config: USB4TestSequenceConfig,
    signal_data: Dict[int, Dict[str, np.ndarray]],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run Thunderbolt 4 certification sequence
    
    Args:
        config: Test configuration
        signal_data: Signal data for each lane
        output_dir: Output directory for results
        
    Returns:
        Certification results dictionary
    """
    logger.info("Starting Thunderbolt 4 certification sequence")
    
    # Create test sequence
    test_sequence = USB4TestSequence(config)
    
    try:
        # Run complete certification sequence
        results = test_sequence.run_complete_sequence(signal_data)
        
        # Analyze certification results
        certification_results = analyze_certification_results(results)
        
        # Generate certification report
        generate_certification_report(results, certification_results, output_dir)
        
        # Generate certification plots
        generate_certification_plots(signal_data, results, certification_results, output_dir)
        
        logger.info("Thunderbolt 4 certification completed successfully")
        return certification_results
        
    except Exception as e:
        logger.error(f"Thunderbolt 4 certification failed: {e}")
        raise


def analyze_certification_results(results) -> Dict[str, Any]:
    """
    Analyze results for Thunderbolt 4 certification compliance
    
    Args:
        results: Test sequence results
        
    Returns:
        Certification analysis results
    """
    logger.info("Analyzing certification compliance")
    
    certification_results = {
        'overall_certification_status': 'PASS',
        'certification_level': 'Thunderbolt 4 Certified',
        'test_categories': {},
        'critical_failures': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Analyze each test phase for certification compliance
    for phase_result in results.phase_results:
        category_name = phase_result.phase.name.lower()
        
        category_result = {
            'status': phase_result.status.name,
            'duration': phase_result.duration,
            'critical_metrics': {},
            'compliance_score': 0.0
        }
        
        # Extract critical metrics for certification
        if phase_result.phase.name == 'SIGNAL_ANALYSIS':
            # Signal integrity requirements
            for metric, value in phase_result.metrics.items():
                if 'eye_height' in metric:
                    category_result['critical_metrics']['eye_height'] = value
                    if value < 0.5:  # Thunderbolt requirement
                        certification_results['critical_failures'].append(
                            f"Eye height {value:.3f} below Thunderbolt minimum 0.5"
                        )
                elif 'jitter' in metric and 'rms' in metric:
                    category_result['critical_metrics']['rms_jitter'] = value
                    if value > 5e-12:  # 5 ps RMS jitter limit
                        certification_results['critical_failures'].append(
                            f"RMS jitter {value*1e12:.2f} ps exceeds 5 ps limit"
                        )
        
        elif phase_result.phase.name == 'LINK_TRAINING':
            # Link training requirements
            for metric, value in phase_result.metrics.items():
                if 'training_time' in metric:
                    category_result['critical_metrics']['training_time'] = value
                    if value > 0.1:  # 100 ms training time limit
                        certification_results['warnings'].append(
                            f"Link training time {value*1000:.1f} ms exceeds recommended 100 ms"
                        )
        
        elif phase_result.phase.name == 'THUNDERBOLT':
            # Thunderbolt-specific requirements
            for metric, value in phase_result.metrics.items():
                if 'security' in metric:
                    category_result['critical_metrics']['security_score'] = value
                    if value < 0.95:  # 95% security compliance
                        certification_results['critical_failures'].append(
                            f"Security compliance {value*100:.1f}% below required 95%"
                        )
                elif 'daisy_chain' in metric:
                    category_result['critical_metrics']['daisy_chain_support'] = value
        
        # Calculate compliance score for this category
        if phase_result.status.name == 'PASS':
            category_result['compliance_score'] = 1.0
        elif phase_result.status.name == 'WARNING':
            category_result['compliance_score'] = 0.8
        else:
            category_result['compliance_score'] = 0.0
        
        certification_results['test_categories'][category_name] = category_result
    
    # Determine overall certification status
    if certification_results['critical_failures']:
        certification_results['overall_certification_status'] = 'FAIL'
        certification_results['certification_level'] = 'Not Certified'
    elif len(certification_results['warnings']) > 3:
        certification_results['overall_certification_status'] = 'CONDITIONAL'
        certification_results['certification_level'] = 'Conditional Certification'
    
    # Add recommendations
    if not certification_results['critical_failures']:
        certification_results['recommendations'].extend([
            "Signal quality meets Thunderbolt 4 requirements",
            "Link training performance is acceptable",
            "Security features are properly implemented"
        ])
    
    return certification_results


def generate_certification_report(
    results,
    certification_results: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate comprehensive certification report"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / "thunderbolt4_certification_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("THUNDERBOLT 4 CERTIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Executive summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Test Name: {results.config.test_name}\n")
        f.write(f"Certification Status: {certification_results['overall_certification_status']}\n")
        f.write(f"Certification Level: {certification_results['certification_level']}\n")
        f.write(f"Total Test Duration: {results.total_duration:.2f} seconds\n")
        f.write(f"Test Date: {results.config.test_name}\n\n")
        
        # Critical failures
        if certification_results['critical_failures']:
            f.write("CRITICAL FAILURES\n")
            f.write("-" * 30 + "\n")
            for failure in certification_results['critical_failures']:
                f.write(f"❌ {failure}\n")
            f.write("\n")
        
        # Warnings
        if certification_results['warnings']:
            f.write("WARNINGS\n")
            f.write("-" * 30 + "\n")
            for warning in certification_results['warnings']:
                f.write(f"⚠️  {warning}\n")
            f.write("\n")
        
        # Test category results
        f.write("TEST CATEGORY RESULTS\n")
        f.write("-" * 30 + "\n")
        for category, result in certification_results['test_categories'].items():
            status_symbol = "✅" if result['status'] == "PASS" else "❌"
            f.write(f"{status_symbol} {category.upper():<20} {result['status']:<8} "
                   f"(Score: {result['compliance_score']:.2f})\n")
            
            # Critical metrics
            if result['critical_metrics']:
                for metric, value in result['critical_metrics'].items():
                    f.write(f"    {metric}: {value:.6f}\n")
        f.write("\n")
        
        # Recommendations
        if certification_results['recommendations']:
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for recommendation in certification_results['recommendations']:
                f.write(f"• {recommendation}\n")
            f.write("\n")
        
        # Detailed phase results
        f.write("DETAILED PHASE RESULTS\n")
        f.write("-" * 30 + "\n")
        for phase_result in results.phase_results:
            f.write(f"\n{phase_result.phase.name}:\n")
            f.write(f"  Status: {phase_result.status.name}\n")
            f.write(f"  Duration: {phase_result.duration:.3f} seconds\n")
            f.write(f"  Metrics: {len(phase_result.metrics)} measurements\n")
            if phase_result.error_message:
                f.write(f"  Error: {phase_result.error_message}\n")
        
        # Compliance summary
        f.write("\nCOMPLIANCE SUMMARY\n")
        f.write("-" * 30 + "\n")
        for test, result in results.compliance_summary.items():
            f.write(f"{test}: {result}\n")
    
    logger.info(f"Certification report saved to {report_file}")


def generate_certification_plots(
    signal_data: Dict[int, Dict[str, np.ndarray]],
    results,
    certification_results: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate certification-specific plots"""
    try:
        plots_dir = output_dir / "certification_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Certification status summary
        categories = list(certification_results['test_categories'].keys())
        scores = [certification_results['test_categories'][cat]['compliance_score'] 
                 for cat in categories]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(categories, scores)
        
        # Color bars based on score
        for i, score in enumerate(scores):
            if score >= 1.0:
                bars[i].set_color('green')
            elif score >= 0.8:
                bars[i].set_color('orange')
            else:
                bars[i].set_color('red')
        
        plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Certification Threshold')
        plt.xlabel('Test Category')
        plt.ylabel('Compliance Score')
        plt.title('Thunderbolt 4 Certification Compliance Scores')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "certification_compliance_scores.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Signal quality visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        for lane_id, data in signal_data.items():
            time_ns = data['time'] * 1e9  # Convert to nanoseconds
            # Show only first 100 ns for detail
            mask = time_ns <= 100
            axes[lane_id].plot(time_ns[mask], data['voltage'][mask], 'b-', linewidth=0.8)
            axes[lane_id].set_title(f'Thunderbolt 4 Lane {lane_id} Signal Quality')
            axes[lane_id].set_xlabel('Time (ns)')
            axes[lane_id].set_ylabel('Voltage (V)')
            axes[lane_id].grid(True, alpha=0.3)
            
            # Add certification limits
            axes[lane_id].axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Max Level')
            axes[lane_id].axhline(y=-0.4, color='red', linestyle='--', alpha=0.5, label='Min Level')
            axes[lane_id].legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "thunderbolt4_signal_quality.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Certification plots saved to {plots_dir}")
        
    except Exception as e:
        logger.warning(f"Failed to generate certification plots: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Thunderbolt 4 Certification Example")
    parser.add_argument("--device-type", type=str, default="HOST_CONTROLLER",
                       choices=["HOST_CONTROLLER", "DEVICE", "HUB", "DOCK"],
                       help="Type of Thunderbolt device being certified")
    parser.add_argument("--output-dir", type=str, default="thunderbolt4_certification_results",
                       help="Output directory for certification results")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    
    try:
        logger.info("Starting Thunderbolt 4 certification example")
        
        # Map device type string to enum
        device_type_map = {
            "HOST_CONTROLLER": ThunderboltDeviceType.HOST_CONTROLLER,
            "DEVICE": ThunderboltDeviceType.DEVICE,
            "HUB": ThunderboltDeviceType.HUB,
            "DOCK": ThunderboltDeviceType.DOCK
        }
        device_type = device_type_map[args.device_type]
        
        # Create certification configuration
        config = create_thunderbolt_certification_config(device_type)
        
        # Generate certification test signals
        signal_data = generate_thunderbolt_test_signals(
            config,
            duration=100e-6,  # 100 μs for comprehensive testing
            include_security_patterns=True
        )
        
        # Run certification
        certification_results = run_thunderbolt_certification(config, signal_data, output_dir)
        
        # Print certification summary
        print("\n" + "="*70)
        print("THUNDERBOLT 4 CERTIFICATION SUMMARY")
        print("="*70)
        print(f"Device Type: {device_type.name}")
        print(f"Certification Status: {certification_results['overall_certification_status']}")
        print(f"Certification Level: {certification_results['certification_level']}")
        
        if certification_results['critical_failures']:
            print(f"\nCritical Failures: {len(certification_results['critical_failures'])}")
            for failure in certification_results['critical_failures']:
                print(f"  ❌ {failure}")
        
        if certification_results['warnings']:
            print(f"\nWarnings: {len(certification_results['warnings'])}")
            for warning in certification_results['warnings'][:3]:  # Show first 3
                print(f"  ⚠️  {warning}")
        
        print(f"\nDetailed results saved to: {output_dir.absolute()}")
        
        logger.info("Thunderbolt 4 certification example completed successfully")
        
    except Exception as e:
        logger.error(f"Certification example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
