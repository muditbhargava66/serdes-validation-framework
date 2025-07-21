#!/usr/bin/env python3
"""
PCIe Validation Example

This example demonstrates comprehensive PCIe 6.0 validation including:
- Dual-mode operation (NRZ/PAM4)
- Link training and equalization
- Compliance testing
- Signal analysis

Usage:
    python examples/pcie_example.py
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
from serdes_validation_framework.instrument_control.mode_switcher import create_mode_switcher
from serdes_validation_framework.instrument_control.pcie_analyzer import PCIeAnalyzer, PCIeConfig
from serdes_validation_framework.protocols.pcie.compliance import ComplianceConfig, ComplianceTestSuite, ComplianceType
from serdes_validation_framework.protocols.pcie.constants import PCIE_SPECS, SignalMode
from serdes_validation_framework.protocols.pcie.equalization import create_lms_equalizer, create_rls_equalizer
from serdes_validation_framework.protocols.pcie.link_training import create_nrz_trainer, create_pam4_trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_signal(
    mode: SignalMode,
    num_samples: int = 10000,
    sample_rate: float = 100e9,
    snr_db: float = 25.0
) -> dict[str, np.ndarray]:
    """
    Generate test signal for PCIe validation
    
    Args:
        mode: Signal mode (NRZ or PAM4)
        num_samples: Number of samples
        sample_rate: Sample rate in Hz
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Dictionary with time and voltage arrays
        
    Raises:
        AssertionError: If parameters are invalid
    """
    # Validate inputs
    assert isinstance(mode, SignalMode), f"Mode must be SignalMode, got {type(mode)}"
    assert isinstance(num_samples, int), f"Num samples must be int, got {type(num_samples)}"
    assert isinstance(sample_rate, float), f"Sample rate must be float, got {type(sample_rate)}"
    assert isinstance(snr_db, float), f"SNR must be float, got {type(snr_db)}"
    
    assert num_samples > 0, f"Num samples must be positive, got {num_samples}"
    assert sample_rate > 0, f"Sample rate must be positive, got {sample_rate}"
    
    logger.info(f"Generating {mode.name} test signal: {num_samples} samples at {sample_rate/1e9:.1f} GSa/s")
    
    # Time vector
    time = np.linspace(0, num_samples/sample_rate, num_samples, dtype=np.float64)
    
    # Generate data pattern
    if mode == SignalMode.NRZ:
        # NRZ signal: +/-1V levels
        data_bits = np.random.choice([-1.0, 1.0], size=num_samples)
        signal_power = 1.0
    else:  # PAM4
        # PAM4 signal: 4 levels
        pam4_levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
        data_symbols = np.random.choice(pam4_levels, size=num_samples)
        data_bits = data_symbols
        signal_power = np.mean(pam4_levels**2)
    
    # Add noise based on SNR
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), num_samples).astype(np.float64)
    
    # Combine signal and noise
    voltage = data_bits + noise
    
    return {
        'time': time,
        'voltage': voltage
    }


def demonstrate_mode_switching():
    """Demonstrate dual-mode switching capabilities"""
    logger.info("=== PCIe Dual-Mode Switching Demo ===")
    
    try:
        # Create mode switcher
        mode_switcher = create_mode_switcher(
            default_mode=SignalMode.NRZ,
            sample_rate=50e9,
            bandwidth=25e9
        )
        
        logger.info(f"Initial mode: {mode_switcher.get_current_mode().name}")
        
        # Switch to PAM4
        result = mode_switcher.switch_mode(SignalMode.PAM4)
        logger.info(f"Switch to PAM4: {'SUCCESS' if result.success else 'FAILED'}")
        logger.info(f"Switch time: {result.switch_time*1000:.2f} ms")
        
        # Switch back to NRZ
        result = mode_switcher.switch_mode(SignalMode.NRZ)
        logger.info(f"Switch to NRZ: {'SUCCESS' if result.success else 'FAILED'}")
        
        logger.info("Mode switching demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Mode switching demo failed: {e}")
        raise


def demonstrate_signal_analysis():
    """Demonstrate PCIe signal analysis"""
    logger.info("=== PCIe Signal Analysis Demo ===")
    
    try:
        # Test both modes
        for mode in [SignalMode.NRZ, SignalMode.PAM4]:
            logger.info(f"\nAnalyzing {mode.name} signal...")
            
            # Generate test signal
            signal_data = generate_test_signal(mode, num_samples=5000, snr_db=20.0)
            
            # Create analyzer
            config = PCIeConfig(
                mode=mode,
                sample_rate=100e9,
                bandwidth=50e9,
                voltage_range=2.0,
                link_speed=32e9,
                lane_count=1
            )
            
            analyzer = PCIeAnalyzer(config)
            
            # Analyze signal
            results = analyzer.analyze_signal(signal_data)
            
            # Display results
            logger.info(f"Analysis results for {mode.name}:")
            for metric, value in results.items():
                logger.info(f"  {metric}: {value:.3f}")
        
        logger.info("Signal analysis demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Signal analysis demo failed: {e}")
        raise


def demonstrate_link_training():
    """Demonstrate PCIe link training"""
    logger.info("=== PCIe Link Training Demo ===")
    
    try:
        # Test both modes
        for mode in [SignalMode.NRZ, SignalMode.PAM4]:
            logger.info(f"\nTraining {mode.name} link...")
            
            # Generate test signal
            signal_data = generate_test_signal(mode, num_samples=8000, snr_db=15.0)
            
            # Create trainer
            if mode == SignalMode.NRZ:
                trainer = create_nrz_trainer(target_ber=1e-9, max_iterations=500)
            else:
                trainer = create_pam4_trainer(target_ber=1e-9, max_iterations=1000)
            
            # Run training
            result = trainer.run_training(signal_data)
            
            # Display results
            logger.info(f"Training results for {mode.name}:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Final BER: {result.final_ber:.2e}")
            logger.info(f"  Iterations: {result.iterations}")
            logger.info(f"  Final SNR: {result.snr_history[-1]:.1f} dB")
        
        logger.info("Link training demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Link training demo failed: {e}")
        raise


def demonstrate_equalization():
    """Demonstrate adaptive equalization"""
    logger.info("=== PCIe Equalization Demo ===")
    
    try:
        # Generate test signal with ISI
        signal_data = generate_test_signal(SignalMode.NRZ, num_samples=5000, snr_db=20.0)
        
        # Add ISI (inter-symbol interference)
        isi_filter = np.array([0.1, 0.8, 0.1], dtype=np.float64)
        distorted_signal = np.convolve(signal_data['voltage'], isi_filter, mode='same')
        
        # Test different equalizers
        equalizers = [
            ("LMS", create_lms_equalizer(num_forward_taps=7, step_size=0.01)),
            ("RLS", create_rls_equalizer(num_forward_taps=7, forgetting_factor=0.99))
        ]
        
        for name, equalizer in equalizers:
            logger.info(f"\nTesting {name} equalizer...")
            
            # Run equalization
            result = equalizer.equalize_signal(distorted_signal)
            
            logger.info(f"{name} results:")
            logger.info(f"  Converged: {result.converged}")
            logger.info(f"  Final MSE: {result.final_mse:.6f}")
            logger.info(f"  Iterations: {result.iterations}")
        
        logger.info("Equalization demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Equalization demo failed: {e}")
        raise


def demonstrate_compliance_testing():
    """Demonstrate PCIe compliance testing"""
    logger.info("=== PCIe Compliance Testing Demo ===")
    
    try:
        # Generate test signal
        signal_data = generate_test_signal(SignalMode.NRZ, num_samples=10000, snr_db=25.0)
        
        # Create compliance test suite
        config = ComplianceConfig(
            test_pattern="PRBS31",
            sample_rate=100e9,
            record_length=100e-6,
            voltage_range=2.0,
            test_types=[ComplianceType.FULL]
        )
        
        test_suite = ComplianceTestSuite(config)
        
        # Run compliance tests
        results = test_suite.run_compliance_tests(
            signal_data['time'],
            signal_data['voltage']
        )
        
        # Display results
        logger.info("Compliance test results:")
        for category, tests in results.items():
            logger.info(f"\n{category.upper()} Tests:")
            for test_name, result in tests.items():
                status = "PASS" if result.status else "FAIL"
                logger.info(f"  {test_name}: {status} ({result.measured_value:.3f})")
        
        # Overall status
        overall_status = test_suite.get_overall_status()
        logger.info(f"\nOverall compliance: {'PASS' if overall_status else 'FAIL'}")
        
        logger.info("Compliance testing demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Compliance testing demo failed: {e}")
        raise


def plot_signal_comparison():
    """Plot NRZ vs PAM4 signal comparison"""
    logger.info("=== Signal Comparison Plot ===")
    
    try:
        # Generate signals
        nrz_data = generate_test_signal(SignalMode.NRZ, num_samples=1000, snr_db=30.0)
        pam4_data = generate_test_signal(SignalMode.PAM4, num_samples=1000, snr_db=30.0)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # NRZ plot
        ax1.plot(nrz_data['time'][:200] * 1e9, nrz_data['voltage'][:200], 'b-', linewidth=1.5)
        ax1.set_title('PCIe NRZ Signal')
        ax1.set_ylabel('Voltage (V)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-2, 2)
        
        # PAM4 plot
        ax2.plot(pam4_data['time'][:200] * 1e9, pam4_data['voltage'][:200], 'r-', linewidth=1.5)
        ax2.set_title('PCIe PAM4 Signal')
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Voltage (V)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-4, 4)
        
        plt.tight_layout()
        plt.savefig('pcie_signal_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("Signal comparison plot saved as 'pcie_signal_comparison.png'")
        
    except Exception as e:
        logger.error(f"Plotting failed: {e}")


def main():
    """Main demonstration function"""
    logger.info("Starting PCIe Validation Framework Demonstration")
    logger.info(f"PCIe 6.0 Specifications: {PCIE_SPECS['base'].GEN6_RATE/1e9:.1f} GT/s")
    
    try:
        # Run all demonstrations
        demonstrate_mode_switching()
        demonstrate_signal_analysis()
        demonstrate_link_training()
        demonstrate_equalization()
        demonstrate_compliance_testing()
        
        # Generate plots if matplotlib is available
        try:
            plot_signal_comparison()
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")
        
        logger.info("\n" + "="*60)
        logger.info("PCIe Validation Framework Demonstration COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
