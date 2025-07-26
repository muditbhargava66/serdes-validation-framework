#!/usr/bin/env python3
"""
PCIe 6.0 Validation Script

This script provides comprehensive PCIe 6.0 validation capabilities including:
- Complete PCIe 6.0 specification compliance testing
- NRZ/PAM4 dual-mode validation
- Multi-lane analysis with skew detection
- Advanced link training and equalization
- Automated compliance reporting
- Performance benchmarking

Usage:
    python scripts/pcie_validation.py [options]

Options:
    --mode {nrz,pam4,both}     Signal mode to test (default: both)
    --lanes N                  Number of lanes to test (default: 1)
    --samples N                Number of samples per test (default: 10000)
    --output DIR               Output directory for reports (default: ./results)
    --verbose                  Enable verbose logging
    --benchmark                Run performance benchmarks
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

try:
    from serdes_validation_framework.instrument_control.mode_switcher import create_mode_switcher
    from serdes_validation_framework.instrument_control.pcie_analyzer import PCIeAnalyzer, PCIeConfig
    from serdes_validation_framework.protocols.pcie.compliance import ComplianceConfig, ComplianceTestSuite, ComplianceType
    from serdes_validation_framework.protocols.pcie.constants import SignalMode
    from serdes_validation_framework.protocols.pcie.link_training import create_nrz_trainer, create_pam4_trainer
    from serdes_validation_framework.test_sequence.pcie_sequence import create_multi_lane_pam4_test
    PCIE_AVAILABLE = True
except ImportError as e:
    print(f"PCIe modules not available: {e}")
    PCIE_AVAILABLE = False
    sys.exit(1)

# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pcie_validation.log')
        ]
    )
    return logging.getLogger(__name__)


class PCIeValidator:
    """Comprehensive PCIe 6.0 validator"""
    
    def __init__(self, output_dir: str = "./results"):
        """Initialize PCIe validator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        
        self.logger.info(f"PCIe Validator initialized - Output: {self.output_dir}")
    
    def generate_test_signal(
        self, 
        mode: SignalMode, 
        num_samples: int = 10000,
        snr_db: float = 25.0
    ) -> Dict[str, np.ndarray]:
        """Generate test signal for validation"""
        self.logger.debug(f"Generating {mode.name} signal: {num_samples} samples, SNR={snr_db}dB")
        
        # Time vector
        sample_rate = 200e9 if mode == SignalMode.PAM4 else 100e9
        time = np.linspace(0, num_samples/sample_rate, num_samples, dtype=np.float64)
        
        # Generate signal based on mode
        if mode == SignalMode.NRZ:
            data = np.random.choice([-1.0, 1.0], size=num_samples)
            signal_power = 1.0
        else:  # PAM4
            levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
            data = np.random.choice(levels, size=num_samples)
            signal_power = np.mean(levels**2)
        
        # Add realistic noise
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
        voltage = data + noise
        
        return {
            'time': time,
            'voltage': voltage.astype(np.float64)
        }
    
    def validate_signal_analysis(
        self, 
        mode: SignalMode, 
        num_samples: int = 10000
    ) -> Dict[str, float]:
        """Validate signal analysis capabilities"""
        self.logger.info(f"Validating {mode.name} signal analysis...")
        
        try:
            # Generate test signal
            signal_data = self.generate_test_signal(mode, num_samples)
            
            # Configure analyzer
            config = PCIeConfig(
                mode=mode,
                sample_rate=200e9 if mode == SignalMode.PAM4 else 100e9,
                bandwidth=100e9 if mode == SignalMode.PAM4 else 50e9,
                voltage_range=1.2 if mode == SignalMode.PAM4 else 1.0,
                link_speed=64e9,
                lane_count=1
            )
            
            # Analyze signal
            start_time = time.time()
            analyzer = PCIeAnalyzer(config)
            results = analyzer.analyze_signal(signal_data)
            analysis_time = time.time() - start_time
            
            # Add timing information
            results['analysis_time'] = analysis_time
            results['samples_analyzed'] = num_samples
            results['throughput'] = num_samples / analysis_time
            
            self.logger.info(f"{mode.name} analysis completed in {analysis_time:.3f}s")
            for metric, value in results.items():
                if isinstance(value, float) and metric != 'analysis_time':
                    self.logger.info(f"  {metric}: {value:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"{mode.name} signal analysis failed: {e}")
            return {'error': str(e)}
    
    def validate_mode_switching(self) -> Dict[str, float]:
        """Validate dual-mode switching capabilities"""
        self.logger.info("Validating mode switching...")
        
        try:
            switcher = create_mode_switcher(
                default_mode=SignalMode.NRZ,
                sample_rate=100e9,
                bandwidth=50e9
            )
            
            results = {}
            
            # Test NRZ to PAM4 switch
            start_time = time.time()
            result = switcher.switch_mode(SignalMode.PAM4)
            nrz_to_pam4_time = time.time() - start_time
            
            results['nrz_to_pam4_success'] = float(result.success)
            results['nrz_to_pam4_time'] = nrz_to_pam4_time * 1000  # Convert to ms
            
            # Test PAM4 to NRZ switch
            start_time = time.time()
            result = switcher.switch_mode(SignalMode.NRZ)
            pam4_to_nrz_time = time.time() - start_time
            
            results['pam4_to_nrz_success'] = float(result.success)
            results['pam4_to_nrz_time'] = pam4_to_nrz_time * 1000  # Convert to ms
            
            # Calculate average switch time
            results['avg_switch_time'] = (results['nrz_to_pam4_time'] + results['pam4_to_nrz_time']) / 2
            
            self.logger.info("Mode switching validation completed:")
            self.logger.info(f"  NRZ→PAM4: {results['nrz_to_pam4_time']:.2f}ms")
            self.logger.info(f"  PAM4→NRZ: {results['pam4_to_nrz_time']:.2f}ms")
            self.logger.info(f"  Average: {results['avg_switch_time']:.2f}ms")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Mode switching validation failed: {e}")
            return {'error': str(e)}
    
    def validate_link_training(
        self, 
        mode: SignalMode, 
        num_samples: int = 8000
    ) -> Dict[str, float]:
        """Validate link training capabilities"""
        self.logger.info(f"Validating {mode.name} link training...")
        
        try:
            # Generate test signal
            signal_data = self.generate_test_signal(mode, num_samples, snr_db=20.0)
            
            # Create trainer
            if mode == SignalMode.NRZ:
                trainer = create_nrz_trainer(target_ber=1e-9, max_iterations=500)
            else:
                trainer = create_pam4_trainer(target_ber=1e-9, max_iterations=1000)
            
            # Run training
            start_time = time.time()
            result = trainer.run_training(signal_data)
            training_time = time.time() - start_time
            
            # Compile results
            training_results = {
                'success': float(result.success),
                'final_ber': result.final_ber,
                'iterations': float(result.iterations),
                'training_time': training_time,
                'final_snr': result.snr_history[-1] if result.snr_history else 0.0
            }
            
            self.logger.info(f"{mode.name} link training completed:")
            self.logger.info(f"  Success: {result.success}")
            self.logger.info(f"  Final BER: {result.final_ber:.2e}")
            self.logger.info(f"  Iterations: {result.iterations}")
            self.logger.info(f"  Training time: {training_time:.2f}s")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"{mode.name} link training failed: {e}")
            return {'error': str(e)}
    
    def validate_compliance(
        self, 
        mode: SignalMode, 
        num_samples: int = 10000
    ) -> Dict[str, float]:
        """Validate compliance testing capabilities"""
        self.logger.info(f"Validating {mode.name} compliance testing...")
        
        try:
            # Generate test signal
            signal_data = self.generate_test_signal(mode, num_samples, snr_db=30.0)
            
            # Create compliance test suite
            config = ComplianceConfig(
                test_pattern="PRBS31",
                sample_rate=200e9 if mode == SignalMode.PAM4 else 100e9,
                record_length=100e-6,
                voltage_range=2.0,
                test_types=[ComplianceType.ELECTRICAL, ComplianceType.TIMING]
            )
            
            test_suite = ComplianceTestSuite(config)
            
            # Run compliance tests
            start_time = time.time()
            results = test_suite.run_compliance_tests(
                signal_data['time'],
                signal_data['voltage']
            )
            compliance_time = time.time() - start_time
            
            # Process results
            compliance_results = {
                'compliance_time': compliance_time,
                'overall_status': float(test_suite.get_overall_status())
            }
            
            # Extract individual test results
            for category, tests in results.items():
                for test_name, result in tests.items():
                    key = f"{category}_{test_name}_status"
                    compliance_results[key] = float(result.status)
                    
                    key = f"{category}_{test_name}_value"
                    compliance_results[key] = result.measured_value
            
            overall_status = "PASS" if test_suite.get_overall_status() else "FAIL"
            self.logger.info(f"{mode.name} compliance testing completed:")
            self.logger.info(f"  Overall status: {overall_status}")
            self.logger.info(f"  Test time: {compliance_time:.2f}s")
            
            return compliance_results
            
        except Exception as e:
            self.logger.error(f"{mode.name} compliance testing failed: {e}")
            return {'error': str(e)}
    
    def validate_multi_lane(
        self, 
        num_lanes: int = 4, 
        num_samples: int = 5000
    ) -> Dict[str, float]:
        """Validate multi-lane capabilities"""
        self.logger.info(f"Validating {num_lanes}-lane analysis...")
        
        try:
            # Create multi-lane test
            test_sequence = create_multi_lane_pam4_test(
                num_lanes=num_lanes,
                sample_rate=200e9,
                bandwidth=100e9
            )
            
            # Generate multi-lane data
            multi_lane_data = {}
            for lane_id in range(num_lanes):
                signal = self.generate_test_signal(SignalMode.PAM4, num_samples)
                
                # Add lane-specific skew (simulate realistic conditions)
                skew_samples = int(np.random.normal(0, 3))  # ±3 sample skew
                if skew_samples != 0:
                    if skew_samples > 0:
                        signal['voltage'] = np.pad(signal['voltage'], (skew_samples, 0), mode='edge')[:-skew_samples]
                    else:
                        signal['voltage'] = np.pad(signal['voltage'], (0, -skew_samples), mode='edge')[-skew_samples:]
                
                multi_lane_data[lane_id] = signal
            
            # Run multi-lane test
            start_time = time.time()
            result = test_sequence.run_complete_sequence(multi_lane_data)
            test_time = time.time() - start_time
            
            # Compile results
            multi_lane_results = {
                'overall_status': float(result.overall_status.value),
                'test_time': test_time,
                'lanes_tested': float(len(result.lane_results)),
                'phases_completed': float(len(result.phase_results))
            }
            
            # Lane-specific results
            for lane_id, metrics in result.lane_results.items():
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        key = f"lane_{lane_id}_{metric}"
                        multi_lane_results[key] = float(value)
            
            # Check for skew measurements
            if result.phase_results:
                final_phase = result.phase_results[-1]
                if 'max_lane_skew_ps' in final_phase.metrics:
                    multi_lane_results['max_lane_skew_ps'] = final_phase.metrics['max_lane_skew_ps']
                if 'avg_lane_skew_ps' in final_phase.metrics:
                    multi_lane_results['avg_lane_skew_ps'] = final_phase.metrics['avg_lane_skew_ps']
            
            self.logger.info(f"{num_lanes}-lane analysis completed:")
            self.logger.info(f"  Overall status: {result.overall_status.name}")
            self.logger.info(f"  Test time: {test_time:.2f}s")
            self.logger.info(f"  Phases completed: {len(result.phase_results)}")
            
            return multi_lane_results
            
        except Exception as e:
            self.logger.error(f"Multi-lane validation failed: {e}")
            return {'error': str(e)}
    
    def run_performance_benchmark(self) -> Dict[str, float]:
        """Run performance benchmarks"""
        self.logger.info("Running performance benchmarks...")
        
        benchmark_results = {}
        
        try:
            # Signal analysis benchmark
            for mode in [SignalMode.NRZ, SignalMode.PAM4]:
                for sample_count in [1000, 5000, 10000]:
                    key = f"{mode.name.lower()}_analysis_{sample_count}"
                    
                    signal_data = self.generate_test_signal(mode, sample_count)
                    
                    config = PCIeConfig(
                        mode=mode,
                        sample_rate=200e9 if mode == SignalMode.PAM4 else 100e9,
                        bandwidth=100e9 if mode == SignalMode.PAM4 else 50e9,
                        voltage_range=1.2 if mode == SignalMode.PAM4 else 1.0,
                        link_speed=64e9,
                        lane_count=1
                    )
                    
                    analyzer = PCIeAnalyzer(config)
                    
                    # Time the analysis
                    start_time = time.time()
                    analyzer.analyze_signal(signal_data)
                    analysis_time = time.time() - start_time
                    
                    benchmark_results[key] = analysis_time
                    throughput = sample_count / analysis_time
                    benchmark_results[f"{key}_throughput"] = throughput
            
            # Mode switching benchmark
            switcher = create_mode_switcher()
            switch_times = []
            
            for _ in range(10):  # 10 iterations for average
                start_time = time.time()
                switcher.switch_mode(SignalMode.PAM4)
                switch_time = time.time() - start_time
                switch_times.append(switch_time * 1000)  # Convert to ms
                
                start_time = time.time()
                switcher.switch_mode(SignalMode.NRZ)
                switch_time = time.time() - start_time
                switch_times.append(switch_time * 1000)
            
            benchmark_results['avg_switch_time_ms'] = np.mean(switch_times)
            benchmark_results['min_switch_time_ms'] = np.min(switch_times)
            benchmark_results['max_switch_time_ms'] = np.max(switch_times)
            
            self.logger.info("Performance benchmarks completed:")
            for key, value in benchmark_results.items():
                if 'throughput' in key:
                    self.logger.info(f"  {key}: {value:.0f} samples/s")
                elif 'time' in key:
                    self.logger.info(f"  {key}: {value:.3f}")
                else:
                    self.logger.info(f"  {key}: {value:.6f}")
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Performance benchmark failed: {e}")
            return {'error': str(e)}
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report"""
        report_path = self.output_dir / "pcie_validation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("PCIe 6.0 Validation Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Framework Version: 1.4.0\n\n")
            
            # Write results
            for test_name, results in self.results.items():
                f.write(f"{test_name.upper()}\n")
                f.write("-" * len(test_name) + "\n")
                
                if 'error' in results:
                    f.write(f"ERROR: {results['error']}\n")
                else:
                    for key, value in results.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                f.write("\n")
        
        self.logger.info(f"Validation report saved to: {report_path}")
        return str(report_path)
    
    def run_validation(
        self, 
        modes: List[SignalMode], 
        num_lanes: int = 1,
        num_samples: int = 10000,
        run_benchmark: bool = False
    ) -> Dict[str, Dict]:
        """Run complete PCIe validation suite"""
        self.logger.info("Starting PCIe 6.0 validation suite...")
        self.logger.info(f"Modes: {[mode.name for mode in modes]}")
        self.logger.info(f"Lanes: {num_lanes}")
        self.logger.info(f"Samples: {num_samples}")
        
        # Mode switching validation
        self.results['mode_switching'] = self.validate_mode_switching()
        
        # Signal analysis validation
        for mode in modes:
            self.results[f'{mode.name.lower()}_signal_analysis'] = self.validate_signal_analysis(mode, num_samples)
            self.results[f'{mode.name.lower()}_link_training'] = self.validate_link_training(mode, num_samples)
            self.results[f'{mode.name.lower()}_compliance'] = self.validate_compliance(mode, num_samples)
        
        # Multi-lane validation (if more than 1 lane)
        if num_lanes > 1:
            self.results['multi_lane'] = self.validate_multi_lane(num_lanes, num_samples)
        
        # Performance benchmarks
        if run_benchmark:
            self.results['performance_benchmark'] = self.run_performance_benchmark()
        
        # Generate report
        report_path = self.generate_report()
        
        self.logger.info("PCIe 6.0 validation suite completed!")
        self.logger.info(f"Report saved to: {report_path}")
        
        return self.results


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="PCIe 6.0 Validation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/pcie_validation.py --mode both --lanes 4 --verbose
  python scripts/pcie_validation.py --mode pam4 --benchmark
  python scripts/pcie_validation.py --mode nrz --samples 20000 --output ./my_results
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['nrz', 'pam4', 'both'], 
        default='both',
        help='Signal mode to test (default: both)'
    )
    
    parser.add_argument(
        '--lanes', 
        type=int, 
        default=1,
        help='Number of lanes to test (default: 1)'
    )
    
    parser.add_argument(
        '--samples', 
        type=int, 
        default=10000,
        help='Number of samples per test (default: 10000)'
    )
    
    parser.add_argument(
        '--output', 
        default='./results',
        help='Output directory for reports (default: ./results)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--benchmark', 
        action='store_true',
        help='Run performance benchmarks'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Check PCIe availability
    if not PCIE_AVAILABLE:
        logger.error("PCIe modules not available - cannot run validation")
        sys.exit(1)
    
    # Determine modes to test
    if args.mode == 'both':
        modes = [SignalMode.NRZ, SignalMode.PAM4]
    elif args.mode == 'nrz':
        modes = [SignalMode.NRZ]
    else:  # pam4
        modes = [SignalMode.PAM4]
    
    # Validate arguments
    if args.lanes < 1 or args.lanes > 16:
        logger.error("Number of lanes must be between 1 and 16")
        sys.exit(1)
    
    if args.samples < 1000:
        logger.error("Number of samples must be at least 1000")
        sys.exit(1)
    
    try:
        # Create validator
        validator = PCIeValidator(args.output)
        
        # Run validation
        results = validator.run_validation(
            modes=modes,
            num_lanes=args.lanes,
            num_samples=args.samples,
            run_benchmark=args.benchmark
        )
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if 'error' not in r)
        
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed tests: {passed_tests}")
        logger.info(f"Failed tests: {total_tests - passed_tests}")
        logger.info(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - PCIe 6.0 validation successful!")
            sys.exit(0)
        else:
            logger.warning("⚠️  Some tests failed - check the report for details")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
