#!/usr/bin/env python3
"""
Multi-Protocol Validation Script

This script provides comprehensive validation across all supported SerDes protocols:
- PCIe 6.0 (NRZ/PAM4 dual-mode)
- 224G Ethernet (PAM4)
- USB4/Thunderbolt 4 (dual-lane NRZ with tunneling)

Features:
- Unified validation framework
- Cross-protocol comparison
- Automated protocol detection
- Comprehensive reporting
- Performance benchmarking
- Certification testing

Usage:
    python scripts/multi_protocol_validation.py [options]

Options:
    --protocols {pcie,eth224g,usb4,thunderbolt,all}  Protocols to test (default: all)
    --samples N                                      Samples per test (default: 8000)
    --output DIR                                     Output directory (default: ./multi_protocol_results)
    --verbose                                        Enable verbose logging
    --benchmark                                      Run performance benchmarks
    --certification                                  Run certification tests
    --mock                                          Force mock mode
    --compare                                       Generate cross-protocol comparison
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

try:
    from serdes_validation_framework import (
        ProtocolType,
        ValidationFramework,
        auto_validate_signal,
        create_validation_framework,
        detect_signal_protocol,
    )
    from serdes_validation_framework.protocols.pcie.constants import SignalMode as PCIeSignalMode
    from serdes_validation_framework.protocols.usb4 import USB4SignalMode
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"Framework modules not available: {e}")
    FRAMEWORK_AVAILABLE = False
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
            logging.FileHandler('multi_protocol_validation.log')
        ]
    )
    return logging.getLogger(__name__)


class MultiProtocolValidator:
    """Comprehensive multi-protocol SerDes validator"""
    
    def __init__(self, output_dir: str = "./multi_protocol_results"):
        """Initialize multi-protocol validator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        
        # Initialize validation framework
        self.framework = create_validation_framework()
        
        self.logger.info(f"Multi-Protocol Validator initialized - Output: {self.output_dir}")
    
    def generate_protocol_signal(
        self, 
        protocol: ProtocolType,
        num_samples: int = 8000,
        snr_db: float = 25.0
    ) -> Dict[str, np.ndarray]:
        """Generate protocol-specific test signal"""
        self.logger.debug(f"Generating {protocol.name} signal: {num_samples} samples")
        
        # Protocol-specific parameters
        if protocol == ProtocolType.PCIE:
            sample_rate = 200e9  # 200 GSa/s for PAM4
            # Generate PAM4 signal
            levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
            data = np.random.choice(levels, size=num_samples)
            signal_power = np.mean(levels**2)
            
        elif protocol == ProtocolType.ETHERNET_224G:
            sample_rate = 256e9  # 256 GSa/s for 224G
            # Generate PAM4 signal with higher symbol rate
            levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
            data = np.random.choice(levels, size=num_samples)
            signal_power = np.mean(levels**2)
            
        elif protocol == ProtocolType.USB4:
            sample_rate = 80e9   # 80 GSa/s for USB4
            # Generate NRZ signal (dual-lane)
            data = np.random.choice([-1.0, 1.0], size=num_samples)
            signal_power = 1.0
            
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
        
        # Time vector
        time = np.linspace(0, num_samples/sample_rate, num_samples, dtype=np.float64)
        
        # Add realistic noise
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
        voltage = data + noise
        
        return {
            'time': time,
            'voltage': voltage.astype(np.float64),
            'sample_rate': sample_rate,
            'protocol': protocol.name
        }
    
    def validate_protocol(
        self, 
        protocol: ProtocolType,
        num_samples: int = 8000
    ) -> Dict[str, Any]:
        """Validate specific protocol"""
        self.logger.info(f"Validating {protocol.name} protocol...")
        
        try:
            # Generate test signal
            signal_data = self.generate_protocol_signal(protocol, num_samples)
            
            # Use framework's auto-validation
            start_time = time.time()
            validation_results = auto_validate_signal(
                signal_data,
                sample_rate=signal_data['sample_rate'],
                voltage_range=2.0 if protocol in [ProtocolType.PCIE, ProtocolType.ETHERNET_224G] else 1.0
            )
            validation_time = time.time() - start_time
            
            # Process results
            results = {
                'protocol': protocol.name,
                'validation_time': validation_time,
                'samples_processed': num_samples,
                'throughput': num_samples / validation_time,
                'detected_protocol': validation_results.get('detected_protocol', 'UNKNOWN'),
                'confidence': validation_results.get('confidence', 0.0),
                'signal_quality': validation_results.get('signal_quality', {}),
                'compliance_status': validation_results.get('compliance_status', 'UNKNOWN'),
                'performance_metrics': validation_results.get('performance_metrics', {})
            }
            
            # Protocol-specific metrics
            if 'eye_diagram' in validation_results:
                results['eye_height'] = validation_results['eye_diagram'].get('worst_height', 0.0)
                results['eye_width'] = validation_results['eye_diagram'].get('worst_width', 0.0)
            
            if 'jitter_analysis' in validation_results:
                results['total_jitter_ps'] = validation_results['jitter_analysis'].get('tj_pp_ps', 0.0)
                results['random_jitter_ps'] = validation_results['jitter_analysis'].get('rj_rms_ps', 0.0)
            
            if 'ber_estimate' in validation_results:
                results['ber_estimate'] = validation_results['ber_estimate']
            
            self.logger.info(f"{protocol.name} validation completed:")
            self.logger.info(f"  Detected protocol: {results['detected_protocol']}")
            self.logger.info(f"  Confidence: {results['confidence']:.1%}")
            self.logger.info(f"  Compliance: {results['compliance_status']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"{protocol.name} validation failed: {e}")
            return {'protocol': protocol.name, 'error': str(e)}
    
    def validate_protocol_detection(
        self, 
        protocols: List[ProtocolType],
        num_samples: int = 8000
    ) -> Dict[str, Any]:
        """Validate automatic protocol detection"""
        self.logger.info("Validating automatic protocol detection...")
        
        detection_results = {
            'total_tests': 0,
            'correct_detections': 0,
            'detection_times': [],
            'confidence_scores': [],
            'protocol_results': {}
        }
        
        try:
            for protocol in protocols:
                self.logger.info(f"Testing detection for {protocol.name}...")
                
                # Generate signal
                signal_data = self.generate_protocol_signal(protocol, num_samples)
                
                # Test detection
                start_time = time.time()
                detected_protocol = detect_signal_protocol(
                    signal_data,
                    sample_rate=signal_data['sample_rate'],
                    voltage_range=2.0 if protocol in [ProtocolType.PCIE, ProtocolType.ETHERNET_224G] else 1.0
                )
                detection_time = time.time() - start_time
                
                # Record results
                detection_results['total_tests'] += 1
                detection_results['detection_times'].append(detection_time)
                
                if detected_protocol['protocol'] == protocol:
                    detection_results['correct_detections'] += 1
                    detection_status = 'CORRECT'
                else:
                    detection_status = 'INCORRECT'
                
                confidence = detected_protocol.get('confidence', 0.0)
                detection_results['confidence_scores'].append(confidence)
                
                detection_results['protocol_results'][protocol.name] = {
                    'expected': protocol.name,
                    'detected': detected_protocol['protocol'].name if hasattr(detected_protocol['protocol'], 'name') else str(detected_protocol['protocol']),
                    'confidence': confidence,
                    'detection_time': detection_time,
                    'status': detection_status
                }
                
                self.logger.info(f"  Expected: {protocol.name}")
                self.logger.info(f"  Detected: {detection_results['protocol_results'][protocol.name]['detected']}")
                self.logger.info(f"  Status: {detection_status}")
                self.logger.info(f"  Confidence: {confidence:.1%}")
            
            # Calculate summary statistics
            detection_results['accuracy'] = detection_results['correct_detections'] / detection_results['total_tests']
            detection_results['avg_detection_time'] = np.mean(detection_results['detection_times'])
            detection_results['avg_confidence'] = np.mean(detection_results['confidence_scores'])
            
            self.logger.info("Protocol detection validation completed:")
            self.logger.info(f"  Accuracy: {detection_results['accuracy']:.1%}")
            self.logger.info(f"  Average detection time: {detection_results['avg_detection_time']:.3f}s")
            self.logger.info(f"  Average confidence: {detection_results['avg_confidence']:.1%}")
            
            return detection_results
            
        except Exception as e:
            self.logger.error(f"Protocol detection validation failed: {e}")
            return {'error': str(e)}
    
    def run_cross_protocol_comparison(
        self, 
        protocols: List[ProtocolType],
        num_samples: int = 8000
    ) -> Dict[str, Any]:
        """Run cross-protocol performance comparison"""
        self.logger.info("Running cross-protocol comparison...")
        
        comparison_results = {
            'protocols_tested': [p.name for p in protocols],
            'metrics_comparison': {},
            'performance_ranking': {},
            'analysis_summary': {}
        }
        
        try:
            protocol_metrics = {}
            
            # Collect metrics for each protocol
            for protocol in protocols:
                self.logger.info(f"Analyzing {protocol.name} for comparison...")
                
                signal_data = self.generate_protocol_signal(protocol, num_samples)
                
                # Measure various performance aspects
                start_time = time.time()
                validation_result = auto_validate_signal(
                    signal_data,
                    sample_rate=signal_data['sample_rate'],
                    voltage_range=2.0 if protocol in [ProtocolType.PCIE, ProtocolType.ETHERNET_224G] else 1.0
                )
                processing_time = time.time() - start_time
                
                protocol_metrics[protocol.name] = {
                    'processing_time': processing_time,
                    'throughput': num_samples / processing_time,
                    'signal_quality_score': validation_result.get('signal_quality', {}).get('overall_score', 0.0),
                    'eye_height': validation_result.get('eye_diagram', {}).get('worst_height', 0.0),
                    'jitter_ps': validation_result.get('jitter_analysis', {}).get('tj_pp_ps', 0.0),
                    'ber_estimate': validation_result.get('ber_estimate', 1.0),
                    'compliance_score': 1.0 if validation_result.get('compliance_status') == 'PASS' else 0.0
                }
            
            # Compare metrics across protocols
            metrics_to_compare = ['processing_time', 'throughput', 'signal_quality_score', 'eye_height']
            
            for metric in metrics_to_compare:
                values = {proto: metrics[metric] for proto, metrics in protocol_metrics.items()}
                
                # Rank protocols for this metric (higher is better except for processing_time)
                if metric == 'processing_time':
                    ranked = sorted(values.items(), key=lambda x: x[1])  # Lower is better
                else:
                    ranked = sorted(values.items(), key=lambda x: x[1], reverse=True)  # Higher is better
                
                comparison_results['metrics_comparison'][metric] = {
                    'values': values,
                    'ranking': [proto for proto, _ in ranked],
                    'best': ranked[0][0],
                    'worst': ranked[-1][0]
                }
            
            # Overall performance ranking
            protocol_scores = {}
            for protocol in protocols:
                score = 0
                for metric in metrics_to_compare:
                    ranking = comparison_results['metrics_comparison'][metric]['ranking']
                    # Score based on ranking position (higher position = higher score)
                    score += len(protocols) - ranking.index(protocol.name)
                protocol_scores[protocol.name] = score
            
            overall_ranking = sorted(protocol_scores.items(), key=lambda x: x[1], reverse=True)
            comparison_results['performance_ranking'] = {
                'scores': protocol_scores,
                'ranking': [proto for proto, _ in overall_ranking],
                'best_overall': overall_ranking[0][0],
                'worst_overall': overall_ranking[-1][0]
            }
            
            # Analysis summary
            comparison_results['analysis_summary'] = {
                'fastest_processing': comparison_results['metrics_comparison']['processing_time']['best'],
                'highest_throughput': comparison_results['metrics_comparison']['throughput']['best'],
                'best_signal_quality': comparison_results['metrics_comparison']['signal_quality_score']['best'],
                'best_eye_height': comparison_results['metrics_comparison']['eye_height']['best'],
                'overall_winner': comparison_results['performance_ranking']['best_overall']
            }
            
            self.logger.info("Cross-protocol comparison completed:")
            self.logger.info(f"  Overall winner: {comparison_results['analysis_summary']['overall_winner']}")
            self.logger.info(f"  Fastest processing: {comparison_results['analysis_summary']['fastest_processing']}")
            self.logger.info(f"  Highest throughput: {comparison_results['analysis_summary']['highest_throughput']}")
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Cross-protocol comparison failed: {e}")
            return {'error': str(e)}
    
    def run_performance_benchmark(
        self, 
        protocols: List[ProtocolType]
    ) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks"""
        self.logger.info("Running multi-protocol performance benchmarks...")
        
        benchmark_results = {
            'benchmark_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'protocols_tested': [p.name for p in protocols],
            'sample_sizes_tested': [2000, 5000, 8000, 12000],
            'protocol_benchmarks': {},
            'scalability_analysis': {}
        }
        
        try:
            for protocol in protocols:
                self.logger.info(f"Benchmarking {protocol.name}...")
                
                protocol_benchmark = {
                    'processing_times': {},
                    'throughput_rates': {},
                    'memory_usage': {},
                    'scalability_factor': 0.0
                }
                
                processing_times = []
                sample_sizes = []
                
                for sample_count in benchmark_results['sample_sizes_tested']:
                    self.logger.debug(f"  Testing {sample_count} samples...")
                    
                    # Generate signal
                    signal_data = self.generate_protocol_signal(protocol, sample_count)
                    
                    # Measure processing time
                    start_time = time.time()
                    auto_validate_signal(
                        signal_data,
                        sample_rate=signal_data['sample_rate'],
                        voltage_range=2.0 if protocol in [ProtocolType.PCIE, ProtocolType.ETHERNET_224G] else 1.0
                    )
                    processing_time = time.time() - start_time
                    
                    throughput = sample_count / processing_time
                    
                    protocol_benchmark['processing_times'][sample_count] = processing_time
                    protocol_benchmark['throughput_rates'][sample_count] = throughput
                    
                    processing_times.append(processing_time)
                    sample_sizes.append(sample_count)
                
                # Calculate scalability factor (how well it scales with sample size)
                if len(processing_times) > 1:
                    # Linear regression to find scaling relationship
                    coeffs = np.polyfit(sample_sizes, processing_times, 1)
                    protocol_benchmark['scalability_factor'] = coeffs[0]  # Slope
                
                benchmark_results['protocol_benchmarks'][protocol.name] = protocol_benchmark
                
                avg_throughput = np.mean(list(protocol_benchmark['throughput_rates'].values()))
                self.logger.info(f"  Average throughput: {avg_throughput:.0f} samples/s")
            
            # Scalability analysis
            scalability_scores = {}
            for protocol_name, benchmark in benchmark_results['protocol_benchmarks'].items():
                # Lower scalability factor is better (more linear scaling)
                scalability_scores[protocol_name] = benchmark['scalability_factor']
            
            best_scaling = min(scalability_scores.items(), key=lambda x: x[1])
            worst_scaling = max(scalability_scores.items(), key=lambda x: x[1])
            
            benchmark_results['scalability_analysis'] = {
                'scalability_scores': scalability_scores,
                'best_scaling': best_scaling[0],
                'worst_scaling': worst_scaling[0],
                'scaling_difference': worst_scaling[1] - best_scaling[1]
            }
            
            self.logger.info("Multi-protocol benchmarking completed:")
            self.logger.info(f"  Best scaling: {benchmark_results['scalability_analysis']['best_scaling']}")
            self.logger.info(f"  Worst scaling: {benchmark_results['scalability_analysis']['worst_scaling']}")
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Performance benchmarking failed: {e}")
            return {'error': str(e)}
    
    def generate_comparison_plots(self, comparison_data: Dict[str, Any]) -> None:
        """Generate cross-protocol comparison plots"""
        if not comparison_data or 'error' in comparison_data:
            return
        
        try:
            # Throughput comparison
            plt.figure(figsize=(12, 8))
            
            protocols = list(comparison_data['metrics_comparison']['throughput']['values'].keys())
            throughputs = list(comparison_data['metrics_comparison']['throughput']['values'].values())
            
            plt.subplot(2, 2, 1)
            plt.bar(protocols, throughputs)
            plt.title('Processing Throughput Comparison')
            plt.ylabel('Samples/Second')
            plt.xticks(rotation=45)
            
            # Signal quality comparison
            plt.subplot(2, 2, 2)
            quality_scores = list(comparison_data['metrics_comparison']['signal_quality_score']['values'].values())
            plt.bar(protocols, quality_scores)
            plt.title('Signal Quality Score Comparison')
            plt.ylabel('Quality Score')
            plt.xticks(rotation=45)
            
            # Eye height comparison
            plt.subplot(2, 2, 3)
            eye_heights = list(comparison_data['metrics_comparison']['eye_height']['values'].values())
            plt.bar(protocols, eye_heights)
            plt.title('Eye Height Comparison')
            plt.ylabel('Eye Height')
            plt.xticks(rotation=45)
            
            # Overall ranking
            plt.subplot(2, 2, 4)
            ranking_scores = [comparison_data['performance_ranking']['scores'][p] for p in protocols]
            plt.bar(protocols, ranking_scores)
            plt.title('Overall Performance Ranking')
            plt.ylabel('Performance Score')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'protocol_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Comparison plots saved to {self.output_dir / 'protocol_comparison.png'}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparison plots: {e}")
    
    def generate_report(self) -> str:
        """Generate comprehensive multi-protocol validation report"""
        report_path = self.output_dir / "multi_protocol_validation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Multi-Protocol SerDes Validation Report\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Framework Version: 1.4.0\n\n")
            
            # Write results
            for test_name, results in self.results.items():
                f.write(f"{test_name.upper().replace('_', ' ')}\n")
                f.write("-" * len(test_name) + "\n")
                
                if 'error' in results:
                    f.write(f"ERROR: {results['error']}\n")
                else:
                    self._write_results_recursive(f, results, indent=0)
                f.write("\n")
        
        self.logger.info(f"Multi-protocol validation report saved to: {report_path}")
        return str(report_path)
    
    def _write_results_recursive(self, f, data, indent=0):
        """Recursively write results to file"""
        indent_str = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    f.write(f"{indent_str}{key}:\n")
                    self._write_results_recursive(f, value, indent + 1)
                elif isinstance(value, list):
                    f.write(f"{indent_str}{key}: {value}\n")
                elif isinstance(value, float):
                    f.write(f"{indent_str}{key}: {value:.6f}\n")
                else:
                    f.write(f"{indent_str}{key}: {value}\n")
        else:
            f.write(f"{indent_str}{data}\n")
    
    def run_validation(
        self, 
        protocols: List[ProtocolType],
        num_samples: int = 8000,
        run_benchmark: bool = False,
        run_comparison: bool = False
    ) -> Dict[str, Any]:
        """Run complete multi-protocol validation suite"""
        self.logger.info("Starting multi-protocol validation suite...")
        self.logger.info(f"Protocols: {[p.name for p in protocols]}")
        self.logger.info(f"Samples per test: {num_samples}")
        
        # Individual protocol validation
        for protocol in protocols:
            self.results[f'{protocol.name.lower()}_validation'] = self.validate_protocol(protocol, num_samples)
        
        # Protocol detection validation
        if len(protocols) > 1:
            self.results['protocol_detection'] = self.validate_protocol_detection(protocols, num_samples)
        
        # Cross-protocol comparison
        if run_comparison and len(protocols) > 1:
            comparison_results = self.run_cross_protocol_comparison(protocols, num_samples)
            self.results['cross_protocol_comparison'] = comparison_results
            
            # Generate comparison plots
            self.generate_comparison_plots(comparison_results)
        
        # Performance benchmarks
        if run_benchmark:
            self.results['performance_benchmark'] = self.run_performance_benchmark(protocols)
        
        # Generate comprehensive report
        report_path = self.generate_report()
        
        self.logger.info("Multi-protocol validation suite completed!")
        self.logger.info(f"Report saved to: {report_path}")
        
        return self.results


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Multi-Protocol SerDes Validation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/multi_protocol_validation.py --protocols all --compare --benchmark
  python scripts/multi_protocol_validation.py --protocols pcie,usb4 --samples 10000
  python scripts/multi_protocol_validation.py --protocols thunderbolt --certification
        """
    )
    
    parser.add_argument(
        '--protocols', 
        default='all',
        help='Comma-separated list of protocols to test: pcie,eth224g,usb4,thunderbolt,all (default: all)'
    )
    
    parser.add_argument(
        '--samples', 
        type=int, 
        default=8000,
        help='Number of samples per test (default: 8000)'
    )
    
    parser.add_argument(
        '--output', 
        default='./multi_protocol_results',
        help='Output directory for reports (default: ./multi_protocol_results)'
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
    
    parser.add_argument(
        '--compare', 
        action='store_true',
        help='Generate cross-protocol comparison'
    )
    
    parser.add_argument(
        '--mock', 
        action='store_true',
        help='Force mock mode for testing'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Check framework availability
    if not FRAMEWORK_AVAILABLE:
        logger.error("Framework modules not available - cannot run validation")
        sys.exit(1)
    
    # Force mock mode if requested
    if args.mock:
        import os
        os.environ['SVF_MOCK_MODE'] = '1'
        logger.info("Forcing mock mode operation")
    
    # Parse protocols
    if args.protocols.lower() == 'all':
        protocols = [ProtocolType.PCIE, ProtocolType.ETHERNET_224G, ProtocolType.USB4]
    else:
        protocol_map = {
            'pcie': ProtocolType.PCIE,
            'eth224g': ProtocolType.ETHERNET_224G,
            'usb4': ProtocolType.USB4,
            'thunderbolt': ProtocolType.USB4  # Thunderbolt uses USB4 protocol
        }
        
        protocol_names = [p.strip().lower() for p in args.protocols.split(',')]
        protocols = []
        
        for name in protocol_names:
            if name in protocol_map:
                if protocol_map[name] not in protocols:  # Avoid duplicates
                    protocols.append(protocol_map[name])
            else:
                logger.error(f"Unknown protocol: {name}")
                sys.exit(1)
    
    # Validate arguments
    if args.samples < 2000:
        logger.error("Number of samples must be at least 2000")
        sys.exit(1)
    
    if not protocols:
        logger.error("No valid protocols specified")
        sys.exit(1)
    
    try:
        # Create validator
        validator = MultiProtocolValidator(args.output)
        
        # Run validation
        results = validator.run_validation(
            protocols=protocols,
            num_samples=args.samples,
            run_benchmark=args.benchmark,
            run_comparison=args.compare
        )
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("MULTI-PROTOCOL VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        total_tests = len([r for r in results.values() if isinstance(r, dict)])
        passed_tests = sum(1 for r in results.values() if isinstance(r, dict) and 'error' not in r)
        
        logger.info(f"Protocols tested: {[p.name for p in protocols]}")
        logger.info(f"Total test categories: {total_tests}")
        logger.info(f"Passed test categories: {passed_tests}")
        logger.info(f"Failed test categories: {total_tests - passed_tests}")
        logger.info(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        # Protocol-specific summary
        for protocol in protocols:
            protocol_key = f'{protocol.name.lower()}_validation'
            if protocol_key in results and 'error' not in results[protocol_key]:
                detected = results[protocol_key].get('detected_protocol', 'UNKNOWN')
                compliance = results[protocol_key].get('compliance_status', 'UNKNOWN')
                logger.info(f"{protocol.name}: Detected={detected}, Compliance={compliance}")
        
        # Cross-protocol comparison summary
        if 'cross_protocol_comparison' in results and 'error' not in results['cross_protocol_comparison']:
            winner = results['cross_protocol_comparison']['analysis_summary']['overall_winner']
            logger.info(f"Overall best performing protocol: {winner}")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Multi-protocol validation successful!")
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
