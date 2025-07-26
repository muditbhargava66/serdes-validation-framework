#!/usr/bin/env python3
"""
Multi-Protocol Comparison Example

This example demonstrates cross-protocol comparison capabilities of the
SerDes Validation Framework v1.4.0, comparing PCIe 6.0, 224G Ethernet,
and USB4/Thunderbolt 4 protocols.

Features demonstrated:
- Automatic protocol detection
- Cross-protocol performance comparison
- Signal quality analysis across protocols
- Unified validation framework usage
- Comprehensive reporting and visualization

Usage:
    python examples/multi_protocol_comparison.py [--protocols PROTOCOLS] [--verbose]
"""

import argparse
import logging
import sys
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
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"Framework not available: {e}")
    FRAMEWORK_AVAILABLE = False
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProtocolSignalGenerator:
    """Generate realistic test signals for different protocols"""
    
    @staticmethod
    def generate_pcie_signal(num_samples: int = 8000, mode: str = 'PAM4') -> Dict[str, np.ndarray]:
        """Generate PCIe 6.0 test signal"""
        sample_rate = 200e9 if mode == 'PAM4' else 100e9
        time = np.linspace(0, num_samples/sample_rate, num_samples, dtype=np.float64)
        
        if mode == 'PAM4':
            # PAM4 signal with 4 levels
            levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
            data = np.random.choice(levels, size=num_samples)
            signal_power = np.mean(levels**2)
        else:  # NRZ
            # NRZ signal
            data = np.random.choice([-1.0, 1.0], size=num_samples)
            signal_power = 1.0
        
        # Add realistic noise (SNR ~25dB)
        noise_power = signal_power / (10**(25.0/10))
        noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
        voltage = data + noise
        
        return {
            'time': time,
            'voltage': voltage.astype(np.float64),
            'sample_rate': sample_rate,
            'protocol': 'PCIe',
            'mode': mode
        }
    
    @staticmethod
    def generate_ethernet_224g_signal(num_samples: int = 8000) -> Dict[str, np.ndarray]:
        """Generate 224G Ethernet test signal"""
        sample_rate = 256e9  # 256 GSa/s for 224G
        time = np.linspace(0, num_samples/sample_rate, num_samples, dtype=np.float64)
        
        # PAM4 signal for 224G Ethernet
        levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
        data = np.random.choice(levels, size=num_samples)
        signal_power = np.mean(levels**2)
        
        # Add realistic noise (SNR ~28dB for 224G)
        noise_power = signal_power / (10**(28.0/10))
        noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
        voltage = data + noise
        
        return {
            'time': time,
            'voltage': voltage.astype(np.float64),
            'sample_rate': sample_rate,
            'protocol': '224G_Ethernet',
            'mode': 'PAM4'
        }
    
    @staticmethod
    def generate_usb4_signal(num_samples: int = 8000, lanes: int = 2) -> Dict[str, np.ndarray]:
        """Generate USB4 test signal"""
        sample_rate = 80e9  # 80 GSa/s for USB4
        time = np.linspace(0, num_samples/sample_rate, num_samples, dtype=np.float64)
        
        # NRZ signal for USB4
        data = np.random.choice([-1.0, 1.0], size=num_samples)
        signal_power = 1.0
        
        # Add spread spectrum clocking effect
        ssc_freq = 33e3  # 33 kHz SSC
        ssc_deviation = 0.005  # 0.5% deviation
        ssc_modulation = ssc_deviation * np.sin(2 * np.pi * ssc_freq * time)
        phase_modulation = np.cumsum(ssc_modulation) * 2 * np.pi
        data = data * np.cos(phase_modulation)
        
        # Add realistic noise (SNR ~30dB for USB4)
        noise_power = signal_power / (10**(30.0/10))
        noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
        voltage = data + noise
        
        # For dual-lane, create lane data
        if lanes == 2:
            # Add slight skew between lanes
            skew_samples = 2
            lane1_voltage = voltage
            lane2_voltage = np.roll(voltage, skew_samples) + 0.01 * np.random.randn(num_samples)
            
            return {
                'lane_0': {'time': time, 'voltage': lane1_voltage.astype(np.float64)},
                'lane_1': {'time': time, 'voltage': lane2_voltage.astype(np.float64)},
                'sample_rate': sample_rate,
                'protocol': 'USB4',
                'mode': 'NRZ',
                'lanes': lanes
            }
        else:
            return {
                'time': time,
                'voltage': voltage.astype(np.float64),
                'sample_rate': sample_rate,
                'protocol': 'USB4',
                'mode': 'NRZ',
                'lanes': 1
            }


class MultiProtocolComparator:
    """Compare performance across multiple SerDes protocols"""
    
    def __init__(self):
        """Initialize the comparator"""
        self.framework = create_validation_framework()
        self.results = {}
        self.signal_generator = ProtocolSignalGenerator()
        
    def validate_protocol(self, protocol_name: str, signal_data: Dict, **kwargs) -> Dict[str, Any]:
        """Validate a specific protocol"""
        logger.info(f"Validating {protocol_name}...")
        
        try:
            # Use framework's auto-validation
            start_time = time.time()
            
            if protocol_name == 'USB4' and 'lanes' in signal_data:
                # Handle dual-lane USB4 signal
                validation_results = auto_validate_signal(
                    signal_data=signal_data,
                    sample_rate=signal_data['sample_rate'],
                    voltage_range=0.8,
                    protocol_hint="usb4"
                )
            else:
                # Handle single-lane signals
                validation_results = auto_validate_signal(
                    signal_data=signal_data,
                    sample_rate=signal_data['sample_rate'],
                    voltage_range=2.0 if protocol_name in ['PCIe', '224G_Ethernet'] else 0.8
                )
            
            validation_time = time.time() - start_time
            
            # Extract key metrics
            results = {
                'protocol': protocol_name,
                'validation_time': validation_time,
                'detected_protocol': validation_results.get('protocol_type', 'UNKNOWN'),
                'confidence': validation_results.get('confidence', 0.0),
                'overall_status': validation_results.get('validation_results', {}).get('overall_status', 'UNKNOWN'),
                'signal_quality_score': 0.0,
                'eye_height': 0.0,
                'eye_width': 0.0,
                'jitter_ps': 0.0,
                'ber_estimate': 1.0,
                'throughput': len(signal_data.get('voltage', signal_data.get('lane_0', {}).get('voltage', []))) / validation_time
            }
            
            # Extract detailed metrics if available
            validation_data = validation_results.get('validation_results', {})
            if hasattr(validation_data, 'phase_results'):
                # Extract metrics from phase results
                total_metrics = 0
                quality_sum = 0.0
                
                for phase_result in validation_data.phase_results:
                    if hasattr(phase_result, 'metrics'):
                        for metric_name, value in phase_result.metrics.items():
                            if 'eye_height' in metric_name.lower():
                                results['eye_height'] = max(results['eye_height'], float(value))
                            elif 'eye_width' in metric_name.lower():
                                results['eye_width'] = max(results['eye_width'], float(value))
                            elif 'jitter' in metric_name.lower() and 'ps' in metric_name.lower():
                                results['jitter_ps'] = max(results['jitter_ps'], float(value))
                            elif 'ber' in metric_name.lower():
                                results['ber_estimate'] = min(results['ber_estimate'], float(value))
                            elif 'quality' in metric_name.lower() or 'snr' in metric_name.lower():
                                quality_sum += float(value)
                                total_metrics += 1
                
                if total_metrics > 0:
                    results['signal_quality_score'] = quality_sum / total_metrics
            
            # Set overall status
            if hasattr(validation_data, 'overall_status'):
                results['overall_status'] = validation_data.overall_status.name
            
            logger.info(f"{protocol_name} validation completed:")
            logger.info(f"  Status: {results['overall_status']}")
            logger.info(f"  Validation time: {results['validation_time']:.3f}s")
            logger.info(f"  Throughput: {results['throughput']:.0f} samples/s")
            
            return results
            
        except Exception as e:
            logger.error(f"{protocol_name} validation failed: {e}")
            return {
                'protocol': protocol_name,
                'error': str(e),
                'validation_time': 0.0,
                'throughput': 0.0
            }
    
    def run_comparison(self, protocols: List[str], num_samples: int = 8000) -> Dict[str, Any]:
        """Run comprehensive protocol comparison"""
        logger.info(f"Starting multi-protocol comparison: {protocols}")
        
        comparison_results = {
            'protocols_tested': protocols,
            'num_samples': num_samples,
            'individual_results': {},
            'performance_comparison': {},
            'ranking': {},
            'summary': {}
        }
        
        # Generate signals and validate each protocol
        for protocol in protocols:
            logger.info(f"\n--- Testing {protocol} ---")
            
            try:
                # Generate appropriate signal
                if protocol == 'PCIe':
                    signal_data = self.signal_generator.generate_pcie_signal(num_samples, 'PAM4')
                elif protocol == '224G_Ethernet':
                    signal_data = self.signal_generator.generate_ethernet_224g_signal(num_samples)
                elif protocol == 'USB4':
                    signal_data = self.signal_generator.generate_usb4_signal(num_samples, lanes=2)
                else:
                    logger.error(f"Unknown protocol: {protocol}")
                    continue
                
                # Validate protocol
                result = self.validate_protocol(protocol, signal_data)
                comparison_results['individual_results'][protocol] = result
                
            except Exception as e:
                logger.error(f"Failed to test {protocol}: {e}")
                comparison_results['individual_results'][protocol] = {
                    'protocol': protocol,
                    'error': str(e)
                }
        
        # Perform comparison analysis
        self._analyze_performance(comparison_results)
        self._rank_protocols(comparison_results)
        self._generate_summary(comparison_results)
        
        return comparison_results
    
    def _analyze_performance(self, results: Dict[str, Any]) -> None:
        """Analyze performance across protocols"""
        metrics_to_compare = ['validation_time', 'throughput', 'signal_quality_score', 'eye_height', 'jitter_ps']
        
        performance_data = {}
        
        for metric in metrics_to_compare:
            metric_values = {}
            for protocol, data in results['individual_results'].items():
                if 'error' not in data:
                    metric_values[protocol] = data.get(metric, 0.0)
            
            if metric_values:
                # Find best and worst
                if metric in ['validation_time', 'jitter_ps']:  # Lower is better
                    best = min(metric_values.items(), key=lambda x: x[1])
                    worst = max(metric_values.items(), key=lambda x: x[1])
                else:  # Higher is better
                    best = max(metric_values.items(), key=lambda x: x[1])
                    worst = min(metric_values.items(), key=lambda x: x[1])
                
                performance_data[metric] = {
                    'values': metric_values,
                    'best': best,
                    'worst': worst,
                    'range': abs(best[1] - worst[1])
                }
        
        results['performance_comparison'] = performance_data
    
    def _rank_protocols(self, results: Dict[str, Any]) -> None:
        """Rank protocols based on overall performance"""
        protocols = [p for p in results['individual_results'].keys() 
                    if 'error' not in results['individual_results'][p]]
        
        if not protocols:
            return
        
        # Calculate scores for each protocol
        protocol_scores = {protocol: 0 for protocol in protocols}
        
        for metric, data in results['performance_comparison'].items():
            if not data['values']:
                continue
                
            # Assign points based on ranking
            if metric in ['validation_time', 'jitter_ps']:  # Lower is better
                sorted_protocols = sorted(data['values'].items(), key=lambda x: x[1])
            else:  # Higher is better
                sorted_protocols = sorted(data['values'].items(), key=lambda x: x[1], reverse=True)
            
            # Award points (highest rank gets most points)
            for i, (protocol, _) in enumerate(sorted_protocols):
                protocol_scores[protocol] += len(protocols) - i
        
        # Sort by total score
        ranked_protocols = sorted(protocol_scores.items(), key=lambda x: x[1], reverse=True)
        
        results['ranking'] = {
            'scores': protocol_scores,
            'ranked_list': [p[0] for p in ranked_protocols],
            'winner': ranked_protocols[0][0] if ranked_protocols else None
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> None:
        """Generate comparison summary"""
        summary = {
            'total_protocols_tested': len(results['individual_results']),
            'successful_validations': len([r for r in results['individual_results'].values() if 'error' not in r]),
            'overall_winner': results['ranking'].get('winner', 'None'),
            'key_findings': []
        }
        
        # Add key findings
        if results['performance_comparison']:
            for metric, data in results['performance_comparison'].items():
                if data['values']:
                    best_protocol, best_value = data['best']
                    summary['key_findings'].append(
                        f"Best {metric}: {best_protocol} ({best_value:.3f})"
                    )
        
        results['summary'] = summary
    
    def plot_comparison(self, results: Dict[str, Any], save_path: str = 'protocol_comparison.png') -> None:
        """Generate comparison plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Multi-Protocol SerDes Comparison', fontsize=16, fontweight='bold')
            
            protocols = list(results['individual_results'].keys())
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            # Validation time comparison
            ax = axes[0, 0]
            times = [results['individual_results'][p].get('validation_time', 0) for p in protocols]
            bars = ax.bar(protocols, times, color=colors[:len(protocols)])
            ax.set_title('Validation Time Comparison')
            ax.set_ylabel('Time (seconds)')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, time in zip(bars, times, strict=False):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{time:.3f}s', ha='center', va='bottom')
            
            # Throughput comparison
            ax = axes[0, 1]
            throughputs = [results['individual_results'][p].get('throughput', 0) for p in protocols]
            bars = ax.bar(protocols, throughputs, color=colors[:len(protocols)])
            ax.set_title('Processing Throughput Comparison')
            ax.set_ylabel('Samples/Second')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, throughput in zip(bars, throughputs, strict=False):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{throughput:.0f}', ha='center', va='bottom')
            
            # Signal quality comparison
            ax = axes[1, 0]
            qualities = [results['individual_results'][p].get('signal_quality_score', 0) for p in protocols]
            bars = ax.bar(protocols, qualities, color=colors[:len(protocols)])
            ax.set_title('Signal Quality Score Comparison')
            ax.set_ylabel('Quality Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Overall ranking
            ax = axes[1, 1]
            if 'ranking' in results and results['ranking']['scores']:
                scores = [results['ranking']['scores'].get(p, 0) for p in protocols]
                bars = ax.bar(protocols, scores, color=colors[:len(protocols)])
                ax.set_title('Overall Performance Ranking')
                ax.set_ylabel('Performance Score')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, score in zip(bars, scores, strict=False):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{score}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plots saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
    
    def print_detailed_results(self, results: Dict[str, Any]) -> None:
        """Print detailed comparison results"""
        print("\n" + "="*80)
        print("MULTI-PROTOCOL COMPARISON RESULTS")
        print("="*80)
        
        # Summary
        summary = results['summary']
        print("\nSUMMARY:")
        print(f"  Protocols tested: {summary['total_protocols_tested']}")
        print(f"  Successful validations: {summary['successful_validations']}")
        print(f"  Overall winner: {summary['overall_winner']}")
        
        # Individual results
        print("\nINDIVIDUAL RESULTS:")
        for protocol, data in results['individual_results'].items():
            print(f"\n{protocol}:")
            if 'error' in data:
                print(f"  ERROR: {data['error']}")
            else:
                print(f"  Status: {data.get('overall_status', 'UNKNOWN')}")
                print(f"  Validation time: {data.get('validation_time', 0):.3f}s")
                print(f"  Throughput: {data.get('throughput', 0):.0f} samples/s")
                print(f"  Signal quality: {data.get('signal_quality_score', 0):.3f}")
                print(f"  Eye height: {data.get('eye_height', 0):.3f}")
                print(f"  Jitter: {data.get('jitter_ps', 0):.2f} ps")
        
        # Performance comparison
        if 'performance_comparison' in results:
            print("\nPERFORMANCE COMPARISON:")
            for metric, data in results['performance_comparison'].items():
                if data['values']:
                    best_protocol, best_value = data['best']
                    print(f"  Best {metric}: {best_protocol} ({best_value:.3f})")
        
        # Ranking
        if 'ranking' in results and results['ranking']['ranked_list']:
            print("\nOVERALL RANKING:")
            for i, protocol in enumerate(results['ranking']['ranked_list'], 1):
                score = results['ranking']['scores'][protocol]
                print(f"  {i}. {protocol} (Score: {score})")
        
        print("\n" + "="*80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Multi-Protocol Comparison Example")
    parser.add_argument(
        '--protocols',
        default='PCIe,224G_Ethernet,USB4',
        help='Comma-separated list of protocols to compare (default: PCIe,224G_Ethernet,USB4)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=8000,
        help='Number of samples per test (default: 8000)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check framework availability
    if not FRAMEWORK_AVAILABLE:
        logger.error("Framework not available")
        sys.exit(1)
    
    # Parse protocols
    protocols = [p.strip() for p in args.protocols.split(',')]
    valid_protocols = ['PCIe', '224G_Ethernet', 'USB4']
    
    for protocol in protocols:
        if protocol not in valid_protocols:
            logger.error(f"Invalid protocol: {protocol}. Valid options: {valid_protocols}")
            sys.exit(1)
    
    try:
        # Create comparator and run comparison
        comparator = MultiProtocolComparator()
        results = comparator.run_comparison(protocols, args.samples)
        
        # Display results
        comparator.print_detailed_results(results)
        
        # Generate plots
        try:
            comparator.plot_comparison(results)
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")
        
        logger.info("Multi-protocol comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import time
    main()
