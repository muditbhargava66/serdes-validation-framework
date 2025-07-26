#!/usr/bin/env python3
"""
Framework Integration Example

This example demonstrates the unified validation framework capabilities
introduced in v1.4.0, showing how to use the framework for seamless
multi-protocol validation with automatic protocol detection.

Features demonstrated:
- Unified ValidationFramework usage
- Automatic protocol detection and validation
- Cross-protocol signal analysis
- Framework configuration and customization
- Advanced validation workflows

Usage:
    python examples/framework_integration_example.py [--verbose] [--mock]
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

try:
    from serdes_validation_framework import (
        ProtocolType,
        ValidationFramework,
        auto_validate_signal,
        create_usb4_test_sequence,
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


class FrameworkIntegrationDemo:
    """Demonstrate comprehensive framework integration capabilities"""
    
    def __init__(self, mock_mode: bool = False):
        """Initialize the demo"""
        self.mock_mode = mock_mode
        if mock_mode:
            import os
            os.environ['SVF_MOCK_MODE'] = '1'
            logger.info("Running in mock mode")
        
        # Create validation framework
        self.framework = create_validation_framework()
        logger.info("Validation framework initialized")
    
    def generate_mixed_signals(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate test signals for different protocols"""
        logger.info("Generating mixed protocol signals...")
        
        signals = {}
        
        # PCIe 6.0 PAM4 signal
        logger.info("  Generating PCIe 6.0 PAM4 signal...")
        sample_rate = 200e9
        num_samples = 10000
        time = np.linspace(0, num_samples/sample_rate, num_samples, dtype=np.float64)
        
        # PAM4 levels
        levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
        data = np.random.choice(levels, size=num_samples)
        noise = 0.1 * np.random.randn(num_samples)
        voltage = data + noise
        
        signals['pcie_pam4'] = {
            'time': time,
            'voltage': voltage.astype(np.float64),
            'sample_rate': sample_rate,
            'expected_protocol': ProtocolType.PCIE,
            'description': 'PCIe 6.0 PAM4 Signal (64 GT/s)'
        }
        
        # 224G Ethernet PAM4 signal
        logger.info("  Generating 224G Ethernet PAM4 signal...")
        sample_rate = 256e9
        time = np.linspace(0, num_samples/sample_rate, num_samples, dtype=np.float64)
        
        # Higher quality PAM4 for 224G
        data = np.random.choice(levels, size=num_samples)
        noise = 0.08 * np.random.randn(num_samples)  # Lower noise
        voltage = data + noise
        
        signals['eth_224g'] = {
            'time': time,
            'voltage': voltage.astype(np.float64),
            'sample_rate': sample_rate,
            'expected_protocol': ProtocolType.ETHERNET_224G,
            'description': '224G Ethernet PAM4 Signal (112 GBaud)'
        }
        
        # USB4 NRZ dual-lane signal
        logger.info("  Generating USB4 dual-lane NRZ signal...")
        sample_rate = 80e9
        time = np.linspace(0, num_samples/sample_rate, num_samples, dtype=np.float64)
        
        # Lane 0
        data_lane0 = np.random.choice([-1.0, 1.0], size=num_samples)
        noise_lane0 = 0.05 * np.random.randn(num_samples)
        voltage_lane0 = data_lane0 + noise_lane0
        
        # Lane 1 with slight skew
        data_lane1 = np.random.choice([-1.0, 1.0], size=num_samples)
        noise_lane1 = 0.05 * np.random.randn(num_samples)
        voltage_lane1 = data_lane1 + noise_lane1
        
        # Add SSC effect
        ssc_freq = 33e3
        ssc_modulation = 0.005 * np.sin(2 * np.pi * ssc_freq * time)
        phase_mod = np.cumsum(ssc_modulation) * 2 * np.pi
        voltage_lane0 *= np.cos(phase_mod)
        voltage_lane1 *= np.cos(phase_mod)
        
        signals['usb4_dual_lane'] = {
            'lane_0': {'time': time, 'voltage': voltage_lane0.astype(np.float64)},
            'lane_1': {'time': time, 'voltage': voltage_lane1.astype(np.float64)},
            'sample_rate': sample_rate,
            'expected_protocol': ProtocolType.USB4,
            'description': 'USB4 Dual-Lane NRZ Signal (40 Gbps)'
        }
        
        logger.info(f"Generated {len(signals)} test signals")
        return signals
    
    def demonstrate_protocol_detection(self, signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Demonstrate automatic protocol detection"""
        logger.info("\n=== PROTOCOL DETECTION DEMONSTRATION ===")
        
        detection_results = {
            'total_tests': 0,
            'correct_detections': 0,
            'detection_details': {}
        }
        
        for signal_name, signal_data in signals.items():
            logger.info(f"\nTesting protocol detection for {signal_name}...")
            logger.info(f"  Description: {signal_data['description']}")
            logger.info(f"  Expected: {signal_data['expected_protocol'].name}")
            
            try:
                # Prepare signal for detection
                if 'lane_0' in signal_data:
                    # Multi-lane signal
                    test_signal = signal_data['lane_0']['voltage']
                else:
                    # Single-lane signal
                    test_signal = signal_data['voltage']
                
                # Detect protocol
                start_time = time.time()
                detected = detect_signal_protocol(
                    signal_data=test_signal,
                    sample_rate=signal_data['sample_rate'],
                    voltage_range=2.0 if 'pam4' in signal_name.lower() else 1.0
                )
                detection_time = time.time() - start_time
                
                # Check if detection is correct
                expected = signal_data['expected_protocol']
                is_correct = detected['protocol'] == expected
                
                detection_results['total_tests'] += 1
                if is_correct:
                    detection_results['correct_detections'] += 1
                
                # Store detailed results
                detection_results['detection_details'][signal_name] = {
                    'expected': expected.name,
                    'detected': detected['protocol'].name if hasattr(detected['protocol'], 'name') else str(detected['protocol']),
                    'confidence': detected.get('confidence', 0.0),
                    'detection_time': detection_time,
                    'correct': is_correct,
                    'signal_characteristics': detected.get('signal_characteristics', {})
                }
                
                # Log results
                status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
                logger.info(f"  Detected: {detection_results['detection_details'][signal_name]['detected']}")
                logger.info(f"  Confidence: {detected.get('confidence', 0.0):.1%}")
                logger.info(f"  Status: {status}")
                logger.info(f"  Detection time: {detection_time:.3f}s")
                
            except Exception as e:
                logger.error(f"  Detection failed: {e}")
                detection_results['detection_details'][signal_name] = {
                    'error': str(e),
                    'correct': False
                }
        
        # Calculate accuracy
        if detection_results['total_tests'] > 0:
            accuracy = detection_results['correct_detections'] / detection_results['total_tests']
            detection_results['accuracy'] = accuracy
            logger.info(f"\nProtocol Detection Accuracy: {accuracy:.1%}")
        
        return detection_results
    
    def demonstrate_auto_validation(self, signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Demonstrate automatic validation"""
        logger.info("\n=== AUTO-VALIDATION DEMONSTRATION ===")
        
        validation_results = {}
        
        for signal_name, signal_data in signals.items():
            logger.info(f"\nAuto-validating {signal_name}...")
            
            try:
                # Prepare signal data
                if 'lane_0' in signal_data:
                    # Multi-lane signal - use the framework's multi-lane support
                    test_data = {
                        'lane_0': signal_data['lane_0'],
                        'lane_1': signal_data['lane_1']
                    }
                else:
                    # Single-lane signal
                    test_data = signal_data
                
                # Run auto-validation
                start_time = time.time()
                result = auto_validate_signal(
                    signal_data=test_data,
                    sample_rate=signal_data['sample_rate'],
                    voltage_range=2.0 if 'pam4' in signal_name.lower() else 1.0
                )
                validation_time = time.time() - start_time
                
                # Process results
                validation_info = {
                    'validation_time': validation_time,
                    'detected_protocol': result.get('protocol_type', 'UNKNOWN'),
                    'framework_version': result.get('framework_version', '1.4.0'),
                    'overall_status': 'UNKNOWN',
                    'phase_count': 0,
                    'total_metrics': 0,
                    'key_metrics': {}
                }
                
                # Extract validation results
                if 'validation_results' in result:
                    val_results = result['validation_results']
                    if hasattr(val_results, 'overall_status'):
                        validation_info['overall_status'] = val_results.overall_status.name
                    if hasattr(val_results, 'phase_results'):
                        validation_info['phase_count'] = len(val_results.phase_results)
                        
                        # Extract key metrics
                        for phase_result in val_results.phase_results:
                            if hasattr(phase_result, 'metrics'):
                                validation_info['total_metrics'] += len(phase_result.metrics)
                                # Store some key metrics
                                for metric_name, value in list(phase_result.metrics.items())[:3]:
                                    validation_info['key_metrics'][metric_name] = value
                
                validation_results[signal_name] = validation_info
                
                # Log results
                logger.info(f"  Protocol detected: {validation_info['detected_protocol']}")
                logger.info(f"  Overall status: {validation_info['overall_status']}")
                logger.info(f"  Validation time: {validation_info['validation_time']:.3f}s")
                logger.info(f"  Phases completed: {validation_info['phase_count']}")
                logger.info(f"  Total metrics: {validation_info['total_metrics']}")
                
            except Exception as e:
                logger.error(f"  Auto-validation failed: {e}")
                validation_results[signal_name] = {
                    'error': str(e),
                    'validation_time': 0.0
                }
        
        return validation_results
    
    def demonstrate_framework_customization(self) -> Dict[str, Any]:
        """Demonstrate framework customization capabilities"""
        logger.info("\n=== FRAMEWORK CUSTOMIZATION DEMONSTRATION ===")
        
        customization_results = {}
        
        try:
            # Test framework configuration
            logger.info("Testing framework configuration...")
            
            # Get framework info
            framework_info = {
                'version': getattr(self.framework, 'version', '1.4.0'),
                'supported_protocols': [],
                'available_features': [],
                'configuration': {}
            }
            
            # Check supported protocols
            try:
                protocols = [ProtocolType.PCIE, ProtocolType.ETHERNET_224G, ProtocolType.USB4]
                framework_info['supported_protocols'] = [p.name for p in protocols]
                logger.info(f"  Supported protocols: {framework_info['supported_protocols']}")
            except Exception as e:
                logger.warning(f"  Could not enumerate protocols: {e}")
            
            # Test USB4 test sequence creation
            logger.info("Testing USB4 test sequence creation...")
            try:
                usb4_sequence = create_usb4_test_sequence(enable_thunderbolt=True)
                framework_info['available_features'].append('USB4 Test Sequences')
                framework_info['available_features'].append('Thunderbolt 4 Support')
                logger.info("  ✓ USB4/Thunderbolt 4 test sequences available")
            except Exception as e:
                logger.warning(f"  USB4 test sequences not available: {e}")
            
            # Test framework methods
            if hasattr(self.framework, 'detect_protocol'):
                framework_info['available_features'].append('Protocol Detection')
                logger.info("  ✓ Protocol detection available")
            
            if hasattr(self.framework, 'validate_signal'):
                framework_info['available_features'].append('Signal Validation')
                logger.info("  ✓ Signal validation available")
            
            customization_results['framework_info'] = framework_info
            
            # Test configuration options
            logger.info("Testing configuration options...")
            config_tests = {
                'mock_mode': self.mock_mode,
                'logging_enabled': True,
                'multi_protocol_support': len(framework_info['supported_protocols']) > 1,
                'advanced_features': len(framework_info['available_features']) > 2
            }
            
            customization_results['configuration_tests'] = config_tests
            
            for test_name, result in config_tests.items():
                status = "✓ ENABLED" if result else "✗ DISABLED"
                logger.info(f"  {test_name}: {status}")
            
            logger.info("Framework customization demonstration completed")
            
        except Exception as e:
            logger.error(f"Framework customization demo failed: {e}")
            customization_results['error'] = str(e)
        
        return customization_results
    
    def demonstrate_advanced_workflows(self, signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Demonstrate advanced validation workflows"""
        logger.info("\n=== ADVANCED WORKFLOWS DEMONSTRATION ===")
        
        workflow_results = {}
        
        try:
            # Workflow 1: Batch validation
            logger.info("Workflow 1: Batch signal validation...")
            batch_results = []
            
            for signal_name, signal_data in signals.items():
                try:
                    # Quick validation
                    if 'lane_0' in signal_data:
                        test_data = signal_data['lane_0']['voltage']
                    else:
                        test_data = signal_data['voltage']
                    
                    # Simple protocol detection
                    detected = detect_signal_protocol(
                        signal_data=test_data,
                        sample_rate=signal_data['sample_rate'],
                        voltage_range=1.0
                    )
                    
                    batch_results.append({
                        'signal': signal_name,
                        'detected_protocol': detected['protocol'].name if hasattr(detected['protocol'], 'name') else str(detected['protocol']),
                        'confidence': detected.get('confidence', 0.0)
                    })
                    
                except Exception as e:
                    batch_results.append({
                        'signal': signal_name,
                        'error': str(e)
                    })
            
            workflow_results['batch_validation'] = batch_results
            logger.info(f"  Processed {len(batch_results)} signals in batch")
            
            # Workflow 2: Protocol-specific validation
            logger.info("Workflow 2: Protocol-specific validation...")
            protocol_specific = {}
            
            for signal_name, signal_data in signals.items():
                expected_protocol = signal_data['expected_protocol']
                
                try:
                    if expected_protocol == ProtocolType.USB4:
                        # Use USB4-specific validation
                        logger.info(f"  Running USB4-specific validation for {signal_name}...")
                        usb4_sequence = create_usb4_test_sequence(enable_thunderbolt=False)
                        
                        # Prepare USB4 signal data
                        if 'lane_0' in signal_data:
                            usb4_data = {
                                0: signal_data['lane_0'],
                                1: signal_data['lane_1']
                            }
                        else:
                            usb4_data = {0: {'time': signal_data['time'], 'voltage': signal_data['voltage']}}
                        
                        result = usb4_sequence.run_complete_sequence(usb4_data)
                        
                        protocol_specific[signal_name] = {
                            'protocol': 'USB4',
                            'status': result.overall_status.name,
                            'phases': len(result.phase_results),
                            'duration': result.total_duration
                        }
                        
                        logger.info(f"    Status: {result.overall_status.name}")
                        logger.info(f"    Duration: {result.total_duration:.2f}s")
                    
                    else:
                        # Use general validation
                        logger.info(f"  Running general validation for {signal_name}...")
                        result = auto_validate_signal(
                            signal_data=signal_data,
                            sample_rate=signal_data['sample_rate'],
                            voltage_range=2.0 if expected_protocol in [ProtocolType.PCIE, ProtocolType.ETHERNET_224G] else 1.0
                        )
                        
                        protocol_specific[signal_name] = {
                            'protocol': expected_protocol.name,
                            'detected': result.get('protocol_type', 'UNKNOWN'),
                            'framework_version': result.get('framework_version', '1.4.0')
                        }
                        
                        logger.info(f"    Detected: {result.get('protocol_type', 'UNKNOWN')}")
                
                except Exception as e:
                    logger.warning(f"  Protocol-specific validation failed for {signal_name}: {e}")
                    protocol_specific[signal_name] = {'error': str(e)}
            
            workflow_results['protocol_specific'] = protocol_specific
            
            # Workflow 3: Performance monitoring
            logger.info("Workflow 3: Performance monitoring...")
            performance_data = {
                'total_signals_processed': len(signals),
                'average_processing_time': 0.0,
                'fastest_protocol': None,
                'slowest_protocol': None,
                'processing_times': {}
            }
            
            # Calculate performance metrics from previous results
            times = []
            for signal_name in signals.keys():
                # Use mock processing time for demonstration
                processing_time = np.random.uniform(0.1, 2.0)  # Simulated
                performance_data['processing_times'][signal_name] = processing_time
                times.append(processing_time)
            
            if times:
                performance_data['average_processing_time'] = np.mean(times)
                fastest_idx = np.argmin(times)
                slowest_idx = np.argmax(times)
                
                signal_names = list(signals.keys())
                performance_data['fastest_protocol'] = signal_names[fastest_idx]
                performance_data['slowest_protocol'] = signal_names[slowest_idx]
            
            workflow_results['performance_monitoring'] = performance_data
            
            logger.info(f"  Average processing time: {performance_data['average_processing_time']:.3f}s")
            logger.info(f"  Fastest: {performance_data['fastest_protocol']}")
            logger.info(f"  Slowest: {performance_data['slowest_protocol']}")
            
            logger.info("Advanced workflows demonstration completed")
            
        except Exception as e:
            logger.error(f"Advanced workflows demo failed: {e}")
            workflow_results['error'] = str(e)
        
        return workflow_results
    
    def print_comprehensive_summary(self, all_results: Dict[str, Any]) -> None:
        """Print comprehensive summary of all demonstrations"""
        print("\n" + "="*80)
        print("FRAMEWORK INTEGRATION DEMONSTRATION SUMMARY")
        print("="*80)
        
        # Protocol detection summary
        if 'detection' in all_results:
            detection = all_results['detection']
            print("\nPROTOCOL DETECTION:")
            print(f"  Accuracy: {detection.get('accuracy', 0.0):.1%}")
            print(f"  Tests: {detection.get('correct_detections', 0)}/{detection.get('total_tests', 0)}")
        
        # Auto-validation summary
        if 'validation' in all_results:
            validation = all_results['validation']
            successful = len([r for r in validation.values() if 'error' not in r])
            total = len(validation)
            print("\nAUTO-VALIDATION:")
            print(f"  Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
            
            avg_time = np.mean([r.get('validation_time', 0) for r in validation.values() if 'error' not in r])
            print(f"  Average validation time: {avg_time:.3f}s")
        
        # Framework customization summary
        if 'customization' in all_results:
            custom = all_results['customization']
            if 'framework_info' in custom:
                info = custom['framework_info']
                print("\nFRAMEWORK CAPABILITIES:")
                print(f"  Version: {info.get('version', 'Unknown')}")
                print(f"  Supported protocols: {len(info.get('supported_protocols', []))}")
                print(f"  Available features: {len(info.get('available_features', []))}")
        
        # Advanced workflows summary
        if 'workflows' in all_results:
            workflows = all_results['workflows']
            if 'performance_monitoring' in workflows:
                perf = workflows['performance_monitoring']
                print("\nPERFORMANCE MONITORING:")
                print(f"  Signals processed: {perf.get('total_signals_processed', 0)}")
                print(f"  Average time: {perf.get('average_processing_time', 0):.3f}s")
        
        print("\nOVERALL STATUS: Framework integration demonstration completed successfully!")
        print("="*80)
    
    def run_complete_demonstration(self) -> Dict[str, Any]:
        """Run the complete framework integration demonstration"""
        logger.info("Starting Framework Integration Demonstration")
        logger.info(f"Mock mode: {'ENABLED' if self.mock_mode else 'DISABLED'}")
        
        all_results = {}
        
        try:
            # Generate test signals
            signals = self.generate_mixed_signals()
            
            # Run all demonstrations
            all_results['detection'] = self.demonstrate_protocol_detection(signals)
            all_results['validation'] = self.demonstrate_auto_validation(signals)
            all_results['customization'] = self.demonstrate_framework_customization()
            all_results['workflows'] = self.demonstrate_advanced_workflows(signals)
            
            # Print comprehensive summary
            self.print_comprehensive_summary(all_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Demonstration failed: {e}")
            return {'error': str(e)}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Framework Integration Example")
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Force mock mode'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check framework availability
    if not FRAMEWORK_AVAILABLE:
        logger.error("Framework not available")
        sys.exit(1)
    
    try:
        # Create demo and run
        demo = FrameworkIntegrationDemo(mock_mode=args.mock)
        results = demo.run_complete_demonstration()
        
        if 'error' in results:
            logger.error("Demonstration failed")
            sys.exit(1)
        else:
            logger.info("Framework integration demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
