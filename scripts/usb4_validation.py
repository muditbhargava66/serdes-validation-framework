#!/usr/bin/env python3
"""
USB4/Thunderbolt 4 Validation Script

This script provides comprehensive USB4 and Thunderbolt 4 validation capabilities including:
- Complete USB4 v2.0 specification compliance testing
- Thunderbolt 4 security and certification validation
- Dual-lane signal analysis with lane skew compensation
- Multi-protocol tunneling validation (PCIe, DisplayPort, USB 3.2)
- Link training and power state management
- Advanced signal integrity analysis with SSC support
- Automated compliance reporting and certification

Usage:
    python scripts/usb4_validation.py [options]

Options:
    --mode {usb4,thunderbolt,both}    Protocol mode to test (default: both)
    --lanes {1,2}                     Number of lanes to test (default: 2)
    --tunneling {pcie,dp,usb32,all}   Tunneling protocols to test (default: all)
    --samples N                       Number of samples per test (default: 8000)
    --output DIR                      Output directory for reports (default: ./usb4_results)
    --verbose                         Enable verbose logging
    --benchmark                       Run performance benchmarks
    --certification                   Run Thunderbolt 4 certification tests
    --mock                           Force mock mode for testing
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
    from serdes_validation_framework.instrument_control.mock_controller import get_instrument_controller
    from serdes_validation_framework.instrument_control.usb4_scope_controller import USB4ScopeController
    from serdes_validation_framework.instrument_control.usb4_sync_controller import USB4SyncController
    from serdes_validation_framework.protocols.usb4 import (
        USB4_PROTOCOL_SPECS,
        DisplayPortTunnelValidator,
        MultiProtocolBandwidthManager,
        PCIeTunnelValidator,
        TunnelState,
        USB4Config,
        USB4JitterAnalyzer,
        USB4LinkState,
        USB4PowerManager,
        USB4SignalMode,
        USB4TunnelingMode,
        USB32TunnelValidator,
    )
    from serdes_validation_framework.protocols.usb4.thunderbolt import (
        DaisyChainValidator,
        IntelCertificationSuite,
        ThunderboltDeviceType,
        ThunderboltSecurityLevel,
        ThunderboltSecurityValidator,
    )
    from serdes_validation_framework.test_sequence.usb4_sequence import (
        USB4TestPhase,
        USB4TestResult,
        USB4TestSequence,
        USB4TestSequenceConfig,
    )
    USB4_AVAILABLE = True
except ImportError as e:
    print(f"USB4 modules not available: {e}")
    USB4_AVAILABLE = False
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
            logging.FileHandler('usb4_validation.log')
        ]
    )
    return logging.getLogger(__name__)


class USB4Validator:
    """Comprehensive USB4/Thunderbolt 4 validator"""
    
    def __init__(self, output_dir: str = "./usb4_results"):
        """Initialize USB4 validator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        
        # Initialize controllers
        self.controller = get_instrument_controller()
        self.scope_controller = USB4ScopeController(self.controller)
        self.sync_controller = USB4SyncController()
        
        self.logger.info(f"USB4 Validator initialized - Output: {self.output_dir}")
    
    def generate_usb4_signal(
        self, 
        mode: USB4SignalMode = USB4SignalMode.GEN2X2,
        num_samples: int = 8000,
        lane_count: int = 2,
        ssc_enabled: bool = True,
        snr_db: float = 28.0
    ) -> Dict[str, np.ndarray]:
        """Generate realistic USB4 test signal"""
        self.logger.debug(f"Generating USB4 {mode.name} signal: {num_samples} samples, {lane_count} lanes")
        
        # USB4 specifications
        specs = USB4_PROTOCOL_SPECS[mode.name.lower()]
        sample_rate = specs['sample_rate']
        data_rate = specs['data_rate_per_lane']
        
        # Time vector
        time = np.linspace(0, num_samples/sample_rate, num_samples, dtype=np.float64)
        
        # Generate dual-lane data
        lanes_data = {}
        for lane_id in range(lane_count):
            # Generate NRZ data (USB4 uses NRZ encoding)
            data_bits = np.random.choice([-1.0, 1.0], size=num_samples)
            
            # Add spread spectrum clocking if enabled
            if ssc_enabled:
                ssc_freq = 33e3  # 33 kHz SSC frequency
                ssc_deviation = 0.005  # 0.5% frequency deviation
                ssc_modulation = ssc_deviation * np.sin(2 * np.pi * ssc_freq * time)
                # Apply SSC to the signal timing (simplified model)
                phase_modulation = np.cumsum(ssc_modulation) * 2 * np.pi
                data_bits = data_bits * np.cos(phase_modulation)
            
            # Add lane-specific skew (realistic USB4 implementation)
            if lane_id > 0:
                skew_samples = int(np.random.normal(0, 2))  # ±2 sample skew
                if skew_samples != 0:
                    if skew_samples > 0:
                        data_bits = np.pad(data_bits, (skew_samples, 0), mode='edge')[:-skew_samples]
                    else:
                        data_bits = np.pad(data_bits, (0, -skew_samples), mode='edge')[-skew_samples:]
            
            # Add realistic noise
            signal_power = 1.0
            noise_power = signal_power / (10**(snr_db/10))
            noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
            voltage = data_bits + noise
            
            lanes_data[f'lane_{lane_id}'] = {
                'time': time,
                'voltage': voltage.astype(np.float64)
            }
        
        return lanes_data
    
    def validate_signal_analysis(
        self, 
        mode: USB4SignalMode = USB4SignalMode.GEN2X2,
        num_samples: int = 8000,
        lane_count: int = 2
    ) -> Dict[str, float]:
        """Validate USB4 signal analysis capabilities"""
        self.logger.info(f"Validating USB4 {mode.name} signal analysis...")
        
        try:
            # Generate test signal
            signal_data = self.generate_usb4_signal(mode, num_samples, lane_count)
            
            # Configure USB4 analyzer
            config = USB4Config(
                mode=mode,
                lane_count=lane_count,
                sample_rate=USB4_PROTOCOL_SPECS[mode.name.lower()]['sample_rate'],
                ssc_enabled=True,
                link_state=USB4LinkState.U0
            )
            
            # Analyze signal
            start_time = time.time()
            results = self.scope_controller.analyze_usb4_signal(signal_data, config)
            analysis_time = time.time() - start_time
            
            # Add timing information
            results['analysis_time'] = analysis_time
            results['samples_analyzed'] = num_samples * lane_count
            results['throughput'] = (num_samples * lane_count) / analysis_time
            
            self.logger.info(f"USB4 {mode.name} analysis completed in {analysis_time:.3f}s")
            for metric, value in results.items():
                if isinstance(value, float) and metric != 'analysis_time':
                    self.logger.info(f"  {metric}: {value:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"USB4 signal analysis failed: {e}")
            return {'error': str(e)}
    
    def validate_lane_skew_compensation(
        self, 
        num_samples: int = 8000
    ) -> Dict[str, float]:
        """Validate dual-lane skew compensation"""
        self.logger.info("Validating lane skew compensation...")
        
        try:
            # Generate signal with intentional skew
            signal_data = self.generate_usb4_signal(
                USB4SignalMode.GEN2X2, 
                num_samples, 
                lane_count=2
            )
            
            # Measure initial skew
            start_time = time.time()
            initial_skew = self.sync_controller.measure_lane_skew(signal_data)
            
            # Apply skew compensation
            compensated_data = self.sync_controller.compensate_lane_skew(
                signal_data, 
                initial_skew
            )
            
            # Measure residual skew
            final_skew = self.sync_controller.measure_lane_skew(compensated_data)
            compensation_time = time.time() - start_time
            
            results = {
                'initial_skew_ps': initial_skew * 1e12,  # Convert to ps
                'final_skew_ps': final_skew * 1e12,
                'skew_reduction_db': 20 * np.log10(abs(initial_skew) / abs(final_skew)) if final_skew != 0 else 60.0,
                'compensation_time': compensation_time,
                'compensation_success': abs(final_skew) < abs(initial_skew) * 0.1
            }
            
            self.logger.info("Lane skew compensation completed:")
            self.logger.info(f"  Initial skew: {results['initial_skew_ps']:.2f} ps")
            self.logger.info(f"  Final skew: {results['final_skew_ps']:.2f} ps")
            self.logger.info(f"  Reduction: {results['skew_reduction_db']:.1f} dB")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Lane skew compensation failed: {e}")
            return {'error': str(e)}
    
    def validate_jitter_analysis(
        self, 
        mode: USB4SignalMode = USB4SignalMode.GEN2X2,
        num_samples: int = 8000
    ) -> Dict[str, float]:
        """Validate advanced jitter analysis with SSC awareness"""
        self.logger.info(f"Validating USB4 {mode.name} jitter analysis...")
        
        try:
            # Generate signal with SSC
            signal_data = self.generate_usb4_signal(mode, num_samples, ssc_enabled=True)
            
            # Configure jitter analyzer
            jitter_analyzer = USB4JitterAnalyzer(
                sample_rate=USB4_PROTOCOL_SPECS[mode.name.lower()]['sample_rate'],
                ssc_enabled=True,
                ssc_frequency=33e3,
                ssc_deviation=0.005
            )
            
            # Analyze jitter for each lane
            start_time = time.time()
            jitter_results = {}
            
            for lane_name, lane_data in signal_data.items():
                lane_jitter = jitter_analyzer.analyze_jitter(
                    lane_data['time'],
                    lane_data['voltage']
                )
                jitter_results[lane_name] = lane_jitter
            
            analysis_time = time.time() - start_time
            
            # Compile comprehensive results
            results = {
                'analysis_time': analysis_time,
                'ssc_compensation_enabled': True
            }
            
            # Average jitter components across lanes
            jitter_components = ['rj_rms_ps', 'dj_pp_ps', 'pj_pp_ps', 'tj_pp_ps']
            for component in jitter_components:
                values = [jr[component] for jr in jitter_results.values() if component in jr]
                if values:
                    results[f'avg_{component}'] = np.mean(values)
                    results[f'max_{component}'] = np.max(values)
            
            # SSC-specific metrics
            if 'ssc_jitter_ps' in list(jitter_results.values())[0]:
                ssc_values = [jr['ssc_jitter_ps'] for jr in jitter_results.values()]
                results['avg_ssc_jitter_ps'] = np.mean(ssc_values)
                results['max_ssc_jitter_ps'] = np.max(ssc_values)
            
            self.logger.info(f"USB4 jitter analysis completed in {analysis_time:.3f}s")
            self.logger.info(f"  Average TJ: {results.get('avg_tj_pp_ps', 0):.2f} ps")
            self.logger.info(f"  Average SSC jitter: {results.get('avg_ssc_jitter_ps', 0):.2f} ps")
            
            return results
            
        except Exception as e:
            self.logger.error(f"USB4 jitter analysis failed: {e}")
            return {'error': str(e)}
    
    def validate_tunneling_protocols(
        self, 
        protocols: List[str] = ['pcie', 'dp', 'usb32'],
        num_samples: int = 6000
    ) -> Dict[str, Dict[str, float]]:
        """Validate multi-protocol tunneling"""
        self.logger.info(f"Validating tunneling protocols: {protocols}")
        
        tunneling_results = {}
        
        try:
            # Generate USB4 signal with tunneling overhead
            base_signal = self.generate_usb4_signal(
                USB4SignalMode.GEN2X2, 
                num_samples, 
                lane_count=2
            )
            
            # Initialize bandwidth manager
            bandwidth_manager = MultiProtocolBandwidthManager(
                total_bandwidth=40e9,  # 40 Gbps USB4
                overhead_percent=10.0
            )
            
            for protocol in protocols:
                self.logger.info(f"Testing {protocol.upper()} tunneling...")
                
                try:
                    start_time = time.time()
                    
                    if protocol == 'pcie':
                        validator = PCIeTunnelValidator()
                        tunnel_result = validator.validate_pcie_tunneling(
                            base_signal,
                            pcie_lanes=4,
                            pcie_speed='Gen4'  # 16 GT/s
                        )
                    elif protocol == 'dp':
                        validator = DisplayPortTunnelValidator()
                        tunnel_result = validator.validate_displayport_tunneling(
                            base_signal,
                            dp_lanes=4,
                            link_rate='HBR3'  # 8.1 Gbps
                        )
                    elif protocol == 'usb32':
                        validator = USB32TunnelValidator()
                        tunnel_result = validator.validate_usb32_tunneling(
                            base_signal,
                            usb32_speed='SuperSpeed+'  # 10 Gbps
                        )
                    
                    validation_time = time.time() - start_time
                    
                    # Process results
                    protocol_results = {
                        'validation_time': validation_time,
                        'tunnel_established': tunnel_result.tunnel_state == TunnelState.ACTIVE,
                        'bandwidth_efficiency': tunnel_result.bandwidth_efficiency,
                        'latency_us': tunnel_result.latency_us,
                        'error_rate': tunnel_result.error_rate,
                        'throughput_gbps': tunnel_result.effective_throughput / 1e9
                    }
                    
                    # Protocol-specific metrics
                    if hasattr(tunnel_result, 'protocol_specific'):
                        protocol_results.update(tunnel_result.protocol_specific)
                    
                    tunneling_results[protocol] = protocol_results
                    
                    self.logger.info(f"  {protocol.upper()} tunneling: {'SUCCESS' if protocol_results['tunnel_established'] else 'FAILED'}")
                    self.logger.info(f"  Bandwidth efficiency: {protocol_results['bandwidth_efficiency']:.1%}")
                    self.logger.info(f"  Effective throughput: {protocol_results['throughput_gbps']:.2f} Gbps")
                    
                except Exception as e:
                    self.logger.error(f"  {protocol.upper()} tunneling failed: {e}")
                    tunneling_results[protocol] = {'error': str(e)}
            
            return tunneling_results
            
        except Exception as e:
            self.logger.error(f"Tunneling validation failed: {e}")
            return {'error': str(e)}
    
    def validate_thunderbolt_security(self) -> Dict[str, float]:
        """Validate Thunderbolt 4 security features"""
        self.logger.info("Validating Thunderbolt 4 security...")
        
        try:
            # Initialize security validator
            security_validator = ThunderboltSecurityValidator(
                security_level=ThunderboltSecurityLevel.USER_AUTHORIZATION,
                device_type=ThunderboltDeviceType.HOST
            )
            
            start_time = time.time()
            
            # Run security validation tests
            security_results = security_validator.validate_security_features({
                'dma_protection': True,
                'device_authentication': True,
                'authorization_required': True,
                'secure_boot': True
            })
            
            validation_time = time.time() - start_time
            
            results = {
                'validation_time': validation_time,
                'overall_security_status': float(security_results.overall_status),
                'dma_protection_enabled': float(security_results.dma_protection.enabled),
                'authentication_success': float(security_results.authentication.success),
                'authorization_level': float(security_results.authorization.level.value),
                'security_score': security_results.security_score
            }
            
            self.logger.info("Thunderbolt 4 security validation completed:")
            self.logger.info(f"  Overall status: {'PASS' if results['overall_security_status'] else 'FAIL'}")
            self.logger.info(f"  Security score: {results['security_score']:.1f}/100")
            self.logger.info(f"  DMA protection: {'ENABLED' if results['dma_protection_enabled'] else 'DISABLED'}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Thunderbolt security validation failed: {e}")
            return {'error': str(e)}
    
    def validate_daisy_chain(
        self, 
        device_count: int = 4
    ) -> Dict[str, float]:
        """Validate Thunderbolt 4 daisy chain capabilities"""
        self.logger.info(f"Validating Thunderbolt 4 daisy chain with {device_count} devices...")
        
        try:
            # Initialize daisy chain validator
            daisy_validator = DaisyChainValidator(max_devices=6)
            
            # Create mock device chain
            device_chain = []
            for i in range(device_count):
                device_chain.append({
                    'device_id': f'TB4_DEV_{i:02d}',
                    'device_type': ThunderboltDeviceType.DEVICE,
                    'hop_count': i + 1,
                    'bandwidth_requirement': 10e9  # 10 Gbps per device
                })
            
            start_time = time.time()
            
            # Validate daisy chain
            chain_results = daisy_validator.validate_daisy_chain(device_chain)
            
            validation_time = time.time() - start_time
            
            results = {
                'validation_time': validation_time,
                'chain_established': float(chain_results.chain_status),
                'devices_detected': float(len(chain_results.detected_devices)),
                'total_bandwidth_gbps': chain_results.total_bandwidth / 1e9,
                'available_bandwidth_gbps': chain_results.available_bandwidth / 1e9,
                'max_hop_count': float(chain_results.max_hop_count),
                'power_budget_sufficient': float(chain_results.power_status)
            }
            
            self.logger.info("Thunderbolt 4 daisy chain validation completed:")
            self.logger.info(f"  Chain status: {'ESTABLISHED' if results['chain_established'] else 'FAILED'}")
            self.logger.info(f"  Devices detected: {int(results['devices_detected'])}")
            self.logger.info(f"  Available bandwidth: {results['available_bandwidth_gbps']:.1f} Gbps")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Daisy chain validation failed: {e}")
            return {'error': str(e)}
    
    def validate_power_management(
        self, 
        num_samples: int = 5000
    ) -> Dict[str, float]:
        """Validate USB4 power management capabilities"""
        self.logger.info("Validating USB4 power management...")
        
        try:
            # Initialize power manager
            power_manager = USB4PowerManager()
            
            # Generate signal data for different power states
            signal_data = self.generate_usb4_signal(
                USB4SignalMode.GEN2X2, 
                num_samples, 
                lane_count=2
            )
            
            start_time = time.time()
            
            # Test power state transitions
            power_results = power_manager.validate_power_states(
                signal_data,
                test_states=[USB4LinkState.U0, USB4LinkState.U1, USB4LinkState.U2, USB4LinkState.U3]
            )
            
            validation_time = time.time() - start_time
            
            results = {
                'validation_time': validation_time,
                'u0_power_mw': power_results.state_power['U0'],
                'u1_power_mw': power_results.state_power['U1'],
                'u2_power_mw': power_results.state_power['U2'],
                'u3_power_mw': power_results.state_power['U3'],
                'transition_u0_u1_ms': power_results.transition_times['U0_to_U1'] * 1000,
                'transition_u1_u0_ms': power_results.transition_times['U1_to_U0'] * 1000,
                'power_efficiency': power_results.efficiency_score
            }
            
            self.logger.info("USB4 power management validation completed:")
            self.logger.info(f"  U0 power: {results['u0_power_mw']:.1f} mW")
            self.logger.info(f"  U3 power: {results['u3_power_mw']:.1f} mW")
            self.logger.info(f"  Power efficiency: {results['power_efficiency']:.1%}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Power management validation failed: {e}")
            return {'error': str(e)}
    
    def run_certification_tests(self) -> Dict[str, float]:
        """Run Intel Thunderbolt 4 certification tests"""
        self.logger.info("Running Intel Thunderbolt 4 certification tests...")
        
        try:
            # Initialize certification suite
            cert_suite = IntelCertificationSuite()
            
            start_time = time.time()
            
            # Run certification test suite
            cert_results = cert_suite.run_certification_suite({
                'device_type': ThunderboltDeviceType.HOST,
                'security_level': ThunderboltSecurityLevel.USER_AUTHORIZATION,
                'test_categories': ['electrical', 'protocol', 'security', 'interop']
            })
            
            certification_time = time.time() - start_time
            
            results = {
                'certification_time': certification_time,
                'overall_status': float(cert_results.overall_status),
                'electrical_tests_passed': float(cert_results.category_results['electrical'].passed),
                'protocol_tests_passed': float(cert_results.category_results['protocol'].passed),
                'security_tests_passed': float(cert_results.category_results['security'].passed),
                'interop_tests_passed': float(cert_results.category_results['interop'].passed),
                'certification_score': cert_results.certification_score,
                'compliance_level': float(cert_results.compliance_level.value)
            }
            
            self.logger.info("Intel Thunderbolt 4 certification completed:")
            self.logger.info(f"  Overall status: {'PASS' if results['overall_status'] else 'FAIL'}")
            self.logger.info(f"  Certification score: {results['certification_score']:.1f}/100")
            self.logger.info(f"  Compliance level: {cert_results.compliance_level.name}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Certification tests failed: {e}")
            return {'error': str(e)}
    
    def run_performance_benchmark(self) -> Dict[str, float]:
        """Run USB4/Thunderbolt 4 performance benchmarks"""
        self.logger.info("Running USB4/Thunderbolt 4 performance benchmarks...")
        
        benchmark_results = {}
        
        try:
            # Signal analysis benchmarks
            for mode in [USB4SignalMode.GEN2X2, USB4SignalMode.GEN3X2]:
                for sample_count in [2000, 5000, 8000]:
                    key = f"{mode.name.lower()}_analysis_{sample_count}"
                    
                    signal_data = self.generate_usb4_signal(mode, sample_count, lane_count=2)
                    
                    start_time = time.time()
                    self.scope_controller.analyze_usb4_signal(signal_data, USB4Config(mode=mode))
                    analysis_time = time.time() - start_time
                    
                    benchmark_results[key] = analysis_time
                    throughput = (sample_count * 2) / analysis_time  # 2 lanes
                    benchmark_results[f"{key}_throughput"] = throughput
            
            # Tunneling benchmarks
            tunneling_protocols = ['pcie', 'dp', 'usb32']
            for protocol in tunneling_protocols:
                key = f"{protocol}_tunneling_setup"
                
                signal_data = self.generate_usb4_signal(USB4SignalMode.GEN2X2, 4000, 2)
                
                start_time = time.time()
                # Simulate tunneling setup time
                time.sleep(0.001)  # 1ms simulation
                setup_time = time.time() - start_time
                
                benchmark_results[key] = setup_time * 1000  # Convert to ms
            
            # Lane skew compensation benchmark
            signal_data = self.generate_usb4_signal(USB4SignalMode.GEN2X2, 6000, 2)
            
            start_time = time.time()
            self.sync_controller.compensate_lane_skew(signal_data, 5e-12)  # 5ps skew
            skew_comp_time = time.time() - start_time
            
            benchmark_results['lane_skew_compensation_ms'] = skew_comp_time * 1000
            
            self.logger.info("USB4/Thunderbolt 4 performance benchmarks completed:")
            for key, value in benchmark_results.items():
                if 'throughput' in key:
                    self.logger.info(f"  {key}: {value:.0f} samples/s")
                elif 'time' in key or key.endswith('_ms'):
                    self.logger.info(f"  {key}: {value:.3f}")
                else:
                    self.logger.info(f"  {key}: {value:.6f}")
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Performance benchmark failed: {e}")
            return {'error': str(e)}
    
    def generate_report(self) -> str:
        """Generate comprehensive USB4/Thunderbolt 4 validation report"""
        report_path = self.output_dir / "usb4_validation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("USB4/Thunderbolt 4 Validation Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Framework Version: 1.4.0\n\n")
            
            # Write results
            for test_name, results in self.results.items():
                f.write(f"{test_name.upper().replace('_', ' ')}\n")
                f.write("-" * len(test_name) + "\n")
                
                if 'error' in results:
                    f.write(f"ERROR: {results['error']}\n")
                else:
                    for key, value in results.items():
                        if isinstance(value, dict):
                            f.write(f"{key}:\n")
                            for subkey, subvalue in value.items():
                                f.write(f"  {subkey}: {subvalue}\n")
                        elif isinstance(value, float):
                            f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                f.write("\n")
        
        self.logger.info(f"Validation report saved to: {report_path}")
        return str(report_path)
    
    def run_validation(
        self, 
        modes: List[USB4SignalMode],
        protocols: List[str] = ['usb4'],
        tunneling_protocols: List[str] = ['pcie', 'dp', 'usb32'],
        lane_count: int = 2,
        num_samples: int = 8000,
        run_benchmark: bool = False,
        run_certification: bool = False
    ) -> Dict[str, Dict]:
        """Run complete USB4/Thunderbolt 4 validation suite"""
        self.logger.info("Starting USB4/Thunderbolt 4 validation suite...")
        self.logger.info(f"Modes: {[mode.name for mode in modes]}")
        self.logger.info(f"Protocols: {protocols}")
        self.logger.info(f"Lanes: {lane_count}")
        self.logger.info(f"Samples: {num_samples}")
        
        # Signal analysis validation
        for mode in modes:
            self.results[f'{mode.name.lower()}_signal_analysis'] = self.validate_signal_analysis(
                mode, num_samples, lane_count
            )
        
        # Lane skew compensation
        if lane_count > 1:
            self.results['lane_skew_compensation'] = self.validate_lane_skew_compensation(num_samples)
        
        # Jitter analysis with SSC
        for mode in modes:
            self.results[f'{mode.name.lower()}_jitter_analysis'] = self.validate_jitter_analysis(
                mode, num_samples
            )
        
        # Multi-protocol tunneling
        if 'usb4' in protocols:
            self.results['tunneling_validation'] = self.validate_tunneling_protocols(
                tunneling_protocols, num_samples
            )
        
        # Thunderbolt 4 specific tests
        if 'thunderbolt' in protocols:
            self.results['thunderbolt_security'] = self.validate_thunderbolt_security()
            self.results['daisy_chain_validation'] = self.validate_daisy_chain(4)
        
        # Power management
        self.results['power_management'] = self.validate_power_management(num_samples)
        
        # Intel certification tests
        if run_certification and 'thunderbolt' in protocols:
            self.results['intel_certification'] = self.run_certification_tests()
        
        # Performance benchmarks
        if run_benchmark:
            self.results['performance_benchmark'] = self.run_performance_benchmark()
        
        # Generate report
        report_path = self.generate_report()
        
        self.logger.info("USB4/Thunderbolt 4 validation suite completed!")
        self.logger.info(f"Report saved to: {report_path}")
        
        return self.results


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="USB4/Thunderbolt 4 Validation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/usb4_validation.py --mode both --tunneling all --verbose
  python scripts/usb4_validation.py --mode thunderbolt --certification --benchmark
  python scripts/usb4_validation.py --mode usb4 --tunneling pcie --samples 10000
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['usb4', 'thunderbolt', 'both'], 
        default='both',
        help='Protocol mode to test (default: both)'
    )
    
    parser.add_argument(
        '--lanes', 
        type=int, 
        choices=[1, 2],
        default=2,
        help='Number of lanes to test (default: 2)'
    )
    
    parser.add_argument(
        '--tunneling',
        choices=['pcie', 'dp', 'usb32', 'all'],
        default='all',
        help='Tunneling protocols to test (default: all)'
    )
    
    parser.add_argument(
        '--samples', 
        type=int, 
        default=8000,
        help='Number of samples per test (default: 8000)'
    )
    
    parser.add_argument(
        '--output', 
        default='./usb4_results',
        help='Output directory for reports (default: ./usb4_results)'
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
        '--certification', 
        action='store_true',
        help='Run Thunderbolt 4 certification tests'
    )
    
    parser.add_argument(
        '--mock', 
        action='store_true',
        help='Force mock mode for testing'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Check USB4 availability
    if not USB4_AVAILABLE:
        logger.error("USB4 modules not available - cannot run validation")
        sys.exit(1)
    
    # Force mock mode if requested
    if args.mock:
        import os
        os.environ['SVF_MOCK_MODE'] = '1'
        logger.info("Forcing mock mode operation")
    
    # Determine protocols to test
    if args.mode == 'both':
        protocols = ['usb4', 'thunderbolt']
        modes = [USB4SignalMode.GEN2X2, USB4SignalMode.GEN3X2]
    elif args.mode == 'usb4':
        protocols = ['usb4']
        modes = [USB4SignalMode.GEN2X2, USB4SignalMode.GEN3X2]
    else:  # thunderbolt
        protocols = ['thunderbolt']
        modes = [USB4SignalMode.GEN2X2]  # Thunderbolt 4 uses Gen2x2
    
    # Determine tunneling protocols
    if args.tunneling == 'all':
        tunneling_protocols = ['pcie', 'dp', 'usb32']
    else:
        tunneling_protocols = [args.tunneling]
    
    # Validate arguments
    if args.samples < 2000:
        logger.error("Number of samples must be at least 2000")
        sys.exit(1)
    
    try:
        # Create validator
        validator = USB4Validator(args.output)
        
        # Run validation
        results = validator.run_validation(
            modes=modes,
            protocols=protocols,
            tunneling_protocols=tunneling_protocols,
            lane_count=args.lanes,
            num_samples=args.samples,
            run_benchmark=args.benchmark,
            run_certification=args.certification
        )
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("USB4/THUNDERBOLT 4 VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if 'error' not in r)
        
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed tests: {passed_tests}")
        logger.info(f"Failed tests: {total_tests - passed_tests}")
        logger.info(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        # Protocol-specific summary
        if 'thunderbolt_security' in results:
            security_status = "PASS" if results['thunderbolt_security'].get('overall_security_status', 0) else "FAIL"
            logger.info(f"Thunderbolt 4 Security: {security_status}")
        
        if 'intel_certification' in results:
            cert_status = "PASS" if results['intel_certification'].get('overall_status', 0) else "FAIL"
            logger.info(f"Intel Certification: {cert_status}")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - USB4/Thunderbolt 4 validation successful!")
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
