#!/usr/bin/env python3
"""
Comprehensive SerDes Validation Suite

This script provides a complete validation suite for USB4/Thunderbolt 4 devices
using the enhanced SerDes Validation Framework v1.4.0. It includes:

- Enhanced daisy chain validation with real device monitoring
- Advanced security testing with DMA attack simulation  
- Comprehensive certification testing
- Performance benchmarking and stress testing
- Professional reporting with detailed metrics

Usage:
    python comprehensive_validation_suite.py [options]

Options:
    --device-chain PATH     Path to device chain configuration file
    --security-config PATH Path to security configuration file
    --output-dir PATH       Output directory for reports (default: ./reports)
    --format FORMAT         Report format: html, json, pdf, all (default: html)
    --test-duration SECONDS Test duration in seconds (default: 60)
    --enable-stress         Enable stress testing
    --enable-security       Enable security testing
    --enable-certification  Enable certification testing
    --mock-mode            Enable mock mode for testing without hardware
    --verbose              Enable verbose logging
    --help                 Show this help message
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from serdes_validation_framework.protocols.usb4.constants import USB4SignalMode
from serdes_validation_framework.protocols.usb4.reporting import (
    ReportFormat,
    ReportTemplate,
    ReportType,
    TestSession,
    USB4TestReporter,
)
from serdes_validation_framework.protocols.usb4.thunderbolt.certification import (
    CertificationConfig,
    CertificationTestType,
    IntelCertificationSuite,
)
from serdes_validation_framework.protocols.usb4.thunderbolt.constants import ThunderboltDeviceType
from serdes_validation_framework.protocols.usb4.thunderbolt.daisy_chain import (
    ChainDevice,
    DaisyChainTestConfig,
    DaisyChainValidator,
)
from serdes_validation_framework.protocols.usb4.thunderbolt.security import (
    SecurityTestConfig,
    ThunderboltAuthMethod,
    ThunderboltSecurityValidator,
)


class ComprehensiveValidationSuite:
    """Comprehensive validation suite for USB4/Thunderbolt 4 devices"""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize validation suite with command line arguments"""
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.output_dir / 'validation_suite.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Enable mock mode if requested
        if args.mock_mode:
            os.environ['SVF_MOCK_MODE'] = '1'
            self.logger.info("Mock mode enabled")
        
        # Initialize components
        self.reporter = USB4TestReporter()
        self.test_results = {}
        self.session_id = f"validation_suite_{int(time.time())}"
        
        self.logger.info(f"Initialized validation suite with session ID: {self.session_id}")
    
    def load_device_chain_config(self, config_path: Optional[str] = None) -> List[ChainDevice]:
        """Load device chain configuration from file or create default"""
        if config_path and Path(config_path).exists():
            self.logger.info(f"Loading device chain configuration from {config_path}")
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                devices = []
                for device_data in config_data.get('devices', []):
                    device = ChainDevice(
                        device_id=device_data['device_id'],
                        device_type=ThunderboltDeviceType[device_data['device_type']],
                        position=device_data['position'],
                        vendor_id=device_data['vendor_id'],
                        product_id=device_data['product_id'],
                        firmware_version=device_data['firmware_version'],
                        power_consumption=device_data['power_consumption'],
                        bandwidth_requirement=device_data['bandwidth_requirement'],
                        downstream_ports=device_data.get('downstream_ports', []),
                        is_hub=device_data.get('is_hub', False),
                        max_downstream_devices=device_data.get('max_downstream_devices', 1)
                    )
                    devices.append(device)
                
                return devices
                
            except Exception as e:
                self.logger.error(f"Failed to load device chain configuration: {e}")
                self.logger.info("Using default device chain configuration")
        
        # Create default device chain
        return self._create_default_device_chain()
    
    def _create_default_device_chain(self) -> List[ChainDevice]:
        """Create default device chain for testing"""
        return [
            ChainDevice(
                device_id="host_controller",
                device_type=ThunderboltDeviceType.HOST,
                position=0,
                vendor_id="8086",
                product_id="1234",
                firmware_version="1.0.0",
                power_consumption=5.0,
                bandwidth_requirement=10e9,
                is_hub=False
            ),
            ChainDevice(
                device_id="thunderbolt_dock",
                device_type=ThunderboltDeviceType.HUB,
                position=1,
                vendor_id="8086",
                product_id="5678",
                firmware_version="2.1.0",
                power_consumption=20.0,
                bandwidth_requirement=25e9,
                downstream_ports=["port1", "port2", "port3", "port4"],
                is_hub=True,
                max_downstream_devices=4
            ),
            ChainDevice(
                device_id="4k_display",
                device_type=ThunderboltDeviceType.DISPLAY,
                position=2,
                vendor_id="1002",
                product_id="9ABC",
                firmware_version="3.2.1",
                power_consumption=30.0,
                bandwidth_requirement=35e9,
                is_hub=False
            ),
            ChainDevice(
                device_id="nvme_ssd",
                device_type=ThunderboltDeviceType.DEVICE,
                position=3,
                vendor_id="144D",
                product_id="DEF0",
                firmware_version="1.5.2",
                power_consumption=12.0,
                bandwidth_requirement=20e9,
                is_hub=False
            ),
            ChainDevice(
                device_id="audio_interface",
                device_type=ThunderboltDeviceType.DEVICE,
                position=4,
                vendor_id="1234",
                product_id="ABCD",
                firmware_version="2.0.1",
                power_consumption=8.0,
                bandwidth_requirement=5e9,
                is_hub=False
            )
        ]
    
    def load_security_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load security configuration from file or create default"""
        if config_path and Path(config_path).exists():
            self.logger.info(f"Loading security configuration from {config_path}")
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load security configuration: {e}")
                self.logger.info("Using default security configuration")
        
        # Create default security configuration
        import base64
        return {
            'device_id': 'secure_device_001',
            'vendor_id': '8086',
            'product_id': 'ABCD',
            'device_type': 'hub',
            'firmware_version': '2.1.0',
            'security_level': 'SL1',
            'supported_protocols': ['USB4_v2.0', 'TB4', 'USB3.2'],
            'max_power_consumption': 25.0,
            'max_power_delivery': 100.0,
            'max_operating_temperature': 85.0,
            'power_requirement': 15.0,
            'device_key': '0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF',
            'expected_key': '0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF',
            'challenge': 'FEDCBA9876543210FEDCBA9876543210FEDCBA9876543210FEDCBA9876543210',
            'certificate': '''-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAKoK/heBjcOuMA0GCSqGSIb3DQEBBQUAMBQxEjAQBgNVBAMMCVRo
dW5kZXJib2x0MB4XDTIzMDEwMTAwMDAwMFoXDTI0MDEwMTAwMDAwMFowFDESMBAG
A1UEAwwJVGh1bmRlcmJvbHQwXDANBgkqhkiG9w0BAQEFAANLADBIAkEAuGaP/fcj
Subject: CN=Thunderbolt Device
Issuer: CN=Thunderbolt CA
Serial Number: 789012
Not Before: 2023-01-01T00:00:00Z
Not After: 2024-12-31T23:59:59Z
-----END CERTIFICATE-----''',
            'biometric_data': base64.b64encode(b'comprehensive_fingerprint_data' * 100).decode(),
            'biometric_type': 'fingerprint',
            'reference_template': base64.b64encode(b'reference_fingerprint_template' * 100).decode()
        }
    
    def run_daisy_chain_validation(self, device_chain: List[ChainDevice]) -> Dict[str, Any]:
        """Run comprehensive daisy chain validation"""
        self.logger.info("Starting daisy chain validation")
        
        # Create configuration
        config = DaisyChainTestConfig(
            max_devices=6,
            test_duration=self.args.test_duration,
            bandwidth_test_enabled=True,
            power_test_enabled=True,
            stability_test_enabled=True,
            topology_validation_enabled=True,
            stress_test_enabled=self.args.enable_stress,
            hot_plug_test_enabled=True
        )
        
        # Create validator and run tests
        validator = DaisyChainValidator(config)
        
        start_time = time.time()
        results = validator.run_comprehensive_chain_test(device_chain)
        execution_time = time.time() - start_time
        
        # Convert results to dictionary for JSON serialization
        results_dict = {
            'overall_status': results.overall_status.name,
            'topology': {
                'total_devices': results.topology.total_devices,
                'hub_count': results.topology.hub_count,
                'display_count': results.topology.display_count,
                'max_chain_length': results.topology.max_chain_length,
                'topology_valid': results.topology.topology_valid,
                'topology_map': results.topology.topology_map
            },
            'bandwidth_allocation': {
                'total_bandwidth': results.bandwidth_allocation.total_bandwidth,
                'allocated_bandwidth': results.bandwidth_allocation.allocated_bandwidth,
                'available_bandwidth': results.bandwidth_allocation.available_bandwidth,
                'efficiency': results.bandwidth_allocation.efficiency,
                'congestion_points': results.bandwidth_allocation.congestion_points
            },
            'power_distribution': {
                'total_power_budget': results.power_distribution.total_power_budget,
                'consumed_power': results.power_distribution.consumed_power,
                'available_power': results.power_distribution.available_power,
                'delivery_efficiency': results.power_distribution.delivery_efficiency,
                'power_violations': results.power_distribution.power_violations
            },
            'stability_score': results.stability_score,
            'hot_plug_events': results.hot_plug_events,
            'performance_metrics': results.performance_metrics,
            'recommendations': results.recommendations,
            'test_duration': results.test_duration,
            'execution_time': execution_time
        }
        
        self.logger.info(f"Daisy chain validation completed in {execution_time:.2f}s")
        self.logger.info(f"Overall status: {results.overall_status.name}")
        
        return results_dict
    
    def run_security_validation(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive security validation"""
        self.logger.info("Starting security validation")
        
        # Create configuration
        config = SecurityTestConfig()
        
        # Create validator and run tests
        validator = ThunderboltSecurityValidator(config)
        
        start_time = time.time()
        results = validator.validate_device_security(device_info)
        execution_time = time.time() - start_time
        
        # Test individual components
        dma_attacks_blocked = validator._simulate_dma_attacks()
        
        # Test authentication methods
        auth_results = {}
        auth_methods = [
            ('key_based', ThunderboltAuthMethod.KEY_BASED),
            ('certificate', ThunderboltAuthMethod.CERTIFICATE),
            ('biometric', ThunderboltAuthMethod.BIOMETRIC)
        ]
        
        for auth_name, auth_method in auth_methods:
            auth_results[auth_name] = validator._perform_authentication(device_info, auth_method)
        
        # Test certificate validation
        cert_result = validator._validate_device_certificate(device_info.get('certificate', ''))
        
        # Convert results to dictionary
        results_dict = {
            'overall_status': results.overall_status.name,
            'security_score': results.security_score,
            'dma_protection_status': results.dma_protection_status.name,
            'dma_attacks_blocked': dma_attacks_blocked,
            'authentication_results': auth_results,
            'certificate_validation': cert_result,
            'test_duration': results.test_duration,
            'execution_time': execution_time
        }
        
        self.logger.info(f"Security validation completed in {execution_time:.2f}s")
        self.logger.info(f"Security score: {results.security_score:.2%}")
        
        return results_dict
    
    def run_certification_testing(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive certification testing"""
        self.logger.info("Starting certification testing")
        
        # Create configuration
        config = CertificationConfig()
        
        # Create certification suite
        cert_suite = IntelCertificationSuite(config)
        
        start_time = time.time()
        
        # Run basic certification
        basic_results = cert_suite.run_basic_certification(device_info)
        
        # Run individual component tests
        from serdes_validation_framework.protocols.usb4.thunderbolt.certification import CertificationTestCase
        
        component_tests = [
            ('link_training', 'Link Training Test'),
            ('power_management', 'Power Management Test'),
            ('tunneling', 'Tunneling Test'),
            ('basic_interop', 'Basic Interoperability Test'),
            ('multi_vendor', 'Multi-vendor Compatibility Test'),
            ('usb_pd', 'USB-PD Compliance Test'),
            ('power_efficiency', 'Power Efficiency Test'),
            ('thermal_management', 'Thermal Management Test')
        ]
        
        component_results = {}
        
        for test_id, test_name in component_tests:
            test_case = CertificationTestCase(
                test_id=test_id,
                test_name=test_name,
                test_type=CertificationTestType.BASIC,
                required=True,
                timeout=30.0
            )
            
            # Run specific test
            if test_id == 'link_training':
                result = cert_suite._run_link_training_test(test_case, device_info)
            elif test_id == 'power_management':
                result = cert_suite._run_power_management_test(test_case, device_info)
            elif test_id == 'tunneling':
                result = cert_suite._run_tunneling_test(test_case, device_info)
            elif test_id == 'basic_interop':
                result = cert_suite._run_basic_interop_test(test_case, device_info)
            elif test_id == 'multi_vendor':
                result = cert_suite._run_multi_vendor_test(test_case, device_info)
            elif test_id == 'usb_pd':
                result = cert_suite._run_usb_pd_test(test_case, device_info)
            elif test_id == 'power_efficiency':
                result = cert_suite._run_power_efficiency_test(test_case, device_info)
            elif test_id == 'thermal_management':
                result = cert_suite._run_thermal_management_test(test_case, device_info)
            else:
                continue
            
            component_results[test_id] = {
                'test_name': test_name,
                'result': result.result.name,
                'score': result.score,
                'execution_time': result.execution_time,
                'details': result.details
            }
        
        execution_time = time.time() - start_time
        
        # Convert results to dictionary
        results_dict = {
            'basic_certification': {
                'overall_status': basic_results.overall_status.name,
                'certification_score': basic_results.certification_score,
                'tests_passed': basic_results.tests_passed,
                'total_tests': basic_results.total_tests,
                'test_duration': basic_results.test_duration
            },
            'component_results': component_results,
            'execution_time': execution_time
        }
        
        self.logger.info(f"Certification testing completed in {execution_time:.2f}s")
        self.logger.info(f"Certification score: {basic_results.certification_score:.2%}")
        
        return results_dict
    
    def generate_reports(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive reports"""
        self.logger.info("Generating reports")
        
        # Create test session
        session = TestSession(
            session_id=self.session_id,
            timestamp=time.time(),
            test_type=ReportType.COMPLIANCE,
            signal_mode=USB4SignalMode.GEN3X2,
            device_info={
                'session_id': self.session_id,
                'test_suite': 'Comprehensive Validation Suite',
                'framework_version': '1.4.0'
            }
        )
        
        # Add test session to reporter
        self.reporter.add_test_session(session)
        
        # Determine report formats
        if self.args.format == 'all':
            formats = [ReportFormat.HTML, ReportFormat.JSON, ReportFormat.PDF]
        else:
            format_map = {
                'html': ReportFormat.HTML,
                'json': ReportFormat.JSON,
                'pdf': ReportFormat.PDF
            }
            formats = [format_map.get(self.args.format, ReportFormat.HTML)]
        
        generated_reports = []
        
        for report_format in formats:
            try:
                # Create report template
                template = ReportTemplate(
                    name=f"comprehensive_validation_{report_format.name.lower()}",
                    format=report_format,
                    sections=['summary', 'daisy_chain', 'security', 'certification', 'recommendations']
                )
                
                # Generate report
                report_path = self.reporter.generate_compliance_report(
                    session.session_id,
                    template
                )
                
                generated_reports.append(report_path)
                self.logger.info(f"Generated {report_format.name} report: {report_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate {report_format.name} report: {e}")
        
        # Save raw test results as JSON
        results_file = self.output_dir / f"test_results_{self.session_id}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            generated_reports.append(str(results_file))
            self.logger.info(f"Saved raw test results: {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save test results: {e}")
        
        return generated_reports
    
    def run_comprehensive_validation(self) -> int:
        """Run comprehensive validation suite"""
        self.logger.info("Starting comprehensive validation suite")
        
        try:
            # Load configurations
            device_chain = self.load_device_chain_config(self.args.device_chain)
            security_config = self.load_security_config(self.args.security_config)
            
            # Initialize test results
            test_results = {
                'session_id': self.session_id,
                'timestamp': time.time(),
                'configuration': {
                    'test_duration': self.args.test_duration,
                    'enable_stress': self.args.enable_stress,
                    'enable_security': self.args.enable_security,
                    'enable_certification': self.args.enable_certification,
                    'mock_mode': self.args.mock_mode
                },
                'device_chain': [
                    {
                        'device_id': device.device_id,
                        'device_type': device.device_type.name,
                        'position': device.position,
                        'power_consumption': device.power_consumption,
                        'bandwidth_requirement': device.bandwidth_requirement
                    }
                    for device in device_chain
                ]
            }
            
            # Run daisy chain validation
            print("🔗 Running daisy chain validation...")
            daisy_chain_results = self.run_daisy_chain_validation(device_chain)
            test_results['daisy_chain'] = daisy_chain_results
            
            # Run security validation if enabled
            if self.args.enable_security:
                print("🔒 Running security validation...")
                security_results = self.run_security_validation(security_config)
                test_results['security'] = security_results
            
            # Run certification testing if enabled
            if self.args.enable_certification:
                print("🏆 Running certification testing...")
                certification_results = self.run_certification_testing(security_config)
                test_results['certification'] = certification_results
            
            # Generate reports
            print("📊 Generating reports...")
            generated_reports = self.generate_reports(test_results)
            test_results['generated_reports'] = generated_reports
            
            # Display summary
            self._display_summary(test_results)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Comprehensive validation failed: {e}")
            print(f"❌ Validation suite failed: {e}")
            return 1
    
    def _display_summary(self, test_results: Dict[str, Any]):
        """Display validation summary"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE VALIDATION SUMMARY")
        print("="*80)
        
        # Session info
        print(f"Session ID: {test_results['session_id']}")
        print(f"Timestamp: {time.ctime(test_results['timestamp'])}")
        print(f"Test Duration: {test_results['configuration']['test_duration']}s")
        
        # Daisy chain results
        if 'daisy_chain' in test_results:
            dc_results = test_results['daisy_chain']
            print("\n🔗 Daisy Chain Validation:")
            print(f"  Status: {dc_results['overall_status']}")
            print(f"  Stability Score: {dc_results['stability_score']:.2%}")
            print(f"  Bandwidth Efficiency: {dc_results['bandwidth_allocation']['efficiency']:.2%}")
            print(f"  Power Efficiency: {dc_results['power_distribution']['delivery_efficiency']:.2%}")
            print(f"  Devices Tested: {dc_results['topology']['total_devices']}")
            print(f"  Hot Plug Events: {len(dc_results['hot_plug_events'])}")
        
        # Security results
        if 'security' in test_results:
            sec_results = test_results['security']
            print("\n🔒 Security Validation:")
            print(f"  Status: {sec_results['overall_status']}")
            print(f"  Security Score: {sec_results['security_score']:.2%}")
            print(f"  DMA Protection: {sec_results['dma_protection_status']}")
            print(f"  Attacks Blocked: {sec_results['dma_attacks_blocked']}")
            print(f"  Certificate Valid: {sec_results['certificate_validation']}")
        
        # Certification results
        if 'certification' in test_results:
            cert_results = test_results['certification']
            basic = cert_results['basic_certification']
            components = cert_results['component_results']
            
            print("\n🏆 Certification Testing:")
            print(f"  Overall Status: {basic['overall_status']}")
            print(f"  Certification Score: {basic['certification_score']:.2%}")
            print(f"  Tests Passed: {basic['tests_passed']}/{basic['total_tests']}")
            
            passed_components = sum(1 for comp in components.values() if comp['result'] == 'PASS')
            print(f"  Component Tests Passed: {passed_components}/{len(components)}")
        
        # Reports generated
        if 'generated_reports' in test_results:
            reports = test_results['generated_reports']
            print(f"\n📊 Generated Reports: {len(reports)}")
            for report in reports:
                print(f"  • {Path(report).name}")
        
        print("\n" + "="*80)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Comprehensive SerDes Validation Suite v1.4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation with HTML report
  python comprehensive_validation_suite.py
  
  # Full validation with all report formats
  python comprehensive_validation_suite.py --enable-stress --enable-security --enable-certification --format all
  
  # Custom configuration with extended test duration
  python comprehensive_validation_suite.py --device-chain config/devices.json --test-duration 120
  
  # Mock mode for CI/CD testing
  python comprehensive_validation_suite.py --mock-mode --format json --output-dir ./ci_reports
        """
    )
    
    parser.add_argument(
        '--device-chain',
        type=str,
        help='Path to device chain configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--security-config',
        type=str,
        help='Path to security configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./reports',
        help='Output directory for reports (default: ./reports)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['html', 'json', 'pdf', 'all'],
        default='html',
        help='Report format (default: html)'
    )
    
    parser.add_argument(
        '--test-duration',
        type=float,
        default=60.0,
        help='Test duration in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--enable-stress',
        action='store_true',
        help='Enable stress testing'
    )
    
    parser.add_argument(
        '--enable-security',
        action='store_true',
        help='Enable security testing'
    )
    
    parser.add_argument(
        '--enable-certification',
        action='store_true',
        help='Enable certification testing'
    )
    
    parser.add_argument(
        '--mock-mode',
        action='store_true',
        help='Enable mock mode for testing without hardware'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main():
    """Main function"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    print("🚀 SerDes Validation Framework v1.4.0 - Comprehensive Validation Suite")
    print("="*80)
    
    # Create and run validation suite
    suite = ComprehensiveValidationSuite(args)
    return suite.run_comprehensive_validation()


if __name__ == '__main__':
    sys.exit(main())
