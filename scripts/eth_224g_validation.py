"""
224G Ethernet Validation Test Script

This script provides automated validation testing for 224G Ethernet devices,
supporting both real hardware and mock testing modes. It includes comprehensive
link training and compliance testing capabilities.

Features:
- Flexible hardware/mock mode operation
- Automated test sequencing
- Compliance verification
- Detailed result reporting
- Configurable test parameters

Usage:
    python eth_224g_validation.py [options]

Environment Variables:
    SVF_MOCK_MODE:
        - Set to 1 to force mock mode (no hardware required)
        - Set to 0 to force real hardware mode
        - If not set, automatically detects available hardware
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import sys
import os
import json
import argparse
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.test_sequence.eth_224g_sequence import (
    Ethernet224GTestSequence,
    TrainingResults,
    ComplianceResults
)
from src.serdes_validation_framework.instrument_control.scope_224g import (
    HighBandwidthScope,
    ScopeConfig
)
from src.serdes_validation_framework.protocols.ethernet_224g import (
    ComplianceSpecification,
    ETHERNET_224G_SPECS
)
from src.serdes_validation_framework.instrument_control.mock_controller import (
    get_instrument_controller,
    get_instrument_mode,
    InstrumentMode,
    MockInstrumentController
)

# Get the controller based on environment and availability
controller = get_instrument_controller()
mode = get_instrument_mode()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

@dataclass
class ValidationConfig:
    """
    Validation test configuration with parameter validation
    
    Attributes:
        scope_address: VISA address for oscilloscope
        pattern_gen_address: VISA address for pattern generator
        output_dir: Directory for test results
        test_duration: Test duration in seconds
        skip_training: Whether to skip link training tests
        skip_compliance: Whether to skip compliance tests
        plot_results: Whether to generate result plots
        debug_mode: Enable debug logging
        config_file: Optional JSON configuration file path
        force_mock: Force mock mode operation
    """
    scope_address: str
    pattern_gen_address: str
    output_dir: str
    test_duration: float
    skip_training: bool = False
    skip_compliance: bool = False
    plot_results: bool = True
    debug_mode: bool = False
    config_file: Optional[str] = None
    force_mock: bool = False
    
    # Validation parameters
    min_test_duration: float = field(default=1.0, init=False)
    max_test_duration: float = field(default=3600.0, init=False)
    valid_visa_pattern: str = field(
        default=r'^(GPIB|USB|TCPIP|VXI|ASRL)\d*::.+::INSTR$',
        init=False
    )
    
    def __post_init__(self) -> None:
        """
        Validate configuration parameters after initialization
        
        Raises:
            ValidationError: If any parameters are invalid
        """
        self.validate_visa_address(self.scope_address, "Scope")
        self.validate_visa_address(self.pattern_gen_address, "Pattern generator")
        self.validate_test_duration()
        self.validate_output_dir()
        self.validate_config_file()

    def validate_visa_address(self, address: str, device: str) -> None:
        """
        Validate VISA address format
        
        Args:
            address: VISA address to validate
            device: Device type for error messages
        
        Raises:
            ValidationError: If address is invalid
        """
        if not isinstance(address, str):
            raise ValidationError(f"{device} address must be a string")
            
        if not re.match(self.valid_visa_pattern, address):
            raise ValidationError(
                f"{device} VISA address '{address}' is invalid. "
                "Must match pattern: INTERFACE::BOARD::RESOURCE::INSTR"
            )

    def validate_test_duration(self) -> None:
        """
        Validate test duration
        
        Raises:
            ValidationError: If duration is invalid
        """
        if not isinstance(self.test_duration, (int, float)):
            raise ValidationError("Test duration must be a number")
        
        if not self.min_test_duration <= self.test_duration <= self.max_test_duration:
            raise ValidationError(
                f"Test duration must be between {self.min_test_duration} "
                f"and {self.max_test_duration} seconds"
            )

    def validate_output_dir(self) -> None:
        """
        Validate output directory path
        
        Raises:
            ValidationError: If directory path is invalid
        """
        try:
            path = Path(self.output_dir)
            if path.exists() and not path.is_dir():
                raise ValidationError(
                    f"Output path '{self.output_dir}' exists but is not a directory"
                )
        except Exception as e:
            raise ValidationError(f"Invalid output directory path: {e}")

    def validate_config_file(self) -> None:
        """
        Validate configuration file if specified
        
        Raises:
            ValidationError: If config file is invalid
        """
        if self.config_file:
            path = Path(self.config_file)
            if not path.exists():
                raise ValidationError(
                    f"Configuration file '{self.config_file}' does not exist"
                )
            if not path.is_file():
                raise ValidationError(
                    f"Configuration file path '{self.config_file}' is not a file"
                )
            try:
                with open(path, 'r') as f:
                    json.load(f)  # Validate JSON format
            except json.JSONDecodeError as e:
                raise ValidationError(
                    f"Configuration file '{self.config_file}' is not valid JSON: {e}"
                )

def validate_config_schema(config: Dict[str, Any]) -> None:
    """
    Validate configuration file schema
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValidationError: If schema is invalid
    """
    required_keys = {
        'equipment': {'scope', 'pattern_gen'},
        'test_params': {'duration', 'compliance_limits'},
        'output': {'save_plots', 'save_raw_data'}
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            raise ValidationError(f"Missing required section '{section}'")
        if not isinstance(config[section], dict):
            raise ValidationError(f"Section '{section}' must be a dictionary")
        missing_keys = keys - set(config[section].keys())
        if missing_keys:
            raise ValidationError(
                f"Missing required keys in '{section}': {missing_keys}"
            )

def validate_test_requirements(
    scope: HighBandwidthScope,
    pattern_gen_address: str
) -> None:
    """
    Validate test equipment requirements
    
    Args:
        scope: Configured scope instance
        pattern_gen_address: Pattern generator address
        
    Raises:
        ValidationError: If requirements not met
    """
    # Skip validation in mock mode
    if isinstance(scope.controller, MockInstrumentController):
        logger.info("Skipping hardware validation in mock mode")
        return
        
    # Validate scope bandwidth
    scope_config = scope.default_config
    if scope_config.bandwidth < 100e9:  # 100 GHz minimum
        raise ValidationError(
            f"Insufficient scope bandwidth: {scope_config.bandwidth/1e9:.1f} GHz. "
            "Requires at least 100 GHz"
        )
    
    # Validate sample rate
    if scope_config.sampling_rate < 200e9:  # 200 GSa/s minimum
        raise ValidationError(
            f"Insufficient sample rate: {scope_config.sampling_rate/1e9:.1f} GSa/s. "
            "Requires at least 200 GSa/s"
        )

def parse_arguments() -> ValidationConfig:
    """
    Parse and validate command line arguments
    
    Returns:
        Validated ValidationConfig object
        
    Raises:
        ValidationError: If arguments are invalid
    """
    parser = argparse.ArgumentParser(
        description='224G Ethernet Validation Test Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Equipment configuration
    equipment_group = parser.add_argument_group('Equipment Configuration')
    equipment_group.add_argument(
        '--scope',
        default='GPIB0::7::INSTR',
        help='VISA address for oscilloscope'
    )
    equipment_group.add_argument(
        '--pattern-gen',
        default='GPIB0::10::INSTR',
        help='VISA address for pattern generator'
    )
    equipment_group.add_argument(
        '--force-mock',
        action='store_true',
        help='Force mock mode operation'
    )
    
    # Test configuration
    test_group = parser.add_argument_group('Test Configuration')
    test_group.add_argument(
        '--duration',
        type=float,
        default=10.0,
        help='Test duration in seconds (1-3600)'
    )
    test_group.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip link training tests'
    )
    test_group.add_argument(
        '--skip-compliance',
        action='store_true',
        help='Skip compliance tests'
    )
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output-dir',
        default='validation_results',
        help='Directory for test results'
    )
    output_group.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable result plotting'
    )
    
    # Advanced options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    advanced_group.add_argument(
        '--config',
        help='Path to JSON configuration file'
    )
    
    args = parser.parse_args()
    
    try:
        # Create and validate configuration
        config = ValidationConfig(
            scope_address=args.scope,
            pattern_gen_address=args.pattern_gen,
            output_dir=args.output_dir,
            test_duration=args.duration,
            skip_training=args.skip_training,
            skip_compliance=args.skip_compliance,
            plot_results=not args.no_plots,
            debug_mode=args.debug,
            config_file=args.config,
            force_mock=args.force_mock
        )
        return config
        
    except ValidationError as e:
        parser.error(str(e))

class ValidationTest:
    """224G Ethernet validation test controller"""
    
    def __init__(self, config: ValidationConfig) -> None:
        """
        Initialize validation test with validated configuration
        
        Args:
            config: Validated test configuration
            
        Raises:
            ValidationError: If initialization fails
        """
        self.config = config
        
        try:
            # Set up logging
            log_level = logging.DEBUG if config.debug_mode else logging.INFO
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(f"validation_{datetime.now():%Y%m%d_%H%M%S}.log")
                ]
            )
            
            # Initialize paths
            self.output_dir = Path(config.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.session_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir.mkdir(parents=True)
            
            # Load external configuration if provided
            self.test_config = {}
            if config.config_file:
                self.test_config = self._load_and_validate_config()
            
            # Force mock mode if requested
            if config.force_mock:
                os.environ['SVF_MOCK_MODE'] = '1'
                logger.info("Forcing mock mode operation")
            
            # Initialize equipment using the global controller
            self.scope = HighBandwidthScope(config.scope_address, controller=controller)
            self.sequencer = Ethernet224GTestSequence()
            self.compliance_spec = ComplianceSpecification()
            
            # Validate equipment requirements
            if mode != InstrumentMode.MOCK:
                validate_test_requirements(self.scope, config.pattern_gen_address)
            else:
                logger.info("Skipping hardware requirements validation in mock mode")
            
            logger.info(f"Validation test initialized with session dir: {self.session_dir}")
            logger.info(f"Operating in {mode.value} mode")
            
        except Exception as e:
            logger.error(f"Failed to initialize validation test: {e}")
            raise

    def _load_and_validate_config(self) -> Dict[str, Any]:
        """
        Load and validate external configuration file
        
        Returns:
            Validated configuration dictionary
        
        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            with open(self.config.config_file, 'r') as f:
                config = json.load(f)
            
            # Validate schema
            validate_config_schema(config)
            
            # Validate specific parameters
            self._validate_config_parameters(config)
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _validate_config_parameters(self, config: Dict[str, Any]) -> None:
        """
        Validate specific configuration parameters
        
        Args:
            config: Configuration dictionary to validate
        
        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate compliance limits
        limits = config.get('test_params', {}).get('compliance_limits', {})
        required_limits = {
            'evm_max': (0.0, 10.0),
            'eye_height_min': (0.0, 1.0),
            'jitter_max_ps': (0.0, 100.0)
        }
        
        for limit_name, (min_val, max_val) in required_limits.items():
            if limit_name not in limits:
                raise ValidationError(f"Missing compliance limit: {limit_name}")
            
            value = limits[limit_name]
            if not isinstance(value, (int, float)):
                raise ValidationError(
                    f"Invalid type for {limit_name}: {type(value)}"
                )
            
            if not min_val <= value <= max_val:
                raise ValidationError(
                    f"{limit_name} must be between {min_val} and {max_val}"
                )

    def validate_results(self, results: Dict[str, Any]) -> None:
        """
        Validate test results against requirements
        
        Args:
            results: Test results dictionary
            
        Raises:
            ValidationError: If results are invalid
        """
        try:
            if 'compliance' in results['tests']:
                compliance = results['tests']['compliance']
                
                # Validate EVM
                evm = compliance['evm']['rms_percent']
                if evm > self.test_config.get('test_params', {}).get('compliance_limits', {}).get('evm_max', 5.0):
                    raise ValidationError(f"EVM {evm:.2f}% exceeds maximum limit")
                
                # Validate eye height
                eye_height = compliance['eye_diagram']['worst_height']
                if eye_height < self.test_config.get('test_params', {}).get('compliance_limits', {}).get('eye_height_min', 0.2):
                    raise ValidationError(f"Eye height {eye_height:.3f} below minimum")
                
                # Validate jitter
                jitter_ps = compliance['jitter']['tj'] * 1e12  # Convert to ps
                if jitter_ps > self.test_config.get('test_params', {}).get('compliance_limits', {}).get('jitter_max_ps', 50.0):
                    raise ValidationError(f"Total jitter {jitter_ps:.2f}ps exceeds maximum")
            
        except Exception as e:
            logger.error(f"Results validation failed: {e}")
            raise

    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save test results with validation
        
        Args:
            results: Test results dictionary
            
        Raises:
            ValidationError: If results cannot be saved
        """
        try:
            # Validate results before saving
            self.validate_results(results)
            
            # Save to JSON file
            results_file = self.session_dir / 'validation_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Save summary report
            summary_file = self.session_dir / 'validation_summary.txt'
            with open(summary_file, 'w') as f:
                f.write("224G Ethernet Validation Test Summary\n")
                f.write("=====================================\n\n")
                
                f.write(f"Test Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                f.write(f"Equipment:\n")
                f.write(f"  Scope: {self.config.scope_address}\n")
                f.write(f"  Pattern Generator: {self.config.pattern_gen_address}\n\n")
                
                if 'link_training' in results['tests']:
                    training = results['tests']['link_training']
                    f.write("Link Training Results:\n")
                    f.write(f"  Status: {training['status']}\n")
                    f.write(f"  Training Time: {training['training_time']:.2f} s\n")
                    f.write(f"  Final Error: {training['final_error']:.6f}\n\n")
                
                if 'compliance' in results['tests']:
                    compliance = results['tests']['compliance']
                    f.write("Compliance Test Results:\n")
                    f.write(f"  Overall Status: {compliance['status']}\n")
                    f.write(f"  RMS EVM: {compliance['evm']['rms_percent']:.2f}%\n")
                    f.write(f"  Peak EVM: {compliance['evm']['peak_percent']:.2f}%\n")
                    f.write(f"  Worst Eye Height: {compliance['eye_diagram']['worst_height']:.3f}\n")
                    f.write(f"  Worst Eye Width: {compliance['eye_diagram']['worst_width']:.3f}\n")
                    f.write("\nJitter Components:\n")
                    for jitter_type, value in compliance['jitter'].items():
                        f.write(f"  {jitter_type.upper()}: {value:.3f} ps\n")
            
            logger.info(f"Results saved to {self.session_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

    def run_validation_suite(self) -> Dict[str, Any]:
        """
        Run validation test suite with enhanced validation
        
        Returns:
            Dictionary containing validated test results
            
        Raises:
            ValidationError: If test suite fails
        """
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'equipment': {
                        'scope': self.config.scope_address,
                        'pattern_gen': self.config.pattern_gen_address
                    },
                    'test_duration': self.config.test_duration,
                    'skip_training': self.config.skip_training,
                    'skip_compliance': self.config.skip_compliance,
                    'external_config': self.test_config
                },
                'tests': {}
            }
            
            # Setup phase
            logger.info("Setting up instruments...")
            self.sequencer.setup_instruments([
                self.config.scope_address,
                self.config.pattern_gen_address
            ])
            
            # Configure scope with validation
            scope_config = ScopeConfig(
                sampling_rate=256e9,
                bandwidth=120e9,
                timebase=5e-12,
                voltage_range=0.8
            )
            self.scope.configure_for_224g(scope_config)
            
            # Link training test if not skipped 
            if not self.config.skip_training:
                logger.info("Running link training test...")
                training_results = self.sequencer.run_link_training_test(
                    self.config.scope_address,
                    self.config.pattern_gen_address,
                    timeout_seconds=self.config.test_duration
                )
                results['tests']['link_training'] = self._process_training_results(
                    training_results
                )
                
                if self.config.plot_results:
                    self._plot_training_results(training_results)
            
            # Compliance tests if not skipped
            if not self.config.skip_compliance:
                logger.info("Running compliance tests...")
                compliance_results = self.sequencer.run_compliance_test_suite(
                    self.config.scope_address,
                    self.config.pattern_gen_address,
                    test_duration=self.config.test_duration
                )
                results['tests']['compliance'] = self._process_compliance_results(
                    compliance_results
                )
                
                if self.config.plot_results:
                    self._plot_compliance_results(compliance_results)
            
            # Validate and save results
            self.validate_results(results)
            self.save_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Validation suite failed: {e}")
            raise
        finally:
            self._cleanup()

    def _process_training_results(self, results: TrainingResults) -> Dict[str, Any]:
        """
        Process link training results into report format
        
        Args:
            results: TrainingResults object
            
        Returns:
            Dictionary of processed results
        """
        return {
            'status': results.convergence_status,
            'training_time': results.training_time,
            'converged': results.convergence_status == 'PASS',
            'final_error': results.adaptation_error,
            'equalizer_taps': results.final_eq_settings
        }

    def _process_compliance_results(self, results: ComplianceResults) -> Dict[str, Any]:
        """
        Process compliance test results into report format
        
        Args:
            results: ComplianceResults object
            
        Returns:
            Dictionary of processed results
        """
        return {
            'status': results.test_status,
            'evm': {
                'rms_percent': results.evm_results.rms_evm_percent,
                'peak_percent': results.evm_results.peak_evm_percent
            },
            'eye_diagram': {
                'worst_height': results.eye_results.worst_eye_height,
                'worst_width': results.eye_results.worst_eye_width
            },
            'jitter': results.jitter_results
        }

    def _plot_training_results(self, results: TrainingResults) -> None:
        """
        Generate training results plots
        
        Args:
            results: TrainingResults object
        """
        if not self.config.plot_results:
            return
            
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(results.error_history)
            plt.title('Link Training Convergence')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.grid(True)
            plt.savefig(self.session_dir / 'training_convergence.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot training results: {e}")

    def _plot_compliance_results(self, results: ComplianceResults) -> None:
        """
        Generate compliance test plots
        
        Args:
            results: ComplianceResults object
        """
        if not self.config.plot_results:
            return
            
        try:
            # Eye diagram plot
            plt.figure(figsize=(10, 6))
            heights = results.eye_results.eye_heights
            plt.bar(range(len(heights)), heights)
            plt.title('Eye Heights by Level')
            plt.xlabel('Level')
            plt.ylabel('Eye Height')
            plt.grid(True)
            plt.savefig(self.session_dir / 'eye_heights.png')
            plt.close()
            
            # Jitter components plot
            plt.figure(figsize=(10, 6))
            jitter_data = results.jitter_results
            plt.bar(jitter_data.keys(), jitter_data.values())
            plt.title('Jitter Components')
            plt.xlabel('Component')
            plt.ylabel('Time (ps)')
            plt.grid(True)
            plt.savefig(self.session_dir / 'jitter_components.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot compliance results: {e}")

    def _cleanup(self) -> None:
        """Clean up resources and connections"""
        try:
            # List of instruments to disconnect
            instruments = [
                self.config.scope_address,
                self.config.pattern_gen_address
            ]
            
            # Clean up scope
            if hasattr(self, 'scope'):
                try:
                    self.scope.cleanup()
                except Exception as e:
                    logger.warning(f"Scope cleanup failed: {e}")
                    
            # Clean up sequencer
            if hasattr(self, 'sequencer'):
                try:
                    self.sequencer.cleanup(instruments)
                except Exception as e:
                    logger.warning(f"Sequencer cleanup failed: {e}")
                    
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

def main() -> None:
    """Main validation script function with enhanced error handling"""
    try:
        # Parse and validate command line arguments
        config = parse_arguments()
        
        # Run validation with progress updates
        validator = ValidationTest(config)
        print("\nStarting 224G Ethernet Validation...")
        print(f"Operating in {mode.value} mode")
        
        results = validator.run_validation_suite()
        
        # Print formatted summary
        print("\n=== Validation Test Summary ===")
        print(f"Test Date: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"\nResults Directory: {validator.session_dir}")
        
        if not config.skip_training and 'link_training' in results['tests']:
            print("\nLink Training:")
            training = results['tests']['link_training']
            print(f"- Status: {training['status']}")
            print(f"- Training Time: {training['training_time']:.2f} s")
            print(f"- Converged: {training['converged']}")
        
        if not config.skip_compliance and 'compliance' in results['tests']:
            print("\nCompliance Tests:")
            compliance = results['tests']['compliance']
            print(f"- Status: {compliance['status']}")
            print(f"- RMS EVM: {compliance['evm']['rms_percent']:.2f}%")
            print(f"- Worst Eye Height: {compliance['eye_diagram']['worst_height']:.3f}")
            print(f"- Total Jitter: {compliance['jitter']['tj']:.3f} ps")
        
        # Set exit code based on test status
        if ('compliance' in results['tests'] and 
            results['tests']['compliance']['status'] != 'PASS'):
            sys.exit(1)
        
    except ValidationError as e:
        logger.error(f"Validation Error: {e}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()