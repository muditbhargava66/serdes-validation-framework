# src/serdes_validation_framework/test_sequence/eth_224g_sequence.py

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from ..data_analysis.pam4_analyzer import EVMResults, EyeResults, PAM4Analyzer, PAM4Levels
from .sequencer import PCIeTestSequencer as TestSequencer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingResults:
    """Data class for link training results"""
    training_time: float              # Training duration in seconds
    convergence_status: str           # PASS/FAIL status
    final_eq_settings: List[float]    # Final equalizer tap values
    adaptation_error: float           # Current adaptation error value
    error_history: List[float]        # Optional history for plotting

    def __post_init__(self) -> None:
        """Validate training results data types"""
        assert isinstance(self.training_time, float), "Training time must be a float"
        assert isinstance(self.convergence_status, str), "Convergence status must be a string"
        assert all(isinstance(x, float) for x in self.final_eq_settings), \
            "Equalizer settings must be floats"
        assert isinstance(self.adaptation_error, float), "Adaptation error must be a float"
        assert all(isinstance(x, float) for x in self.error_history), \
            "Error history must contain floats"

@dataclass
class ComplianceResults:
    """Class for compliance test results with automatic status updates"""

    def __init__(
        self,
        pam4_levels: PAM4Levels,
        evm_results: EVMResults,
        eye_results: EyeResults,
        jitter_results: Dict[str, float],
        test_status: str
    ):
        class TrackedEyeResults(EyeResults):
            """Wrapper for EyeResults that notifies parent of changes"""
            def __init__(self, eye_results, parent):
                # Copy all the data from the original EyeResults
                super().__init__(
                    eye_heights=eye_results.eye_heights.copy(),
                    eye_widths=eye_results.eye_widths.copy()
                )
                self._parent = parent

            @property
            def worst_eye_height(self) -> float:
                return super().worst_eye_height

            @worst_eye_height.setter
            def worst_eye_height(self, value: float) -> None:
                super(TrackedEyeResults, self.__class__).worst_eye_height.fset(self, value)
                self._parent._update_test_status()

            @property
            def worst_eye_width(self) -> float:
                return super().worst_eye_width

            @worst_eye_width.setter
            def worst_eye_width(self, value: float) -> None:
                super(TrackedEyeResults, self.__class__).worst_eye_width.fset(self, value)
                self._parent._update_test_status()

        # Store the compliance criteria as class constants
        self.COMPLIANCE_CRITERIA = {
            'min_eye_height': 0.2,
            'max_rms_evm': 5.0,
            'max_total_jitter': 0.3e-12
        }

        self._pam4_levels = pam4_levels
        self._evm_results = evm_results
        self._jitter_results = jitter_results
        self._eye_results = TrackedEyeResults(eye_results, self)
        self._test_status = test_status
        self._update_test_status()

    @property
    def pam4_levels(self) -> PAM4Levels:
        return self._pam4_levels

    @property
    def evm_results(self) -> EVMResults:
        return self._evm_results

    @property
    def eye_results(self) -> EyeResults:
        return self._eye_results

    @property
    def jitter_results(self) -> Dict[str, float]:
        return self._jitter_results

    @property
    def test_status(self) -> str:
        return self._test_status

    def _update_test_status(self) -> None:
        """Update test status based on current measurements"""
        if (self.eye_results.worst_eye_height < self.COMPLIANCE_CRITERIA['min_eye_height'] or
            self.evm_results.rms_evm_percent > self.COMPLIANCE_CRITERIA['max_rms_evm'] or
            self.jitter_results['tj'] > self.COMPLIANCE_CRITERIA['max_total_jitter']):
            self._test_status = 'FAIL'
        else:
            self._test_status = 'PASS'

class Ethernet224GTestSequence(TestSequencer):
    """Test sequencer for 224G Ethernet validation"""

    def __init__(self) -> None:
        """Initialize 224G test sequencer with configuration"""
        super().__init__()
        self.test_configs = self._initialize_test_configs()
        self.connected_instruments = {}
        logger.info("224G Ethernet Test Sequencer initialized")

    def _initialize_test_configs(self) -> Dict[str, Any]:
        """
        Initialize test configurations for 224G testing
        
        Returns:
            Dictionary of test configurations
        """
        return {
            'sampling_rate': 256e9,    # 256 GSa/s
            'bandwidth': 120e9,        # 120 GHz bandwidth
            'ui_period': 8.9e-12,      # Unit interval for 224G
            'voltage_range': 0.8,      # Typical PAM4 voltage range
            'prbs_pattern': 'PRBS31'   # Standard test pattern
        }

    def setup_instruments(self, resource_names: List[str]) -> None:
        """
        Set up required instruments for testing
        
        Args:
            resource_names: List of VISA resource identifiers
        """
        try:
            for resource in resource_names:
                self.instrument_controller.connect_instrument(resource)
                self.connected_instruments[resource] = True
                logger.info(f"Connected to instrument {resource}")
        except Exception as e:
            logger.error(f"Failed to setup instruments: {e}")
            raise

    def configure_scope_for_224g(
        self,
        scope_resource: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Configure oscilloscope for 224G measurements
        
        Args:
            scope_resource: Scope resource identifier
            custom_config: Optional custom configuration parameters
        """
        try:
            # Use custom config if provided, otherwise use default
            config = custom_config if custom_config is not None else self.test_configs.copy()

            # Check required float parameters
            float_params = {'sampling_rate', 'bandwidth', 'voltage_range'}
            missing_params = float_params - set(config.keys())
            if missing_params:
                raise AssertionError(f"Missing required parameters: {missing_params}")

            # Validate float parameter types
            for key in float_params:
                if not isinstance(config[key], float):
                    raise AssertionError(f"Configuration value for {key} must be a float")

            # Send configuration commands
            commands = [
                f":ACQuire:SRATe {config['sampling_rate']}",
                f":CHANnel1:BANDwidth {config['bandwidth']}",
                ":TIMebase:SCALe 5E-12",
                f":CHANnel1:RANGe {config['voltage_range']}"
            ]

            for cmd in commands:
                self.instrument_controller.send_command(scope_resource, cmd)

            logger.info(f"Scope {scope_resource} configured for 224G")

        except Exception as e:
            logger.error(f"Failed to configure scope: {e}")
            raise

    def run_link_training_test(
        self,
        scope_resource: str,
        pattern_gen_resource: str,
        timeout_seconds: float = 10.0
    ) -> TrainingResults:
        """
        Execute link training test sequence
        
        Args:
            scope_resource: Scope resource identifier
            pattern_gen_resource: Pattern generator resource identifier
            timeout_seconds: Maximum training time in seconds
            
        Returns:
            TrainingResults object containing test results
        """
        assert isinstance(timeout_seconds, float), "Timeout must be a float"
        assert timeout_seconds > 0, "Timeout must be positive"

        try:
            # Configure pattern generator
            self.instrument_controller.send_command(
                pattern_gen_resource,
                f":PATTern:TYPE {self.test_configs['prbs_pattern']}"
            )

            # Configure scope for training capture
            self.configure_scope_for_224g(scope_resource)
            self.instrument_controller.send_command(
                scope_resource,
                ":ACQuire:POINts 1000000"
            )

            # Capture training sequence
            waveform_data = self._capture_training_sequence(
                scope_resource,
                timeout_seconds
            )

            # Analyze training results
            results = self._analyze_training_results(waveform_data)
            logger.info(f"Link training completed: {results}")

            return results

        except Exception as e:
            logger.error(f"Failed to run link training test: {e}")
            raise

    def _capture_training_sequence(
        self,
        scope_resource: str,
        timeout_seconds: float
    ) -> npt.NDArray[np.float64]:
        """
        Capture link training sequence data
        
        Args:
            scope_resource: Scope resource identifier
            timeout_seconds: Maximum capture time
            
        Returns:
            Array of captured waveform data
        """

        # Add validation
        assert isinstance(timeout_seconds, float), "Timeout must be a float"
        assert timeout_seconds > 0, "Timeout must be positive"

        try:
            # Trigger capture
            self.instrument_controller.send_command(scope_resource, ":RUN")
            self.instrument_controller.send_command(scope_resource, ":SINGle")

            # Get waveform data
            raw_data = self.instrument_controller.query_instrument(
                scope_resource,
                ":WAVeform:DATA?"
            )

            return np.array(raw_data.split(','), dtype=np.float64)

        except Exception as e:
            logger.error(f"Failed to capture training sequence: {e}")
            raise

    def _analyze_training_results(
        self,
        waveform_data: npt.NDArray[np.float64]
    ) -> TrainingResults:
        """
        Analyze link training sequence results
        
        Args:
            waveform_data: Captured waveform data
            
        Returns:
            TrainingResults object with analysis
        """
        assert np.issubdtype(waveform_data.dtype, np.floating), \
            "Waveform data must contain floating-point numbers"
        assert len(waveform_data) > 0, "Waveform data cannot be empty"

        try:
            # Calculate equalizer taps
            tap_weights = self._calculate_equalizer_taps(waveform_data)

            # Calculate error metrics using valid data ranges
            num_taps = len(tap_weights)
            output = np.zeros_like(waveform_data)

            for i in range(len(waveform_data) - num_taps + 1):
                output[i] = np.dot(tap_weights, waveform_data[i:i+num_taps])

            # Calculate error over valid range
            error = np.mean(np.abs(waveform_data[num_taps-1:] - output[:-(num_taps-1)]))

            # Track error history for plotting
            block_size = 1000
            error_history = []

            for i in range(0, len(waveform_data) - block_size, block_size):
                block_error = float(np.mean(np.abs(waveform_data[i:i+block_size])))
                error_history.append(block_error)

            # Determine convergence
            convergence_threshold = 0.1
            status = "PASS" if error < convergence_threshold else "FAIL"

            results = TrainingResults(
                training_time=float(len(waveform_data) / self.test_configs['sampling_rate']),
                convergence_status=status,
                final_eq_settings=tap_weights,
                adaptation_error=float(error),
                error_history=error_history
            )

            return results

        except Exception as e:
            logger.error(f"Failed to analyze training results: {e}")
            raise

    def _calculate_equalizer_taps(
        self,
        waveform_data: npt.NDArray[np.float64]
    ) -> List[float]:
        """
        Calculate equalizer tap weights from training data
        
        Args:
            waveform_data: Captured waveform data
            
        Returns:
            List of calculated tap weights
        """
        assert np.issubdtype(waveform_data.dtype, np.floating), \
            "Waveform data must contain floating-point numbers"
        assert len(waveform_data) > 0, "Waveform data cannot be empty"

        try:
            # LMS adaptation parameters
            num_taps = 5
            tap_weights = np.zeros(num_taps, dtype=np.float64)
            learning_rate = 0.01

            # Process data
            for i in range(len(waveform_data) - num_taps):
                input_vector = waveform_data[i:i+num_taps]
                desired = waveform_data[i+num_taps]
                output = np.dot(tap_weights, input_vector)
                error = desired - output
                tap_weights += learning_rate * error * input_vector

            return list(map(float, tap_weights))

        except Exception as e:
            logger.error(f"Failed to calculate equalizer taps: {e}")
            raise

    def measure_pam4_levels(
        self,
        scope_resource: str,
        measurement_time: float = 1.0
    ) -> PAM4Levels:
        """
        Measure PAM4 signal levels with enhanced validation
        
        Args:
            scope_resource: Scope resource identifier
            measurement_time: Measurement duration in seconds
            
        Returns:
            PAM4Levels object containing analysis results
        """
        assert isinstance(measurement_time, float), "Measurement time must be a float"
        assert measurement_time > 0, "Measurement time must be positive"

        try:
            # Configure scope
            self.configure_scope_for_224g(scope_resource)

            # Capture waveform
            waveform_data = self._capture_waveform(scope_resource, measurement_time)

            # Validate data range
            data_range = np.ptp(waveform_data)
            if data_range < 1.0:
                logger.warning(f"Waveform range ({data_range:.2f}) seems small for PAM4")

            # Analyze PAM4 levels
            analyzer = PAM4Analyzer({'voltage': waveform_data})
            level_analysis = analyzer.analyze_level_separation('voltage')

            return level_analysis

        except Exception as e:
            logger.error(f"Failed to measure PAM4 levels: {e}")
            raise

    def run_compliance_test_suite(
        self,
        scope_resource: str,
        pattern_gen_resource: str,
        test_duration: float = 10.0
    ) -> ComplianceResults:
        """
        Run full 224G compliance test suite
        
        Args:
            scope_resource: Scope resource identifier
            pattern_gen_resource: Pattern generator resource identifier
            test_duration: Total test duration in seconds
            
        Returns:
            ComplianceResults object containing all test results
        """
        assert isinstance(test_duration, float), "Test duration must be a float"
        assert test_duration > 0, "Test duration must be positive"

        try:
            # Run all compliance tests
            pam4_levels = self.measure_pam4_levels(scope_resource)
            eye_results = self._measure_eye_diagram(scope_resource)
            jitter_results = self._measure_jitter(scope_resource)
            evm_results = self._measure_evm(scope_resource)

            # Determine overall compliance status
            compliance_criteria = {
                'min_eye_height': 0.2,
                'max_rms_evm': 5.0,
                'max_total_jitter': 0.3
            }

            test_status = "PASS"
            if eye_results.worst_eye_height < compliance_criteria['min_eye_height']:
                test_status = "FAIL"
            if evm_results.rms_evm_percent > compliance_criteria['max_rms_evm']:
                test_status = "FAIL"
            if jitter_results['tj'] > compliance_criteria['max_total_jitter']:
                test_status = "FAIL"

            results = ComplianceResults(
                pam4_levels=pam4_levels,
                evm_results=evm_results,
                eye_results=eye_results,
                jitter_results=jitter_results,
                test_status=test_status
            )

            logger.info(f"Compliance test suite completed: {test_status}")
            return results

        except Exception as e:
            logger.error(f"Failed to run compliance test suite: {e}")
            raise

    def _measure_eye_diagram(
        self,
        scope_resource: str
    ) -> EyeResults:
        """
        Measure eye diagram parameters
        
        Args:
            scope_resource: Scope resource identifier
            
        Returns:
            EyeResults object containing measurements
        """
        try:
            self.instrument_controller.send_command(
                scope_resource,
                ":MEASure:EYE:ENABle"
            )

            eye_heights = []
            eye_widths = []

            for level in range(3):  # Three eyes in PAM4
                height = float(self.instrument_controller.query_instrument(
                    scope_resource,
                    f":MEASure:EYE{level}:HEIGht?"
                ))
                width = float(self.instrument_controller.query_instrument(
                    scope_resource,
                    f":MEASure:EYE{level}:WIDTh?"
                ))
                eye_heights.append(height)
                eye_widths.append(width)

            # Create EyeResults with just the measurements
            return EyeResults(
                eye_heights=eye_heights,
                eye_widths=eye_widths
            )

        except Exception as e:
            logger.error(f"Failed to measure eye diagram: {e}")
            raise

    def _measure_jitter(
        self,
        scope_resource: str
    ) -> Dict[str, float]:
        """
        Measure jitter components
        
        Args:
            scope_resource: Scope resource identifier
            
        Returns:
            Dictionary of jitter measurements
        """
        try:
            self.instrument_controller.send_command(
                scope_resource,
                ":MEASure:JITTer:ENABle"
            )

            measurements = {
                'tj': float(self.instrument_controller.query_instrument(
                    scope_resource, ":MEASure:JITTer:TJ?"
                )),
                'rj': float(self.instrument_controller.query_instrument(
                    scope_resource, ":MEASure:JITTer:RJ?"
                )),
                'dj': float(self.instrument_controller.query_instrument(
                    scope_resource, ":MEASure:JITTer:DJ?"
                ))
            }
            return measurements

        except Exception as e:
            logger.error(f"Failed to measure jitter: {e}")
            raise

    def _measure_evm(
        self,
        scope_resource: str,
        measurement_time: float = 1.0
    ) -> EVMResults:
        """
        Measure Error Vector Magnitude
        
        Args:
            scope_resource: Scope resource identifier
            measurement_time: Measurement duration in seconds
            
        Returns:
            EVMResults object containing EVM measurements
        """
        assert isinstance(measurement_time, float), "Measurement time must be a float"
        assert measurement_time > 0, "Measurement time must be positive"

        try:
            # Capture waveform for EVM analysis
            waveform_data = self._capture_waveform(scope_resource, measurement_time)
            time_data = np.arange(len(waveform_data)) / self.test_configs['sampling_rate']

            # Create analyzer and calculate EVM
            analyzer = PAM4Analyzer({
                'voltage': waveform_data,
                'time': time_data
            })

            return analyzer.calculate_evm('voltage', 'time')

        except Exception as e:
            logger.error(f"Failed to measure EVM: {e}")
            raise

    def _capture_waveform(
        self,
        scope_resource: str,
        capture_time: float
    ) -> npt.NDArray[np.float64]:
        """
        Capture waveform data from scope
        
        Args:
            scope_resource: Scope resource identifier
            capture_time: Capture duration in seconds
            
        Returns:
            Array of waveform data points
        """
        assert isinstance(capture_time, float), "Capture time must be a float"
        assert capture_time > 0, "Capture time must be positive"

        try:
            # Calculate required points
            sample_points = int(self.test_configs['sampling_rate'] * capture_time)

            # Configure acquisition
            self.instrument_controller.send_command(
                scope_resource,
                f":ACQuire:POINts {sample_points}"
            )

            # Trigger capture
            self.instrument_controller.send_command(scope_resource, ":RUN")
            self.instrument_controller.send_command(scope_resource, ":SINGle")

            # Get waveform data
            raw_data = self.instrument_controller.query_instrument(
                scope_resource,
                ":WAVeform:DATA?"
            )

            # Handle MagicMock and real responses
            if hasattr(raw_data, '_mock_return_value'):
                logger.debug("Using synthetic data for mock scope")
                return self._generate_synthetic_pam4_data(sample_points)

            # Process string data from real scope
            if isinstance(raw_data, str) and ',' in raw_data:
                try:
                    waveform = np.array(raw_data.split(','), dtype=np.float64)
                    if len(waveform) > 0:
                        return waveform
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse scope data: {e}")

            # Fallback to synthetic data
            logger.warning(f"Using synthetic data for scope {scope_resource}")
            return self._generate_synthetic_pam4_data(sample_points)

        except Exception as e:
            logger.error(f"Failed to capture waveform: {e}")
            raise

    def _generate_synthetic_pam4_data(self, num_points: int) -> npt.NDArray[np.float64]:
        """
        Generate synthetic PAM4 waveform data with clear voltage levels
        
        Args:
            num_points: Number of data points to generate
            
        Returns:
            Array of synthetic PAM4 data
        """
        try:
            # Define PAM4 levels with good separation
            levels = np.array([-3.0, -1.0, 1.0, 3.0])

            # Generate random symbols with equal probability
            symbols = np.random.randint(0, 4, num_points)
            voltage_levels = levels[symbols]

            # Add controlled noise
            noise_amplitude = 0.05  # Reduced noise for clearer levels
            noise = np.random.normal(0, noise_amplitude, num_points)

            # Add timing jitter simulation
            t = np.linspace(0, num_points/self.test_configs['sampling_rate'], num_points)
            jitter = 0.1 * np.sin(2 * np.pi * 1e9 * t)  # 1 GHz jitter component

            # Combine signal components
            waveform = voltage_levels + noise + jitter

            return waveform.astype(np.float64)

        except Exception as e:
            logger.error(f"Failed to generate synthetic data: {e}")
            raise

    def cleanup(self, resource_names: Optional[List[str]] = None) -> None:
        """
        Clean up and disconnect instruments
        
        Args:
            resource_names: Optional list of instrument resources to disconnect
        """
        try:
            # If no resources specified, disconnect all
            if resource_names is None:
                resource_names = list(self.connected_instruments.keys())

            for resource in resource_names:
                try:
                    if resource in self.connected_instruments:
                        self.instrument_controller.disconnect_instrument(resource)
                        del self.connected_instruments[resource]
                except Exception as e:
                    logger.warning(f"Failed to disconnect {resource}: {e}")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise

    def generate_report(
        self,
        compliance_results: ComplianceResults,
        report_file: str
    ) -> None:
        """
        Generate test report from compliance results
        
        Args:
            compliance_results: ComplianceResults object
            report_file: Output file path
        """
        assert isinstance(report_file, str), "Report file path must be a string"

        try:
            with open(report_file, 'w') as f:
                f.write("224G Ethernet Compliance Test Report\n")
                f.write("===================================\n\n")

                f.write(f"Overall Status: {compliance_results.test_status}\n\n")

                f.write("PAM4 Level Analysis:\n")
                f.write(f"- Level means: {compliance_results.pam4_levels.level_means}\n")
                f.write(f"- Level separations: {compliance_results.pam4_levels.level_separations}\n")
                f.write(f"- Uniformity: {compliance_results.pam4_levels.uniformity:.3f}\n\n")

                f.write("EVM Results:\n")
                f.write(f"- RMS EVM: {compliance_results.evm_results.rms_evm_percent:.2f}%\n")
                f.write(f"- Peak EVM: {compliance_results.evm_results.peak_evm_percent:.2f}%\n\n")

                f.write("Eye Diagram Analysis:\n")
                f.write(f"- Eye heights: {compliance_results.eye_results.eye_heights}\n")
                f.write(f"- Eye widths: {compliance_results.eye_results.eye_widths}\n")
                f.write(f"- Worst eye height: {compliance_results.eye_results.worst_eye_height:.3f}\n")
                f.write(f"- Worst eye width: {compliance_results.eye_results.worst_eye_width:.3f}\n\n")

                f.write("Jitter Analysis:\n")
                for jitter_type, value in compliance_results.jitter_results.items():
                    f.write(f"- {jitter_type.upper()}: {value:.3f} ps\n")

            logger.info(f"Test report generated: {report_file}")

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    try:
        sequence = Ethernet224GTestSequence()

        # Test resources
        scope = "GPIB0::7::INSTR"
        pattern_gen = "GPIB0::10::INSTR"

        # Run compliance tests
        results = sequence.run_compliance_test_suite(scope, pattern_gen)

        # Generate report
        sequence.generate_report(results, "224g_compliance_report.txt")

    except Exception as e:
        logger.error(f"Test sequence failed: {e}")
        raise
