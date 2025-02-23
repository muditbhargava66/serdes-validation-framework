# src/serdes_validation_framework/instrument_control/scope_224g.py

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import numpy.typing as npt

from .mock_controller import InstrumentMode, get_instrument_controller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScopeConfig:
    """Data class for scope configuration parameters"""
    sampling_rate: float  # In samples/second
    bandwidth: float      # In Hz
    timebase: float      # In seconds/division
    voltage_range: float # In volts

    def __post_init__(self) -> None:
        """Validate configuration parameters"""
        assert isinstance(self.sampling_rate, float), "Sampling rate must be a float"
        assert isinstance(self.bandwidth, float), "Bandwidth must be a float"
        assert isinstance(self.timebase, float), "Timebase must be a float"
        assert isinstance(self.voltage_range, float), "Voltage range must be a float"

        assert self.sampling_rate > 0, "Sampling rate must be positive"
        assert self.bandwidth > 0, "Bandwidth must be positive"
        assert self.timebase > 0, "Timebase must be positive"
        assert self.voltage_range > 0, "Voltage range must be positive"

@dataclass
class WaveformData:
    """Data class for captured waveform data"""
    time: npt.NDArray[np.float64]
    voltage: npt.NDArray[np.float64]
    sample_rate: float
    time_scale: float

class HighBandwidthScope:
    """Controller for high-bandwidth oscilloscopes supporting 224G measurements"""

    def __init__(
        self,
        resource_name: str,
        controller = None
    ) -> None:
        """
        Initialize scope controller
        
        Args:
            resource_name: VISA resource name for the scope
            controller: Optional existing instrument controller
        """
        self.controller = controller or get_instrument_controller()
        self.resource_name = resource_name
        self.default_config = ScopeConfig(
            sampling_rate=256e9,  # 256 GSa/s
            bandwidth=120e9,      # 120 GHz
            timebase=5e-12,       # 5 ps/div
            voltage_range=0.8     # 0.8V
        )

        try:
            self.controller.connect_instrument(resource_name)
            logger.info(f"Connected to scope at {resource_name}")

            # If we're in mock mode, configure mock responses for common scope queries
            if getattr(self.controller, 'mode', None) == InstrumentMode.MOCK:
                self._configure_mock_responses()

        except Exception as e:
            logger.error(f"Failed to connect to scope: {e}")
            raise

    def _configure_mock_responses(self) -> None:
        """Configure mock responses for common scope queries"""
        try:
            self.controller.add_mock_response(
                ':WAVeform:XINCrement?',
                lambda: f"{1/self.default_config.sampling_rate}"
            )
            self.controller.add_mock_response(
                ':WAVeform:XORigin?',
                '0.0'
            )
            self.controller.add_mock_response(
                ':MEASure:EYE:HEIGht?',
                lambda: f"{np.random.normal(0.4, 0.05):.6f}"
            )
            self.controller.add_mock_response(
                ':MEASure:EYE:WIDTh?',
                lambda: f"{np.random.normal(0.6, 0.05):.6f}"
            )
            self.controller.add_mock_response(
                ':MEASure:EYE:JITTer?',
                lambda: f"{np.random.normal(1e-12, 1e-13):.3e}"
            )
            logger.debug("Mock responses configured for scope")
        except Exception as e:
            logger.error(f"Failed to configure mock responses: {e}")
            raise

    def configure_for_224g(
        self,
        config: Optional[ScopeConfig] = None
    ) -> None:
        """
        Configure scope settings for 224G PAM4 capture
        
        Args:
            config: Optional custom configuration, uses default if None
        """
        config = config or self.default_config

        try:
            commands = [
                f":ACQuire:SRATe {config.sampling_rate:.6e}",
                f":CHANnel1:BANDwidth {config.bandwidth:.6e}",
                f":TIMebase:SCALe {config.timebase:.6e}",
                f":CHANnel1:RANGe {config.voltage_range:.6e}",
                ":ACQuire:MODE HRES",
                ":TRIGger:MODE AUTO",
                ":WAVeform:FORMat REAL"
            ]

            for cmd in commands:
                self.controller.send_command(self.resource_name, cmd)

            logger.info(f"Scope configured for 224G with settings: {config}")

        except Exception as e:
            logger.error(f"Failed to configure scope: {e}")
            raise

    def capture_eye_diagram(
        self,
        duration_seconds: float = 1.0,
        num_ui: int = 1000
    ) -> Dict[str, Union[WaveformData, float]]:
        """
        Capture eye diagram with optimal settings for 224G
        
        Args:
            duration_seconds: Capture duration in seconds
            num_ui: Number of Unit Intervals to capture
            
        Returns:
            Dictionary containing waveform data and measurements
        """
        assert isinstance(duration_seconds, float), "Duration must be a float"
        assert isinstance(num_ui, int), "num_ui must be an integer"
        assert duration_seconds > 0, "Duration must be positive"
        assert num_ui > 0, "num_ui must be positive"

        try:
            # Configure acquisition
            sample_points = int(self.default_config.sampling_rate * duration_seconds)
            self.controller.send_command(
                self.resource_name,
                f":ACQuire:POINts {sample_points}"
            )

            # Trigger acquisition
            self.controller.send_command(self.resource_name, ":RUN")
            self.controller.send_command(self.resource_name, ":SINGle")

            # Get waveform data
            raw_data = self._get_waveform_data()

            # Process data
            waveform = self._process_waveform(raw_data)

            # Calculate eye parameters
            eye_params = self._measure_eye_parameters(waveform)

            results = {
                'waveform': waveform,
                'eye_height': float(eye_params['height']),
                'eye_width': float(eye_params['width']),
                'jitter': float(eye_params['jitter'])
            }

            logger.info("Eye diagram capture completed")
            return results

        except Exception as e:
            logger.error(f"Failed to capture eye diagram: {e}")
            raise

    def measure_jitter(self) -> Dict[str, float]:
        """
        Measure various jitter components
        
        Returns:
            Dictionary containing jitter measurements
        """
        try:
            # Enable jitter analysis
            self.controller.send_command(
                self.resource_name,
                ":MEASure:JITTer:ENABle"
            )

            # Measure different jitter components
            measurements = {
                'tj': self._query_float(":MEASure:JITTer:TJ?"),
                'rj': self._query_float(":MEASure:JITTer:RJ?"),
                'dj': self._query_float(":MEASure:JITTer:DJ?"),
                'pj': self._query_float(":MEASure:JITTer:PJ?")
            }

            logger.info(f"Jitter measurements completed: {measurements}")
            return measurements

        except Exception as e:
            logger.error(f"Failed to measure jitter: {e}")
            raise

    def _get_waveform_data(self) -> npt.NDArray[np.float64]:
        """
        Get raw waveform data from scope
        
        Returns:
            Array of waveform data points
        """
        try:
            raw_data = self.controller.query_instrument(
                self.resource_name,
                ":WAVeform:DATA?"
            )
            return np.array(raw_data.split(','), dtype=np.float64)

        except Exception as e:
            logger.error(f"Failed to get waveform data: {e}")
            raise

    def _process_waveform(
        self,
        raw_data: npt.NDArray[np.float64]
    ) -> WaveformData:
        """
        Process raw waveform data into time and voltage arrays
        
        Args:
            raw_data: Raw data from scope
            
        Returns:
            WaveformData object with processed arrays
        """
        assert np.issubdtype(raw_data.dtype, np.floating), \
            "Raw data must contain floating-point numbers"

        try:
            # Get time base information
            x_increment = float(self._query_float(":WAVeform:XINCrement?"))
            x_origin = float(self._query_float(":WAVeform:XORigin?"))

            # Create time array
            time_array = x_origin + np.arange(len(raw_data)) * x_increment

            return WaveformData(
                time=time_array,
                voltage=raw_data,
                sample_rate=float(self.default_config.sampling_rate),
                time_scale=float(self.default_config.timebase)
            )

        except Exception as e:
            logger.error(f"Failed to process waveform: {e}")
            raise

    def _measure_eye_parameters(
        self,
        waveform: WaveformData
    ) -> Dict[str, float]:
        """
        Calculate eye diagram parameters
        
        Args:
            waveform: Processed waveform data
            
        Returns:
            Dictionary containing eye measurements
        """
        try:
            # Enable eye measurements
            self.controller.send_command(
                self.resource_name,
                ":MEASure:EYE:ENABle"
            )

            # Get measurements
            measurements = {
                'height': self._query_float(":MEASure:EYE:HEIGht?"),
                'width': self._query_float(":MEASure:EYE:WIDTh?"),
                'jitter': self._query_float(":MEASure:EYE:JITTer?")
            }

            return measurements

        except Exception as e:
            logger.error(f"Failed to measure eye parameters: {e}")
            raise

    def _query_float(self, query: str) -> float:
        """
        Query scope and convert response to float
        
        Args:
            query: SCPI query string
            
        Returns:
            Float value from scope
        """
        try:
            response = self.controller.query_instrument(self.resource_name, query)
            return float(response.strip())

        except Exception as e:
            logger.error(f"Failed to query float value: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up scope connection"""
        try:
            self.controller.disconnect_instrument(self.resource_name)
            logger.info("Scope disconnected successfully")
        except Exception as e:
            logger.error(f"Failed to disconnect scope: {e}")
            raise
