"""
USB4 Pattern Generator Module

This module provides pattern generation capabilities specifically optimized for USB4 and
Thunderbolt 4 testing with support for training sequences, compliance patterns, and
stress testing patterns.

Features:
- USB4 v2.0 pattern generation
- Thunderbolt 4 certification patterns
- Link training sequence generation
- Multi-protocol tunneling patterns
- Error injection capabilities
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Union

import numpy as np
import numpy.typing as npt

from ..protocols.usb4.constants import USB4SignalMode, USB4Specs
from .controller import InstrumentController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USB4PatternType(Enum):
    """USB4 pattern types"""

    PRBS7 = auto()
    PRBS15 = auto()
    PRBS31 = auto()
    TRAINING_SEQUENCE_1 = auto()
    TRAINING_SEQUENCE_2 = auto()
    IDLE_PATTERN = auto()
    COMPLIANCE_PATTERN = auto()
    STRESS_PATTERN = auto()
    CUSTOM = auto()


class USB4OutputMode(Enum):
    """USB4 output modes"""

    DIFFERENTIAL = auto()
    SINGLE_ENDED = auto()
    DUAL_LANE = auto()


@dataclass
class USB4PatternConfig:
    """USB4 pattern generator configuration"""

    pattern_type: USB4PatternType
    signal_mode: USB4SignalMode
    bit_rate: float
    amplitude: float
    offset: float
    output_mode: USB4OutputMode
    enable_ssc: bool = True
    ssc_frequency: float = 31.5e3  # 31.5 kHz
    ssc_deviation: float = 0.5  # 0.5%

    def __post_init__(self) -> None:
        """Validate pattern configuration"""
        assert isinstance(
            self.pattern_type, USB4PatternType
        ), f"Pattern type must be USB4PatternType, got {type(self.pattern_type)}"
        assert isinstance(self.signal_mode, USB4SignalMode), f"Signal mode must be USB4SignalMode, got {type(self.signal_mode)}"
        assert isinstance(self.bit_rate, float), f"Bit rate must be float, got {type(self.bit_rate)}"
        assert isinstance(self.amplitude, float), f"Amplitude must be float, got {type(self.amplitude)}"
        assert isinstance(self.offset, float), f"Offset must be float, got {type(self.offset)}"
        assert isinstance(self.output_mode, USB4OutputMode), f"Output mode must be USB4OutputMode, got {type(self.output_mode)}"

        assert self.bit_rate > 0, f"Bit rate must be positive, got {self.bit_rate}"
        assert self.amplitude > 0, f"Amplitude must be positive, got {self.amplitude}"
        assert self.ssc_frequency > 0, f"SSC frequency must be positive, got {self.ssc_frequency}"
        assert 0 < self.ssc_deviation <= 1.0, f"SSC deviation must be between 0 and 1, got {self.ssc_deviation}"


@dataclass
class USB4PatternResult:
    """USB4 pattern generation result"""

    pattern_data: npt.NDArray[np.complex128]
    pattern_length: int
    actual_bit_rate: float
    pattern_type: USB4PatternType
    metadata: Dict[str, Union[str, float, bool]]

    def __post_init__(self) -> None:
        """Validate pattern result"""
        assert isinstance(self.pattern_data, np.ndarray), f"Pattern data must be numpy array, got {type(self.pattern_data)}"
        assert isinstance(self.pattern_length, int), f"Pattern length must be int, got {type(self.pattern_length)}"
        assert isinstance(self.actual_bit_rate, float), f"Actual bit rate must be float, got {type(self.actual_bit_rate)}"
        assert isinstance(
            self.pattern_type, USB4PatternType
        ), f"Pattern type must be USB4PatternType, got {type(self.pattern_type)}"
        assert isinstance(self.metadata, dict), f"Metadata must be dict, got {type(self.metadata)}"

        assert len(self.pattern_data) > 0, "Pattern data cannot be empty"
        assert self.pattern_length > 0, f"Pattern length must be positive, got {self.pattern_length}"
        assert self.actual_bit_rate > 0, f"Actual bit rate must be positive, got {self.actual_bit_rate}"


class USB4PatternGenerator:
    """Pattern generator for USB4 signal testing"""

    def __init__(self, resource_name: str) -> None:
        """
        Initialize USB4 pattern generator

        Args:
            resource_name: VISA resource identifier

        Raises:
            ValueError: If initialization fails
        """
        assert isinstance(resource_name, str), f"Resource name must be string, got {type(resource_name)}"

        self.resource_name = resource_name
        self.usb4_specs = USB4Specs()

        # Initialize instrument controller
        self.controller = InstrumentController()
        self.connected = False

        # Pattern generator specific settings
        self.output_mapping = {
            "lane0_p": 1,  # Lane 0 positive output
            "lane0_n": 2,  # Lane 0 negative output
            "lane1_p": 3,  # Lane 1 positive output
            "lane1_n": 4,  # Lane 1 negative output
        }

        # Pattern cache
        self.pattern_cache: Dict[str, USB4PatternResult] = {}

        logger.info(f"USB4 pattern generator initialized for {resource_name}")

    def connect(self) -> bool:
        """
        Connect to pattern generator

        Returns:
            True if connection successful

        Raises:
            ValueError: If connection fails
        """
        try:
            self.controller.connect_instrument(self.resource_name)

            # Verify instrument identity
            idn = self.controller.query_instrument(self.resource_name, "*IDN?")
            logger.info(f"Connected to: {idn.strip()}")

            # Initialize pattern generator for USB4
            self._initialize_generator()

            self.connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to pattern generator: {e}")
            raise ValueError(f"Pattern generator connection failed: {e}")

    def disconnect(self) -> None:
        """Disconnect from pattern generator"""
        try:
            if self.connected:
                # Turn off all outputs
                for output in range(1, 5):
                    self.controller.send_command(self.resource_name, f":OUTPut{output}:STATe OFF")

                self.controller.disconnect_instrument(self.resource_name)
                self.connected = False
                logger.info("Disconnected from pattern generator")
        except Exception as e:
            logger.error(f"Failed to disconnect from pattern generator: {e}")

    def _initialize_generator(self) -> None:
        """Initialize pattern generator for USB4"""
        try:
            # Reset instrument
            self.controller.send_command(self.resource_name, "*RST")
            time.sleep(2.0)  # Wait for reset

            # Set reference clock to internal
            self.controller.send_command(self.resource_name, ":ROSCillator:SOURce INTernal")

            # Configure outputs for differential signaling
            for output in range(1, 5):
                # Set output impedance to 50 ohms
                self.controller.send_command(self.resource_name, f":OUTPut{output}:IMPedance 50")

                # Set output format to NRZ
                self.controller.send_command(self.resource_name, f":OUTPut{output}:FORMat NRZ")

                # Enable output
                self.controller.send_command(self.resource_name, f":OUTPut{output}:STATe ON")

            # Configure differential pairs
            self._configure_differential_outputs()

            logger.info("Pattern generator initialized for USB4")

        except Exception as e:
            raise ValueError(f"Pattern generator initialization failed: {e}")

    def _configure_differential_outputs(self) -> None:
        """Configure differential output pairs for USB4 lanes"""
        try:
            # Configure Lane 0 differential pair (outputs 1 and 2)
            self.controller.send_command(self.resource_name, ":OUTPut1:COMPlement:SOURce OUTPut2")
            self.controller.send_command(self.resource_name, ":OUTPut2:COMPlement:SOURce OUTPut1")
            self.controller.send_command(self.resource_name, ":OUTPut1:COMPlement:STATe ON")
            self.controller.send_command(self.resource_name, ":OUTPut2:COMPlement:STATe ON")

            # Configure Lane 1 differential pair (outputs 3 and 4)
            self.controller.send_command(self.resource_name, ":OUTPut3:COMPlement:SOURce OUTPut4")
            self.controller.send_command(self.resource_name, ":OUTPut4:COMPlement:SOURce OUTPut3")
            self.controller.send_command(self.resource_name, ":OUTPut3:COMPlement:STATe ON")
            self.controller.send_command(self.resource_name, ":OUTPut4:COMPlement:STATe ON")

        except Exception as e:
            raise ValueError(f"Differential output configuration failed: {e}")

    def generate_usb4_pattern(self, config: USB4PatternConfig, duration: Optional[float] = None) -> USB4PatternResult:
        """
        Generate USB4 pattern

        Args:
            config: Pattern configuration
            duration: Optional pattern duration in seconds

        Returns:
            Pattern generation result

        Raises:
            ValueError: If pattern generation fails
        """
        assert isinstance(config, USB4PatternConfig), f"Config must be USB4PatternConfig, got {type(config)}"

        if not self.connected:
            raise ValueError("Pattern generator not connected")

        try:
            # Check cache first
            cache_key = self._generate_cache_key(config)
            if cache_key in self.pattern_cache:
                logger.info(f"Using cached pattern for {config.pattern_type.name}")
                return self.pattern_cache[cache_key]

            # Configure pattern generator for this pattern
            self._configure_for_pattern(config)

            # Generate pattern data
            pattern_result = self._generate_pattern_data(config, duration)

            # Load pattern to instrument
            self._load_pattern_to_instrument(pattern_result, config)

            # Cache the result
            self.pattern_cache[cache_key] = pattern_result

            logger.info(f"Generated {config.pattern_type.name} pattern with {pattern_result.pattern_length} bits")
            return pattern_result

        except Exception as e:
            logger.error(f"USB4 pattern generation failed: {e}")
            raise ValueError(f"Pattern generation failed: {e}")

    def _generate_cache_key(self, config: USB4PatternConfig) -> str:
        """Generate cache key for pattern configuration"""
        return f"{config.pattern_type.name}_{config.signal_mode.name}_{config.bit_rate}_{config.amplitude}_{config.enable_ssc}"

    def _configure_for_pattern(self, config: USB4PatternConfig) -> None:
        """Configure pattern generator for specific pattern"""
        try:
            # Set bit rate
            self.controller.send_command(self.resource_name, f":PULSe:PERiod {1.0/config.bit_rate}")

            # Configure amplitude and offset for all outputs
            for output in range(1, 5):
                self.controller.send_command(self.resource_name, f":OUTPut{output}:AMPLitude {config.amplitude}")
                self.controller.send_command(self.resource_name, f":OUTPut{output}:OFFSet {config.offset}")

            # Configure SSC if enabled
            if config.enable_ssc:
                self._configure_ssc(config)

            # Configure for specific signal mode
            if config.signal_mode == USB4SignalMode.GEN2X2:
                # Gen 2 configuration (20 Gbps per lane)
                self.controller.send_command(self.resource_name, ":PATTern:MODE NORMal")

            elif config.signal_mode == USB4SignalMode.GEN3X2:
                # Gen 3 configuration (20 Gbps per lane, enhanced signaling)
                self.controller.send_command(self.resource_name, ":PATTern:MODE ENHanced")

            elif config.signal_mode == USB4SignalMode.ASYMMETRIC:
                # Asymmetric configuration
                self.controller.send_command(self.resource_name, ":PATTern:MODE ASYMmetric")

        except Exception as e:
            raise ValueError(f"Pattern configuration failed: {e}")

    def _configure_ssc(self, config: USB4PatternConfig) -> None:
        """Configure spread spectrum clocking"""
        try:
            # Enable SSC
            self.controller.send_command(self.resource_name, ":SSC:STATe ON")

            # Set SSC frequency (30-33 kHz for USB4)
            self.controller.send_command(self.resource_name, f":SSC:FREQuency {config.ssc_frequency}")

            # Set SSC deviation (0.5% for USB4)
            deviation_ppm = config.ssc_deviation * 10000  # Convert to ppm
            self.controller.send_command(self.resource_name, f":SSC:DEViation {deviation_ppm}")

            # Set SSC profile to down-spread
            self.controller.send_command(self.resource_name, ":SSC:PROFile DOWN")

        except Exception as e:
            logger.warning(f"SSC configuration failed: {e}")

    def _generate_pattern_data(self, config: USB4PatternConfig, duration: Optional[float]) -> USB4PatternResult:
        """Generate pattern data based on configuration"""
        try:
            if config.pattern_type == USB4PatternType.PRBS7:
                pattern_data = self._generate_prbs_pattern(7, config.bit_rate, duration)
            elif config.pattern_type == USB4PatternType.PRBS15:
                pattern_data = self._generate_prbs_pattern(15, config.bit_rate, duration)
            elif config.pattern_type == USB4PatternType.PRBS31:
                pattern_data = self._generate_prbs_pattern(31, config.bit_rate, duration)
            elif config.pattern_type == USB4PatternType.TRAINING_SEQUENCE_1:
                pattern_data = self._generate_training_sequence_1(config.bit_rate, duration)
            elif config.pattern_type == USB4PatternType.TRAINING_SEQUENCE_2:
                pattern_data = self._generate_training_sequence_2(config.bit_rate, duration)
            elif config.pattern_type == USB4PatternType.IDLE_PATTERN:
                pattern_data = self._generate_idle_pattern(config.bit_rate, duration)
            elif config.pattern_type == USB4PatternType.COMPLIANCE_PATTERN:
                pattern_data = self._generate_compliance_pattern(config.bit_rate, duration)
            elif config.pattern_type == USB4PatternType.STRESS_PATTERN:
                pattern_data = self._generate_stress_pattern(config.bit_rate, duration)
            else:
                raise ValueError(f"Unsupported pattern type: {config.pattern_type}")

            # Create metadata
            metadata = {
                "pattern_type": config.pattern_type.name,
                "signal_mode": config.signal_mode.name,
                "bit_rate": config.bit_rate,
                "amplitude": config.amplitude,
                "offset": config.offset,
                "output_mode": config.output_mode.name,
                "enable_ssc": config.enable_ssc,
                "generation_time": time.time(),
            }

            if config.enable_ssc:
                metadata["ssc_frequency"] = config.ssc_frequency
                metadata["ssc_deviation"] = config.ssc_deviation

            return USB4PatternResult(
                pattern_data=pattern_data,
                pattern_length=len(pattern_data),
                actual_bit_rate=config.bit_rate,
                pattern_type=config.pattern_type,
                metadata=metadata,
            )

        except Exception as e:
            raise ValueError(f"Pattern data generation failed: {e}")

    def _generate_prbs_pattern(self, order: int, bit_rate: float, duration: Optional[float]) -> npt.NDArray[np.complex128]:
        """Generate PRBS pattern"""
        try:
            # Calculate pattern length
            if duration is not None:
                pattern_length = int(bit_rate * duration)
            else:
                pattern_length = (2**order) - 1  # Full PRBS sequence

            # Generate PRBS sequence
            if order == 7:
                # PRBS7: x^7 + x^6 + 1
                taps = [7, 6]
            elif order == 15:
                # PRBS15: x^15 + x^14 + 1
                taps = [15, 14]
            elif order == 31:
                # PRBS31: x^31 + x^28 + 1
                taps = [31, 28]
            else:
                raise ValueError(f"Unsupported PRBS order: {order}")

            # Initialize shift register
            shift_register = np.ones(order, dtype=int)
            pattern = []

            for _ in range(pattern_length):
                # Output current bit
                output_bit = shift_register[-1]
                pattern.append(output_bit)

                # Calculate feedback
                feedback = 0
                for tap in taps:
                    feedback ^= shift_register[tap - 1]

                # Shift register
                shift_register[1:] = shift_register[:-1]
                shift_register[0] = feedback

            # Convert to complex array (NRZ encoding: 0 -> -1, 1 -> +1)
            pattern_array = np.array(pattern, dtype=int)
            pattern_array = 2 * pattern_array - 1  # Convert 0,1 to -1,+1

            return pattern_array.astype(np.complex128)

        except Exception as e:
            raise ValueError(f"PRBS pattern generation failed: {e}")

    def _generate_training_sequence_1(self, bit_rate: float, duration: Optional[float]) -> npt.NDArray[np.complex128]:
        """Generate USB4 Training Sequence 1 (TS1)"""
        try:
            # USB4 TS1 pattern: alternating pattern for link training
            base_pattern = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.complex128)

            if duration is not None:
                pattern_length = int(bit_rate * duration)
                repeats = pattern_length // len(base_pattern) + 1
                pattern = np.tile(base_pattern, repeats)[:pattern_length]
            else:
                pattern = base_pattern

            return pattern

        except Exception as e:
            raise ValueError(f"TS1 pattern generation failed: {e}")

    def _generate_training_sequence_2(self, bit_rate: float, duration: Optional[float]) -> npt.NDArray[np.complex128]:
        """Generate USB4 Training Sequence 2 (TS2)"""
        try:
            # USB4 TS2 pattern: different pattern for link training
            base_pattern = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1], dtype=np.complex128)

            if duration is not None:
                pattern_length = int(bit_rate * duration)
                repeats = pattern_length // len(base_pattern) + 1
                pattern = np.tile(base_pattern, repeats)[:pattern_length]
            else:
                pattern = base_pattern

            return pattern

        except Exception as e:
            raise ValueError(f"TS2 pattern generation failed: {e}")

    def _generate_idle_pattern(self, bit_rate: float, duration: Optional[float]) -> npt.NDArray[np.complex128]:
        """Generate USB4 idle pattern"""
        try:
            # USB4 idle pattern: all zeros (electrical idle)
            if duration is not None:
                pattern_length = int(bit_rate * duration)
            else:
                pattern_length = 1000  # Default length

            pattern = np.zeros(pattern_length, dtype=np.complex128)

            return pattern

        except Exception as e:
            raise ValueError(f"Idle pattern generation failed: {e}")

    def _generate_compliance_pattern(self, bit_rate: float, duration: Optional[float]) -> npt.NDArray[np.complex128]:
        """Generate USB4 compliance test pattern"""
        try:
            # Compliance pattern: mix of PRBS and specific sequences
            prbs_part = self._generate_prbs_pattern(31, bit_rate, duration)

            # Add compliance-specific sequences
            compliance_sequence = np.array([1, -1, 1, 1, -1, -1, 1, -1], dtype=np.complex128)

            if duration is not None:
                pattern_length = int(bit_rate * duration)
                # Interleave PRBS with compliance sequences
                pattern = np.zeros(pattern_length, dtype=np.complex128)

                prbs_len = min(len(prbs_part), pattern_length)
                pattern[:prbs_len] = prbs_part[:prbs_len]

                # Insert compliance sequences every 100 bits
                for i in range(0, pattern_length - len(compliance_sequence), 100):
                    end_idx = min(i + len(compliance_sequence), pattern_length)
                    pattern[i:end_idx] = compliance_sequence[: end_idx - i]
            else:
                pattern = prbs_part

            return pattern

        except Exception as e:
            raise ValueError(f"Compliance pattern generation failed: {e}")

    def _generate_stress_pattern(self, bit_rate: float, duration: Optional[float]) -> npt.NDArray[np.complex128]:
        """Generate USB4 stress test pattern"""
        try:
            # Stress pattern: worst-case transitions and patterns
            if duration is not None:
                pattern_length = int(bit_rate * duration)
            else:
                pattern_length = 10000  # Default length

            # Create pattern with maximum transitions
            pattern = np.zeros(pattern_length, dtype=np.complex128)

            # Alternating pattern for maximum transitions
            pattern[::2] = 1
            pattern[1::2] = -1

            # Add some longer runs to stress clock recovery
            for i in range(0, pattern_length, 200):
                end_idx = min(i + 20, pattern_length)
                pattern[i:end_idx] = 1  # Long run of 1s

                if end_idx < pattern_length:
                    end_idx2 = min(end_idx + 20, pattern_length)
                    pattern[end_idx:end_idx2] = -1  # Long run of -1s

            return pattern

        except Exception as e:
            raise ValueError(f"Stress pattern generation failed: {e}")

    def _load_pattern_to_instrument(self, pattern_result: USB4PatternResult, config: USB4PatternConfig) -> None:
        """Load generated pattern to instrument memory"""
        try:
            # Convert pattern to instrument format
            pattern_data = pattern_result.pattern_data

            # For simplicity, we'll use built-in patterns when possible
            if config.pattern_type == USB4PatternType.PRBS7:
                self.controller.send_command(self.resource_name, ":PATTern:TYPE PRBS7")
            elif config.pattern_type == USB4PatternType.PRBS15:
                self.controller.send_command(self.resource_name, ":PATTern:TYPE PRBS15")
            elif config.pattern_type == USB4PatternType.PRBS31:
                self.controller.send_command(self.resource_name, ":PATTern:TYPE PRBS31")
            else:
                # Load custom pattern
                self.controller.send_command(self.resource_name, ":PATTern:TYPE USER")

                # Convert pattern to binary string
                binary_pattern = ""
                for bit in pattern_data:
                    binary_pattern += "1" if bit.real > 0 else "0"

                # Load pattern (simplified - real implementation would handle binary data)
                self.controller.send_command(self.resource_name, f":PATTern:DATA '{binary_pattern}'")

            # Configure output routing based on output mode
            if config.output_mode == USB4OutputMode.DUAL_LANE:
                # Route pattern to both lanes
                self.controller.send_command(self.resource_name, ":OUTPut1:PATTern:SOURce PATTern1")
                self.controller.send_command(self.resource_name, ":OUTPut2:PATTern:SOURce PATTern1")
                self.controller.send_command(self.resource_name, ":OUTPut3:PATTern:SOURce PATTern1")
                self.controller.send_command(self.resource_name, ":OUTPut4:PATTern:SOURce PATTern1")
            elif config.output_mode == USB4OutputMode.DIFFERENTIAL:
                # Route to differential pair
                self.controller.send_command(self.resource_name, ":OUTPut1:PATTern:SOURce PATTern1")
                self.controller.send_command(self.resource_name, ":OUTPut2:PATTern:SOURce PATTern1")

        except Exception as e:
            raise ValueError(f"Pattern loading failed: {e}")

    def start_pattern_generation(self) -> bool:
        """
        Start pattern generation

        Returns:
            True if start successful

        Raises:
            ValueError: If start fails
        """
        if not self.connected:
            raise ValueError("Pattern generator not connected")

        try:
            # Start pattern generation
            self.controller.send_command(self.resource_name, ":OUTPut:STATe ON")

            # Verify outputs are active
            for output in range(1, 5):
                status = self.controller.query_instrument(self.resource_name, f":OUTPut{output}:STATe?")
                if "OFF" in status.upper():
                    logger.warning(f"Output {output} failed to start")

            logger.info("Pattern generation started")
            return True

        except Exception as e:
            logger.error(f"Failed to start pattern generation: {e}")
            raise ValueError(f"Pattern generation start failed: {e}")

    def stop_pattern_generation(self) -> bool:
        """
        Stop pattern generation

        Returns:
            True if stop successful

        Raises:
            ValueError: If stop fails
        """
        if not self.connected:
            raise ValueError("Pattern generator not connected")

        try:
            # Stop pattern generation
            self.controller.send_command(self.resource_name, ":OUTPut:STATe OFF")

            logger.info("Pattern generation stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop pattern generation: {e}")
            raise ValueError(f"Pattern generation stop failed: {e}")

    def inject_error(self, error_type: str, error_rate: float, duration: float) -> bool:
        """
        Inject errors into pattern for testing

        Args:
            error_type: Type of error to inject ('bit', 'symbol', 'burst')
            error_rate: Error rate (errors per second)
            duration: Duration of error injection in seconds

        Returns:
            True if error injection successful

        Raises:
            ValueError: If error injection fails
        """
        assert isinstance(error_type, str), f"Error type must be string, got {type(error_type)}"
        assert isinstance(error_rate, float), f"Error rate must be float, got {type(error_rate)}"
        assert isinstance(duration, float), f"Duration must be float, got {type(duration)}"
        assert error_rate >= 0, f"Error rate must be non-negative, got {error_rate}"
        assert duration > 0, f"Duration must be positive, got {duration}"

        if not self.connected:
            raise ValueError("Pattern generator not connected")

        try:
            # Configure error injection
            if error_type.lower() == "bit":
                self.controller.send_command(self.resource_name, ":ERRor:TYPE BIT")
            elif error_type.lower() == "symbol":
                self.controller.send_command(self.resource_name, ":ERRor:TYPE SYMBol")
            elif error_type.lower() == "burst":
                self.controller.send_command(self.resource_name, ":ERRor:TYPE BURSt")
            else:
                raise ValueError(f"Unknown error type: {error_type}")

            # Set error rate
            self.controller.send_command(self.resource_name, f":ERRor:RATE {error_rate}")

            # Enable error injection
            self.controller.send_command(self.resource_name, ":ERRor:STATe ON")

            # Wait for specified duration
            time.sleep(duration)

            # Disable error injection
            self.controller.send_command(self.resource_name, ":ERRor:STATe OFF")

            logger.info(f"Injected {error_type} errors at rate {error_rate} for {duration} seconds")
            return True

        except Exception as e:
            logger.error(f"Error injection failed: {e}")
            raise ValueError(f"Error injection failed: {e}")

    def get_generator_status(self) -> Dict[str, Union[str, float, bool]]:
        """
        Get pattern generator status

        Returns:
            Dictionary with generator status information

        Raises:
            ValueError: If status query fails
        """
        if not self.connected:
            raise ValueError("Pattern generator not connected")

        try:
            status = {}

            # Output status
            for output in range(1, 5):
                output_state = self.controller.query_instrument(self.resource_name, f":OUTPut{output}:STATe?")
                status[f"output_{output}_enabled"] = "ON" in output_state.upper()

                amplitude = self.controller.query_instrument(self.resource_name, f":OUTPut{output}:AMPLitude?")
                status[f"output_{output}_amplitude"] = float(amplitude)

                offset = self.controller.query_instrument(self.resource_name, f":OUTPut{output}:OFFSet?")
                status[f"output_{output}_offset"] = float(offset)

            # Pattern status
            pattern_type = self.controller.query_instrument(self.resource_name, ":PATTern:TYPE?")
            status["pattern_type"] = pattern_type.strip()

            # Clock status
            clock_freq = self.controller.query_instrument(self.resource_name, ":PULSe:FREQuency?")
            status["clock_frequency"] = float(clock_freq)

            # SSC status
            ssc_state = self.controller.query_instrument(self.resource_name, ":SSC:STATe?")
            status["ssc_enabled"] = "ON" in ssc_state.upper()

            if status["ssc_enabled"]:
                ssc_freq = self.controller.query_instrument(self.resource_name, ":SSC:FREQuency?")
                status["ssc_frequency"] = float(ssc_freq)

                ssc_dev = self.controller.query_instrument(self.resource_name, ":SSC:DEViation?")
                status["ssc_deviation"] = float(ssc_dev)

            return status

        except Exception as e:
            logger.error(f"Generator status query failed: {e}")
            raise ValueError(f"Status query failed: {e}")


__all__ = [
    "USB4PatternType",
    "USB4OutputMode",
    "USB4PatternConfig",
    "USB4PatternResult",
    "USB4PatternGenerator",
]
