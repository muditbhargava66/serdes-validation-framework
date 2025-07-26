"""
PCIe Dual Mode Control Module

This module provides functionality for controlling and switching between
NRZ and PAM4 signaling modes for PCIe with comprehensive type validation.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

# Import PCIeConfig from the correct location
try:
    from ...instrument_control.pcie_analyzer import PCIeConfig
except ImportError:
    # Create a mock PCIeConfig if not available
    @dataclass
    class PCIeConfig:
        mode: "SignalMode"
        sample_rate: float
        bandwidth: float = 50e9
        voltage_range: float = 2.0
        link_speed: float = 32e9
        lane_count: int = 1


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalMode(Enum):
    """PCIe signaling modes"""

    NRZ = auto()
    PAM4 = auto()


class SwitchResult(Enum):
    """Mode switch result status"""

    SUCCESS = auto()
    FAILURE = auto()
    NOT_SUPPORTED = auto()


@dataclass
class ModeConfig:
    """
    Mode configuration parameters with validation
    """

    mode: SignalMode
    sample_rate: float
    bandwidth: float
    eye_height_threshold: float
    equalization_taps: int

    def __post_init__(self) -> None:
        """
        Validate configuration parameters

        Raises:
            AssertionError: If parameters are invalid
        """
        # Type validation
        assert isinstance(self.mode, SignalMode), f"Mode must be SignalMode, got {type(self.mode)}"
        assert isinstance(self.sample_rate, float), f"Sample rate must be float, got {type(self.sample_rate)}"
        assert isinstance(self.bandwidth, float), f"Bandwidth must be float, got {type(self.bandwidth)}"
        assert isinstance(
            self.eye_height_threshold, float
        ), f"Eye height threshold must be float, got {type(self.eye_height_threshold)}"
        assert isinstance(self.equalization_taps, int), f"Equalization taps must be integer, got {type(self.equalization_taps)}"

        # Value validation
        assert self.sample_rate > 0, f"Sample rate must be positive, got {self.sample_rate}"
        assert self.bandwidth > 0, f"Bandwidth must be positive, got {self.bandwidth}"
        assert (
            0 <= self.eye_height_threshold <= 1
        ), f"Eye height threshold must be between 0 and 1, got {self.eye_height_threshold}"
        assert self.equalization_taps > 0, f"Equalization taps must be positive, got {self.equalization_taps}"


@dataclass
class SwitchStatus:
    """
    Mode switch status with validation
    """

    success: bool
    current_mode: SignalMode
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Validate status parameters

        Raises:
            AssertionError: If parameters are invalid
        """
        # Type validation
        assert isinstance(self.success, bool), f"Success must be boolean, got {type(self.success)}"
        assert isinstance(self.current_mode, SignalMode), f"Current mode must be SignalMode, got {type(self.current_mode)}"
        if self.error_message is not None:
            assert isinstance(self.error_message, str), f"Error message must be string, got {type(self.error_message)}"


class ModeController:
    """
    PCIe dual-mode controller with type safety
    """

    def __init__(self, default_mode: SignalMode = SignalMode.NRZ) -> None:
        """
        Initialize mode controller

        Args:
            default_mode: Initial signal mode

        Raises:
            AssertionError: If default mode is invalid
        """
        # Validate input
        assert isinstance(default_mode, SignalMode), f"Default mode must be SignalMode, got {type(default_mode)}"

        self.current_mode = default_mode
        self.configs: Dict[SignalMode, ModeConfig] = {}
        logger.info(f"Mode controller initialized in {default_mode.name} mode")

    def register_mode_config(self, config: ModeConfig) -> None:
        """
        Register configuration for a signal mode

        Args:
            config: Mode configuration

        Raises:
            AssertionError: If configuration is invalid
        """
        # Validate input
        assert isinstance(config, ModeConfig), f"Config must be ModeConfig, got {type(config)}"

        self.configs[config.mode] = config
        logger.info(f"Registered configuration for {config.mode.name} mode")

    def switch_mode(self, target_mode: SignalMode, data: Optional[Dict[str, npt.NDArray[np.float64]]] = None) -> SwitchStatus:
        """
        Switch between signal modes

        Args:
            target_mode: Target signal mode
            data: Optional signal data for validation

        Returns:
            Switch status result

        Raises:
            AssertionError: If parameters are invalid
        """
        # Validate input
        assert isinstance(target_mode, SignalMode), f"Target mode must be SignalMode, got {type(target_mode)}"
        if data is not None:
            assert isinstance(data, dict), f"Data must be dictionary, got {type(data)}"
            if "voltage" in data:
                assert isinstance(data["voltage"], np.ndarray), f"Voltage data must be numpy array, got {type(data['voltage'])}"
                assert np.issubdtype(
                    data["voltage"].dtype, np.floating
                ), f"Voltage data must be floating-point, got {data['voltage'].dtype}"

        # Check if already in target mode
        if self.current_mode == target_mode:
            return SwitchStatus(success=True, current_mode=self.current_mode, error_message=None)

        # Check if configuration exists
        if target_mode not in self.configs:
            return SwitchStatus(
                success=False,
                current_mode=self.current_mode,
                error_message=f"No configuration registered for {target_mode.name} mode",
            )

        try:
            # Validate signal quality if data provided
            if data is not None and "voltage" in data:
                valid = self._validate_signal_quality(
                    data["voltage"], target_mode, self.configs[target_mode].eye_height_threshold
                )

                if not valid:
                    return SwitchStatus(
                        success=False,
                        current_mode=self.current_mode,
                        error_message=f"Signal quality insufficient for {target_mode.name} mode",
                    )

            # Perform mode switch
            self._perform_mode_switch(target_mode)

            # Update current mode
            self.current_mode = target_mode

            return SwitchStatus(success=True, current_mode=self.current_mode, error_message=None)

        except Exception as e:
            logger.error(f"Mode switch failed: {str(e)}")
            return SwitchStatus(success=False, current_mode=self.current_mode, error_message=f"Mode switch failed: {str(e)}")

    def _validate_signal_quality(self, voltage_data: npt.NDArray[np.float64], target_mode: SignalMode, threshold: float) -> bool:
        """
        Validate signal quality for mode switch

        Args:
            voltage_data: Voltage measurements array
            target_mode: Target signal mode
            threshold: Quality threshold

        Returns:
            True if signal quality sufficient, False otherwise

        Raises:
            AssertionError: If parameters are invalid
        """
        # Validate inputs
        assert isinstance(voltage_data, np.ndarray), f"Voltage data must be numpy array, got {type(voltage_data)}"
        assert np.issubdtype(voltage_data.dtype, np.floating), f"Voltage data must be floating-point, got {voltage_data.dtype}"
        assert isinstance(target_mode, SignalMode), f"Target mode must be SignalMode, got {type(target_mode)}"
        assert isinstance(threshold, float), f"Threshold must be float, got {type(threshold)}"
        assert 0 <= threshold <= 1, f"Threshold must be between 0 and 1, got {threshold}"

        try:
            if target_mode == SignalMode.PAM4:
                # Check PAM4 levels
                hist, bins = np.histogram(voltage_data, bins=100)
                peaks, _ = self._find_peaks(hist)

                # Need at least 4 clear peaks for PAM4
                if len(peaks) < 4:
                    logger.warning(f"Found only {len(peaks)} levels, need 4 for PAM4")
                    return False

                # Check level separation
                levels = bins[peaks]
                sorted_levels = np.sort(levels)
                level_gaps = np.diff(sorted_levels[:4])  # Use first 4 if more found
                min_gap = float(np.min(level_gaps))

                # Calculate peak-to-noise ratio
                noise = float(np.std(voltage_data))
                snr = min_gap / noise

                return snr >= threshold

            else:  # NRZ mode
                # Check NRZ levels
                hist, bins = np.histogram(voltage_data, bins=100)
                peaks, _ = self._find_peaks(hist)

                # Need at least 2 clear peaks for NRZ
                if len(peaks) < 2:
                    logger.warning(f"Found only {len(peaks)} levels, need 2 for NRZ")
                    return False

                # For NRZ, just ensure signal amplitude is sufficient
                v_pp = float(np.max(voltage_data) - np.min(voltage_data))
                noise = float(np.std(voltage_data))
                snr = v_pp / noise

                return snr >= threshold

        except Exception as e:
            logger.error(f"Signal quality validation failed: {str(e)}")
            return False

    def _find_peaks(self, histogram: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """
        Find peaks in histogram for level detection

        Args:
            histogram: Signal histogram

        Returns:
            Tuple of (peak indices, peak heights)

        Raises:
            AssertionError: If histogram is invalid
        """
        # Validate input
        assert isinstance(histogram, np.ndarray), f"Histogram must be numpy array, got {type(histogram)}"

        # Find local maxima
        peak_indices = []
        peak_heights = []

        for i in range(1, len(histogram) - 1):
            if histogram[i] > histogram[i - 1] and histogram[i] > histogram[i + 1]:
                peak_indices.append(i)
                peak_heights.append(histogram[i])

        # Sort by height (descending)
        sorted_indices = np.argsort(peak_heights)[::-1]
        peak_indices = np.array(peak_indices, dtype=np.int64)[sorted_indices]
        peak_heights = np.array(peak_heights, dtype=np.float64)[sorted_indices]

        return peak_indices, peak_heights

    def _perform_mode_switch(self, target_mode: SignalMode) -> None:
        """
        Perform actual mode switch operation

        Args:
            target_mode: Target signal mode

        Raises:
            AssertionError: If target mode is invalid
        """
        # Validate input
        assert isinstance(target_mode, SignalMode), f"Target mode must be SignalMode, got {type(target_mode)}"

        # Get configuration
        config = self.configs[target_mode]

        # Apply mode-specific settings
        logger.info(f"Switching to {target_mode.name} mode")
        logger.info(f"Setting sample rate to {config.sample_rate:.2e} Hz")
        logger.info(f"Setting bandwidth to {config.bandwidth:.2e} Hz")
        logger.info(f"Setting {config.equalization_taps} equalization taps")

        # Implement hardware control for PCIe configuration
        try:
            # Configure PCIe link parameters
            self._configure_pcie_link_parameters(config)

            # Set equalization parameters
            self._configure_equalization_hardware(config)

            # Apply bandwidth settings
            self._configure_bandwidth_hardware(config)

            # Validate hardware configuration
            self._validate_hardware_configuration(config)

            logger.info("Hardware configuration applied successfully")

        except Exception as e:
            logger.error(f"Hardware configuration failed: {e}")
            raise RuntimeError(f"Failed to configure PCIe hardware: {e}")

    def _configure_pcie_link_parameters(self, config: PCIeConfig) -> None:
        """Configure PCIe link parameters in hardware"""
        try:
            # Set link width
            link_width = getattr(config, "link_width", 4)  # Default x4
            logger.debug(f"Configuring PCIe link width: x{link_width}")

            # Set link speed
            link_speed = getattr(config, "link_speed", "Gen4")  # Default Gen4
            logger.debug(f"Configuring PCIe link speed: {link_speed}")

            # Configure link training parameters
            training_params = {"max_retries": 10, "timeout_ms": 1000, "enable_compliance": True}

            # Apply link configuration
            # In actual hardware implementation, this would write to PCIe registers
            self._write_pcie_register(
                "LINK_CONTROL", {"width": link_width, "speed": link_speed, "training_params": training_params}
            )

        except Exception as e:
            logger.error(f"PCIe link parameter configuration failed: {e}")
            raise

    def _configure_equalization_hardware(self, config: PCIeConfig) -> None:
        """Configure equalization hardware settings"""
        try:
            eq_taps = config.equalization_taps
            logger.debug(f"Configuring {eq_taps} equalization taps")

            # Configure transmitter equalization
            tx_eq_settings = {
                "pre_cursor": -2,  # dB
                "cursor": 0,  # dB (reference)
                "post_cursor": -6,  # dB
            }

            # Configure receiver equalization
            rx_eq_settings = {
                "ctle_gain": 12,  # dB
                "dfe_taps": eq_taps,
                "adaptation_mode": "continuous",
            }

            # Apply equalization settings
            self._write_pcie_register("TX_EQUALIZATION", tx_eq_settings)
            self._write_pcie_register("RX_EQUALIZATION", rx_eq_settings)

            # Enable equalization adaptation
            self._write_pcie_register("EQ_CONTROL", {"enable_adaptation": True, "adaptation_rate": "medium"})

        except Exception as e:
            logger.error(f"Equalization hardware configuration failed: {e}")
            raise

    def _configure_bandwidth_hardware(self, config: PCIeConfig) -> None:
        """Configure bandwidth-related hardware settings"""
        try:
            bandwidth = config.bandwidth
            logger.debug(f"Configuring bandwidth: {bandwidth:.2e} Hz")

            # Calculate required clock settings
            symbol_rate = bandwidth / 2  # Assuming NRZ encoding

            # Configure PLL settings
            pll_settings = {
                "reference_clock": 100e6,  # 100 MHz reference
                "output_frequency": symbol_rate,
                "loop_bandwidth": symbol_rate / 1000,  # 1/1000 of symbol rate
                "phase_margin": 45,  # degrees
            }

            # Configure clock distribution
            clock_settings = {
                "enable_spread_spectrum": True,
                "ssc_frequency": 33000,  # 33 kHz
                "ssc_deviation": 0.005,  # 0.5%
            }

            # Apply clock configuration
            self._write_pcie_register("PLL_CONTROL", pll_settings)
            self._write_pcie_register("CLOCK_CONTROL", clock_settings)

        except Exception as e:
            logger.error(f"Bandwidth hardware configuration failed: {e}")
            raise

    def _validate_hardware_configuration(self, config: PCIeConfig) -> None:
        """Validate that hardware configuration was applied correctly"""
        try:
            # Read back configuration registers
            link_status = self._read_pcie_register("LINK_STATUS")
            eq_status = self._read_pcie_register("EQ_STATUS")
            clock_status = self._read_pcie_register("CLOCK_STATUS")

            # Validate link configuration
            if not link_status.get("link_up", False):
                raise RuntimeError("PCIe link failed to come up")

            # Validate equalization
            if not eq_status.get("equalization_complete", False):
                logger.warning("Equalization not complete, may affect performance")

            # Validate clock stability
            if not clock_status.get("pll_locked", False):
                raise RuntimeError("PLL failed to lock")

            logger.info("Hardware configuration validation passed")

        except Exception as e:
            logger.error(f"Hardware configuration validation failed: {e}")
            raise

    def _write_pcie_register(self, register_name: str, value: Any) -> None:
        """Write to PCIe hardware register"""
        try:
            # In actual implementation, this would write to hardware registers
            # For now, simulate register write with validation

            logger.debug(f"Writing PCIe register {register_name}: {value}")

            # Simulate register write delay
            import time

            time.sleep(0.001)  # 1ms delay

            # Store register value for readback
            if not hasattr(self, "_register_cache"):
                self._register_cache = {}

            self._register_cache[register_name] = value

        except Exception as e:
            logger.error(f"Failed to write PCIe register {register_name}: {e}")
            raise

    def _read_pcie_register(self, register_name: str) -> Dict[str, Any]:
        """Read from PCIe hardware register"""
        try:
            # In actual implementation, this would read from hardware registers
            # For now, simulate register read with cached values

            if not hasattr(self, "_register_cache"):
                self._register_cache = {}

            # Return cached value or simulate default status
            if register_name in self._register_cache:
                base_value = self._register_cache[register_name]
            else:
                base_value = {}

            # Simulate status registers
            if register_name == "LINK_STATUS":
                return {"link_up": True, "link_width": 4, "link_speed": "Gen4", "training_complete": True, **base_value}
            elif register_name == "EQ_STATUS":
                return {
                    "equalization_complete": True,
                    "adaptation_active": True,
                    "eye_height": 0.8,
                    "eye_width": 0.9,
                    **base_value,
                }
            elif register_name == "CLOCK_STATUS":
                return {
                    "pll_locked": True,
                    "clock_stable": True,
                    "frequency_error": 0.001,  # 0.1% error
                    **base_value,
                }
            else:
                return base_value

        except Exception as e:
            logger.error(f"Failed to read PCIe register {register_name}: {e}")
            raise


def create_mode_controller() -> ModeController:
    """
    Create and configure mode controller

    Returns:
        Configured mode controller
    """
    # Create controller
    controller = ModeController(default_mode=SignalMode.NRZ)

    # Register NRZ configuration
    nrz_config = ModeConfig(
        mode=SignalMode.NRZ,
        sample_rate=20e9,  # 20 GSa/s
        bandwidth=10e9,  # 10 GHz
        eye_height_threshold=0.5,
        equalization_taps=3,
    )
    controller.register_mode_config(nrz_config)

    # Register PAM4 configuration
    pam4_config = ModeConfig(
        mode=SignalMode.PAM4,
        sample_rate=40e9,  # 40 GSa/s
        bandwidth=20e9,  # 20 GHz
        eye_height_threshold=0.4,
        equalization_taps=5,
    )
    controller.register_mode_config(pam4_config)

    return controller
