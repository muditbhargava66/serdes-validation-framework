"""
USB4/Thunderbolt 4 Protocol Constants

This module defines constants, specifications, and enumerations for USB4 and Thunderbolt 4
validation, including operating modes, link states, tunneling protocols, and compliance limits.

Features:
- USB4 v2.0 specifications
- Thunderbolt 4 specifications
- Signal modes and link states
- Tunneling protocol definitions
- Power management parameters
- Compliance limits and thresholds
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Final, Tuple

import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USB4SignalMode(Enum):
    """USB4 signal operating modes"""

    GEN2X2 = auto()  # Gen 2 x2 lanes (20 Gbps per lane)
    GEN3X2 = auto()  # Gen 3 x2 lanes (20 Gbps per lane, enhanced signaling)
    ASYMMETRIC = auto()  # Asymmetric TX/RX configuration


class USB4LinkState(Enum):
    """USB4 link power states"""

    U0 = auto()  # Active state - full power operation
    U1 = auto()  # Standby state - reduced power, quick wake
    U2 = auto()  # Sleep state - low power, longer wake time
    U3 = auto()  # Suspend state - minimal power, full wake sequence


class USB4TunnelingMode(Enum):
    """USB4 tunneling protocol types"""

    PCIE = auto()  # PCIe tunneling over USB4
    DISPLAYPORT = auto()  # DisplayPort tunneling over USB4
    USB32 = auto()  # USB 3.2 tunneling over USB4
    NATIVE = auto()  # Native USB4 protocol


class ThunderboltSecurityLevel(Enum):
    """Thunderbolt security levels"""

    NONE = auto()  # No security (legacy mode)
    USER = auto()  # User approval required
    SECURE = auto()  # Secure connect with device authentication
    DPONLY = auto()  # DisplayPort only mode


class USB4ErrorType(Enum):
    """USB4 error types for testing"""

    SIGNAL_INTEGRITY = auto()  # Signal quality issues
    LINK_TRAINING = auto()  # Link training failures
    PROTOCOL = auto()  # Protocol violations
    POWER_MANAGEMENT = auto()  # Power state issues
    TUNNELING = auto()  # Tunneling protocol errors


@dataclass(frozen=True)
class USB4Specs:
    """USB4 v2.0 core specifications"""

    # Link parameters
    GEN2_RATE: float = 20.0e9  # 20 Gbps per lane
    GEN3_RATE: float = 20.0e9  # 20 Gbps per lane (enhanced signaling)
    MAX_LANES: int = 2  # USB4 uses 2 lanes
    TOTAL_BANDWIDTH: float = 40.0e9  # 40 Gbps total bandwidth

    # Timing parameters
    UI_PERIOD_GEN2: float = 50.0e-12  # Unit interval Gen 2 (50 ps)
    UI_PERIOD_GEN3: float = 50.0e-12  # Unit interval Gen 3 (50 ps)
    SSC_MODULATION: float = 0.5  # Spread spectrum modulation (%)
    SSC_FREQUENCY: float = 33.0e3  # SSC modulation frequency (33 kHz)

    # Signal integrity parameters
    DIFFERENTIAL_VOLTAGE: float = 1.2  # Differential voltage swing (V)
    COMMON_MODE_VOLTAGE: float = 0.0  # Common mode voltage (V)
    MAX_LANE_SKEW: float = 20.0e-12  # Maximum lane skew (20 ps)

    # Jitter specifications
    MAX_TOTAL_JITTER: float = 0.35  # Maximum total jitter (UI)
    MAX_RANDOM_JITTER: float = 0.1  # Maximum random jitter (UI)
    MAX_DETERMINISTIC_JITTER: float = 0.25  # Maximum deterministic jitter (UI)

    # Eye diagram specifications
    MIN_EYE_HEIGHT: float = 0.4  # Minimum eye height (normalized)
    MIN_EYE_WIDTH: float = 0.6  # Minimum eye width (normalized)

    # Power parameters
    MAX_POWER_DELIVERY: float = 100.0  # Maximum USB-PD power (W)
    IDLE_POWER_U0: float = 2.5  # U0 idle power consumption (W)
    IDLE_POWER_U1: float = 0.5  # U1 standby power consumption (W)
    IDLE_POWER_U2: float = 0.1  # U2 sleep power consumption (W)
    IDLE_POWER_U3: float = 0.01  # U3 suspend power consumption (W)

    # Link training parameters
    MAX_TRAINING_TIME: float = 100.0e-3  # Maximum link training time (100 ms)
    TRAINING_TIMEOUT: float = 1.0  # Training timeout (s)
    MAX_RETRIES: int = 3  # Maximum training retries

    def validate_link_parameters(self) -> None:
        """Validate USB4 link parameters with assertions"""
        assert self.GEN2_RATE > 0, "Gen2 rate must be positive"
        assert self.GEN3_RATE > 0, "Gen3 rate must be positive"
        assert self.MAX_LANES == 2, "USB4 must use exactly 2 lanes"
        assert self.TOTAL_BANDWIDTH == self.GEN2_RATE * self.MAX_LANES, "Total bandwidth must equal rate * lanes"

    def validate_timing_parameters(self) -> None:
        """Validate USB4 timing parameters with assertions"""
        assert self.UI_PERIOD_GEN2 > 0, "UI period Gen2 must be positive"
        assert self.UI_PERIOD_GEN3 > 0, "UI period Gen3 must be positive"
        assert 0 < self.SSC_MODULATION <= 1.0, "SSC modulation must be between 0 and 1"
        assert self.SSC_FREQUENCY > 0, "SSC frequency must be positive"

    def validate_signal_parameters(self) -> None:
        """Validate USB4 signal integrity parameters with assertions"""
        assert self.DIFFERENTIAL_VOLTAGE > 0, "Differential voltage must be positive"
        assert self.MAX_LANE_SKEW > 0, "Max lane skew must be positive"
        assert 0 < self.MAX_TOTAL_JITTER < 1.0, "Total jitter must be between 0 and 1 UI"
        assert 0 < self.MAX_RANDOM_JITTER < self.MAX_TOTAL_JITTER, "Random jitter must be less than total jitter"
        assert 0 < self.MAX_DETERMINISTIC_JITTER < self.MAX_TOTAL_JITTER, "Deterministic jitter must be less than total jitter"

    def validate_eye_parameters(self) -> None:
        """Validate USB4 eye diagram parameters with assertions"""
        assert 0 < self.MIN_EYE_HEIGHT < 1.0, "Eye height must be between 0 and 1"
        assert 0 < self.MIN_EYE_WIDTH < 1.0, "Eye width must be between 0 and 1"

    def validate_power_parameters(self) -> None:
        """Validate USB4 power parameters with assertions"""
        assert self.MAX_POWER_DELIVERY > 0, "Max power delivery must be positive"
        assert self.IDLE_POWER_U0 > self.IDLE_POWER_U1, "U0 power must be greater than U1"
        assert self.IDLE_POWER_U1 > self.IDLE_POWER_U2, "U1 power must be greater than U2"
        assert self.IDLE_POWER_U2 > self.IDLE_POWER_U3, "U2 power must be greater than U3"
        assert self.IDLE_POWER_U3 > 0, "U3 power must be positive"

    def validate_training_parameters(self) -> None:
        """Validate USB4 link training parameters with assertions"""
        assert self.MAX_TRAINING_TIME > 0, "Max training time must be positive"
        assert self.TRAINING_TIMEOUT > self.MAX_TRAINING_TIME, "Training timeout must be greater than max training time"
        assert self.MAX_RETRIES > 0, "Max retries must be positive"

    def validate_all(self) -> None:
        """Validate all USB4 specifications with comprehensive assertions"""
        self.validate_link_parameters()
        self.validate_timing_parameters()
        self.validate_signal_parameters()
        self.validate_eye_parameters()
        self.validate_power_parameters()
        self.validate_training_parameters()


@dataclass(frozen=True)
class ThunderboltSpecs:
    """Thunderbolt 4 specific specifications"""

    # Thunderbolt parameters
    TB4_BANDWIDTH: float = 40.0e9  # Thunderbolt 4 bandwidth (40 Gbps)
    TB4_LANES: int = 2  # Thunderbolt 4 lanes

    # Security parameters
    DMA_PROTECTION: bool = True  # DMA protection required
    DEVICE_AUTH_REQUIRED: bool = True  # Device authentication required
    SECURE_BOOT: bool = True  # Secure boot support

    # Device authentication constants
    AUTH_TIMEOUT: float = 5.0  # Device authentication timeout (s)
    AUTH_RETRY_COUNT: int = 3  # Authentication retry attempts
    CHALLENGE_SIZE: int = 32  # Authentication challenge size (bytes)
    CERTIFICATE_SIZE: int = 512  # Device certificate size (bytes)

    # Daisy chain configuration limits and specifications
    MAX_DAISY_DEVICES: int = 6  # Maximum devices in daisy chain
    MAX_CABLE_LENGTH: float = 2.0  # Maximum cable length (m)
    DAISY_CHAIN_LATENCY: float = 1.0e-6  # Additional latency per hop (1 μs)
    CHAIN_BANDWIDTH_DEGRADATION: float = 0.05  # Bandwidth loss per hop (5%)

    # Power delivery
    TB4_POWER_DELIVERY: float = 100.0  # Thunderbolt 4 power delivery (W)
    MIN_POWER_LAPTOP: float = 15.0  # Minimum power for laptop charging (W)
    POWER_NEGOTIATION_TIME: float = 1.0  # Power negotiation timeout (s)

    # Display support
    MAX_4K_DISPLAYS: int = 2  # Maximum 4K displays supported
    MAX_8K_DISPLAYS: int = 1  # Maximum 8K displays supported
    DISPLAY_BANDWIDTH_4K: float = 12.54e9  # 4K display bandwidth (12.54 Gbps)
    DISPLAY_BANDWIDTH_8K: float = 50.16e9  # 8K display bandwidth (50.16 Gbps)

    # PCIe tunneling
    PCIE_BANDWIDTH: float = 32.0e9  # PCIe bandwidth over Thunderbolt (32 Gbps)
    PCIE_LANES: int = 4  # PCIe lanes supported

    # Intel certification test parameters
    CERT_TEST_DURATION: float = 3600.0  # Certification test duration (1 hour)
    CERT_ERROR_THRESHOLD: float = 1e-12  # Maximum error rate for certification
    CERT_TEMPERATURE_RANGE: Tuple[float, float] = (-40.0, 85.0)  # Operating temperature range (°C)
    CERT_VOLTAGE_TOLERANCE: float = 0.05  # Voltage tolerance for certification (5%)
    CERT_JITTER_LIMIT: float = 0.3  # Jitter limit for certification (UI)
    CERT_EYE_MARGIN: float = 0.1  # Eye margin requirement for certification

    def validate_security_parameters(self) -> None:
        """Validate Thunderbolt security parameters with assertions"""
        assert self.AUTH_TIMEOUT > 0, "Authentication timeout must be positive"
        assert self.AUTH_RETRY_COUNT > 0, "Authentication retry count must be positive"
        assert self.CHALLENGE_SIZE > 0, "Challenge size must be positive"
        assert self.CERTIFICATE_SIZE > 0, "Certificate size must be positive"

    def validate_daisy_chain_specs(self) -> None:
        """Validate daisy chain specifications with assertions"""
        assert 1 <= self.MAX_DAISY_DEVICES <= 6, "Daisy chain devices must be between 1 and 6"
        assert self.MAX_CABLE_LENGTH > 0, "Cable length must be positive"
        assert self.DAISY_CHAIN_LATENCY > 0, "Daisy chain latency must be positive"
        assert 0 < self.CHAIN_BANDWIDTH_DEGRADATION < 1.0, "Bandwidth degradation must be between 0 and 100%"

    def validate_power_parameters(self) -> None:
        """Validate power delivery parameters with assertions"""
        assert self.TB4_POWER_DELIVERY > 0, "Power delivery must be positive"
        assert self.MIN_POWER_LAPTOP > 0, "Minimum laptop power must be positive"
        assert self.MIN_POWER_LAPTOP < self.TB4_POWER_DELIVERY, "Minimum power must be less than maximum"
        assert self.POWER_NEGOTIATION_TIME > 0, "Power negotiation time must be positive"

    def validate_display_parameters(self) -> None:
        """Validate display support parameters with assertions"""
        assert self.MAX_4K_DISPLAYS > 0, "4K display count must be positive"
        assert self.MAX_8K_DISPLAYS > 0, "8K display count must be positive"
        assert self.DISPLAY_BANDWIDTH_4K > 0, "4K display bandwidth must be positive"
        assert self.DISPLAY_BANDWIDTH_8K > self.DISPLAY_BANDWIDTH_4K, "8K bandwidth must be greater than 4K"

    def validate_pcie_parameters(self) -> None:
        """Validate PCIe tunneling parameters with assertions"""
        assert self.PCIE_BANDWIDTH > 0, "PCIe bandwidth must be positive"
        assert self.PCIE_BANDWIDTH < self.TB4_BANDWIDTH, "PCIe bandwidth must be less than total bandwidth"
        assert self.PCIE_LANES > 0, "PCIe lanes must be positive"

    def validate_certification_parameters(self) -> None:
        """Validate Intel certification parameters with assertions"""
        assert self.CERT_TEST_DURATION > 0, "Certification test duration must be positive"
        assert 0 < self.CERT_ERROR_THRESHOLD < 1.0, "Error threshold must be between 0 and 1"
        assert len(self.CERT_TEMPERATURE_RANGE) == 2, "Temperature range must have min and max"
        assert self.CERT_TEMPERATURE_RANGE[0] < self.CERT_TEMPERATURE_RANGE[1], "Temperature range must be valid"
        assert 0 < self.CERT_VOLTAGE_TOLERANCE < 1.0, "Voltage tolerance must be between 0 and 100%"
        assert 0 < self.CERT_JITTER_LIMIT < 1.0, "Jitter limit must be between 0 and 1 UI"
        assert 0 < self.CERT_EYE_MARGIN < 1.0, "Eye margin must be between 0 and 1"

    def validate_all(self) -> None:
        """Validate all Thunderbolt specifications with comprehensive assertions"""
        self.validate_security_parameters()
        self.validate_daisy_chain_specs()
        self.validate_power_parameters()
        self.validate_display_parameters()
        self.validate_pcie_parameters()
        self.validate_certification_parameters()


@dataclass(frozen=True)
class USB4SignalSpecs:
    """USB4 signal integrity specifications"""

    # Eye diagram limits
    EYE_HEIGHT_MIN: float = 0.4  # Minimum eye height (normalized)
    EYE_WIDTH_MIN: float = 0.6  # Minimum eye width (normalized)
    EYE_CROSSING_MAX: float = 0.05  # Maximum eye crossing percentage

    # Jitter limits
    TOTAL_JITTER_MAX: float = 0.35  # Maximum total jitter (UI)
    RANDOM_JITTER_MAX: float = 0.1  # Maximum random jitter (UI)
    DETERMINISTIC_JITTER_MAX: float = 0.25  # Maximum deterministic jitter (UI)
    PERIODIC_JITTER_MAX: float = 0.1  # Maximum periodic jitter (UI)

    # Voltage levels
    VOD_MIN: float = 0.8  # Minimum output differential voltage (V)
    VOD_MAX: float = 1.6  # Maximum output differential voltage (V)
    VCM_MIN: float = -0.3  # Minimum common mode voltage (V)
    VCM_MAX: float = 0.3  # Maximum common mode voltage (V)

    # Rise/fall times
    RISE_TIME_MAX: float = 35.0e-12  # Maximum rise time (35 ps)
    FALL_TIME_MAX: float = 35.0e-12  # Maximum fall time (35 ps)

    # Skew specifications
    INTRA_PAIR_SKEW_MAX: float = 5.0e-12  # Maximum intra-pair skew (5 ps)
    INTER_PAIR_SKEW_MAX: float = 20.0e-12  # Maximum inter-pair skew (20 ps)

    # Spread Spectrum Clocking (SSC) specifications
    SSC_MODULATION_DEPTH: float = 0.5  # SSC modulation depth (%)
    SSC_FREQUENCY_MIN: float = 30.0e3  # Minimum SSC frequency (30 kHz)
    SSC_FREQUENCY_MAX: float = 33.0e3  # Maximum SSC frequency (33 kHz)
    SSC_PROFILE: str = "down_spread"  # SSC profile type

    # Lane skew tolerance and compensation
    LANE_SKEW_TOLERANCE: float = 20.0e-12  # Lane skew tolerance (20 ps)
    SKEW_COMPENSATION_RANGE: float = 100.0e-12  # Skew compensation range (100 ps)
    SKEW_COMPENSATION_RESOLUTION: float = 1.0e-12  # Skew compensation resolution (1 ps)

    def validate_eye_diagram_limits(self) -> None:
        """Validate eye diagram specifications with assertions"""
        assert 0 < self.EYE_HEIGHT_MIN < 1.0, "Eye height minimum must be between 0 and 1"
        assert 0 < self.EYE_WIDTH_MIN < 1.0, "Eye width minimum must be between 0 and 1"
        assert 0 < self.EYE_CROSSING_MAX < 0.1, "Eye crossing maximum must be less than 10%"

    def validate_jitter_limits(self) -> None:
        """Validate jitter specifications with assertions"""
        assert 0 < self.TOTAL_JITTER_MAX < 1.0, "Total jitter must be between 0 and 1 UI"
        assert 0 < self.RANDOM_JITTER_MAX < self.TOTAL_JITTER_MAX, "Random jitter must be less than total jitter"
        assert 0 < self.DETERMINISTIC_JITTER_MAX < self.TOTAL_JITTER_MAX, "Deterministic jitter must be less than total jitter"
        assert (
            0 < self.PERIODIC_JITTER_MAX < self.DETERMINISTIC_JITTER_MAX
        ), "Periodic jitter must be less than deterministic jitter"

    def validate_voltage_levels(self) -> None:
        """Validate voltage level specifications with assertions"""
        assert self.VOD_MIN > 0, "Minimum differential voltage must be positive"
        assert self.VOD_MAX > self.VOD_MIN, "Maximum differential voltage must be greater than minimum"
        assert self.VCM_MIN < self.VCM_MAX, "Common mode voltage range must be valid"

    def validate_timing_specs(self) -> None:
        """Validate timing specifications with assertions"""
        assert self.RISE_TIME_MAX > 0, "Rise time maximum must be positive"
        assert self.FALL_TIME_MAX > 0, "Fall time maximum must be positive"

    def validate_skew_specs(self) -> None:
        """Validate skew specifications with assertions"""
        assert self.INTRA_PAIR_SKEW_MAX > 0, "Intra-pair skew maximum must be positive"
        assert self.INTER_PAIR_SKEW_MAX > self.INTRA_PAIR_SKEW_MAX, "Inter-pair skew must be greater than intra-pair skew"
        assert self.LANE_SKEW_TOLERANCE > 0, "Lane skew tolerance must be positive"
        assert self.SKEW_COMPENSATION_RANGE > self.LANE_SKEW_TOLERANCE, "Compensation range must be greater than tolerance"
        assert self.SKEW_COMPENSATION_RESOLUTION > 0, "Compensation resolution must be positive"

    def validate_ssc_specs(self) -> None:
        """Validate spread spectrum clocking specifications with assertions"""
        assert 0 < self.SSC_MODULATION_DEPTH <= 1.0, "SSC modulation depth must be between 0 and 1%"
        assert self.SSC_FREQUENCY_MIN > 0, "SSC frequency minimum must be positive"
        assert self.SSC_FREQUENCY_MAX > self.SSC_FREQUENCY_MIN, "SSC frequency maximum must be greater than minimum"
        assert self.SSC_PROFILE in ["down_spread", "center_spread"], "SSC profile must be valid"

    def validate_all(self) -> None:
        """Validate all signal integrity specifications with comprehensive assertions"""
        self.validate_eye_diagram_limits()
        self.validate_jitter_limits()
        self.validate_voltage_levels()
        self.validate_timing_specs()
        self.validate_skew_specs()
        self.validate_ssc_specs()


@dataclass(frozen=True)
class USB4TunnelingSpecs:
    """USB4 multi-protocol tunneling specifications"""

    # Bandwidth allocation
    MIN_TUNNEL_BANDWIDTH: float = 1.0e9  # Minimum tunnel bandwidth (1 Gbps)
    MAX_TUNNEL_OVERHEAD: float = 0.1  # Maximum tunneling overhead (10%)

    # PCIe tunneling
    PCIE_MIN_BANDWIDTH: float = 8.0e9  # Minimum PCIe tunnel bandwidth (8 Gbps)
    PCIE_MAX_LATENCY: float = 1.0e-6  # Maximum PCIe tunnel latency (1 μs)
    PCIE_MAX_PACKET_SIZE: int = 4096  # Maximum PCIe packet size (bytes)

    # DisplayPort tunneling
    DP_MIN_BANDWIDTH: float = 5.4e9  # Minimum DP tunnel bandwidth (5.4 Gbps)
    DP_MAX_LATENCY: float = 100.0e-6  # Maximum DP tunnel latency (100 μs)
    DP_MAX_STREAMS: int = 4  # Maximum DisplayPort streams

    # USB 3.2 tunneling
    USB32_MIN_BANDWIDTH: float = 5.0e9  # Minimum USB 3.2 tunnel bandwidth (5 Gbps)
    USB32_MAX_LATENCY: float = 10.0e-6  # Maximum USB 3.2 tunnel latency (10 μs)
    USB32_MAX_DEVICES: int = 127  # Maximum USB devices per tunnel

    # Flow control
    CREDIT_LIMIT: int = 256  # Flow control credit limit
    BUFFER_SIZE: int = 4096  # Tunnel buffer size (bytes)

    # Multi-protocol parameters
    MAX_CONCURRENT_TUNNELS: int = 8  # Maximum concurrent tunnels
    TUNNEL_SETUP_TIMEOUT: float = 1.0  # Tunnel setup timeout (s)
    BANDWIDTH_ALLOCATION_QUANTUM: float = 1.0e6  # Bandwidth allocation quantum (1 Mbps)

    def validate_bandwidth_specs(self) -> None:
        """Validate bandwidth specifications with assertions"""
        assert self.MIN_TUNNEL_BANDWIDTH > 0, "Minimum tunnel bandwidth must be positive"
        assert 0 < self.MAX_TUNNEL_OVERHEAD < 1.0, "Tunnel overhead must be between 0 and 100%"
        assert self.BANDWIDTH_ALLOCATION_QUANTUM > 0, "Bandwidth allocation quantum must be positive"

    def validate_pcie_tunneling(self) -> None:
        """Validate PCIe tunneling specifications with assertions"""
        assert self.PCIE_MIN_BANDWIDTH >= self.MIN_TUNNEL_BANDWIDTH, "PCIe bandwidth must meet minimum"
        assert self.PCIE_MAX_LATENCY > 0, "PCIe latency must be positive"
        assert self.PCIE_MAX_PACKET_SIZE > 0, "PCIe packet size must be positive"

    def validate_displayport_tunneling(self) -> None:
        """Validate DisplayPort tunneling specifications with assertions"""
        assert self.DP_MIN_BANDWIDTH >= self.MIN_TUNNEL_BANDWIDTH, "DisplayPort bandwidth must meet minimum"
        assert self.DP_MAX_LATENCY > 0, "DisplayPort latency must be positive"
        assert self.DP_MAX_STREAMS > 0, "DisplayPort streams must be positive"

    def validate_usb32_tunneling(self) -> None:
        """Validate USB 3.2 tunneling specifications with assertions"""
        assert self.USB32_MIN_BANDWIDTH >= self.MIN_TUNNEL_BANDWIDTH, "USB 3.2 bandwidth must meet minimum"
        assert self.USB32_MAX_LATENCY > 0, "USB 3.2 latency must be positive"
        assert self.USB32_MAX_DEVICES > 0, "USB 3.2 device count must be positive"

    def validate_flow_control(self) -> None:
        """Validate flow control specifications with assertions"""
        assert self.CREDIT_LIMIT > 0, "Credit limit must be positive"
        assert self.BUFFER_SIZE > 0, "Buffer size must be positive"

    def validate_multi_protocol(self) -> None:
        """Validate multi-protocol specifications with assertions"""
        assert self.MAX_CONCURRENT_TUNNELS > 0, "Maximum concurrent tunnels must be positive"
        assert self.TUNNEL_SETUP_TIMEOUT > 0, "Tunnel setup timeout must be positive"

    def validate_all(self) -> None:
        """Validate all tunneling specifications with comprehensive assertions"""
        self.validate_bandwidth_specs()
        self.validate_pcie_tunneling()
        self.validate_displayport_tunneling()
        self.validate_usb32_tunneling()
        self.validate_flow_control()
        self.validate_multi_protocol()


# Protocol constants collection
USB4_PROTOCOL_SPECS: Final[Dict[str, object]] = {
    "usb4": USB4Specs(),
    "thunderbolt": ThunderboltSpecs(),
    "signal": USB4SignalSpecs(),
    "tunneling": USB4TunnelingSpecs(),
}

# USB4 compliance patterns for testing
USB4_COMPLIANCE_PATTERNS: Final[Dict[str, npt.NDArray[np.complex128]]] = {
    "prbs7": np.array([]),  # Will be populated with actual PRBS patterns
    "prbs15": np.array([]),  # Will be populated with actual PRBS patterns
    "prbs31": np.array([]),  # Will be populated with actual PRBS patterns
}

# USB4 training patterns
USB4_TRAINING_PATTERNS: Final[Dict[str, npt.NDArray[np.complex128]]] = {
    "ts1": np.array([]),  # Training Sequence 1
    "ts2": np.array([]),  # Training Sequence 2
    "idle": np.array([]),  # Idle pattern
}


def validate_usb4_signal_mode(mode: USB4SignalMode) -> bool:
    """
    Validate USB4 signal mode

    Args:
        mode: USB4 signal mode to validate

    Returns:
        True if mode is valid

    Raises:
        ValueError: If mode is invalid
    """
    if not isinstance(mode, USB4SignalMode):
        raise ValueError(f"Invalid USB4 signal mode: {mode}")
    return True


def validate_usb4_link_state(state: USB4LinkState) -> bool:
    """
    Validate USB4 link state

    Args:
        state: USB4 link state to validate

    Returns:
        True if state is valid

    Raises:
        ValueError: If state is invalid
    """
    if not isinstance(state, USB4LinkState):
        raise ValueError(f"Invalid USB4 link state: {state}")
    return True


def calculate_usb4_ui_parameters(mode: USB4SignalMode) -> Tuple[float, float]:
    """
    Calculate USB4 unit interval parameters for given mode

    Args:
        mode: USB4 signal mode

    Returns:
        Tuple of (ui_period, symbol_rate)
    """
    validate_usb4_signal_mode(mode)
    specs = USB4Specs()

    if mode in [USB4SignalMode.GEN2X2, USB4SignalMode.GEN3X2]:
        ui_period = specs.UI_PERIOD_GEN2 if mode == USB4SignalMode.GEN2X2 else specs.UI_PERIOD_GEN3
        symbol_rate = 1.0 / ui_period
        return ui_period, symbol_rate
    else:
        # Asymmetric mode - use Gen 2 parameters as default
        return specs.UI_PERIOD_GEN2, 1.0 / specs.UI_PERIOD_GEN2


def get_usb4_power_consumption(state: USB4LinkState) -> float:
    """
    Get expected power consumption for USB4 link state

    Args:
        state: USB4 link state

    Returns:
        Power consumption in watts
    """
    validate_usb4_link_state(state)
    specs = USB4Specs()

    power_map = {
        USB4LinkState.U0: specs.IDLE_POWER_U0,
        USB4LinkState.U1: specs.IDLE_POWER_U1,
        USB4LinkState.U2: specs.IDLE_POWER_U2,
        USB4LinkState.U3: specs.IDLE_POWER_U3,
    }

    return power_map[state]


def get_tunneling_bandwidth_limit(tunnel_mode: USB4TunnelingMode) -> float:
    """
    Get bandwidth limit for tunneling mode

    Args:
        tunnel_mode: USB4 tunneling mode

    Returns:
        Bandwidth limit in bps
    """
    if not isinstance(tunnel_mode, USB4TunnelingMode):
        raise ValueError(f"Invalid tunneling mode: {tunnel_mode}")

    specs = USB4TunnelingSpecs()

    bandwidth_map = {
        USB4TunnelingMode.PCIE: specs.PCIE_MIN_BANDWIDTH,
        USB4TunnelingMode.DISPLAYPORT: specs.DP_MIN_BANDWIDTH,
        USB4TunnelingMode.USB32: specs.USB32_MIN_BANDWIDTH,
        USB4TunnelingMode.NATIVE: USB4Specs().TOTAL_BANDWIDTH,
    }

    return bandwidth_map[tunnel_mode]


__all__ = [
    # Enumerations
    "USB4SignalMode",
    "USB4LinkState",
    "USB4TunnelingMode",
    "ThunderboltSecurityLevel",
    "USB4ErrorType",
    # Specifications
    "USB4Specs",
    "ThunderboltSpecs",
    "USB4SignalSpecs",
    "USB4TunnelingSpecs",
    # Constants
    "USB4_PROTOCOL_SPECS",
    "USB4_COMPLIANCE_PATTERNS",
    "USB4_TRAINING_PATTERNS",
    # Utility functions
    "validate_usb4_signal_mode",
    "validate_usb4_link_state",
    "calculate_usb4_ui_parameters",
    "get_usb4_power_consumption",
    "get_tunneling_bandwidth_limit",
]
