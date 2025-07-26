"""
Thunderbolt 4 Mock Device Simulator Module

This module provides simulation of Thunderbolt 4 devices for testing without physical
hardware. Simulates device authentication, daisy chaining, power delivery, and
security features.

Features:
- Thunderbolt 4 device simulation
- Device authentication simulation
- Daisy chain topology simulation
- Power delivery negotiation
- Security feature simulation
- Performance characteristics modeling
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Union
from uuid import uuid4

import numpy as np

from ..constants import USB4TunnelingMode
from .constants import ThunderboltSpecs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThunderboltDeviceType(Enum):
    """Thunderbolt device types"""

    HOST = auto()
    DEVICE = auto()
    HUB = auto()
    DOCK = auto()
    STORAGE = auto()
    DISPLAY = auto()
    EGPU = auto()


class ThunderboltSecurityLevel(Enum):
    """Thunderbolt security levels"""

    NONE = auto()
    USER = auto()
    SECURE = auto()
    DPONLY = auto()


class ThunderboltDeviceState(Enum):
    """Thunderbolt device states"""

    DISCONNECTED = auto()
    CONNECTING = auto()
    AUTHENTICATING = auto()
    CONNECTED = auto()
    SUSPENDED = auto()
    ERROR = auto()


@dataclass
class ThunderboltDeviceInfo:
    """Thunderbolt device information"""

    device_id: str
    device_type: ThunderboltDeviceType
    vendor_id: int
    device_name: str
    firmware_version: str
    max_bandwidth: float  # Gbps
    power_consumption: float  # Watts
    supports_daisy_chain: bool
    security_level: ThunderboltSecurityLevel

    def __post_init__(self) -> None:
        """Validate device info"""
        assert isinstance(self.device_id, str), f"Device ID must be string, got {type(self.device_id)}"
        assert isinstance(
            self.device_type, ThunderboltDeviceType
        ), f"Device type must be ThunderboltDeviceType, got {type(self.device_type)}"
        assert isinstance(self.vendor_id, int), f"Vendor ID must be int, got {type(self.vendor_id)}"
        assert isinstance(self.device_name, str), f"Device name must be string, got {type(self.device_name)}"
        assert isinstance(self.firmware_version, str), f"Firmware version must be string, got {type(self.firmware_version)}"
        assert isinstance(self.max_bandwidth, float), f"Max bandwidth must be float, got {type(self.max_bandwidth)}"
        assert isinstance(self.power_consumption, float), f"Power consumption must be float, got {type(self.power_consumption)}"
        assert isinstance(
            self.supports_daisy_chain, bool
        ), f"Supports daisy chain must be bool, got {type(self.supports_daisy_chain)}"
        assert isinstance(
            self.security_level, ThunderboltSecurityLevel
        ), f"Security level must be ThunderboltSecurityLevel, got {type(self.security_level)}"

        assert self.vendor_id >= 0, f"Vendor ID must be non-negative, got {self.vendor_id}"
        assert self.max_bandwidth > 0, f"Max bandwidth must be positive, got {self.max_bandwidth}"
        assert self.power_consumption >= 0, f"Power consumption must be non-negative, got {self.power_consumption}"


@dataclass
class ThunderboltAuthResult:
    """Thunderbolt authentication result"""

    success: bool
    auth_time: float
    challenge_response: bytes
    certificate_valid: bool
    security_level_achieved: ThunderboltSecurityLevel
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate auth result"""
        assert isinstance(self.success, bool), f"Success must be bool, got {type(self.success)}"
        assert isinstance(self.auth_time, float), f"Auth time must be float, got {type(self.auth_time)}"
        assert isinstance(
            self.challenge_response, bytes
        ), f"Challenge response must be bytes, got {type(self.challenge_response)}"
        assert isinstance(self.certificate_valid, bool), f"Certificate valid must be bool, got {type(self.certificate_valid)}"
        assert isinstance(
            self.security_level_achieved, ThunderboltSecurityLevel
        ), f"Security level must be ThunderboltSecurityLevel, got {type(self.security_level_achieved)}"

        assert self.auth_time >= 0, f"Auth time must be non-negative, got {self.auth_time}"


@dataclass
class ThunderboltPowerNegotiation:
    """Thunderbolt power delivery negotiation"""

    requested_power: float
    allocated_power: float
    voltage: float
    current: float
    negotiation_time: float
    success: bool

    def __post_init__(self) -> None:
        """Validate power negotiation"""
        assert isinstance(self.requested_power, float), f"Requested power must be float, got {type(self.requested_power)}"
        assert isinstance(self.allocated_power, float), f"Allocated power must be float, got {type(self.allocated_power)}"
        assert isinstance(self.voltage, float), f"Voltage must be float, got {type(self.voltage)}"
        assert isinstance(self.current, float), f"Current must be float, got {type(self.current)}"
        assert isinstance(self.negotiation_time, float), f"Negotiation time must be float, got {type(self.negotiation_time)}"
        assert isinstance(self.success, bool), f"Success must be bool, got {type(self.success)}"

        assert self.requested_power >= 0, f"Requested power must be non-negative, got {self.requested_power}"
        assert self.allocated_power >= 0, f"Allocated power must be non-negative, got {self.allocated_power}"
        assert self.voltage >= 0, f"Voltage must be non-negative, got {self.voltage}"
        assert self.current >= 0, f"Current must be non-negative, got {self.current}"
        assert self.negotiation_time >= 0, f"Negotiation time must be non-negative, got {self.negotiation_time}"


class ThunderboltMockDevice:
    """Mock Thunderbolt 4 device simulator"""

    def __init__(self, device_info: ThunderboltDeviceInfo) -> None:
        """
        Initialize Thunderbolt mock device

        Args:
            device_info: Device information

        Raises:
            ValueError: If initialization fails
        """
        assert isinstance(
            device_info, ThunderboltDeviceInfo
        ), f"Device info must be ThunderboltDeviceInfo, got {type(device_info)}"

        self.device_info = device_info
        self.thunderbolt_specs = ThunderboltSpecs()

        # Device state
        self.state = ThunderboltDeviceState.DISCONNECTED
        self.connection_time = 0.0
        self.last_auth_result: Optional[ThunderboltAuthResult] = None
        self.power_negotiation: Optional[ThunderboltPowerNegotiation] = None

        # Daisy chain support
        self.upstream_device: Optional["ThunderboltMockDevice"] = None
        self.downstream_devices: List["ThunderboltMockDevice"] = []
        self.chain_position = 0

        # Performance tracking
        self.bandwidth_usage = 0.0
        self.latency_measurements: List[float] = []

        # Security features
        self.certificate = self._generate_device_certificate()
        self.private_key = self._generate_private_key()

        # Random number generator for simulation
        self.rng = np.random.RandomState(hash(device_info.device_id) % 2**32)

        logger.info(f"Thunderbolt mock device initialized: {device_info.device_name}")

    def connect_to_host(self, host_security_level: ThunderboltSecurityLevel, timeout: float = 10.0) -> bool:
        """
        Connect device to Thunderbolt host

        Args:
            host_security_level: Host security level requirement
            timeout: Connection timeout in seconds

        Returns:
            True if connection successful

        Raises:
            ValueError: If connection fails
        """
        assert isinstance(
            host_security_level, ThunderboltSecurityLevel
        ), f"Host security level must be ThunderboltSecurityLevel, got {type(host_security_level)}"
        assert isinstance(timeout, float), f"Timeout must be float, got {type(timeout)}"
        assert timeout > 0, f"Timeout must be positive, got {timeout}"

        try:
            logger.info(f"Connecting {self.device_info.device_name} to host...")

            self.state = ThunderboltDeviceState.CONNECTING
            start_time = time.time()

            # Simulate connection process
            connection_steps = [
                ("Physical layer detection", 0.1),
                ("Link training", 0.2),
                ("Device enumeration", 0.1),
                ("Security negotiation", 0.3),
                ("Power negotiation", 0.2),
            ]

            for step_name, step_duration in connection_steps:
                logger.debug(f"Connection step: {step_name}")
                time.sleep(step_duration)

                # Check timeout
                if time.time() - start_time > timeout:
                    self.state = ThunderboltDeviceState.ERROR
                    raise ValueError("Connection timeout")

                # Simulate occasional connection failures
                if self.rng.random() < 0.02:  # 2% failure rate
                    self.state = ThunderboltDeviceState.ERROR
                    raise ValueError(f"Connection failed at: {step_name}")

            # Perform authentication if required
            if host_security_level != ThunderboltSecurityLevel.NONE:
                auth_success = self.authenticate(host_security_level)
                if not auth_success:
                    self.state = ThunderboltDeviceState.ERROR
                    raise ValueError("Authentication failed")

            # Negotiate power delivery
            power_success = self._negotiate_power_delivery()
            if not power_success:
                logger.warning("Power negotiation failed, continuing with default power")

            # Connection successful
            self.state = ThunderboltDeviceState.CONNECTED
            self.connection_time = time.time()

            logger.info(f"Device {self.device_info.device_name} connected successfully")
            return True

        except Exception as e:
            logger.error(f"Device connection failed: {e}")
            self.state = ThunderboltDeviceState.ERROR
            raise ValueError(f"Device connection failed: {e}")

    def disconnect(self) -> bool:
        """
        Disconnect device from host

        Returns:
            True if disconnection successful

        Raises:
            ValueError: If disconnection fails
        """
        try:
            if self.state == ThunderboltDeviceState.DISCONNECTED:
                logger.warning("Device already disconnected")
                return True

            logger.info(f"Disconnecting {self.device_info.device_name}...")

            # Disconnect downstream devices first
            for downstream_device in self.downstream_devices:
                downstream_device.disconnect()

            # Reset state
            self.state = ThunderboltDeviceState.DISCONNECTED
            self.connection_time = 0.0
            self.last_auth_result = None
            self.power_negotiation = None
            self.bandwidth_usage = 0.0
            self.latency_measurements.clear()

            # Clear daisy chain connections
            if self.upstream_device:
                self.upstream_device.downstream_devices.remove(self)
                self.upstream_device = None

            self.downstream_devices.clear()
            self.chain_position = 0

            logger.info(f"Device {self.device_info.device_name} disconnected")
            return True

        except Exception as e:
            logger.error(f"Device disconnection failed: {e}")
            raise ValueError(f"Device disconnection failed: {e}")

    def authenticate(self, required_security_level: ThunderboltSecurityLevel) -> bool:
        """
        Perform Thunderbolt device authentication

        Args:
            required_security_level: Required security level

        Returns:
            True if authentication successful

        Raises:
            ValueError: If authentication fails
        """
        assert isinstance(
            required_security_level, ThunderboltSecurityLevel
        ), f"Required security level must be ThunderboltSecurityLevel, got {type(required_security_level)}"

        try:
            logger.info(f"Authenticating device with {required_security_level.name} security level...")

            self.state = ThunderboltDeviceState.AUTHENTICATING
            start_time = time.time()

            # Generate authentication challenge
            challenge = self._generate_auth_challenge()

            # Process challenge (simulate cryptographic operations)
            challenge_response = self._process_auth_challenge(challenge)

            # Validate certificate
            certificate_valid = self._validate_certificate()

            # Determine if authentication succeeds
            auth_success = self._determine_auth_success(required_security_level, certificate_valid)

            # Calculate authentication time
            auth_time = time.time() - start_time

            # Determine achieved security level
            if auth_success:
                if required_security_level == ThunderboltSecurityLevel.SECURE and certificate_valid:
                    achieved_level = ThunderboltSecurityLevel.SECURE
                elif required_security_level == ThunderboltSecurityLevel.USER:
                    achieved_level = ThunderboltSecurityLevel.USER
                elif required_security_level == ThunderboltSecurityLevel.DPONLY:
                    achieved_level = ThunderboltSecurityLevel.DPONLY
                else:
                    achieved_level = ThunderboltSecurityLevel.NONE
            else:
                achieved_level = ThunderboltSecurityLevel.NONE

            # Create authentication result
            self.last_auth_result = ThunderboltAuthResult(
                success=auth_success,
                auth_time=auth_time,
                challenge_response=challenge_response,
                certificate_valid=certificate_valid,
                security_level_achieved=achieved_level,
                error_message=None if auth_success else "Authentication failed",
            )

            if auth_success:
                logger.info(f"Authentication successful (level: {achieved_level.name})")
            else:
                logger.warning("Authentication failed")

            return auth_success

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            self.last_auth_result = ThunderboltAuthResult(
                success=False,
                auth_time=time.time() - start_time,
                challenge_response=b"",
                certificate_valid=False,
                security_level_achieved=ThunderboltSecurityLevel.NONE,
                error_message=str(e),
            )
            raise ValueError(f"Authentication failed: {e}")

    def _generate_auth_challenge(self) -> bytes:
        """Generate authentication challenge"""
        try:
            # Generate random challenge
            challenge_size = self.thunderbolt_specs.CHALLENGE_SIZE
            challenge = self.rng.bytes(challenge_size)
            return challenge

        except Exception as e:
            logger.error(f"Challenge generation failed: {e}")
            return b"\x00" * 32  # Default challenge

    def _process_auth_challenge(self, challenge: bytes) -> bytes:
        """Process authentication challenge"""
        try:
            # Simulate cryptographic processing
            # In real implementation, this would involve proper cryptography
            response = bytearray(challenge)

            # Simple XOR with private key (not secure, just for simulation)
            for i in range(len(response)):
                response[i] ^= self.private_key[i % len(self.private_key)]

            return bytes(response)

        except Exception as e:
            logger.error(f"Challenge processing failed: {e}")
            return b""

    def _validate_certificate(self) -> bool:
        """Validate device certificate"""
        try:
            # Simulate certificate validation
            # Check certificate expiry, signature, etc.

            # Simulate occasional certificate validation failures
            if self.rng.random() < 0.05:  # 5% failure rate
                return False

            # Check if certificate matches device info
            cert_valid = (
                len(self.certificate) == self.thunderbolt_specs.CERTIFICATE_SIZE
                and self.device_info.security_level != ThunderboltSecurityLevel.NONE
            )

            return cert_valid

        except Exception as e:
            logger.error(f"Certificate validation failed: {e}")
            return False

    def _determine_auth_success(self, required_level: ThunderboltSecurityLevel, cert_valid: bool) -> bool:
        """Determine if authentication should succeed"""
        try:
            # Device security level must meet or exceed required level
            device_level_value = self._get_security_level_value(self.device_info.security_level)
            required_level_value = self._get_security_level_value(required_level)

            if device_level_value < required_level_value:
                return False

            # SECURE level requires valid certificate
            if required_level == ThunderboltSecurityLevel.SECURE and not cert_valid:
                return False

            # Simulate occasional authentication failures
            if self.rng.random() < 0.02:  # 2% failure rate
                return False

            return True

        except Exception as e:
            logger.error(f"Auth success determination failed: {e}")
            return False

    def _get_security_level_value(self, level: ThunderboltSecurityLevel) -> int:
        """Get numeric value for security level comparison"""
        level_values = {
            ThunderboltSecurityLevel.NONE: 0,
            ThunderboltSecurityLevel.DPONLY: 1,
            ThunderboltSecurityLevel.USER: 2,
            ThunderboltSecurityLevel.SECURE: 3,
        }
        return level_values.get(level, 0)

    def _negotiate_power_delivery(self) -> bool:
        """Negotiate power delivery with host"""
        try:
            start_time = time.time()

            # Determine power requirements based on device type
            if self.device_info.device_type == ThunderboltDeviceType.EGPU:
                requested_power = 100.0  # eGPU needs maximum power
            elif self.device_info.device_type == ThunderboltDeviceType.DOCK:
                requested_power = 60.0  # Dock needs substantial power
            elif self.device_info.device_type == ThunderboltDeviceType.DISPLAY:
                requested_power = 15.0  # Display needs moderate power
            else:
                requested_power = self.device_info.power_consumption

            # Simulate power negotiation
            max_available_power = self.thunderbolt_specs.TB4_POWER_DELIVERY
            allocated_power = min(requested_power, max_available_power)

            # Calculate voltage and current
            if allocated_power > 60:
                voltage = 20.0  # 20V for high power
            elif allocated_power > 15:
                voltage = 12.0  # 12V for medium power
            else:
                voltage = 5.0  # 5V for low power

            current = allocated_power / voltage if voltage > 0 else 0.0

            # Determine negotiation success
            success = allocated_power >= self.device_info.power_consumption

            negotiation_time = time.time() - start_time

            self.power_negotiation = ThunderboltPowerNegotiation(
                requested_power=requested_power,
                allocated_power=allocated_power,
                voltage=voltage,
                current=current,
                negotiation_time=negotiation_time,
                success=success,
            )

            if success:
                logger.info(f"Power negotiation successful: {allocated_power}W at {voltage}V")
            else:
                logger.warning(f"Power negotiation failed: requested {requested_power}W, got {allocated_power}W")

            return success

        except Exception as e:
            logger.error(f"Power negotiation failed: {e}")
            return False

    def add_to_daisy_chain(self, upstream_device: "ThunderboltMockDevice", position: int) -> bool:
        """
        Add device to daisy chain

        Args:
            upstream_device: Upstream device in chain
            position: Position in chain (1-based)

        Returns:
            True if successfully added to chain

        Raises:
            ValueError: If daisy chain operation fails
        """
        assert isinstance(
            upstream_device, ThunderboltMockDevice
        ), f"Upstream device must be ThunderboltMockDevice, got {type(upstream_device)}"
        assert isinstance(position, int), f"Position must be int, got {type(position)}"
        assert position > 0, f"Position must be positive, got {position}"

        try:
            # Check if device supports daisy chaining
            if not self.device_info.supports_daisy_chain:
                raise ValueError("Device does not support daisy chaining")

            # Check chain length limit
            if position > self.thunderbolt_specs.MAX_DAISY_DEVICES:
                raise ValueError(f"Chain position {position} exceeds maximum {self.thunderbolt_specs.MAX_DAISY_DEVICES}")

            # Check if upstream device supports daisy chaining
            if not upstream_device.device_info.supports_daisy_chain:
                raise ValueError("Upstream device does not support daisy chaining")

            # Add to chain
            self.upstream_device = upstream_device
            self.chain_position = position
            upstream_device.downstream_devices.append(self)

            # Calculate bandwidth degradation due to daisy chaining
            degradation_factor = 1.0 - (position - 1) * self.thunderbolt_specs.CHAIN_BANDWIDTH_DEGRADATION
            self.bandwidth_usage *= degradation_factor

            logger.info(f"Device {self.device_info.device_name} added to daisy chain at position {position}")
            return True

        except Exception as e:
            logger.error(f"Daisy chain addition failed: {e}")
            raise ValueError(f"Daisy chain addition failed: {e}")

    def remove_from_daisy_chain(self) -> bool:
        """
        Remove device from daisy chain

        Returns:
            True if successfully removed from chain

        Raises:
            ValueError: If daisy chain operation fails
        """
        try:
            if self.upstream_device is None:
                logger.warning("Device not in daisy chain")
                return True

            # Remove from upstream device's downstream list
            self.upstream_device.downstream_devices.remove(self)

            # Reconnect downstream devices to upstream device
            for downstream_device in self.downstream_devices:
                downstream_device.upstream_device = self.upstream_device
                downstream_device.chain_position = self.chain_position
                self.upstream_device.downstream_devices.append(downstream_device)

            # Reset chain state
            self.upstream_device = None
            self.chain_position = 0
            self.downstream_devices.clear()

            logger.info(f"Device {self.device_info.device_name} removed from daisy chain")
            return True

        except Exception as e:
            logger.error(f"Daisy chain removal failed: {e}")
            raise ValueError(f"Daisy chain removal failed: {e}")

    def simulate_data_transfer(self, data_size: int, tunnel_mode: USB4TunnelingMode) -> Dict[str, float]:
        """
        Simulate data transfer through device

        Args:
            data_size: Size of data to transfer in bytes
            tunnel_mode: Tunneling mode to use

        Returns:
            Dictionary with transfer performance metrics

        Raises:
            ValueError: If data transfer simulation fails
        """
        assert isinstance(data_size, int), f"Data size must be int, got {type(data_size)}"
        assert isinstance(tunnel_mode, USB4TunnelingMode), f"Tunnel mode must be USB4TunnelingMode, got {type(tunnel_mode)}"
        assert data_size > 0, f"Data size must be positive, got {data_size}"

        if self.state != ThunderboltDeviceState.CONNECTED:
            raise ValueError("Device not connected")

        try:
            start_time = time.time()

            # Calculate available bandwidth considering daisy chain position
            base_bandwidth = self.device_info.max_bandwidth * 1e9  # Convert to bps
            if self.chain_position > 0:
                degradation = (self.chain_position - 1) * self.thunderbolt_specs.CHAIN_BANDWIDTH_DEGRADATION
                available_bandwidth = base_bandwidth * (1.0 - degradation)
            else:
                available_bandwidth = base_bandwidth

            # Calculate transfer time
            transfer_time = (data_size * 8) / available_bandwidth

            # Add latency based on chain position
            base_latency = 100e-9  # 100 ns base latency
            chain_latency = self.chain_position * self.thunderbolt_specs.DAISY_CHAIN_LATENCY
            total_latency = base_latency + chain_latency

            # Add protocol overhead
            protocol_overhead = self._calculate_protocol_overhead(tunnel_mode, data_size)
            total_transfer_time = transfer_time + protocol_overhead + total_latency

            # Simulate transfer
            time.sleep(min(total_transfer_time, 0.1))  # Cap simulation time

            # Calculate performance metrics
            actual_throughput = (data_size * 8) / total_transfer_time
            bandwidth_utilization = actual_throughput / available_bandwidth

            # Update device statistics
            self.bandwidth_usage = bandwidth_utilization
            self.latency_measurements.append(total_latency)

            # Keep only recent latency measurements
            if len(self.latency_measurements) > 100:
                self.latency_measurements = self.latency_measurements[-100:]

            results = {
                "transfer_time": total_transfer_time,
                "throughput_gbps": actual_throughput / 1e9,
                "bandwidth_utilization": bandwidth_utilization,
                "latency_us": total_latency * 1e6,
                "protocol_overhead": protocol_overhead,
                "chain_position": float(self.chain_position),
                "available_bandwidth_gbps": available_bandwidth / 1e9,
            }

            return results

        except Exception as e:
            logger.error(f"Data transfer simulation failed: {e}")
            raise ValueError(f"Data transfer simulation failed: {e}")

    def _calculate_protocol_overhead(self, tunnel_mode: USB4TunnelingMode, data_size: int) -> float:
        """Calculate protocol overhead time"""
        try:
            # Base USB4 overhead
            base_overhead = 32  # bytes

            # Mode-specific overhead
            if tunnel_mode == USB4TunnelingMode.PCIE:
                mode_overhead = 16
            elif tunnel_mode == USB4TunnelingMode.DISPLAYPORT:
                mode_overhead = 12
            elif tunnel_mode == USB4TunnelingMode.USB32:
                mode_overhead = 8
            else:
                mode_overhead = 4

            # Calculate overhead packets
            max_payload = 4096  # Maximum payload size
            num_packets = (data_size + max_payload - 1) // max_payload
            total_overhead_bytes = num_packets * (base_overhead + mode_overhead)

            # Convert to time
            bandwidth = self.device_info.max_bandwidth * 1e9
            overhead_time = (total_overhead_bytes * 8) / bandwidth

            return overhead_time

        except Exception as e:
            logger.warning(f"Protocol overhead calculation failed: {e}")
            return 1e-6  # Default 1 μs

    def _generate_device_certificate(self) -> bytes:
        """Generate device certificate"""
        try:
            # Generate mock certificate
            cert_size = self.thunderbolt_specs.CERTIFICATE_SIZE
            certificate = self.rng.bytes(cert_size)
            return certificate

        except Exception as e:
            logger.error(f"Certificate generation failed: {e}")
            return b"\x00" * 512  # Default certificate

    def _generate_private_key(self) -> bytes:
        """Generate device private key"""
        try:
            # Generate mock private key
            key_size = 32  # 256-bit key
            private_key = self.rng.bytes(key_size)
            return private_key

        except Exception as e:
            logger.error(f"Private key generation failed: {e}")
            return b"\x00" * 32  # Default key

    def get_device_status(self) -> Dict[str, Union[str, float, bool, int]]:
        """
        Get current device status

        Returns:
            Dictionary with device status information
        """
        try:
            status = {
                "device_id": self.device_info.device_id,
                "device_name": self.device_info.device_name,
                "device_type": self.device_info.device_type.name,
                "state": self.state.name,
                "connected": self.state == ThunderboltDeviceState.CONNECTED,
                "connection_time": self.connection_time,
                "chain_position": self.chain_position,
                "downstream_device_count": len(self.downstream_devices),
                "bandwidth_usage": self.bandwidth_usage,
                "security_level": self.device_info.security_level.name,
            }

            # Add authentication info if available
            if self.last_auth_result:
                status.update(
                    {
                        "last_auth_success": self.last_auth_result.success,
                        "last_auth_time": self.last_auth_result.auth_time,
                        "certificate_valid": self.last_auth_result.certificate_valid,
                        "achieved_security_level": self.last_auth_result.security_level_achieved.name,
                    }
                )

            # Add power info if available
            if self.power_negotiation:
                status.update(
                    {
                        "allocated_power": self.power_negotiation.allocated_power,
                        "power_voltage": self.power_negotiation.voltage,
                        "power_current": self.power_negotiation.current,
                        "power_negotiation_success": self.power_negotiation.success,
                    }
                )

            # Add latency statistics
            if self.latency_measurements:
                status.update(
                    {
                        "average_latency_us": np.mean(self.latency_measurements) * 1e6,
                        "max_latency_us": np.max(self.latency_measurements) * 1e6,
                        "latency_measurements_count": len(self.latency_measurements),
                    }
                )

            return status

        except Exception as e:
            logger.error(f"Device status retrieval failed: {e}")
            return {"error": str(e)}


# Factory functions for common device types
def create_thunderbolt_dock() -> ThunderboltMockDevice:
    """Create Thunderbolt dock device"""
    device_info = ThunderboltDeviceInfo(
        device_id=str(uuid4()),
        device_type=ThunderboltDeviceType.DOCK,
        vendor_id=0x8086,  # Intel vendor ID
        device_name="Thunderbolt 4 Dock",
        firmware_version="1.0.0",
        max_bandwidth=40.0,  # 40 Gbps
        power_consumption=60.0,  # 60W
        supports_daisy_chain=True,
        security_level=ThunderboltSecurityLevel.SECURE,
    )
    return ThunderboltMockDevice(device_info)


def create_thunderbolt_egpu() -> ThunderboltMockDevice:
    """Create Thunderbolt eGPU device"""
    device_info = ThunderboltDeviceInfo(
        device_id=str(uuid4()),
        device_type=ThunderboltDeviceType.EGPU,
        vendor_id=0x10DE,  # NVIDIA vendor ID
        device_name="Thunderbolt 4 eGPU",
        firmware_version="2.1.0",
        max_bandwidth=40.0,  # 40 Gbps
        power_consumption=100.0,  # 100W
        supports_daisy_chain=False,  # eGPUs typically don't support daisy chaining
        security_level=ThunderboltSecurityLevel.SECURE,
    )
    return ThunderboltMockDevice(device_info)


def create_thunderbolt_display() -> ThunderboltMockDevice:
    """Create Thunderbolt display device"""
    device_info = ThunderboltDeviceInfo(
        device_id=str(uuid4()),
        device_type=ThunderboltDeviceType.DISPLAY,
        vendor_id=0x1002,  # AMD vendor ID
        device_name="Thunderbolt 4 Display",
        firmware_version="1.5.2",
        max_bandwidth=25.0,  # 25 Gbps for 4K display
        power_consumption=15.0,  # 15W
        supports_daisy_chain=True,
        security_level=ThunderboltSecurityLevel.USER,
    )
    return ThunderboltMockDevice(device_info)


def create_thunderbolt_storage() -> ThunderboltMockDevice:
    """Create Thunderbolt storage device"""
    device_info = ThunderboltDeviceInfo(
        device_id=str(uuid4()),
        device_type=ThunderboltDeviceType.STORAGE,
        vendor_id=0x1058,  # Western Digital vendor ID
        device_name="Thunderbolt 4 SSD",
        firmware_version="3.0.1",
        max_bandwidth=20.0,  # 20 Gbps for high-speed storage
        power_consumption=5.0,  # 5W
        supports_daisy_chain=True,
        security_level=ThunderboltSecurityLevel.SECURE,
    )
    return ThunderboltMockDevice(device_info)


__all__ = [
    "ThunderboltDeviceType",
    "ThunderboltSecurityLevel",
    "ThunderboltDeviceState",
    "ThunderboltDeviceInfo",
    "ThunderboltAuthResult",
    "ThunderboltPowerNegotiation",
    "ThunderboltMockDevice",
    "create_thunderbolt_dock",
    "create_thunderbolt_egpu",
    "create_thunderbolt_display",
    "create_thunderbolt_storage",
]
