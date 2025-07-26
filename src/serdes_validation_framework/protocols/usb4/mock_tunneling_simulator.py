"""
USB4 Tunneling Protocol Simulation Module

This module provides simulation of USB4 multi-protocol tunneling for testing
without physical tunneling hardware. Simulates PCIe, DisplayPort, and USB 3.2
tunneling over USB4.

Features:
- PCIe tunneling simulation
- DisplayPort tunneling simulation
- USB 3.2 tunneling simulation
- Bandwidth allocation simulation
- Protocol encapsulation/decapsulation
- Latency and performance modeling
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from .constants import USB4Specs, USB4TunnelingMode, USB4TunnelingSpecs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USB4TunnelState(Enum):
    """USB4 tunnel states"""

    IDLE = auto()
    ESTABLISHING = auto()
    ACTIVE = auto()
    TEARING_DOWN = auto()
    ERROR = auto()


class USB4PacketType(Enum):
    """USB4 packet types"""

    CONTROL = auto()
    DATA = auto()
    CREDIT = auto()
    ERROR = auto()


@dataclass
class USB4TunnelConfig:
    """USB4 tunnel configuration"""

    tunnel_mode: USB4TunnelingMode
    bandwidth_allocation: float  # Gbps
    max_packet_size: int
    buffer_size: int
    enable_flow_control: bool = True
    enable_error_correction: bool = True
    latency_target: float = 1e-6  # 1 μs

    def __post_init__(self) -> None:
        """Validate tunnel configuration"""
        assert isinstance(
            self.tunnel_mode, USB4TunnelingMode
        ), f"Tunnel mode must be USB4TunnelingMode, got {type(self.tunnel_mode)}"
        assert isinstance(
            self.bandwidth_allocation, float
        ), f"Bandwidth allocation must be float, got {type(self.bandwidth_allocation)}"
        assert isinstance(self.max_packet_size, int), f"Max packet size must be int, got {type(self.max_packet_size)}"
        assert isinstance(self.buffer_size, int), f"Buffer size must be int, got {type(self.buffer_size)}"
        assert isinstance(self.latency_target, float), f"Latency target must be float, got {type(self.latency_target)}"

        assert self.bandwidth_allocation > 0, f"Bandwidth allocation must be positive, got {self.bandwidth_allocation}"
        assert self.max_packet_size > 0, f"Max packet size must be positive, got {self.max_packet_size}"
        assert self.buffer_size > 0, f"Buffer size must be positive, got {self.buffer_size}"
        assert self.latency_target > 0, f"Latency target must be positive, got {self.latency_target}"


@dataclass
class USB4TunnelPacket:
    """USB4 tunnel packet"""

    packet_id: int
    packet_type: USB4PacketType
    tunnel_id: int
    payload_size: int
    payload_data: npt.NDArray[np.uint8]
    timestamp: float
    sequence_number: int
    metadata: Dict[str, Union[str, int, float]]

    def __post_init__(self) -> None:
        """Validate tunnel packet"""
        assert isinstance(self.packet_id, int), f"Packet ID must be int, got {type(self.packet_id)}"
        assert isinstance(self.packet_type, USB4PacketType), f"Packet type must be USB4PacketType, got {type(self.packet_type)}"
        assert isinstance(self.tunnel_id, int), f"Tunnel ID must be int, got {type(self.tunnel_id)}"
        assert isinstance(self.payload_size, int), f"Payload size must be int, got {type(self.payload_size)}"
        assert isinstance(self.payload_data, np.ndarray), f"Payload data must be numpy array, got {type(self.payload_data)}"
        assert isinstance(self.timestamp, float), f"Timestamp must be float, got {type(self.timestamp)}"
        assert isinstance(self.sequence_number, int), f"Sequence number must be int, got {type(self.sequence_number)}"
        assert isinstance(self.metadata, dict), f"Metadata must be dict, got {type(self.metadata)}"

        assert self.packet_id >= 0, f"Packet ID must be non-negative, got {self.packet_id}"
        assert self.tunnel_id >= 0, f"Tunnel ID must be non-negative, got {self.tunnel_id}"
        assert self.payload_size >= 0, f"Payload size must be non-negative, got {self.payload_size}"
        assert len(self.payload_data) == self.payload_size, "Payload data size mismatch"
        assert self.sequence_number >= 0, f"Sequence number must be non-negative, got {self.sequence_number}"


@dataclass
class USB4TunnelStats:
    """USB4 tunnel statistics"""

    packets_sent: int
    packets_received: int
    bytes_sent: int
    bytes_received: int
    packet_loss_rate: float
    average_latency: float
    bandwidth_utilization: float
    error_count: int

    def __post_init__(self) -> None:
        """Validate tunnel statistics"""
        assert isinstance(self.packets_sent, int), f"Packets sent must be int, got {type(self.packets_sent)}"
        assert isinstance(self.packets_received, int), f"Packets received must be int, got {type(self.packets_received)}"
        assert isinstance(self.bytes_sent, int), f"Bytes sent must be int, got {type(self.bytes_sent)}"
        assert isinstance(self.bytes_received, int), f"Bytes received must be int, got {type(self.bytes_received)}"
        assert isinstance(self.packet_loss_rate, float), f"Packet loss rate must be float, got {type(self.packet_loss_rate)}"
        assert isinstance(self.average_latency, float), f"Average latency must be float, got {type(self.average_latency)}"
        assert isinstance(
            self.bandwidth_utilization, float
        ), f"Bandwidth utilization must be float, got {type(self.bandwidth_utilization)}"
        assert isinstance(self.error_count, int), f"Error count must be int, got {type(self.error_count)}"

        assert self.packets_sent >= 0, f"Packets sent must be non-negative, got {self.packets_sent}"
        assert self.packets_received >= 0, f"Packets received must be non-negative, got {self.packets_received}"
        assert self.bytes_sent >= 0, f"Bytes sent must be non-negative, got {self.bytes_sent}"
        assert self.bytes_received >= 0, f"Bytes received must be non-negative, got {self.bytes_received}"
        assert 0 <= self.packet_loss_rate <= 1, f"Packet loss rate must be between 0 and 1, got {self.packet_loss_rate}"
        assert self.average_latency >= 0, f"Average latency must be non-negative, got {self.average_latency}"
        assert (
            0 <= self.bandwidth_utilization <= 1
        ), f"Bandwidth utilization must be between 0 and 1, got {self.bandwidth_utilization}"
        assert self.error_count >= 0, f"Error count must be non-negative, got {self.error_count}"


class USB4TunnelingSimulator:
    """USB4 tunneling protocol simulator"""

    def __init__(self, config: USB4TunnelConfig) -> None:
        """
        Initialize USB4 tunneling simulator

        Args:
            config: Tunnel configuration

        Raises:
            ValueError: If initialization fails
        """
        assert isinstance(config, USB4TunnelConfig), f"Config must be USB4TunnelConfig, got {type(config)}"

        self.config = config
        self.usb4_specs = USB4Specs()
        self.tunneling_specs = USB4TunnelingSpecs()

        # Tunnel state
        self.tunnel_state = USB4TunnelState.IDLE
        self.tunnel_id = 0
        self.next_packet_id = 0
        self.next_sequence_number = 0

        # Packet buffers
        self.tx_buffer: List[USB4TunnelPacket] = []
        self.rx_buffer: List[USB4TunnelPacket] = []

        # Statistics
        self.stats = USB4TunnelStats(
            packets_sent=0,
            packets_received=0,
            bytes_sent=0,
            bytes_received=0,
            packet_loss_rate=0.0,
            average_latency=0.0,
            bandwidth_utilization=0.0,
            error_count=0,
        )

        # Flow control
        self.credits_available = self.tunneling_specs.CREDIT_LIMIT
        self.credits_consumed = 0

        # Random number generator for simulation
        self.rng = np.random.RandomState(42)

        logger.info(f"USB4 tunneling simulator initialized for {config.tunnel_mode.name}")

    def establish_tunnel(self, timeout: float = 5.0) -> bool:
        """
        Establish USB4 tunnel

        Args:
            timeout: Establishment timeout in seconds

        Returns:
            True if tunnel established successfully

        Raises:
            ValueError: If tunnel establishment fails
        """
        assert isinstance(timeout, float), f"Timeout must be float, got {type(timeout)}"
        assert timeout > 0, f"Timeout must be positive, got {timeout}"

        try:
            logger.info(f"Establishing {self.config.tunnel_mode.name} tunnel...")

            self.tunnel_state = USB4TunnelState.ESTABLISHING
            start_time = time.time()

            # Simulate tunnel establishment process
            establishment_steps = [
                ("Requesting tunnel allocation", 0.1),
                ("Negotiating bandwidth", 0.2),
                ("Configuring flow control", 0.1),
                ("Validating tunnel parameters", 0.1),
                ("Activating tunnel", 0.1),
            ]

            for step_name, step_duration in establishment_steps:
                logger.debug(f"Tunnel establishment: {step_name}")
                time.sleep(step_duration)

                # Check timeout
                if time.time() - start_time > timeout:
                    self.tunnel_state = USB4TunnelState.ERROR
                    raise ValueError("Tunnel establishment timeout")

                # Simulate occasional establishment failures
                if self.rng.random() < 0.01:  # 1% failure rate
                    self.tunnel_state = USB4TunnelState.ERROR
                    raise ValueError(f"Tunnel establishment failed at: {step_name}")

            # Tunnel established successfully
            self.tunnel_state = USB4TunnelState.ACTIVE
            self.tunnel_id = self.rng.randint(1, 256)  # Assign tunnel ID

            logger.info(f"Tunnel established successfully (ID: {self.tunnel_id})")
            return True

        except Exception as e:
            logger.error(f"Tunnel establishment failed: {e}")
            self.tunnel_state = USB4TunnelState.ERROR
            raise ValueError(f"Tunnel establishment failed: {e}")

    def tear_down_tunnel(self) -> bool:
        """
        Tear down USB4 tunnel

        Returns:
            True if tunnel torn down successfully

        Raises:
            ValueError: If tunnel tear down fails
        """
        try:
            if self.tunnel_state != USB4TunnelState.ACTIVE:
                logger.warning("Tunnel not active, cannot tear down")
                return False

            logger.info(f"Tearing down tunnel {self.tunnel_id}...")

            self.tunnel_state = USB4TunnelState.TEARING_DOWN

            # Flush buffers
            self.tx_buffer.clear()
            self.rx_buffer.clear()

            # Reset state
            self.tunnel_state = USB4TunnelState.IDLE
            self.tunnel_id = 0
            self.next_packet_id = 0
            self.next_sequence_number = 0
            self.credits_available = self.tunneling_specs.CREDIT_LIMIT
            self.credits_consumed = 0

            logger.info("Tunnel torn down successfully")
            return True

        except Exception as e:
            logger.error(f"Tunnel tear down failed: {e}")
            self.tunnel_state = USB4TunnelState.ERROR
            raise ValueError(f"Tunnel tear down failed: {e}")

    def send_data(self, data: npt.NDArray[np.uint8]) -> bool:
        """
        Send data through tunnel

        Args:
            data: Data to send

        Returns:
            True if data sent successfully

        Raises:
            ValueError: If data transmission fails
        """
        assert isinstance(data, np.ndarray), f"Data must be numpy array, got {type(data)}"
        assert data.dtype == np.uint8, f"Data must be uint8 array, got {data.dtype}"

        if self.tunnel_state != USB4TunnelState.ACTIVE:
            raise ValueError("Tunnel not active")

        try:
            # Fragment data into packets if necessary
            packets = self._fragment_data(data)

            # Send each packet
            for packet_data in packets:
                if not self._send_packet(packet_data):
                    return False

            return True

        except Exception as e:
            logger.error(f"Data transmission failed: {e}")
            raise ValueError(f"Data transmission failed: {e}")

    def receive_data(self, timeout: float = 1.0) -> Optional[npt.NDArray[np.uint8]]:
        """
        Receive data from tunnel

        Args:
            timeout: Receive timeout in seconds

        Returns:
            Received data or None if timeout

        Raises:
            ValueError: If data reception fails
        """
        assert isinstance(timeout, float), f"Timeout must be float, got {type(timeout)}"
        assert timeout > 0, f"Timeout must be positive, got {timeout}"

        if self.tunnel_state != USB4TunnelState.ACTIVE:
            raise ValueError("Tunnel not active")

        try:
            start_time = time.time()
            received_packets = []

            # Wait for packets
            while time.time() - start_time < timeout:
                if self.rx_buffer:
                    packet = self.rx_buffer.pop(0)
                    received_packets.append(packet)

                    # Check if we have a complete message
                    if self._is_complete_message(received_packets):
                        break

                time.sleep(0.001)  # Small delay

            if not received_packets:
                return None

            # Reassemble data from packets
            return self._reassemble_data(received_packets)

        except Exception as e:
            logger.error(f"Data reception failed: {e}")
            raise ValueError(f"Data reception failed: {e}")

    def _fragment_data(self, data: npt.NDArray[np.uint8]) -> List[npt.NDArray[np.uint8]]:
        """Fragment data into packets"""
        try:
            max_payload = self.config.max_packet_size - 32  # Account for headers
            fragments = []

            for i in range(0, len(data), max_payload):
                fragment = data[i : i + max_payload]
                fragments.append(fragment)

            return fragments

        except Exception as e:
            raise ValueError(f"Data fragmentation failed: {e}")

    def _send_packet(self, data: npt.NDArray[np.uint8]) -> bool:
        """Send a single packet"""
        try:
            # Check flow control
            if not self._check_flow_control():
                logger.warning("Flow control limit reached")
                return False

            # Create packet
            packet = USB4TunnelPacket(
                packet_id=self.next_packet_id,
                packet_type=USB4PacketType.DATA,
                tunnel_id=self.tunnel_id,
                payload_size=len(data),
                payload_data=data,
                timestamp=time.time(),
                sequence_number=self.next_sequence_number,
                metadata={"tunnel_mode": self.config.tunnel_mode.name, "fragmented": len(data) < self.config.max_packet_size},
            )

            # Simulate transmission
            success = self._simulate_transmission(packet)

            if success:
                self.tx_buffer.append(packet)
                self.next_packet_id += 1
                self.next_sequence_number += 1
                self.stats.packets_sent += 1
                self.stats.bytes_sent += len(data)

                # Consume credits
                if self.config.enable_flow_control:
                    self.credits_consumed += 1

                # Simulate packet arrival at receiver
                self._simulate_packet_arrival(packet)

            return success

        except Exception as e:
            logger.error(f"Packet transmission failed: {e}")
            return False

    def _check_flow_control(self) -> bool:
        """Check flow control credits"""
        if not self.config.enable_flow_control:
            return True

        return self.credits_consumed < self.credits_available

    def _simulate_transmission(self, packet: USB4TunnelPacket) -> bool:
        """Simulate packet transmission with realistic behavior"""
        try:
            # Simulate transmission delay based on bandwidth
            transmission_time = packet.payload_size * 8 / (self.config.bandwidth_allocation * 1e9)

            # Add protocol overhead
            protocol_overhead = self._calculate_protocol_overhead(packet)
            transmission_time += protocol_overhead

            # Simulate occasional packet loss
            loss_probability = self._calculate_loss_probability()
            if self.rng.random() < loss_probability:
                logger.debug(f"Packet {packet.packet_id} lost during transmission")
                return False

            # Simulate transmission errors
            if self.config.enable_error_correction:
                error_probability = 1e-9  # Very low with error correction
            else:
                error_probability = 1e-6  # Higher without error correction

            if self.rng.random() < error_probability:
                logger.debug(f"Packet {packet.packet_id} corrupted during transmission")
                self.stats.error_count += 1
                return False

            # Update latency statistics
            self._update_latency_stats(transmission_time)

            return True

        except Exception as e:
            logger.error(f"Transmission simulation failed: {e}")
            return False

    def _calculate_protocol_overhead(self, packet: USB4TunnelPacket) -> float:
        """Calculate protocol overhead time"""
        try:
            # Base overhead for USB4 tunneling
            base_overhead = 32  # bytes

            # Mode-specific overhead
            if self.config.tunnel_mode == USB4TunnelingMode.PCIE:
                mode_overhead = 16  # PCIe TLP overhead
            elif self.config.tunnel_mode == USB4TunnelingMode.DISPLAYPORT:
                mode_overhead = 12  # DisplayPort overhead
            elif self.config.tunnel_mode == USB4TunnelingMode.USB32:
                mode_overhead = 8  # USB 3.2 overhead
            else:
                mode_overhead = 4  # Native USB4

            total_overhead_bytes = base_overhead + mode_overhead
            overhead_time = total_overhead_bytes * 8 / (self.config.bandwidth_allocation * 1e9)

            return overhead_time

        except Exception as e:
            logger.warning(f"Protocol overhead calculation failed: {e}")
            return 1e-9  # Default 1 ns

    def _calculate_loss_probability(self) -> float:
        """Calculate packet loss probability based on conditions"""
        try:
            base_loss_rate = 1e-12  # Very low base loss rate

            # Increase loss rate based on buffer utilization
            buffer_utilization = len(self.tx_buffer) / self.config.buffer_size
            utilization_factor = buffer_utilization**2

            # Increase loss rate based on bandwidth utilization
            bandwidth_factor = self.stats.bandwidth_utilization**3

            total_loss_rate = base_loss_rate * (1 + utilization_factor + bandwidth_factor)

            return min(total_loss_rate, 1e-6)  # Cap at reasonable maximum

        except Exception as e:
            logger.warning(f"Loss probability calculation failed: {e}")
            return 1e-12  # Default very low loss rate

    def _update_latency_stats(self, transmission_time: float) -> None:
        """Update latency statistics"""
        try:
            # Simple exponential moving average
            alpha = 0.1
            if self.stats.average_latency == 0:
                self.stats.average_latency = transmission_time
            else:
                self.stats.average_latency = alpha * transmission_time + (1 - alpha) * self.stats.average_latency

        except Exception as e:
            logger.warning(f"Latency stats update failed: {e}")

    def _simulate_packet_arrival(self, packet: USB4TunnelPacket) -> None:
        """Simulate packet arrival at receiver"""
        try:
            # Simulate network delay
            network_delay = self.rng.exponential(self.config.latency_target)

            # Schedule packet arrival (simplified - just add to rx buffer)
            rx_packet = USB4TunnelPacket(
                packet_id=packet.packet_id,
                packet_type=packet.packet_type,
                tunnel_id=packet.tunnel_id,
                payload_size=packet.payload_size,
                payload_data=packet.payload_data.copy(),
                timestamp=packet.timestamp + network_delay,
                sequence_number=packet.sequence_number,
                metadata=packet.metadata.copy(),
            )

            self.rx_buffer.append(rx_packet)
            self.stats.packets_received += 1
            self.stats.bytes_received += packet.payload_size

        except Exception as e:
            logger.warning(f"Packet arrival simulation failed: {e}")

    def _is_complete_message(self, packets: List[USB4TunnelPacket]) -> bool:
        """Check if received packets form a complete message"""
        try:
            if not packets:
                return False

            # Simple check: if last packet is not fragmented, message is complete
            last_packet = packets[-1]
            is_fragmented = last_packet.metadata.get("fragmented", False)

            return not is_fragmented

        except Exception as e:
            logger.warning(f"Complete message check failed: {e}")
            return True  # Assume complete on error

    def _reassemble_data(self, packets: List[USB4TunnelPacket]) -> npt.NDArray[np.uint8]:
        """Reassemble data from packets"""
        try:
            # Sort packets by sequence number
            sorted_packets = sorted(packets, key=lambda p: p.sequence_number)

            # Concatenate payload data
            data_parts = [packet.payload_data for packet in sorted_packets]
            reassembled_data = np.concatenate(data_parts)

            return reassembled_data

        except Exception as e:
            logger.error(f"Data reassembly failed: {e}")
            raise ValueError(f"Data reassembly failed: {e}")

    def simulate_protocol_specific_behavior(
        self, data: npt.NDArray[np.uint8], duration: float
    ) -> Dict[str, Union[float, bool, int]]:
        """
        Simulate protocol-specific tunneling behavior

        Args:
            data: Protocol data to tunnel
            duration: Simulation duration in seconds

        Returns:
            Dictionary with protocol-specific results

        Raises:
            ValueError: If simulation fails
        """
        assert isinstance(data, np.ndarray), f"Data must be numpy array, got {type(data)}"
        assert isinstance(duration, float), f"Duration must be float, got {type(duration)}"
        assert duration > 0, f"Duration must be positive, got {duration}"

        try:
            if self.config.tunnel_mode == USB4TunnelingMode.PCIE:
                return self._simulate_pcie_tunneling(data, duration)
            elif self.config.tunnel_mode == USB4TunnelingMode.DISPLAYPORT:
                return self._simulate_displayport_tunneling(data, duration)
            elif self.config.tunnel_mode == USB4TunnelingMode.USB32:
                return self._simulate_usb32_tunneling(data, duration)
            else:
                return self._simulate_native_usb4(data, duration)

        except Exception as e:
            logger.error(f"Protocol-specific simulation failed: {e}")
            raise ValueError(f"Protocol-specific simulation failed: {e}")

    def _simulate_pcie_tunneling(self, data: npt.NDArray[np.uint8], duration: float) -> Dict[str, Union[float, bool, int]]:
        """Simulate PCIe tunneling over USB4"""
        try:
            results = {}

            # PCIe-specific parameters
            tlp_overhead = 16  # Transaction Layer Packet overhead
            max_payload_size = 256  # Typical PCIe max payload

            # Calculate PCIe metrics
            num_tlps = len(data) // max_payload_size + (1 if len(data) % max_payload_size else 0)
            total_overhead = num_tlps * tlp_overhead
            efficiency = len(data) / (len(data) + total_overhead)

            # Simulate PCIe latency characteristics
            base_latency = 100e-9  # 100 ns base PCIe latency
            tunneling_latency = self.config.latency_target
            total_latency = base_latency + tunneling_latency

            # Simulate PCIe bandwidth utilization
            pcie_bandwidth = min(self.config.bandwidth_allocation, self.tunneling_specs.PCIE_MIN_BANDWIDTH)
            utilization = (len(data) * 8) / (pcie_bandwidth * duration)

            results = {
                "protocol": "PCIe",
                "num_tlps": num_tlps,
                "efficiency": efficiency,
                "total_latency": total_latency,
                "bandwidth_utilization": min(utilization, 1.0),
                "max_payload_size": max_payload_size,
                "overhead_bytes": total_overhead,
                "compliance_check": total_latency <= self.tunneling_specs.PCIE_MAX_LATENCY,
            }

            return results

        except Exception as e:
            raise ValueError(f"PCIe tunneling simulation failed: {e}")

    def _simulate_displayport_tunneling(self, data: npt.NDArray[np.uint8], duration: float) -> Dict[str, Union[float, bool, int]]:
        """Simulate DisplayPort tunneling over USB4"""
        try:
            results = {}

            # DisplayPort-specific parameters
            mst_overhead = 12  # Multi-Stream Transport overhead
            video_frame_size = 1920 * 1080 * 3  # 1080p RGB frame

            # Calculate DisplayPort metrics
            num_frames = len(data) // video_frame_size + (1 if len(data) % video_frame_size else 0)
            total_overhead = num_frames * mst_overhead
            efficiency = len(data) / (len(data) + total_overhead)

            # Simulate DisplayPort latency characteristics
            frame_latency = 16.67e-3  # 60 Hz frame time
            tunneling_latency = self.config.latency_target
            total_latency = tunneling_latency  # Frame latency is separate

            # Simulate DisplayPort bandwidth requirements
            dp_bandwidth = min(self.config.bandwidth_allocation, self.tunneling_specs.DP_MIN_BANDWIDTH)
            utilization = (len(data) * 8) / (dp_bandwidth * duration)

            results = {
                "protocol": "DisplayPort",
                "num_frames": num_frames,
                "efficiency": efficiency,
                "total_latency": total_latency,
                "frame_latency": frame_latency,
                "bandwidth_utilization": min(utilization, 1.0),
                "video_frame_size": video_frame_size,
                "overhead_bytes": total_overhead,
                "compliance_check": total_latency <= self.tunneling_specs.DP_MAX_LATENCY,
            }

            return results

        except Exception as e:
            raise ValueError(f"DisplayPort tunneling simulation failed: {e}")

    def _simulate_usb32_tunneling(self, data: npt.NDArray[np.uint8], duration: float) -> Dict[str, Union[float, bool, int]]:
        """Simulate USB 3.2 tunneling over USB4"""
        try:
            results = {}

            # USB 3.2 specific parameters
            usb_overhead = 8  # USB packet overhead
            max_packet_size = 1024  # USB 3.2 max packet size

            # Calculate USB 3.2 metrics
            num_packets = len(data) // max_packet_size + (1 if len(data) % max_packet_size else 0)
            total_overhead = num_packets * usb_overhead
            efficiency = len(data) / (len(data) + total_overhead)

            # Simulate USB 3.2 latency characteristics
            usb_latency = 1e-6  # 1 μs typical USB latency
            tunneling_latency = self.config.latency_target
            total_latency = usb_latency + tunneling_latency

            # Simulate USB 3.2 bandwidth utilization
            usb_bandwidth = min(self.config.bandwidth_allocation, self.tunneling_specs.USB32_MIN_BANDWIDTH)
            utilization = (len(data) * 8) / (usb_bandwidth * duration)

            results = {
                "protocol": "USB 3.2",
                "num_packets": num_packets,
                "efficiency": efficiency,
                "total_latency": total_latency,
                "bandwidth_utilization": min(utilization, 1.0),
                "max_packet_size": max_packet_size,
                "overhead_bytes": total_overhead,
                "compliance_check": total_latency <= self.tunneling_specs.USB32_MAX_LATENCY,
            }

            return results

        except Exception as e:
            raise ValueError(f"USB 3.2 tunneling simulation failed: {e}")

    def _simulate_native_usb4(self, data: npt.NDArray[np.uint8], duration: float) -> Dict[str, Union[float, bool, int]]:
        """Simulate native USB4 protocol"""
        try:
            results = {}

            # Native USB4 parameters
            usb4_overhead = 4  # Minimal overhead for native protocol
            max_packet_size = self.config.max_packet_size

            # Calculate native USB4 metrics
            num_packets = len(data) // max_packet_size + (1 if len(data) % max_packet_size else 0)
            total_overhead = num_packets * usb4_overhead
            efficiency = len(data) / (len(data) + total_overhead)

            # Native USB4 has minimal additional latency
            total_latency = self.config.latency_target

            # Full bandwidth utilization possible
            utilization = (len(data) * 8) / (self.config.bandwidth_allocation * duration)

            results = {
                "protocol": "Native USB4",
                "num_packets": num_packets,
                "efficiency": efficiency,
                "total_latency": total_latency,
                "bandwidth_utilization": min(utilization, 1.0),
                "max_packet_size": max_packet_size,
                "overhead_bytes": total_overhead,
                "compliance_check": True,  # Native protocol always compliant
            }

            return results

        except Exception as e:
            raise ValueError(f"Native USB4 simulation failed: {e}")

    def get_tunnel_statistics(self) -> USB4TunnelStats:
        """
        Get current tunnel statistics

        Returns:
            Current tunnel statistics
        """
        try:
            # Update bandwidth utilization
            if self.stats.packets_sent > 0:
                total_time = time.time() - (self.tx_buffer[0].timestamp if self.tx_buffer else time.time())
                if total_time > 0:
                    throughput = (self.stats.bytes_sent * 8) / total_time
                    self.stats.bandwidth_utilization = throughput / (self.config.bandwidth_allocation * 1e9)

            # Update packet loss rate
            if self.stats.packets_sent > 0:
                self.stats.packet_loss_rate = 1.0 - (self.stats.packets_received / self.stats.packets_sent)

            return self.stats

        except Exception as e:
            logger.error(f"Statistics retrieval failed: {e}")
            return self.stats

    def reset_statistics(self) -> None:
        """Reset tunnel statistics"""
        self.stats = USB4TunnelStats(
            packets_sent=0,
            packets_received=0,
            bytes_sent=0,
            bytes_received=0,
            packet_loss_rate=0.0,
            average_latency=0.0,
            bandwidth_utilization=0.0,
            error_count=0,
        )
        logger.info("Tunnel statistics reset")


__all__ = [
    "USB4TunnelState",
    "USB4PacketType",
    "USB4TunnelConfig",
    "USB4TunnelPacket",
    "USB4TunnelStats",
    "USB4TunnelingSimulator",
]
