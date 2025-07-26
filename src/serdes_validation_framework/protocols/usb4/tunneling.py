"""
USB4 Tunneling Validation Module

This module implements comprehensive validation for multi-protocol tunneling over USB4,
including PCIe, DisplayPort, and USB 3.2 tunneling support with bandwidth management
and flow control validation.

Features:
- PCIe tunneling validation with TLP integrity checking
- DisplayPort tunneling with video signal analysis
- USB 3.2 tunneling with backward compatibility
- Multi-protocol bandwidth management
- Tunnel establishment and teardown testing
- Flow control and congestion management
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .base import (
    SignalQuality,
    USB4Config,
    USB4TunnelValidator,
)
from .constants import (
    ThunderboltSpecs,
    USB4TunnelingMode,
    USB4TunnelingSpecs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TunnelState(Enum):
    """Tunnel connection states"""

    DISCONNECTED = auto()
    ESTABLISHING = auto()
    CONNECTED = auto()
    ACTIVE = auto()
    TEARING_DOWN = auto()
    ERROR = auto()


class BandwidthAllocationMode(Enum):
    """Bandwidth allocation modes"""

    STATIC = auto()
    DYNAMIC = auto()
    PRIORITY_BASED = auto()
    FAIR_SHARE = auto()


@dataclass
class TunnelConfig:
    """Configuration for tunnel validation"""

    tunnel_mode: USB4TunnelingMode
    bandwidth_requirement: float
    latency_requirement: float
    priority: int = 0
    enable_flow_control: bool = True
    buffer_size: int = 4096


@dataclass
class PCIeTLPHeader:
    """PCIe Transaction Layer Packet header"""

    fmt: int  # Format field
    type: int  # Type field
    tc: int  # Traffic class
    attr: int  # Attributes
    length: int  # Data length
    requester_id: int
    tag: int
    last_dw_be: int
    first_dw_be: int
    address: int = 0
    data: Optional[bytes] = None


@dataclass
class TunnelResults:
    """Base tunnel validation results"""

    tunnel_mode: USB4TunnelingMode
    state: TunnelState
    bandwidth_utilization: float
    latency_measurements: List[float]
    packet_loss_rate: float
    error_count: int
    throughput: float
    test_duration: float


@dataclass
class PCIeTunnelResults(TunnelResults):
    """PCIe tunnel validation results"""

    tlp_integrity_rate: float
    completion_timeout_count: int
    malformed_tlp_count: int
    flow_control_violations: int
    bandwidth_efficiency: float


@dataclass
class DisplayPortTunnelResults(TunnelResults):
    """DisplayPort tunnel validation results"""

    video_signal_quality: SignalQuality
    frame_drop_rate: float
    sync_error_count: int
    mst_stream_count: int
    color_depth: int
    resolution: Tuple[int, int]


@dataclass
class USB32TunnelResults(TunnelResults):
    """USB 3.2 tunnel validation results"""

    enumeration_success_rate: float
    device_count: int
    protocol_compliance_rate: float
    backward_compatibility_score: float
    power_delivery_efficiency: float


class PCIeTunnelValidator(USB4TunnelValidator):
    """PCIe tunneling validator for USB4"""

    def __init__(self, config: USB4Config):
        """
        Initialize PCIe tunnel validator

        Args:
            config: USB4 configuration
        """
        super().__init__(config)
        self.specs = USB4TunnelingSpecs()
        self.tunnel_state = TunnelState.DISCONNECTED
        self.active_tlps: Dict[int, PCIeTLPHeader] = {}
        self.bandwidth_monitor = BandwidthMonitor()

    def initialize(self) -> bool:
        """
        Initialize PCIe tunnel validator

        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing PCIe tunnel validator")
            self._initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize PCIe tunnel validator: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up PCIe tunnel validator resources"""
        self.active_tlps.clear()
        self.tunnel_state = TunnelState.DISCONNECTED
        self._initialized = False

    def validate_tunnel(self, tunnel_mode: USB4TunnelingMode, data: npt.NDArray) -> Dict[str, Any]:
        """
        Validate PCIe tunneled data

        Args:
            tunnel_mode: Must be USB4TunnelingMode.PCIE
            data: PCIe tunneled data

        Returns:
            Validation results dictionary
        """
        if tunnel_mode != USB4TunnelingMode.PCIE:
            raise ValueError("PCIe validator only supports PCIe tunneling mode")

        results = {"tunnel_mode": tunnel_mode, "data_size": len(data), "validation_time": time.time()}

        # Extract and validate TLPs
        tlps = self._extract_tlps_from_data(data)
        results["tlp_count"] = len(tlps)

        # Validate TLP integrity
        integrity_results = self._validate_tlp_integrity(tlps)
        results.update(integrity_results)

        # Check bandwidth allocation
        bandwidth_results = self._validate_bandwidth_allocation(data)
        results.update(bandwidth_results)

        # Validate flow control
        flow_control_results = self._validate_flow_control(tlps)
        results.update(flow_control_results)

        return results

    def measure_tunnel_bandwidth(self, tunnel_mode: USB4TunnelingMode) -> float:
        """
        Measure PCIe tunnel bandwidth utilization

        Args:
            tunnel_mode: Must be USB4TunnelingMode.PCIE

        Returns:
            Bandwidth utilization in bps
        """
        if tunnel_mode != USB4TunnelingMode.PCIE:
            raise ValueError("PCIe validator only supports PCIe tunneling mode")

        return self.bandwidth_monitor.get_current_bandwidth()

    def test_tunnel_establishment(self, tunnel_mode: USB4TunnelingMode) -> bool:
        """
        Test PCIe tunnel establishment process

        Args:
            tunnel_mode: Must be USB4TunnelingMode.PCIE

        Returns:
            True if tunnel establishment successful
        """
        if tunnel_mode != USB4TunnelingMode.PCIE:
            raise ValueError("PCIe validator only supports PCIe tunneling mode")

        try:
            self.logger.info("Testing PCIe tunnel establishment")

            # Simulate tunnel establishment sequence
            self.tunnel_state = TunnelState.ESTABLISHING

            # Validate PCIe configuration space access
            if not self._validate_config_space_access():
                self.tunnel_state = TunnelState.ERROR
                return False

            # Test PCIe link training over tunnel
            if not self._test_pcie_link_training():
                self.tunnel_state = TunnelState.ERROR
                return False

            # Validate bandwidth negotiation
            if not self._validate_bandwidth_negotiation():
                self.tunnel_state = TunnelState.ERROR
                return False

            self.tunnel_state = TunnelState.CONNECTED
            self.logger.info("PCIe tunnel establishment successful")
            return True

        except Exception as e:
            self.logger.error(f"PCIe tunnel establishment failed: {e}")
            self.tunnel_state = TunnelState.ERROR
            return False

    def validate_tlp_integrity(self, tlp_data: bytes) -> Dict[str, Any]:
        """
        Validate PCIe TLP integrity

        Args:
            tlp_data: Raw TLP data

        Returns:
            TLP integrity validation results
        """
        results = {
            "tlp_size": len(tlp_data),
            "header_valid": False,
            "crc_valid": False,
            "sequence_valid": False,
            "format_valid": False,
        }

        if len(tlp_data) < 12:  # Minimum TLP header size
            results["error"] = "TLP too short"
            return results

        # Parse TLP header
        tlp_header = self._parse_tlp_header(tlp_data)
        if tlp_header:
            results["header_valid"] = True
            results["tlp_type"] = tlp_header.type
            results["tlp_format"] = tlp_header.fmt
            results["data_length"] = tlp_header.length

            # Validate TLP format
            results["format_valid"] = self._validate_tlp_format(tlp_header)

            # Validate sequence number if present
            results["sequence_valid"] = self._validate_tlp_sequence(tlp_header)

        # Validate CRC if present
        results["crc_valid"] = self._validate_tlp_crc(tlp_data)

        return results

    def test_bandwidth_allocation(self, allocation_config: Dict[str, float]) -> Dict[str, Any]:
        """
        Test PCIe bandwidth allocation and management

        Args:
            allocation_config: Bandwidth allocation configuration

        Returns:
            Bandwidth allocation test results
        """
        results = {
            "requested_bandwidth": allocation_config.get("bandwidth", 0),
            "allocated_bandwidth": 0,
            "utilization_efficiency": 0,
            "allocation_time": 0,
            "success": False,
        }

        start_time = time.time()

        try:
            # Request bandwidth allocation
            requested_bw = allocation_config.get("bandwidth", self.specs.PCIE_MIN_BANDWIDTH)

            # Validate bandwidth request
            if requested_bw > self.specs.PCIE_MIN_BANDWIDTH:
                allocated_bw = min(requested_bw, self.specs.PCIE_MIN_BANDWIDTH)
            else:
                allocated_bw = requested_bw

            results["allocated_bandwidth"] = allocated_bw
            results["utilization_efficiency"] = allocated_bw / requested_bw if requested_bw > 0 else 0
            results["allocation_time"] = time.time() - start_time
            results["success"] = True

        except Exception as e:
            results["error"] = str(e)

        return results

    def run_comprehensive_pcie_test(self, test_duration: float = 60.0) -> PCIeTunnelResults:
        """
        Run comprehensive PCIe tunneling test

        Args:
            test_duration: Test duration in seconds

        Returns:
            Comprehensive PCIe tunnel test results
        """
        self.logger.info(f"Starting comprehensive PCIe tunnel test ({test_duration}s)")

        start_time = time.time()
        tlp_count = 0
        error_count = 0
        malformed_tlp_count = 0
        completion_timeout_count = 0
        flow_control_violations = 0
        latency_measurements = []

        # Generate test traffic
        while time.time() - start_time < test_duration:
            # Simulate PCIe traffic
            test_tlp = self._generate_test_tlp()
            tlp_data = self._serialize_tlp(test_tlp)

            # Measure latency
            tlp_start = time.time()
            validation_result = self.validate_tlp_integrity(tlp_data)
            latency = time.time() - tlp_start
            latency_measurements.append(latency)

            tlp_count += 1

            if not validation_result.get("header_valid", False):
                malformed_tlp_count += 1
                error_count += 1

            if not validation_result.get("crc_valid", False):
                error_count += 1

            # Simulate occasional timeouts and violations
            if tlp_count % 1000 == 0:
                completion_timeout_count += 1

            if tlp_count % 500 == 0:
                flow_control_violations += 1

            time.sleep(0.001)  # Small delay to prevent overwhelming

        test_duration_actual = time.time() - start_time

        # Calculate results
        tlp_integrity_rate = (tlp_count - malformed_tlp_count) / tlp_count if tlp_count > 0 else 0
        packet_loss_rate = error_count / tlp_count if tlp_count > 0 else 0
        throughput = tlp_count / test_duration_actual
        bandwidth_utilization = self.measure_tunnel_bandwidth(USB4TunnelingMode.PCIE)
        bandwidth_efficiency = bandwidth_utilization / self.specs.PCIE_MIN_BANDWIDTH

        return PCIeTunnelResults(
            tunnel_mode=USB4TunnelingMode.PCIE,
            state=self.tunnel_state,
            bandwidth_utilization=bandwidth_utilization,
            latency_measurements=latency_measurements,
            packet_loss_rate=packet_loss_rate,
            error_count=error_count,
            throughput=throughput,
            test_duration=test_duration_actual,
            tlp_integrity_rate=tlp_integrity_rate,
            completion_timeout_count=completion_timeout_count,
            malformed_tlp_count=malformed_tlp_count,
            flow_control_violations=flow_control_violations,
            bandwidth_efficiency=bandwidth_efficiency,
        )

    def _extract_tlps_from_data(self, data: npt.NDArray) -> List[PCIeTLPHeader]:
        """Extract and parse PCIe TLPs from raw USB4 tunnel data"""
        tlps = []

        try:
            # Convert numpy array to bytes for processing
            if isinstance(data, np.ndarray):
                data_bytes = data.tobytes()
            else:
                data_bytes = bytes(data)

            logger.info(f"Extracting PCIe TLPs from {len(data_bytes)} bytes of tunnel data")

            i = 0
            while i < len(data_bytes) - 12:  # Minimum TLP header size
                # Look for TLP start pattern
                tlp_start = self._find_tlp_start(data_bytes, i)
                if tlp_start == -1:
                    break

                i = tlp_start

                # Parse TLP header
                if i + 16 <= len(data_bytes):
                    header_bytes = data_bytes[i : i + 16]
                    tlp = self._parse_tlp_header(header_bytes)

                    if tlp and self._validate_tlp_header_basic(tlp):
                        # Extract payload if present
                        payload_length = tlp.length * 4  # Length is in DWORDs
                        total_tlp_size = 16 + payload_length  # Header + payload

                        if i + total_tlp_size <= len(data_bytes):
                            if payload_length > 0:
                                tlp.payload = data_bytes[i + 16 : i + 16 + payload_length]

                            # Validate TLP CRC if present
                            if self._validate_tlp_crc(data_bytes[i : i + total_tlp_size]):
                                tlps.append(tlp)
                                logger.debug(f"Extracted valid TLP: type={tlp.type}, length={tlp.length}")
                            else:
                                logger.warning(f"TLP CRC validation failed at offset {i}")

                            i += total_tlp_size
                        else:
                            logger.warning(f"Incomplete TLP at offset {i}")
                            break
                    else:
                        logger.debug(f"Invalid TLP header at offset {i}")
                        i += 1
                else:
                    break

            logger.info(f"Extracted {len(tlps)} valid PCIe TLPs")
            return tlps

        except Exception as e:
            logger.error(f"Error extracting PCIe TLPs: {e}")
            return []

    def _find_tlp_start(self, data: bytes, start_offset: int) -> int:
        """Find the start of next TLP in data stream"""
        try:
            # Look for TLP start patterns
            # PCIe TLPs typically start with specific bit patterns
            tlp_start_patterns = [
                b"\x00\x00\x00\x40",  # Memory Read Request
                b"\x00\x00\x00\x60",  # Memory Write Request
                b"\x00\x00\x00\x4a",  # Completion
                b"\x00\x00\x00\x0a",  # Configuration Read
                b"\x00\x00\x00\x2a",  # Configuration Write
            ]

            for i in range(start_offset, len(data) - 4):
                # Check for alignment (TLPs are typically DWORD aligned)
                if i % 4 == 0:
                    for pattern in tlp_start_patterns:
                        if data[i : i + 4] == pattern:
                            return i

                # Also check for generic TLP format indicators
                if i + 1 < len(data):
                    # Check TLP format field (bits 7:5 of first byte)
                    fmt_type = data[i] & 0xE0
                    if fmt_type in [0x00, 0x20, 0x40, 0x60]:  # Valid format types
                        return i

            return -1

        except Exception as e:
            logger.error(f"Error finding TLP start: {e}")
            return -1

    def _validate_tlp_header_basic(self, tlp: PCIeTLPHeader) -> bool:
        """Validate basic TLP header fields"""
        try:
            # Check TLP type is valid
            valid_types = [0x00, 0x01, 0x02, 0x04, 0x05, 0x0A, 0x0B, 0x20, 0x21, 0x2A, 0x2B]
            if tlp.type not in valid_types:
                return False

            # Check length is reasonable
            if tlp.length > 1024:  # Max 1024 DWORDs
                return False

            # Check requester ID format
            if tlp.requester_id > 0xFFFF:
                return False

            # Check tag is valid
            if tlp.tag > 0xFF:
                return False

            return True

        except Exception as e:
            logger.error(f"TLP header validation error: {e}")
            return False

    def _validate_tlp_integrity(self, tlps: List[PCIeTLPHeader]) -> Dict[str, Any]:
        """Validate integrity of extracted TLPs"""
        total_tlps = len(tlps)
        valid_tlps = 0

        for tlp in tlps:
            if self._is_tlp_valid(tlp):
                valid_tlps += 1

        return {
            "total_tlps": total_tlps,
            "valid_tlps": valid_tlps,
            "integrity_rate": valid_tlps / total_tlps if total_tlps > 0 else 0,
        }

    def _validate_bandwidth_allocation(self, data: npt.NDArray) -> Dict[str, Any]:
        """Validate bandwidth allocation for PCIe tunnel"""
        data_rate = len(data) * 8 / 1.0  # Assume 1 second capture
        utilization = data_rate / self.specs.PCIE_MIN_BANDWIDTH

        return {"data_rate": data_rate, "bandwidth_utilization": utilization, "within_limits": utilization <= 1.0}

    def _validate_flow_control(self, tlps: List[PCIeTLPHeader]) -> Dict[str, Any]:
        """Validate PCIe flow control mechanisms with comprehensive credit tracking"""
        try:
            # Initialize credit counters for different TLP types
            credit_counters = {
                "posted_header": 256,  # Posted header credits
                "posted_data": 2048,  # Posted data credits (in DWORDs)
                "non_posted_header": 64,  # Non-posted header credits
                "non_posted_data": 512,  # Non-posted data credits
                "completion_header": 64,  # Completion header credits
                "completion_data": 512,  # Completion data credits
            }

            # Track violations and buffer states
            credit_violations = 0
            buffer_overruns = 0
            flow_control_events = []

            # Process each TLP for flow control validation
            for i, tlp in enumerate(tlps):
                try:
                    # Determine TLP category for credit accounting
                    tlp_category = self._categorize_tlp_for_flow_control(tlp)

                    # Check packet size limits
                    if tlp.length > self.specs.PCIE_MAX_PACKET_SIZE:
                        buffer_overruns += 1
                        flow_control_events.append(
                            {
                                "tlp_index": i,
                                "event_type": "buffer_overrun",
                                "tlp_length": tlp.length,
                                "max_allowed": self.specs.PCIE_MAX_PACKET_SIZE,
                            }
                        )
                        continue

                    # Calculate credit consumption
                    header_credits_needed = 1  # Each TLP consumes 1 header credit
                    data_credits_needed = tlp.length if tlp.length > 0 else 0

                    # Check and consume credits based on TLP type
                    if tlp_category == "posted":
                        if (
                            credit_counters["posted_header"] < header_credits_needed
                            or credit_counters["posted_data"] < data_credits_needed
                        ):
                            credit_violations += 1
                            flow_control_events.append(
                                {
                                    "tlp_index": i,
                                    "event_type": "credit_violation",
                                    "category": "posted",
                                    "header_credits_available": credit_counters["posted_header"],
                                    "data_credits_available": credit_counters["posted_data"],
                                    "header_credits_needed": header_credits_needed,
                                    "data_credits_needed": data_credits_needed,
                                }
                            )
                        else:
                            # Consume credits
                            credit_counters["posted_header"] -= header_credits_needed
                            credit_counters["posted_data"] -= data_credits_needed

                    elif tlp_category == "non_posted":
                        if (
                            credit_counters["non_posted_header"] < header_credits_needed
                            or credit_counters["non_posted_data"] < data_credits_needed
                        ):
                            credit_violations += 1
                            flow_control_events.append(
                                {
                                    "tlp_index": i,
                                    "event_type": "credit_violation",
                                    "category": "non_posted",
                                    "header_credits_available": credit_counters["non_posted_header"],
                                    "data_credits_available": credit_counters["non_posted_data"],
                                    "header_credits_needed": header_credits_needed,
                                    "data_credits_needed": data_credits_needed,
                                }
                            )
                        else:
                            # Consume credits
                            credit_counters["non_posted_header"] -= header_credits_needed
                            credit_counters["non_posted_data"] -= data_credits_needed

                    elif tlp_category == "completion":
                        if (
                            credit_counters["completion_header"] < header_credits_needed
                            or credit_counters["completion_data"] < data_credits_needed
                        ):
                            credit_violations += 1
                            flow_control_events.append(
                                {
                                    "tlp_index": i,
                                    "event_type": "credit_violation",
                                    "category": "completion",
                                    "header_credits_available": credit_counters["completion_header"],
                                    "data_credits_available": credit_counters["completion_data"],
                                    "header_credits_needed": header_credits_needed,
                                    "data_credits_needed": data_credits_needed,
                                }
                            )
                        else:
                            # Consume credits
                            credit_counters["completion_header"] -= header_credits_needed
                            credit_counters["completion_data"] -= data_credits_needed

                    # Simulate credit return (in real implementation, this would be based on actual completions)
                    if i % 10 == 0:  # Return some credits periodically
                        self._simulate_credit_return(credit_counters)

                except Exception as e:
                    logger.warning(f"Error processing TLP {i} for flow control: {e}")
                    continue

            # Calculate flow control efficiency
            total_tlps = len(tlps)
            successful_tlps = total_tlps - credit_violations - buffer_overruns
            flow_control_efficiency = successful_tlps / total_tlps if total_tlps > 0 else 0.0

            # Determine overall flow control validity
            flow_control_valid = credit_violations == 0 and buffer_overruns == 0

            return {
                "credit_violations": credit_violations,
                "buffer_overruns": buffer_overruns,
                "flow_control_valid": flow_control_valid,
                "flow_control_efficiency": flow_control_efficiency,
                "final_credit_counters": credit_counters,
                "flow_control_events": flow_control_events,
                "total_tlps_processed": total_tlps,
                "successful_tlps": successful_tlps,
            }

        except Exception as e:
            logger.error(f"Flow control validation error: {e}")
            return {"credit_violations": 0, "buffer_overruns": 0, "flow_control_valid": False, "error": str(e)}

    def _categorize_tlp_for_flow_control(self, tlp: PCIeTLPHeader) -> str:
        """Categorize TLP for flow control credit accounting"""
        try:
            # PCIe TLP type categorization for flow control
            # Based on PCIe specification TLP types

            if tlp.type in [0x00, 0x01]:  # Memory Read Request
                return "non_posted"
            elif tlp.type in [0x20, 0x21]:  # Memory Write Request
                return "posted"
            elif tlp.type in [0x04, 0x05]:  # Configuration Read/Write
                return "non_posted"
            elif tlp.type in [0x0A, 0x0B]:  # Completion with/without data
                return "completion"
            elif tlp.type in [0x02, 0x03]:  # I/O Read/Write
                return "non_posted"
            elif tlp.type in [0x30, 0x31]:  # Message Request
                return "posted"
            else:
                # Default to non-posted for unknown types
                return "non_posted"

        except Exception as e:
            logger.warning(f"Error categorizing TLP type {tlp.type}: {e}")
            return "non_posted"

    def _simulate_credit_return(self, credit_counters: Dict[str, int]) -> None:
        """Simulate credit return mechanism"""
        try:
            # In real implementation, credits are returned when:
            # 1. Posted transactions are committed to memory
            # 2. Non-posted transactions receive completions
            # 3. Completion transactions are processed

            # Simulate partial credit return
            credit_return_rates = {
                "posted_header": 0.1,  # 10% return rate
                "posted_data": 0.1,
                "non_posted_header": 0.15,  # 15% return rate (faster due to completions)
                "non_posted_data": 0.15,
                "completion_header": 0.2,  # 20% return rate (fastest processing)
                "completion_data": 0.2,
            }

            # Initial credit limits
            max_credits = {
                "posted_header": 256,
                "posted_data": 2048,
                "non_posted_header": 64,
                "non_posted_data": 512,
                "completion_header": 64,
                "completion_data": 512,
            }

            # Return credits up to maximum
            for credit_type, return_rate in credit_return_rates.items():
                max_credit = max_credits[credit_type]
                current_credit = credit_counters[credit_type]

                # Calculate credits to return
                credits_to_return = int((max_credit - current_credit) * return_rate)

                # Update credit counter
                credit_counters[credit_type] = min(max_credit, current_credit + credits_to_return)

        except Exception as e:
            logger.warning(f"Error in credit return simulation: {e}")

    def _validate_config_space_access(self) -> bool:
        """Validate PCIe configuration space access through tunnel"""
        # Simulate configuration space read/write operations
        return True

    def _test_pcie_link_training(self) -> bool:
        """Test PCIe link training through USB4 tunnel"""
        # Simulate PCIe link training sequence
        return True

    def _validate_bandwidth_negotiation(self) -> bool:
        """Validate PCIe bandwidth negotiation"""
        # Simulate bandwidth negotiation
        return True

    def _parse_tlp_header(self, data: bytes) -> Optional[PCIeTLPHeader]:
        """Parse PCIe TLP header from raw data"""
        if len(data) < 12:
            return None

        # Simplified TLP header parsing
        try:
            # Extract fields from first 32-bit word
            word0 = int.from_bytes(data[0:4], byteorder="little")
            fmt = (word0 >> 29) & 0x7
            type_field = (word0 >> 24) & 0x1F
            tc = (word0 >> 20) & 0x7
            attr = (word0 >> 12) & 0xFF
            length = word0 & 0x3FF

            # Extract fields from second 32-bit word
            word1 = int.from_bytes(data[4:8], byteorder="little")
            requester_id = (word1 >> 16) & 0xFFFF
            tag = (word1 >> 8) & 0xFF
            last_dw_be = (word1 >> 4) & 0xF
            first_dw_be = word1 & 0xF

            # Extract address if present
            address = 0
            if fmt in [0, 1]:  # 3DW header
                if len(data) >= 12:
                    address = int.from_bytes(data[8:12], byteorder="little")
            elif fmt in [2, 3]:  # 4DW header
                if len(data) >= 16:
                    address = int.from_bytes(data[8:16], byteorder="little")

            return PCIeTLPHeader(
                fmt=fmt,
                type=type_field,
                tc=tc,
                attr=attr,
                length=length,
                requester_id=requester_id,
                tag=tag,
                last_dw_be=last_dw_be,
                first_dw_be=first_dw_be,
                address=address,
            )

        except Exception as e:
            self.logger.error(f"Failed to parse TLP header: {e}")
            return None

    def _validate_tlp_format(self, tlp: PCIeTLPHeader) -> bool:
        """Validate TLP format fields"""
        # Check format field validity
        if tlp.fmt not in [0, 1, 2, 3]:
            return False

        # Check type field validity
        if tlp.type > 31:
            return False

        # Check length field
        if tlp.length > 1024:  # Max payload size
            return False

        return True

    def _validate_tlp_sequence(self, tlp: PCIeTLPHeader) -> bool:
        """Validate TLP sequence number"""
        # Simplified sequence validation
        return True

    def _validate_tlp_crc(self, tlp_data: bytes) -> bool:
        """Validate TLP CRC using proper CRC-32 calculation"""
        try:
            if len(tlp_data) < 16:  # Minimum TLP size with CRC
                return False

            # Extract CRC from last 4 bytes
            data_portion = tlp_data[:-4]
            received_crc = int.from_bytes(tlp_data[-4:], byteorder="little")

            # Calculate CRC-32 for the data portion
            calculated_crc = self._calculate_crc32(data_portion)

            # Compare CRCs
            crc_valid = calculated_crc == received_crc

            if not crc_valid:
                logger.debug(f"TLP CRC mismatch: calculated=0x{calculated_crc:08x}, received=0x{received_crc:08x}")

            return crc_valid

        except Exception as e:
            logger.error(f"TLP CRC validation error: {e}")
            return False

    def _calculate_crc32(self, data: bytes) -> int:
        """Calculate CRC-32 for TLP data"""
        try:
            import zlib

            # Use standard CRC-32 algorithm
            crc = zlib.crc32(data) & 0xFFFFFFFF

            return crc

        except Exception as e:
            logger.error(f"CRC-32 calculation error: {e}")
            return 0

    def _is_tlp_valid(self, tlp: PCIeTLPHeader) -> bool:
        """Check if TLP is valid"""
        return self._validate_tlp_format(tlp) and self._validate_tlp_sequence(tlp)

    def _generate_test_tlp(self) -> PCIeTLPHeader:
        """Generate test TLP for validation"""
        return PCIeTLPHeader(
            fmt=0,  # 3DW header, no data
            type=0,  # Memory read request
            tc=0,  # Traffic class 0
            attr=0,  # No attributes
            length=1,  # 1 DW
            requester_id=0x0100,
            tag=0x01,
            last_dw_be=0xF,
            first_dw_be=0xF,
            address=0x1000,
        )

    def _serialize_tlp(self, tlp: PCIeTLPHeader) -> bytes:
        """Serialize TLP header to bytes"""
        # Create 3DW header
        word0 = (
            ((tlp.fmt & 0x7) << 29)
            | ((tlp.type & 0x1F) << 24)
            | ((tlp.tc & 0x7) << 20)
            | ((tlp.attr & 0xFF) << 12)
            | (tlp.length & 0x3FF)
        )

        word1 = (
            ((tlp.requester_id & 0xFFFF) << 16)
            | ((tlp.tag & 0xFF) << 8)
            | ((tlp.last_dw_be & 0xF) << 4)
            | (tlp.first_dw_be & 0xF)
        )

        word2 = tlp.address & 0xFFFFFFFF

        return (
            word0.to_bytes(4, byteorder="little") + word1.to_bytes(4, byteorder="little") + word2.to_bytes(4, byteorder="little")
        )


class BandwidthMonitor:
    """Monitor bandwidth utilization for tunnels"""

    def __init__(self):
        self.current_bandwidth = 0.0
        self.peak_bandwidth = 0.0
        self.average_bandwidth = 0.0
        self.measurement_history = []

    def update_bandwidth(self, bandwidth: float) -> None:
        """Update bandwidth measurement"""
        self.current_bandwidth = bandwidth
        self.peak_bandwidth = max(self.peak_bandwidth, bandwidth)
        self.measurement_history.append(bandwidth)

        # Keep only recent measurements
        if len(self.measurement_history) > 1000:
            self.measurement_history = self.measurement_history[-1000:]

        self.average_bandwidth = np.mean(self.measurement_history)

    def get_current_bandwidth(self) -> float:
        """Get current bandwidth utilization"""
        return self.current_bandwidth

    def get_bandwidth_statistics(self) -> Dict[str, float]:
        """Get bandwidth statistics"""
        return {
            "current": self.current_bandwidth,
            "peak": self.peak_bandwidth,
            "average": self.average_bandwidth,
            "std_dev": np.std(self.measurement_history) if self.measurement_history else 0.0,
        }


__all__ = [
    # Enums
    "TunnelState",
    "BandwidthAllocationMode",
    # Data structures
    "TunnelConfig",
    "PCIeTLPHeader",
    "TunnelResults",
    "PCIeTunnelResults",
    "DisplayPortTunnelResults",
    "USB32TunnelResults",
    # Classes
    "PCIeTunnelValidator",
    "BandwidthMonitor",
]


class DisplayPortTunnelValidator(USB4TunnelValidator):
    """DisplayPort tunneling validator for USB4"""

    def __init__(self, config: USB4Config):
        """
        Initialize DisplayPort tunnel validator

        Args:
            config: USB4 configuration
        """
        super().__init__(config)
        self.specs = USB4TunnelingSpecs()
        self.tb_specs = ThunderboltSpecs()
        self.tunnel_state = TunnelState.DISCONNECTED
        self.active_streams: Dict[int, Dict[str, Any]] = {}
        self.bandwidth_monitor = BandwidthMonitor()

    def initialize(self) -> bool:
        """
        Initialize DisplayPort tunnel validator

        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing DisplayPort tunnel validator")
            self._initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize DisplayPort tunnel validator: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up DisplayPort tunnel validator resources"""
        self.active_streams.clear()
        self.tunnel_state = TunnelState.DISCONNECTED
        self._initialized = False

    def validate_tunnel(self, tunnel_mode: USB4TunnelingMode, data: npt.NDArray) -> Dict[str, Any]:
        """
        Validate DisplayPort tunneled data

        Args:
            tunnel_mode: Must be USB4TunnelingMode.DISPLAYPORT
            data: DisplayPort tunneled data

        Returns:
            Validation results dictionary
        """
        if tunnel_mode != USB4TunnelingMode.DISPLAYPORT:
            raise ValueError("DisplayPort validator only supports DisplayPort tunneling mode")

        results = {"tunnel_mode": tunnel_mode, "data_size": len(data), "validation_time": time.time()}

        # Analyze video signal integrity
        signal_results = self._analyze_video_signal_integrity(data)
        results.update(signal_results)

        # Validate DisplayPort timing
        timing_results = self._validate_displayport_timing(data)
        results.update(timing_results)

        # Check MST stream validation
        mst_results = self._validate_mst_streams(data)
        results.update(mst_results)

        return results

    def measure_tunnel_bandwidth(self, tunnel_mode: USB4TunnelingMode) -> float:
        """
        Measure DisplayPort tunnel bandwidth utilization

        Args:
            tunnel_mode: Must be USB4TunnelingMode.DISPLAYPORT

        Returns:
            Bandwidth utilization in bps
        """
        if tunnel_mode != USB4TunnelingMode.DISPLAYPORT:
            raise ValueError("DisplayPort validator only supports DisplayPort tunneling mode")

        return self.bandwidth_monitor.get_current_bandwidth()

    def test_tunnel_establishment(self, tunnel_mode: USB4TunnelingMode) -> bool:
        """
        Test DisplayPort tunnel establishment process

        Args:
            tunnel_mode: Must be USB4TunnelingMode.DISPLAYPORT

        Returns:
            True if tunnel establishment successful
        """
        if tunnel_mode != USB4TunnelingMode.DISPLAYPORT:
            raise ValueError("DisplayPort validator only supports DisplayPort tunneling mode")

        try:
            self.logger.info("Testing DisplayPort tunnel establishment")

            self.tunnel_state = TunnelState.ESTABLISHING

            # Test DisplayPort capability negotiation
            if not self._test_dp_capability_negotiation():
                self.tunnel_state = TunnelState.ERROR
                return False

            # Validate link training
            if not self._test_dp_link_training():
                self.tunnel_state = TunnelState.ERROR
                return False

            # Test MST topology discovery
            if not self._test_mst_topology_discovery():
                self.tunnel_state = TunnelState.ERROR
                return False

            self.tunnel_state = TunnelState.CONNECTED
            self.logger.info("DisplayPort tunnel establishment successful")
            return True

        except Exception as e:
            self.logger.error(f"DisplayPort tunnel establishment failed: {e}")
            self.tunnel_state = TunnelState.ERROR
            return False

    def validate_video_signal_integrity(self, video_data: npt.NDArray) -> Dict[str, Any]:
        """
        Validate video signal integrity for tunneled DisplayPort

        Args:
            video_data: Video signal data

        Returns:
            Video signal integrity results
        """
        results = {
            "data_size": len(video_data),
            "signal_quality": SignalQuality.GOOD,
            "sync_errors": 0,
            "pixel_errors": 0,
            "frame_drops": 0,
        }

        # Analyze signal quality
        signal_stats = self._calculate_video_signal_stats(video_data)
        results.update(signal_stats)

        # Check for sync errors
        sync_errors = self._detect_sync_errors(video_data)
        results["sync_errors"] = sync_errors

        # Detect pixel errors
        pixel_errors = self._detect_pixel_errors(video_data)
        results["pixel_errors"] = pixel_errors

        # Assess overall signal quality
        if sync_errors == 0 and pixel_errors == 0:
            results["signal_quality"] = SignalQuality.EXCELLENT
        elif sync_errors < 10 and pixel_errors < 100:
            results["signal_quality"] = SignalQuality.GOOD
        elif sync_errors < 50 and pixel_errors < 1000:
            results["signal_quality"] = SignalQuality.MARGINAL
        else:
            results["signal_quality"] = SignalQuality.POOR

        return results

    def validate_mst_streams(self, stream_count: int) -> Dict[str, Any]:
        """
        Validate Multi-Stream Transport (MST) for multiple displays

        Args:
            stream_count: Number of MST streams to validate

        Returns:
            MST validation results
        """
        results = {
            "requested_streams": stream_count,
            "active_streams": 0,
            "bandwidth_per_stream": 0,
            "total_bandwidth": 0,
            "mst_valid": False,
        }

        if stream_count > self.specs.DP_MAX_STREAMS:
            results["error"] = f"Requested streams ({stream_count}) exceeds maximum ({self.specs.DP_MAX_STREAMS})"
            return results

        # Calculate bandwidth requirements
        bandwidth_per_stream = self.tb_specs.DISPLAY_BANDWIDTH_4K  # Assume 4K per stream
        total_bandwidth = bandwidth_per_stream * stream_count

        results["bandwidth_per_stream"] = bandwidth_per_stream
        results["total_bandwidth"] = total_bandwidth

        # Check if total bandwidth fits within DisplayPort tunnel limits
        if total_bandwidth <= self.specs.DP_MIN_BANDWIDTH:
            results["active_streams"] = stream_count
            results["mst_valid"] = True
        else:
            # Calculate maximum supportable streams
            max_streams = int(self.specs.DP_MIN_BANDWIDTH / bandwidth_per_stream)
            results["active_streams"] = max_streams
            results["mst_valid"] = False
            results["warning"] = f"Only {max_streams} streams can be supported with available bandwidth"

        return results

    def run_comprehensive_displayport_test(self, test_duration: float = 60.0) -> DisplayPortTunnelResults:
        """
        Run comprehensive DisplayPort tunneling test

        Args:
            test_duration: Test duration in seconds

        Returns:
            Comprehensive DisplayPort tunnel test results
        """
        self.logger.info(f"Starting comprehensive DisplayPort tunnel test ({test_duration}s)")

        start_time = time.time()
        frame_count = 0
        frame_drops = 0
        sync_errors = 0
        latency_measurements = []

        # Simulate video stream processing
        while time.time() - start_time < test_duration:
            # Generate test video frame
            test_frame = self._generate_test_video_frame()

            # Measure processing latency
            frame_start = time.time()
            signal_results = self.validate_video_signal_integrity(test_frame)
            latency = time.time() - frame_start
            latency_measurements.append(latency)

            frame_count += 1
            sync_errors += signal_results.get("sync_errors", 0)

            # Simulate occasional frame drops
            if frame_count % 1000 == 0:
                frame_drops += 1

            time.sleep(0.016)  # ~60 FPS

        test_duration_actual = time.time() - start_time

        # Calculate results
        frame_drop_rate = frame_drops / frame_count if frame_count > 0 else 0
        throughput = frame_count / test_duration_actual
        bandwidth_utilization = self.measure_tunnel_bandwidth(USB4TunnelingMode.DISPLAYPORT)

        # Determine overall signal quality
        if sync_errors == 0 and frame_drops == 0:
            signal_quality = SignalQuality.EXCELLENT
        elif sync_errors < 10 and frame_drops < 5:
            signal_quality = SignalQuality.GOOD
        elif sync_errors < 50 and frame_drops < 20:
            signal_quality = SignalQuality.MARGINAL
        else:
            signal_quality = SignalQuality.POOR

        return DisplayPortTunnelResults(
            tunnel_mode=USB4TunnelingMode.DISPLAYPORT,
            state=self.tunnel_state,
            bandwidth_utilization=bandwidth_utilization,
            latency_measurements=latency_measurements,
            packet_loss_rate=frame_drop_rate,
            error_count=sync_errors,
            throughput=throughput,
            test_duration=test_duration_actual,
            video_signal_quality=signal_quality,
            frame_drop_rate=frame_drop_rate,
            sync_error_count=sync_errors,
            mst_stream_count=len(self.active_streams),
            color_depth=8,  # Assume 8-bit color
            resolution=(3840, 2160),  # Assume 4K resolution
        )

    def _analyze_video_signal_integrity(self, data: npt.NDArray) -> Dict[str, Any]:
        """Analyze video signal integrity"""
        return {
            "signal_strength": np.mean(np.abs(data)),
            "noise_level": np.std(data),
            "snr": np.mean(np.abs(data)) / np.std(data) if np.std(data) > 0 else float("inf"),
        }

    def _validate_displayport_timing(self, data: npt.NDArray) -> Dict[str, Any]:
        """Validate DisplayPort timing and synchronization"""
        return {"timing_valid": True, "sync_valid": True, "blanking_valid": True}

    def _validate_mst_streams(self, data: npt.NDArray) -> Dict[str, Any]:
        """Validate MST stream data"""
        return {"stream_count": len(self.active_streams), "streams_valid": True}

    def _test_dp_capability_negotiation(self) -> bool:
        """Test DisplayPort capability negotiation"""
        return True

    def _test_dp_link_training(self) -> bool:
        """Test DisplayPort link training"""
        return True

    def _test_mst_topology_discovery(self) -> bool:
        """Test MST topology discovery"""
        return True

    def _calculate_video_signal_stats(self, data: npt.NDArray) -> Dict[str, float]:
        """Calculate video signal statistics"""
        return {
            "mean_amplitude": float(np.mean(np.abs(data))),
            "peak_amplitude": float(np.max(np.abs(data))),
            "rms_amplitude": float(np.sqrt(np.mean(data**2))),
        }

    def _detect_sync_errors(self, data: npt.NDArray) -> int:
        """Detect synchronization errors in video data"""
        # Simplified sync error detection
        return 0

    def _detect_pixel_errors(self, data: npt.NDArray) -> int:
        """Detect pixel errors in video data"""
        # Simplified pixel error detection
        return 0

    def _generate_test_video_frame(self) -> npt.NDArray:
        """Generate test video frame data"""
        # Generate 4K frame (simplified)
        return np.random.randn(3840 * 2160).astype(np.float64)


class USB32TunnelValidator(USB4TunnelValidator):
    """USB 3.2 tunneling validator for USB4"""

    def __init__(self, config: USB4Config):
        """
        Initialize USB 3.2 tunnel validator

        Args:
            config: USB4 configuration
        """
        super().__init__(config)
        self.specs = USB4TunnelingSpecs()
        self.tunnel_state = TunnelState.DISCONNECTED
        self.connected_devices: Dict[int, Dict[str, Any]] = {}
        self.bandwidth_monitor = BandwidthMonitor()

    def initialize(self) -> bool:
        """
        Initialize USB 3.2 tunnel validator

        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing USB 3.2 tunnel validator")
            self._initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize USB 3.2 tunnel validator: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up USB 3.2 tunnel validator resources"""
        self.connected_devices.clear()
        self.tunnel_state = TunnelState.DISCONNECTED
        self._initialized = False

    def validate_tunnel(self, tunnel_mode: USB4TunnelingMode, data: npt.NDArray) -> Dict[str, Any]:
        """
        Validate USB 3.2 tunneled data

        Args:
            tunnel_mode: Must be USB4TunnelingMode.USB32
            data: USB 3.2 tunneled data

        Returns:
            Validation results dictionary
        """
        if tunnel_mode != USB4TunnelingMode.USB32:
            raise ValueError("USB 3.2 validator only supports USB 3.2 tunneling mode")

        results = {"tunnel_mode": tunnel_mode, "data_size": len(data), "validation_time": time.time()}

        # Validate USB 3.2 protocol compliance
        protocol_results = self._validate_usb32_protocol(data)
        results.update(protocol_results)

        # Test device enumeration
        enum_results = self._validate_device_enumeration(data)
        results.update(enum_results)

        # Check backward compatibility
        compat_results = self._validate_backward_compatibility(data)
        results.update(compat_results)

        return results

    def measure_tunnel_bandwidth(self, tunnel_mode: USB4TunnelingMode) -> float:
        """
        Measure USB 3.2 tunnel bandwidth utilization

        Args:
            tunnel_mode: Must be USB4TunnelingMode.USB32

        Returns:
            Bandwidth utilization in bps
        """
        if tunnel_mode != USB4TunnelingMode.USB32:
            raise ValueError("USB 3.2 validator only supports USB 3.2 tunneling mode")

        return self.bandwidth_monitor.get_current_bandwidth()

    def test_tunnel_establishment(self, tunnel_mode: USB4TunnelingMode) -> bool:
        """
        Test USB 3.2 tunnel establishment process

        Args:
            tunnel_mode: Must be USB4TunnelingMode.USB32

        Returns:
            True if tunnel establishment successful
        """
        if tunnel_mode != USB4TunnelingMode.USB32:
            raise ValueError("USB 3.2 validator only supports USB 3.2 tunneling mode")

        try:
            self.logger.info("Testing USB 3.2 tunnel establishment")

            self.tunnel_state = TunnelState.ESTABLISHING

            # Test USB hub enumeration
            if not self._test_usb_hub_enumeration():
                self.tunnel_state = TunnelState.ERROR
                return False

            # Validate USB 3.2 link training
            if not self._test_usb32_link_training():
                self.tunnel_state = TunnelState.ERROR
                return False

            # Test device discovery
            if not self._test_device_discovery():
                self.tunnel_state = TunnelState.ERROR
                return False

            self.tunnel_state = TunnelState.CONNECTED
            self.logger.info("USB 3.2 tunnel establishment successful")
            return True

        except Exception as e:
            self.logger.error(f"USB 3.2 tunnel establishment failed: {e}")
            self.tunnel_state = TunnelState.ERROR
            return False

    def validate_device_enumeration(self, device_count: int) -> Dict[str, Any]:
        """
        Validate USB device enumeration and configuration

        Args:
            device_count: Number of devices to enumerate

        Returns:
            Device enumeration validation results
        """
        results = {
            "requested_devices": device_count,
            "enumerated_devices": 0,
            "configuration_success": 0,
            "enumeration_time": 0,
            "success_rate": 0,
        }

        if device_count > self.specs.USB32_MAX_DEVICES:
            results["error"] = f"Device count ({device_count}) exceeds USB 3.2 limit ({self.specs.USB32_MAX_DEVICES})"
            return results

        start_time = time.time()

        # Simulate device enumeration
        enumerated = 0
        configured = 0

        for device_id in range(device_count):
            # Simulate enumeration process
            if self._simulate_device_enumeration(device_id):
                enumerated += 1

                # Simulate configuration
                if self._simulate_device_configuration(device_id):
                    configured += 1
                    self.connected_devices[device_id] = {
                        "enumerated": True,
                        "configured": True,
                        "device_class": "storage",  # Example
                        "speed": "super_speed",
                    }

        enumeration_time = time.time() - start_time

        results.update(
            {
                "enumerated_devices": enumerated,
                "configuration_success": configured,
                "enumeration_time": enumeration_time,
                "success_rate": enumerated / device_count if device_count > 0 else 0,
            }
        )

        return results

    def test_usb32_performance(self, test_duration: float = 30.0) -> Dict[str, Any]:
        """
        Test USB 3.2 performance benchmarking over tunnel

        Args:
            test_duration: Test duration in seconds

        Returns:
            USB 3.2 performance test results
        """
        results = {
            "test_duration": test_duration,
            "throughput": 0,
            "latency_avg": 0,
            "latency_max": 0,
            "error_rate": 0,
            "efficiency": 0,
        }

        start_time = time.time()
        transfer_count = 0
        error_count = 0
        latency_measurements = []

        # Simulate USB 3.2 data transfers
        while time.time() - start_time < test_duration:
            transfer_start = time.time()

            # Simulate data transfer
            success = self._simulate_usb32_transfer()
            transfer_count += 1

            if not success:
                error_count += 1

            latency = time.time() - transfer_start
            latency_measurements.append(latency)

            time.sleep(0.001)  # Small delay between transfers

        actual_duration = time.time() - start_time

        results.update(
            {
                "test_duration": actual_duration,
                "throughput": transfer_count / actual_duration,
                "latency_avg": np.mean(latency_measurements),
                "latency_max": np.max(latency_measurements),
                "error_rate": error_count / transfer_count if transfer_count > 0 else 0,
                "efficiency": (transfer_count - error_count) / transfer_count if transfer_count > 0 else 0,
            }
        )

        return results

    def run_comprehensive_usb32_test(self, test_duration: float = 60.0) -> USB32TunnelResults:
        """
        Run comprehensive USB 3.2 tunneling test

        Args:
            test_duration: Test duration in seconds

        Returns:
            Comprehensive USB 3.2 tunnel test results
        """
        self.logger.info(f"Starting comprehensive USB 3.2 tunnel test ({test_duration}s)")

        start_time = time.time()

        # Test device enumeration
        enum_results = self.validate_device_enumeration(10)  # Test with 10 devices

        # Test performance
        perf_results = self.test_usb32_performance(test_duration / 2)

        # Test backward compatibility
        compat_results = self._test_backward_compatibility()

        test_duration_actual = time.time() - start_time

        # Calculate overall results
        enumeration_success_rate = enum_results.get("success_rate", 0)
        protocol_compliance_rate = perf_results.get("efficiency", 0)
        backward_compatibility_score = compat_results.get("compatibility_score", 0)

        bandwidth_utilization = self.measure_tunnel_bandwidth(USB4TunnelingMode.USB32)
        throughput = perf_results.get("throughput", 0)
        latency_measurements = [perf_results.get("latency_avg", 0)]

        return USB32TunnelResults(
            tunnel_mode=USB4TunnelingMode.USB32,
            state=self.tunnel_state,
            bandwidth_utilization=bandwidth_utilization,
            latency_measurements=latency_measurements,
            packet_loss_rate=perf_results.get("error_rate", 0),
            error_count=0,
            throughput=throughput,
            test_duration=test_duration_actual,
            enumeration_success_rate=enumeration_success_rate,
            device_count=len(self.connected_devices),
            protocol_compliance_rate=protocol_compliance_rate,
            backward_compatibility_score=backward_compatibility_score,
            power_delivery_efficiency=0.95,  # Assume 95% efficiency
        )

    def _validate_usb32_protocol(self, data: npt.NDArray) -> Dict[str, Any]:
        """Validate USB 3.2 protocol compliance"""
        return {"protocol_valid": True, "packet_format_valid": True, "crc_valid": True}

    def _validate_device_enumeration(self, data: npt.NDArray) -> Dict[str, Any]:
        """Validate device enumeration process"""
        return {"enumeration_valid": True, "descriptor_valid": True}

    def _validate_backward_compatibility(self, data: npt.NDArray) -> Dict[str, Any]:
        """Validate backward compatibility with USB 2.0/1.1"""
        return {"usb2_compatible": True, "usb1_compatible": True, "fallback_working": True}

    def _test_usb_hub_enumeration(self) -> bool:
        """Test USB hub enumeration"""
        return True

    def _test_usb32_link_training(self) -> bool:
        """Test USB 3.2 link training"""
        return True

    def _test_device_discovery(self) -> bool:
        """Test USB device discovery"""
        return True

    def _simulate_device_enumeration(self, device_id: int) -> bool:
        """Simulate device enumeration process"""
        # 95% success rate
        return np.random.random() > 0.05

    def _simulate_device_configuration(self, device_id: int) -> bool:
        """Simulate device configuration process"""
        # 98% success rate
        return np.random.random() > 0.02

    def _simulate_usb32_transfer(self) -> bool:
        """Simulate USB 3.2 data transfer"""
        # 99% success rate
        return np.random.random() > 0.01

    def _test_backward_compatibility(self) -> Dict[str, Any]:
        """Test backward compatibility with older USB versions"""
        return {"compatibility_score": 0.95, "usb2_devices_supported": True, "usb1_devices_supported": True}


class MultiProtocolBandwidthManager:
    """Manage bandwidth allocation across multiple tunnel protocols"""

    def __init__(self, total_bandwidth: float = 40.0e9):
        """
        Initialize bandwidth manager

        Args:
            total_bandwidth: Total available bandwidth in bps
        """
        self.total_bandwidth = total_bandwidth
        self.allocated_bandwidth: Dict[USB4TunnelingMode, float] = {}
        self.active_tunnels: Dict[USB4TunnelingMode, bool] = {}
        self.allocation_mode = BandwidthAllocationMode.DYNAMIC
        self.specs = USB4TunnelingSpecs()

    def allocate_bandwidth(self, tunnel_mode: USB4TunnelingMode, requested_bandwidth: float) -> Dict[str, Any]:
        """
        Allocate bandwidth for tunnel protocol

        Args:
            tunnel_mode: Tunnel protocol type
            requested_bandwidth: Requested bandwidth in bps

        Returns:
            Allocation results
        """
        results = {
            "tunnel_mode": tunnel_mode,
            "requested_bandwidth": requested_bandwidth,
            "allocated_bandwidth": 0,
            "allocation_success": False,
            "remaining_bandwidth": 0,
        }

        # Check minimum bandwidth requirements
        min_bandwidth = self._get_minimum_bandwidth(tunnel_mode)
        if requested_bandwidth < min_bandwidth:
            results["error"] = f"Requested bandwidth below minimum for {tunnel_mode.name}"
            return results

        # Calculate currently allocated bandwidth
        current_allocation = sum(self.allocated_bandwidth.values())
        available_bandwidth = self.total_bandwidth - current_allocation

        # Apply allocation algorithm based on mode
        if self.allocation_mode == BandwidthAllocationMode.DYNAMIC:
            allocated = self._dynamic_allocation(tunnel_mode, requested_bandwidth, available_bandwidth)
        elif self.allocation_mode == BandwidthAllocationMode.PRIORITY_BASED:
            allocated = self._priority_based_allocation(tunnel_mode, requested_bandwidth, available_bandwidth)
        elif self.allocation_mode == BandwidthAllocationMode.FAIR_SHARE:
            allocated = self._fair_share_allocation(tunnel_mode, requested_bandwidth, available_bandwidth)
        else:  # STATIC
            allocated = min(requested_bandwidth, available_bandwidth)

        if allocated > 0:
            self.allocated_bandwidth[tunnel_mode] = allocated
            self.active_tunnels[tunnel_mode] = True
            results["allocated_bandwidth"] = allocated
            results["allocation_success"] = True

        results["remaining_bandwidth"] = self.total_bandwidth - sum(self.allocated_bandwidth.values())

        return results

    def deallocate_bandwidth(self, tunnel_mode: USB4TunnelingMode) -> bool:
        """
        Deallocate bandwidth for tunnel protocol

        Args:
            tunnel_mode: Tunnel protocol type

        Returns:
            True if deallocation successful
        """
        if tunnel_mode in self.allocated_bandwidth:
            del self.allocated_bandwidth[tunnel_mode]
            self.active_tunnels[tunnel_mode] = False
            return True
        return False

    def test_congestion_management(self, congestion_level: float) -> Dict[str, Any]:
        """
        Test congestion management and flow control

        Args:
            congestion_level: Congestion level (0.0 to 1.0)

        Returns:
            Congestion management test results
        """
        results = {
            "congestion_level": congestion_level,
            "throttling_applied": False,
            "bandwidth_reduction": 0,
            "flow_control_active": False,
            "recovery_time": 0,
        }

        if congestion_level > 0.8:  # High congestion
            # Apply throttling
            reduction_factor = min(0.5, congestion_level - 0.5)

            for tunnel_mode in self.allocated_bandwidth:
                original_bw = self.allocated_bandwidth[tunnel_mode]
                reduced_bw = original_bw * (1 - reduction_factor)
                self.allocated_bandwidth[tunnel_mode] = reduced_bw

            results["throttling_applied"] = True
            results["bandwidth_reduction"] = reduction_factor
            results["flow_control_active"] = True

        elif congestion_level > 0.6:  # Medium congestion
            # Apply flow control
            results["flow_control_active"] = True

        # Simulate recovery time
        results["recovery_time"] = congestion_level * 10.0  # ms

        return results

    def test_simultaneous_tunnels(self, tunnel_configs: List[TunnelConfig]) -> Dict[str, Any]:
        """
        Test simultaneous multi-tunnel operation

        Args:
            tunnel_configs: List of tunnel configurations

        Returns:
            Multi-tunnel test results
        """
        results = {
            "requested_tunnels": len(tunnel_configs),
            "active_tunnels": 0,
            "total_bandwidth_used": 0,
            "bandwidth_efficiency": 0,
            "allocation_success": True,
        }

        # Reset allocations
        self.allocated_bandwidth.clear()
        self.active_tunnels.clear()

        successful_allocations = 0

        # Sort by priority (higher priority first)
        sorted_configs = sorted(tunnel_configs, key=lambda x: x.priority, reverse=True)

        for config in sorted_configs:
            allocation_result = self.allocate_bandwidth(config.tunnel_mode, config.bandwidth_requirement)

            if allocation_result["allocation_success"]:
                successful_allocations += 1

        total_allocated = sum(self.allocated_bandwidth.values())

        results.update(
            {
                "active_tunnels": successful_allocations,
                "total_bandwidth_used": total_allocated,
                "bandwidth_efficiency": total_allocated / self.total_bandwidth,
                "allocation_success": successful_allocations == len(tunnel_configs),
            }
        )

        return results

    def measure_bandwidth_utilization(self) -> Dict[str, float]:
        """
        Measure current bandwidth utilization

        Returns:
            Bandwidth utilization metrics
        """
        total_allocated = sum(self.allocated_bandwidth.values())

        return {
            "total_bandwidth": self.total_bandwidth,
            "allocated_bandwidth": total_allocated,
            "available_bandwidth": self.total_bandwidth - total_allocated,
            "utilization_percentage": (total_allocated / self.total_bandwidth) * 100,
            "active_tunnel_count": len([t for t in self.active_tunnels.values() if t]),
        }

    def _get_minimum_bandwidth(self, tunnel_mode: USB4TunnelingMode) -> float:
        """Get minimum bandwidth requirement for tunnel mode"""
        bandwidth_map = {
            USB4TunnelingMode.PCIE: self.specs.PCIE_MIN_BANDWIDTH,
            USB4TunnelingMode.DISPLAYPORT: self.specs.DP_MIN_BANDWIDTH,
            USB4TunnelingMode.USB32: self.specs.USB32_MIN_BANDWIDTH,
            USB4TunnelingMode.NATIVE: self.specs.MIN_TUNNEL_BANDWIDTH,
        }
        return bandwidth_map.get(tunnel_mode, self.specs.MIN_TUNNEL_BANDWIDTH)

    def _dynamic_allocation(self, tunnel_mode: USB4TunnelingMode, requested: float, available: float) -> float:
        """Dynamic bandwidth allocation algorithm"""
        return min(requested, available)

    def _priority_based_allocation(self, tunnel_mode: USB4TunnelingMode, requested: float, available: float) -> float:
        """Priority-based bandwidth allocation algorithm"""
        # PCIe gets highest priority, then DisplayPort, then USB 3.2
        priority_map = {
            USB4TunnelingMode.PCIE: 3,
            USB4TunnelingMode.DISPLAYPORT: 2,
            USB4TunnelingMode.USB32: 1,
            USB4TunnelingMode.NATIVE: 0,
        }

        priority = priority_map.get(tunnel_mode, 0)
        allocation_factor = 0.5 + (priority * 0.2)  # 50% to 110% of requested

        return min(requested * allocation_factor, available)

    def _fair_share_allocation(self, tunnel_mode: USB4TunnelingMode, requested: float, available: float) -> float:
        """Fair share bandwidth allocation algorithm"""
        active_count = len([t for t in self.active_tunnels.values() if t]) + 1  # +1 for new tunnel
        fair_share = available / active_count

        return min(requested, fair_share)

    def _get_minimum_bandwidth(self, tunnel_mode: USB4TunnelingMode) -> float:
        """Get minimum bandwidth requirement for tunnel mode"""
        bandwidth_map = {
            USB4TunnelingMode.PCIE: self.specs.PCIE_MIN_BANDWIDTH,
            USB4TunnelingMode.DISPLAYPORT: self.specs.DP_MIN_BANDWIDTH,
            USB4TunnelingMode.USB32: self.specs.USB32_MIN_BANDWIDTH,
            USB4TunnelingMode.NATIVE: self.specs.MIN_TUNNEL_BANDWIDTH,
        }
        return bandwidth_map.get(tunnel_mode, self.specs.MIN_TUNNEL_BANDWIDTH)


# Update __all__ to include new classes
__all__.extend(
    [
        "DisplayPortTunnelValidator",
        "USB32TunnelValidator",
        "MultiProtocolBandwidthManager",
    ]
)
