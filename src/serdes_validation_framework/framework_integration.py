"""
Framework Integration Module

This module provides the main integration point for all protocols and test sequences
in the SerDes Validation Framework. It handles protocol detection, test sequence
creation, and provides a unified interface for validation workflows.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from .protocol_detector import ProtocolDetector, ProtocolType
from .protocols import SUPPORTED_PROTOCOLS
from .protocols.usb4 import USB4LinkState, USB4TunnelingMode
from .protocols.usb4.constants import USB4SignalMode
from .test_sequence import (
    PCIeTestSequence,
    USB4LaneConfig,
    USB4TestPhase,
    USB4TestSequence,
    USB4TestSequenceConfig,
)

logger = logging.getLogger(__name__)


class ValidationFramework:
    """
    Main SerDes Validation Framework class

    This class provides a unified interface for protocol detection,
    test sequence creation, and validation workflows across all
    supported protocols.
    """

    def __init__(self) -> None:
        """Initialize the validation framework"""
        self.protocol_detector = ProtocolDetector()
        self.supported_protocols = SUPPORTED_PROTOCOLS
        logger.info("SerDes Validation Framework initialized")
        logger.info(f"Supported protocols: {list(self.supported_protocols.keys())}")

    def detect_protocol(self, signal_data: npt.NDArray[np.float64], sample_rate: float, voltage_range: float) -> ProtocolType:
        """
        Detect protocol from signal characteristics

        Args:
            signal_data: Signal voltage data
            sample_rate: Sampling rate in Hz
            voltage_range: Signal voltage range in V

        Returns:
            Detected protocol type
        """
        return self.protocol_detector.detect_protocol_from_signal(signal_data, sample_rate, voltage_range)

    def select_protocol(self, protocol_name: str) -> ProtocolType:
        """
        Select protocol by name

        Args:
            protocol_name: Name of the protocol

        Returns:
            Selected protocol type
        """
        return self.protocol_detector.select_protocol_by_name(protocol_name)

    def get_protocol_config(self, protocol_type: ProtocolType) -> Dict[str, Any]:
        """
        Get configuration for specified protocol

        Args:
            protocol_type: Protocol type

        Returns:
            Protocol configuration dictionary
        """
        return self.protocol_detector.get_protocol_config(protocol_type)

    def create_test_sequence(
        self, protocol_type: ProtocolType, test_config: Optional[Dict[str, Any]] = None
    ) -> Union[PCIeTestSequence, USB4TestSequence, Any]:
        """
        Create test sequence for specified protocol

        Args:
            protocol_type: Protocol type
            test_config: Optional test configuration

        Returns:
            Test sequence instance
        """
        if protocol_type == ProtocolType.USB4 or protocol_type == ProtocolType.THUNDERBOLT4:
            return self._create_usb4_test_sequence(test_config, protocol_type == ProtocolType.THUNDERBOLT4)
        elif protocol_type == ProtocolType.PCIE:
            return self._create_pcie_test_sequence(test_config)
        elif protocol_type == ProtocolType.ETHERNET_224G:
            return self._create_ethernet_test_sequence(test_config)
        else:
            raise ValueError(f"Unsupported protocol type: {protocol_type}")

    def run_auto_validation(
        self,
        signal_data: Union[npt.NDArray[np.float64], Dict[int, Dict[str, npt.NDArray[np.float64]]]],
        sample_rate: float,
        voltage_range: float,
        protocol_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run automatic validation with protocol detection

        Args:
            signal_data: Signal data (single array or multi-lane dictionary)
            sample_rate: Sampling rate in Hz
            voltage_range: Signal voltage range in V
            protocol_hint: Optional protocol hint

        Returns:
            Validation results
        """
        logger.info("Starting automatic validation")

        # Determine protocol
        if protocol_hint:
            protocol_type = self.select_protocol(protocol_hint)
        else:
            # Use first signal for detection if multi-lane
            detection_signal = signal_data if isinstance(signal_data, np.ndarray) else list(signal_data.values())[0]["voltage"]
            protocol_type = self.detect_protocol(detection_signal, sample_rate, voltage_range)

        logger.info(f"Using protocol: {protocol_type.name}")

        # Create appropriate test sequence
        test_sequence = self.create_test_sequence(protocol_type)

        # Format signal data appropriately
        formatted_signal_data = self._format_signal_data(signal_data, protocol_type)

        # Run validation
        results = test_sequence.run_complete_sequence(formatted_signal_data)

        return {
            "protocol_type": protocol_type.name,
            "validation_results": results,
            "framework_version": self._get_framework_version(),
        }

    def list_supported_protocols(self) -> List[Dict[str, Any]]:
        """
        List all supported protocols

        Returns:
            List of protocol information dictionaries
        """
        return self.protocol_detector.list_supported_protocols()

    def get_protocol_capabilities(self, protocol_type: ProtocolType) -> Dict[str, Any]:
        """
        Get capabilities for specified protocol

        Args:
            protocol_type: Protocol type

        Returns:
            Protocol capabilities dictionary
        """
        protocol_map = {
            ProtocolType.USB4: "usb4",
            ProtocolType.THUNDERBOLT4: "usb4",  # Uses same implementation
            ProtocolType.ETHERNET_224G: "ethernet_224g",
            ProtocolType.PCIE: "pcie",
        }

        protocol_key = protocol_map.get(protocol_type)
        if not protocol_key:
            raise ValueError(f"Unknown protocol type: {protocol_type}")

        if protocol_key in self.supported_protocols:
            return self.supported_protocols[protocol_key]
        else:
            raise ValueError(f"Protocol not supported: {protocol_key}")

    def _create_usb4_test_sequence(
        self, test_config: Optional[Dict[str, Any]] = None, enable_thunderbolt: bool = False
    ) -> USB4TestSequence:
        """Create USB4/Thunderbolt 4 test sequence"""
        if test_config is None:
            # Create default USB4 configuration
            lane_configs = [
                USB4LaneConfig(
                    lane_id=0, mode=USB4SignalMode.GEN2X2, sample_rate=200e9, bandwidth=25e9, voltage_range=0.8, enable_ssc=True
                ),
                USB4LaneConfig(
                    lane_id=1, mode=USB4SignalMode.GEN2X2, sample_rate=200e9, bandwidth=25e9, voltage_range=0.8, enable_ssc=True
                ),
            ]

            test_phases = [
                USB4TestPhase.INITIALIZATION,
                USB4TestPhase.SIGNAL_ANALYSIS,
                USB4TestPhase.LINK_TRAINING,
                USB4TestPhase.COMPLIANCE,
                USB4TestPhase.TUNNELING,
                USB4TestPhase.POWER_MANAGEMENT,
                USB4TestPhase.VALIDATION,
            ]

            if enable_thunderbolt:
                test_phases.insert(-1, USB4TestPhase.THUNDERBOLT)

            config = USB4TestSequenceConfig(
                test_name="Auto-Generated USB4 Test" + (" with Thunderbolt" if enable_thunderbolt else ""),
                lanes=lane_configs,
                test_phases=test_phases,
                tunneling_modes=[USB4TunnelingMode.PCIE, USB4TunnelingMode.DISPLAYPORT, USB4TunnelingMode.USB32],
                stress_duration=60.0,
                compliance_patterns=["PRBS31", "PRBS15"],
                target_ber=1e-12,
                enable_thunderbolt=enable_thunderbolt,
                power_states_to_test=[USB4LinkState.U0, USB4LinkState.U1, USB4LinkState.U2],
            )
        else:
            # Use provided configuration
            config = USB4TestSequenceConfig(**test_config)

        return USB4TestSequence(config)

    def _create_pcie_test_sequence(self, test_config: Optional[Dict[str, Any]] = None):
        """Create PCIe test sequence"""
        # Implementation would create PCIe test sequence
        # For now, raise not implemented as PCIe is already integrated
        raise NotImplementedError("PCIe test sequence creation via framework integration")

    def _create_ethernet_test_sequence(self, test_config: Optional[Dict[str, Any]] = None):
        """Create Ethernet test sequence"""
        # Implementation would create Ethernet test sequence
        # For now, raise not implemented as Ethernet is already integrated
        raise NotImplementedError("Ethernet test sequence creation via framework integration")

    def _format_signal_data(
        self,
        signal_data: Union[npt.NDArray[np.float64], Dict[int, Dict[str, npt.NDArray[np.float64]]]],
        protocol_type: ProtocolType,
    ) -> Dict[int, Dict[str, npt.NDArray[np.float64]]]:
        """Format signal data for the specified protocol"""
        if isinstance(signal_data, np.ndarray):
            # Single signal array - create time array and format for protocol
            duration = len(signal_data) / 200e9  # Assume 200 GSa/s
            time_array = np.linspace(0, duration, len(signal_data))

            if protocol_type in [ProtocolType.USB4, ProtocolType.THUNDERBOLT4]:
                # USB4 expects dual-lane data
                return {
                    0: {"time": time_array, "voltage": signal_data},
                    1: {"time": time_array, "voltage": signal_data.copy()},  # Duplicate for second lane
                }
            else:
                # Single lane protocols
                return {0: {"time": time_array, "voltage": signal_data}}
        else:
            # Already in correct format
            return signal_data

    def _get_framework_version(self) -> str:
        """Get framework version"""
        try:
            from . import __version__

            return __version__
        except ImportError:
            return "1.4.0"


# Convenience functions for direct usage
def create_validation_framework() -> ValidationFramework:
    """
    Create a validation framework instance

    Returns:
        Configured validation framework
    """
    return ValidationFramework()


def auto_validate_signal(
    signal_data: Union[npt.NDArray[np.float64], Dict[int, Dict[str, npt.NDArray[np.float64]]]],
    sample_rate: float,
    voltage_range: float,
    protocol_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function for automatic signal validation

    Args:
        signal_data: Signal data
        sample_rate: Sampling rate in Hz
        voltage_range: Signal voltage range in V
        protocol_hint: Optional protocol hint

    Returns:
        Validation results
    """
    framework = create_validation_framework()
    return framework.run_auto_validation(signal_data, sample_rate, voltage_range, protocol_hint)


def detect_signal_protocol(signal_data: npt.NDArray[np.float64], sample_rate: float, voltage_range: float) -> ProtocolType:
    """
    Convenience function for protocol detection

    Args:
        signal_data: Signal voltage data
        sample_rate: Sampling rate in Hz
        voltage_range: Signal voltage range in V

    Returns:
        Detected protocol type
    """
    framework = create_validation_framework()
    return framework.detect_protocol(signal_data, sample_rate, voltage_range)


def create_usb4_test_sequence(
    enable_thunderbolt: bool = False, custom_config: Optional[Dict[str, Any]] = None
) -> USB4TestSequence:
    """
    Convenience function to create USB4 test sequence

    Args:
        enable_thunderbolt: Enable Thunderbolt 4 specific tests
        custom_config: Optional custom configuration

    Returns:
        USB4 test sequence
    """
    framework = create_validation_framework()
    return framework._create_usb4_test_sequence(custom_config, enable_thunderbolt)


# Export main classes and functions
__all__ = [
    "ValidationFramework",
    "create_validation_framework",
    "auto_validate_signal",
    "detect_signal_protocol",
    "create_usb4_test_sequence",
]
