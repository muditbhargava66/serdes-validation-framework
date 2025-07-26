"""
Protocol Detection and Selection Module

This module provides utilities for automatically detecting and selecting
the appropriate protocol validation modules based on signal characteristics
or user configuration.
"""

import logging
from enum import Enum, auto
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt

from .protocols import SUPPORTED_PROTOCOLS

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Supported protocol types"""

    PCIE = auto()
    ETHERNET_224G = auto()
    USB4 = auto()
    THUNDERBOLT4 = auto()
    UNKNOWN = auto()


class DetectionMethod(Enum):
    """Protocol detection methods"""

    SIGNAL_ANALYSIS = auto()
    USER_SPECIFIED = auto()
    AUTO_DETECT = auto()


class ProtocolDetector:
    """
    Protocol detection and selection utility

    This class provides methods to detect the appropriate protocol
    based on signal characteristics, user input, or configuration.
    """

    def __init__(self) -> None:
        """Initialize protocol detector"""
        self.supported_protocols = SUPPORTED_PROTOCOLS
        logger.info("Protocol detector initialized")

    def detect_protocol_from_signal(
        self, signal_data: npt.NDArray[np.float64], sample_rate: float, voltage_range: float
    ) -> ProtocolType:
        """
        Detect protocol type from signal characteristics

        Args:
            signal_data: Signal voltage data
            sample_rate: Sampling rate in Hz
            voltage_range: Signal voltage range in V

        Returns:
            Detected protocol type
        """
        try:
            # Analyze signal characteristics
            signal_stats = self._analyze_signal_characteristics(signal_data, sample_rate, voltage_range)

            # Apply detection heuristics
            detected_protocol = self._apply_detection_heuristics(signal_stats)

            logger.info(f"Detected protocol: {detected_protocol.name}")
            return detected_protocol

        except Exception as e:
            logger.error(f"Protocol detection failed: {e}")
            return ProtocolType.UNKNOWN

    def select_protocol_by_name(self, protocol_name: str) -> ProtocolType:
        """
        Select protocol by name

        Args:
            protocol_name: Name of the protocol

        Returns:
            Selected protocol type
        """
        protocol_mapping = {
            "pcie": ProtocolType.PCIE,
            "pcie6": ProtocolType.PCIE,
            "ethernet": ProtocolType.ETHERNET_224G,
            "ethernet_224g": ProtocolType.ETHERNET_224G,
            "224g": ProtocolType.ETHERNET_224G,
            "usb4": ProtocolType.USB4,
            "usb": ProtocolType.USB4,
            "thunderbolt": ProtocolType.THUNDERBOLT4,
            "thunderbolt4": ProtocolType.THUNDERBOLT4,
            "tb4": ProtocolType.THUNDERBOLT4,
        }

        normalized_name = protocol_name.lower().strip()
        protocol_type = protocol_mapping.get(normalized_name, ProtocolType.UNKNOWN)

        logger.info(f"Selected protocol by name '{protocol_name}': {protocol_type.name}")
        return protocol_type

    def get_protocol_config(self, protocol_type: ProtocolType) -> Dict[str, Any]:
        """
        Get configuration for specified protocol

        Args:
            protocol_type: Protocol type

        Returns:
            Protocol configuration dictionary
        """
        config_mapping = {
            ProtocolType.PCIE: self._get_pcie_config(),
            ProtocolType.ETHERNET_224G: self._get_ethernet_224g_config(),
            ProtocolType.USB4: self._get_usb4_config(),
            ProtocolType.THUNDERBOLT4: self._get_thunderbolt4_config(),
        }

        if protocol_type not in config_mapping:
            raise ValueError(f"Unsupported protocol type: {protocol_type}")

        return config_mapping[protocol_type]

    def list_supported_protocols(self) -> List[Dict[str, Any]]:
        """
        List all supported protocols with their information

        Returns:
            List of protocol information dictionaries
        """
        protocols = []
        for name, info in self.supported_protocols.items():
            protocol_info = {
                "name": name,
                "display_name": info["name"],
                "version": info["version"],
                "description": info["description"],
                "features": info["features"],
            }
            protocols.append(protocol_info)

        return protocols

    def _analyze_signal_characteristics(
        self, signal_data: npt.NDArray[np.float64], sample_rate: float, voltage_range: float
    ) -> Dict[str, float]:
        """Analyze signal characteristics for protocol detection"""
        # Basic signal statistics
        signal_mean = float(np.mean(signal_data))
        signal_std = float(np.std(signal_data))
        signal_peak_to_peak = float(np.ptp(signal_data))

        # Frequency domain analysis
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data), 1 / sample_rate)
        power_spectrum = np.abs(fft) ** 2

        # Find dominant frequency
        dominant_freq_idx = np.argmax(power_spectrum[1 : len(power_spectrum) // 2]) + 1
        dominant_frequency = float(abs(freqs[dominant_freq_idx]))

        # Estimate symbol rate (approximate)
        estimated_symbol_rate = dominant_frequency * 2  # Nyquist approximation

        # Signal level analysis
        signal_levels = self._estimate_signal_levels(signal_data)

        return {
            "mean": signal_mean,
            "std": signal_std,
            "peak_to_peak": signal_peak_to_peak,
            "dominant_frequency": dominant_frequency,
            "estimated_symbol_rate": estimated_symbol_rate,
            "voltage_range": voltage_range,
            "sample_rate": sample_rate,
            "num_levels": len(signal_levels),
            "level_separation": float(np.mean(np.diff(sorted(signal_levels)))) if len(signal_levels) > 1 else 0.0,
        }

    def _estimate_signal_levels(self, signal_data: npt.NDArray[np.float64]) -> List[float]:
        """Estimate discrete signal levels (for PAM detection)"""
        # Simple histogram-based level detection
        hist, bin_edges = np.histogram(signal_data, bins=50)

        # Find peaks in histogram
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > np.max(hist) * 0.1:
                peak_voltage = (bin_edges[i] + bin_edges[i + 1]) / 2
                peaks.append(float(peak_voltage))

        return sorted(peaks)

    def _apply_detection_heuristics(self, signal_stats: Dict[str, float]) -> ProtocolType:
        """Apply heuristics to detect protocol type"""
        symbol_rate = signal_stats["estimated_symbol_rate"]
        num_levels = signal_stats["num_levels"]
        voltage_range = signal_stats["voltage_range"]

        # PCIe 6.0 detection (64 GT/s, PAM4)
        if (
            30e9 <= symbol_rate <= 35e9  # ~32 GBaud for PCIe 6.0
            and num_levels >= 3  # PAM4 has 4 levels
            and 0.3 <= voltage_range <= 1.2
        ):  # Typical PCIe voltage range
            return ProtocolType.PCIE

        # Ethernet 224G detection (112 GBaud, PAM4)
        if (
            100e9 <= symbol_rate <= 120e9  # ~112 GBaud
            and num_levels >= 3  # PAM4 has 4 levels
            and 0.4 <= voltage_range <= 1.0
        ):  # Typical Ethernet voltage range
            return ProtocolType.ETHERNET_224G

        # USB4 detection (20 Gbps per lane, NRZ)
        if (
            18e9 <= symbol_rate <= 22e9  # ~20 Gbps
            and num_levels <= 2  # NRZ has 2 levels
            and 0.3 <= voltage_range <= 0.8
        ):  # Typical USB4 voltage range
            return ProtocolType.USB4

        # Thunderbolt 4 detection (similar to USB4 but may have different characteristics)
        if 18e9 <= symbol_rate <= 22e9 and num_levels <= 2 and 0.3 <= voltage_range <= 0.8:
            # Additional heuristics could be added here to distinguish from USB4
            return ProtocolType.THUNDERBOLT4

        return ProtocolType.UNKNOWN

    def _get_pcie_config(self) -> Dict[str, Any]:
        """Get PCIe protocol configuration"""
        return {
            "protocol_type": "pcie",
            "symbol_rate": 32e9,  # 32 GBaud
            "modulation": "PAM4",
            "voltage_range": 0.8,
            "lanes": [1, 2, 4, 8, 16],
            "test_patterns": ["PRBS31", "PRBS15", "PRBS7"],
            "compliance_tests": ["signal_integrity", "link_training", "equalization"],
        }

    def _get_ethernet_224g_config(self) -> Dict[str, Any]:
        """Get Ethernet 224G protocol configuration"""
        return {
            "protocol_type": "ethernet_224g",
            "symbol_rate": 112e9,  # 112 GBaud
            "modulation": "PAM4",
            "voltage_range": 0.6,
            "lanes": [1, 2, 4],
            "test_patterns": ["PRBS31", "PRBS13"],
            "compliance_tests": ["signal_integrity", "equalization", "performance"],
        }

    def _get_usb4_config(self) -> Dict[str, Any]:
        """Get USB4 protocol configuration"""
        return {
            "protocol_type": "usb4",
            "symbol_rate": 20e9,  # 20 Gbps per lane
            "modulation": "NRZ",
            "voltage_range": 0.6,
            "lanes": [2],  # USB4 uses 2 lanes
            "total_bandwidth": 40e9,  # 40 Gbps total
            "test_patterns": ["PRBS31", "PRBS15"],
            "compliance_tests": ["signal_integrity", "link_training", "tunneling", "power_management"],
            "tunneling_modes": ["pcie", "displayport", "usb32"],
            "power_states": ["U0", "U1", "U2", "U3"],
        }

    def _get_thunderbolt4_config(self) -> Dict[str, Any]:
        """Get Thunderbolt 4 protocol configuration"""
        usb4_config = self._get_usb4_config()
        usb4_config.update(
            {
                "protocol_type": "thunderbolt4",
                "security_features": True,
                "daisy_chain_support": True,
                "max_chain_devices": 6,
                "power_delivery": 100.0,  # 100W
                "compliance_tests": [
                    "signal_integrity",
                    "link_training",
                    "tunneling",
                    "power_management",
                    "security",
                    "certification",
                ],
            }
        )
        return usb4_config


def create_protocol_detector() -> ProtocolDetector:
    """
    Factory function to create a protocol detector instance

    Returns:
        Configured protocol detector
    """
    return ProtocolDetector()


def detect_protocol_auto(signal_data: npt.NDArray[np.float64], sample_rate: float, voltage_range: float) -> ProtocolType:
    """
    Convenience function for automatic protocol detection

    Args:
        signal_data: Signal voltage data
        sample_rate: Sampling rate in Hz
        voltage_range: Signal voltage range in V

    Returns:
        Detected protocol type
    """
    detector = create_protocol_detector()
    return detector.detect_protocol_from_signal(signal_data, sample_rate, voltage_range)
