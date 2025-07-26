"""
USB4 Base Interfaces and Abstract Classes

This module defines the core interfaces and abstract base classes for USB4 protocol
components, providing a consistent API for signal analysis, link training, tunneling,
and compliance testing.

Features:
- Abstract base classes for USB4 components
- Common interfaces for signal analysis
- Base classes for tunneling protocols
- Configuration and result data structures
- Error handling interfaces
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from .constants import (
    USB4ErrorType,
    USB4LinkState,
    USB4SignalMode,
    USB4TunnelingMode,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestResult(Enum):
    """Test result status"""

    PASS = auto()
    FAIL = auto()
    WARNING = auto()
    NOT_TESTED = auto()


class SignalQuality(Enum):
    """Signal quality assessment"""

    EXCELLENT = auto()
    GOOD = auto()
    MARGINAL = auto()
    POOR = auto()
    FAILED = auto()


@dataclass
class USB4Config:
    """Base configuration for USB4 components"""

    signal_mode: USB4SignalMode
    sample_rate: float
    capture_length: int
    enable_ssc: bool = True
    enable_equalization: bool = True
    debug_mode: bool = False


@dataclass
class USB4TestResult:
    """Base test result structure"""

    test_name: str
    result: TestResult
    measured_value: Optional[float] = None
    limit_value: Optional[float] = None
    units: Optional[str] = None
    message: Optional[str] = None
    timestamp: Optional[float] = None


@dataclass
class USB4SignalData:
    """USB4 signal data container"""

    lane0_data: npt.NDArray[np.float64]
    lane1_data: npt.NDArray[np.float64]
    sample_rate: float
    timestamp: npt.NDArray[np.float64]
    metadata: Dict[str, Any]


class USB4Component(ABC):
    """Abstract base class for all USB4 components"""

    def __init__(self, config: USB4Config):
        """
        Initialize USB4 component

        Args:
            config: USB4 configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the component

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up component resources"""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized"""
        return self._initialized

    def validate_config(self) -> bool:
        """
        Validate component configuration

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, USB4Config):
            raise ValueError("Invalid configuration type")
        return True


class USB4SignalAnalyzer(USB4Component):
    """Abstract base class for USB4 signal analyzers"""

    @abstractmethod
    def analyze_signal(self, signal_data: USB4SignalData) -> Dict[str, Any]:
        """
        Analyze USB4 signal data

        Args:
            signal_data: USB4 signal data to analyze

        Returns:
            Analysis results dictionary
        """
        pass

    @abstractmethod
    def measure_eye_diagram(self, signal_data: USB4SignalData) -> Dict[str, float]:
        """
        Measure eye diagram parameters

        Args:
            signal_data: USB4 signal data

        Returns:
            Eye diagram measurements
        """
        pass

    @abstractmethod
    def analyze_jitter(self, signal_data: USB4SignalData) -> Dict[str, float]:
        """
        Analyze signal jitter

        Args:
            signal_data: USB4 signal data

        Returns:
            Jitter analysis results
        """
        pass

    @abstractmethod
    def measure_lane_skew(self, signal_data: USB4SignalData) -> float:
        """
        Measure skew between USB4 lanes

        Args:
            signal_data: USB4 dual-lane signal data

        Returns:
            Lane skew in seconds
        """
        pass


class USB4LinkTrainer(USB4Component):
    """Abstract base class for USB4 link training"""

    @abstractmethod
    def execute_training(self) -> Dict[str, Any]:
        """
        Execute USB4 link training sequence

        Returns:
            Training results
        """
        pass

    @abstractmethod
    def monitor_link_state(self) -> USB4LinkState:
        """
        Monitor current USB4 link state

        Returns:
            Current link state
        """
        pass

    @abstractmethod
    def validate_training_sequence(self, sequence_data: npt.NDArray) -> bool:
        """
        Validate link training sequence

        Args:
            sequence_data: Training sequence data

        Returns:
            True if sequence is valid
        """
        pass


class USB4TunnelValidator(USB4Component):
    """Abstract base class for USB4 tunneling validation"""

    @abstractmethod
    def validate_tunnel(self, tunnel_mode: USB4TunnelingMode, data: npt.NDArray) -> Dict[str, Any]:
        """
        Validate tunneled protocol data

        Args:
            tunnel_mode: Type of tunneling protocol
            data: Tunneled data to validate

        Returns:
            Validation results
        """
        pass

    @abstractmethod
    def measure_tunnel_bandwidth(self, tunnel_mode: USB4TunnelingMode) -> float:
        """
        Measure tunnel bandwidth utilization

        Args:
            tunnel_mode: Type of tunneling protocol

        Returns:
            Bandwidth utilization in bps
        """
        pass

    @abstractmethod
    def test_tunnel_establishment(self, tunnel_mode: USB4TunnelingMode) -> bool:
        """
        Test tunnel establishment process

        Args:
            tunnel_mode: Type of tunneling protocol

        Returns:
            True if tunnel establishment successful
        """
        pass


class USB4ComplianceTester(USB4Component):
    """Abstract base class for USB4 compliance testing"""

    @abstractmethod
    def run_compliance_suite(self) -> List[USB4TestResult]:
        """
        Run complete USB4 compliance test suite

        Returns:
            List of test results
        """
        pass

    @abstractmethod
    def validate_specification(self, spec_name: str) -> USB4TestResult:
        """
        Validate specific USB4 specification

        Args:
            spec_name: Name of specification to validate

        Returns:
            Test result
        """
        pass

    @abstractmethod
    def generate_report(self, results: List[USB4TestResult]) -> str:
        """
        Generate compliance test report

        Args:
            results: List of test results

        Returns:
            Formatted report string
        """
        pass


class USB4InstrumentController(USB4Component):
    """Abstract base class for USB4 instrument control"""

    @abstractmethod
    def connect_instrument(self, resource_string: str) -> bool:
        """
        Connect to test instrument

        Args:
            resource_string: Instrument resource identifier

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def configure_measurement(self, config: Dict[str, Any]) -> bool:
        """
        Configure instrument measurement

        Args:
            config: Measurement configuration

        Returns:
            True if configuration successful
        """
        pass

    @abstractmethod
    def acquire_data(self) -> USB4SignalData:
        """
        Acquire signal data from instrument

        Returns:
            USB4 signal data
        """
        pass

    @abstractmethod
    def disconnect_instrument(self) -> None:
        """Disconnect from instrument"""
        pass


@dataclass
class USB4ErrorInfo:
    """USB4 error information"""

    error_type: USB4ErrorType
    error_code: int
    message: str
    timestamp: float
    context: Dict[str, Any]


class USB4ErrorHandler:
    """USB4 error handling utilities"""

    def __init__(self):
        self.error_history: List[USB4ErrorInfo] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def handle_error(self, error_info: USB4ErrorInfo) -> None:
        """
        Handle USB4 error

        Args:
            error_info: Error information
        """
        self.error_history.append(error_info)
        self.logger.error(f"USB4 Error [{error_info.error_type.name}]: {error_info.message}")

    def get_error_history(self) -> List[USB4ErrorInfo]:
        """
        Get error history

        Returns:
            List of error information
        """
        return self.error_history.copy()

    def clear_error_history(self) -> None:
        """Clear error history"""
        self.error_history.clear()


class USB4DataProcessor:
    """USB4 data processing utilities"""

    @staticmethod
    def validate_signal_data(signal_data: USB4SignalData) -> bool:
        """
        Validate USB4 signal data structure

        Args:
            signal_data: Signal data to validate

        Returns:
            True if data is valid

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(signal_data, USB4SignalData):
            raise ValueError("Invalid signal data type")

        if signal_data.lane0_data.size == 0 or signal_data.lane1_data.size == 0:
            raise ValueError("Empty signal data")

        if signal_data.lane0_data.size != signal_data.lane1_data.size:
            raise ValueError("Lane data size mismatch")

        if signal_data.sample_rate <= 0:
            raise ValueError("Invalid sample rate")

        return True

    @staticmethod
    def normalize_signal_data(signal_data: USB4SignalData) -> USB4SignalData:
        """
        Normalize USB4 signal data

        Args:
            signal_data: Signal data to normalize

        Returns:
            Normalized signal data
        """
        USB4DataProcessor.validate_signal_data(signal_data)

        # Normalize to zero mean, unit variance
        lane0_normalized = (signal_data.lane0_data - np.mean(signal_data.lane0_data)) / np.std(signal_data.lane0_data)
        lane1_normalized = (signal_data.lane1_data - np.mean(signal_data.lane1_data)) / np.std(signal_data.lane1_data)

        return USB4SignalData(
            lane0_data=lane0_normalized,
            lane1_data=lane1_normalized,
            sample_rate=signal_data.sample_rate,
            timestamp=signal_data.timestamp,
            metadata={**signal_data.metadata, "normalized": True},
        )

    @staticmethod
    def calculate_signal_statistics(signal_data: USB4SignalData) -> Dict[str, Dict[str, float]]:
        """
        Calculate basic signal statistics

        Args:
            signal_data: Signal data to analyze

        Returns:
            Statistics for both lanes
        """
        USB4DataProcessor.validate_signal_data(signal_data)

        def calc_stats(data: npt.NDArray[np.float64]) -> Dict[str, float]:
            return {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "rms": float(np.sqrt(np.mean(data**2))),
                "peak_to_peak": float(np.max(data) - np.min(data)),
            }

        return {"lane0": calc_stats(signal_data.lane0_data), "lane1": calc_stats(signal_data.lane1_data)}


__all__ = [
    # Enums
    "TestResult",
    "SignalQuality",
    # Data structures
    "USB4Config",
    "USB4TestResult",
    "USB4SignalData",
    "USB4ErrorInfo",
    # Abstract base classes
    "USB4Component",
    "USB4SignalAnalyzer",
    "USB4LinkTrainer",
    "USB4TunnelValidator",
    "USB4ComplianceTester",
    "USB4InstrumentController",
    # Utility classes
    "USB4ErrorHandler",
    "USB4DataProcessor",
]
