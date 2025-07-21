"""
PCIe Compliance Test Module

This module provides comprehensive PCIe compliance testing functionality with
type-safe implementation to validate PCIe signals against specification requirements.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceType(Enum):
    """Types of PCIe compliance tests"""
    ELECTRICAL = auto()
    TIMING = auto()
    PROTOCOL = auto()
    FULL = auto()


@dataclass
class ComplianceLimit:
    """Type-safe specification limit with validation"""
    nominal: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        """
        Validate compliance limit values
        
        Raises:
            AssertionError: If limit values are invalid
        """
        # Type validation
        assert isinstance(self.nominal, float), f"Nominal must be float, got {type(self.nominal)}"
        assert isinstance(self.minimum, float), f"Minimum must be float, got {type(self.minimum)}"
        assert isinstance(self.maximum, float), f"Maximum must be float, got {type(self.maximum)}"
        
        # Range validation
        assert self.minimum <= self.nominal <= self.maximum, \
            f"Invalid limit range: {self.minimum} <= {self.nominal} <= {self.maximum} not satisfied"


@dataclass
class ComplianceConfig:
    """
    Configuration for PCIe compliance testing with validation
    """
    test_pattern: str
    sample_rate: float
    record_length: float
    voltage_range: float
    test_types: List[ComplianceType] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Validate configuration parameters
        
        Raises:
            AssertionError: If parameters are invalid
        """
        # Type validation
        assert isinstance(self.test_pattern, str), \
            f"Test pattern must be string, got {type(self.test_pattern)}"
        assert isinstance(self.sample_rate, float), \
            f"Sample rate must be float, got {type(self.sample_rate)}"
        assert isinstance(self.record_length, float), \
            f"Record length must be float, got {type(self.record_length)}"
        assert isinstance(self.voltage_range, float), \
            f"Voltage range must be float, got {type(self.voltage_range)}"
        
        # Value validation
        assert self.sample_rate > 0, f"Sample rate must be positive, got {self.sample_rate}"
        assert self.record_length > 0, f"Record length must be positive, got {self.record_length}"
        assert self.voltage_range > 0, f"Voltage range must be positive, got {self.voltage_range}"
        
        # Test types validation
        assert all(isinstance(t, ComplianceType) for t in self.test_types), \
            "All test types must be ComplianceType enum values"


@dataclass
class ComplianceResult:
    """Type-safe compliance test result with validation"""
    test_name: str
    measured_value: float
    limit: ComplianceLimit
    status: bool = field(init=False)

    def __post_init__(self) -> None:
        """
        Validate result data and calculate status
        
        Raises:
            AssertionError: If result data is invalid
        """
        # Type validation
        assert isinstance(self.test_name, str), \
            f"Test name must be string, got {type(self.test_name)}"
        assert isinstance(self.measured_value, float), \
            f"Measured value must be float, got {type(self.measured_value)}"
        assert isinstance(self.limit, ComplianceLimit), \
            f"Limit must be ComplianceLimit, got {type(self.limit)}"
        
        # Calculate pass/fail status
        self.status = self.limit.minimum <= self.measured_value <= self.limit.maximum


class ComplianceTestSuite:
    """PCIe compliance test suite with type safety"""
    
    def __init__(self, config: ComplianceConfig) -> None:
        """
        Initialize test suite with validated configuration
        
        Args:
            config: Test configuration
            
        Raises:
            AssertionError: If configuration is invalid
        """
        # Validate input
        assert isinstance(config, ComplianceConfig), \
            f"Config must be ComplianceConfig, got {type(config)}"
        
        self.config = config
        self.results: List[ComplianceResult] = []
        logger.info("Compliance test suite initialized")

    def validate_signal_data(
        self, 
        time_data: npt.NDArray[np.float64],
        voltage_data: npt.NDArray[np.float64]
    ) -> None:
        """
        Validate signal data arrays
        
        Args:
            time_data: Time points array
            voltage_data: Voltage measurements array
            
        Raises:
            AssertionError: If arrays are invalid
        """
        # Type validation
        assert isinstance(time_data, np.ndarray), \
            f"Time data must be numpy array, got {type(time_data)}"
        assert isinstance(voltage_data, np.ndarray), \
            f"Voltage data must be numpy array, got {type(voltage_data)}"
        
        # Data type validation
        assert np.issubdtype(time_data.dtype, np.floating), \
            f"Time data must be floating-point, got {time_data.dtype}"
        assert np.issubdtype(voltage_data.dtype, np.floating), \
            f"Voltage data must be floating-point, got {voltage_data.dtype}"
        
        # Array validation
        assert len(time_data) == len(voltage_data), \
            f"Array length mismatch: {len(time_data)} != {len(voltage_data)}"
        assert len(time_data) > 0, "Arrays cannot be empty"
        
        # Value validation
        assert not np.any(np.isnan(time_data)), "Time data contains NaN values"
        assert not np.any(np.isnan(voltage_data)), "Voltage data contains NaN values"
        assert not np.any(np.isinf(time_data)), "Time data contains infinite values"
        assert not np.any(np.isinf(voltage_data)), "Voltage data contains infinite values"

    def run_electrical_tests(
        self, 
        voltage_data: npt.NDArray[np.float64]
    ) -> Dict[str, ComplianceResult]:
        """
        Run electrical compliance tests
        
        Args:
            voltage_data: Voltage measurements array
            
        Returns:
            Dictionary mapping test names to results
            
        Raises:
            AssertionError: If input data is invalid
        """
        # Validate input
        assert isinstance(voltage_data, np.ndarray), \
            f"Voltage data must be numpy array, got {type(voltage_data)}"
        assert np.issubdtype(voltage_data.dtype, np.floating), \
            f"Voltage data must be floating-point, got {voltage_data.dtype}"
        
        results = {}
        
        try:
            # Measure voltage levels
            v_max = float(np.max(voltage_data))
            v_min = float(np.min(voltage_data))
            v_pp = float(v_max - v_min)
            
            # Voltage swing test
            results['voltage_swing'] = ComplianceResult(
                test_name="Voltage Swing",
                measured_value=v_pp,
                limit=ComplianceLimit(
                    nominal=1.0,
                    minimum=0.8,
                    maximum=1.2
                )
            )
            
            # Common mode voltage test
            v_cm = float((v_max + v_min) / 2)
            results['common_mode'] = ComplianceResult(
                test_name="Common Mode Voltage",
                measured_value=v_cm,
                limit=ComplianceLimit(
                    nominal=0.0,
                    minimum=-0.2,
                    maximum=0.2
                )
            )
            
            # Add results to suite
            self.results.extend(results.values())
            
            return results
            
        except Exception as e:
            logger.error(f"Electrical tests failed: {e}")
            raise

    def run_timing_tests(
        self, 
        time_data: npt.NDArray[np.float64],
        voltage_data: npt.NDArray[np.float64]
    ) -> Dict[str, ComplianceResult]:
        """
        Run timing compliance tests
        
        Args:
            time_data: Time points array
            voltage_data: Voltage measurements array
            
        Returns:
            Dictionary mapping test names to results
            
        Raises:
            AssertionError: If input data is invalid
        """
        # Validate inputs
        self.validate_signal_data(time_data, voltage_data)
        
        results = {}
        
        try:
            # Find zero crossings
            zero_crossings = np.where(np.diff(np.signbit(voltage_data)))[0]
            if len(zero_crossings) < 2:
                raise ValueError("Insufficient zero crossings for timing analysis")
            
            # Calculate crossing times
            crossing_times = time_data[zero_crossings]
            time_diffs = np.diff(crossing_times)
            
            # Unit interval test
            ui = float(np.mean(time_diffs))
            results['unit_interval'] = ComplianceResult(
                test_name="Unit Interval",
                measured_value=ui,
                limit=ComplianceLimit(
                    nominal=250e-12,  # 250 ps for PCIe Gen3
                    minimum=245e-12,
                    maximum=255e-12
                )
            )
            
            # Jitter test
            jitter = float(np.std(time_diffs))
            results['jitter'] = ComplianceResult(
                test_name="Jitter",
                measured_value=jitter,
                limit=ComplianceLimit(
                    nominal=15e-12,  # 15 ps
                    minimum=0.0,
                    maximum=30e-12
                )
            )
            
            # Add results to suite
            self.results.extend(results.values())
            
            return results
            
        except Exception as e:
            logger.error(f"Timing tests failed: {e}")
            raise

    def run_compliance_tests(
        self, 
        time_data: npt.NDArray[np.float64],
        voltage_data: npt.NDArray[np.float64]
    ) -> Dict[str, Dict[str, ComplianceResult]]:
        """
        Run full compliance test suite
        
        Args:
            time_data: Time points array
            voltage_data: Voltage measurements array
            
        Returns:
            Dictionary mapping test categories to results
            
        Raises:
            AssertionError: If input data is invalid
        """
        # Validate inputs
        self.validate_signal_data(time_data, voltage_data)
        
        results = {}
        
        try:
            # Run test categories based on configuration
            if ComplianceType.ELECTRICAL in self.config.test_types or ComplianceType.FULL in self.config.test_types:
                results['electrical'] = self.run_electrical_tests(voltage_data)
                
            if ComplianceType.TIMING in self.config.test_types or ComplianceType.FULL in self.config.test_types:
                results['timing'] = self.run_timing_tests(time_data, voltage_data)
                
            return results
            
        except Exception as e:
            logger.error(f"Compliance test suite failed: {e}")
            raise

    def get_overall_status(self) -> bool:
        """
        Get overall compliance status
        
        Returns:
            True if all tests passed, False otherwise
        """
        return all(result.status for result in self.results)

    def generate_report(self) -> Dict[str, Union[bool, List[Dict[str, Union[str, float, bool]]]]]:
        """
        Generate compliance test report
        
        Returns:
            Dictionary containing:
                - status: Overall pass/fail status
                - results: List of test results
        """
        # Format results
        formatted_results = []
        for result in self.results:
            formatted_results.append({
                'test_name': result.test_name,
                'measured_value': result.measured_value,
                'nominal': result.limit.nominal,
                'minimum': result.limit.minimum,
                'maximum': result.limit.maximum,
                'status': result.status
            })
        
        return {
            'status': self.get_overall_status(),
            'results': formatted_results
        }


# Example usage
def create_test_suite(
    sample_rate: float = 20e9,
    record_length: float = 1e-6,
    voltage_range: float = 1.0
) -> ComplianceTestSuite:
    """
    Create PCIe compliance test suite
    
    Args:
        sample_rate: Sample rate in Hz
        record_length: Record length in seconds
        voltage_range: Voltage range in volts
        
    Returns:
        Configured test suite
        
    Raises:
        AssertionError: If parameters are invalid
    """
    # Validate inputs
    assert isinstance(sample_rate, float), f"Sample rate must be float, got {type(sample_rate)}"
    assert isinstance(record_length, float), f"Record length must be float, got {type(record_length)}"
    assert isinstance(voltage_range, float), f"Voltage range must be float, got {type(voltage_range)}"
    
    # Create configuration
    config = ComplianceConfig(
        test_pattern="PRBS31",
        sample_rate=sample_rate,
        record_length=record_length,
        voltage_range=voltage_range,
        test_types=[ComplianceType.FULL]
    )
    
    return ComplianceTestSuite(config)
