# src/serdes_validation_framework/protocols/ethernet_224g/compliance.py

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from .constants import COMPLIANCE_PATTERNS, ETHERNET_224G_SPECS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ComplianceTestConfig:
    """Configuration for compliance testing"""
    test_pattern: str
    sample_rate: float
    record_length: float
    voltage_range: float

    def __post_init__(self) -> None:
        """Validate configuration parameters"""
        assert isinstance(self.test_pattern, str), "Test pattern must be a string"
        assert isinstance(self.sample_rate, float), "Sample rate must be a float"
        assert isinstance(self.record_length, float), "Record length must be a float"
        assert isinstance(self.voltage_range, float), "Voltage range must be a float"

        assert self.sample_rate > 0, "Sample rate must be positive"
        assert self.record_length > 0, "Record length must be positive"
        assert self.voltage_range > 0, "Voltage range must be positive"

@dataclass
class ComplianceLimit:
    """Specification limit with tolerance"""
    nominal: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        """Validate limit values"""
        for value in [self.nominal, self.minimum, self.maximum]:
            assert isinstance(value, float), "All limits must be floats"
        assert self.minimum <= self.nominal <= self.maximum, "Invalid limit range"

class ComplianceSpecification:
    """224G Ethernet compliance specification checker"""

    def __init__(self) -> None:
        """Initialize compliance specifications"""
        self.specs = ETHERNET_224G_SPECS
        self.test_patterns = COMPLIANCE_PATTERNS
        self._initialize_limits()
        logger.info("Compliance specification checker initialized")

    def _initialize_limits(self) -> None:
        """Initialize specification limits"""
        self.limits: Dict[str, ComplianceLimit] = {
            'level_separation': ComplianceLimit(
                nominal=0.5,
                minimum=0.4,
                maximum=0.6
            ),
            'eye_height': ComplianceLimit(
                nominal=0.3,
                minimum=0.2,
                maximum=0.4
            ),
            'eye_width': ComplianceLimit(
                nominal=0.5,
                minimum=0.4,
                maximum=0.6
            ),
            'rms_evm': ComplianceLimit(
                nominal=3.5,
                minimum=0.0,
                maximum=5.0
            )
        }

    def check_pam4_levels(
        self,
        measured_levels: npt.NDArray[np.float64]
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check PAM4 level compliance
        
        Args:
            measured_levels: Array of measured voltage levels
            
        Returns:
            Tuple of (pass/fail status, measurements)
        """
        assert isinstance(measured_levels, np.ndarray), "Levels must be a numpy array"
        assert np.issubdtype(measured_levels.dtype, np.floating), \
            "Levels must be floating-point numbers"
        assert len(measured_levels) == 4, "Must have exactly 4 PAM4 levels"

        try:
            # Calculate level separations
            level_gaps = np.diff(sorted(measured_levels))
            min_separation = float(np.min(level_gaps))

            # Check against limits
            limit = self.limits['level_separation']
            passed = limit.minimum <= min_separation <= limit.maximum

            measurements = {
                'min_separation': min_separation,
                'uniformity': float(np.std(level_gaps) / np.mean(level_gaps))
            }

            return passed, measurements

        except Exception as e:
            logger.error(f"Failed to check PAM4 levels: {e}")
            raise

    def check_eye_diagram(
        self,
        eye_height: float,
        eye_width: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check eye diagram compliance
        
        Args:
            eye_height: Measured eye height
            eye_width: Measured eye width
            
        Returns:
            Tuple of (pass/fail status, measurements)
        """
        assert isinstance(eye_height, float), "Eye height must be a float"
        assert isinstance(eye_width, float), "Eye width must be a float"
        assert eye_height >= 0, "Eye height must be non-negative"
        assert eye_width >= 0, "Eye width must be non-negative"

        try:
            height_limit = self.limits['eye_height']
            width_limit = self.limits['eye_width']

            height_passed = height_limit.minimum <= eye_height <= height_limit.maximum
            width_passed = width_limit.minimum <= eye_width <= width_limit.maximum

            measurements = {
                'eye_height': eye_height,
                'eye_width': eye_width,
                'height_margin': eye_height - height_limit.minimum,
                'width_margin': eye_width - width_limit.minimum
            }

            return (height_passed and width_passed), measurements

        except Exception as e:
            logger.error(f"Failed to check eye diagram: {e}")
            raise

    def check_evm(
        self,
        rms_evm: float,
        peak_evm: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check EVM compliance
        
        Args:
            rms_evm: Measured RMS EVM percentage
            peak_evm: Measured peak EVM percentage
            
        Returns:
            Tuple of (pass/fail status, measurements)
        """
        assert isinstance(rms_evm, float), "RMS EVM must be a float"
        assert isinstance(peak_evm, float), "Peak EVM must be a float"
        assert rms_evm >= 0, "RMS EVM must be non-negative"
        assert peak_evm >= 0, "Peak EVM must be non-negative"

        try:
            evm_limit = self.limits['rms_evm']
            passed = rms_evm <= evm_limit.maximum

            measurements = {
                'rms_evm': rms_evm,
                'peak_evm': peak_evm,
                'margin': evm_limit.maximum - rms_evm
            }

            return passed, measurements

        except Exception as e:
            logger.error(f"Failed to check EVM: {e}")
            raise

    def get_test_config(
        self,
        test_type: str
    ) -> ComplianceTestConfig:
        """
        Get test configuration for specific compliance test
        
        Args:
            test_type: Type of compliance test
            
        Returns:
            ComplianceTestConfig object
        """
        assert isinstance(test_type, str), "Test type must be a string"
        assert test_type in self.test_patterns, f"Unknown test type: {test_type}"

        try:
            configs = {
                'jitter': ComplianceTestConfig(
                    test_pattern=self.test_patterns['jitter'],
                    sample_rate=256e9,
                    record_length=1e-6,
                    voltage_range=0.8
                ),
                'eye': ComplianceTestConfig(
                    test_pattern=self.test_patterns['eye'],
                    sample_rate=256e9,
                    record_length=2e-6,
                    voltage_range=0.8
                ),
                'levels': ComplianceTestConfig(
                    test_pattern=self.test_patterns['levels'],
                    sample_rate=224e9,
                    record_length=500e-9,
                    voltage_range=1.0
                )
            }

            return configs[test_type]

        except Exception as e:
            logger.error(f"Failed to get test config: {e}")
            raise
