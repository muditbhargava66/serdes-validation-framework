"""
Thunderbolt 4 Support Module

This module provides Thunderbolt 4 specific validation capabilities including
security validation, daisy chain testing, and Intel certification support.

Features:
- Thunderbolt 4 security validation (DMA protection)
- Device authentication and authorization
- Daisy chain validation (up to 6 devices)
- Intel certification test suite
- Thunderbolt-specific power delivery validation
"""

from .certification import (
    CertificationConfig,
    CertificationReport,
    CertificationStatus,
    CertificationTestStatus,
    IntelCertificationSuite,
    TestCategory,
)
from .constants import (
    ThunderboltAuthMethod,
    ThunderboltCertificationLevel,
    ThunderboltCertificationSpecs,
    ThunderboltDaisyChainSpecs,
    ThunderboltDeviceType,
    ThunderboltDisplaySpecs,
    ThunderboltPowerSpecs,
    ThunderboltSecurityLevel,
    ThunderboltSecuritySpecs,
)
from .daisy_chain import (
    ChainDevice,
    ChainTestResult,
    DaisyChainResults,
    DaisyChainTestConfig,
    DaisyChainValidator,
)
from .security import (
    AuthenticationStatus,
    DMAProtectionStatus,
    SecurityTestConfig,
    SecurityTestResult,
    ThunderboltSecurityResults,
    ThunderboltSecurityValidator,
)

__version__ = "1.4.0"

__all__ = [
    # Constants and enumerations
    "ThunderboltSecurityLevel",
    "ThunderboltDeviceType",
    "ThunderboltCertificationLevel",
    "ThunderboltAuthMethod",
    "ThunderboltSecuritySpecs",
    "ThunderboltDaisyChainSpecs",
    "ThunderboltDisplaySpecs",
    "ThunderboltPowerSpecs",
    "ThunderboltCertificationSpecs",
    # Security validation
    "ThunderboltSecurityValidator",
    "SecurityTestConfig",
    "SecurityTestResult",
    "DMAProtectionStatus",
    "AuthenticationStatus",
    "ThunderboltSecurityResults",
    # Daisy chain validation
    "DaisyChainValidator",
    "DaisyChainTestConfig",
    "ChainDevice",
    "ChainTestResult",
    "DaisyChainResults",
    # Intel certification
    "IntelCertificationSuite",
    "CertificationConfig",
    "CertificationReport",
    "CertificationStatus",
    "CertificationTestStatus",
    "TestCategory",
    # Version
    "__version__",
]

# Thunderbolt 4 module metadata
THUNDERBOLT_INFO = {
    "name": "Thunderbolt 4 Support",
    "standard": "Intel Thunderbolt 4",
    "version": __version__,
    "supported_features": [
        "DMA protection validation",
        "Device authentication testing",
        "Daisy chain validation (up to 6 devices)",
        "Intel certification support",
        "Security policy enforcement",
        "Power delivery validation",
        "Display support validation",
        "PCIe tunneling validation",
    ],
}
