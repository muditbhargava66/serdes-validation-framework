"""
Thunderbolt 4 Constants

This module defines Thunderbolt 4 specific constants, specifications, and enumerations
for security validation, device authentication, and Intel certification.

Features:
- Thunderbolt 4 security levels and policies
- Device authentication parameters
- Daisy chain specifications
- Intel certification requirements
- Power delivery specifications
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Final, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThunderboltSecurityLevel(Enum):
    """Thunderbolt security levels"""

    NONE = auto()  # No security (legacy mode)
    USER = auto()  # User approval required
    SECURE = auto()  # Secure connect with device authentication
    DPONLY = auto()  # DisplayPort only mode


class ThunderboltDeviceType(Enum):
    """Thunderbolt device types"""

    HOST = auto()  # Thunderbolt host controller
    DEVICE = auto()  # Thunderbolt device
    HUB = auto()  # Thunderbolt hub/dock
    DISPLAY = auto()  # Thunderbolt display
    STORAGE = auto()  # Thunderbolt storage device
    EGPU = auto()  # External GPU


class ThunderboltCertificationLevel(Enum):
    """Intel Thunderbolt certification levels"""

    BASIC = auto()  # Basic Thunderbolt certification
    PREMIUM = auto()  # Premium Thunderbolt certification
    PRO = auto()  # Professional Thunderbolt certification


class ThunderboltAuthMethod(Enum):
    """Thunderbolt authentication methods"""

    NONE = auto()  # No authentication
    KEY_BASED = auto()  # Key-based authentication
    CERTIFICATE = auto()  # Certificate-based authentication
    BIOMETRIC = auto()  # Biometric authentication


@dataclass(frozen=True)
class ThunderboltSecuritySpecs:
    """Thunderbolt 4 security specifications"""

    # DMA protection
    DMA_PROTECTION_REQUIRED: bool = True
    IOMMU_REQUIRED: bool = True
    VT_D_REQUIRED: bool = True  # Intel VT-d support

    # Device authentication
    DEVICE_AUTH_TIMEOUT: float = 5.0  # Authentication timeout (s)
    MAX_AUTH_RETRIES: int = 3  # Maximum authentication retries
    AUTH_KEY_LENGTH: int = 256  # Authentication key length (bits)

    # Security policies
    ALLOW_LEGACY_DEVICES: bool = False  # Allow non-Thunderbolt 4 devices
    REQUIRE_APPROVAL: bool = True  # Require user approval
    AUTO_APPROVE_DISPLAYS: bool = True  # Auto-approve displays

    # Encryption
    LINK_ENCRYPTION: bool = True  # Link-level encryption
    ENCRYPTION_ALGORITHM: str = "AES-256"  # Encryption algorithm


@dataclass(frozen=True)
class ThunderboltDaisyChainSpecs:
    """Thunderbolt daisy chain specifications"""

    # Chain limits
    MAX_DEVICES: int = 6  # Maximum devices in chain
    MAX_HUBS: int = 2  # Maximum hubs in chain
    MAX_DISPLAYS: int = 2  # Maximum displays in chain

    # Cable specifications
    MAX_CABLE_LENGTH: float = 2.0  # Maximum cable length (m)
    MIN_CABLE_RATING: str = "40Gbps"  # Minimum cable rating

    # Power delivery in chain
    MAX_CHAIN_POWER: float = 100.0  # Maximum power through chain (W)
    POWER_BUDGET_PER_DEVICE: float = 15.0  # Power budget per device (W)

    # Bandwidth allocation
    BANDWIDTH_RESERVATION: float = 0.1  # Bandwidth reservation (10%)
    MIN_DEVICE_BANDWIDTH: float = 1.0e9  # Minimum per-device bandwidth (1 Gbps)


@dataclass(frozen=True)
class ThunderboltDisplaySpecs:
    """Thunderbolt display specifications"""

    # Display support
    MAX_4K_60HZ: int = 2  # Maximum 4K@60Hz displays
    MAX_8K_30HZ: int = 1  # Maximum 8K@30Hz displays
    MAX_5K_60HZ: int = 1  # Maximum 5K@60Hz displays

    # DisplayPort tunneling
    DP_VERSION: str = "2.0"  # DisplayPort version
    DP_BANDWIDTH: float = 80.0e9  # DisplayPort bandwidth (80 Gbps)
    DP_LANES: int = 4  # DisplayPort lanes

    # HDR support
    HDR10_SUPPORT: bool = True  # HDR10 support
    DOLBY_VISION_SUPPORT: bool = True  # Dolby Vision support
    HDR_PEAK_BRIGHTNESS: float = 1000.0  # HDR peak brightness (nits)


@dataclass(frozen=True)
class ThunderboltPowerSpecs:
    """Thunderbolt power delivery specifications"""

    # USB Power Delivery
    USB_PD_VERSION: str = "3.0"  # USB-PD version
    MAX_POWER: float = 100.0  # Maximum power delivery (W)
    MIN_LAPTOP_POWER: float = 15.0  # Minimum laptop charging power (W)

    # Power profiles
    POWER_PROFILES: Dict[str, float] = field(
        default_factory=lambda: {"phone": 15.0, "tablet": 30.0, "laptop": 65.0, "workstation": 100.0}
    )

    # Power negotiation
    NEGOTIATION_TIMEOUT: float = 1.0  # Power negotiation timeout (s)
    VOLTAGE_TOLERANCE: float = 0.05  # Voltage tolerance (5%)
    CURRENT_TOLERANCE: float = 0.1  # Current tolerance (10%)


@dataclass(frozen=True)
class ThunderboltCertificationSpecs:
    """Intel Thunderbolt certification specifications"""

    # Certification requirements
    INTEL_CERTIFICATION_REQUIRED: bool = True
    CERTIFICATION_VALIDITY: float = 365.0 * 24 * 3600  # 1 year in seconds

    # Test requirements
    REQUIRED_TESTS: List[str] = field(
        default_factory=lambda: [
            "signal_integrity",
            "protocol_compliance",
            "interoperability",
            "security_validation",
            "power_delivery",
            "thermal_validation",
        ]
    )

    # Performance requirements
    MIN_BANDWIDTH: float = 40.0e9  # Minimum bandwidth (40 Gbps)
    MAX_LATENCY: float = 1.0e-6  # Maximum latency (1 μs)
    MIN_RELIABILITY: float = 0.999999  # Minimum reliability (99.9999%)

    # Environmental requirements
    OPERATING_TEMP_MIN: float = 0.0  # Minimum operating temperature (°C)
    OPERATING_TEMP_MAX: float = 70.0  # Maximum operating temperature (°C)
    STORAGE_TEMP_MIN: float = -20.0  # Minimum storage temperature (°C)
    STORAGE_TEMP_MAX: float = 85.0  # Maximum storage temperature (°C)


# Thunderbolt 4 specifications collection
THUNDERBOLT_SPECS: Final[Dict[str, object]] = {
    "security": ThunderboltSecuritySpecs(),
    "daisy_chain": ThunderboltDaisyChainSpecs(),
    "display": ThunderboltDisplaySpecs(),
    "power": ThunderboltPowerSpecs(),
    "certification": ThunderboltCertificationSpecs(),
}

# Thunderbolt device authentication keys (example structure)
THUNDERBOLT_AUTH_KEYS: Final[Dict[str, str]] = {
    "intel_root_ca": "example_root_certificate_authority_key",
    "device_cert": "example_device_certificate_key",
    "host_cert": "example_host_certificate_key",
}

# Thunderbolt security policies
THUNDERBOLT_SECURITY_POLICIES: Final[Dict[str, Dict[str, bool]]] = {
    "strict": {
        "require_authentication": True,
        "allow_legacy_devices": False,
        "auto_approve_displays": False,
        "require_user_approval": True,
        "enable_dma_protection": True,
    },
    "balanced": {
        "require_authentication": True,
        "allow_legacy_devices": True,
        "auto_approve_displays": True,
        "require_user_approval": True,
        "enable_dma_protection": True,
    },
    "permissive": {
        "require_authentication": False,
        "allow_legacy_devices": True,
        "auto_approve_displays": True,
        "require_user_approval": False,
        "enable_dma_protection": True,
    },
}


def validate_thunderbolt_security_level(level: ThunderboltSecurityLevel) -> bool:
    """
    Validate Thunderbolt security level

    Args:
        level: Security level to validate

    Returns:
        True if level is valid

    Raises:
        ValueError: If level is invalid
    """
    if not isinstance(level, ThunderboltSecurityLevel):
        raise ValueError(f"Invalid Thunderbolt security level: {level}")
    return True


def get_security_policy(policy_name: str) -> Dict[str, bool]:
    """
    Get Thunderbolt security policy

    Args:
        policy_name: Name of security policy

    Returns:
        Security policy configuration

    Raises:
        ValueError: If policy name is invalid
    """
    if policy_name not in THUNDERBOLT_SECURITY_POLICIES:
        raise ValueError(f"Unknown security policy: {policy_name}")
    return THUNDERBOLT_SECURITY_POLICIES[policy_name].copy()


def validate_daisy_chain_config(device_count: int, hub_count: int, display_count: int) -> bool:
    """
    Validate Thunderbolt daisy chain configuration

    Args:
        device_count: Number of devices in chain
        hub_count: Number of hubs in chain
        display_count: Number of displays in chain

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration exceeds limits
    """
    specs = ThunderboltDaisyChainSpecs()

    if device_count > specs.MAX_DEVICES:
        raise ValueError(f"Too many devices: {device_count} > {specs.MAX_DEVICES}")
    if hub_count > specs.MAX_HUBS:
        raise ValueError(f"Too many hubs: {hub_count} > {specs.MAX_HUBS}")
    if display_count > specs.MAX_DISPLAYS:
        raise ValueError(f"Too many displays: {display_count} > {specs.MAX_DISPLAYS}")

    return True


def calculate_chain_power_budget(device_count: int) -> float:
    """
    Calculate power budget for Thunderbolt daisy chain

    Args:
        device_count: Number of devices in chain

    Returns:
        Required power budget in watts
    """
    specs = ThunderboltDaisyChainSpecs()
    return min(device_count * specs.POWER_BUDGET_PER_DEVICE, specs.MAX_CHAIN_POWER)


def get_certification_requirements(level: ThunderboltCertificationLevel) -> List[str]:
    """
    Get certification requirements for given level

    Args:
        level: Certification level

    Returns:
        List of required tests
    """
    if not isinstance(level, ThunderboltCertificationLevel):
        raise ValueError(f"Invalid certification level: {level}")

    base_tests = ThunderboltCertificationSpecs().REQUIRED_TESTS.copy()

    if level == ThunderboltCertificationLevel.PREMIUM:
        base_tests.extend(["advanced_interop", "stress_testing"])
    elif level == ThunderboltCertificationLevel.PRO:
        base_tests.extend(["advanced_interop", "stress_testing", "enterprise_security"])

    return base_tests


__all__ = [
    # Enumerations
    "ThunderboltSecurityLevel",
    "ThunderboltDeviceType",
    "ThunderboltCertificationLevel",
    "ThunderboltAuthMethod",
    # Specifications
    "ThunderboltSecuritySpecs",
    "ThunderboltDaisyChainSpecs",
    "ThunderboltDisplaySpecs",
    "ThunderboltPowerSpecs",
    "ThunderboltCertificationSpecs",
    # Constants
    "THUNDERBOLT_SPECS",
    "THUNDERBOLT_AUTH_KEYS",
    "THUNDERBOLT_SECURITY_POLICIES",
    # Utility functions
    "validate_thunderbolt_security_level",
    "get_security_policy",
    "validate_daisy_chain_config",
    "calculate_chain_power_budget",
    "get_certification_requirements",
]
