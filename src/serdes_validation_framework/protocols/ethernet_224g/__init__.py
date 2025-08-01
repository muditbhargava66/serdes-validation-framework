# src/serdes_validation_framework/protocols/ethernet_224g/__init__.py

"""
224G Ethernet Protocol Implementation

This module provides constants, compliance specifications, training patterns,
and enhanced equalization algorithms for 224G Ethernet validation.
"""

from .compliance import ComplianceLimit, ComplianceSpecification, ComplianceTestConfig
from .constants import (
    COMPLIANCE_PATTERNS,
    ETHERNET_224G_SPECS,
    TRAINING_PATTERNS,
    EyeSpecs,
    JitterSpecs,
    PAM4Specs,
    TrainingSpecs,
    calculate_ui_parameters,
    validate_pam4_levels,
)
from .training import (
    EnhancedEqualizer,
    # TrainingConfig,
    # TrainingStatus,
    # LinkTraining,
    # New enhanced equalization components
    EqualizerConfig,
    EqualizerState,
)

__version__ = "1.4.0"

__all__ = [
    # Constants
    "PAM4Specs",
    "JitterSpecs",
    "EyeSpecs",
    "TrainingSpecs",
    "ETHERNET_224G_SPECS",
    "TRAINING_PATTERNS",
    "COMPLIANCE_PATTERNS",
    "validate_pam4_levels",
    "calculate_ui_parameters",
    # Compliance
    "ComplianceTestConfig",
    "ComplianceLimit",
    "ComplianceSpecification",
    # Training and Equalization
    "TrainingConfig",
    "TrainingStatus",
    "LinkTraining",
    "EqualizerConfig",
    "EqualizerState",
    "EnhancedEqualizer",
    # Version
    "__version__",
]

# Protocol metadata
PROTOCOL_INFO = {
    "name": "224G Ethernet",
    "standard": "IEEE 802.3",
    "version": __version__,
    "specs": ETHERNET_224G_SPECS,
    "supported_features": [
        "PAM4 modulation",
        "Link training",
        "Compliance testing",
        "Eye diagram analysis",
        "Jitter measurements",
        # New features
        "Advanced equalization algorithms",
        "LMS adaptation",
        "RLS adaptation",
        "CMA adaptation",
        "Performance analysis",
    ],
}
