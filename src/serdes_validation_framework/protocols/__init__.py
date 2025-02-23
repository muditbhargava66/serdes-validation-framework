# src/serdes_validation_framework/protocols/__init__.py

"""
Protocol-specific modules for SerDes validation.

This package contains protocol-specific implementations and specifications
for various high-speed serial interfaces.
"""

from typing import Any, Dict, List

# Protocol registry
SUPPORTED_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    'ethernet_224g': {
        'name': '224G Ethernet',
        'version': '1.1.0',
        'description': 'IEEE 802.3 224G Ethernet protocol specification',
        'symbol_rate': 112e9,  # 112 GBaud
        'modulation': 'PAM4',
        'features': [
            'Advanced equalization',
            'Multi-algorithm adaptation',
            'Performance analysis',
            'Compliance testing'
        ],
        'supported_algorithms': [
            'LMS (Least Mean Squares)',
            'RLS (Recursive Least Squares)',
            'CMA (Constant Modulus Algorithm)'
        ]
    }
}

def get_protocol_info(protocol_name: str) -> Dict[str, Any]:
    """
    Get information about a supported protocol
    
    Args:
        protocol_name: Name of the protocol
        
    Returns:
        Dictionary containing protocol information
    """
    assert isinstance(protocol_name, str), "Protocol name must be a string"
    if protocol_name not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported protocol: {protocol_name}")
    return SUPPORTED_PROTOCOLS[protocol_name]

def list_supported_features(protocol_name: str) -> List[str]:
    """
    List supported features for a protocol
    
    Args:
        protocol_name: Name of the protocol
        
    Returns:
        List of supported features
    """
    protocol_info = get_protocol_info(protocol_name)
    return protocol_info.get('features', [])

__all__ = [
    'SUPPORTED_PROTOCOLS',
    'get_protocol_info',
    'list_supported_features'
]
