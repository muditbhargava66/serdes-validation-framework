"""
Mock Multi-Protocol Comparator

Provides mock implementation of multi-protocol comparison functionality for testing.
"""

from typing import Any, Dict, List

import numpy as np


class MultiProtocolComparator:
    """Mock multi-protocol comparator for testing"""

    def __init__(self):
        self.protocols = ["USB4", "PCIe", "Ethernet", "Thunderbolt"]
        self.initialized = True

    def compare_protocols(self, data: np.ndarray) -> Dict[str, Any]:
        """Mock protocol comparison"""
        # Simple mock logic based on data characteristics
        data_mean = np.mean(np.abs(data))

        if data_mean > 0.8:
            best_match = "USB4"
            confidence = 0.95
        elif data_mean > 0.6:
            best_match = "PCIe"
            confidence = 0.88
        elif data_mean > 0.4:
            best_match = "Thunderbolt"
            confidence = 0.82
        else:
            best_match = "Ethernet"
            confidence = 0.75

        return {
            "best_match": best_match,
            "confidence": confidence,
            "scores": {
                "USB4": confidence if best_match == "USB4" else confidence * 0.7,
                "PCIe": confidence if best_match == "PCIe" else confidence * 0.6,
                "Thunderbolt": confidence if best_match == "Thunderbolt" else confidence * 0.5,
                "Ethernet": confidence if best_match == "Ethernet" else confidence * 0.4,
            },
        }

    def benchmark_performance(self, protocols: List[str]) -> Dict[str, Dict[str, float]]:
        """Mock performance benchmarking"""
        performance_data = {
            "USB4": {"throughput": 40e9, "latency": 1e-6, "power": 5.0},
            "PCIe": {"throughput": 64e9, "latency": 100e-9, "power": 10.0},
            "Thunderbolt": {"throughput": 40e9, "latency": 1e-6, "power": 7.0},
            "Ethernet": {"throughput": 224e9, "latency": 10e-6, "power": 15.0},
        }

        return {protocol: performance_data.get(protocol, {"throughput": 0, "latency": 0, "power": 0}) for protocol in protocols}

    def analyze_compatibility(self, protocol1: str, protocol2: str) -> Dict[str, Any]:
        """Mock compatibility analysis"""
        compatible_pairs = [("USB4", "Thunderbolt"), ("PCIe", "USB4"), ("Thunderbolt", "PCIe")]

        is_compatible = (protocol1, protocol2) in compatible_pairs or (protocol2, protocol1) in compatible_pairs

        return {
            "compatible": is_compatible,
            "compatibility_score": 0.9 if is_compatible else 0.3,
            "interop_features": ["tunneling", "power_delivery"] if is_compatible else [],
        }
