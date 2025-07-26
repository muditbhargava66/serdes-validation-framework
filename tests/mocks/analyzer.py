"""
Mock Data Analyzer

Provides mock implementation of data analysis functionality for testing.
"""

from typing import Any, Dict

import numpy as np


class DataAnalyzer:
    """Mock data analyzer for testing"""

    def __init__(self):
        self.initialized = True

    def analyze_signal(self, data: np.ndarray) -> Dict[str, Any]:
        """Mock signal analysis"""
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "samples": len(data),
        }

    def calculate_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """Mock statistics calculation"""
        return {
            "count": len(data),
            "mean": float(np.mean(data)),
            "variance": float(np.var(data)),
            "skewness": 0.0,
            "kurtosis": 0.0,
        }


class SignalProcessor:
    """Mock signal processor for testing"""

    def __init__(self):
        self.sample_rate = 100e9

    def process_signal(self, data: np.ndarray) -> np.ndarray:
        """Mock signal processing"""
        # Simple filtering simulation
        return data * 0.95

    def apply_filter(self, data: np.ndarray, filter_type: str = "lowpass") -> np.ndarray:
        """Mock filter application"""
        if filter_type == "lowpass":
            return data * 0.9
        elif filter_type == "highpass":
            return data * 1.1
        else:
            return data
