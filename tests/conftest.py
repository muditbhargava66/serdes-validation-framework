import os
import sys
from pathlib import Path

import pytest

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Enable mock mode for all tests
os.environ["SVF_MOCK_MODE"] = "1"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment with mock mode"""
    os.environ["SVF_MOCK_MODE"] = "1"
    yield


def pytest_collection_modifyitems(config, items):
    """Add skip markers for hardware-dependent tests"""
    skip_hardware = pytest.mark.skip(reason="Hardware not available in test environment")
    skip_visualization = pytest.mark.skip(reason="Visualization dependencies not available")

    for item in items:
        # Skip tests that require actual hardware
        if any(marker in str(item.fspath).lower() for marker in ["hardware", "instrument", "scope"]):
            item.add_marker(skip_hardware)

        # Skip visualization tests if dependencies not available
        if "visualization" in str(item.fspath).lower():
            try:
                import matplotlib
                import plotly
            except ImportError:
                item.add_marker(skip_visualization)

        # Skip legacy tests by default
        if "legacy" in str(item.fspath):
            item.add_marker(pytest.mark.skip(reason="Legacy test - may have import issues"))


@pytest.fixture
def mock_usb4_config():
    """Provide mock USB4 configuration"""
    return {"signal_mode": "GEN3X2", "lane_count": 2, "bit_rate": 20e9, "voltage_swing": 0.8, "mock_mode": True}


@pytest.fixture
def mock_tunnel_config():
    """Provide mock tunnel configuration"""
    return {
        "tunnel_mode": "PCIE",
        "bandwidth_requirement": 10e9,
        "latency_requirement": 1e-6,
        "priority": 1,
        "enable_flow_control": True,
        "buffer_size": 4096,
    }
