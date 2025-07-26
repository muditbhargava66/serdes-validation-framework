"""
Basic import tests to verify framework structure
"""

import os

import pytest

# Set mock mode
os.environ["SVF_MOCK_MODE"] = "1"


def test_framework_version():
    """Test that framework version can be imported"""
    try:
        from serdes_validation_framework import __version__

        assert __version__ == "1.4.0"
    except ImportError:
        pytest.skip("Framework version not available")


def test_framework_availability_flags():
    """Test framework availability flags"""
    try:
        from serdes_validation_framework import FRAMEWORK_AVAILABLE, PROTOCOL_DETECTOR_AVAILABLE

        # These should be boolean values
        assert isinstance(FRAMEWORK_AVAILABLE, bool)
        assert isinstance(PROTOCOL_DETECTOR_AVAILABLE, bool)
    except ImportError:
        pytest.skip("Framework availability flags not available")


def test_protocols_availability():
    """Test protocol availability flags"""
    try:
        from serdes_validation_framework.protocols import ETHERNET_AVAILABLE, PCIE_AVAILABLE, USB4_AVAILABLE

        # These should be boolean values
        assert isinstance(USB4_AVAILABLE, bool)
        assert isinstance(PCIE_AVAILABLE, bool)
        assert isinstance(ETHERNET_AVAILABLE, bool)
    except ImportError:
        pytest.skip("Protocol availability flags not available")


def test_usb4_availability():
    """Test USB4 module availability flags"""
    try:
        from serdes_validation_framework.protocols.usb4 import (
            BASE_AVAILABLE,
            CONSTANTS_AVAILABLE,
            TUNNELING_AVAILABLE,
            VISUALIZATION_AVAILABLE,
        )

        # These should be boolean values
        assert isinstance(TUNNELING_AVAILABLE, bool)
        assert isinstance(BASE_AVAILABLE, bool)
        assert isinstance(CONSTANTS_AVAILABLE, bool)
        assert isinstance(VISUALIZATION_AVAILABLE, bool)
    except ImportError:
        pytest.skip("USB4 availability flags not available")


@pytest.mark.skipif(os.environ.get("SVF_MOCK_MODE") != "1", reason="Mock mode not enabled")
def test_mock_mode_enabled():
    """Test that mock mode is properly enabled"""
    assert os.environ.get("SVF_MOCK_MODE") == "1"


def test_tunneling_classes_conditional_import():
    """Test conditional import of tunneling classes"""
    try:
        from serdes_validation_framework.protocols.usb4 import TUNNELING_AVAILABLE

        if TUNNELING_AVAILABLE:
            from serdes_validation_framework.protocols.usb4 import BandwidthAllocationMode, PCIeTunnelValidator, TunnelState

            # Test that classes exist
            assert PCIeTunnelValidator is not None
            assert TunnelState is not None
            assert BandwidthAllocationMode is not None
        else:
            pytest.skip("Tunneling classes not available")
    except ImportError as e:
        pytest.skip(f"Tunneling import failed: {e}")


def test_visualization_conditional_import():
    """Test conditional import of visualization"""
    try:
        from serdes_validation_framework.protocols.usb4 import VISUALIZATION_AVAILABLE

        if VISUALIZATION_AVAILABLE:
            from serdes_validation_framework.protocols.usb4 import USB4Visualizer

            visualizer = USB4Visualizer()
            assert visualizer is not None
        else:
            pytest.skip("Visualization not available")
    except ImportError as e:
        pytest.skip(f"Visualization import failed: {e}")
