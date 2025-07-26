"""
Test USB4 Tunneling Validation Module

This module contains comprehensive tests for USB4 tunneling validation,
including PCIe, DisplayPort, and USB 3.2 tunneling support.
"""

import numpy as np
import pytest

from src.serdes_validation_framework.protocols.usb4.base import (
    SignalQuality,
    USB4Config,
)
from src.serdes_validation_framework.protocols.usb4.constants import (
    USB4TunnelingMode,
    USB4TunnelingSpecs,
)
from src.serdes_validation_framework.protocols.usb4.tunneling import (
    BandwidthAllocationMode,
    BandwidthMonitor,
    DisplayPortTunnelValidator,
    MultiProtocolBandwidthManager,
    PCIeTLPHeader,
    PCIeTunnelValidator,
    TunnelConfig,
    TunnelState,
    USB32TunnelValidator,
)


class TestPCIeTunnelValidator:
    """Test PCIe tunneling validator"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        from src.serdes_validation_framework.protocols.usb4.constants import USB4SignalMode

        return USB4Config(signal_mode=USB4SignalMode.GEN2X2, sample_rate=100e9, capture_length=10000)

    @pytest.fixture
    def validator(self, config):
        """Create PCIe tunnel validator"""
        return PCIeTunnelValidator(config)

    def test_initialization(self, validator):
        """Test validator initialization"""
        assert validator.initialize()
        assert validator.is_initialized
        assert validator.tunnel_state == TunnelState.DISCONNECTED

    def test_cleanup(self, validator):
        """Test validator cleanup"""
        validator.initialize()
        validator.cleanup()
        assert not validator.is_initialized
        assert validator.tunnel_state == TunnelState.DISCONNECTED
        assert len(validator.active_tlps) == 0

    def test_validate_tunnel_pcie_mode(self, validator):
        """Test tunnel validation with PCIe mode"""
        validator.initialize()

        # Create test data
        test_data = np.random.randn(1000).astype(np.float64)

        # Validate tunnel
        results = validator.validate_tunnel(USB4TunnelingMode.PCIE, test_data)

        assert results["tunnel_mode"] == USB4TunnelingMode.PCIE
        assert results["data_size"] == len(test_data)
        assert "tlp_count" in results
        assert "validation_time" in results

    def test_validate_tunnel_invalid_mode(self, validator):
        """Test tunnel validation with invalid mode"""
        validator.initialize()
        test_data = np.random.randn(100).astype(np.float64)

        with pytest.raises(ValueError, match="PCIe validator only supports PCIe tunneling mode"):
            validator.validate_tunnel(USB4TunnelingMode.DISPLAYPORT, test_data)

    def test_measure_tunnel_bandwidth(self, validator):
        """Test bandwidth measurement"""
        validator.initialize()

        bandwidth = validator.measure_tunnel_bandwidth(USB4TunnelingMode.PCIE)
        assert isinstance(bandwidth, float)
        assert bandwidth >= 0

    def test_tunnel_establishment(self, validator):
        """Test tunnel establishment process"""
        validator.initialize()

        success = validator.test_tunnel_establishment(USB4TunnelingMode.PCIE)
        assert success
        assert validator.tunnel_state == TunnelState.CONNECTED

    def test_tlp_integrity_validation(self, validator):
        """Test TLP integrity validation"""
        validator.initialize()

        # Create test TLP data (minimum 12 bytes for header)
        tlp_data = bytes(
            [
                0x00,
                0x00,
                0x00,
                0x01,  # Header word 0
                0x01,
                0x00,
                0xFF,
                0x0F,  # Header word 1
                0x00,
                0x10,
                0x00,
                0x00,
            ]
        )  # Header word 2

        results = validator.validate_tlp_integrity(tlp_data)

        assert results["tlp_size"] == len(tlp_data)
        assert "header_valid" in results
        assert "crc_valid" in results
        assert "sequence_valid" in results
        assert "format_valid" in results

    def test_tlp_integrity_short_data(self, validator):
        """Test TLP integrity with short data"""
        validator.initialize()

        # Create too-short TLP data
        tlp_data = bytes([0x00, 0x01, 0x02])

        results = validator.validate_tlp_integrity(tlp_data)

        assert results["tlp_size"] == len(tlp_data)
        assert "error" in results
        assert results["error"] == "TLP too short"

    def test_bandwidth_allocation(self, validator):
        """Test bandwidth allocation"""
        validator.initialize()

        allocation_config = {"bandwidth": 10e9}  # 10 Gbps
        results = validator.test_bandwidth_allocation(allocation_config)

        assert results["requested_bandwidth"] == 10e9
        assert results["allocated_bandwidth"] > 0
        assert results["success"]
        assert "utilization_efficiency" in results
        assert "allocation_time" in results

    def test_comprehensive_pcie_test(self, validator):
        """Test comprehensive PCIe test"""
        validator.initialize()

        # Run short test
        results = validator.run_comprehensive_pcie_test(test_duration=1.0)

        assert isinstance(results.tunnel_mode, USB4TunnelingMode)
        assert results.tunnel_mode == USB4TunnelingMode.PCIE
        assert results.test_duration >= 1.0
        assert results.throughput > 0
        assert len(results.latency_measurements) > 0
        assert 0 <= results.tlp_integrity_rate <= 1.0
        assert 0 <= results.bandwidth_efficiency <= 1.0


class TestDisplayPortTunnelValidator:
    """Test DisplayPort tunneling validator"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        from src.serdes_validation_framework.protocols.usb4.constants import USB4SignalMode

        return USB4Config(signal_mode=USB4SignalMode.GEN2X2, sample_rate=100e9, capture_length=10000)

    @pytest.fixture
    def validator(self, config):
        """Create DisplayPort tunnel validator"""
        return DisplayPortTunnelValidator(config)

    def test_initialization(self, validator):
        """Test validator initialization"""
        assert validator.initialize()
        assert validator.is_initialized
        assert validator.tunnel_state == TunnelState.DISCONNECTED

    def test_validate_tunnel_displayport_mode(self, validator):
        """Test tunnel validation with DisplayPort mode"""
        validator.initialize()

        test_data = np.random.randn(1000).astype(np.float64)
        results = validator.validate_tunnel(USB4TunnelingMode.DISPLAYPORT, test_data)

        assert results["tunnel_mode"] == USB4TunnelingMode.DISPLAYPORT
        assert results["data_size"] == len(test_data)
        assert "validation_time" in results

    def test_validate_tunnel_invalid_mode(self, validator):
        """Test tunnel validation with invalid mode"""
        validator.initialize()
        test_data = np.random.randn(100).astype(np.float64)

        with pytest.raises(ValueError, match="DisplayPort validator only supports DisplayPort tunneling mode"):
            validator.validate_tunnel(USB4TunnelingMode.PCIE, test_data)

    def test_tunnel_establishment(self, validator):
        """Test DisplayPort tunnel establishment"""
        validator.initialize()

        success = validator.test_tunnel_establishment(USB4TunnelingMode.DISPLAYPORT)
        assert success
        assert validator.tunnel_state == TunnelState.CONNECTED

    def test_video_signal_integrity(self, validator):
        """Test video signal integrity validation"""
        validator.initialize()

        # Create test video data
        video_data = np.random.randn(3840 * 2160).astype(np.float64)

        results = validator.validate_video_signal_integrity(video_data)

        assert results["data_size"] == len(video_data)
        assert isinstance(results["signal_quality"], SignalQuality)
        assert "sync_errors" in results
        assert "pixel_errors" in results
        assert "frame_drops" in results

    def test_mst_stream_validation(self, validator):
        """Test MST stream validation"""
        validator.initialize()

        # Test with valid stream count
        results = validator.validate_mst_streams(2)

        assert results["requested_streams"] == 2
        assert results["active_streams"] >= 0
        assert "bandwidth_per_stream" in results
        assert "total_bandwidth" in results
        assert "mst_valid" in results

    def test_mst_stream_validation_excessive(self, validator):
        """Test MST stream validation with excessive streams"""
        validator.initialize()

        # Test with too many streams
        results = validator.validate_mst_streams(10)

        assert results["requested_streams"] == 10
        assert "error" in results

    def test_comprehensive_displayport_test(self, validator):
        """Test comprehensive DisplayPort test"""
        validator.initialize()

        # Run short test
        results = validator.run_comprehensive_displayport_test(test_duration=1.0)

        assert results.tunnel_mode == USB4TunnelingMode.DISPLAYPORT
        assert results.test_duration >= 1.0
        assert results.throughput > 0
        assert len(results.latency_measurements) > 0
        assert isinstance(results.video_signal_quality, SignalQuality)
        assert results.resolution == (3840, 2160)
        assert results.color_depth == 8


class TestUSB32TunnelValidator:
    """Test USB 3.2 tunneling validator"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        from src.serdes_validation_framework.protocols.usb4.constants import USB4SignalMode

        return USB4Config(signal_mode=USB4SignalMode.GEN2X2, sample_rate=100e9, capture_length=10000)

    @pytest.fixture
    def validator(self, config):
        """Create USB 3.2 tunnel validator"""
        return USB32TunnelValidator(config)

    def test_initialization(self, validator):
        """Test validator initialization"""
        assert validator.initialize()
        assert validator.is_initialized
        assert validator.tunnel_state == TunnelState.DISCONNECTED

    def test_validate_tunnel_usb32_mode(self, validator):
        """Test tunnel validation with USB 3.2 mode"""
        validator.initialize()

        test_data = np.random.randn(1000).astype(np.float64)
        results = validator.validate_tunnel(USB4TunnelingMode.USB32, test_data)

        assert results["tunnel_mode"] == USB4TunnelingMode.USB32
        assert results["data_size"] == len(test_data)
        assert "validation_time" in results

    def test_validate_tunnel_invalid_mode(self, validator):
        """Test tunnel validation with invalid mode"""
        validator.initialize()
        test_data = np.random.randn(100).astype(np.float64)

        with pytest.raises(ValueError, match="USB 3.2 validator only supports USB 3.2 tunneling mode"):
            validator.validate_tunnel(USB4TunnelingMode.PCIE, test_data)

    def test_tunnel_establishment(self, validator):
        """Test USB 3.2 tunnel establishment"""
        validator.initialize()

        success = validator.test_tunnel_establishment(USB4TunnelingMode.USB32)
        assert success
        assert validator.tunnel_state == TunnelState.CONNECTED

    def test_device_enumeration(self, validator):
        """Test USB device enumeration"""
        validator.initialize()

        results = validator.validate_device_enumeration(5)

        assert results["requested_devices"] == 5
        assert results["enumerated_devices"] >= 0
        assert results["configuration_success"] >= 0
        assert "enumeration_time" in results
        assert "success_rate" in results

    def test_device_enumeration_excessive(self, validator):
        """Test device enumeration with excessive device count"""
        validator.initialize()

        results = validator.validate_device_enumeration(200)

        assert results["requested_devices"] == 200
        assert "error" in results

    def test_usb32_performance(self, validator):
        """Test USB 3.2 performance benchmarking"""
        validator.initialize()

        results = validator.test_usb32_performance(test_duration=1.0)

        assert results["test_duration"] >= 1.0
        assert results["throughput"] > 0
        assert results["latency_avg"] >= 0
        assert results["latency_max"] >= results["latency_avg"]
        assert 0 <= results["error_rate"] <= 1.0
        assert 0 <= results["efficiency"] <= 1.0

    def test_comprehensive_usb32_test(self, validator):
        """Test comprehensive USB 3.2 test"""
        validator.initialize()

        # Run short test
        results = validator.run_comprehensive_usb32_test(test_duration=1.0)

        assert results.tunnel_mode == USB4TunnelingMode.USB32
        assert results.test_duration >= 0.5  # Allow for some timing variance
        assert results.throughput > 0
        assert len(results.latency_measurements) > 0
        assert 0 <= results.enumeration_success_rate <= 1.0
        assert 0 <= results.protocol_compliance_rate <= 1.0
        assert 0 <= results.backward_compatibility_score <= 1.0
        assert 0 <= results.power_delivery_efficiency <= 1.0


class TestMultiProtocolBandwidthManager:
    """Test multi-protocol bandwidth manager"""

    @pytest.fixture
    def manager(self):
        """Create bandwidth manager"""
        return MultiProtocolBandwidthManager(total_bandwidth=40e9)

    def test_initialization(self, manager):
        """Test manager initialization"""
        assert manager.total_bandwidth == 40e9
        assert len(manager.allocated_bandwidth) == 0
        assert len(manager.active_tunnels) == 0
        assert manager.allocation_mode == BandwidthAllocationMode.DYNAMIC

    def test_allocate_bandwidth_success(self, manager):
        """Test successful bandwidth allocation"""
        results = manager.allocate_bandwidth(USB4TunnelingMode.PCIE, 10e9)

        assert results["tunnel_mode"] == USB4TunnelingMode.PCIE
        assert results["requested_bandwidth"] == 10e9
        assert results["allocated_bandwidth"] > 0
        assert results["allocation_success"]
        assert results["remaining_bandwidth"] > 0

    def test_allocate_bandwidth_insufficient(self, manager):
        """Test bandwidth allocation with insufficient request"""
        specs = USB4TunnelingSpecs()

        # Request less than minimum
        results = manager.allocate_bandwidth(USB4TunnelingMode.PCIE, specs.PCIE_MIN_BANDWIDTH / 2)

        assert not results["allocation_success"]
        assert "error" in results

    def test_deallocate_bandwidth(self, manager):
        """Test bandwidth deallocation"""
        # First allocate
        manager.allocate_bandwidth(USB4TunnelingMode.PCIE, 10e9)

        # Then deallocate
        success = manager.deallocate_bandwidth(USB4TunnelingMode.PCIE)
        assert success
        assert USB4TunnelingMode.PCIE not in manager.allocated_bandwidth

    def test_deallocate_nonexistent(self, manager):
        """Test deallocation of non-existent tunnel"""
        success = manager.deallocate_bandwidth(USB4TunnelingMode.PCIE)
        assert not success

    def test_congestion_management_high(self, manager):
        """Test congestion management with high congestion"""
        # Allocate some bandwidth first
        manager.allocate_bandwidth(USB4TunnelingMode.PCIE, 10e9)

        results = manager.test_congestion_management(0.9)  # High congestion

        assert results["congestion_level"] == 0.9
        assert results["throttling_applied"]
        assert results["bandwidth_reduction"] > 0
        assert results["flow_control_active"]
        assert results["recovery_time"] > 0

    def test_congestion_management_medium(self, manager):
        """Test congestion management with medium congestion"""
        results = manager.test_congestion_management(0.7)  # Medium congestion

        assert results["congestion_level"] == 0.7
        assert not results["throttling_applied"]
        assert results["flow_control_active"]

    def test_congestion_management_low(self, manager):
        """Test congestion management with low congestion"""
        results = manager.test_congestion_management(0.3)  # Low congestion

        assert results["congestion_level"] == 0.3
        assert not results["throttling_applied"]
        assert not results["flow_control_active"]

    def test_simultaneous_tunnels(self, manager):
        """Test simultaneous tunnel operation"""
        tunnel_configs = [
            TunnelConfig(USB4TunnelingMode.PCIE, 8e9, 1e-6, priority=3),
            TunnelConfig(USB4TunnelingMode.DISPLAYPORT, 5e9, 100e-6, priority=2),
            TunnelConfig(USB4TunnelingMode.USB32, 5e9, 10e-6, priority=1),
        ]

        results = manager.test_simultaneous_tunnels(tunnel_configs)

        assert results["requested_tunnels"] == 3
        assert results["active_tunnels"] >= 0
        assert results["total_bandwidth_used"] >= 0
        assert 0 <= results["bandwidth_efficiency"] <= 1.0

    def test_bandwidth_utilization_measurement(self, manager):
        """Test bandwidth utilization measurement"""
        # Allocate some bandwidth (use valid minimum bandwidths)
        manager.allocate_bandwidth(USB4TunnelingMode.PCIE, 10e9)
        manager.allocate_bandwidth(USB4TunnelingMode.DISPLAYPORT, 6e9)  # Above 5.4 Gbps minimum

        metrics = manager.measure_bandwidth_utilization()

        assert metrics["total_bandwidth"] == 40e9
        assert metrics["allocated_bandwidth"] == 16e9
        assert metrics["available_bandwidth"] == 24e9
        assert metrics["utilization_percentage"] == 40.0
        assert metrics["active_tunnel_count"] == 2


class TestBandwidthMonitor:
    """Test bandwidth monitor"""

    @pytest.fixture
    def monitor(self):
        """Create bandwidth monitor"""
        return BandwidthMonitor()

    def test_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.current_bandwidth == 0.0
        assert monitor.peak_bandwidth == 0.0
        assert monitor.average_bandwidth == 0.0
        assert len(monitor.measurement_history) == 0

    def test_update_bandwidth(self, monitor):
        """Test bandwidth update"""
        monitor.update_bandwidth(10e9)

        assert monitor.current_bandwidth == 10e9
        assert monitor.peak_bandwidth == 10e9
        assert monitor.average_bandwidth == 10e9
        assert len(monitor.measurement_history) == 1

    def test_multiple_updates(self, monitor):
        """Test multiple bandwidth updates"""
        bandwidths = [5e9, 10e9, 8e9, 12e9, 6e9]

        for bw in bandwidths:
            monitor.update_bandwidth(bw)

        assert monitor.current_bandwidth == 6e9
        assert monitor.peak_bandwidth == 12e9
        assert monitor.average_bandwidth == np.mean(bandwidths)
        assert len(monitor.measurement_history) == len(bandwidths)

    def test_get_bandwidth_statistics(self, monitor):
        """Test bandwidth statistics"""
        bandwidths = [5e9, 10e9, 8e9, 12e9, 6e9]

        for bw in bandwidths:
            monitor.update_bandwidth(bw)

        stats = monitor.get_bandwidth_statistics()

        assert stats["current"] == 6e9
        assert stats["peak"] == 12e9
        assert stats["average"] == np.mean(bandwidths)
        assert stats["std_dev"] == np.std(bandwidths)

    def test_history_limit(self, monitor):
        """Test measurement history limit"""
        # Add more than 1000 measurements
        for i in range(1200):
            monitor.update_bandwidth(float(i))

        # Should be limited to 1000
        assert len(monitor.measurement_history) == 1000
        # Should contain the most recent 1000 measurements
        assert monitor.measurement_history[0] == 200.0  # 1200 - 1000


class TestPCIeTLPHeader:
    """Test PCIe TLP header structure"""

    def test_tlp_header_creation(self):
        """Test TLP header creation"""
        header = PCIeTLPHeader(
            fmt=0, type=0, tc=0, attr=0, length=1, requester_id=0x0100, tag=0x01, last_dw_be=0xF, first_dw_be=0xF, address=0x1000
        )

        assert header.fmt == 0
        assert header.type == 0
        assert header.tc == 0
        assert header.attr == 0
        assert header.length == 1
        assert header.requester_id == 0x0100
        assert header.tag == 0x01
        assert header.last_dw_be == 0xF
        assert header.first_dw_be == 0xF
        assert header.address == 0x1000
        assert header.data is None


class TestTunnelConfig:
    """Test tunnel configuration"""

    def test_tunnel_config_creation(self):
        """Test tunnel config creation"""
        config = TunnelConfig(
            tunnel_mode=USB4TunnelingMode.PCIE,
            bandwidth_requirement=10e9,
            latency_requirement=1e-6,
            priority=3,
            enable_flow_control=True,
            buffer_size=8192,
        )

        assert config.tunnel_mode == USB4TunnelingMode.PCIE
        assert config.bandwidth_requirement == 10e9
        assert config.latency_requirement == 1e-6
        assert config.priority == 3
        assert config.enable_flow_control
        assert config.buffer_size == 8192

    def test_tunnel_config_defaults(self):
        """Test tunnel config with defaults"""
        config = TunnelConfig(tunnel_mode=USB4TunnelingMode.DISPLAYPORT, bandwidth_requirement=5e9, latency_requirement=100e-6)

        assert config.priority == 0
        assert config.enable_flow_control
        assert config.buffer_size == 4096


if __name__ == "__main__":
    pytest.main([__file__])
