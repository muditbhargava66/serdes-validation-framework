#!/usr/bin/env python3
"""
USB4 Tunneling Validation Example

This example demonstrates how to use the USB4 tunneling validation capabilities
to test PCIe, DisplayPort, and USB 3.2 tunneling over USB4.

Features demonstrated:
- PCIe tunneling with TLP integrity checking
- DisplayPort tunneling with video signal analysis
- USB 3.2 tunneling with device enumeration
- Multi-protocol bandwidth management
- Comprehensive tunneling test suites
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.serdes_validation_framework.protocols.usb4.base import USB4Config
from src.serdes_validation_framework.protocols.usb4.constants import (
    USB4SignalMode,
    USB4TunnelingMode,
)
from src.serdes_validation_framework.protocols.usb4.tunneling import (
    DisplayPortTunnelValidator,
    MultiProtocolBandwidthManager,
    PCIeTunnelValidator,
    TunnelConfig,
    USB32TunnelValidator,
)


def create_usb4_config() -> USB4Config:
    """Create USB4 configuration for tunneling tests"""
    return USB4Config(
        signal_mode=USB4SignalMode.GEN2X2,
        sample_rate=100e9,  # 100 GS/s
        capture_length=10000,
        enable_ssc=True,
        enable_equalization=True,
        debug_mode=False
    )


def demonstrate_pcie_tunneling():
    """Demonstrate PCIe tunneling validation"""
    print("=" * 60)
    print("PCIe Tunneling Validation Demo")
    print("=" * 60)
    
    # Create validator
    config = create_usb4_config()
    validator = PCIeTunnelValidator(config)
    
    # Initialize validator
    if not validator.initialize():
        print("Failed to initialize PCIe tunnel validator")
        return
        
    print("✓ PCIe tunnel validator initialized")
    
    # Test tunnel establishment
    success = validator.test_tunnel_establishment(USB4TunnelingMode.PCIE)
    print(f"✓ Tunnel establishment: {'SUCCESS' if success else 'FAILED'}")
    
    # Generate test PCIe data
    test_data = np.random.randn(1000).astype(np.float64)
    
    # Validate tunneled data
    results = validator.validate_tunnel(USB4TunnelingMode.PCIE, test_data)
    print(f"✓ Validated {results['tlp_count']} TLPs")
    print(f"✓ Data size: {results['data_size']} samples")
    
    # Test TLP integrity
    tlp_data = bytes([0x00, 0x00, 0x00, 0x01,  # Header word 0
                     0x01, 0x00, 0xFF, 0x0F,  # Header word 1
                     0x00, 0x10, 0x00, 0x00]) # Header word 2
    
    tlp_results = validator.validate_tlp_integrity(tlp_data)
    print(f"✓ TLP header valid: {tlp_results['header_valid']}")
    print(f"✓ TLP format valid: {tlp_results['format_valid']}")
    
    # Test bandwidth allocation
    bandwidth_config = {'bandwidth': 10e9}  # 10 Gbps
    bw_results = validator.test_bandwidth_allocation(bandwidth_config)
    print(f"✓ Bandwidth allocation: {bw_results['allocated_bandwidth']/1e9:.1f} Gbps")
    print(f"✓ Utilization efficiency: {bw_results['utilization_efficiency']:.2%}")
    
    # Run comprehensive test
    print("\nRunning comprehensive PCIe test (5 seconds)...")
    comprehensive_results = validator.run_comprehensive_pcie_test(test_duration=5.0)
    
    print(f"✓ Test duration: {comprehensive_results.test_duration:.2f}s")
    print(f"✓ Throughput: {comprehensive_results.throughput:.0f} TLPs/s")
    print(f"✓ TLP integrity rate: {comprehensive_results.tlp_integrity_rate:.2%}")
    print(f"✓ Bandwidth efficiency: {comprehensive_results.bandwidth_efficiency:.2%}")
    print(f"✓ Average latency: {np.mean(comprehensive_results.latency_measurements)*1000:.2f} ms")
    
    validator.cleanup()
    print("✓ PCIe tunnel validator cleaned up\n")


def demonstrate_displayport_tunneling():
    """Demonstrate DisplayPort tunneling validation"""
    print("=" * 60)
    print("DisplayPort Tunneling Validation Demo")
    print("=" * 60)
    
    # Create validator
    config = create_usb4_config()
    validator = DisplayPortTunnelValidator(config)
    
    # Initialize validator
    if not validator.initialize():
        print("Failed to initialize DisplayPort tunnel validator")
        return
        
    print("✓ DisplayPort tunnel validator initialized")
    
    # Test tunnel establishment
    success = validator.test_tunnel_establishment(USB4TunnelingMode.DISPLAYPORT)
    print(f"✓ Tunnel establishment: {'SUCCESS' if success else 'FAILED'}")
    
    # Generate test video data (4K frame)
    video_data = np.random.randn(3840 * 2160).astype(np.float64)
    
    # Validate video signal integrity
    signal_results = validator.validate_video_signal_integrity(video_data)
    print(f"✓ Video signal quality: {signal_results['signal_quality'].name}")
    print(f"✓ Sync errors: {signal_results['sync_errors']}")
    print(f"✓ Pixel errors: {signal_results['pixel_errors']}")
    
    # Test MST stream validation
    mst_results = validator.validate_mst_streams(2)  # 2 displays
    print(f"✓ MST streams requested: {mst_results['requested_streams']}")
    print(f"✓ MST streams active: {mst_results['active_streams']}")
    print(f"✓ MST validation: {'PASS' if mst_results['mst_valid'] else 'FAIL'}")
    
    # Run comprehensive test
    print("\nRunning comprehensive DisplayPort test (3 seconds)...")
    comprehensive_results = validator.run_comprehensive_displayport_test(test_duration=3.0)
    
    print(f"✓ Test duration: {comprehensive_results.test_duration:.2f}s")
    print(f"✓ Throughput: {comprehensive_results.throughput:.0f} frames/s")
    print(f"✓ Video signal quality: {comprehensive_results.video_signal_quality.name}")
    print(f"✓ Frame drop rate: {comprehensive_results.frame_drop_rate:.2%}")
    print(f"✓ Resolution: {comprehensive_results.resolution[0]}x{comprehensive_results.resolution[1]}")
    print(f"✓ Color depth: {comprehensive_results.color_depth} bits")
    
    validator.cleanup()
    print("✓ DisplayPort tunnel validator cleaned up\n")


def demonstrate_usb32_tunneling():
    """Demonstrate USB 3.2 tunneling validation"""
    print("=" * 60)
    print("USB 3.2 Tunneling Validation Demo")
    print("=" * 60)
    
    # Create validator
    config = create_usb4_config()
    validator = USB32TunnelValidator(config)
    
    # Initialize validator
    if not validator.initialize():
        print("Failed to initialize USB 3.2 tunnel validator")
        return
        
    print("✓ USB 3.2 tunnel validator initialized")
    
    # Test tunnel establishment
    success = validator.test_tunnel_establishment(USB4TunnelingMode.USB32)
    print(f"✓ Tunnel establishment: {'SUCCESS' if success else 'FAILED'}")
    
    # Test device enumeration
    enum_results = validator.validate_device_enumeration(5)  # 5 devices
    print(f"✓ Devices requested: {enum_results['requested_devices']}")
    print(f"✓ Devices enumerated: {enum_results['enumerated_devices']}")
    print(f"✓ Configuration success: {enum_results['configuration_success']}")
    print(f"✓ Enumeration success rate: {enum_results['success_rate']:.2%}")
    
    # Test USB 3.2 performance
    perf_results = validator.test_usb32_performance(test_duration=2.0)
    print(f"✓ Performance throughput: {perf_results['throughput']:.0f} transfers/s")
    print(f"✓ Average latency: {perf_results['latency_avg']*1000:.2f} ms")
    print(f"✓ Error rate: {perf_results['error_rate']:.2%}")
    print(f"✓ Efficiency: {perf_results['efficiency']:.2%}")
    
    # Run comprehensive test
    print("\nRunning comprehensive USB 3.2 test (3 seconds)...")
    comprehensive_results = validator.run_comprehensive_usb32_test(test_duration=3.0)
    
    print(f"✓ Test duration: {comprehensive_results.test_duration:.2f}s")
    print(f"✓ Throughput: {comprehensive_results.throughput:.0f} transfers/s")
    print(f"✓ Enumeration success rate: {comprehensive_results.enumeration_success_rate:.2%}")
    print(f"✓ Protocol compliance rate: {comprehensive_results.protocol_compliance_rate:.2%}")
    print(f"✓ Backward compatibility: {comprehensive_results.backward_compatibility_score:.2%}")
    print(f"✓ Power delivery efficiency: {comprehensive_results.power_delivery_efficiency:.2%}")
    print(f"✓ Connected devices: {comprehensive_results.device_count}")
    
    validator.cleanup()
    print("✓ USB 3.2 tunnel validator cleaned up\n")


def demonstrate_bandwidth_management():
    """Demonstrate multi-protocol bandwidth management"""
    print("=" * 60)
    print("Multi-Protocol Bandwidth Management Demo")
    print("=" * 60)
    
    # Create bandwidth manager with 40 Gbps total bandwidth
    manager = MultiProtocolBandwidthManager(total_bandwidth=40e9)
    print("✓ Bandwidth manager initialized with 40 Gbps total bandwidth")
    
    # Test individual bandwidth allocations
    print("\nTesting individual bandwidth allocations:")
    
    # Allocate PCIe bandwidth
    pcie_result = manager.allocate_bandwidth(USB4TunnelingMode.PCIE, 10e9)
    print(f"✓ PCIe allocation: {pcie_result['allocated_bandwidth']/1e9:.1f} Gbps "
          f"({'SUCCESS' if pcie_result['allocation_success'] else 'FAILED'})")
    
    # Allocate DisplayPort bandwidth
    dp_result = manager.allocate_bandwidth(USB4TunnelingMode.DISPLAYPORT, 6e9)
    print(f"✓ DisplayPort allocation: {dp_result['allocated_bandwidth']/1e9:.1f} Gbps "
          f"({'SUCCESS' if dp_result['allocation_success'] else 'FAILED'})")
    
    # Allocate USB 3.2 bandwidth
    usb32_result = manager.allocate_bandwidth(USB4TunnelingMode.USB32, 5e9)
    print(f"✓ USB 3.2 allocation: {usb32_result['allocated_bandwidth']/1e9:.1f} Gbps "
          f"({'SUCCESS' if usb32_result['allocation_success'] else 'FAILED'})")
    
    # Check bandwidth utilization
    metrics = manager.measure_bandwidth_utilization()
    print("\nBandwidth utilization:")
    print(f"✓ Total bandwidth: {metrics['total_bandwidth']/1e9:.1f} Gbps")
    print(f"✓ Allocated bandwidth: {metrics['allocated_bandwidth']/1e9:.1f} Gbps")
    print(f"✓ Available bandwidth: {metrics['available_bandwidth']/1e9:.1f} Gbps")
    print(f"✓ Utilization: {metrics['utilization_percentage']:.1f}%")
    print(f"✓ Active tunnels: {metrics['active_tunnel_count']}")
    
    # Test congestion management
    print("\nTesting congestion management:")
    congestion_result = manager.test_congestion_management(0.85)  # High congestion
    print(f"✓ Congestion level: {congestion_result['congestion_level']:.0%}")
    print(f"✓ Throttling applied: {congestion_result['throttling_applied']}")
    print(f"✓ Bandwidth reduction: {congestion_result['bandwidth_reduction']:.0%}")
    print(f"✓ Flow control active: {congestion_result['flow_control_active']}")
    print(f"✓ Recovery time: {congestion_result['recovery_time']:.1f} ms")
    
    # Test simultaneous tunnels
    print("\nTesting simultaneous tunnel allocation:")
    tunnel_configs = [
        TunnelConfig(USB4TunnelingMode.PCIE, 8e9, 1e-6, priority=3),
        TunnelConfig(USB4TunnelingMode.DISPLAYPORT, 6e9, 100e-6, priority=2),
        TunnelConfig(USB4TunnelingMode.USB32, 5e9, 10e-6, priority=1),
    ]
    
    simultaneous_result = manager.test_simultaneous_tunnels(tunnel_configs)
    print(f"✓ Requested tunnels: {simultaneous_result['requested_tunnels']}")
    print(f"✓ Active tunnels: {simultaneous_result['active_tunnels']}")
    print(f"✓ Total bandwidth used: {simultaneous_result['total_bandwidth_used']/1e9:.1f} Gbps")
    print(f"✓ Bandwidth efficiency: {simultaneous_result['bandwidth_efficiency']:.2%}")
    print(f"✓ Allocation success: {'YES' if simultaneous_result['allocation_success'] else 'NO'}")
    
    print("✓ Bandwidth management demo completed\n")


def main():
    """Main demonstration function"""
    print("USB4 Tunneling Validation Framework Demo")
    print("========================================")
    print("This demo showcases USB4 tunneling validation capabilities")
    print("including PCIe, DisplayPort, and USB 3.2 tunneling support.\n")
    
    try:
        # Demonstrate each tunneling protocol
        demonstrate_pcie_tunneling()
        demonstrate_displayport_tunneling()
        demonstrate_usb32_tunneling()
        demonstrate_bandwidth_management()
        
        print("=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        print("\nKey features demonstrated:")
        print("• PCIe tunneling with TLP integrity checking")
        print("• DisplayPort tunneling with video signal analysis")
        print("• USB 3.2 tunneling with device enumeration")
        print("• Multi-protocol bandwidth management")
        print("• Congestion management and flow control")
        print("• Comprehensive test suites for each protocol")
        print("\nThe USB4 tunneling validation framework provides")
        print("comprehensive testing capabilities for all major")
        print("protocols tunneled over USB4 infrastructure.")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
