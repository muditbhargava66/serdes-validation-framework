# USB4 Quick Start Guide

This guide will walk you through getting started with USB4/Thunderbolt 4 validation using the SerDes Validation Framework.

## Prerequisites

- SerDes Validation Framework installed
- Python 3.9 or higher
- Optional: USB4-capable hardware or mock mode for testing

## Basic USB4 Validation

### 1. Setting Up USB4 Validation

```python
from serdes_validation_framework.protocols.usb4 import (
    USB4Validator,
    USB4SignalAnalyzer,
    USB4ComplianceValidator
)
from serdes_validation_framework.protocols.usb4.constants import (
    USB4SignalMode,
    USB4LinkState
)

# Initialize USB4 validator
validator = USB4Validator()

# Configure for Gen3x2 mode
signal_mode = USB4SignalMode.GEN3_X2
validator.configure_signal_mode(signal_mode)
```

### 2. Signal Analysis {#signal-analysis}

```python
# Analyze USB4 dual-lane signals
analyzer = USB4SignalAnalyzer()

# Load signal data (or use mock data)
signal_data = analyzer.load_signal_data("usb4_capture.csv")

# Perform dual-lane analysis
lane_results = analyzer.analyze_dual_lanes(signal_data)
print(f"Lane skew: {lane_results.skew_ps:.2f} ps")
print(f"Lane correlation: {lane_results.correlation:.3f}")

# Eye diagram analysis
eye_results = analyzer.analyze_eye_diagram(signal_data)
print(f"Eye height: {eye_results.eye_height:.3f}")
print(f"Eye width: {eye_results.eye_width:.3f}")
```

### 3. Compliance Testing {#compliance-testing}

```python
# Run USB4 compliance tests
compliance_validator = USB4ComplianceValidator()

# Execute full compliance suite
compliance_results = compliance_validator.validate_full_compliance(
    signal_data=signal_data,
    test_config={
        'signal_mode': USB4SignalMode.GEN3_X2,
        'link_state': USB4LinkState.U0,
        'enable_ssc': True
    }
)

# Check results
for result in compliance_results:
    status = "PASS" if result.status else "FAIL"
    print(f"{result.test_name}: {status} ({result.measured_value:.3f} {result.unit})")
```

## Thunderbolt 4 Certification

### 1. Security Validation

```python
from serdes_validation_framework.protocols.usb4.thunderbolt import (
    ThunderboltSecurityValidator,
    DaisyChainValidator
)

# Test Thunderbolt 4 security features
security_validator = ThunderboltSecurityValidator()

security_results = security_validator.validate_security_features({
    'dma_protection': True,
    'device_authentication': True,
    'secure_boot': True
})

print(f"Security validation: {'PASS' if all(security_results.values()) else 'FAIL'}")
```

### 2. Daisy Chain Testing

```python
# Test daisy chain configuration
daisy_validator = DaisyChainValidator()

# Test up to 6 devices in chain
chain_results = daisy_validator.validate_chain_topology(
    device_count=4,
    bandwidth_requirements=[10, 5, 2, 1]  # Gbps per device
)

print(f"Daisy chain validation: {'PASS' if chain_results.valid else 'FAIL'}")
print(f"Total bandwidth utilization: {chain_results.bandwidth_utilization:.1%}")
```

## USB4 Tunneling Validation

### 1. PCIe Tunneling

```python
from serdes_validation_framework.protocols.usb4.tunneling import (
    PCIeTunnelValidator,
    DisplayPortTunnelValidator,
    USB32TunnelValidator
)

# Validate PCIe over USB4 tunneling
pcie_tunnel = PCIeTunnelValidator()

pcie_results = pcie_tunnel.validate_tunnel_integrity({
    'pcie_gen': 4,
    'lane_count': 4,
    'bandwidth_allocation': 0.6  # 60% of USB4 bandwidth
})

print(f"PCIe tunnel validation: {'PASS' if pcie_results.valid else 'FAIL'}")
```

### 2. DisplayPort Tunneling

```python
# Validate DisplayPort over USB4
dp_tunnel = DisplayPortTunnelValidator()

dp_results = dp_tunnel.validate_video_tunnel({
    'resolution': '4K',
    'refresh_rate': 60,
    'color_depth': 10,
    'display_count': 2
})

print(f"DisplayPort tunnel validation: {'PASS' if dp_results.valid else 'FAIL'}")
```

## Mock Mode Testing

For development and CI/CD environments, you can use mock mode:

```python
import os

# Enable mock mode
os.environ['SVF_MOCK_MODE'] = '1'

# All USB4 classes will now use mock implementations
validator = USB4Validator()  # Uses mock data
results = validator.validate_compliance()  # Returns mock results

print("Mock mode testing completed successfully!")
```

## Generating Reports

### 1. Compliance Report

```python
from serdes_validation_framework.reporting import USB4TestReporter

# Initialize reporter
reporter = USB4TestReporter()

# Add test session
from datetime import datetime
session = TestSession(
    session_id="usb4_test_001",
    timestamp=datetime.now(),
    test_type=ReportType.COMPLIANCE,
    signal_mode=USB4SignalMode.GEN3_X2,
    device_info={"vendor": "Test Corp", "model": "USB4-DEV-001"}
)

reporter.add_test_session(session)
reporter.add_compliance_results("usb4_test_001", compliance_results)

# Generate HTML report
report_path = reporter.generate_compliance_report("usb4_test_001")
print(f"Report generated: {report_path}")
```

### 2. Certification Report

```python
# Generate Thunderbolt 4 certification report
cert_report = reporter.generate_certification_report(
    session_id="usb4_test_001",
    template_name="certification_pdf"
)
print(f"Certification report: {cert_report}")
```

## Visualization

### 1. Eye Diagram Plotting

```python
from serdes_validation_framework.visualization import USB4Visualizer

visualizer = USB4Visualizer()

# Plot eye diagram
eye_plot, eye_data = visualizer.plot_eye_diagram(
    signal_data=signal_data,
    config=PlotConfiguration(
        plot_type=PlotType.EYE_DIAGRAM,
        title="USB4 Gen3x2 Eye Diagram"
    )
)

print(f"Eye diagram saved: {eye_plot}")
```

### 2. Dual-Lane Analysis

```python
# Plot dual-lane signals with skew analysis
lane_plot, skew_data = visualizer.plot_dual_lane_signals(
    signal_data=signal_data,
    config=PlotConfiguration(
        plot_type=PlotType.DUAL_LANE_SIGNAL,
        title="USB4 Dual-Lane Analysis"
    )
)

print(f"Lane analysis plot: {lane_plot}")
print(f"Measured skew: {skew_data.skew_ps:.2f} ps")
```

## Best Practices

### 1. Test Configuration

```python
# Always configure test parameters explicitly
test_config = {
    'signal_mode': USB4SignalMode.GEN3_X2,
    'link_state': USB4LinkState.U0,
    'enable_ssc': True,
    'temperature': 25.0,  # °C
    'voltage': 3.3,       # V
    'test_duration': 60   # seconds
}
```

### 2. Error Handling

```python
try:
    results = validator.validate_compliance(signal_data, test_config)
except USB4ValidationError as e:
    print(f"Validation error: {e}")
    # Handle specific USB4 errors
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle general errors
```

### 3. Resource Management

```python
# Use context managers for proper cleanup
with USB4Validator() as validator:
    results = validator.validate_compliance(signal_data)
    # Validator automatically cleaned up
```

## Next Steps

1. **Advanced Features**: Explore the [USB4 API documentation](api-reference.md)
2. **Certification**: Follow the [Thunderbolt 4 certification guide](certification/thunderbolt4.md)
3. **Custom Tests**: Learn about creating custom validation tests
4. **Integration**: Set up CI/CD integration
5. **Troubleshooting**: Check the troubleshooting guide

## Example Scripts

Complete example scripts are available in the examples directory:

- Basic Validation - Basic USB4 compliance testing
- Advanced Testing - Complex validation scenarios
- Mock Testing - Mock mode development example
- Reporting - Report generation examples

## Support

For questions or issues with USB4 validation:

1. Check the troubleshooting guide
2. Review the [API documentation](api-reference.md)
3. Explore the examples
4. Open an issue on GitHub with your specific use case

Happy validating! 🚀