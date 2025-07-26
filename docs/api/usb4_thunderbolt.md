# USB4/Thunderbolt 4 API Reference

The USB4/Thunderbolt 4 module provides comprehensive validation capabilities for USB4 2.0 and Thunderbolt 4 protocols, including dual-lane analysis, tunneling validation, and certification testing.

## Core Classes

### USB4Validator

The main class for USB4 compliance validation.

```python
from serdes_validation_framework.protocols.usb4 import USB4Validator

validator = USB4Validator()
results = validator.validate_compliance(signal_data)
```

#### Methods

##### `validate_compliance(signal_data: USB4SignalData) -> USB4ComplianceResult`

Performs comprehensive USB4 compliance validation.

**Parameters:**
- `signal_data`: USB4 signal data containing dual-lane measurements

**Returns:**
- `USB4ComplianceResult`: Detailed compliance test results

**Example:**
```python
# Load signal data
signal_data = USB4SignalData(
    lane0_data=lane0_voltage,
    lane1_data=lane1_voltage,
    timestamp=time_array,
    sample_rate=80e9,
    signal_mode=USB4SignalMode.GEN2X2
)

# Run compliance validation
results = validator.validate_compliance(signal_data)
print(f"Overall Status: {results.overall_status}")
```

##### `analyze_signal_integrity(signal_data: USB4SignalData) -> SignalIntegrityResults`

Analyzes signal integrity parameters including eye diagrams and jitter.

##### `validate_tunneling(tunnel_data: TunnelingData) -> TunnelingResults`

Validates tunneling protocols (PCIe, DisplayPort, USB 3.2).

### USB4SignalAnalyzer

Advanced signal analysis for USB4 dual-lane signals.

```python
from serdes_validation_framework.protocols.usb4 import USB4SignalAnalyzer

analyzer = USB4SignalAnalyzer()
eye_results = analyzer.analyze_eye_diagram(signal_data)
skew_results = analyzer.measure_lane_skew(signal_data)
```

#### Methods

##### `analyze_eye_diagram(signal_data: USB4SignalData) -> EyeDiagramResults`

Performs comprehensive eye diagram analysis with USB4-specific compliance masks.

##### `measure_lane_skew(signal_data: USB4SignalData) -> LaneSkewResults`

Measures and analyzes dual-lane skew with compensation algorithms.

##### `analyze_jitter(signal_data: USB4SignalData) -> JitterResults`

Performs advanced jitter decomposition (RJ, DJ, PJ) with SSC awareness.

### USB4LinkTraining

USB4 link training state machine and validation.

```python
from serdes_validation_framework.protocols.usb4 import USB4LinkTraining

link_trainer = USB4LinkTraining()
training_results = link_trainer.run_training_sequence(config)
```

### USB4PowerManager

USB4 power state management and validation.

```python
from serdes_validation_framework.protocols.usb4 import USB4PowerManager

power_mgr = USB4PowerManager()
power_results = power_mgr.validate_power_states(signal_data)
```

## Thunderbolt 4 Support

### ThunderboltSecurityValidator

Validates Thunderbolt 4 security features including DMA protection.

```python
from serdes_validation_framework.protocols.usb4.thunderbolt import ThunderboltSecurityValidator

security_validator = ThunderboltSecurityValidator()
security_results = security_validator.validate_security_features(device_data)
```

### DaisyChainValidator

Validates Thunderbolt 4 daisy-chain configurations up to 6 devices.

```python
from serdes_validation_framework.protocols.usb4.thunderbolt import DaisyChainValidator

chain_validator = DaisyChainValidator()
chain_results = chain_validator.validate_chain(chain_config)
```

### IntelCertificationSuite

Complete Intel Thunderbolt 4 certification test suite.

```python
from serdes_validation_framework.protocols.usb4.thunderbolt import IntelCertificationSuite

cert_suite = IntelCertificationSuite()
cert_results = cert_suite.run_certification_tests(device_config)
```

## Tunneling Validation

### PCIeTunnelValidator

Validates PCIe tunneling over USB4.

```python
from serdes_validation_framework.protocols.usb4 import PCIeTunnelValidator

pcie_validator = PCIeTunnelValidator()
pcie_results = pcie_validator.validate_pcie_tunnel(tunnel_data)
```

### DisplayPortTunnelValidator

Validates DisplayPort tunneling with video signal integrity analysis.

```python
from serdes_validation_framework.protocols.usb4 import DisplayPortTunnelValidator

dp_validator = DisplayPortTunnelValidator()
dp_results = dp_validator.validate_displayport_tunnel(video_data)
```

### USB32TunnelValidator

Validates USB 3.2 backward compatibility over USB4 tunneling.

```python
from serdes_validation_framework.protocols.usb4 import USB32TunnelValidator

usb32_validator = USB32TunnelValidator()
usb32_results = usb32_validator.validate_usb32_tunnel(usb_data)
```

## Data Structures

### USB4SignalData

Container for USB4 signal measurements.

```python
@dataclass
class USB4SignalData:
    lane0_data: np.ndarray      # Lane 0 voltage measurements
    lane1_data: np.ndarray      # Lane 1 voltage measurements
    timestamp: np.ndarray       # Time base
    sample_rate: float          # Sampling rate (Hz)
    signal_mode: USB4SignalMode # Signal mode (Gen2x2, Gen3x2, etc.)
    metadata: Dict[str, Any]    # Additional metadata
```

### USB4ComplianceResult

Results from USB4 compliance validation.

```python
@dataclass
class USB4ComplianceResult:
    overall_status: TestResult
    signal_integrity: SignalIntegrityResults
    protocol_tests: ProtocolTestResults
    tunneling_tests: TunnelingTestResults
    power_tests: PowerTestResults
    recommendations: List[str]
    test_duration: float
```

### USB4SignalMode

Enumeration of USB4 signal modes.

```python
class USB4SignalMode(Enum):
    GEN2 = "Gen2"           # 10 Gbps single lane
    GEN3 = "Gen3"           # 20 Gbps single lane
    GEN2X2 = "Gen2x2"       # 20 Gbps dual lane
    GEN3X2 = "Gen3x2"       # 40 Gbps dual lane
    ASYMMETRIC = "Asymmetric" # Asymmetric configuration
```

### USB4LinkState

USB4 power states.

```python
class USB4LinkState(Enum):
    U0 = "U0"  # Active
    U1 = "U1"  # Standby
    U2 = "U2"  # Sleep
    U3 = "U3"  # Suspend
```

### USB4TunnelingMode

Supported tunneling protocols.

```python
class USB4TunnelingMode(Enum):
    PCIE = "PCIe"
    DISPLAYPORT = "DisplayPort"
    USB32 = "USB32"
    THUNDERBOLT = "Thunderbolt"
```

## Configuration Classes

### USB4Specs

USB4 specification constants and limits.

```python
from serdes_validation_framework.protocols.usb4 import USB4Specs

specs = USB4Specs()
print(f"Max bandwidth: {specs.GEN3X2_BANDWIDTH / 1e9} Gbps")
print(f"Max lane skew: {specs.MAX_LANE_SKEW * 1e12} ps")
```

### USB4SignalSpecs

Signal integrity specifications for USB4.

```python
from serdes_validation_framework.protocols.usb4 import USB4SignalSpecs

signal_specs = USB4SignalSpecs()
print(f"Min eye height: {signal_specs.MIN_EYE_HEIGHT}")
print(f"Max jitter: {signal_specs.MAX_RJ_RMS * 1e12} ps")
```

### ThunderboltSpecs

Thunderbolt 4 specific specifications.

```python
from serdes_validation_framework.protocols.usb4.thunderbolt import ThunderboltSpecs

tb_specs = ThunderboltSpecs()
print(f"TB4 bandwidth: {tb_specs.TB4_BANDWIDTH / 1e9} Gbps")
print(f"Max daisy devices: {tb_specs.MAX_DAISY_DEVICES}")
```

## Test Automation

### USB4TestSequence

Automated USB4 test sequence orchestration.

```python
from serdes_validation_framework.test_sequence import USB4TestSequence

# Configure test sequence
config = USB4TestSequenceConfig(
    test_name="USB4 Compliance Test",
    signal_mode=USB4SignalMode.GEN3X2,
    enable_tunneling_tests=True,
    enable_power_tests=True
)

# Run test sequence
sequence = USB4TestSequence(config)
results = sequence.run_complete_sequence(signal_data)
```

### USB4PerformanceBenchmark

Performance benchmarking and regression testing.

```python
from serdes_validation_framework.protocols.usb4 import USB4PerformanceBenchmark

benchmark = USB4PerformanceBenchmark()
perf_results = benchmark.run_performance_tests(test_config)
```

### USB4StressTester

Long-duration stability and stress testing.

```python
from serdes_validation_framework.protocols.usb4 import USB4StressTester

stress_tester = USB4StressTester()
stress_results = stress_tester.run_stress_tests(stress_config)
```

## Instrument Integration

### USB4ScopeController

High-speed oscilloscope control for USB4 measurements.

```python
from serdes_validation_framework.instrument_control import USB4ScopeController

scope = USB4ScopeController()
scope.configure_for_usb4(signal_mode=USB4SignalMode.GEN3X2)
signal_data = scope.capture_dual_lane_signal()
```

### USB4PatternGenerator

Test pattern generation for USB4 validation.

```python
from serdes_validation_framework.instrument_control import USB4PatternGenerator

pattern_gen = USB4PatternGenerator()
pattern_gen.generate_compliance_patterns(USB4SignalMode.GEN3X2)
```

### USB4PowerMeter

Power measurement and validation.

```python
from serdes_validation_framework.instrument_control import USB4PowerMeter

power_meter = USB4PowerMeter()
power_data = power_meter.measure_power_consumption(test_duration=60)
```

## Mock Testing Support

For testing without hardware, the framework provides comprehensive mock implementations:

```python
import os
os.environ['SVF_MOCK_MODE'] = '1'

from serdes_validation_framework.protocols.usb4 import USB4Validator

# This will use mock implementations
validator = USB4Validator()
mock_results = validator.validate_compliance(mock_signal_data)
```

## Examples

### Basic USB4 Validation

```python
from serdes_validation_framework.protocols.usb4 import USB4Validator, USB4SignalData, USB4SignalMode
import numpy as np

# Create sample signal data
duration = 5e-6  # 5 microseconds
sample_rate = 80e9  # 80 GSa/s
num_samples = int(duration * sample_rate)

time = np.linspace(0, duration, num_samples)
lane0_data = np.random.choice([-0.4, 0.4], size=num_samples)
lane1_data = np.random.choice([-0.4, 0.4], size=num_samples)

signal_data = USB4SignalData(
    lane0_data=lane0_data,
    lane1_data=lane1_data,
    timestamp=time,
    sample_rate=sample_rate,
    signal_mode=USB4SignalMode.GEN2X2
)

# Run validation
validator = USB4Validator()
results = validator.validate_compliance(signal_data)

print(f"Overall Status: {results.overall_status}")
print(f"Test Duration: {results.test_duration:.2f} seconds")
```

### Thunderbolt 4 Certification

```python
from serdes_validation_framework.protocols.usb4.thunderbolt import IntelCertificationSuite

# Configure device for certification
device_config = {
    'device_id': 'TB4_Device_001',
    'vendor': 'Example Corp',
    'model': 'TB4 Hub Pro',
    'firmware_version': '1.2.3'
}

# Run certification tests
cert_suite = IntelCertificationSuite()
cert_results = cert_suite.run_certification_tests(device_config)

if cert_results.certification_status == 'CERTIFIED':
    print("Device passed Thunderbolt 4 certification!")
else:
    print("Certification failed. Check detailed results.")
```

### Multi-Protocol Tunneling

```python
from serdes_validation_framework.protocols.usb4 import (
    PCIeTunnelValidator, 
    DisplayPortTunnelValidator,
    USB32TunnelValidator
)

# Validate multiple tunneling protocols
pcie_validator = PCIeTunnelValidator()
dp_validator = DisplayPortTunnelValidator()
usb32_validator = USB32TunnelValidator()

# Run tunneling tests
pcie_results = pcie_validator.validate_pcie_tunnel(pcie_tunnel_data)
dp_results = dp_validator.validate_displayport_tunnel(dp_tunnel_data)
usb32_results = usb32_validator.validate_usb32_tunnel(usb32_tunnel_data)

print(f"PCIe Tunneling: {pcie_results.status}")
print(f"DisplayPort Tunneling: {dp_results.status}")
print(f"USB 3.2 Tunneling: {usb32_results.status}")
```

## Best Practices

1. **Signal Quality**: Always validate signal integrity before protocol testing
2. **Dual-Lane Analysis**: Use both lanes for comprehensive USB4 validation
3. **Mock Testing**: Leverage mock mode for CI/CD and development
4. **Reporting**: Generate detailed reports for compliance documentation
5. **Performance Monitoring**: Use trend analysis for regression detection

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure mock mode is enabled for testing without hardware
2. **Signal Quality**: Check sample rate and signal conditioning
3. **Lane Skew**: Verify differential pair routing and termination
4. **Tunneling Issues**: Validate bandwidth allocation and flow control

### Debug Mode

Enable debug logging for detailed analysis:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your USB4 validation code here
```

For more troubleshooting tips, see the [Troubleshooting Guide](../guides/troubleshooting.md).