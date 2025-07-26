# USB4/Thunderbolt 4 API Reference

The USB4/Thunderbolt 4 module provides comprehensive validation capabilities for USB4 2.0 and Thunderbolt 4 protocols, including dual-lane analysis, tunneling validation, and certification testing.

## Signal Analysis {#signal-analysis}

### Jitter Analysis {#jitter-analysis}

Advanced jitter decomposition for USB4 signals.

### Lane Skew Analysis {#lane-skew-analysis}

Dual-lane skew measurement and analysis.

## Compliance Testing {#compliance-testing}

USB4 compliance validation and testing.

## Performance Testing {#performance-testing}

Performance benchmarking and stress testing.

## Visualization {#visualization}

Signal visualization and plotting capabilities.

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

**Parameters:**
- `signal_data`: USB4 signal data for analysis

**Returns:**
- `SignalIntegrityResults`: Signal integrity analysis results

**Example:**
```python
integrity_results = validator.analyze_signal_integrity(signal_data)
print(f"Eye Height: {integrity_results.eye_height:.3f}")
print(f"Eye Width: {integrity_results.eye_width:.3f}")
print(f"RMS Jitter: {integrity_results.jitter_rms:.4f} UI")
```

##### `validate_tunneling(tunnel_data: TunnelingData) -> TunnelingResults`

Validates USB4 tunneling protocols (PCIe, DisplayPort, USB 3.2).

**Parameters:**
- `tunnel_data`: Tunneling protocol data

**Returns:**
- `TunnelingResults`: Tunneling validation results

### USB4SignalAnalyzer

Specialized class for USB4 signal analysis.

```python
from serdes_validation_framework.protocols.usb4 import USB4SignalAnalyzer

analyzer = USB4SignalAnalyzer()
```

#### Methods

##### `analyze_eye_diagram(signal_data: USB4SignalData) -> EyeDiagramResults`

Performs detailed eye diagram analysis with USB4-specific compliance masks.

**Parameters:**
- `signal_data`: USB4 signal data

**Returns:**
- `EyeDiagramResults`: Eye diagram analysis results

**Example:**
```python
eye_results = analyzer.analyze_eye_diagram(signal_data)
print(f"Eye Height: {eye_results.eye_height:.3f}")
print(f"Eye Width: {eye_results.eye_width:.3f}")
print(f"Crossing Percentage: {eye_results.crossing_percentage:.1f}%")
```

##### `analyze_jitter(signal_data: USB4SignalData) -> JitterResults`

Performs comprehensive jitter analysis including RJ, DJ, and PJ decomposition.

**Parameters:**
- `signal_data`: USB4 signal data

**Returns:**
- `JitterResults`: Jitter analysis results

**Example:**
```python
jitter_results = analyzer.analyze_jitter(signal_data)
print(f"RMS Jitter: {jitter_results.rms_jitter:.4f} UI")
print(f"Peak-to-Peak Jitter: {jitter_results.pp_jitter:.4f} UI")
print(f"Random Jitter: {jitter_results.random_jitter:.4f} UI")
print(f"Deterministic Jitter: {jitter_results.deterministic_jitter:.4f} UI")
```

##### `analyze_lane_skew(signal_data: USB4SignalData) -> LaneSkewResults`

Analyzes skew between USB4 dual lanes.

**Parameters:**
- `signal_data`: USB4 dual-lane signal data

**Returns:**
- `LaneSkewResults`: Lane skew analysis results

**Example:**
```python
skew_results = analyzer.analyze_lane_skew(signal_data)
print(f"Lane Skew: {skew_results.skew_ps:.2f} ps")
print(f"Lane Correlation: {skew_results.correlation:.3f}")
print(f"Skew in UI: {skew_results.skew_ui:.4f}")
```

### USB4ComplianceValidator

Specialized validator for USB4 compliance testing.

```python
from serdes_validation_framework.protocols.usb4 import USB4ComplianceValidator

compliance_validator = USB4ComplianceValidator()
```

#### Methods

##### `validate_full_compliance(signal_data: USB4SignalData, config: dict) -> List[USB4ComplianceResult]`

Runs the complete USB4 compliance test suite.

**Parameters:**
- `signal_data`: USB4 signal data
- `config`: Test configuration dictionary

**Returns:**
- `List[USB4ComplianceResult]`: List of compliance test results

**Example:**
```python
config = {
    'signal_mode': USB4SignalMode.GEN3_X2,
    'link_state': USB4LinkState.U0,
    'enable_ssc': True,
    'temperature': 25.0,
    'voltage': 3.3
}

compliance_results = compliance_validator.validate_full_compliance(
    signal_data, config
)

for result in compliance_results:
    status = "PASS" if result.status else "FAIL"
    print(f"{result.test_name}: {status}")
```

## Thunderbolt 4 Classes

### ThunderboltSecurityValidator

Validates Thunderbolt 4 security features.

```python
from serdes_validation_framework.protocols.usb4.thunderbolt import ThunderboltSecurityValidator

security_validator = ThunderboltSecurityValidator()
```

#### Methods

##### `test_dma_protection(config: dict) -> DMAProtectionResult`

Tests DMA protection capabilities.

**Parameters:**
- `config`: DMA protection configuration

**Returns:**
- `DMAProtectionResult`: DMA protection test results

**Example:**
```python
dma_config = {
    'iommu_enabled': True,
    'vt_d_support': True,
    'secure_boot': True
}

dma_result = security_validator.test_dma_protection(dma_config)
print(f"DMA Protection: {'ENABLED' if dma_result.protected else 'DISABLED'}")
```

##### `test_device_authentication(config: dict) -> AuthenticationResult`

Tests device authentication mechanisms.

**Parameters:**
- `config`: Authentication configuration

**Returns:**
- `AuthenticationResult`: Authentication test results

### DaisyChainValidator

Validates Thunderbolt 4 daisy chain functionality.

```python
from serdes_validation_framework.protocols.usb4.thunderbolt import DaisyChainValidator

daisy_validator = DaisyChainValidator()
```

#### Methods

##### `validate_chain_configuration(config: dict) -> DaisyChainResult`

Validates daisy chain configuration and bandwidth allocation.

**Parameters:**
- `config`: Daisy chain configuration

**Returns:**
- `DaisyChainResult`: Daisy chain validation results

**Example:**
```python
chain_config = {
    'device_count': 4,
    'bandwidth_requirements': [10, 8, 6, 4]  # Gbps per device
}

chain_result = daisy_validator.validate_chain_configuration(chain_config)
print(f"Chain Valid: {chain_result.valid}")
print(f"Total Bandwidth: {chain_result.total_bandwidth:.1f} Gbps")
```

### IntelCertificationSuite

Complete Intel Thunderbolt 4 certification test suite.

```python
from serdes_validation_framework.protocols.usb4.thunderbolt import IntelCertificationSuite

cert_suite = IntelCertificationSuite()
```

#### Methods

##### `run_pre_certification_checks() -> dict`

Runs pre-certification requirement checks.

**Returns:**
- `dict`: Pre-certification check results

##### `create_submission_package(session_id: str, **kwargs) -> str`

Creates certification submission package.

**Parameters:**
- `session_id`: Test session identifier
- `**kwargs`: Additional package options

**Returns:**
- `str`: Path to submission package

## Tunneling Classes

### PCIeTunnelValidator

Validates PCIe over USB4 tunneling.

```python
from serdes_validation_framework.protocols.usb4.tunneling import PCIeTunnelValidator

pcie_validator = PCIeTunnelValidator()
```

#### Methods

##### `validate_tunnel_integrity(config: dict) -> PCIeTunnelResult`

Validates PCIe tunnel integrity and performance.

**Parameters:**
- `config`: PCIe tunnel configuration

**Returns:**
- `PCIeTunnelResult`: PCIe tunnel validation results

**Example:**
```python
pcie_config = {
    'pcie_generation': 4,
    'lane_count': 4,
    'bandwidth_allocation': 0.6,
    'latency_requirement': 1e-6
}

pcie_result = pcie_validator.validate_tunnel_integrity(pcie_config)
print(f"Tunnel Valid: {pcie_result.valid}")
print(f"Bandwidth: {pcie_result.bandwidth/1e9:.1f} Gbps")
```

### DisplayPortTunnelValidator

Validates DisplayPort over USB4 tunneling.

```python
from serdes_validation_framework.protocols.usb4.tunneling import DisplayPortTunnelValidator

dp_validator = DisplayPortTunnelValidator()
```

#### Methods

##### `validate_display_configuration(config: dict) -> DisplayPortResult`

Validates DisplayPort display configuration.

**Parameters:**
- `config`: Display configuration

**Returns:**
- `DisplayPortResult`: DisplayPort validation results

**Example:**
```python
display_config = {
    'display_count': 2,
    'resolution': '4K',
    'refresh_rate': 60,
    'color_depth': 10
}

dp_result = dp_validator.validate_display_configuration(display_config)
print(f"Display Supported: {dp_result.supported}")
print(f"Required Bandwidth: {dp_result.bandwidth_required/1e9:.1f} Gbps")
```

### USB32TunnelValidator

Validates USB 3.2 over USB4 tunneling.

```python
from serdes_validation_framework.protocols.usb4.tunneling import USB32TunnelValidator

usb32_validator = USB32TunnelValidator()
```

## Performance Classes

### USB4PerformanceBenchmark

Performance benchmarking for USB4 devices.

```python
from serdes_validation_framework.protocols.usb4.performance import USB4PerformanceBenchmark

benchmark = USB4PerformanceBenchmark()
```

#### Methods

##### `test_sustained_bandwidth(duration: int, target_bandwidth: float) -> BandwidthResult`

Tests sustained bandwidth performance.

**Parameters:**
- `duration`: Test duration in seconds
- `target_bandwidth`: Target bandwidth in bps

**Returns:**
- `BandwidthResult`: Bandwidth test results

**Example:**
```python
bandwidth_result = benchmark.test_sustained_bandwidth(
    duration=300,  # 5 minutes
    target_bandwidth=32e9  # 32 Gbps
)

print(f"Average Bandwidth: {bandwidth_result.average/1e9:.1f} Gbps")
print(f"Peak Bandwidth: {bandwidth_result.peak/1e9:.1f} Gbps")
```

##### `test_latency_performance(packet_sizes: list, duration: int) -> LatencyResult`

Tests latency performance with various packet sizes.

**Parameters:**
- `packet_sizes`: List of packet sizes to test
- `duration`: Test duration in seconds

**Returns:**
- `LatencyResult`: Latency test results

### USB4StressTester

Stress testing for USB4 devices.

```python
from serdes_validation_framework.protocols.usb4.stress import USB4StressTester

stress_tester = USB4StressTester()
```

#### Methods

##### `run_thermal_stress_test(duration: int, max_temp: float) -> ThermalStressResult`

Runs thermal stress testing.

**Parameters:**
- `duration`: Test duration in minutes
- `max_temp`: Maximum allowed temperature in °C

**Returns:**
- `ThermalStressResult`: Thermal stress test results

## Data Classes

### USB4SignalData

Container for USB4 signal data.

```python
@dataclass
class USB4SignalData:
    lane0_data: np.ndarray      # Lane 0 voltage data
    lane1_data: np.ndarray      # Lane 1 voltage data
    timestamp: np.ndarray       # Time array
    sample_rate: float          # Sample rate in Hz
    signal_mode: USB4SignalMode # Signal mode
    metadata: dict = field(default_factory=dict)
```

### USB4ComplianceResult

Result of USB4 compliance testing.

```python
@dataclass
class USB4ComplianceResult:
    test_name: str              # Test name
    status: bool                # Pass/fail status
    measured_value: float       # Measured value
    limit: USB4Limit           # Test limit
    margin: float              # Margin to limit
    timestamp: datetime        # Test timestamp
    metadata: dict = field(default_factory=dict)
```

### USB4Limit

Test limit specification.

```python
@dataclass
class USB4Limit:
    minimum: float             # Minimum allowed value
    maximum: float             # Maximum allowed value
    unit: str                  # Unit of measurement
    tolerance: float = 0.0     # Measurement tolerance
```

## Constants and Enums

### USB4SignalMode

USB4 signal mode enumeration.

```python
class USB4SignalMode(Enum):
    GEN2_X2 = "Gen2x2"        # 20 Gbps
    GEN3_X2 = "Gen3x2"        # 40 Gbps
    ASYMMETRIC = "Asymmetric"  # Asymmetric mode
```

### USB4LinkState

USB4 link state enumeration.

```python
class USB4LinkState(Enum):
    U0 = "U0"                  # Active state
    U1 = "U1"                  # Low power state 1
    U2 = "U2"                  # Low power state 2
    U3 = "U3"                  # Suspend state
```

### USB4TunnelingMode

USB4 tunneling mode enumeration.

```python
class USB4TunnelingMode(Enum):
    PCIE = "PCIe"              # PCIe tunneling
    DISPLAYPORT = "DisplayPort" # DisplayPort tunneling
    USB32 = "USB3.2"           # USB 3.2 tunneling
```

## Specifications Classes

### USB4Specs

USB4 specification constants.

```python
class USB4Specs:
    # Data rates
    GEN2_RATE = 20e9           # 20 Gbps
    GEN3_RATE = 40e9           # 40 Gbps
    
    # Signal integrity limits
    MIN_EYE_HEIGHT = 0.65      # Normalized
    MIN_EYE_WIDTH = 0.65       # Normalized
    MAX_RMS_JITTER = 0.025     # UI
    MAX_LANE_SKEW = 20e-12     # 20 ps
    
    # Power specifications
    MAX_POWER_CONSUMPTION = 7.5  # W
    STANDBY_POWER = 0.5        # W
```

### ThunderboltSpecs

Thunderbolt 4 specification constants.

```python
class ThunderboltSpecs:
    # Performance requirements
    TB4_BANDWIDTH = 40e9       # 40 Gbps
    TB4_POWER_DELIVERY = 100   # W
    
    # Device limits
    MAX_DAISY_DEVICES = 6      # devices
    MAX_4K_DISPLAYS = 2        # displays
    MAX_8K_DISPLAYS = 1        # display
    
    # Security requirements
    DMA_PROTECTION_REQUIRED = True
    DEVICE_AUTH_REQUIRED = True
```

## Usage Examples

### Basic USB4 Validation

```python
# Initialize validator
validator = USB4Validator()

# Load signal data
signal_data = load_usb4_signal_data("capture.csv")

# Run validation
results = validator.validate_compliance(signal_data)

# Check results
for result in results:
    print(f"{result.test_name}: {'PASS' if result.status else 'FAIL'}")
```

### Thunderbolt 4 Certification

```python
# Initialize certification suite
cert_suite = IntelCertificationSuite()

# Configure device
device_config = {
    'vendor_id': 0x8086,
    'device_id': 0x1234,
    'firmware_version': '1.0.0'
}
cert_suite.configure_device(device_config)

# Run pre-certification checks
pre_checks = cert_suite.run_pre_certification_checks()

# Run full certification if pre-checks pass
if all(pre_checks.values()):
    cert_results = cert_suite.run_full_certification()
    print(f"Certification: {'PASS' if cert_results.passed else 'FAIL'}")
```

### Mock Mode Testing

```python
import os

# Enable mock mode
os.environ['SVF_MOCK_MODE'] = '1'

# All classes use mock implementations
validator = USB4Validator()
mock_data = validator.generate_mock_signal_data()
results = validator.validate_compliance(mock_data)

print("Mock validation completed successfully!")
```

For more examples and detailed usage, see:
- [USB4 Examples](examples/basic-validation.md)
- [Certification Guide](certification/thunderbolt4.md)
- [Best Practices](guides/best-practices.md)