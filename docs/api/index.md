# API Reference

This section provides comprehensive API documentation for the SerDes Validation Framework.

## Core APIs

### Protocol Support
- [USB4/Thunderbolt 4 API](../usb4/api-reference.md) - Complete USB4 and Thunderbolt 4 validation
- [PCIe API](pcie.md) - PCIe signal analysis and compliance testing
- [224G Ethernet API](eth_224g.md) - High-speed Ethernet PHY validation
- [PAM4 Analysis API](pam4_analysis.md) - PAM4 signal analysis tools

### Analysis & Testing
- [Signal Analysis](../usb4/api-reference.md#signal-analysis) - Eye diagrams, jitter analysis, and signal integrity
- [Compliance Testing](../usb4/api-reference.md#compliance-testing) - Automated compliance test execution
- [Performance Testing](../usb4/api-reference.md#performance-testing) - Benchmarking and performance analysis

### Reporting & Visualization
- [Reporting API](reporting.md) - Test reporting and documentation generation
- [🎨 Visualization API](visualization.md) - Comprehensive visualization system
  - Advanced eye diagram analysis with automatic measurements
  - Interactive dashboards using Plotly
  - Protocol-specific visualizations (USB4, PCIe, Ethernet)
  - Multi-protocol comparison tools

### Infrastructure
- [Instrument Control API](instrument_control.md) - Hardware instrument integration
- [Mock Controller API](mock_controller.md) - Advanced mock testing framework with 91+ tests
- [Testing Framework](../guides/testing.md) - Comprehensive testing infrastructure
- [Configuration](../reference/configuration.md) - Framework configuration management

## Quick Reference

### Common Classes

#### USB4 Validation
```python
from serdes_validation_framework.protocols.usb4 import (
    USB4Validator,
    USB4SignalAnalyzer,
    USB4ComplianceValidator
)
```

#### Thunderbolt 4 Certification
```python
from serdes_validation_framework.protocols.usb4.thunderbolt import (
    ThunderboltSecurityValidator,
    DaisyChainValidator,
    IntelCertificationSuite
)
```

#### Visualization
```python
from serdes_validation_framework.visualization import (
    USB4Visualizer,
    PCIeVisualizer,
    EthernetVisualizer,
    EyeDiagramVisualizer,
    ProtocolComparison
)
```

#### Reporting
```python
from serdes_validation_framework.reporting import (
    USB4TestReporter,
    ReportTemplate,
    TestSession
)
```

#### Visualization
```python
from serdes_validation_framework.visualization import (
    USB4Visualizer,
    PlotConfiguration,
    PlotType
)
```

### Common Enums and Constants

#### USB4 Signal Modes
```python
from serdes_validation_framework.protocols.usb4.constants import (
    USB4SignalMode,
    USB4LinkState,
    USB4TunnelingMode
)
```

#### Report Types
```python
from serdes_validation_framework.reporting import (
    ReportType,
    ReportFormat
)
```

## API Categories

### 🔌 Protocol APIs
APIs for specific protocol validation and testing.

| Protocol | Status | Key Features |
|----------|--------|--------------|
| USB4 2.0 | ✅ Complete | Dual-lane analysis, tunneling, power management |
| Thunderbolt 4 | ✅ Complete | Security, daisy-chain, certification |
| PCIe 6.0 | ✅ Complete | NRZ/PAM4, link training, compliance |
| 224G Ethernet | ✅ Complete | PAM4 analysis, equalization |

### 📊 Analysis APIs
APIs for signal analysis and measurement.

- **Eye Diagram Analysis**: Comprehensive eye pattern analysis with compliance masks
- **Jitter Analysis**: Advanced jitter decomposition (RJ, DJ, PJ)
- **Signal Integrity**: Real-time signal quality assessment
- **Lane Skew Analysis**: Multi-lane skew measurement and compensation

### 🧪 Testing APIs
APIs for automated testing and validation.

- **Compliance Testing**: Automated compliance test execution
- **Performance Testing**: Benchmarking and performance analysis
- **Multi-Protocol Integration**: Cross-protocol validation testing
- **Mock Testing**: Advanced mock implementations with intelligent protocol detection
- **Stress Testing**: Long-duration and thermal stress testing
- **Regression Testing**: Performance regression analysis with 91+ core tests

### 📈 Reporting APIs
APIs for generating reports and documentation.

- **Test Reports**: HTML, PDF, JSON, and XML report generation
- **Trend Analysis**: Performance trend monitoring and analysis
- **Certification Reports**: Thunderbolt 4 certification documentation
- **Custom Templates**: Customizable report templates

### 🎨 Visualization APIs
APIs for real-time monitoring and plotting.

- **Eye Diagrams**: Interactive eye diagram plotting
- **Signal Plots**: Dual-lane signal visualization
- **Dashboards**: Real-time monitoring dashboards
- **Trend Charts**: Performance trend visualization

### 🔧 Infrastructure APIs
APIs for framework infrastructure and utilities.

- **Instrument Control**: Hardware instrument integration
- **Mock Testing**: Comprehensive mock testing framework with 91+ core tests
- **Protocol Detection**: Intelligent protocol detection from signal characteristics
- **Test Organization**: Core, integration, and performance test categories
- **Configuration**: Framework configuration management with conditional imports
- **Utilities**: Helper functions and common utilities

## Usage Patterns

### Basic Validation Workflow
```python
# 1. Initialize validator
validator = USB4Validator()

# 2. Load or generate signal data
signal_data = validator.load_signal_data("capture.csv")

# 3. Run validation
results = validator.validate_compliance(signal_data)

# 4. Generate report
reporter = USB4TestReporter()
report_path = reporter.generate_compliance_report(session_id)
```

### Mock Mode Development
```python
import os

# Enable mock mode
os.environ['SVF_MOCK_MODE'] = '1'

# All APIs now use mock implementations
validator = USB4Validator()
mock_data = validator.generate_mock_signal_data()
results = validator.validate_compliance(mock_data)
```

### Custom Report Generation
```python
# Create custom template
template = ReportTemplate(
    name="Custom Report",
    format=ReportFormat.HTML,
    sections=['summary', 'results', 'recommendations']
)

# Generate report
reporter = USB4TestReporter()
report = reporter.generate_compliance_report(
    session_id="test_001",
    custom_template=template
)
```

## Error Handling

### Common Exceptions
```python
from serdes_validation_framework.exceptions import (
    USB4ValidationError,
    SignalAnalysisError,
    ComplianceTestError,
    InstrumentError
)

try:
    results = validator.validate_compliance(signal_data)
except USB4ValidationError as e:
    print(f"USB4 validation error: {e}")
except SignalAnalysisError as e:
    print(f"Signal analysis error: {e}")
```

### Best Practices
- Always use try-catch blocks for validation operations
- Check mock mode status when debugging
- Validate input parameters before processing
- Use context managers for resource management

## Performance Considerations

### Optimization Tips
- Use mock mode for development and CI/CD
- Cache signal data for repeated analysis
- Use appropriate sample rates for your analysis needs
- Enable parallel processing for large datasets

### Memory Management
- Process large datasets in chunks
- Clean up resources after use
- Monitor memory usage during long-running tests
- Use generators for streaming data processing

## Version Compatibility

### API Stability
- **Stable APIs**: Core validation and reporting APIs
- **Beta APIs**: Advanced visualization features
- **Experimental APIs**: New protocol support

### Deprecation Policy
- Deprecated features are marked in documentation
- Minimum 2 release cycles before removal
- Migration guides provided for breaking changes

## Support and Resources

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community support and questions
- **Examples**: Comprehensive usage examples
- **Tutorials**: Step-by-step guides

For detailed information on each API, click on the links above or use the navigation menu.