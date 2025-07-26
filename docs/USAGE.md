# Basic Usage

This document provides a quick overview of basic framework usage. For comprehensive guides, see the tutorials directory.

## Quick Start

### 1. Basic USB4 Validation

```python
from serdes_validation_framework.protocols.usb4 import USB4Validator

# Initialize validator
validator = USB4Validator()

# Load signal data (or use mock data)
signal_data = validator.load_signal_data("usb4_capture.csv")

# Run compliance validation
results = validator.validate_compliance(signal_data)

# Check results
for result in results:
    status = "PASS" if result.status else "FAIL"
    print(f"{result.test_name}: {status}")
```

### 2. Generate Reports

```python
from serdes_validation_framework.reporting import USB4TestReporter

# Initialize reporter
reporter = USB4TestReporter()

# Add test session and results
session = TestSession(
    session_id="test_001",
    timestamp=datetime.now(),
    test_type=ReportType.COMPLIANCE,
    signal_mode=USB4SignalMode.GEN3_X2
)

reporter.add_test_session(session)
reporter.add_compliance_results(session.session_id, results)

# Generate HTML report
report_path = reporter.generate_compliance_report(session.session_id)
print(f"Report generated: {report_path}")
```

### 3. Mock Mode for Development

```python
import os

# Enable mock mode
os.environ['SVF_MOCK_MODE'] = '1'

# All operations now use mock data
validator = USB4Validator()
mock_data = validator.generate_mock_signal_data()
results = validator.validate_compliance(mock_data)

print("Mock validation completed successfully!")
```

## Common Use Cases

### Signal Analysis
- [Eye Diagram Analysis](usb4/quickstart.md#signal-analysis)
- [Jitter Analysis](usb4/api-reference.md#jitter-analysis)
- [Lane Skew Measurement](usb4/api-reference.md#lane-skew-analysis)

### Compliance Testing
- [USB4 Compliance](usb4/quickstart.md#compliance-testing)
- [Thunderbolt Certification](usb4/certification/thunderbolt4.md)
- [Custom Test Sequences](guides/testing.md)

### Reporting and Visualization
- [Custom Reports](examples/reporting_examples.md)
- Real-time Monitoring
- [Trend Analysis](api/reporting.md#trend-analysis)

## Next Steps

1. **Detailed Tutorials**: Start with [Getting Started](tutorials/getting_started.md)
2. **USB4 Specific**: Follow [USB4 Quick Start](usb4/quickstart.md)
3. **API Reference**: Explore the [API Documentation](api/index.md)
4. **Examples**: Check out comprehensive examples

For complete usage information, see the tutorials and API documentation.