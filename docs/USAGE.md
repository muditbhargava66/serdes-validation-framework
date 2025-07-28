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

### 3. REST API Usage (NEW in v1.4.1)

#### Starting the API Server

```bash
# Start the API server
python -m serdes_validation_framework.api.cli server --host 0.0.0.0 --port 8000

# Or use the run script
python run_api_server.py
```

#### Using the API

```python
import requests
import numpy as np

# API base URL
BASE_URL = "http://localhost:8000/api/v1"

# Generate test signal
signal_data = np.random.randn(1000) * 0.4

# Analyze eye diagram via API
response = requests.post(f"{BASE_URL}/eye-diagram/analyze", json={
    "signal_data": signal_data.tolist(),
    "sample_rate": 40e9,
    "protocol": "USB4",
    "show_mask": True
})

if response.status_code == 200:
    result = response.json()
    print(f"Eye Height: {result['eye_height']:.4f}V")
    print(f"Q-Factor: {result['q_factor']:.2f}")
    print(f"Mask Compliance: {result['mask_analysis']['compliance_level']}")

# Start stress test
stress_response = requests.post(f"{BASE_URL}/stress-test/start", json={
    "protocol": "USB4",
    "num_cycles": 100,
    "cycle_duration": 1.0
})

test_id = stress_response.json()["test_id"]
print(f"Started stress test: {test_id}")
```

#### CLI Client Usage

```bash
# Analyze eye diagram
python -m serdes_validation_framework.api.cli analyze-eye \
    --signal-file signal_data.csv \
    --protocol USB4 \
    --sample-rate 40e9 \
    --output results.json

# Start stress test
python -m serdes_validation_framework.api.cli start-stress-test \
    --protocol PCIe \
    --cycles 500 \
    --duration 2.0

# Get system status
python -m serdes_validation_framework.api.cli status
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