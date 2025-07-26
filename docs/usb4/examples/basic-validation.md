# Basic USB4 Validation Examples

This document provides fundamental examples for USB4 validation using the SerDes Validation Framework.

## Simple Eye Diagram Analysis

### Example 1: Basic Eye Diagram Measurement

```python
from serdes_validation_framework.protocols.usb4 import USB4SignalAnalyzer
from serdes_validation_framework.protocols.usb4.constants import USB4SignalMode

# Initialize analyzer
analyzer = USB4SignalAnalyzer()

# Configure for Gen3x2 mode
analyzer.configure_signal_mode(USB4SignalMode.GEN3_X2)

# Load signal data (replace with your data source)
signal_data = analyzer.load_signal_data("usb4_capture.csv")

# Analyze eye diagram
eye_results = analyzer.analyze_eye_diagram(signal_data)

print(f"Eye Height: {eye_results.eye_height:.3f}")
print(f"Eye Width: {eye_results.eye_width:.3f}")
print(f"Quality Factor: {eye_results.quality_factor:.2f}")

# Check against USB4 specifications
from serdes_validation_framework.protocols.usb4.constants import USB4Specs
specs = USB4Specs()

if eye_results.eye_height >= specs.MIN_EYE_HEIGHT:
    print("✓ Eye height meets specification")
else:
    print("✗ Eye height below specification")
```

### Example 2: Dual-Lane Skew Analysis

```python
# Analyze lane skew
skew_results = analyzer.analyze_lane_skew(signal_data)

print(f"Lane Skew: {skew_results.skew_ps:.2f} ps")
print(f"Lane Correlation: {skew_results.correlation:.3f}")

# Check against specification
if abs(skew_results.skew_ps) <= specs.MAX_LANE_SKEW * 1e12:
    print("✓ Lane skew within specification")
else:
    print("✗ Lane skew exceeds specification")
```

## Basic Compliance Testing

### Example 3: Simple Compliance Check

```python
from serdes_validation_framework.protocols.usb4 import USB4ComplianceValidator

# Initialize compliance validator
validator = USB4ComplianceValidator()

# Run basic compliance tests
results = validator.validate_signal_integrity(signal_data)

# Display results
passed_tests = sum(1 for r in results if r.status)
total_tests = len(results)

print(f"Compliance Results: {passed_tests}/{total_tests} tests passed")

for result in results:
    status = "PASS" if result.status else "FAIL"
    print(f"  {result.test_name}: {status}")
```

### Example 4: Custom Test Configuration

```python
# Create custom test configuration
test_config = {
    'signal_mode': USB4SignalMode.GEN2_X2,
    'link_state': USB4LinkState.U0,
    'enable_ssc': True,
    'ssc_frequency': 33000,  # Hz
    'ssc_deviation': 0.005,  # 0.5%
    'test_duration': 60,     # seconds
    'temperature': 25.0,     # °C
    'voltage_tolerance': 0.05  # 5%
}

# Run targeted compliance tests
targeted_results = validator.validate_signal_integrity(
    signal_data=signal_data,
    config=test_config
)

print("Signal Integrity Test Results:")
for test_name, result in targeted_results.items():
    print(f"  {test_name}: {result.status} ({result.measured_value:.3f} {result.unit})")
```

## Mock Mode Examples

### Example 5: Development with Mock Data

```python
import os

# Enable mock mode for development
os.environ['SVF_MOCK_MODE'] = '1'

# All classes now use mock implementations
from serdes_validation_framework.protocols.usb4 import USB4Validator

validator = USB4Validator()

# Generate mock signal data
mock_signal = validator.generate_mock_signal_data(
    mode=USB4SignalMode.GEN3_X2,
    duration=1e-6,  # 1 microsecond
    sample_rate=100e9  # 100 GSa/s
)

# Run validation with mock data
mock_results = validator.validate_compliance(mock_signal)

print("Mock validation completed successfully!")
print(f"Generated {len(mock_results)} test results")

# Display mock results
for result in mock_results[:5]:  # Show first 5 results
    status = "PASS" if result.status else "FAIL"
    print(f"  {result.test_name}: {status}")
```

### Example 6: CI/CD Testing

```python
def test_usb4_compliance():
    """Test function for CI/CD pipeline"""
    
    # Ensure mock mode is enabled
    os.environ['SVF_MOCK_MODE'] = '1'
    
    validator = USB4Validator()
    
    # Test all signal modes
    modes = [USB4SignalMode.GEN2_X2, USB4SignalMode.GEN3_X2]
    
    for mode in modes:
        signal_data = validator.generate_mock_signal_data(mode)
        results = validator.validate_compliance(signal_data)
        
        # Assert all tests pass in mock mode
        assert all(r.status for r in results), f"Mock tests failed for {mode}"
        
    print("All CI/CD tests passed!")

# Run the test
if __name__ == "__main__":
    test_usb4_compliance()
```

## Basic Reporting

### Example 7: Simple Report Generation

```python
from serdes_validation_framework.reporting import USB4TestReporter
from datetime import datetime

# Initialize reporter
reporter = USB4TestReporter()

# Create test session
session = TestSession(
    session_id="basic_test_001",
    timestamp=datetime.now(),
    test_type=ReportType.COMPLIANCE,
    signal_mode=USB4SignalMode.GEN3_X2,
    device_info={
        "vendor": "Example Corp",
        "model": "USB4-DEV-001"
    }
)

reporter.add_test_session(session)
reporter.add_compliance_results(session.session_id, results)

# Generate HTML report
report_path = reporter.generate_compliance_report(session.session_id)
print(f"Report generated: {report_path}")
```

### Example 8: JSON Report for API Integration

```python
# Generate JSON report for API integration
json_template = ReportTemplate(
    name="USB4 API Report",
    format=ReportFormat.JSON,
    sections=['summary', 'results'],
    include_raw_data=True
)

json_report = reporter.generate_compliance_report(
    session_id=session.session_id,
    custom_template=json_template
)

# Load and process JSON data
import json
with open(json_report, 'r') as f:
    report_data = json.load(f)

print(f"JSON report contains {len(report_data['test_results'])} test results")
```

## Basic Visualization

### Example 9: Eye Diagram Plotting

```python
from serdes_validation_framework.visualization import USB4Visualizer

# Initialize visualizer
visualizer = USB4Visualizer()

# Plot eye diagram
eye_plot, eye_data = visualizer.plot_eye_diagram(
    signal_data=signal_data,
    config=PlotConfiguration(
        title="USB4 Gen3x2 Eye Diagram",
        width=1200,
        height=800,
        save_format=VisualizationFormat.PNG
    )
)

print(f"Eye diagram saved to: {eye_plot}")
print(f"Eye height: {eye_data.eye_height:.3f}")
print(f"Eye width: {eye_data.eye_width:.3f}")
```

### Example 10: Dual-Lane Signal Plot

```python
# Plot dual-lane signals with skew analysis
lane_plot, skew_data = visualizer.plot_dual_lane_signals(
    signal_data=signal_data,
    config=PlotConfiguration(
        title="USB4 Dual-Lane Analysis",
        width=1400,
        height=600
    )
)

print(f"Lane analysis plot: {lane_plot}")
print(f"Measured skew: {skew_data.skew_ps:.2f} ps")
print(f"Lane correlation: {skew_data.correlation:.3f}")
```

## Error Handling Examples

### Example 11: Robust Error Handling

```python
from serdes_validation_framework.exceptions import (
    USB4ValidationError,
    SignalAnalysisError,
    ComplianceTestError
)

def robust_usb4_validation(signal_data):
    """Example of robust USB4 validation with error handling"""
    
    try:
        validator = USB4Validator()
        results = validator.validate_compliance(signal_data)
        
        return {
            'success': True,
            'results': results,
            'error': None
        }
        
    except USB4ValidationError as e:
        print(f"USB4 validation error: {e}")
        return {
            'success': False,
            'results': None,
            'error': str(e)
        }
        
    except SignalAnalysisError as e:
        print(f"Signal analysis error: {e}")
        return {
            'success': False,
            'results': None,
            'error': str(e)
        }
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {
            'success': False,
            'results': None,
            'error': str(e)
        }

# Usage
result = robust_usb4_validation(signal_data)
if result['success']:
    print("Validation completed successfully")
    # Process results
    for test_result in result['results']:
        print(f"  {test_result.test_name}: {'PASS' if test_result.status else 'FAIL'}")
else:
    print(f"Validation failed: {result['error']}")
```

## Configuration Examples

### Example 12: Environment Configuration

```python
# Configure environment for USB4 testing
import os

# Set USB4-specific environment variables
os.environ['SVF_USB4_DEFAULT_MODE'] = 'GEN3_X2'
os.environ['SVF_USB4_SSC_ENABLED'] = 'true'
os.environ['SVF_REPORT_FORMAT'] = 'HTML'

# Load configuration
from serdes_validation_framework.config import ConfigManager
config = ConfigManager()

# Access USB4 configuration
usb4_mode = config.get('usb4.default_mode', 'GEN3_X2')
ssc_enabled = config.get('usb4.ssc_enabled', True)

print(f"USB4 Mode: {usb4_mode}")
print(f"SSC Enabled: {ssc_enabled}")
```

### Example 13: Custom Test Limits

```python
# Define custom test limits
custom_limits = {
    'eye_height_min': 0.7,      # 70% minimum
    'eye_width_min': 0.7,       # 70% minimum
    'jitter_rms_max': 0.02,     # 2% maximum
    'lane_skew_max': 15e-12,    # 15 ps maximum
}

# Apply custom limits to validator
validator = USB4ComplianceValidator(custom_limits=custom_limits)

# Run validation with custom limits
results = validator.validate_compliance(signal_data)

print("Validation with custom limits:")
for result in results:
    status = "PASS" if result.status else "FAIL"
    margin = result.margin if hasattr(result, 'margin') else 'N/A'
    print(f"  {result.test_name}: {status} (margin: {margin})")
```

## Integration Examples

### Example 14: Simple Hardware Integration

```python
# Example with simulated hardware connection
class SimpleUSB4Tester:
    def __init__(self):
        self.validator = USB4Validator()
        self.analyzer = USB4SignalAnalyzer()
    
    def run_basic_test(self, device_id):
        """Run basic USB4 test on a device"""
        
        print(f"Testing device: {device_id}")
        
        # In real implementation, this would capture from hardware
        # For this example, we'll use mock data
        os.environ['SVF_MOCK_MODE'] = '1'
        signal_data = self.analyzer.generate_mock_signal_data()
        
        # Run compliance tests
        results = self.validator.validate_compliance(signal_data)
        
        # Analyze results
        passed = sum(1 for r in results if r.status)
        total = len(results)
        
        print(f"Test Results: {passed}/{total} tests passed")
        
        return {
            'device_id': device_id,
            'passed': passed,
            'total': total,
            'pass_rate': passed / total if total > 0 else 0,
            'results': results
        }

# Usage
tester = SimpleUSB4Tester()
test_result = tester.run_basic_test("USB4-DEV-001")

if test_result['pass_rate'] >= 0.95:  # 95% pass rate required
    print("✓ Device passed basic validation")
else:
    print("✗ Device failed basic validation")
```

## Next Steps

After mastering these basic examples:

1. **Advanced Testing**: Move to [advanced testing examples](advanced-testing.md)
2. **Certification**: Follow the [Thunderbolt 4 certification guide](../certification/thunderbolt4.md)
3. **Mock Testing**: Explore [mock testing examples](mock-testing.md)
4. **Reporting**: Learn [advanced reporting techniques](reporting.md)
5. **API Reference**: Study the complete [USB4 API documentation](../api-reference.md)

## Common Issues and Solutions

### Issue 1: Import Errors
```python
# Solution: Check mock mode and installation
import os
print(f"Mock mode: {os.environ.get('SVF_MOCK_MODE', '0')}")

try:
    from serdes_validation_framework.protocols.usb4 import USB4Validator
    print("✓ USB4 module imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Try: pip install serdes-validation-framework[all]")
```

### Issue 2: No Signal Data
```python
# Solution: Use mock mode for development
if not os.path.exists("usb4_capture.csv"):
    print("No signal data file found, enabling mock mode")
    os.environ['SVF_MOCK_MODE'] = '1'
    
    validator = USB4Validator()
    signal_data = validator.generate_mock_signal_data()
else:
    analyzer = USB4SignalAnalyzer()
    signal_data = analyzer.load_signal_data("usb4_capture.csv")
```

These basic examples provide a solid foundation for USB4 validation. For more complex scenarios, see the [advanced testing examples](advanced-testing.md).