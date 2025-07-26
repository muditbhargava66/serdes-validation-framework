# Reporting API Reference

The SerDes Validation Framework provides comprehensive reporting capabilities for generating certification-ready reports, trend analysis, and performance monitoring. The reporting system supports multiple output formats and customizable templates.

## Core Classes

### USB4TestReporter

The main class for generating USB4 and Thunderbolt test reports.

```python
from serdes_validation_framework.protocols.usb4.reporting import USB4TestReporter

reporter = USB4TestReporter(output_directory="reports")
```

#### Constructor

```python
USB4TestReporter(output_directory: str = "usb4_reports")
```

**Parameters:**
- `output_directory`: Directory for report output files

#### Methods

##### `add_test_session(session: TestSession) -> None`

Add test session information to the reporter.

```python
from datetime import datetime
from serdes_validation_framework.protocols.usb4.reporting import TestSession, ReportType
from serdes_validation_framework.protocols.usb4 import USB4SignalMode

session = TestSession(
    session_id="USB4_TEST_001",
    timestamp=datetime.now(),
    test_type=ReportType.COMPLIANCE,
    signal_mode=USB4SignalMode.GEN3X2,
    device_info={
        "vendor": "Example Corp",
        "model": "USB4 Hub Pro",
        "serial": "12345"
    },
    operator="Test Engineer"
)

reporter.add_test_session(session)
```

##### `add_test_results(session_id: str, results: List[USB4TestResult]) -> None`

Add test results for a specific session.

```python
test_results = [
    USB4TestResult(
        test_name="Eye Height Measurement",
        status=TestResult.PASS,
        measured_value=0.85,
        limit=ComplianceLimit(minimum=0.7, maximum=1.0, unit="normalized"),
        margin=0.15
    ),
    USB4TestResult(
        test_name="Jitter Analysis",
        status=TestResult.PASS,
        measured_value=12.5,
        limit=ComplianceLimit(minimum=0.0, maximum=25.0, unit="ps"),
        margin=12.5
    )
]

reporter.add_test_results("USB4_TEST_001", test_results)
```

##### `generate_compliance_report(session_id: str, template_name: str = 'compliance_html') -> str`

Generate a USB4 compliance test report.

```python
report_path = reporter.generate_compliance_report(
    session_id="USB4_TEST_001",
    template_name="compliance_html"
)
print(f"Report generated: {report_path}")
```

**Returns:** Path to the generated report file

##### `generate_certification_report(session_id: str, template_name: str = 'certification_pdf') -> str`

Generate a Thunderbolt 4 certification report.

```python
cert_report_path = reporter.generate_certification_report(
    session_id="USB4_TEST_001",
    template_name="certification_pdf"
)
```

##### `generate_trend_analysis_report(test_names: Optional[List[str]] = None, time_range_days: int = 30) -> str`

Generate trend analysis report for performance monitoring.

```python
trend_report = reporter.generate_trend_analysis_report(
    test_names=["Eye Height Measurement", "Jitter Analysis"],
    time_range_days=30
)
```

##### `generate_regression_report(baseline_session_id: str, current_session_id: str) -> str`

Generate regression testing report comparing two test sessions.

```python
regression_report = reporter.generate_regression_report(
    baseline_session_id="USB4_BASELINE_001",
    current_session_id="USB4_TEST_001"
)
```

## Data Structures

### TestSession

Container for test session metadata.

```python
@dataclass
class TestSession:
    session_id: str                    # Unique session identifier
    timestamp: datetime                # Test execution timestamp
    test_type: ReportType             # Type of test report
    signal_mode: USB4SignalMode       # USB4 signal mode
    device_info: Dict[str, str]       # Device information
    test_config: Dict[str, Any]       # Test configuration
    environment: Dict[str, Any]       # Test environment info
    operator: Optional[str]           # Test operator name
    notes: Optional[str]              # Additional notes
```

### ReportTemplate

Customizable report template configuration.

```python
@dataclass
class ReportTemplate:
    name: str                         # Template name
    format: ReportFormat             # Output format
    sections: List[str]              # Report sections to include
    include_charts: bool = True      # Include charts and graphs
    include_raw_data: bool = False   # Include raw measurement data
    custom_css: Optional[str] = None # Custom CSS for HTML reports
    custom_header: Optional[str] = None  # Custom header content
    custom_footer: Optional[str] = None  # Custom footer content
    logo_path: Optional[str] = None  # Path to logo image
```

### TrendDataPoint

Data point for trend analysis.

```python
@dataclass
class TrendDataPoint:
    timestamp: datetime               # Measurement timestamp
    test_name: str                   # Name of the test
    measured_value: float            # Measured value
    limit_value: float               # Specification limit
    status: TestResult               # Test result status
    session_id: str                  # Associated session ID
    metadata: Dict[str, Any]         # Additional metadata
```

### RegressionAnalysis

Results from regression analysis between test sessions.

```python
@dataclass
class RegressionAnalysis:
    test_name: str                   # Test name
    baseline_value: float            # Baseline measurement
    current_value: float             # Current measurement
    change_percent: float            # Percentage change
    trend_direction: str             # 'improving', 'degrading', 'stable'
    significance: str                # 'critical', 'warning', 'normal'
    recommendation: str              # Recommended action
```

## Enumerations

### ReportFormat

Supported report output formats.

```python
class ReportFormat(Enum):
    HTML = auto()    # HTML format with interactive elements
    PDF = auto()     # PDF format for printing/archiving
    JSON = auto()    # JSON format for programmatic access
    XML = auto()     # XML format for data exchange
    TEXT = auto()    # Plain text format
    CSV = auto()     # CSV format for data analysis
```

### ReportType

Types of reports that can be generated.

```python
class ReportType(Enum):
    COMPLIANCE = auto()      # USB4 compliance test report
    CERTIFICATION = auto()   # Thunderbolt certification report
    PERFORMANCE = auto()     # Performance benchmark report
    TREND_ANALYSIS = auto()  # Trend analysis report
    REGRESSION = auto()      # Regression test report
    SUMMARY = auto()         # Executive summary report
    DETAILED = auto()        # Detailed technical report
```

## Report Templates

### Default Templates

The framework includes several built-in templates:

#### Compliance HTML Template
```python
compliance_template = ReportTemplate(
    name="USB4 Compliance Report",
    format=ReportFormat.HTML,
    sections=[
        'executive_summary',
        'test_configuration', 
        'test_results',
        'signal_integrity',
        'protocol_tests',
        'recommendations'
    ],
    include_charts=True,
    include_raw_data=False
)
```

#### Certification PDF Template
```python
certification_template = ReportTemplate(
    name="Thunderbolt 4 Certification Report",
    format=ReportFormat.PDF,
    sections=[
        'cover_page',
        'executive_summary',
        'device_info',
        'test_results',
        'certification_status',
        'appendix'
    ],
    include_charts=True,
    include_raw_data=True
)
```

#### Trend Analysis Template
```python
trend_template = ReportTemplate(
    name="USB4 Trend Analysis Report",
    format=ReportFormat.HTML,
    sections=[
        'trend_overview',
        'performance_trends',
        'regression_analysis',
        'recommendations'
    ],
    include_charts=True,
    include_raw_data=False
)
```

### Custom Templates

Create custom report templates:

```python
custom_template = ReportTemplate(
    name="Custom USB4 Report",
    format=ReportFormat.HTML,
    sections=['executive_summary', 'test_results'],
    include_charts=True,
    custom_css="""
        .summary { background-color: #f0f8ff; }
        .pass { color: green; }
        .fail { color: red; }
    """,
    custom_header="<h1>Company Logo</h1>",
    logo_path="assets/company_logo.png"
)

# Use custom template
report_path = reporter.generate_compliance_report(
    session_id="USB4_TEST_001",
    custom_template=custom_template
)
```

## Advanced Features

### Trend Analysis {#trend-analysis}

Track performance trends over time:

```python
# Add trend data points
trend_data = [
    TrendDataPoint(
        timestamp=datetime.now(),
        test_name="Eye Height Measurement",
        measured_value=0.85,
        limit_value=0.7,
        status=TestResult.PASS,
        session_id="USB4_TEST_001"
    ),
    TrendDataPoint(
        timestamp=datetime.now() - timedelta(days=1),
        test_name="Eye Height Measurement", 
        measured_value=0.82,
        limit_value=0.7,
        status=TestResult.PASS,
        session_id="USB4_TEST_000"
    )
]

reporter.add_trend_data(trend_data)

# Generate trend analysis
trend_report = reporter.generate_trend_analysis_report(
    test_names=["Eye Height Measurement"],
    time_range_days=30
)
```

### Performance Regression Detection

Compare test sessions for regression analysis:

```python
# Run baseline tests
baseline_session = TestSession(
    session_id="BASELINE_001",
    timestamp=datetime.now() - timedelta(days=30),
    test_type=ReportType.PERFORMANCE,
    signal_mode=USB4SignalMode.GEN3X2
)

# Run current tests
current_session = TestSession(
    session_id="CURRENT_001", 
    timestamp=datetime.now(),
    test_type=ReportType.PERFORMANCE,
    signal_mode=USB4SignalMode.GEN3X2
)

# Generate regression report
regression_report = reporter.generate_regression_report(
    baseline_session_id="BASELINE_001",
    current_session_id="CURRENT_001"
)
```

### Multi-Format Reports

Generate reports in multiple formats:

```python
session_id = "USB4_TEST_001"

# Generate HTML report for web viewing
html_report = reporter.generate_compliance_report(
    session_id=session_id,
    template_name="compliance_html"
)

# Generate PDF report for archiving
pdf_report = reporter.generate_compliance_report(
    session_id=session_id,
    template_name="compliance_pdf"
)

# Generate JSON report for programmatic access
json_report = reporter.generate_compliance_report(
    session_id=session_id,
    template_name="summary_json"
)
```

## Integration Examples

### Automated Reporting in Test Sequences

```python
from serdes_validation_framework.test_sequence import USB4TestSequence
from serdes_validation_framework.protocols.usb4.reporting import USB4TestReporter

# Run test sequence
sequence = USB4TestSequence(config)
test_results = sequence.run_complete_sequence(signal_data)

# Generate reports automatically
reporter = USB4TestReporter()

# Add session and results
session = TestSession(
    session_id=f"AUTO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    timestamp=datetime.now(),
    test_type=ReportType.COMPLIANCE,
    signal_mode=config.signal_mode
)

reporter.add_test_session(session)
reporter.add_test_results(session.session_id, test_results.test_results)

# Generate compliance report
compliance_report = reporter.generate_compliance_report(session.session_id)
print(f"Compliance report: {compliance_report}")
```

### CI/CD Integration

```python
def generate_ci_reports(test_results, build_info):
    """Generate reports for CI/CD pipeline"""
    
    reporter = USB4TestReporter(output_directory="ci_reports")
    
    # Create session from CI/CD context
    session = TestSession(
        session_id=f"CI_{build_info['build_number']}",
        timestamp=datetime.now(),
        test_type=ReportType.REGRESSION,
        signal_mode=USB4SignalMode.GEN3X2,
        environment={
            'ci_system': build_info['ci_system'],
            'branch': build_info['branch'],
            'commit': build_info['commit_sha']
        }
    )
    
    reporter.add_test_session(session)
    reporter.add_test_results(session.session_id, test_results)
    
    # Generate multiple report formats
    reports = {}
    reports['html'] = reporter.generate_compliance_report(
        session.session_id, 
        template_name="compliance_html"
    )
    reports['json'] = reporter.generate_compliance_report(
        session.session_id,
        template_name="summary_json"
    )
    
    return reports
```

### Performance Monitoring Dashboard

```python
class PerformanceMonitor:
    """Performance monitoring with automated reporting"""
    
    def __init__(self):
        self.reporter = USB4TestReporter(output_directory="performance_reports")
    
    def record_performance_data(self, test_results, session_info):
        """Record performance data for trend analysis"""
        
        # Convert test results to trend data points
        trend_points = []
        for result in test_results:
            trend_points.append(TrendDataPoint(
                timestamp=datetime.now(),
                test_name=result.test_name,
                measured_value=result.measured_value,
                limit_value=result.limit.maximum,
                status=result.status,
                session_id=session_info['session_id']
            ))
        
        self.reporter.add_trend_data(trend_points)
    
    def generate_weekly_report(self):
        """Generate weekly performance trend report"""
        return self.reporter.generate_trend_analysis_report(
            time_range_days=7
        )
    
    def check_performance_regression(self, baseline_session, current_session):
        """Check for performance regression"""
        regression_report = self.reporter.generate_regression_report(
            baseline_session_id=baseline_session,
            current_session_id=current_session
        )
        
        # Parse regression report to determine if action is needed
        # Implementation depends on specific requirements
        return regression_report
```

## Best Practices

### 1. Report Organization
- Use consistent session naming conventions
- Include comprehensive metadata in test sessions
- Organize reports by date, protocol, or test type

### 2. Template Customization
- Create organization-specific templates
- Include company branding and logos
- Customize sections based on audience needs

### 3. Data Management
- Regularly archive old reports
- Implement data retention policies
- Use appropriate storage for large datasets

### 4. Performance Monitoring
- Set up automated trend analysis
- Define regression thresholds
- Implement alerting for critical issues

### 5. Integration
- Automate report generation in CI/CD pipelines
- Integrate with existing quality management systems
- Provide programmatic access via JSON/XML formats

## Troubleshooting

### Common Issues

1. **Template Not Found**
   ```python
   # Solution: Check available templates
   print(reporter.templates.keys())
   ```

2. **Missing Test Data**
   ```python
   # Solution: Verify session and results are added
   reporter.add_test_session(session)
   reporter.add_test_results(session_id, results)
   ```

3. **Report Generation Errors**
   ```python
   # Solution: Enable debug logging
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Large Report Files**
   ```python
   # Solution: Exclude raw data for large datasets
   template.include_raw_data = False
   ```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your reporting code here
reporter = USB4TestReporter()
# ... rest of code
```

## Next Steps

- Explore [USB4/Thunderbolt API](../usb4/api-reference.md) for test result generation
- Check [USB4 Visualization](../usb4/api-reference.md#visualization) for advanced charting
- See [CI/CD Integration Guide](../guides/cicd.md) for automated reporting
- Review [Testing Guide](../guides/testing.md) for comprehensive testing strategies