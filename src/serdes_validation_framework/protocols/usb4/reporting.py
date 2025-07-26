"""
USB4/Thunderbolt 4 Test Reporting System

This module provides comprehensive USB4 and Thunderbolt 4 test reporting capabilities
including certification-ready report generation, trend analysis, regression testing reports,
and customizable report templates and formatting.

Features:
- USB4TestReporter class with comprehensive reporting
- Certification-ready report generation
- Trend analysis and regression testing reports
- Customizable report templates and formatting
- Multiple output formats (HTML, PDF, JSON, XML)
- Test result aggregation and analysis
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np

from .base import TestResult, USB4TestResult
from .compliance import USB4ComplianceResult
from .constants import (
    ThunderboltSpecs,
    USB4SignalMode,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report output formats"""

    HTML = auto()
    PDF = auto()
    JSON = auto()
    XML = auto()
    TEXT = auto()
    CSV = auto()


class ReportType(Enum):
    """Types of USB4 test reports"""

    COMPLIANCE = auto()  # USB4 compliance test report
    CERTIFICATION = auto()  # Thunderbolt certification report
    PERFORMANCE = auto()  # Performance benchmark report
    TREND_ANALYSIS = auto()  # Trend analysis report
    REGRESSION = auto()  # Regression test report
    SUMMARY = auto()  # Executive summary report
    DETAILED = auto()  # Detailed technical report


@dataclass
class ReportTemplate:
    """Customizable report template configuration"""

    name: str
    format: ReportFormat
    sections: List[str] = field(default_factory=list)
    include_charts: bool = True
    include_raw_data: bool = False
    custom_css: Optional[str] = None
    custom_header: Optional[str] = None
    custom_footer: Optional[str] = None
    logo_path: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate template configuration"""
        assert isinstance(self.name, str) and self.name.strip(), "Template name must be non-empty string"
        assert isinstance(self.format, ReportFormat), "Format must be ReportFormat enum"
        assert isinstance(self.sections, list), "Sections must be a list"
        assert all(isinstance(s, str) for s in self.sections), "All sections must be strings"


@dataclass
class TestSession:
    """USB4 test session information"""

    session_id: str
    timestamp: datetime
    test_type: ReportType
    signal_mode: USB4SignalMode
    device_info: Dict[str, str] = field(default_factory=dict)
    test_config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    operator: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate test session data"""
        assert isinstance(self.session_id, str) and self.session_id.strip(), "Session ID must be non-empty string"
        assert isinstance(self.timestamp, datetime), "Timestamp must be datetime object"
        assert isinstance(self.test_type, ReportType), "Test type must be ReportType enum"
        assert isinstance(self.signal_mode, USB4SignalMode), "Signal mode must be USB4SignalMode enum"


@dataclass
class TrendDataPoint:
    """Data point for trend analysis"""

    timestamp: datetime
    test_name: str
    measured_value: float
    limit_value: float
    status: TestResult
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate trend data point"""
        assert isinstance(self.timestamp, datetime), "Timestamp must be datetime object"
        assert isinstance(self.test_name, str) and self.test_name.strip(), "Test name must be non-empty string"
        assert isinstance(self.measured_value, (int, float)), "Measured value must be numeric"
        assert isinstance(self.limit_value, (int, float)), "Limit value must be numeric"
        assert isinstance(self.status, TestResult), "Status must be TestResult enum"
        assert isinstance(self.session_id, str) and self.session_id.strip(), "Session ID must be non-empty string"


@dataclass
class RegressionAnalysis:
    """Regression analysis results"""

    test_name: str
    baseline_value: float
    current_value: float
    change_percent: float
    trend_direction: str  # 'improving', 'degrading', 'stable'
    significance: str  # 'critical', 'warning', 'normal'
    recommendation: str

    def __post_init__(self) -> None:
        """Validate regression analysis data"""
        assert isinstance(self.test_name, str) and self.test_name.strip(), "Test name must be non-empty string"
        assert isinstance(self.baseline_value, (int, float)), "Baseline value must be numeric"
        assert isinstance(self.current_value, (int, float)), "Current value must be numeric"
        assert isinstance(self.change_percent, (int, float)), "Change percent must be numeric"
        assert self.trend_direction in ["improving", "degrading", "stable"], "Invalid trend direction"
        assert self.significance in ["critical", "warning", "normal"], "Invalid significance level"
        assert isinstance(self.recommendation, str), "Recommendation must be string"


class USB4TestReporter:
    """Comprehensive USB4 test reporting system"""

    def __init__(self, output_directory: str = "usb4_reports") -> None:
        """
        Initialize USB4 test reporter

        Args:
            output_directory: Directory for report output files
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.test_sessions: List[TestSession] = []
        self.test_results: Dict[str, List[USB4TestResult]] = {}
        self.compliance_results: Dict[str, List[USB4ComplianceResult]] = {}
        self.trend_data: List[TrendDataPoint] = []

        # Initialize default templates
        self._initialize_default_templates()

        logger.info(f"USB4 test reporter initialized with output directory: {self.output_directory}")

    def _initialize_default_templates(self) -> None:
        """Initialize default report templates"""
        self.templates: Dict[str, ReportTemplate] = {
            "compliance_html": ReportTemplate(
                name="USB4 Compliance Report",
                format=ReportFormat.HTML,
                sections=[
                    "executive_summary",
                    "test_configuration",
                    "test_results",
                    "signal_integrity",
                    "protocol_tests",
                    "recommendations",
                ],
                include_charts=True,
                include_raw_data=False,
            ),
            "certification_pdf": ReportTemplate(
                name="Thunderbolt 4 Certification Report",
                format=ReportFormat.PDF,
                sections=["cover_page", "executive_summary", "device_info", "test_results", "certification_status", "appendix"],
                include_charts=True,
                include_raw_data=True,
            ),
            "trend_analysis": ReportTemplate(
                name="USB4 Trend Analysis Report",
                format=ReportFormat.HTML,
                sections=["trend_overview", "performance_trends", "regression_analysis", "recommendations"],
                include_charts=True,
                include_raw_data=False,
            ),
            "summary_json": ReportTemplate(
                name="USB4 Test Summary",
                format=ReportFormat.JSON,
                sections=["summary", "results", "statistics"],
                include_charts=False,
                include_raw_data=True,
            ),
        }

    def add_test_session(self, session: TestSession) -> None:
        """
        Add test session information

        Args:
            session: Test session to add
        """
        assert isinstance(session, TestSession), "Session must be TestSession object"
        self.test_sessions.append(session)
        logger.info(f"Added test session: {session.session_id}")

    def add_test_results(self, session_id: str, results: List[USB4TestResult]) -> None:
        """
        Add test results for a session

        Args:
            session_id: Test session identifier
            results: List of test results
        """
        assert isinstance(session_id, str) and session_id.strip(), "Session ID must be non-empty string"
        assert isinstance(results, list), "Results must be a list"
        assert all(isinstance(r, USB4TestResult) for r in results), "All results must be USB4TestResult objects"

        if session_id not in self.test_results:
            self.test_results[session_id] = []
        self.test_results[session_id].extend(results)
        logger.info(f"Added {len(results)} test results for session {session_id}")

    def add_compliance_results(self, session_id: str, results: List[USB4ComplianceResult]) -> None:
        """
        Add compliance test results for a session

        Args:
            session_id: Test session identifier
            results: List of compliance results
        """
        assert isinstance(session_id, str) and session_id.strip(), "Session ID must be non-empty string"
        assert isinstance(results, list), "Results must be a list"
        assert all(isinstance(r, USB4ComplianceResult) for r in results), "All results must be USB4ComplianceResult objects"

        if session_id not in self.compliance_results:
            self.compliance_results[session_id] = []
        self.compliance_results[session_id].extend(results)
        logger.info(f"Added {len(results)} compliance results for session {session_id}")

    def add_trend_data(self, data_points: List[TrendDataPoint]) -> None:
        """
        Add trend analysis data points

        Args:
            data_points: List of trend data points
        """
        assert isinstance(data_points, list), "Data points must be a list"
        assert all(isinstance(dp, TrendDataPoint) for dp in data_points), "All data points must be TrendDataPoint objects"

        self.trend_data.extend(data_points)
        logger.info(f"Added {len(data_points)} trend data points")

    def generate_compliance_report(
        self, session_id: str, template_name: str = "compliance_html", custom_template: Optional[ReportTemplate] = None
    ) -> str:
        """
        Generate USB4 compliance test report

        Args:
            session_id: Test session identifier
            template_name: Name of template to use
            custom_template: Custom template override

        Returns:
            Path to generated report file
        """
        # Get session and results
        session = self._get_session(session_id)
        compliance_results = self.compliance_results.get(session_id, [])
        test_results = self.test_results.get(session_id, [])

        # Use custom template or default
        template = custom_template or self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        # Generate report content
        report_data = self._build_compliance_report_data(session, compliance_results, test_results)

        # Format and save report
        if template.format == ReportFormat.HTML:
            return self._generate_html_report(report_data, template, f"compliance_{session_id}")
        elif template.format == ReportFormat.JSON:
            return self._generate_json_report(report_data, template, f"compliance_{session_id}")
        elif template.format == ReportFormat.XML:
            return self._generate_xml_report(report_data, template, f"compliance_{session_id}")
        else:
            # Implement support for additional formats
            logger.warning(f"Format {template.format.name} not fully optimized, using HTML fallback")
            return self._generate_html_report(report_data, template, f"compliance_{session_id}")

    def generate_certification_report(self, session_id: str, template_name: str = "certification_pdf") -> str:
        """
        Generate Thunderbolt 4 certification report

        Args:
            session_id: Test session identifier
            template_name: Name of template to use

        Returns:
            Path to generated report file
        """
        session = self._get_session(session_id)
        compliance_results = self.compliance_results.get(session_id, [])

        # Build certification-specific data
        cert_data = self._build_certification_report_data(session, compliance_results)

        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        # Generate certification report
        if template.format == ReportFormat.HTML:
            return self._generate_html_report(cert_data, template, f"certification_{session_id}")
        elif template.format == ReportFormat.JSON:
            return self._generate_json_report(cert_data, template, f"certification_{session_id}")
        else:
            # Implement support for additional certification formats
            logger.warning(f"Certification format {template.format.name} not fully optimized, using HTML fallback")
            return self._generate_html_report(cert_data, template, f"certification_{session_id}")

    def generate_trend_analysis_report(
        self, test_names: Optional[List[str]] = None, time_range_days: int = 30, template_name: str = "trend_analysis"
    ) -> str:
        """
        Generate trend analysis report

        Args:
            test_names: Specific tests to analyze (None for all)
            time_range_days: Number of days to include in analysis
            template_name: Name of template to use

        Returns:
            Path to generated report file
        """
        # Filter trend data by time range
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - time_range_days)

        filtered_data = [dp for dp in self.trend_data if dp.timestamp >= cutoff_date]

        if test_names:
            filtered_data = [dp for dp in filtered_data if dp.test_name in test_names]

        # Perform trend analysis
        trend_analysis = self._perform_trend_analysis(filtered_data)
        regression_analysis = self._perform_regression_analysis(filtered_data)

        # Build report data
        report_data = {
            "report_type": "trend_analysis",
            "generation_time": datetime.now().isoformat(),
            "time_range_days": time_range_days,
            "data_points_analyzed": len(filtered_data),
            "trend_analysis": trend_analysis,
            "regression_analysis": regression_analysis,
            "recommendations": self._generate_trend_recommendations(regression_analysis),
        }

        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        return self._generate_html_report(report_data, template, f"trend_analysis_{datetime.now().strftime('%Y%m%d')}")

    def generate_regression_report(
        self, baseline_session_id: str, current_session_id: str, template_name: str = "summary_json"
    ) -> str:
        """
        Generate regression testing report

        Args:
            baseline_session_id: Baseline test session ID
            current_session_id: Current test session ID
            template_name: Name of template to use

        Returns:
            Path to generated report file
        """
        baseline_results = self.compliance_results.get(baseline_session_id, [])
        current_results = self.compliance_results.get(current_session_id, [])

        if not baseline_results or not current_results:
            raise ValueError("Both baseline and current results required for regression analysis")

        # Perform regression analysis
        regression_data = self._compare_test_sessions(baseline_results, current_results)

        # Build report data
        report_data = {
            "report_type": "regression",
            "generation_time": datetime.now().isoformat(),
            "baseline_session": baseline_session_id,
            "current_session": current_session_id,
            "regression_analysis": regression_data,
            "summary": self._generate_regression_summary(regression_data),
        }

        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        return self._generate_json_report(report_data, template, f"regression_{current_session_id}")

    def _get_session(self, session_id: str) -> TestSession:
        """Get test session by ID"""
        for session in self.test_sessions:
            if session.session_id == session_id:
                return session
        raise ValueError(f"Session '{session_id}' not found")

    def _build_compliance_report_data(
        self, session: TestSession, compliance_results: List[USB4ComplianceResult], test_results: List[USB4TestResult]
    ) -> Dict[str, Any]:
        """Build compliance report data structure"""
        # Calculate summary statistics
        total_tests = len(compliance_results)
        passed_tests = sum(1 for r in compliance_results if r.status)
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # Group results by category
        results_by_category = {}
        for result in compliance_results:
            category = result.test_category.name
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(
                {
                    "test_name": result.test_name,
                    "measured_value": result.measured_value,
                    "limit_min": result.limit.minimum,
                    "limit_max": result.limit.maximum,
                    "unit": result.limit.unit,
                    "status": "PASS" if result.status else "FAIL",
                    "margin": result.margin,
                    "diagnostic_info": result.diagnostic_info,
                }
            )

        return {
            "report_type": "compliance",
            "generation_time": datetime.now().isoformat(),
            "session_info": {
                "session_id": session.session_id,
                "timestamp": session.timestamp.isoformat(),
                "signal_mode": session.signal_mode.name,
                "device_info": session.device_info,
                "operator": session.operator,
                "notes": session.notes,
            },
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": pass_rate,
                "overall_status": "PASS" if failed_tests == 0 else "FAIL",
            },
            "test_results": results_by_category,
            "recommendations": self._generate_compliance_recommendations(compliance_results),
        }

    def _build_certification_report_data(
        self, session: TestSession, compliance_results: List[USB4ComplianceResult]
    ) -> Dict[str, Any]:
        """Build Thunderbolt certification report data"""
        # Filter for Thunderbolt-specific tests
        tb_results = [r for r in compliance_results if "thunderbolt" in r.test_name.lower()]

        # Determine certification status
        cert_status = all(r.status for r in tb_results) if tb_results else False

        return {
            "report_type": "certification",
            "generation_time": datetime.now().isoformat(),
            "certification_status": "CERTIFIED" if cert_status else "NOT_CERTIFIED",
            "session_info": {
                "session_id": session.session_id,
                "timestamp": session.timestamp.isoformat(),
                "device_info": session.device_info,
            },
            "thunderbolt_results": [
                {
                    "test_name": r.test_name,
                    "status": "PASS" if r.status else "FAIL",
                    "measured_value": r.measured_value,
                    "limit": f"{r.limit.minimum} - {r.limit.maximum} {r.limit.unit}",
                    "margin": r.margin,
                }
                for r in tb_results
            ],
            "certification_requirements": self._get_certification_requirements(),
        }

    def _perform_trend_analysis(self, data_points: List[TrendDataPoint]) -> Dict[str, Any]:
        """Perform trend analysis on data points"""
        if not data_points:
            return {}

        # Group by test name
        tests_data = {}
        for dp in data_points:
            if dp.test_name not in tests_data:
                tests_data[dp.test_name] = []
            tests_data[dp.test_name].append(dp)

        trend_results = {}
        for test_name, test_data in tests_data.items():
            if len(test_data) < 2:
                continue

            # Sort by timestamp
            test_data.sort(key=lambda x: x.timestamp)

            # Calculate trend
            values = [dp.measured_value for dp in test_data]
            timestamps = [(dp.timestamp - test_data[0].timestamp).total_seconds() for dp in test_data]

            # Simple linear regression
            if len(values) > 1:
                slope = np.polyfit(timestamps, values, 1)[0]
                trend_direction = "improving" if slope < 0 else "degrading" if slope > 0 else "stable"
            else:
                slope = 0
                trend_direction = "stable"

            trend_results[test_name] = {
                "data_points": len(test_data),
                "slope": float(slope),
                "trend_direction": trend_direction,
                "latest_value": values[-1],
                "earliest_value": values[0],
                "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0,
            }

        return trend_results

    def _perform_regression_analysis(self, data_points: List[TrendDataPoint]) -> List[RegressionAnalysis]:
        """Perform regression analysis comparing recent vs baseline performance"""
        if len(data_points) < 2:
            return []

        # Group by test name and sort by timestamp
        tests_data = {}
        for dp in data_points:
            if dp.test_name not in tests_data:
                tests_data[dp.test_name] = []
            tests_data[dp.test_name].append(dp)

        regression_results = []
        for test_name, test_data in tests_data.items():
            if len(test_data) < 2:
                continue

            test_data.sort(key=lambda x: x.timestamp)

            # Use first 25% as baseline, last 25% as current
            baseline_count = max(1, len(test_data) // 4)
            current_count = max(1, len(test_data) // 4)

            baseline_values = [dp.measured_value for dp in test_data[:baseline_count]]
            current_values = [dp.measured_value for dp in test_data[-current_count:]]

            baseline_avg = np.mean(baseline_values)
            current_avg = np.mean(current_values)

            change_percent = ((current_avg - baseline_avg) / baseline_avg * 100) if baseline_avg != 0 else 0

            # Determine trend and significance
            if abs(change_percent) < 5:
                trend_direction = "stable"
                significance = "normal"
            elif change_percent > 0:
                trend_direction = "degrading"
                significance = "critical" if change_percent > 20 else "warning"
            else:
                trend_direction = "improving"
                significance = "normal"

            recommendation = self._generate_test_recommendation(test_name, trend_direction, change_percent)

            regression_results.append(
                RegressionAnalysis(
                    test_name=test_name,
                    baseline_value=baseline_avg,
                    current_value=current_avg,
                    change_percent=change_percent,
                    trend_direction=trend_direction,
                    significance=significance,
                    recommendation=recommendation,
                )
            )

        return regression_results

    def _compare_test_sessions(
        self, baseline_results: List[USB4ComplianceResult], current_results: List[USB4ComplianceResult]
    ) -> Dict[str, Any]:
        """Compare two test sessions for regression analysis"""
        # Create lookup for baseline results
        baseline_lookup = {r.test_name: r for r in baseline_results}

        comparisons = []
        for current in current_results:
            if current.test_name in baseline_lookup:
                baseline = baseline_lookup[current.test_name]

                change_percent = (
                    ((current.measured_value - baseline.measured_value) / baseline.measured_value * 100)
                    if baseline.measured_value != 0
                    else 0
                )

                comparisons.append(
                    {
                        "test_name": current.test_name,
                        "baseline_value": baseline.measured_value,
                        "current_value": current.measured_value,
                        "change_percent": change_percent,
                        "baseline_status": "PASS" if baseline.status else "FAIL",
                        "current_status": "PASS" if current.status else "FAIL",
                        "status_changed": baseline.status != current.status,
                    }
                )

        return {
            "comparisons": comparisons,
            "summary": {
                "tests_compared": len(comparisons),
                "status_changes": sum(1 for c in comparisons if c["status_changed"]),
                "significant_changes": sum(1 for c in comparisons if abs(c["change_percent"]) > 10),
            },
        }

    def _generate_html_report(self, data: Dict[str, Any], template: ReportTemplate, filename: str) -> str:
        """Generate HTML report"""
        html_content = self._build_html_content(data, template)

        output_path = self.output_directory / f"{filename}.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Generated HTML report: {output_path}")
        return str(output_path)

    def _generate_json_report(self, data: Dict[str, Any], template: ReportTemplate, filename: str) -> str:
        """Generate JSON report"""
        output_path = self.output_directory / f"{filename}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Generated JSON report: {output_path}")
        return str(output_path)

    def _generate_xml_report(self, data: Dict[str, Any], template: ReportTemplate, filename: str) -> str:
        """Generate XML report"""
        root = Element("usb4_report")
        self._dict_to_xml(data, root)

        # Pretty print XML
        rough_string = tostring(root, "unicode")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        output_path = self.output_directory / f"{filename}.xml"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        logger.info(f"Generated XML report: {output_path}")
        return str(output_path)

    def _build_html_content(self, data: Dict[str, Any], template: ReportTemplate) -> str:
        """Build HTML content from data and template"""
        html = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{template.name}</title>
            <style>
                {self._get_default_css()}
                {template.custom_css or ''}
            </style>
        </head>
        <body>
            {template.custom_header or ''}
            <div class="container">
                <header>
                    <h1>{template.name}</h1>
                    <p>Generated: {data.get('generation_time', 'Unknown')}</p>
                </header>
                
                {self._build_html_sections(data, template.sections)}
            </div>
            {template.custom_footer or ''}
        </body>
        </html>
        """
        return html

    def _build_html_sections(self, data: Dict[str, Any], sections: List[str]) -> str:
        """Build HTML sections based on template configuration"""
        html_sections = []

        for section in sections:
            if section == "executive_summary":
                html_sections.append(self._build_summary_section(data))
            elif section == "test_results":
                html_sections.append(self._build_results_section(data))
            elif section == "recommendations":
                html_sections.append(self._build_recommendations_section(data))
            elif section == "trend_overview":
                html_sections.append(self._build_trend_section(data))
            # Add more sections as needed

        return "\n".join(html_sections)

    def _build_summary_section(self, data: Dict[str, Any]) -> str:
        """Build executive summary section"""
        summary = data.get("summary", {})
        return f"""
        <section class="summary">
            <h2>Executive Summary</h2>
            <div class="summary-stats">
                <div class="stat">
                    <span class="stat-value">{summary.get('total_tests', 0)}</span>
                    <span class="stat-label">Total Tests</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{summary.get('passed_tests', 0)}</span>
                    <span class="stat-label">Passed</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{summary.get('failed_tests', 0)}</span>
                    <span class="stat-label">Failed</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{summary.get('pass_rate', 0):.1f}%</span>
                    <span class="stat-label">Pass Rate</span>
                </div>
            </div>
            <div class="overall-status {summary.get('overall_status', 'UNKNOWN').lower()}">
                Overall Status: {summary.get('overall_status', 'UNKNOWN')}
            </div>
        </section>
        """

    def _build_results_section(self, data: Dict[str, Any]) -> str:
        """Build test results section"""
        test_results = data.get("test_results", {})
        html = '<section class="test-results"><h2>Test Results</h2>'

        for category, results in test_results.items():
            html += f'<h3>{category.replace("_", " ").title()}</h3>'
            html += '<table class="results-table">'
            html += "<thead><tr><th>Test Name</th><th>Measured</th><th>Limit</th><th>Status</th><th>Margin</th></tr></thead>"
            html += "<tbody>"

            for result in results:
                status_class = "pass" if result["status"] == "PASS" else "fail"
                html += f"""
                <tr class="{status_class}">
                    <td>{result['test_name']}</td>
                    <td>{result['measured_value']:.3f} {result['unit']}</td>
                    <td>{result['limit_min']:.3f} - {result['limit_max']:.3f} {result['unit']}</td>
                    <td class="status">{result['status']}</td>
                    <td>{result['margin']:.3f}</td>
                </tr>
                """

            html += "</tbody></table>"

        html += "</section>"
        return html

    def _build_recommendations_section(self, data: Dict[str, Any]) -> str:
        """Build recommendations section"""
        recommendations = data.get("recommendations", [])
        if not recommendations:
            return ""

        html = '<section class="recommendations"><h2>Recommendations</h2><ul>'
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul></section>"
        return html

    def _build_trend_section(self, data: Dict[str, Any]) -> str:
        """Build trend analysis section"""
        trend_data = data.get("trend_analysis", {})
        if not trend_data:
            return ""

        html = '<section class="trend-analysis"><h2>Trend Analysis</h2>'
        html += '<table class="trend-table">'
        html += "<thead><tr><th>Test Name</th><th>Trend</th><th>Change %</th><th>Data Points</th></tr></thead>"
        html += "<tbody>"

        for test_name, trend in trend_data.items():
            trend_class = trend["trend_direction"]
            html += f"""
            <tr class="{trend_class}">
                <td>{test_name}</td>
                <td class="trend-direction">{trend['trend_direction'].title()}</td>
                <td>{trend['change_percent']:.1f}%</td>
                <td>{trend['data_points']}</td>
            </tr>
            """

        html += "</tbody></table></section>"
        return html

    def _get_default_css(self) -> str:
        """Get default CSS styles for HTML reports"""
        return """
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        header { text-align: center; margin-bottom: 30px; border-bottom: 2px solid #007acc; padding-bottom: 20px; }
        h1 { color: #007acc; margin: 0; }
        h2 { color: #333; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
        h3 { color: #555; }
        .summary-stats { display: flex; justify-content: space-around; margin: 20px 0; }
        .stat { text-align: center; }
        .stat-value { display: block; font-size: 2em; font-weight: bold; color: #007acc; }
        .stat-label { color: #666; }
        .overall-status { text-align: center; padding: 15px; margin: 20px 0; border-radius: 5px; font-weight: bold; }
        .overall-status.pass { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .overall-status.fail { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        tr.pass { background-color: #f8fff8; }
        tr.fail { background-color: #fff8f8; }
        .status { font-weight: bold; }
        .pass .status { color: #28a745; }
        .fail .status { color: #dc3545; }
        .trend-direction { font-weight: bold; }
        .improving { color: #28a745; }
        .degrading { color: #dc3545; }
        .stable { color: #6c757d; }
        section { margin: 30px 0; }
        ul { padding-left: 20px; }
        li { margin: 5px 0; }
        """

    def _dict_to_xml(self, data: Dict[str, Any], parent: Element) -> None:
        """Convert dictionary to XML elements"""
        for key, value in data.items():
            element = SubElement(parent, str(key))
            if isinstance(value, dict):
                self._dict_to_xml(value, element)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        item_element = SubElement(element, "item")
                        self._dict_to_xml(item, item_element)
                    else:
                        item_element = SubElement(element, "item")
                        item_element.text = str(item)
            else:
                element.text = str(value)

    def _generate_compliance_recommendations(self, results: List[USB4ComplianceResult]) -> List[str]:
        """Generate recommendations based on compliance results"""
        recommendations = []

        for result in results:
            if not result.status:
                if "jitter" in result.test_name.lower():
                    recommendations.append(
                        f"High jitter detected in {result.test_name}. Consider improving clock quality or reducing noise sources."
                    )
                elif "eye" in result.test_name.lower():
                    recommendations.append(
                        f"Eye diagram issue in {result.test_name}. Check signal integrity and equalization settings."
                    )
                elif "voltage" in result.test_name.lower():
                    recommendations.append(f"Voltage level issue in {result.test_name}. Verify driver settings and termination.")
                elif "skew" in result.test_name.lower():
                    recommendations.append("Lane skew issue detected. Check PCB routing and differential pair matching.")
                else:
                    recommendations.append(
                        f"Test failure in {result.test_name}. Review test conditions and device configuration."
                    )

        if not recommendations:
            recommendations.append("All tests passed. Device meets USB4 compliance requirements.")

        return recommendations

    def _generate_trend_recommendations(self, regression_analysis: List[RegressionAnalysis]) -> List[str]:
        """Generate recommendations based on trend analysis"""
        recommendations = []

        critical_issues = [r for r in regression_analysis if r.significance == "critical"]
        warning_issues = [r for r in regression_analysis if r.significance == "warning"]

        if critical_issues:
            recommendations.append(
                f"Critical performance degradation detected in {len(critical_issues)} tests. Immediate investigation required."
            )

        if warning_issues:
            recommendations.append(
                f"Performance warnings in {len(warning_issues)} tests. Monitor closely for further degradation."
            )

        improving_tests = [r for r in regression_analysis if r.trend_direction == "improving"]
        if improving_tests:
            recommendations.append(f"Performance improvements observed in {len(improving_tests)} tests. Good trend.")

        if not critical_issues and not warning_issues:
            recommendations.append("Performance trends are stable. Continue regular monitoring.")

        return recommendations

    def _generate_test_recommendation(self, test_name: str, trend_direction: str, change_percent: float) -> str:
        """Generate specific recommendation for a test"""
        if trend_direction == "degrading":
            if abs(change_percent) > 20:
                return f"Critical degradation in {test_name}. Immediate action required."
            else:
                return f"Performance degradation in {test_name}. Monitor and investigate."
        elif trend_direction == "improving":
            return f"Performance improvement in {test_name}. Positive trend."
        else:
            return f"Stable performance in {test_name}. Continue monitoring."

    def _generate_regression_summary(self, regression_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of regression analysis"""
        comparisons = regression_data.get("comparisons", [])

        if not comparisons:
            return {"status": "no_data", "message": "No comparable data available"}

        significant_degradations = [c for c in comparisons if c["change_percent"] > 10]
        significant_improvements = [c for c in comparisons if c["change_percent"] < -10]
        status_regressions = [c for c in comparisons if c["status_changed"] and c["current_status"] == "FAIL"]

        overall_status = "pass"
        if status_regressions or significant_degradations:
            overall_status = "fail"
        elif significant_improvements:
            overall_status = "improved"

        return {
            "overall_status": overall_status,
            "significant_degradations": len(significant_degradations),
            "significant_improvements": len(significant_improvements),
            "status_regressions": len(status_regressions),
            "recommendation": self._get_regression_recommendation(
                overall_status, len(significant_degradations), len(status_regressions)
            ),
        }

    def _get_regression_recommendation(self, status: str, degradations: int, regressions: int) -> str:
        """Get recommendation based on regression analysis"""
        if status == "fail":
            return f"Regression detected with {degradations} performance degradations and {regressions} test failures. Investigation required."
        elif status == "improved":
            return "Performance improvements detected. Positive regression."
        else:
            return "No significant regressions detected. Performance is stable."

    def _get_certification_requirements(self) -> Dict[str, Any]:
        """Get Thunderbolt 4 certification requirements"""
        tb_specs = ThunderboltSpecs()
        return {
            "bandwidth": f"{tb_specs.TB4_BANDWIDTH / 1e9:.0f} Gbps",
            "power_delivery": f"{tb_specs.TB4_POWER_DELIVERY:.0f} W",
            "security": "DMA protection required",
            "daisy_chain": f"Up to {tb_specs.MAX_DAISY_DEVICES} devices",
            "displays": f"Up to {tb_specs.MAX_4K_DISPLAYS} 4K displays",
        }


__all__ = [
    "ReportFormat",
    "ReportType",
    "ReportTemplate",
    "TestSession",
    "TrendDataPoint",
    "RegressionAnalysis",
    "USB4TestReporter",
]
