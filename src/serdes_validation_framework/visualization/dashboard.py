"""
Validation Dashboard

Creates comprehensive interactive dashboards for validation results.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ValidationDashboard:
    """Interactive validation dashboard creator"""

    def __init__(self):
        """Initialize dashboard creator"""
        self.plotly_available = False

        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            self.go = go
            self.px = px
            self.make_subplots = make_subplots
            self.plotly_available = True
            logger.info("Plotly available for dashboard creation")
        except ImportError:
            logger.warning("Plotly not available - dashboard creation disabled")

    def create_validation_dashboard(self, results: Dict[str, Any], **kwargs) -> Optional[str]:
        """Create comprehensive validation dashboard"""
        if not self.plotly_available:
            logger.error("Plotly required for dashboard creation")
            return None

        try:
            # Create dashboard layout
            fig = self.make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("Test Results", "Signal Quality", "Performance", "Compliance"),
                specs=[[{"type": "pie"}, {"type": "bar"}], [{"type": "scatter"}, {"type": "bar"}]],
            )

            # Add plots based on available data
            if "test_results" in results:
                self._add_test_results_pie(fig, results["test_results"], row=1, col=1)

            if "signal_quality" in results:
                self._add_signal_quality_bar(fig, results["signal_quality"], row=1, col=2)

            if "performance" in results:
                self._add_performance_scatter(fig, results["performance"], row=2, col=1)

            if "compliance" in results:
                self._add_compliance_bar(fig, results["compliance"], row=2, col=2)

            # Update layout
            fig.update_layout(title_text="SerDes Validation Dashboard", title_x=0.5, showlegend=True, height=800)

            # Save dashboard
            filename = kwargs.get("filename", "validation_dashboard.html")
            fig.write_html(filename)

            logger.info(f"Dashboard saved to {filename}")
            return filename

        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            return None

    def _add_test_results_pie(self, fig, test_results: Dict, row: int, col: int):
        """Add test results pie chart"""
        try:
            passed = sum(1 for r in test_results.values() if r.get("status") == "PASS")
            failed = sum(1 for r in test_results.values() if r.get("status") == "FAIL")

            fig.add_trace(
                self.go.Pie(labels=["Passed", "Failed"], values=[passed, failed], marker_colors=["green", "red"]),
                row=row,
                col=col,
            )
        except Exception as e:
            logger.error(f"Adding test results pie failed: {e}")

    def _add_signal_quality_bar(self, fig, signal_quality: Dict, row: int, col: int):
        """Add signal quality bar chart"""
        try:
            metrics = list(signal_quality.keys())
            values = list(signal_quality.values())

            fig.add_trace(self.go.Bar(x=metrics, y=values, marker_color="blue"), row=row, col=col)
        except Exception as e:
            logger.error(f"Adding signal quality bar failed: {e}")

    def _add_performance_scatter(self, fig, performance: Dict, row: int, col: int):
        """Add performance scatter plot"""
        try:
            if "throughput" in performance and "latency" in performance:
                fig.add_trace(
                    self.go.Scatter(
                        x=[performance["throughput"]],
                        y=[performance["latency"]],
                        mode="markers",
                        marker=dict(size=15, color="orange"),
                        name="Performance",
                    ),
                    row=row,
                    col=col,
                )
        except Exception as e:
            logger.error(f"Adding performance scatter failed: {e}")

    def _add_compliance_bar(self, fig, compliance: Dict, row: int, col: int):
        """Add compliance bar chart"""
        try:
            standards = list(compliance.keys())
            statuses = [1 if compliance[std] else 0 for std in standards]
            colors = ["green" if s else "red" for s in statuses]

            fig.add_trace(self.go.Bar(x=standards, y=statuses, marker_color=colors), row=row, col=col)
        except Exception as e:
            logger.error(f"Adding compliance bar failed: {e}")
