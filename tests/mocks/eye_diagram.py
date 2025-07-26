"""Mock EyeDiagramAnalyzer for testing."""


class EyeDiagramAnalyzer:
    """Mock EyeDiagramAnalyzer for testing purposes."""

    def __init__(self, signal_data=None):
        self.signal_data = signal_data

    def analyze(self, signal_data=None):
        """Mock analyze method."""
        return {"eye_height": 0.8, "eye_width": 0.7, "jitter_rms": 0.05, "snr": 25.0, "ber": 1e-12}

    def generate_eye_diagram(self, signal_data=None):
        """Mock eye diagram generation."""
        return {"diagram_data": [[0.5, 0.6, 0.7], [0.4, 0.5, 0.6]], "time_axis": [0, 1, 2], "voltage_axis": [-1, 0, 1]}

    def calculate_eye_metrics(self, signal_data=None):
        """Mock eye metrics calculation."""
        return {"eye_height": 0.8, "eye_width": 0.7, "crossing_percentage": 50.0, "duty_cycle_distortion": 0.02}
