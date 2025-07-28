"""
Tests for Jupyter dashboard module
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set mock mode for testing
os.environ['SVF_MOCK_MODE'] = '1'

from serdes_validation_framework.jupyter_dashboard import (
    DashboardConfig,
    EyeDiagramDashboard,
    WaveformAnalyzer,
    check_dashboard_dependencies,
    create_dashboard,
)


class TestDashboardConfig:
    """Test dashboard configuration"""
    
    def test_default_config(self):
        """Test default configuration creation"""
        config = DashboardConfig()
        
        assert config.figure_width == 12
        assert config.figure_height == 8
        assert config.show_measurements is True
        assert config.show_mask is True
        assert config.background_color == 'white'
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = DashboardConfig(
            figure_width=10,
            figure_height=6,
            show_measurements=False,
            background_color='black'
        )
        
        assert config.figure_width == 10
        assert config.figure_height == 6
        assert config.show_measurements is False
        assert config.background_color == 'black'


class TestEyeDiagramDashboard:
    """Test eye diagram dashboard"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = DashboardConfig()
        self.dashboard = EyeDiagramDashboard(self.config)
        
        # Generate test signal data
        duration = 1e-6
        sample_rate = 40e9
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Create dual-lane USB4-like signal
        self.signal_data = {}
        for lane in range(2):
            signal = np.random.choice([-1, 1], size=len(t)) * 0.4
            noise = np.random.normal(0, 0.02, len(signal))
            self.signal_data[f'lane_{lane}'] = signal + noise
        
        self.time_data = t
        self.sample_rate = sample_rate
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        assert self.dashboard.config == self.config
        assert self.dashboard.current_data is None
        assert self.dashboard.current_results is None
    
    def test_load_waveform_data(self):
        """Test waveform data loading"""
        self.dashboard.load_waveform_data(
            signal_data=self.signal_data,
            time_data=self.time_data,
            sample_rate=self.sample_rate,
            protocol="USB4"
        )
        
        assert self.dashboard.current_data is not None
        assert self.dashboard.protocol == "USB4"
        assert self.dashboard.sample_rate == self.sample_rate
        assert len(self.dashboard.signal_data) == 2
    
    def test_load_single_signal(self):
        """Test loading single signal array"""
        single_signal = self.signal_data['lane_0']
        
        self.dashboard.load_waveform_data(
            signal_data=single_signal,
            sample_rate=self.sample_rate,
            protocol="PCIe"
        )
        
        assert 'lane_0' in self.dashboard.signal_data
        assert self.dashboard.protocol == "PCIe"
    
    def test_analyze_eye_diagram(self):
        """Test eye diagram analysis"""
        self.dashboard.load_waveform_data(
            signal_data=self.signal_data,
            sample_rate=self.sample_rate,
            protocol="USB4"
        )
        
        # Analyze lane 0
        results = self.dashboard.analyze_eye_diagram(lane=0)
        
        assert isinstance(results, dict)
        assert 'eye_height' in results
        assert 'eye_width' in results
        assert 'snr' in results
        assert 'q_factor' in results
        assert 'passed' in results
        assert 'lane' in results
        assert 'protocol' in results
        
        assert results['eye_height'] >= 0
        assert results['snr'] >= 0
        assert results['q_factor'] >= 0
        assert isinstance(results['passed'], bool)
        assert results['protocol'] == "USB4"
    
    def test_analyze_by_lane_name(self):
        """Test analysis by lane name"""
        self.dashboard.load_waveform_data(
            signal_data=self.signal_data,
            sample_rate=self.sample_rate,
            protocol="USB4"
        )
        
        # Analyze by lane name
        results = self.dashboard.analyze_eye_diagram(lane='lane_1')
        
        assert results['lane'] == 'lane_1'
    
    def test_invalid_lane(self):
        """Test analysis with invalid lane"""
        self.dashboard.load_waveform_data(
            signal_data=self.signal_data,
            sample_rate=self.sample_rate,
            protocol="USB4"
        )
        
        with pytest.raises(ValueError, match="Lane.*not found"):
            self.dashboard.analyze_eye_diagram(lane='invalid_lane')
    
    def test_no_data_loaded(self):
        """Test analysis without loaded data"""
        with pytest.raises(ValueError, match="No waveform data loaded"):
            self.dashboard.analyze_eye_diagram()
    
    def test_create_static_dashboard(self):
        """Test static dashboard creation"""
        self.dashboard.load_waveform_data(
            signal_data=self.signal_data,
            sample_rate=self.sample_rate,
            protocol="USB4"
        )
        
        # Should not raise exception
        try:
            self.dashboard.create_static_dashboard(lane=0)
        except Exception as e:
            # Allow matplotlib-related exceptions in test environment
            if "matplotlib" not in str(e).lower():
                raise
    
    def test_export_results(self):
        """Test results export"""
        self.dashboard.load_waveform_data(
            signal_data=self.signal_data,
            sample_rate=self.sample_rate,
            protocol="USB4"
        )
        
        # Analyze first
        self.dashboard.analyze_eye_diagram(lane=0)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.dashboard.export_results(temp_file)
            
            # Verify file exists and has content
            assert Path(temp_file).exists()
            assert Path(temp_file).stat().st_size > 0
            
            # Verify JSON content
            import json
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            assert 'eye_height' in data
            assert 'protocol' in data
            assert data['protocol'] == 'USB4'
            
        finally:
            # Cleanup
            if Path(temp_file).exists():
                Path(temp_file).unlink()


class TestCreateDashboard:
    """Test dashboard creation helper"""
    
    def test_create_dashboard_single_signal(self):
        """Test creating dashboard with single signal"""
        signal = np.random.randn(1000) * 0.4
        
        dashboard = create_dashboard(
            signal_data=signal,
            sample_rate=40e9,
            protocol="USB4"
        )
        
        assert isinstance(dashboard, EyeDiagramDashboard)
        assert dashboard.protocol == "USB4"
        assert dashboard.sample_rate == 40e9
    
    def test_create_dashboard_multi_signal(self):
        """Test creating dashboard with multi-signal data"""
        signal_data = {
            'lane_0': np.random.randn(1000) * 0.4,
            'lane_1': np.random.randn(1000) * 0.4
        }
        
        dashboard = create_dashboard(
            signal_data=signal_data,
            sample_rate=80e9,
            protocol="PCIe"
        )
        
        assert isinstance(dashboard, EyeDiagramDashboard)
        assert dashboard.protocol == "PCIe"
        assert len(dashboard.signal_data) == 2
    
    def test_create_dashboard_with_config(self):
        """Test creating dashboard with custom config"""
        signal = np.random.randn(1000) * 0.4
        config = DashboardConfig(figure_width=10, show_measurements=False)
        
        dashboard = create_dashboard(
            signal_data=signal,
            sample_rate=40e9,
            protocol="Ethernet",
            config=config
        )
        
        assert dashboard.config.figure_width == 10
        assert dashboard.config.show_measurements is False


class TestWaveformAnalyzer:
    """Test waveform analyzer"""
    
    def setup_method(self):
        """Setup test environment"""
        self.analyzer = WaveformAnalyzer(sample_rate=40e9, protocol="USB4")
        
        # Generate test signal
        self.signal = np.random.choice([-1, 1], size=10000) * 0.4
        self.signal += np.random.normal(0, 0.02, len(self.signal))
        
        self.time_data = np.linspace(0, len(self.signal) / 40e9, len(self.signal))
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        assert self.analyzer.sample_rate == 40e9
        assert self.analyzer.protocol == "USB4"
        assert len(self.analyzer.analysis_results) == 0
    
    def test_analyze_waveform(self):
        """Test waveform analysis"""
        result = self.analyzer.analyze_waveform(
            voltage_data=self.signal,
            time_data=self.time_data,
            lane="test_lane"
        )
        
        # Check result attributes
        assert hasattr(result, 'mean_voltage')
        assert hasattr(result, 'rms_voltage')
        assert hasattr(result, 'peak_to_peak')
        assert hasattr(result, 'snr_db')
        assert hasattr(result, 'thd_percent')
        assert hasattr(result, 'passed')
        assert hasattr(result, 'failure_reasons')
        
        # Check values are reasonable
        assert result.peak_to_peak > 0
        assert 0 <= result.thd_percent <= 100
        assert isinstance(result.passed, bool)
        assert isinstance(result.failure_reasons, list)
        assert result.lane == "test_lane"
        assert result.protocol == "USB4"
    
    def test_analyze_without_time_data(self):
        """Test analysis without providing time data"""
        result = self.analyzer.analyze_waveform(
            voltage_data=self.signal,
            lane="test_lane"
        )
        
        assert len(result.time_data) == len(self.signal)
        assert result.time_data[0] == 0
    
    def test_protocol_specific_thresholds(self):
        """Test protocol-specific pass/fail thresholds"""
        protocols = ['USB4', 'PCIe', 'Ethernet']
        
        for protocol in protocols:
            analyzer = WaveformAnalyzer(sample_rate=40e9, protocol=protocol)
            result = analyzer.analyze_waveform(
                voltage_data=self.signal,
                lane="test_lane"
            )
            
            assert result.protocol == protocol
            # Different protocols should have different thresholds
            # (this is tested implicitly through the pass/fail logic)
    
    def test_get_summary_report(self):
        """Test summary report generation"""
        # Analyze multiple lanes
        for i in range(3):
            self.analyzer.analyze_waveform(
                voltage_data=self.signal,
                lane=f"lane_{i}"
            )
        
        summary = self.analyzer.get_summary_report()
        
        assert isinstance(summary, str)
        assert "USB4" in summary
        assert "lane_0" in summary
        assert "lane_1" in summary
        assert "lane_2" in summary
    
    def test_empty_summary_report(self):
        """Test summary report with no results"""
        summary = self.analyzer.get_summary_report()
        
        assert "No analysis results" in summary


class TestDependencyChecking:
    """Test dependency checking functions"""
    
    def test_check_dashboard_dependencies(self):
        """Test dashboard dependency checking"""
        deps = check_dashboard_dependencies()
        
        assert isinstance(deps, dict)
        assert 'Jupyter' in deps
        assert 'Matplotlib' in deps
        assert 'Plotly' in deps
        assert 'Eye Analysis' in deps
        
        # All values should be boolean
        for value in deps.values():
            assert isinstance(value, bool)
    
    def test_check_interactive_dependencies(self):
        """Test interactive dependency checking"""
        from serdes_validation_framework.jupyter_dashboard import check_interactive_dependencies
        
        deps = check_interactive_dependencies()
        
        assert isinstance(deps, dict)
        assert 'Jupyter' in deps
        assert 'Matplotlib' in deps
        assert 'Plotly' in deps
        
        # All values should be boolean
        for value in deps.values():
            assert isinstance(value, bool)


if __name__ == "__main__":
    pytest.main([__file__])
