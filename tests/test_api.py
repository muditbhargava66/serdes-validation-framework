"""
Tests for REST API functionality
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set mock mode for testing
os.environ['SVF_MOCK_MODE'] = '1'

from serdes_validation_framework.api.core import SerDesAPI, TestManager
from serdes_validation_framework.api.models import (
    EyeDiagramRequest,
    FixtureControlRequest,
    ProtocolType,
    StatusType,
    StressTestRequest,
    WaveformAnalysisRequest,
)


class TestAPICore:
    """Test API core functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.api = SerDesAPI()
        self.test_signal = np.random.randn(1000) * 0.4
    
    def test_api_initialization(self):
        """Test API initialization"""
        assert self.api is not None
        assert isinstance(self.api.test_manager, TestManager)
        assert len(self.api.fixtures) == 0
    
    def test_eye_diagram_analysis(self):
        """Test eye diagram analysis"""
        request = EyeDiagramRequest(
            signal_data=self.test_signal.tolist(),
            sample_rate=40e9,
            protocol=ProtocolType.USB4,
            lane=0,
            show_mask=True
        )
        
        result = self.api.analyze_eye_diagram(request)
        
        assert result.protocol == "USB4"
        assert result.eye_height > 0
        assert result.eye_width >= 0
        assert result.q_factor > 0
        assert isinstance(result.passed, bool)
        assert result.mask_analysis is not None
        assert 'compliance_level' in result.mask_analysis
    
    def test_system_status(self):
        """Test system status"""
        status = self.api.get_system_status()
        
        assert status.version is not None
        assert status.status == "healthy"
        assert status.active_tests >= 0
        assert status.total_tests_run >= 0
        assert status.uptime_seconds >= 0
        assert status.memory_usage_mb > 0
        assert len(status.available_protocols) == 3
        assert "USB4" in status.available_protocols
        assert "PCIe" in status.available_protocols
        assert "Ethernet" in status.available_protocols
        assert status.features['eye_analysis'] is True


class TestTestManager:
    """Test test manager functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_manager = TestManager()
    
    def test_test_manager_initialization(self):
        """Test test manager initialization"""
        assert len(self.test_manager.active_tests) == 0
        assert len(self.test_manager.completed_tests) == 0
        assert self.test_manager.total_tests_run == 0
    
    def test_create_test(self):
        """Test test creation"""
        test_id = self.test_manager.create_test('test_type', param1='value1')
        
        assert test_id.startswith('test_type_')
        assert test_id in self.test_manager.active_tests
        
        test_info = self.test_manager.get_test_status(test_id)
        assert test_info['type'] == 'test_type'
        assert test_info['status'] == StatusType.PENDING
        assert test_info['kwargs']['param1'] == 'value1'


class TestStressTesting:
    """Test stress testing via API"""
    
    def setup_method(self):
        """Setup test environment"""
        self.api = SerDesAPI()
    
    def test_start_stress_test(self):
        """Test starting stress test"""
        request = StressTestRequest(
            protocol=ProtocolType.USB4,
            num_cycles=5,
            cycle_duration=0.1,  # Short duration for testing
            enable_bert_hooks=False
        )
        
        test_id = self.api.start_stress_test(request)
        
        assert test_id is not None
        assert test_id.startswith('stress_test_')
        
        # Check test status
        status = self.api.get_test_status(test_id)
        assert status is not None
        assert status.test_id == test_id


class TestAPIValidation:
    """Test API input validation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.api = SerDesAPI()
    
    def test_invalid_sample_rate(self):
        """Test invalid sample rate"""
        with pytest.raises(ValueError):
            EyeDiagramRequest(
                signal_data=[0.1, -0.1],
                sample_rate=-1000,  # Negative sample rate
                protocol=ProtocolType.USB4
            )
    
    def test_invalid_num_cycles(self):
        """Test invalid number of cycles"""
        with pytest.raises(ValueError):
            StressTestRequest(
                protocol=ProtocolType.USB4,
                num_cycles=20000  # Too many cycles
            )


if __name__ == "__main__":
    pytest.main([__file__])
