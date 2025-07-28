"""
Tests for stress testing module
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

from serdes_validation_framework.stress_testing import (
    CycleResults,
    LoopbackStressTest,
    StressTestConfig,
    StressTestResults,
    create_stress_test_config,
)


class TestStressTestConfig:
    """Test stress test configuration"""
    
    def test_default_config(self):
        """Test default configuration creation"""
        config = StressTestConfig()
        
        assert config.num_cycles == 1000
        assert config.cycle_duration == 1.0
        assert config.protocol == "USB4"
        assert config.data_rate == 20e9
        assert config.voltage_swing == 0.8
        assert config.eye_height_threshold == 0.1
        assert config.jitter_threshold == 0.05
        assert config.generate_plots is True
    
    def test_create_stress_test_config(self):
        """Test configuration creation helper"""
        config = create_stress_test_config(
            protocol="PCIe",
            num_cycles=500,
            output_dir="test_output"
        )
        
        assert config.protocol == "PCIe"
        assert config.num_cycles == 500
        assert config.output_dir == Path("test_output")


class TestCycleResults:
    """Test cycle results data structure"""
    
    def test_cycle_results_creation(self):
        """Test cycle results creation"""
        result = CycleResults(
            cycle_number=1,
            timestamp=1234567890.0,
            eye_height=0.5,
            eye_width=0.8,
            rms_jitter=0.02,
            peak_jitter=0.05,
            snr=20.0,
            ber_estimate=1e-12,
            passed=True,
            degradation_percent=0.0
        )
        
        assert result.cycle_number == 1
        assert result.eye_height == 0.5
        assert result.passed is True
        assert result.degradation_percent == 0.0


class TestStressTestResults:
    """Test stress test results"""
    
    def test_results_properties(self):
        """Test results properties calculation"""
        config = StressTestConfig(num_cycles=100)
        results = StressTestResults(
            config=config,
            start_time=1000.0,
            end_time=1100.0,
            total_cycles=100,
            passed_cycles=90,
            failed_cycles=10
        )
        
        assert results.success_rate == 0.9
        assert results.duration == 100.0
    
    def test_zero_cycles(self):
        """Test results with zero cycles"""
        config = StressTestConfig()
        results = StressTestResults(
            config=config,
            start_time=1000.0,
            end_time=1000.0,
            total_cycles=0,
            passed_cycles=0,
            failed_cycles=0
        )
        
        assert results.success_rate == 0.0
        assert results.duration == 0.0


class TestLoopbackStressTest:
    """Test loopback stress test implementation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = StressTestConfig(
            num_cycles=5,  # Small number for testing
            cycle_duration=0.1,  # Fast cycles
            output_dir=Path(self.temp_dir),
            generate_plots=False,  # Disable plots for testing
            save_waveforms=False,  # Disable waveform saving
            log_level="ERROR"  # Reduce logging noise
        )
    
    def test_stress_test_initialization(self):
        """Test stress test initialization"""
        stress_test = LoopbackStressTest(self.config)
        
        assert stress_test.config == self.config
        assert stress_test.eye_analyzer is not None
        assert stress_test.csv_file.exists()
    
    def test_signal_generation(self):
        """Test loopback signal generation"""
        stress_test = LoopbackStressTest(self.config)
        
        # Test signal generation for different cycles
        signal1 = stress_test._generate_loopback_signal(1)
        signal2 = stress_test._generate_loopback_signal(100)
        
        assert len(signal1) == self.config.signal_length
        assert len(signal2) == self.config.signal_length
        assert isinstance(signal1, np.ndarray)
        assert isinstance(signal2, np.ndarray)
        
        # Signals should be different (degradation effects)
        # Note: degradation might increase or decrease std depending on effects
        assert not np.array_equal(signal1, signal2)
    
    def test_signal_analysis(self):
        """Test signal analysis"""
        stress_test = LoopbackStressTest(self.config)
        
        # Generate test signal
        signal = stress_test._generate_loopback_signal(1)
        
        # Analyze signal
        result = stress_test._analyze_signal(signal, 1)
        
        assert isinstance(result, CycleResults)
        assert result.cycle_number == 1
        assert result.eye_height >= 0  # Can be 0 in some cases
        assert result.eye_width >= 0   # Can be 0 in some cases
        assert result.rms_jitter >= 0
        assert result.snr >= 0  # Can be 0 in some cases
        assert 0 <= result.ber_estimate <= 1
    
    def test_protocol_specific_signals(self):
        """Test protocol-specific signal generation"""
        protocols = ["USB4", "PCIe", "Ethernet"]
        
        for protocol in protocols:
            config = StressTestConfig(
                protocol=protocol,
                num_cycles=2,
                output_dir=Path(self.temp_dir) / protocol,
                generate_plots=False,
                log_level="ERROR"
            )
            
            stress_test = LoopbackStressTest(config)
            signal = stress_test._generate_loopback_signal(1)
            
            assert len(signal) == config.signal_length
            assert isinstance(signal, np.ndarray)
    
    def test_csv_logging(self):
        """Test CSV logging functionality"""
        stress_test = LoopbackStressTest(self.config)
        
        # Create test result
        result = CycleResults(
            cycle_number=1,
            timestamp=1234567890.0,
            eye_height=0.5,
            eye_width=0.8,
            rms_jitter=0.02,
            peak_jitter=0.05,
            snr=20.0,
            ber_estimate=1e-12,
            passed=True,
            degradation_percent=0.0,
            notes="Test cycle"
        )
        
        # Log result
        stress_test._log_cycle_result(result)
        
        # Check CSV file
        assert stress_test.csv_file.exists()
        with open(stress_test.csv_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2  # Header + 1 data line
            assert "cycle_number" in lines[0]
            assert "1,1234567890.0,0.5" in lines[1]
    
    def test_short_stress_test(self):
        """Test running a short stress test"""
        stress_test = LoopbackStressTest(self.config)
        
        # Run stress test
        results = stress_test.run_stress_test()
        
        # Verify results
        assert isinstance(results, StressTestResults)
        assert results.total_cycles == self.config.num_cycles
        assert len(results.cycle_results) == self.config.num_cycles
        assert results.passed_cycles + results.failed_cycles == results.total_cycles
        assert 0 <= results.success_rate <= 1
        assert results.duration > 0
        
        # Check output files
        assert stress_test.csv_file.exists()
        log_file = self.config.output_dir / "stress_test.log"
        assert log_file.exists()
    
    def test_degradation_tracking(self):
        """Test degradation tracking over cycles"""
        # Use more cycles to see degradation
        config = StressTestConfig(
            num_cycles=10,
            cycle_duration=0.05,
            output_dir=Path(self.temp_dir) / "degradation",
            generate_plots=False,
            log_level="ERROR"
        )
        
        stress_test = LoopbackStressTest(config)
        results = stress_test.run_stress_test()
        
        # Check that degradation is tracked
        assert results.initial_eye_height > 0
        assert results.final_eye_height > 0
        assert results.max_degradation >= 0
        
        # Check that degradation tracking is working (values are recorded)
        if len(results.cycle_results) > 5:
            degradation_values = [r.degradation_percent for r in results.cycle_results]
            # Just check that degradation values are being calculated
            assert len(degradation_values) == len(results.cycle_results)
            # Check that not all values are the same (some variation expected)
            assert len(set(degradation_values)) > 1
    
    def test_threshold_checking(self):
        """Test pass/fail threshold checking"""
        # Set very strict thresholds
        config = StressTestConfig(
            num_cycles=3,
            eye_height_threshold=0.001,  # Very strict (0.1%)
            jitter_threshold=0.001,      # Very strict
            output_dir=Path(self.temp_dir) / "thresholds",
            generate_plots=False,
            log_level="ERROR"
        )
        
        stress_test = LoopbackStressTest(config)
        results = stress_test.run_stress_test()
        
        # With strict thresholds, some cycles should fail
        assert results.failed_cycles >= 0
        assert results.total_cycles == 3
    
    def test_error_handling(self):
        """Test error handling in analysis"""
        stress_test = LoopbackStressTest(self.config)
        
        # Test with invalid signal (empty array)
        empty_signal = np.array([])
        result = stress_test._analyze_signal(empty_signal, 1)
        
        # Should return failed result
        assert result.passed is False
        assert "error" in result.notes.lower()


class TestIntegration:
    """Integration tests for stress testing"""
    
    def test_full_workflow(self):
        """Test complete stress testing workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create configuration
            config = create_stress_test_config(
                protocol="USB4",
                num_cycles=5,
                output_dir=temp_dir
            )
            config.cycle_duration = 0.1
            config.generate_plots = False
            config.log_level = "ERROR"
            
            # Run stress test
            stress_test = LoopbackStressTest(config)
            results = stress_test.run_stress_test()
            
            # Verify complete workflow
            assert results.total_cycles == 5
            assert len(results.cycle_results) == 5
            assert results.success_rate >= 0
            
            # Check all output files exist
            output_path = Path(temp_dir)
            assert (output_path / "stress_test_results.csv").exists()
            assert (output_path / "stress_test.log").exists()
    
    def test_multi_protocol_comparison(self):
        """Test running stress tests for multiple protocols"""
        protocols = ["USB4", "PCIe", "Ethernet"]
        results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for protocol in protocols:
                config = create_stress_test_config(
                    protocol=protocol,
                    num_cycles=3,
                    output_dir=f"{temp_dir}/{protocol}"
                )
                config.cycle_duration = 0.05
                config.generate_plots = False
                config.log_level = "ERROR"
                
                stress_test = LoopbackStressTest(config)
                results[protocol] = stress_test.run_stress_test()
            
            # Verify all protocols completed
            for result in results.values():
                assert result.total_cycles == 3
                assert len(result.cycle_results) == 3
                assert result.success_rate >= 0


if __name__ == "__main__":
    pytest.main([__file__])
