#!/usr/bin/env python3
"""
PCIe Integration Tests

Comprehensive integration tests for PCIe protocol implementation
including dual-mode operation, link training, and compliance testing.
"""

import sys
import unittest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

import numpy as np
from serdes_validation_framework.instrument_control.mode_switcher import SignalMode as ModeSwitcherSignalMode
from serdes_validation_framework.instrument_control.mode_switcher import create_mode_switcher
from serdes_validation_framework.instrument_control.pcie_analyzer import PCIeAnalyzer, PCIeConfig
from serdes_validation_framework.protocols.pcie.compliance import ComplianceConfig, ComplianceTestSuite, ComplianceType
from serdes_validation_framework.protocols.pcie.constants import (
    PCIE_SPECS,
    SignalMode,
    calculate_ui_parameters,
    validate_link_width,
)
from serdes_validation_framework.protocols.pcie.equalization import create_lms_equalizer, create_rls_equalizer
from serdes_validation_framework.protocols.pcie.link_training import create_nrz_trainer, create_pam4_trainer

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


class TestPCIeIntegration(unittest.TestCase):
    """Integration tests for PCIe functionality"""
    
    @classmethod
    def setUpClass(cls) -> None:
        """Set up class-level test fixtures"""
        cls.TEST_SIGNAL_PARAMS = {
            'num_samples': 1000,
            'sample_rate': 100e9,
            'snr_db': 25.0
        }
        
        # Generate test signals
        cls.nrz_signal = cls._generate_test_signal(SignalMode.NRZ)
        cls.pam4_signal = cls._generate_test_signal(SignalMode.PAM4)
    
    @classmethod
    def _generate_test_signal(
        cls,
        mode: SignalMode
    ) -> dict[str, np.ndarray]:
        """Generate test signal for given mode"""
        num_samples = cls.TEST_SIGNAL_PARAMS['num_samples']
        sample_rate = cls.TEST_SIGNAL_PARAMS['sample_rate']
        snr_db = cls.TEST_SIGNAL_PARAMS['snr_db']
        
        # Time vector
        time = np.linspace(0, num_samples/sample_rate, num_samples, dtype=np.float64)
        
        # Generate signal based on mode
        if mode == SignalMode.NRZ:
            data = np.random.choice([-1.0, 1.0], size=num_samples)
            signal_power = 1.0
        else:  # PAM4
            levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
            data = np.random.choice(levels, size=num_samples)
            signal_power = np.mean(levels**2)
        
        # Add noise
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
        voltage = data + noise
        
        return {
            'time': time.astype(np.float64),
            'voltage': voltage.astype(np.float64)
        }
    
    def test_constants_validation(self) -> None:
        """Test PCIe constants and validation functions"""
        # Test link width validation
        self.assertTrue(validate_link_width(1))
        self.assertTrue(validate_link_width(16))
        self.assertFalse(validate_link_width(0))
        self.assertFalse(validate_link_width(32))
        
        # Test UI parameter calculation
        ui_period, ui_tolerance = calculate_ui_parameters()
        self.assertIsInstance(ui_period, float)
        self.assertIsInstance(ui_tolerance, float)
        self.assertGreater(ui_period, 0)
        self.assertGreater(ui_tolerance, 0)
        
        # Test specifications access
        self.assertIn('base', PCIE_SPECS)
        self.assertIn('nrz', PCIE_SPECS)
        self.assertIn('pam4', PCIE_SPECS)
        self.assertIn('training', PCIE_SPECS)
    
    def test_mode_switcher_integration(self) -> None:
        """Test mode switcher functionality"""
        # Create mode switcher
        switcher = create_mode_switcher(
            default_mode=ModeSwitcherSignalMode.NRZ,
            sample_rate=50e9,
            bandwidth=25e9
        )
        
        # Test initial state
        self.assertEqual(switcher.get_current_mode(), ModeSwitcherSignalMode.NRZ)
        
        # Test mode switching
        result = switcher.switch_mode(ModeSwitcherSignalMode.PAM4)
        self.assertTrue(result.success)
        self.assertEqual(switcher.get_current_mode(), ModeSwitcherSignalMode.PAM4)
        
        # Test switch back
        result = switcher.switch_mode(ModeSwitcherSignalMode.NRZ)
        self.assertTrue(result.success)
        self.assertEqual(switcher.get_current_mode(), ModeSwitcherSignalMode.NRZ)
    
    def test_pcie_analyzer_integration(self) -> None:
        """Test PCIe analyzer with both signal modes"""
        for mode in [SignalMode.NRZ, SignalMode.PAM4]:
            with self.subTest(mode=mode):
                # Create analyzer configuration
                config = PCIeConfig(
                    mode=mode,
                    sample_rate=100e9,
                    bandwidth=50e9,
                    voltage_range=2.0,
                    link_speed=32e9,
                    lane_count=1
                )
                
                # Create analyzer
                analyzer = PCIeAnalyzer(config)
                
                # Get test signal
                signal_data = self.nrz_signal if mode == SignalMode.NRZ else self.pam4_signal
                
                # Analyze signal
                results = analyzer.analyze_signal(signal_data)
                
                # Validate results
                self.assertIsInstance(results, dict)
                self.assertGreater(len(results), 0)
                
                # Check for expected metrics
                if mode == SignalMode.NRZ:
                    self.assertIn('level_separation', results)
                    self.assertIn('snr_db', results)
                    self.assertIn('jitter_ps', results)
                else:  # PAM4
                    self.assertIn('min_level_separation', results)
                    self.assertIn('rms_evm_percent', results)
                    self.assertIn('snr_db', results)
                
                # Validate metric types and ranges
                for _, value in results.items():
                    self.assertIsInstance(value, float)
                    self.assertFalse(np.isnan(value))
                    self.assertFalse(np.isinf(value))
    
    def test_link_training_integration(self) -> None:
        """Test link training for both modes"""
        for mode in [SignalMode.NRZ, SignalMode.PAM4]:
            with self.subTest(mode=mode):
                # Create trainer
                if mode == SignalMode.NRZ:
                    trainer = create_nrz_trainer(target_ber=1e-6, max_iterations=100)
                else:
                    trainer = create_pam4_trainer(target_ber=1e-6, max_iterations=200)
                
                # Get test signal
                signal_data = self.nrz_signal if mode == SignalMode.NRZ else self.pam4_signal
                
                # Run training
                result = trainer.run_training(signal_data)
                
                # Validate results
                self.assertIsInstance(result.success, bool)
                self.assertIsInstance(result.final_ber, float)
                self.assertIsInstance(result.iterations, int)
                self.assertIsInstance(result.equalizer_coeffs, dict)
                self.assertIsInstance(result.snr_history, list)
                
                # Check ranges
                self.assertGreaterEqual(result.final_ber, 0)
                self.assertLessEqual(result.final_ber, 1)
                self.assertGreaterEqual(result.iterations, 0)
                self.assertGreater(len(result.snr_history), 0)
    
    def test_equalization_integration(self) -> None:
        """Test adaptive equalization algorithms"""
        # Generate distorted signal
        signal = self.nrz_signal['voltage']
        
        # Add ISI
        isi_filter = np.array([0.1, 0.8, 0.1], dtype=np.float64)
        distorted_signal = np.convolve(signal, isi_filter, mode='same')
        
        # Test different equalizers
        equalizers = [
            create_lms_equalizer(num_forward_taps=5, step_size=0.01),
            create_rls_equalizer(num_forward_taps=5, forgetting_factor=0.99)
        ]
        
        for i, equalizer in enumerate(equalizers):
            with self.subTest(equalizer=i):
                # Run equalization
                result = equalizer.equalize_signal(distorted_signal)
                
                # Validate results
                self.assertIsInstance(result.converged, bool)
                self.assertIsInstance(result.final_mse, float)
                self.assertIsInstance(result.iterations, int)
                self.assertIsInstance(result.forward_coeffs, list)
                self.assertIsInstance(result.feedback_coeffs, list)
                
                # Check ranges
                self.assertGreaterEqual(result.final_mse, 0)
                self.assertGreaterEqual(result.iterations, 0)
                self.assertGreater(len(result.forward_coeffs), 0)
                
                # Test equalization application
                equalized = equalizer.apply_equalization(distorted_signal)
                self.assertEqual(len(equalized), len(distorted_signal))
                self.assertEqual(equalized.dtype, np.float64)
    
    def test_compliance_testing_integration(self) -> None:
        """Test compliance testing functionality"""
        # Create compliance configuration
        config = ComplianceConfig(
            test_pattern="PRBS31",
            sample_rate=100e9,
            record_length=10e-6,
            voltage_range=2.0,
            test_types=[ComplianceType.ELECTRICAL, ComplianceType.TIMING]
        )
        
        # Create test suite
        test_suite = ComplianceTestSuite(config)
        
        # Run compliance tests
        results = test_suite.run_compliance_tests(
            self.nrz_signal['time'],
            self.nrz_signal['voltage']
        )
        
        # Validate results structure
        self.assertIsInstance(results, dict)
        self.assertIn('electrical', results)
        self.assertIn('timing', results)
        
        # Validate electrical tests
        electrical_tests = results['electrical']
        self.assertIsInstance(electrical_tests, dict)
        self.assertGreater(len(electrical_tests), 0)
        
        # Validate timing tests
        timing_tests = results['timing']
        self.assertIsInstance(timing_tests, dict)
        self.assertGreater(len(timing_tests), 0)
        
        # Check individual test results
        for category_results in results.values():
            for test_result in category_results.values():
                self.assertIsInstance(test_result.test_name, str)
                self.assertIsInstance(test_result.measured_value, float)
                self.assertIsInstance(test_result.status, bool)
        
        # Test report generation
        report = test_suite.generate_report()
        self.assertIsInstance(report, dict)
        self.assertIn('status', report)
        self.assertIn('results', report)
        self.assertIsInstance(report['status'], bool)
        self.assertIsInstance(report['results'], list)
    
    def test_end_to_end_workflow(self) -> None:
        """Test complete end-to-end PCIe validation workflow"""
        # 1. Mode switching
        switcher = create_mode_switcher()
        switch_result = switcher.switch_mode(ModeSwitcherSignalMode.PAM4)
        self.assertTrue(switch_result.success)
        
        # 2. Signal analysis
        config = PCIeConfig(
            mode=SignalMode.PAM4,
            sample_rate=100e9,
            bandwidth=50e9,
            voltage_range=2.0,
            link_speed=32e9,
            lane_count=1
        )
        analyzer = PCIeAnalyzer(config)
        analysis_results = analyzer.analyze_signal(self.pam4_signal)
        self.assertGreater(len(analysis_results), 0)
        
        # 3. Link training
        trainer = create_pam4_trainer(target_ber=1e-6, max_iterations=100)
        training_result = trainer.run_training(self.pam4_signal)
        self.assertIsInstance(training_result.success, bool)
        
        # 4. Equalization
        equalizer = create_lms_equalizer(num_forward_taps=7)
        eq_result = equalizer.equalize_signal(self.pam4_signal['voltage'])
        self.assertIsInstance(eq_result.converged, bool)
        
        # 5. Compliance testing
        compliance_config = ComplianceConfig(
            test_pattern="PRBS31",
            sample_rate=100e9,
            record_length=10e-6,
            voltage_range=2.0,
            test_types=[ComplianceType.ELECTRICAL]
        )
        compliance_suite = ComplianceTestSuite(compliance_config)
        compliance_results = compliance_suite.run_compliance_tests(
            self.pam4_signal['time'],
            self.pam4_signal['voltage']
        )
        self.assertGreater(len(compliance_results), 0)
        
        # Verify workflow completed successfully
        self.assertTrue(True)  # If we get here, the workflow succeeded
    
    def test_error_handling(self) -> None:
        """Test error handling and validation"""
        # Test invalid signal data
        with self.assertRaises((AssertionError, ValueError)):
            config = PCIeConfig(
                mode=SignalMode.NRZ,
                sample_rate=100e9,
                bandwidth=50e9,
                voltage_range=2.0,
                link_speed=32e9,
                lane_count=1
            )
            analyzer = PCIeAnalyzer(config)
            analyzer.analyze_signal({'invalid': 'data'})
        
        # Test invalid configuration
        with self.assertRaises(AssertionError):
            PCIeConfig(
                mode="invalid_mode",  # type: ignore
                sample_rate=100e9,
                bandwidth=50e9,
                voltage_range=2.0,
                link_speed=32e9,
                lane_count=1
            )
        
        # Test invalid equalizer parameters
        with self.assertRaises(AssertionError):
            create_lms_equalizer(num_forward_taps=-1)  # Invalid tap count
    
    def test_performance_benchmarks(self) -> None:
        """Test performance benchmarks"""
        import time
        
        # Benchmark signal analysis
        config = PCIeConfig(
            mode=SignalMode.NRZ,
            sample_rate=100e9,
            bandwidth=50e9,
            voltage_range=2.0,
            link_speed=32e9,
            lane_count=1
        )
        analyzer = PCIeAnalyzer(config)
        
        start_time = time.time()
        analyzer.analyze_signal(self.nrz_signal)
        analysis_time = time.time() - start_time
        
        # Analysis should complete within reasonable time
        self.assertLess(analysis_time, 5.0)  # 5 seconds max
        
        # Benchmark equalization
        equalizer = create_lms_equalizer(num_forward_taps=11)
        
        start_time = time.time()
        equalizer.equalize_signal(self.nrz_signal['voltage'][:500])  # Smaller signal for speed
        eq_time = time.time() - start_time
        
        # Equalization should complete within reasonable time
        self.assertLess(eq_time, 10.0)  # 10 seconds max


class TestPCIeTypeValidation(unittest.TestCase):
    """Test type validation throughout PCIe implementation"""
    
    def test_signal_mode_validation(self) -> None:
        """Test SignalMode enum validation"""
        # Valid modes
        self.assertIsInstance(SignalMode.NRZ, SignalMode)
        self.assertIsInstance(SignalMode.PAM4, SignalMode)
        
        # Test mode comparison
        self.assertNotEqual(SignalMode.NRZ, SignalMode.PAM4)
    
    def test_floating_point_validation(self) -> None:
        """Test floating-point type validation"""
        # Test with valid float
        ui_period, ui_tolerance = calculate_ui_parameters(64e9)
        self.assertIsInstance(ui_period, float)
        self.assertIsInstance(ui_tolerance, float)
        
        # Test with invalid type
        with self.assertRaises(AssertionError):
            calculate_ui_parameters("invalid")  # type: ignore
    
    def test_array_validation(self) -> None:
        """Test numpy array validation"""
        # Valid arrays
        time_data = np.linspace(0, 1e-6, 1000, dtype=np.float64)
        voltage_data = np.random.randn(1000).astype(np.float64)
        
        config = ComplianceConfig(
            test_pattern="PRBS31",
            sample_rate=100e9,
            record_length=10e-6,
            voltage_range=2.0,
            test_types=[ComplianceType.ELECTRICAL]
        )
        suite = ComplianceTestSuite(config)
        
        # Should not raise exception
        suite.validate_signal_data(time_data, voltage_data)
        
        # Invalid arrays
        with self.assertRaises(AssertionError):
            suite.validate_signal_data(
                np.array([1, 2, 3], dtype=int),  # Wrong dtype
                voltage_data
            )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
