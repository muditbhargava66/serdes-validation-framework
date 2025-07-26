"""
Unit tests for USB4 Jitter Analyzer

This module provides comprehensive unit tests for the USB4 jitter analyzer,
covering all major functionality including:
- Random jitter analysis with Gaussian tail fitting
- Deterministic jitter decomposition
- Periodic jitter analysis with SSC awareness
- SSC-aware jitter measurement
- Compliance validation
"""

import unittest

import numpy as np

from src.serdes_validation_framework.protocols.usb4.constants import USB4SignalMode
from src.serdes_validation_framework.protocols.usb4.jitter_analyzer import (
    DeterministicJitterResults,
    PeriodicJitterResults,
    RandomJitterResults,
    SSCAwareJitterResults,
    USB4JitterAnalyzer,
    USB4JitterConfig,
    USB4JitterResults,
)


class TestUSB4JitterAnalyzer(unittest.TestCase):
    """Test cases for USB4JitterAnalyzer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = USB4JitterConfig(
            sample_rate=256e9,
            symbol_rate=20e9,
            mode=USB4SignalMode.GEN2X2,
            enable_ssc_analysis=True,
            target_ber=1e-12,
            analysis_length=10000,
        )
        self.analyzer = USB4JitterAnalyzer(self.config)

        # Generate test signals with known jitter characteristics
        self.time_length = 10000  # samples
        self.time_data = np.arange(self.time_length) / self.config.sample_rate

        # Create realistic USB4 signal with jitter
        self.signal_data = self._generate_jittered_usb4_signal()

        # Generate timing data for testing
        self.timing_data = self._generate_timing_data_with_jitter()

    def _generate_jittered_usb4_signal(self):
        """Generate USB4 signal with controlled jitter for testing"""
        t = self.time_data
        fundamental = self.config.symbol_rate

        # Base signal
        signal = np.sin(2 * np.pi * fundamental * t)

        # Add random jitter (timing noise)
        rj_amplitude = 0.05  # 0.05 UI RMS
        timing_jitter = np.random.normal(0, rj_amplitude * self.analyzer.ui_period, len(t))
        jittered_time = t + timing_jitter
        signal = np.sin(2 * np.pi * fundamental * jittered_time)

        # Add deterministic jitter (systematic timing error)
        dj_amplitude = 0.03  # 0.03 UI
        deterministic_jitter = dj_amplitude * self.analyzer.ui_period * np.sin(2 * np.pi * 1e6 * t)
        signal = np.sin(2 * np.pi * fundamental * (t + deterministic_jitter))

        # Add periodic jitter (SSC and power supply noise)
        ssc_freq = 33e3
        ssc_amplitude = 0.02  # 0.02 UI
        ssc_jitter = ssc_amplitude * self.analyzer.ui_period * np.sin(2 * np.pi * ssc_freq * t)

        power_freq = 100e3  # 100 kHz switching noise
        power_amplitude = 0.01  # 0.01 UI
        power_jitter = power_amplitude * self.analyzer.ui_period * np.sin(2 * np.pi * power_freq * t)

        total_timing_jitter = timing_jitter + deterministic_jitter + ssc_jitter + power_jitter
        final_time = t + total_timing_jitter

        # Generate final signal
        final_signal = np.sin(2 * np.pi * fundamental * final_time)

        # Add some amplitude noise
        noise = np.random.normal(0, 0.01, len(final_signal))

        return final_signal + noise

    def _generate_timing_data_with_jitter(self):
        """Generate timing deviation data with known jitter components"""
        t = self.time_data

        # Random jitter component
        rj_component = np.random.normal(0, 0.05, len(t))  # 0.05 UI RMS

        # Deterministic jitter component
        dj_component = 0.03 * np.sin(2 * np.pi * 1e6 * t)  # 0.03 UI at 1 MHz

        # Periodic jitter components
        ssc_component = 0.02 * np.sin(2 * np.pi * 33e3 * t)  # SSC at 33 kHz
        power_component = 0.01 * np.sin(2 * np.pi * 100e3 * t)  # Power noise at 100 kHz

        # Combine all components
        total_jitter = rj_component + dj_component + ssc_component + power_component

        return total_jitter

    def test_analyzer_initialization(self):
        """Test USB4JitterAnalyzer initialization"""
        self.assertIsInstance(self.analyzer, USB4JitterAnalyzer)
        self.assertEqual(self.analyzer.config.sample_rate, 256e9)
        self.assertEqual(self.analyzer.config.symbol_rate, 20e9)
        self.assertEqual(self.analyzer.config.mode, USB4SignalMode.GEN2X2)
        self.assertTrue(self.analyzer.config.enable_ssc_analysis)
        self.assertEqual(self.analyzer.config.target_ber, 1e-12)
        self.assertEqual(self.analyzer.samples_per_ui, 256e9 // 20e9)
        self.assertEqual(self.analyzer.ui_period, 1.0 / 20e9)

    def test_config_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        valid_config = USB4JitterConfig(sample_rate=256e9, symbol_rate=20e9, analysis_length=5000)
        self.assertIsInstance(valid_config, USB4JitterConfig)

        # Test invalid configurations
        with self.assertRaises(AssertionError):
            USB4JitterConfig(sample_rate=-1e9, symbol_rate=20e9)

        with self.assertRaises(AssertionError):
            USB4JitterConfig(sample_rate=256e9, symbol_rate=-20e9)

        with self.assertRaises(AssertionError):
            USB4JitterConfig(sample_rate=10e9, symbol_rate=20e9)  # Violates Nyquist

        with self.assertRaises(AssertionError):
            USB4JitterConfig(target_ber=2.0)  # Invalid BER

        with self.assertRaises(AssertionError):
            USB4JitterConfig(analysis_length=500)  # Too short

        with self.assertRaises(AssertionError):
            USB4JitterConfig(confidence_level=1.5)  # Invalid confidence level

    def test_jitter_data_validation(self):
        """Test input jitter data validation"""
        # Test valid data
        self.analyzer._validate_jitter_data(self.signal_data, self.time_data)

        # Test invalid data types
        with self.assertRaises(AssertionError):
            self.analyzer._validate_jitter_data([1, 2, 3], self.time_data)

        # Test mismatched lengths
        short_time = self.time_data[: len(self.time_data) // 2]
        with self.assertRaises(AssertionError):
            self.analyzer._validate_jitter_data(self.signal_data, short_time)

        # Test insufficient data length
        short_signal = self.signal_data[:500]
        short_time = self.time_data[:500]
        with self.assertRaises(AssertionError):
            self.analyzer._validate_jitter_data(short_signal, short_time)

        # Test NaN values
        signal_nan = self.signal_data.copy()
        signal_nan[0] = np.nan
        with self.assertRaises(AssertionError):
            self.analyzer._validate_jitter_data(signal_nan, self.time_data)

        # Test infinite values
        signal_inf = self.signal_data.copy()
        signal_inf[0] = np.inf
        with self.assertRaises(AssertionError):
            self.analyzer._validate_jitter_data(signal_inf, self.time_data)

        # Test very small amplitude
        signal_small = np.ones_like(self.signal_data) * 1e-6
        with self.assertRaises(AssertionError):
            self.analyzer._validate_jitter_data(signal_small, self.time_data)

    def test_timing_data_extraction(self):
        """Test timing data extraction from signal"""
        timing_data = self.analyzer._extract_timing_data(self.signal_data, self.time_data)

        self.assertIsInstance(timing_data, np.ndarray)
        self.assertGreater(len(timing_data), 10)  # Should find multiple zero crossings
        self.assertTrue(np.all(np.isfinite(timing_data)))  # No NaN or inf values

        # Check that timing deviations are reasonable
        self.assertLess(np.std(timing_data), 1.0)  # Should be less than 1 UI
        self.assertGreater(np.std(timing_data), 0.001)  # Should have some variation

    def test_random_jitter_analysis(self):
        """Test random jitter analysis"""
        rj_results = self.analyzer._analyze_random_jitter(self.timing_data)

        self.assertIsInstance(rj_results, RandomJitterResults)
        self.assertGreater(rj_results.rj_rms, 0.0)
        self.assertLess(rj_results.rj_rms, 1.0)  # Should be reasonable
        self.assertGreater(rj_results.rj_pp_1e12, rj_results.rj_rms)  # PP should be larger than RMS
        self.assertGreater(rj_results.rj_pp_1e15, rj_results.rj_pp_1e12)  # 1e-15 should be larger than 1e-12
        # Gaussian fit quality can be negative for poor fits, but should be finite
        self.assertTrue(np.isfinite(rj_results.gaussian_fit_quality))
        self.assertLessEqual(rj_results.gaussian_fit_quality, 1.0)

        # Check confidence interval
        self.assertIsInstance(rj_results.confidence_interval, tuple)
        self.assertEqual(len(rj_results.confidence_interval), 2)
        self.assertLess(rj_results.confidence_interval[0], rj_results.confidence_interval[1])

        # Check distribution parameters
        self.assertIn("mean", rj_results.distribution_parameters)
        self.assertIn("std", rj_results.distribution_parameters)
        self.assertIn("skewness", rj_results.distribution_parameters)
        self.assertIn("kurtosis", rj_results.distribution_parameters)

    def test_deterministic_jitter_analysis(self):
        """Test deterministic jitter analysis"""
        dj_results = self.analyzer._analyze_deterministic_jitter(self.timing_data, self.signal_data)

        self.assertIsInstance(dj_results, DeterministicJitterResults)
        self.assertGreater(dj_results.dj_pp, 0.0)
        self.assertLess(dj_results.dj_pp, 1.0)  # Should be reasonable

        # Check DJ components
        self.assertIsInstance(dj_results.dj_components, dict)
        self.assertIn("pattern_dependent", dj_results.dj_components)
        self.assertIn("duty_cycle_distortion", dj_results.dj_components)
        self.assertIn("intersymbol_interference", dj_results.dj_components)

        # Check individual components
        self.assertGreaterEqual(dj_results.pattern_dependent_jitter, 0.0)
        self.assertGreaterEqual(dj_results.duty_cycle_distortion, 0.0)
        self.assertGreaterEqual(dj_results.intersymbol_interference, 0.0)

        # Check compliance status
        self.assertIsInstance(dj_results.compliance_status, bool)

    def test_periodic_jitter_analysis(self):
        """Test periodic jitter analysis"""
        pj_results = self.analyzer._analyze_periodic_jitter(self.timing_data, self.time_data)

        self.assertIsInstance(pj_results, PeriodicJitterResults)
        self.assertGreater(pj_results.pj_rms, 0.0)
        self.assertLess(pj_results.pj_rms, 1.0)  # Should be reasonable
        self.assertGreater(pj_results.pj_pp, pj_results.pj_rms)  # PP should be larger than RMS

        # Check dominant frequencies
        self.assertIsInstance(pj_results.dominant_frequencies, list)
        self.assertLessEqual(len(pj_results.dominant_frequencies), 5)  # Should be limited

        # Check frequency spectrum
        self.assertIsInstance(pj_results.frequency_spectrum, np.ndarray)
        self.assertTrue(np.iscomplexobj(pj_results.frequency_spectrum))

        # Check contributions
        self.assertGreaterEqual(pj_results.ssc_contribution, 0.0)
        self.assertGreaterEqual(pj_results.power_supply_noise, 0.0)

        # Check compliance status
        self.assertIsInstance(pj_results.compliance_status, bool)

    def test_ssc_aware_jitter_analysis(self):
        """Test SSC-aware jitter analysis"""
        ssc_results = self.analyzer._analyze_ssc_aware_jitter(self.signal_data, self.time_data, self.timing_data)

        self.assertIsInstance(ssc_results, SSCAwareJitterResults)
        self.assertGreater(ssc_results.total_jitter_ssc_on, 0.0)
        self.assertGreater(ssc_results.total_jitter_ssc_off, 0.0)
        self.assertGreaterEqual(ssc_results.ssc_jitter_contribution, 0.0)

        # Check clock recovery quality
        self.assertGreaterEqual(ssc_results.clock_recovery_quality, 0.0)
        self.assertLessEqual(ssc_results.clock_recovery_quality, 1.0)

        # Check tracking error
        self.assertGreaterEqual(ssc_results.ssc_tracking_error, 0.0)

        # Check modulation impact
        self.assertIsInstance(ssc_results.modulation_impact, dict)
        self.assertIn("rj_impact", ssc_results.modulation_impact)
        self.assertIn("dj_impact", ssc_results.modulation_impact)
        self.assertIn("pj_impact", ssc_results.modulation_impact)

    def test_pattern_dependent_jitter_analysis(self):
        """Test pattern-dependent jitter analysis"""
        pdj = self.analyzer._analyze_pattern_dependent_jitter(self.timing_data, self.signal_data)

        self.assertIsInstance(pdj, float)
        self.assertGreaterEqual(pdj, 0.0)
        self.assertLessEqual(pdj, 0.1)  # Should be capped at reasonable value

    def test_duty_cycle_distortion_analysis(self):
        """Test duty cycle distortion analysis"""
        dcd = self.analyzer._analyze_duty_cycle_distortion(self.timing_data)

        self.assertIsInstance(dcd, float)
        self.assertGreaterEqual(dcd, 0.0)
        self.assertLessEqual(dcd, 0.05)  # Should be capped at reasonable value

        # Test with insufficient data
        short_timing = self.timing_data[:2]
        dcd_short = self.analyzer._analyze_duty_cycle_distortion(short_timing)
        self.assertEqual(dcd_short, 0.01)  # Should return default

    def test_intersymbol_interference_analysis(self):
        """Test intersymbol interference analysis"""
        isi = self.analyzer._analyze_intersymbol_interference(self.timing_data, self.signal_data)

        self.assertIsInstance(isi, float)
        self.assertGreaterEqual(isi, 0.0)
        self.assertLessEqual(isi, 0.08)  # Should be capped at reasonable value

        # Test with insufficient data
        short_timing = self.timing_data[:1]
        isi_short = self.analyzer._analyze_intersymbol_interference(short_timing, self.signal_data)
        self.assertEqual(isi_short, 0.01)  # Should return default

    def test_ssc_contribution_analysis(self):
        """Test SSC contribution analysis"""
        # Create test data with known SSC frequency
        pj_data = 0.02 * np.sin(2 * np.pi * 33e3 * self.time_data)  # 33 kHz SSC
        dominant_frequencies = [33e3, 100e3, 1e6]

        ssc_contrib = self.analyzer._analyze_ssc_contribution(pj_data, self.time_data, dominant_frequencies)

        self.assertIsInstance(ssc_contrib, float)
        self.assertGreaterEqual(ssc_contrib, 0.0)
        self.assertLessEqual(ssc_contrib, 0.05)  # Should be capped

        # Should detect SSC frequency
        self.assertGreater(ssc_contrib, 0.0)

    def test_power_supply_noise_analysis(self):
        """Test power supply noise analysis"""
        # Test with typical power supply frequencies
        dominant_frequencies = [50, 100, 120, 100e3]  # Mix of line and switching frequencies
        pj_data = np.random.normal(0, 0.01, len(self.time_data))

        power_noise = self.analyzer._analyze_power_supply_noise(pj_data, dominant_frequencies)

        self.assertIsInstance(power_noise, float)
        self.assertGreaterEqual(power_noise, 0.0)
        self.assertLessEqual(power_noise, 0.03)  # Should be capped

    def test_clock_recovery_simulation(self):
        """Test clock recovery simulation"""
        # Test with SSC tracking enabled
        clock_ssc_on = self.analyzer._recover_clock_with_ssc(self.signal_data, self.time_data, enable_ssc_tracking=True)

        # Test with SSC tracking disabled
        clock_ssc_off = self.analyzer._recover_clock_with_ssc(self.signal_data, self.time_data, enable_ssc_tracking=False)

        self.assertIsInstance(clock_ssc_on, np.ndarray)
        self.assertIsInstance(clock_ssc_off, np.ndarray)
        self.assertEqual(len(clock_ssc_on), len(self.signal_data))
        self.assertEqual(len(clock_ssc_off), len(self.signal_data))

        # Should be different due to different tracking bandwidth
        self.assertFalse(np.array_equal(clock_ssc_on, clock_ssc_off))

    def test_ssc_component_extraction(self):
        """Test SSC component extraction"""
        ssc_component = self.analyzer._extract_ssc_component(self.timing_data, self.time_data)

        self.assertIsInstance(ssc_component, np.ndarray)
        self.assertEqual(len(ssc_component), len(self.timing_data))
        self.assertTrue(np.all(np.isfinite(ssc_component)))

    def test_clock_recovery_quality_assessment(self):
        """Test clock recovery quality assessment"""
        # Create test recovered clock
        ideal_clock = 2 * np.pi * self.config.symbol_rate * self.time_data
        noisy_clock = ideal_clock + np.random.normal(0, 0.1, len(ideal_clock))

        quality = self.analyzer._assess_clock_recovery_quality(noisy_clock, self.time_data)

        self.assertIsInstance(quality, float)
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)

        # Perfect clock should have high quality
        perfect_quality = self.analyzer._assess_clock_recovery_quality(ideal_clock, self.time_data)
        self.assertGreater(perfect_quality, quality)

    def test_ssc_tracking_error_calculation(self):
        """Test SSC tracking error calculation"""
        # Create test clocks with different tracking
        clock1 = 2 * np.pi * self.config.symbol_rate * self.time_data
        clock2 = clock1 + 0.01 * np.sin(2 * np.pi * 33e3 * self.time_data)  # Add SSC component

        tracking_error = self.analyzer._calculate_ssc_tracking_error(clock1, clock2)

        self.assertIsInstance(tracking_error, float)
        self.assertGreaterEqual(tracking_error, 0.0)

        # Should detect the difference
        self.assertGreater(tracking_error, 0.0)

    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation"""
        test_data = np.random.normal(0, 0.1, 1000)

        # Test 95% confidence interval
        ci_95 = self.analyzer._calculate_confidence_interval(test_data, 0.95)
        self.assertIsInstance(ci_95, tuple)
        self.assertEqual(len(ci_95), 2)
        self.assertLess(ci_95[0], ci_95[1])

        # Test 99% confidence interval (should be wider)
        ci_99 = self.analyzer._calculate_confidence_interval(test_data, 0.99)
        self.assertLess(ci_99[0], ci_95[0])  # Lower bound should be lower
        self.assertGreater(ci_99[1], ci_95[1])  # Upper bound should be higher

        # Test with small sample (should use t-distribution)
        small_data = test_data[:20]
        ci_small = self.analyzer._calculate_confidence_interval(small_data, 0.95)
        self.assertIsInstance(ci_small, tuple)
        self.assertEqual(len(ci_small), 2)

    def test_total_jitter_calculation(self):
        """Test total jitter calculation"""
        # Create mock jitter results
        rj_results = RandomJitterResults(
            rj_rms=0.05,
            rj_pp_1e12=0.35,
            rj_pp_1e15=0.42,
            gaussian_fit_quality=0.9,
            confidence_interval=(0.04, 0.06),
            distribution_parameters={"mean": 0.0, "std": 0.05, "skewness": 0.0, "kurtosis": 0.0},
        )

        dj_results = DeterministicJitterResults(
            dj_pp=0.15,
            dj_components={},
            pattern_dependent_jitter=0.05,
            duty_cycle_distortion=0.05,
            intersymbol_interference=0.05,
            compliance_status=True,
        )

        pj_results = PeriodicJitterResults(
            pj_rms=0.03,
            pj_pp=0.12,
            dominant_frequencies=[33e3],
            frequency_spectrum=np.array([0 + 0j]),
            ssc_contribution=0.02,
            power_supply_noise=0.01,
            compliance_status=True,
        )

        total_jitter = self.analyzer._calculate_total_jitter(rj_results, dj_results, pj_results)

        self.assertIsInstance(total_jitter, float)
        self.assertGreater(total_jitter, 0.0)

        # Should be combination of components
        expected_min = dj_results.dj_pp + rj_results.rj_pp_1e12 * 0.5
        self.assertGreater(total_jitter, expected_min)

    def test_jitter_compliance_check(self):
        """Test jitter compliance checking"""
        # Create compliant jitter results
        compliant_rj = RandomJitterResults(
            rj_rms=0.08,
            rj_pp_1e12=0.3,
            rj_pp_1e15=0.35,
            gaussian_fit_quality=0.9,
            confidence_interval=(0.07, 0.09),
            distribution_parameters={"mean": 0.0, "std": 0.08, "skewness": 0.0, "kurtosis": 0.0},
        )

        compliant_dj = DeterministicJitterResults(
            dj_pp=0.2,
            dj_components={},
            pattern_dependent_jitter=0.05,
            duty_cycle_distortion=0.05,
            intersymbol_interference=0.05,
            compliance_status=True,
        )

        compliant_pj = PeriodicJitterResults(
            pj_rms=0.08,
            pj_pp=0.15,
            dominant_frequencies=[33e3],
            frequency_spectrum=np.array([0 + 0j]),
            ssc_contribution=0.02,
            power_supply_noise=0.01,
            compliance_status=True,
        )

        total_jitter = 0.3  # Compliant total jitter

        compliance, margins = self.analyzer._check_jitter_compliance(total_jitter, compliant_rj, compliant_dj, compliant_pj)

        self.assertIsInstance(compliance, bool)
        self.assertIsInstance(margins, dict)

        # Check margin keys
        expected_keys = ["total_jitter", "random_jitter", "deterministic_jitter", "periodic_jitter"]
        for key in expected_keys:
            self.assertIn(key, margins)
            self.assertIsInstance(margins[key], float)

        # Test non-compliant case
        non_compliant_rj = RandomJitterResults(
            rj_rms=0.15,
            rj_pp_1e12=0.5,
            rj_pp_1e15=0.6,  # Exceeds limits
            gaussian_fit_quality=0.9,
            confidence_interval=(0.14, 0.16),
            distribution_parameters={"mean": 0.0, "std": 0.15, "skewness": 0.0, "kurtosis": 0.0},
        )

        non_compliance, non_margins = self.analyzer._check_jitter_compliance(
            0.8,
            non_compliant_rj,
            compliant_dj,
            compliant_pj,  # High total jitter
        )

        self.assertFalse(non_compliance)

        # Margins should be negative for non-compliant cases
        self.assertLess(non_margins["total_jitter"], 0)
        self.assertLess(non_margins["random_jitter"], 0)

    def test_jitter_recommendations_generation(self):
        """Test jitter recommendations generation"""
        # Create jitter results with various issues
        high_rj = RandomJitterResults(
            rj_rms=0.12,
            rj_pp_1e12=0.4,
            rj_pp_1e15=0.5,
            gaussian_fit_quality=0.6,
            confidence_interval=(0.11, 0.13),
            distribution_parameters={"mean": 0.0, "std": 0.12, "skewness": 0.0, "kurtosis": 0.0},
        )

        high_dj = DeterministicJitterResults(
            dj_pp=0.3,
            dj_components={},
            pattern_dependent_jitter=0.08,
            duty_cycle_distortion=0.05,
            intersymbol_interference=0.06,
            compliance_status=False,
        )

        high_pj = PeriodicJitterResults(
            pj_rms=0.12,
            pj_pp=0.2,
            dominant_frequencies=[33e3, 100e3],
            frequency_spectrum=np.array([0 + 0j]),
            ssc_contribution=0.05,
            power_supply_noise=0.03,
            compliance_status=False,
        )

        poor_ssc = SSCAwareJitterResults(
            total_jitter_ssc_on=0.4,
            total_jitter_ssc_off=0.35,
            ssc_jitter_contribution=0.05,
            clock_recovery_quality=0.6,
            ssc_tracking_error=0.03,
            modulation_impact={},
        )

        recommendations = self.analyzer._generate_jitter_recommendations(high_rj, high_dj, high_pj, poor_ssc)

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

        # Check that recommendations contain expected content
        rec_text = " ".join(recommendations)
        self.assertIn("Random jitter", rec_text)
        self.assertIn("Deterministic jitter", rec_text)
        self.assertIn("Periodic jitter", rec_text)
        self.assertIn("SSC", rec_text)

    def test_complete_jitter_analysis(self):
        """Test complete USB4 jitter analysis"""
        results = self.analyzer.analyze_usb4_jitter(self.signal_data, self.time_data)

        self.assertIsInstance(results, USB4JitterResults)

        # Check total jitter
        self.assertGreater(results.total_jitter, 0.0)
        self.assertLess(results.total_jitter, 2.0)  # Should be reasonable

        # Check component results
        self.assertIsInstance(results.random_jitter, RandomJitterResults)
        self.assertIsInstance(results.deterministic_jitter, DeterministicJitterResults)
        self.assertIsInstance(results.periodic_jitter, PeriodicJitterResults)
        self.assertIsInstance(results.ssc_aware_results, SSCAwareJitterResults)

        # Check compliance
        self.assertIsInstance(results.compliance_status, bool)
        self.assertIsInstance(results.compliance_margins, dict)

        # Check recommendations
        self.assertIsInstance(results.recommendations, list)

    def test_jitter_analysis_without_time_data(self):
        """Test jitter analysis without explicit time data"""
        results = self.analyzer.analyze_usb4_jitter(self.signal_data)

        self.assertIsInstance(results, USB4JitterResults)
        self.assertGreater(results.total_jitter, 0.0)

    def test_jitter_analysis_without_ssc(self):
        """Test jitter analysis with SSC disabled"""
        config_no_ssc = USB4JitterConfig(sample_rate=256e9, symbol_rate=20e9, enable_ssc_analysis=False, analysis_length=5000)
        analyzer_no_ssc = USB4JitterAnalyzer(config_no_ssc)

        results = analyzer_no_ssc.analyze_usb4_jitter(self.signal_data[:5000], self.time_data[:5000])

        self.assertIsInstance(results, USB4JitterResults)
        self.assertIsInstance(results.ssc_aware_results, SSCAwareJitterResults)

        # SSC results should be defaults when disabled
        self.assertEqual(results.ssc_aware_results.total_jitter_ssc_on, 0.0)
        self.assertEqual(results.ssc_aware_results.total_jitter_ssc_off, 0.0)

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with insufficient data
        short_signal = self.signal_data[:500]
        short_time = self.time_data[:500]

        with self.assertRaises(ValueError):  # Should raise ValueError, not AssertionError
            self.analyzer.analyze_usb4_jitter(short_signal, short_time)

        # Test with invalid signal (all zeros)
        zero_signal = np.zeros_like(self.signal_data)
        with self.assertRaises(ValueError):  # Should raise ValueError, not AssertionError
            self.analyzer.analyze_usb4_jitter(zero_signal, self.time_data)


class TestUSB4JitterConfig(unittest.TestCase):
    """Test cases for USB4JitterConfig class"""

    def test_default_configuration(self):
        """Test default configuration values"""
        config = USB4JitterConfig()

        self.assertEqual(config.sample_rate, 256e9)
        self.assertEqual(config.symbol_rate, 20e9)
        self.assertEqual(config.mode, USB4SignalMode.GEN2X2)
        self.assertTrue(config.enable_ssc_analysis)
        self.assertEqual(config.target_ber, 1e-12)
        self.assertEqual(config.analysis_length, 100000)
        self.assertEqual(config.confidence_level, 0.95)

    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = USB4JitterConfig(
            sample_rate=512e9,
            symbol_rate=40e9,
            mode=USB4SignalMode.GEN3X2,
            enable_ssc_analysis=False,
            target_ber=1e-15,
            analysis_length=50000,
            confidence_level=0.99,
        )

        self.assertEqual(config.sample_rate, 512e9)
        self.assertEqual(config.symbol_rate, 40e9)
        self.assertEqual(config.mode, USB4SignalMode.GEN3X2)
        self.assertFalse(config.enable_ssc_analysis)
        self.assertEqual(config.target_ber, 1e-15)
        self.assertEqual(config.analysis_length, 50000)
        self.assertEqual(config.confidence_level, 0.99)

    def test_configuration_validation(self):
        """Test configuration parameter validation"""
        # Test negative sample rate
        with self.assertRaises(AssertionError):
            USB4JitterConfig(sample_rate=-256e9)

        # Test negative symbol rate
        with self.assertRaises(AssertionError):
            USB4JitterConfig(symbol_rate=-20e9)

        # Test Nyquist violation
        with self.assertRaises(AssertionError):
            USB4JitterConfig(sample_rate=10e9, symbol_rate=20e9)

        # Test invalid target BER
        with self.assertRaises(AssertionError):
            USB4JitterConfig(target_ber=2.0)

        with self.assertRaises(AssertionError):
            USB4JitterConfig(target_ber=-1e-12)

        # Test insufficient analysis length
        with self.assertRaises(AssertionError):
            USB4JitterConfig(analysis_length=500)

        # Test invalid confidence level
        with self.assertRaises(AssertionError):
            USB4JitterConfig(confidence_level=1.5)

        with self.assertRaises(AssertionError):
            USB4JitterConfig(confidence_level=-0.1)


if __name__ == "__main__":
    unittest.main()
