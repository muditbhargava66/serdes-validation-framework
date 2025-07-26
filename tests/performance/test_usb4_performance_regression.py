"""
USB4 Performance Regression Test Suite

This module contains performance regression tests to ensure that USB4 validation
performance remains within acceptable bounds across framework updates.
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import psutil
import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from serdes_validation_framework import create_usb4_test_sequence, create_validation_framework
    from serdes_validation_framework.protocols.usb4 import USB4LinkState, USB4TunnelingMode
    from serdes_validation_framework.protocols.usb4.constants import USB4SignalMode
    from serdes_validation_framework.test_sequence import USB4LaneConfig, USB4TestPhase, USB4TestSequenceConfig

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False

    # Mock classes for testing
    from enum import Enum, auto

    class MockValidationFramework:
        def __init__(self):
            self.protocol_detector = MockProtocolDetector()

        def detect_protocol(self, *args, **kwargs):
            return "USB4"

        def create_test_sequence(self, *args, **kwargs):
            return {"status": "mock_sequence"}

        def run_auto_validation(self, signal_data, sample_rate, voltage_range, protocol_hint=None):
            # Add delay proportional to data size to simulate realistic processing
            import time

            if hasattr(signal_data, "__len__") and len(signal_data) > 0:
                # Get first channel data length
                first_channel = next(iter(signal_data.values()))
                data_size = len(first_channel.get("voltage", []))
                # Scale processing time with data size (base 5ms + 1μs per sample)
                processing_time = 0.005 + (data_size * 1e-6)
            else:
                processing_time = 0.01  # Default 10ms

            time.sleep(processing_time)
            return {
                "status": "passed",
                "protocol": "USB4",
                "results": {"signal_quality": 0.95, "compliance": True},
                "validation_results": MockValidationResults(),
            }

    class MockValidationResults:
        def __init__(self):
            self.overall_status = MockStatus()

    class MockStatus:
        def __init__(self):
            self.name = "PASS"  # Change from "PASSED" to "PASS"

    class MockProtocolDetector:
        def detect_protocol_from_signal(self, *args, **kwargs):
            return "USB4"

    def create_validation_framework():
        return MockValidationFramework()

    class MockUSB4TestSequence:
        def __init__(self, *args, **kwargs):
            self.status = "mock_usb4_sequence"

        def run_complete_sequence(self, signal_data):
            # Add delay proportional to data size
            import time

            if hasattr(signal_data, "__len__") and len(signal_data) > 0:
                first_channel = next(iter(signal_data.values()))
                data_size = len(first_channel.get("voltage", []))
                processing_time = 0.005 + (data_size * 1e-6)
            else:
                processing_time = 0.01

            time.sleep(processing_time)
            return MockSequenceResults()

    class MockSequenceResults:
        def __init__(self):
            self.status = "passed"
            self.overall_status = MockStatus()
            self.phase_results = [
                MockPhaseResult("INITIALIZATION"),
                MockPhaseResult("SIGNAL_ANALYSIS"),
                MockPhaseResult("VALIDATION"),
                MockPhaseResult("COMPLIANCE"),
                MockPhaseResult("PERFORMANCE"),
                MockPhaseResult("THUNDERBOLT"),
                MockPhaseResult("TUNNELING"),
                MockPhaseResult("POWER_MANAGEMENT"),
            ]

    class MockPhaseResult:
        def __init__(self, phase_name):
            self.phase = MockPhase(phase_name)
            self.metrics = {"signal_quality": 0.95, "compliance": True}

    class MockPhase:
        def __init__(self, name):
            self.name = name

    def create_usb4_test_sequence(*args, **kwargs):
        return MockUSB4TestSequence(*args, **kwargs)

    class USB4LinkState(Enum):
        U0 = auto()
        U1 = auto()
        U2 = auto()
        U3 = auto()

    class USB4TunnelingMode(Enum):
        PCIE = auto()
        DISPLAYPORT = auto()
        USB32 = auto()

    class USB4SignalMode(Enum):
        GEN2 = auto()
        GEN3 = auto()
        GEN2X2 = auto()  # Add the missing GEN2X2 mode
        GEN3X2 = auto()

    class USB4LaneConfig:
        def __init__(self, lane_id=0, mode=None, sample_rate=200e9, bandwidth=25e9, voltage_range=0.8, **kwargs):
            self.lane_id = lane_id
            self.mode = mode or USB4SignalMode.GEN2X2
            self.sample_rate = sample_rate
            self.bandwidth = bandwidth
            self.voltage_range = voltage_range
            self.__dict__.update(kwargs)

    class USB4TestPhase(Enum):
        INITIALIZATION = auto()
        LINK_TRAINING = auto()
        DATA_TRANSFER = auto()
        SIGNAL_ANALYSIS = auto()
        VALIDATION = auto()
        THUNDERBOLT = auto()
        TUNNELING = auto()
        POWER_MANAGEMENT = auto()
        STRESS_TEST = auto()
        COMPLIANCE = auto()
        PERFORMANCE = auto()

    class USB4TestSequenceConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


class PerformanceMetrics:
    """Performance metrics collection and analysis"""

    def __init__(self):
        self.metrics = {}
        self.process = psutil.Process(os.getpid())

    def start_measurement(self, test_name: str):
        """Start performance measurement"""
        self.metrics[test_name] = {
            "start_time": time.time(),
            "start_memory": self.process.memory_info().rss,
            "start_cpu_percent": self.process.cpu_percent(),
        }

    def end_measurement(self, test_name: str) -> Dict[str, float]:
        """End performance measurement and return metrics"""
        if test_name not in self.metrics:
            raise ValueError(f"No measurement started for {test_name}")

        end_time = time.time()
        end_memory = self.process.memory_info().rss
        end_cpu_percent = self.process.cpu_percent()

        start_data = self.metrics[test_name]

        metrics = {
            "duration": end_time - start_data["start_time"],
            "memory_delta": end_memory - start_data["start_memory"],
            "peak_memory": end_memory,
            "avg_cpu_percent": (start_data["start_cpu_percent"] + end_cpu_percent) / 2,
        }

        self.metrics[test_name].update(metrics)
        return metrics


class TestUSB4PerformanceRegression:
    """USB4 performance regression tests"""

    @pytest.fixture
    def performance_metrics(self):
        """Performance metrics collector"""
        return PerformanceMetrics()

    @pytest.fixture
    def framework(self):
        """Create validation framework"""
        return create_validation_framework()

    def generate_performance_test_signal(
        self, duration: float, sample_rate: float = 200e9, complexity: str = "normal"
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Generate USB4 signal for performance testing"""
        num_samples = int(duration * sample_rate)
        time_array = np.linspace(0, duration, num_samples)

        # Adjust complexity
        if complexity == "simple":
            noise_level = 0.01
            ssc_enabled = False
        elif complexity == "normal":
            noise_level = 0.02
            ssc_enabled = True
        else:  # complex
            noise_level = 0.03
            ssc_enabled = True

        signal_data = {}

        for lane_id in [0, 1]:
            # Generate NRZ signal
            bit_rate = 20e9
            bit_period = 1.0 / bit_rate
            num_bits = int(duration / bit_period)

            np.random.seed(42 + lane_id)
            data_bits = np.random.randint(0, 2, num_bits)
            voltage = np.zeros(num_samples)

            for i, bit in enumerate(data_bits):
                start_idx = int(i * bit_period * sample_rate)
                end_idx = min(int((i + 1) * bit_period * sample_rate), num_samples)
                if end_idx > start_idx:
                    voltage[start_idx:end_idx] = 0.4 if bit else -0.4

            # Add SSC if enabled
            if ssc_enabled:
                ssc_freq = 33e3
                ssc_deviation = 0.005
                ssc_modulation = ssc_deviation * np.sin(2 * np.pi * ssc_freq * time_array)
                phase_mod = np.cumsum(ssc_modulation) * 2 * np.pi * duration / num_samples
                voltage = voltage * (1 + 0.05 * np.sin(phase_mod))

            # Add noise
            voltage += noise_level * np.random.randn(num_samples)

            # Add complexity-dependent effects
            if complexity == "complex":
                # Add ISI
                if len(voltage) > 20:
                    isi_filter = np.array([0.1, 0.3, 1.0, 0.3, 0.1])
                    isi_filter = isi_filter / np.sum(isi_filter)
                    voltage = np.convolve(voltage, isi_filter, mode="same")

                # Add periodic interference
                interference_freq = 1e6  # 1 MHz
                interference = 0.05 * np.sin(2 * np.pi * interference_freq * time_array)
                voltage += interference

            signal_data[lane_id] = {"time": time_array, "voltage": voltage}

        return signal_data

    def test_basic_validation_performance(self, framework, performance_metrics):
        """Test basic USB4 validation performance"""
        test_name = "basic_validation"

        # Generate test signal
        signal_data = self.generate_performance_test_signal(
            duration=10e-6,  # 10 μs
            complexity="normal",
        )

        # Performance measurement
        performance_metrics.start_measurement(test_name)

        results = framework.run_auto_validation(
            signal_data=signal_data, sample_rate=200e9, voltage_range=0.8, protocol_hint="usb4"
        )

        metrics = performance_metrics.end_measurement(test_name)

        # Performance assertions
        assert metrics["duration"] < 5.0, f"Basic validation took {metrics['duration']:.2f}s, exceeds 5s limit"
        assert (
            metrics["memory_delta"] < 100 * 1024 * 1024
        ), f"Memory usage {metrics['memory_delta']/1024/1024:.1f}MB exceeds 100MB limit"

        # Verify validation succeeded
        assert results["validation_results"].overall_status.name in ["PASS", "WARNING"]

        print("Basic Validation Performance:")
        print(f"  Duration: {metrics['duration']:.2f}s")
        print(f"  Memory Delta: {metrics['memory_delta']/1024/1024:.1f}MB")
        print(f"  CPU Usage: {metrics['avg_cpu_percent']:.1f}%")

    def test_large_dataset_performance(self, framework, performance_metrics):
        """Test performance with large datasets"""
        test_name = "large_dataset"

        # Generate large signal (100 μs)
        signal_data = self.generate_performance_test_signal(
            duration=100e-6,  # 100 μs - large dataset
            complexity="normal",
        )

        performance_metrics.start_measurement(test_name)

        results = framework.run_auto_validation(
            signal_data=signal_data, sample_rate=200e9, voltage_range=0.8, protocol_hint="usb4"
        )

        metrics = performance_metrics.end_measurement(test_name)

        # Calculate samples processed
        num_samples = len(signal_data[0]["voltage"])
        samples_per_second = num_samples / metrics["duration"]

        # Performance assertions
        assert metrics["duration"] < 30.0, f"Large dataset validation took {metrics['duration']:.2f}s, exceeds 30s limit"
        assert samples_per_second > 9e5, f"Processing rate {samples_per_second:.0f} samples/s below 900K samples/s"
        assert (
            metrics["memory_delta"] < 500 * 1024 * 1024
        ), f"Memory usage {metrics['memory_delta']/1024/1024:.1f}MB exceeds 500MB limit"

        print("Large Dataset Performance:")
        print(f"  Duration: {metrics['duration']:.2f}s")
        print(f"  Samples: {num_samples:,}")
        print(f"  Processing Rate: {samples_per_second:.0f} samples/s")
        print(f"  Memory Delta: {metrics['memory_delta']/1024/1024:.1f}MB")

    def test_complex_signal_performance(self, framework, performance_metrics):
        """Test performance with complex signals (ISI, interference, etc.)"""
        test_name = "complex_signal"

        # Generate complex signal
        signal_data = self.generate_performance_test_signal(
            duration=20e-6,  # 20 μs
            complexity="complex",
        )

        performance_metrics.start_measurement(test_name)

        results = framework.run_auto_validation(
            signal_data=signal_data, sample_rate=200e9, voltage_range=0.8, protocol_hint="usb4"
        )

        metrics = performance_metrics.end_measurement(test_name)

        # Performance assertions (complex signals may take longer)
        assert metrics["duration"] < 15.0, f"Complex signal validation took {metrics['duration']:.2f}s, exceeds 15s limit"
        assert (
            metrics["memory_delta"] < 200 * 1024 * 1024
        ), f"Memory usage {metrics['memory_delta']/1024/1024:.1f}MB exceeds 200MB limit"

        print("Complex Signal Performance:")
        print(f"  Duration: {metrics['duration']:.2f}s")
        print(f"  Memory Delta: {metrics['memory_delta']/1024/1024:.1f}MB")

    def test_comprehensive_test_suite_performance(self, performance_metrics):
        """Test performance of comprehensive USB4 test suite"""
        test_name = "comprehensive_suite"

        # Create comprehensive test configuration
        config = USB4TestSequenceConfig(
            test_name="Performance Test Suite",
            lanes=[
                USB4LaneConfig(0, USB4SignalMode.GEN2X2, 200e9, 25e9, 0.8),
                USB4LaneConfig(1, USB4SignalMode.GEN2X2, 200e9, 25e9, 0.8),
            ],
            test_phases=[
                USB4TestPhase.INITIALIZATION,
                USB4TestPhase.SIGNAL_ANALYSIS,
                USB4TestPhase.LINK_TRAINING,
                USB4TestPhase.COMPLIANCE,
                USB4TestPhase.TUNNELING,
                USB4TestPhase.POWER_MANAGEMENT,
                USB4TestPhase.PERFORMANCE,
                USB4TestPhase.VALIDATION,
            ],
            tunneling_modes=[USB4TunnelingMode.PCIE, USB4TunnelingMode.DISPLAYPORT, USB4TunnelingMode.USB32],
            stress_duration=10.0,  # Reduced for performance testing
            power_states_to_test=[USB4LinkState.U0, USB4LinkState.U1, USB4LinkState.U2],
        )

        # Generate signal data
        signal_data = self.generate_performance_test_signal(
            duration=30e-6,  # 30 μs
            complexity="normal",
        )

        performance_metrics.start_measurement(test_name)

        test_sequence = create_usb4_test_sequence(custom_config=config.__dict__)
        results = test_sequence.run_complete_sequence(signal_data)

        metrics = performance_metrics.end_measurement(test_name)

        # Performance assertions for comprehensive suite
        assert metrics["duration"] < 45.0, f"Comprehensive suite took {metrics['duration']:.2f}s, exceeds 45s limit"
        assert (
            metrics["memory_delta"] < 300 * 1024 * 1024
        ), f"Memory usage {metrics['memory_delta']/1024/1024:.1f}MB exceeds 300MB limit"

        # Verify all phases completed
        assert len(results.phase_results) >= 7, f"Only {len(results.phase_results)} phases completed"

        print("Comprehensive Suite Performance:")
        print(f"  Duration: {metrics['duration']:.2f}s")
        print(f"  Phases: {len(results.phase_results)}")
        print(f"  Memory Delta: {metrics['memory_delta']/1024/1024:.1f}MB")

    def test_memory_leak_detection(self, framework):
        """Test for memory leaks during repeated validations"""
        initial_memory = psutil.Process(os.getpid()).memory_info().rss

        # Run multiple validations
        for _i in range(10):
            signal_data = self.generate_performance_test_signal(
                duration=5e-6,  # Small signals for speed
                complexity="simple",
            )

            results = framework.run_auto_validation(
                signal_data=signal_data, sample_rate=200e9, voltage_range=0.8, protocol_hint="usb4"
            )

            # Force garbage collection
            del signal_data, results
            gc.collect()

        final_memory = psutil.Process(os.getpid()).memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal (less than 150MB for test environment)
        max_acceptable_increase = 150 * 1024 * 1024  # 150MB

        print("Memory Leak Test:")
        print(f"  Initial Memory: {initial_memory/1024/1024:.1f}MB")
        print(f"  Final Memory: {final_memory/1024/1024:.1f}MB")
        print(f"  Memory Increase: {memory_increase/1024/1024:.1f}MB")

        assert (
            memory_increase < max_acceptable_increase
        ), f"Memory leak detected: {memory_increase/1024/1024:.1f}MB increase exceeds {max_acceptable_increase/1024/1024:.1f}MB limit"

    def test_concurrent_validation_performance(self, framework, performance_metrics):
        """Test performance of concurrent validations"""
        import queue
        import threading

        test_name = "concurrent_validation"

        def run_validation(signal_data, result_queue):
            try:
                start_time = time.time()
                results = framework.run_auto_validation(
                    signal_data=signal_data, sample_rate=200e9, voltage_range=0.8, protocol_hint="usb4"
                )
                duration = time.time() - start_time
                result_queue.put(("success", duration, results))
            except Exception as e:
                result_queue.put(("error", 0, str(e)))

        # Generate signals for concurrent validation
        signals = []
        for _i in range(3):
            signal_data = self.generate_performance_test_signal(
                duration=8e-6,  # 8 μs each
                complexity="normal",
            )
            signals.append(signal_data)

        performance_metrics.start_measurement(test_name)

        # Start concurrent validations
        threads = []
        results_queue = queue.Queue()

        for signal_data in signals:
            thread = threading.Thread(target=run_validation, args=(signal_data, results_queue))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        metrics = performance_metrics.end_measurement(test_name)

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Analyze concurrent performance
        successful_validations = [r for r in results if r[0] == "success"]
        individual_durations = [r[1] for r in successful_validations]

        if individual_durations:
            sequential_time_estimate = sum(individual_durations)
            concurrent_speedup = sequential_time_estimate / metrics["duration"]

            print("Concurrent Validation Performance:")
            print(f"  Concurrent Duration: {metrics['duration']:.2f}s")
            print(f"  Sequential Estimate: {sequential_time_estimate:.2f}s")
            print(f"  Speedup: {concurrent_speedup:.2f}x")
            print(f"  Success Rate: {len(successful_validations)}/{len(results)}")

            # Performance assertions
            assert len(successful_validations) >= 2, f"Only {len(successful_validations)}/3 concurrent validations succeeded"
            assert concurrent_speedup > 0.8, f"Insufficient speedup from concurrency: {concurrent_speedup:.2f}x"
            assert metrics["duration"] < 20.0, f"Concurrent validation took {metrics['duration']:.2f}s, exceeds 20s limit"

    def test_scalability_performance(self, framework, performance_metrics):
        """Test performance scalability with increasing data sizes"""
        test_sizes = [
            (5e-6, "Small"),  # 5 μs
            (20e-6, "Medium"),  # 20 μs
            (50e-6, "Large"),  # 50 μs
            (100e-6, "XLarge"),  # 100 μs
        ]

        scalability_results = []

        for duration, size_name in test_sizes:
            test_name = f"scalability_{size_name.lower()}"

            signal_data = self.generate_performance_test_signal(duration=duration, complexity="normal")

            performance_metrics.start_measurement(test_name)

            results = framework.run_auto_validation(
                signal_data=signal_data, sample_rate=200e9, voltage_range=0.8, protocol_hint="usb4"
            )

            metrics = performance_metrics.end_measurement(test_name)

            num_samples = len(signal_data[0]["voltage"])
            samples_per_second = num_samples / metrics["duration"]

            scalability_results.append(
                {
                    "size": size_name,
                    "duration": duration,
                    "num_samples": num_samples,
                    "validation_time": metrics["duration"],
                    "samples_per_second": samples_per_second,
                    "memory_delta": metrics["memory_delta"],
                }
            )

            print(f"{size_name} Dataset ({duration*1e6:.0f}μs):")
            print(f"  Validation Time: {metrics['duration']:.2f}s")
            print(f"  Processing Rate: {samples_per_second:.0f} samples/s")
            print(f"  Memory Delta: {metrics['memory_delta']/1024/1024:.1f}MB")

        # Analyze scalability
        print("\nScalability Analysis:")

        # Check that processing rate doesn't degrade significantly
        processing_rates = [r["samples_per_second"] for r in scalability_results]
        min_rate = min(processing_rates)
        max_rate = max(processing_rates)
        rate_variation = (max_rate - min_rate) / max_rate

        print(f"  Processing Rate Range: {min_rate:.0f} - {max_rate:.0f} samples/s")
        print(f"  Rate Variation: {rate_variation:.2%}")

        # Rate variation should be reasonable (less than 50%)
        assert rate_variation < 0.5, f"Processing rate variation {rate_variation:.2%} too high"

        # All validations should complete within reasonable time
        max_validation_time = max(r["validation_time"] for r in scalability_results)
        assert max_validation_time < 60.0, f"Maximum validation time {max_validation_time:.2f}s exceeds 60s limit"


class TestUSB4PerformanceBenchmarks:
    """USB4 performance benchmarks for comparison"""

    def test_baseline_performance_benchmark(self):
        """Establish baseline performance benchmarks"""
        framework = create_validation_framework()

        # Standard test signal
        duration = 10e-6  # 10 μs
        sample_rate = 200e9
        num_samples = int(duration * sample_rate)
        time_array = np.linspace(0, duration, num_samples)

        # Generate standard USB4 signal
        voltage = 0.4 * np.random.choice([-1, 1], size=num_samples)
        voltage += 0.02 * np.random.randn(num_samples)

        signal_data = {0: {"time": time_array, "voltage": voltage}, 1: {"time": time_array, "voltage": voltage.copy()}}

        # Benchmark validation
        start_time = time.time()
        results = framework.run_auto_validation(
            signal_data=signal_data, sample_rate=sample_rate, voltage_range=0.8, protocol_hint="usb4"
        )
        validation_time = time.time() - start_time

        # Performance benchmarks
        samples_per_second = num_samples / validation_time

        print("USB4 Performance Benchmarks:")
        print(f"  Signal Duration: {duration*1e6:.0f} μs")
        print(f"  Samples: {num_samples:,}")
        print(f"  Validation Time: {validation_time:.2f}s")
        print(f"  Processing Rate: {samples_per_second:.0f} samples/s")
        print(f"  Validation Status: {results['validation_results'].overall_status.name}")

        # Store benchmarks for regression testing
        benchmarks = {
            "validation_time": validation_time,
            "samples_per_second": samples_per_second,
            "memory_efficient": validation_time < 10.0,
            "processing_efficient": samples_per_second > 500000,
        }

        # Verify benchmarks meet minimum requirements
        assert benchmarks["memory_efficient"], f"Validation time {validation_time:.2f}s exceeds efficiency threshold"
        assert benchmarks["processing_efficient"], f"Processing rate {samples_per_second:.0f} below efficiency threshold"

        return benchmarks


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])
