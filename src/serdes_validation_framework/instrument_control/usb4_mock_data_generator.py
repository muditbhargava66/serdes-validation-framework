"""
USB4 Mock Data Generator Module

This module provides realistic USB4 signal simulation for testing without physical hardware.
Generates authentic USB4 waveforms with proper characteristics for validation testing.

Features:
- Realistic USB4 signal generation
- Configurable noise and impairments
- Link training sequence simulation
- Multi-lane signal generation
- Thunderbolt 4 pattern support
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Union

import numpy as np
import numpy.typing as npt

from ..protocols.usb4.constants import USB4SignalMode, USB4SignalSpecs, USB4Specs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USB4MockSignalType(Enum):
    """USB4 mock signal types"""

    IDEAL = auto()
    REALISTIC = auto()
    IMPAIRED = auto()
    STRESSED = auto()


class USB4MockNoiseType(Enum):
    """USB4 mock noise types"""

    THERMAL = auto()
    SHOT = auto()
    FLICKER = auto()
    CROSSTALK = auto()
    POWER_SUPPLY = auto()


@dataclass
class USB4MockConfig:
    """USB4 mock data generator configuration"""

    signal_mode: USB4SignalMode
    signal_type: USB4MockSignalType
    sample_rate: float
    bit_rate: float
    amplitude: float
    noise_level: float
    jitter_rms: float
    enable_ssc: bool = True
    dual_lane: bool = True
    lane_skew: float = 0.0  # seconds

    def __post_init__(self) -> None:
        """Validate mock configuration"""
        assert isinstance(self.signal_mode, USB4SignalMode), f"Signal mode must be USB4SignalMode, got {type(self.signal_mode)}"
        assert isinstance(
            self.signal_type, USB4MockSignalType
        ), f"Signal type must be USB4MockSignalType, got {type(self.signal_type)}"
        assert isinstance(self.sample_rate, float), f"Sample rate must be float, got {type(self.sample_rate)}"
        assert isinstance(self.bit_rate, float), f"Bit rate must be float, got {type(self.bit_rate)}"
        assert isinstance(self.amplitude, float), f"Amplitude must be float, got {type(self.amplitude)}"
        assert isinstance(self.noise_level, float), f"Noise level must be float, got {type(self.noise_level)}"
        assert isinstance(self.jitter_rms, float), f"Jitter RMS must be float, got {type(self.jitter_rms)}"
        assert isinstance(self.lane_skew, float), f"Lane skew must be float, got {type(self.lane_skew)}"

        assert self.sample_rate > 0, f"Sample rate must be positive, got {self.sample_rate}"
        assert self.bit_rate > 0, f"Bit rate must be positive, got {self.bit_rate}"
        assert self.amplitude > 0, f"Amplitude must be positive, got {self.amplitude}"
        assert self.noise_level >= 0, f"Noise level must be non-negative, got {self.noise_level}"
        assert self.jitter_rms >= 0, f"Jitter RMS must be non-negative, got {self.jitter_rms}"


@dataclass
class USB4MockSignalData:
    """USB4 mock signal data result"""

    lane0_voltage: npt.NDArray[np.float64]
    lane1_voltage: Optional[npt.NDArray[np.float64]]
    time_base: npt.NDArray[np.float64]
    bit_pattern: npt.NDArray[np.int32]
    signal_quality: Dict[str, float]
    metadata: Dict[str, Union[str, float, bool]]

    def __post_init__(self) -> None:
        """Validate mock signal data"""
        assert isinstance(self.lane0_voltage, np.ndarray), f"Lane 0 voltage must be numpy array, got {type(self.lane0_voltage)}"
        assert isinstance(self.time_base, np.ndarray), f"Time base must be numpy array, got {type(self.time_base)}"
        assert isinstance(self.bit_pattern, np.ndarray), f"Bit pattern must be numpy array, got {type(self.bit_pattern)}"
        assert isinstance(self.signal_quality, dict), f"Signal quality must be dict, got {type(self.signal_quality)}"
        assert isinstance(self.metadata, dict), f"Metadata must be dict, got {type(self.metadata)}"

        assert len(self.lane0_voltage) > 0, "Lane 0 voltage cannot be empty"
        assert len(self.time_base) == len(self.lane0_voltage), "Time base and voltage length mismatch"

        if self.lane1_voltage is not None:
            assert isinstance(
                self.lane1_voltage, np.ndarray
            ), f"Lane 1 voltage must be numpy array, got {type(self.lane1_voltage)}"
            assert len(self.lane1_voltage) == len(self.lane0_voltage), "Lane voltage length mismatch"


class USB4MockDataGenerator:
    """Mock data generator for USB4 signal simulation"""

    def __init__(self, config: USB4MockConfig) -> None:
        """
        Initialize USB4 mock data generator

        Args:
            config: Mock generator configuration

        Raises:
            ValueError: If initialization fails
        """
        assert isinstance(config, USB4MockConfig), f"Config must be USB4MockConfig, got {type(config)}"

        self.config = config
        self.usb4_specs = USB4Specs()
        self.signal_specs = USB4SignalSpecs()

        # Signal generation state
        self.rng = np.random.RandomState(42)  # Fixed seed for reproducibility

        logger.info(f"USB4 mock data generator initialized for {config.signal_mode.name}")

    def generate_usb4_signal(self, duration: float, pattern_type: str = "PRBS31") -> USB4MockSignalData:
        """
        Generate USB4 mock signal data

        Args:
            duration: Signal duration in seconds
            pattern_type: Bit pattern type ("PRBS31", "PRBS15", "TS1", "TS2", "IDLE")

        Returns:
            Mock signal data

        Raises:
            ValueError: If generation fails
        """
        assert isinstance(duration, float), f"Duration must be float, got {type(duration)}"
        assert isinstance(pattern_type, str), f"Pattern type must be string, got {type(pattern_type)}"
        assert duration > 0, f"Duration must be positive, got {duration}"

        try:
            # Calculate signal parameters
            num_samples = int(self.config.sample_rate * duration)
            time_base = np.linspace(0, duration, num_samples, dtype=np.float64)

            # Generate bit pattern
            bit_pattern = self._generate_bit_pattern(pattern_type, duration)

            # Generate lane 0 signal
            lane0_voltage = self._generate_lane_signal(bit_pattern, time_base, 0)

            # Generate lane 1 signal if dual-lane
            lane1_voltage = None
            if self.config.dual_lane:
                lane1_voltage = self._generate_lane_signal(bit_pattern, time_base, 1)

            # Calculate signal quality metrics
            signal_quality = self._calculate_signal_quality(lane0_voltage, lane1_voltage, time_base)

            # Create metadata
            metadata = {
                "signal_mode": self.config.signal_mode.name,
                "signal_type": self.config.signal_type.name,
                "pattern_type": pattern_type,
                "duration": duration,
                "sample_rate": self.config.sample_rate,
                "bit_rate": self.config.bit_rate,
                "amplitude": self.config.amplitude,
                "noise_level": self.config.noise_level,
                "jitter_rms": self.config.jitter_rms,
                "enable_ssc": self.config.enable_ssc,
                "dual_lane": self.config.dual_lane,
                "lane_skew": self.config.lane_skew,
                "generation_time": time.time(),
            }

            return USB4MockSignalData(
                lane0_voltage=lane0_voltage,
                lane1_voltage=lane1_voltage,
                time_base=time_base,
                bit_pattern=bit_pattern,
                signal_quality=signal_quality,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"USB4 signal generation failed: {e}")
            raise ValueError(f"Signal generation failed: {e}")

    def _generate_bit_pattern(self, pattern_type: str, duration: float) -> npt.NDArray[np.int32]:
        """Generate bit pattern for specified type"""
        try:
            num_bits = int(self.config.bit_rate * duration)

            if pattern_type == "PRBS31":
                return self._generate_prbs_pattern(31, num_bits)
            elif pattern_type == "PRBS15":
                return self._generate_prbs_pattern(15, num_bits)
            elif pattern_type == "PRBS7":
                return self._generate_prbs_pattern(7, num_bits)
            elif pattern_type == "TS1":
                return self._generate_training_sequence_1(num_bits)
            elif pattern_type == "TS2":
                return self._generate_training_sequence_2(num_bits)
            elif pattern_type == "IDLE":
                return self._generate_idle_pattern(num_bits)
            else:
                # Default to PRBS31
                return self._generate_prbs_pattern(31, num_bits)

        except Exception as e:
            raise ValueError(f"Bit pattern generation failed: {e}")

    def _generate_prbs_pattern(self, order: int, length: int) -> npt.NDArray[np.int32]:
        """Generate PRBS pattern of specified order"""
        try:
            # PRBS polynomial taps
            if order == 7:
                taps = [7, 6]
            elif order == 15:
                taps = [15, 14]
            elif order == 31:
                taps = [31, 28]
            else:
                raise ValueError(f"Unsupported PRBS order: {order}")

            # Initialize shift register
            shift_register = np.ones(order, dtype=int)
            pattern = np.zeros(length, dtype=np.int32)

            for i in range(length):
                # Output current bit
                pattern[i] = shift_register[-1]

                # Calculate feedback
                feedback = 0
                for tap in taps:
                    feedback ^= shift_register[tap - 1]

                # Shift register
                shift_register[1:] = shift_register[:-1]
                shift_register[0] = feedback

            return pattern

        except Exception as e:
            raise ValueError(f"PRBS pattern generation failed: {e}")

    def _generate_training_sequence_1(self, length: int) -> npt.NDArray[np.int32]:
        """Generate USB4 Training Sequence 1"""
        try:
            # TS1 pattern: alternating 1010... for link training
            base_pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
            repeats = length // len(base_pattern) + 1
            pattern = np.tile(base_pattern, repeats)[:length]
            return pattern

        except Exception as e:
            raise ValueError(f"TS1 pattern generation failed: {e}")

    def _generate_training_sequence_2(self, length: int) -> npt.NDArray[np.int32]:
        """Generate USB4 Training Sequence 2"""
        try:
            # TS2 pattern: different pattern for link training
            base_pattern = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int32)
            repeats = length // len(base_pattern) + 1
            pattern = np.tile(base_pattern, repeats)[:length]
            return pattern

        except Exception as e:
            raise ValueError(f"TS2 pattern generation failed: {e}")

    def _generate_idle_pattern(self, length: int) -> npt.NDArray[np.int32]:
        """Generate USB4 idle pattern"""
        try:
            # Idle pattern: all zeros (electrical idle)
            return np.zeros(length, dtype=np.int32)

        except Exception as e:
            raise ValueError(f"Idle pattern generation failed: {e}")

    def _generate_lane_signal(
        self, bit_pattern: npt.NDArray[np.int32], time_base: npt.NDArray[np.float64], lane_id: int
    ) -> npt.NDArray[np.float64]:
        """Generate analog signal for a specific lane"""
        try:
            # Convert bits to analog levels
            analog_signal = self._bits_to_analog(bit_pattern, time_base)

            # Apply signal impairments based on signal type
            if self.config.signal_type == USB4MockSignalType.REALISTIC:
                analog_signal = self._apply_realistic_impairments(analog_signal, time_base)
            elif self.config.signal_type == USB4MockSignalType.IMPAIRED:
                analog_signal = self._apply_impairments(analog_signal, time_base)
            elif self.config.signal_type == USB4MockSignalType.STRESSED:
                analog_signal = self._apply_stress_impairments(analog_signal, time_base)

            # Apply lane-specific effects
            if lane_id == 1 and self.config.lane_skew != 0:
                analog_signal = self._apply_lane_skew(analog_signal, time_base, self.config.lane_skew)

            # Apply SSC if enabled
            if self.config.enable_ssc:
                analog_signal = self._apply_ssc_modulation(analog_signal, time_base)

            return analog_signal

        except Exception as e:
            raise ValueError(f"Lane signal generation failed: {e}")

    def _bits_to_analog(self, bit_pattern: npt.NDArray[np.int32], time_base: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert bit pattern to analog signal"""
        try:
            # Calculate samples per bit
            bit_period = 1.0 / self.config.bit_rate
            samples_per_bit = int(self.config.sample_rate * bit_period)

            # Upsample bit pattern to match sample rate
            upsampled_bits = np.repeat(bit_pattern, samples_per_bit)

            # Truncate or pad to match time base length
            if len(upsampled_bits) > len(time_base):
                upsampled_bits = upsampled_bits[: len(time_base)]
            elif len(upsampled_bits) < len(time_base):
                padding = len(time_base) - len(upsampled_bits)
                upsampled_bits = np.pad(upsampled_bits, (0, padding), mode="constant")

            # Convert to NRZ levels (0 -> -amplitude, 1 -> +amplitude)
            analog_signal = (2 * upsampled_bits - 1) * self.config.amplitude

            # Apply rise/fall time filtering
            analog_signal = self._apply_bandwidth_limiting(analog_signal)

            return analog_signal.astype(np.float64)

        except Exception as e:
            raise ValueError(f"Bits to analog conversion failed: {e}")

    def _apply_bandwidth_limiting(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply bandwidth limiting to simulate finite rise/fall times"""
        try:
            # Create low-pass filter for bandwidth limiting
            # USB4 has ~35 ps rise/fall time, corresponding to ~10 GHz bandwidth
            cutoff_freq = 10e9  # 10 GHz
            nyquist_freq = self.config.sample_rate / 2
            normalized_cutoff = cutoff_freq / nyquist_freq

            # Simple first-order filter approximation
            alpha = np.exp(-2 * np.pi * normalized_cutoff)
            filtered_signal = np.zeros_like(signal)
            filtered_signal[0] = signal[0]

            for i in range(1, len(signal)):
                filtered_signal[i] = alpha * filtered_signal[i - 1] + (1 - alpha) * signal[i]

            return filtered_signal

        except Exception as e:
            logger.warning(f"Bandwidth limiting failed: {e}")
            return signal

    def _apply_realistic_impairments(
        self, signal: npt.NDArray[np.float64], time_base: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Apply realistic signal impairments"""
        try:
            impaired_signal = signal.copy()

            # Add thermal noise
            noise_power = self.config.noise_level * self.config.amplitude
            thermal_noise = self.rng.normal(0, noise_power, len(signal))
            impaired_signal += thermal_noise

            # Add jitter
            if self.config.jitter_rms > 0:
                impaired_signal = self._apply_jitter(impaired_signal, time_base)

            # Add small amount of ISI (inter-symbol interference)
            impaired_signal = self._apply_isi(impaired_signal)

            return impaired_signal

        except Exception as e:
            logger.warning(f"Realistic impairments failed: {e}")
            return signal

    def _apply_impairments(self, signal: npt.NDArray[np.float64], time_base: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply moderate signal impairments"""
        try:
            impaired_signal = signal.copy()

            # Higher noise level
            noise_power = self.config.noise_level * self.config.amplitude * 2
            noise = self.rng.normal(0, noise_power, len(signal))
            impaired_signal += noise

            # More jitter
            if self.config.jitter_rms > 0:
                impaired_signal = self._apply_jitter(impaired_signal, time_base, scale=2.0)

            # Add crosstalk
            crosstalk = self._generate_crosstalk(len(signal))
            impaired_signal += crosstalk * self.config.amplitude * 0.1

            # Add ISI
            impaired_signal = self._apply_isi(impaired_signal, severity=2.0)

            return impaired_signal

        except Exception as e:
            logger.warning(f"Impairments failed: {e}")
            return signal

    def _apply_stress_impairments(
        self, signal: npt.NDArray[np.float64], time_base: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Apply severe stress test impairments"""
        try:
            impaired_signal = signal.copy()

            # High noise level
            noise_power = self.config.noise_level * self.config.amplitude * 5
            noise = self.rng.normal(0, noise_power, len(signal))
            impaired_signal += noise

            # Severe jitter
            if self.config.jitter_rms > 0:
                impaired_signal = self._apply_jitter(impaired_signal, time_base, scale=5.0)

            # Strong crosstalk
            crosstalk = self._generate_crosstalk(len(signal))
            impaired_signal += crosstalk * self.config.amplitude * 0.3

            # Severe ISI
            impaired_signal = self._apply_isi(impaired_signal, severity=5.0)

            # Add power supply noise
            ps_noise = self._generate_power_supply_noise(len(signal), time_base)
            impaired_signal += ps_noise * self.config.amplitude * 0.2

            # Add amplitude variation
            amplitude_variation = 1 + 0.1 * self.rng.normal(0, 1, len(signal))
            impaired_signal *= amplitude_variation

            return impaired_signal

        except Exception as e:
            logger.warning(f"Stress impairments failed: {e}")
            return signal

    def _apply_jitter(
        self, signal: npt.NDArray[np.float64], time_base: npt.NDArray[np.float64], scale: float = 1.0
    ) -> npt.NDArray[np.float64]:
        """Apply timing jitter to signal"""
        try:
            # Generate jitter sequence
            jitter_std = self.config.jitter_rms * scale
            jitter_samples = self.rng.normal(0, jitter_std * self.config.sample_rate, len(signal))

            # Apply jitter by shifting samples
            jittered_signal = np.zeros_like(signal)
            for i in range(len(signal)):
                jitter_offset = int(jitter_samples[i])
                source_idx = max(0, min(len(signal) - 1, i + jitter_offset))
                jittered_signal[i] = signal[source_idx]

            return jittered_signal

        except Exception as e:
            logger.warning(f"Jitter application failed: {e}")
            return signal

    def _apply_isi(self, signal: npt.NDArray[np.float64], severity: float = 1.0) -> npt.NDArray[np.float64]:
        """Apply inter-symbol interference"""
        try:
            # Simple ISI model: current symbol affected by previous symbols
            isi_taps = np.array([0.05, 0.02, 0.01]) * severity
            isi_signal = signal.copy()

            for i, tap in enumerate(isi_taps):
                if i + 1 < len(signal):
                    isi_signal[i + 1 :] += tap * signal[: -i - 1]

            return isi_signal

        except Exception as e:
            logger.warning(f"ISI application failed: {e}")
            return signal

    def _generate_crosstalk(self, length: int) -> npt.NDArray[np.float64]:
        """Generate crosstalk interference"""
        try:
            # Generate interfering signal (different frequency)
            t = np.arange(length) / self.config.sample_rate
            interferer_freq = self.config.bit_rate * 0.7  # Different frequency
            crosstalk = np.sin(2 * np.pi * interferer_freq * t)

            # Add some randomness
            crosstalk += 0.3 * self.rng.normal(0, 1, length)

            return crosstalk

        except Exception as e:
            logger.warning(f"Crosstalk generation failed: {e}")
            return np.zeros(length)

    def _generate_power_supply_noise(self, length: int, time_base: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Generate power supply noise"""
        try:
            # Low frequency power supply ripple
            ripple_freq = 100e3  # 100 kHz switching frequency
            ps_noise = 0.5 * np.sin(2 * np.pi * ripple_freq * time_base)

            # Add some random variation
            ps_noise += 0.2 * self.rng.normal(0, 1, length)

            return ps_noise

        except Exception as e:
            logger.warning(f"Power supply noise generation failed: {e}")
            return np.zeros(length)

    def _apply_lane_skew(
        self, signal: npt.NDArray[np.float64], time_base: npt.NDArray[np.float64], skew: float
    ) -> npt.NDArray[np.float64]:
        """Apply lane-to-lane skew"""
        try:
            # Convert skew to sample delay
            skew_samples = int(skew * self.config.sample_rate)

            if skew_samples > 0:
                # Positive skew: delay signal
                skewed_signal = np.zeros_like(signal)
                skewed_signal[skew_samples:] = signal[:-skew_samples]
            elif skew_samples < 0:
                # Negative skew: advance signal
                skewed_signal = np.zeros_like(signal)
                skewed_signal[:skew_samples] = signal[-skew_samples:]
            else:
                skewed_signal = signal

            return skewed_signal

        except Exception as e:
            logger.warning(f"Lane skew application failed: {e}")
            return signal

    def _apply_ssc_modulation(
        self, signal: npt.NDArray[np.float64], time_base: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Apply spread spectrum clocking modulation"""
        try:
            # USB4 SSC: 31.5 kHz modulation, 0.5% deviation
            ssc_freq = 31.5e3
            ssc_deviation = 0.005  # 0.5%

            # Generate SSC modulation
            ssc_modulation = ssc_deviation * np.sin(2 * np.pi * ssc_freq * time_base)

            # Apply frequency modulation (simplified)
            # In reality, this would require proper FM modulation
            phase_modulation = np.cumsum(ssc_modulation) * 2 * np.pi / self.config.sample_rate
            ssc_signal = signal * (1 + 0.1 * np.sin(phase_modulation))

            return ssc_signal

        except Exception as e:
            logger.warning(f"SSC modulation failed: {e}")
            return signal

    def _calculate_signal_quality(
        self,
        lane0_voltage: npt.NDArray[np.float64],
        lane1_voltage: Optional[npt.NDArray[np.float64]],
        time_base: npt.NDArray[np.float64],
    ) -> Dict[str, float]:
        """Calculate signal quality metrics"""
        try:
            quality = {}

            # Lane 0 metrics
            quality["lane0_rms"] = float(np.sqrt(np.mean(lane0_voltage**2)))
            quality["lane0_peak"] = float(np.max(np.abs(lane0_voltage)))
            quality["lane0_snr_db"] = self._calculate_snr(lane0_voltage)
            quality["lane0_thd"] = self._calculate_thd(lane0_voltage)

            # Lane 1 metrics if available
            if lane1_voltage is not None:
                quality["lane1_rms"] = float(np.sqrt(np.mean(lane1_voltage**2)))
                quality["lane1_peak"] = float(np.max(np.abs(lane1_voltage)))
                quality["lane1_snr_db"] = self._calculate_snr(lane1_voltage)
                quality["lane1_thd"] = self._calculate_thd(lane1_voltage)

                # Lane-to-lane metrics
                quality["lane_skew_ps"] = self._calculate_lane_skew_metric(lane0_voltage, lane1_voltage) * 1e12
                quality["lane_amplitude_mismatch"] = self._calculate_amplitude_mismatch(lane0_voltage, lane1_voltage)

            # Overall signal quality score
            quality["overall_quality_score"] = self._calculate_overall_quality_score(quality)

            return quality

        except Exception as e:
            logger.warning(f"Signal quality calculation failed: {e}")
            return {"error": 1.0}

    def _calculate_snr(self, signal: npt.NDArray[np.float64]) -> float:
        """Calculate signal-to-noise ratio"""
        try:
            # Estimate signal power (assume signal is mostly signal)
            signal_power = np.var(signal)

            # Estimate noise power (high frequency components)
            # Simple high-pass filter to isolate noise
            diff_signal = np.diff(signal)
            noise_power = np.var(diff_signal) / 2  # Approximate noise power

            if noise_power > 0:
                snr_linear = signal_power / noise_power
                snr_db = 10 * np.log10(snr_linear)
            else:
                snr_db = 60.0  # Very high SNR

            return float(snr_db)

        except Exception as e:
            logger.warning(f"SNR calculation failed: {e}")
            return 20.0  # Default reasonable value

    def _calculate_thd(self, signal: npt.NDArray[np.float64]) -> float:
        """Calculate total harmonic distortion"""
        try:
            # Simple THD estimation using FFT
            fft_signal = np.fft.fft(signal)
            power_spectrum = np.abs(fft_signal) ** 2

            # Find fundamental frequency (largest peak)
            fundamental_idx = np.argmax(power_spectrum[1 : len(power_spectrum) // 2]) + 1
            fundamental_power = power_spectrum[fundamental_idx]

            # Calculate harmonic power (simplified)
            harmonic_power = np.sum(power_spectrum) - fundamental_power

            if fundamental_power > 0:
                thd = np.sqrt(harmonic_power / fundamental_power)
            else:
                thd = 0.0

            return float(thd)

        except Exception as e:
            logger.warning(f"THD calculation failed: {e}")
            return 0.05  # Default reasonable value

    def _calculate_lane_skew_metric(self, lane0: npt.NDArray[np.float64], lane1: npt.NDArray[np.float64]) -> float:
        """Calculate lane-to-lane skew"""
        try:
            # Cross-correlation to find delay
            correlation = np.correlate(lane0[:1000], lane1[:1000], mode="full")
            delay_samples = np.argmax(correlation) - len(lane1[:1000]) + 1

            # Convert to time
            skew_time = delay_samples / self.config.sample_rate

            return float(skew_time)

        except Exception as e:
            logger.warning(f"Lane skew calculation failed: {e}")
            return 0.0

    def _calculate_amplitude_mismatch(self, lane0: npt.NDArray[np.float64], lane1: npt.NDArray[np.float64]) -> float:
        """Calculate amplitude mismatch between lanes"""
        try:
            rms0 = np.sqrt(np.mean(lane0**2))
            rms1 = np.sqrt(np.mean(lane1**2))

            if rms0 > 0:
                mismatch = abs(rms1 - rms0) / rms0
            else:
                mismatch = 0.0

            return float(mismatch)

        except Exception as e:
            logger.warning(f"Amplitude mismatch calculation failed: {e}")
            return 0.0

    def _calculate_overall_quality_score(self, quality_metrics: Dict[str, float]) -> float:
        """Calculate overall signal quality score"""
        try:
            score = 100.0  # Start with perfect score

            # Penalize based on SNR
            snr = quality_metrics.get("lane0_snr_db", 30.0)
            if snr < 20:
                score -= (20 - snr) * 2

            # Penalize based on THD
            thd = quality_metrics.get("lane0_thd", 0.05)
            if thd > 0.1:
                score -= (thd - 0.1) * 100

            # Penalize based on lane skew (if dual lane)
            if "lane_skew_ps" in quality_metrics:
                skew_ps = quality_metrics["lane_skew_ps"]
                if skew_ps > 20:  # USB4 limit
                    score -= (skew_ps - 20) * 0.5

            # Ensure score is between 0 and 100
            score = max(0.0, min(100.0, score))

            return float(score)

        except Exception as e:
            logger.warning(f"Overall quality score calculation failed: {e}")
            return 50.0  # Default moderate score


__all__ = [
    "USB4MockSignalType",
    "USB4MockNoiseType",
    "USB4MockConfig",
    "USB4MockSignalData",
    "USB4MockDataGenerator",
]
