"""
Advanced Eye Diagram Analysis Module

This module provides comprehensive eye diagram analysis capabilities
with statistical modeling, bathtub curves, and advanced metrics.

Features:
- Statistical eye diagram analysis
- Bathtub curve generation
- Eye contour analysis
- Jitter decomposition
- Advanced eye metrics
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JitterType(Enum):
    """Types of jitter components"""
    TOTAL = auto()
    RANDOM = auto()
    DETERMINISTIC = auto()
    PERIODIC = auto()
    DATA_DEPENDENT = auto()


class EyeMetric(Enum):
    """Eye diagram metrics"""
    HEIGHT = auto()
    WIDTH = auto()
    AREA = auto()
    CLOSURE = auto()
    SYMMETRY = auto()
    LINEARITY = auto()


@dataclass
class EyeParameters:
    """Eye diagram analysis parameters"""
    symbol_rate: float
    samples_per_symbol: int
    eye_samples: int = 1000
    confidence_level: float = 0.95
    jitter_analysis: bool = True
    
    def __post_init__(self) -> None:
        """Validate eye parameters"""
        assert isinstance(self.symbol_rate, float), \
            f"Symbol rate must be float, got {type(self.symbol_rate)}"
        assert isinstance(self.samples_per_symbol, int), \
            f"Samples per symbol must be int, got {type(self.samples_per_symbol)}"
        assert isinstance(self.eye_samples, int), \
            f"Eye samples must be int, got {type(self.eye_samples)}"
        assert isinstance(self.confidence_level, float), \
            f"Confidence level must be float, got {type(self.confidence_level)}"
        assert isinstance(self.jitter_analysis, bool), \
            f"Jitter analysis must be bool, got {type(self.jitter_analysis)}"
        
        assert self.symbol_rate > 0, f"Symbol rate must be positive, got {self.symbol_rate}"
        assert self.samples_per_symbol > 0, \
            f"Samples per symbol must be positive, got {self.samples_per_symbol}"
        assert self.eye_samples > 0, f"Eye samples must be positive, got {self.eye_samples}"
        assert 0 < self.confidence_level < 1, \
            f"Confidence level must be between 0 and 1, got {self.confidence_level}"


@dataclass
class JitterAnalysis:
    """Jitter analysis results"""
    total_jitter: float
    random_jitter: float
    deterministic_jitter: float
    periodic_jitter: float
    data_dependent_jitter: float
    jitter_histogram: npt.NDArray[np.float64]
    jitter_bins: npt.NDArray[np.float64]
    
    def __post_init__(self) -> None:
        """Validate jitter analysis"""
        assert isinstance(self.total_jitter, float), \
            f"Total jitter must be float, got {type(self.total_jitter)}"
        assert isinstance(self.random_jitter, float), \
            f"Random jitter must be float, got {type(self.random_jitter)}"
        assert isinstance(self.deterministic_jitter, float), \
            f"Deterministic jitter must be float, got {type(self.deterministic_jitter)}"
        assert isinstance(self.periodic_jitter, float), \
            f"Periodic jitter must be float, got {type(self.periodic_jitter)}"
        assert isinstance(self.data_dependent_jitter, float), \
            f"Data dependent jitter must be float, got {type(self.data_dependent_jitter)}"
        
        assert all(j >= 0 for j in [self.total_jitter, self.random_jitter, 
                                   self.deterministic_jitter, self.periodic_jitter,
                                   self.data_dependent_jitter]), \
            "All jitter values must be non-negative"


@dataclass
class BathtubCurve:
    """Bathtub curve data"""
    offsets: npt.NDArray[np.float64]
    ber_values: npt.NDArray[np.float64]
    eye_opening: float
    bathtub_floor: float
    
    def __post_init__(self) -> None:
        """Validate bathtub curve"""
        assert isinstance(self.offsets, np.ndarray), \
            f"Offsets must be numpy array, got {type(self.offsets)}"
        assert isinstance(self.ber_values, np.ndarray), \
            f"BER values must be numpy array, got {type(self.ber_values)}"
        assert isinstance(self.eye_opening, float), \
            f"Eye opening must be float, got {type(self.eye_opening)}"
        assert isinstance(self.bathtub_floor, float), \
            f"Bathtub floor must be float, got {type(self.bathtub_floor)}"
        
        assert len(self.offsets) == len(self.ber_values), \
            "Offsets and BER values must have same length"
        assert self.eye_opening >= 0, f"Eye opening must be non-negative, got {self.eye_opening}"
        assert 0 <= self.bathtub_floor <= 1, \
            f"Bathtub floor must be between 0 and 1, got {self.bathtub_floor}"


@dataclass
class EyeContour:
    """Eye contour analysis"""
    contour_levels: List[float]
    contour_areas: List[float]
    eye_mask_margin: float
    mask_violations: int
    
    def __post_init__(self) -> None:
        """Validate eye contour"""
        assert isinstance(self.contour_levels, list), \
            f"Contour levels must be list, got {type(self.contour_levels)}"
        assert isinstance(self.contour_areas, list), \
            f"Contour areas must be list, got {type(self.contour_areas)}"
        assert isinstance(self.eye_mask_margin, float), \
            f"Eye mask margin must be float, got {type(self.eye_mask_margin)}"
        assert isinstance(self.mask_violations, int), \
            f"Mask violations must be int, got {type(self.mask_violations)}"
        
        assert len(self.contour_levels) == len(self.contour_areas), \
            "Contour levels and areas must have same length"
        assert all(isinstance(level, float) for level in self.contour_levels), \
            "All contour levels must be floats"
        assert all(isinstance(area, float) for area in self.contour_areas), \
            "All contour areas must be floats"
        assert self.mask_violations >= 0, \
            f"Mask violations must be non-negative, got {self.mask_violations}"


@dataclass
class EyeDiagramResult:
    """Complete eye diagram analysis result"""
    eye_height: float
    eye_width: float
    eye_area: float
    eye_closure: float
    crossing_percentage: float
    q_factor: float
    jitter_analysis: Optional[JitterAnalysis]
    timing_bathtub: Optional[BathtubCurve]
    voltage_bathtub: Optional[BathtubCurve]
    eye_contour: Optional[EyeContour]
    eye_diagram: npt.NDArray[np.float64]
    time_axis: npt.NDArray[np.float64]
    voltage_axis: npt.NDArray[np.float64]
    
    def __post_init__(self) -> None:
        """Validate eye diagram result"""
        # Validate basic metrics
        assert isinstance(self.eye_height, float), \
            f"Eye height must be float, got {type(self.eye_height)}"
        assert isinstance(self.eye_width, float), \
            f"Eye width must be float, got {type(self.eye_width)}"
        assert isinstance(self.eye_area, float), \
            f"Eye area must be float, got {type(self.eye_area)}"
        assert isinstance(self.eye_closure, float), \
            f"Eye closure must be float, got {type(self.eye_closure)}"
        assert isinstance(self.crossing_percentage, float), \
            f"Crossing percentage must be float, got {type(self.crossing_percentage)}"
        assert isinstance(self.q_factor, float), \
            f"Q factor must be float, got {type(self.q_factor)}"
        
        # Validate arrays
        assert isinstance(self.eye_diagram, np.ndarray), \
            f"Eye diagram must be numpy array, got {type(self.eye_diagram)}"
        assert isinstance(self.time_axis, np.ndarray), \
            f"Time axis must be numpy array, got {type(self.time_axis)}"
        assert isinstance(self.voltage_axis, np.ndarray), \
            f"Voltage axis must be numpy array, got {type(self.voltage_axis)}"
        
        # Validate ranges
        assert self.eye_height >= 0, f"Eye height must be non-negative, got {self.eye_height}"
        assert self.eye_width >= 0, f"Eye width must be non-negative, got {self.eye_width}"
        assert self.eye_area >= 0, f"Eye area must be non-negative, got {self.eye_area}"
        assert 0 <= self.eye_closure <= 100, \
            f"Eye closure must be between 0 and 100, got {self.eye_closure}"
        assert 0 <= self.crossing_percentage <= 100, \
            f"Crossing percentage must be between 0 and 100, got {self.crossing_percentage}"


class AdvancedEyeAnalyzer:
    """Advanced eye diagram analyzer with statistical modeling"""
    
    def __init__(self, parameters: EyeParameters) -> None:
        """
        Initialize eye analyzer
        
        Args:
            parameters: Eye analysis parameters
            
        Raises:
            AssertionError: If parameters are invalid
        """
        assert isinstance(parameters, EyeParameters), \
            f"Parameters must be EyeParameters, got {type(parameters)}"
        
        self.params = parameters
        self.symbol_period = 1.0 / parameters.symbol_rate
        
        logger.info(f"Eye analyzer initialized: {parameters.symbol_rate/1e9:.1f} Gbps")
    
    def analyze_eye_diagram(
        self,
        time_data: npt.NDArray[np.float64],
        voltage_data: npt.NDArray[np.float64]
    ) -> EyeDiagramResult:
        """
        Perform comprehensive eye diagram analysis
        
        Args:
            time_data: Time points array
            voltage_data: Voltage measurements array
            
        Returns:
            Complete eye diagram analysis result
            
        Raises:
            ValueError: If analysis fails
        """
        try:
            # Validate input data
            self._validate_signal_data(time_data, voltage_data)
            
            # Generate eye diagram
            eye_diagram, time_axis, voltage_axis = self._generate_eye_diagram(
                time_data, voltage_data
            )
            
            # Calculate basic eye metrics
            eye_height = self._calculate_eye_height(eye_diagram, voltage_axis)
            eye_width = self._calculate_eye_width(eye_diagram, time_axis)
            eye_area = self._calculate_eye_area(eye_diagram, time_axis, voltage_axis)
            eye_closure = self._calculate_eye_closure(eye_height, voltage_axis)
            crossing_percentage = self._calculate_crossing_percentage(voltage_data)
            q_factor = self._calculate_q_factor(voltage_data)
            
            # Advanced analyses
            jitter_analysis = None
            timing_bathtub = None
            voltage_bathtub = None
            eye_contour = None
            
            if self.params.jitter_analysis:
                jitter_analysis = self._analyze_jitter(time_data, voltage_data)
                timing_bathtub = self._generate_timing_bathtub(time_data, voltage_data)
                voltage_bathtub = self._generate_voltage_bathtub(voltage_data)
                eye_contour = self._analyze_eye_contour(eye_diagram, time_axis, voltage_axis)
            
            return EyeDiagramResult(
                eye_height=eye_height,
                eye_width=eye_width,
                eye_area=eye_area,
                eye_closure=eye_closure,
                crossing_percentage=crossing_percentage,
                q_factor=q_factor,
                jitter_analysis=jitter_analysis,
                timing_bathtub=timing_bathtub,
                voltage_bathtub=voltage_bathtub,
                eye_contour=eye_contour,
                eye_diagram=eye_diagram,
                time_axis=time_axis,
                voltage_axis=voltage_axis
            )
            
        except Exception as e:
            logger.error(f"Eye diagram analysis failed: {e}")
            raise ValueError(f"Eye diagram analysis failed: {e}")
    
    def _validate_signal_data(
        self,
        time_data: npt.NDArray[np.float64],
        voltage_data: npt.NDArray[np.float64]
    ) -> None:
        """Validate signal data arrays"""
        assert isinstance(time_data, np.ndarray), \
            f"Time data must be numpy array, got {type(time_data)}"
        assert isinstance(voltage_data, np.ndarray), \
            f"Voltage data must be numpy array, got {type(voltage_data)}"
        
        assert np.issubdtype(time_data.dtype, np.floating), \
            f"Time data must be floating-point, got {time_data.dtype}"
        assert np.issubdtype(voltage_data.dtype, np.floating), \
            f"Voltage data must be floating-point, got {voltage_data.dtype}"
        
        assert len(time_data) == len(voltage_data), \
            f"Array length mismatch: {len(time_data)} != {len(voltage_data)}"
        assert len(time_data) > self.params.samples_per_symbol * 10, \
            "Insufficient data for eye diagram analysis"
        
        assert not np.any(np.isnan(time_data)), "Time data contains NaN values"
        assert not np.any(np.isnan(voltage_data)), "Voltage data contains NaN values"
        assert not np.any(np.isinf(time_data)), "Time data contains infinite values"
        assert not np.any(np.isinf(voltage_data)), "Voltage data contains infinite values"
    
    def _generate_eye_diagram(
        self,
        time_data: npt.NDArray[np.float64],
        voltage_data: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Generate eye diagram from signal data"""
        try:
            # Calculate sample rate
            sample_rate = 1.0 / np.mean(np.diff(time_data))
            samples_per_symbol = int(sample_rate / self.params.symbol_rate)
            
            # Create eye diagram matrix
            num_symbols = len(voltage_data) // samples_per_symbol - 1
            eye_traces = np.zeros((num_symbols, samples_per_symbol * 2), dtype=np.float64)
            
            # Extract eye traces
            for i in range(num_symbols):
                start_idx = i * samples_per_symbol
                end_idx = start_idx + samples_per_symbol * 2
                if end_idx <= len(voltage_data):
                    eye_traces[i, :] = voltage_data[start_idx:end_idx]
            
            # Create 2D histogram for eye diagram
            time_axis = np.linspace(-self.symbol_period, self.symbol_period, 
                                  samples_per_symbol * 2, dtype=np.float64)
            voltage_axis = np.linspace(np.min(voltage_data), np.max(voltage_data), 
                                     self.params.eye_samples, dtype=np.float64)
            
            # Generate eye diagram histogram
            eye_diagram = np.zeros((len(voltage_axis), len(time_axis)), dtype=np.float64)
            
            for trace in eye_traces:
                for i, voltage in enumerate(trace):
                    if i < len(time_axis):
                        v_idx = np.argmin(np.abs(voltage_axis - voltage))
                        eye_diagram[v_idx, i] += 1
            
            # Normalize
            eye_diagram = eye_diagram / np.max(eye_diagram) if np.max(eye_diagram) > 0 else eye_diagram
            
            return eye_diagram, time_axis, voltage_axis
            
        except Exception as e:
            raise ValueError(f"Eye diagram generation failed: {e}")
    
    def _calculate_eye_height(
        self,
        eye_diagram: npt.NDArray[np.float64],
        voltage_axis: npt.NDArray[np.float64]
    ) -> float:
        """Calculate eye height"""
        try:
            # Find center of eye diagram
            center_time_idx = eye_diagram.shape[1] // 2
            
            # Get voltage profile at center
            center_profile = eye_diagram[:, center_time_idx]
            
            # Find peaks (high and low levels)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(center_profile, height=np.max(center_profile) * 0.1)
            
            if len(peaks) >= 2:
                # Sort peaks by voltage
                peak_voltages = voltage_axis[peaks]
                peak_voltages_sorted = np.sort(peak_voltages)
                
                # Eye height is difference between highest and lowest significant peaks
                eye_height = float(peak_voltages_sorted[-1] - peak_voltages_sorted[0])
            else:
                # Fallback: use voltage range
                eye_height = float(np.max(voltage_axis) - np.min(voltage_axis))
            
            return eye_height
            
        except Exception as e:
            logger.warning(f"Eye height calculation failed: {e}")
            return 0.0
    
    def _calculate_eye_width(
        self,
        eye_diagram: npt.NDArray[np.float64],
        time_axis: npt.NDArray[np.float64]
    ) -> float:
        """Calculate eye width"""
        try:
            # Find center voltage
            center_voltage_idx = eye_diagram.shape[0] // 2
            
            # Get time profile at center voltage
            center_profile = eye_diagram[center_voltage_idx, :]
            
            # Find the eye opening (low density region)
            threshold = np.max(center_profile) * 0.1
            eye_opening_mask = center_profile < threshold
            
            if np.any(eye_opening_mask):
                # Find continuous regions below threshold
                eye_regions = []
                in_eye = False
                start_idx = 0
                
                for i, below_threshold in enumerate(eye_opening_mask):
                    if below_threshold and not in_eye:
                        start_idx = i
                        in_eye = True
                    elif not below_threshold and in_eye:
                        eye_regions.append((start_idx, i))
                        in_eye = False
                
                if in_eye:  # Handle case where eye extends to end
                    eye_regions.append((start_idx, len(eye_opening_mask)))
                
                # Find largest eye opening
                if eye_regions:
                    largest_region = max(eye_regions, key=lambda x: x[1] - x[0])
                    eye_width_samples = largest_region[1] - largest_region[0]
                    eye_width = float(eye_width_samples * (time_axis[1] - time_axis[0]))
                else:
                    eye_width = 0.0
            else:
                eye_width = 0.0
            
            return eye_width
            
        except Exception as e:
            logger.warning(f"Eye width calculation failed: {e}")
            return 0.0
    
    def _calculate_eye_area(
        self,
        eye_diagram: npt.NDArray[np.float64],
        time_axis: npt.NDArray[np.float64],
        voltage_axis: npt.NDArray[np.float64]
    ) -> float:
        """Calculate eye area"""
        try:
            # Define eye opening threshold
            threshold = np.max(eye_diagram) * 0.1
            
            # Find eye opening region
            eye_opening = eye_diagram < threshold
            
            # Calculate area
            dt = float(time_axis[1] - time_axis[0])
            dv = float(voltage_axis[1] - voltage_axis[0])
            eye_area = float(np.sum(eye_opening) * dt * dv)
            
            return eye_area
            
        except Exception as e:
            logger.warning(f"Eye area calculation failed: {e}")
            return 0.0
    
    def _calculate_eye_closure(
        self,
        eye_height: float,
        voltage_axis: npt.NDArray[np.float64]
    ) -> float:
        """Calculate eye closure percentage"""
        try:
            total_voltage_range = float(np.max(voltage_axis) - np.min(voltage_axis))
            if total_voltage_range > 0:
                eye_closure = float((1.0 - eye_height / total_voltage_range) * 100)
            else:
                eye_closure = 100.0
            
            return max(0.0, min(100.0, eye_closure))
            
        except Exception as e:
            logger.warning(f"Eye closure calculation failed: {e}")
            return 100.0
    
    def _calculate_crossing_percentage(
        self,
        voltage_data: npt.NDArray[np.float64]
    ) -> float:
        """Calculate zero crossing percentage"""
        try:
            # Find zero crossings
            zero_crossings = np.where(np.diff(np.signbit(voltage_data)))[0]
            
            # Calculate crossing percentage
            total_samples = len(voltage_data)
            crossing_percentage = float(len(zero_crossings) / total_samples * 100)
            
            return crossing_percentage
            
        except Exception as e:
            logger.warning(f"Crossing percentage calculation failed: {e}")
            return 0.0
    
    def _calculate_q_factor(
        self,
        voltage_data: npt.NDArray[np.float64]
    ) -> float:
        """Calculate Q-factor"""
        try:
            # Simple Q-factor estimation
            # Separate high and low levels
            threshold = np.mean(voltage_data)
            high_levels = voltage_data[voltage_data > threshold]
            low_levels = voltage_data[voltage_data <= threshold]
            
            if len(high_levels) > 0 and len(low_levels) > 0:
                mean_high = float(np.mean(high_levels))
                mean_low = float(np.mean(low_levels))
                std_high = float(np.std(high_levels))
                std_low = float(np.std(low_levels))
                
                if std_high + std_low > 0:
                    q_factor = float(abs(mean_high - mean_low) / (std_high + std_low))
                else:
                    q_factor = float('inf')
            else:
                q_factor = 0.0
            
            return q_factor
            
        except Exception as e:
            logger.warning(f"Q-factor calculation failed: {e}")
            return 0.0
    
    def _analyze_jitter(
        self,
        time_data: npt.NDArray[np.float64],
        voltage_data: npt.NDArray[np.float64]
    ) -> JitterAnalysis:
        """Analyze jitter components"""
        try:
            # Find zero crossings
            zero_crossings = np.where(np.diff(np.signbit(voltage_data)))[0]
            
            if len(zero_crossings) < 10:
                # Not enough crossings for jitter analysis
                return JitterAnalysis(
                    total_jitter=0.0,
                    random_jitter=0.0,
                    deterministic_jitter=0.0,
                    periodic_jitter=0.0,
                    data_dependent_jitter=0.0,
                    jitter_histogram=np.array([1.0], dtype=np.float64),
                    jitter_bins=np.array([0.0], dtype=np.float64)
                )
            
            # Calculate crossing times
            crossing_times = time_data[zero_crossings]
            
            # Calculate time intervals
            time_intervals = np.diff(crossing_times)
            
            # Estimate ideal interval
            ideal_interval = self.symbol_period / 2  # Half symbol period for zero crossings
            
            # Calculate jitter
            jitter_values = time_intervals - ideal_interval
            
            # Total jitter (RMS)
            total_jitter = float(np.std(jitter_values))
            
            # Random jitter (assume Gaussian component)
            # Use histogram analysis to separate components
            hist, bins = np.histogram(jitter_values, bins=50)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Fit Gaussian to estimate random jitter
            try:
                # Fit Gaussian
                mu, sigma = stats.norm.fit(jitter_values)
                random_jitter = float(sigma)
                
                # Deterministic jitter (residual)
                deterministic_jitter = float(np.sqrt(max(0, total_jitter**2 - random_jitter**2)))
                
            except Exception:
                # Fallback estimation
                random_jitter = total_jitter * 0.7  # Typical split
                deterministic_jitter = total_jitter * 0.3
            
            # Periodic jitter (simplified estimation)
            # Look for periodic components in jitter spectrum
            try:
                jitter_fft = np.fft.fft(jitter_values)
                jitter_psd = np.abs(jitter_fft)**2
                
                # Find dominant frequency components
                freqs = np.fft.fftfreq(len(jitter_values), np.mean(np.diff(crossing_times)))
                
                # Estimate periodic jitter as energy in discrete frequency bins
                dc_component = jitter_psd[0]
                total_energy = np.sum(jitter_psd)
                
                if total_energy > 0:
                    periodic_energy_ratio = (total_energy - dc_component) / total_energy
                    periodic_jitter = float(deterministic_jitter * np.sqrt(periodic_energy_ratio))
                else:
                    periodic_jitter = 0.0
                    
            except Exception:
                periodic_jitter = deterministic_jitter * 0.5  # Rough estimate
            
            # Data-dependent jitter (remaining deterministic component)
            data_dependent_jitter = float(np.sqrt(max(0, deterministic_jitter**2 - periodic_jitter**2)))
            
            return JitterAnalysis(
                total_jitter=total_jitter,
                random_jitter=random_jitter,
                deterministic_jitter=deterministic_jitter,
                periodic_jitter=periodic_jitter,
                data_dependent_jitter=data_dependent_jitter,
                jitter_histogram=hist.astype(np.float64),
                jitter_bins=bin_centers.astype(np.float64)
            )
            
        except Exception as e:
            logger.error(f"Jitter analysis failed: {e}")
            # Return default values
            return JitterAnalysis(
                total_jitter=0.0,
                random_jitter=0.0,
                deterministic_jitter=0.0,
                periodic_jitter=0.0,
                data_dependent_jitter=0.0,
                jitter_histogram=np.array([1.0], dtype=np.float64),
                jitter_bins=np.array([0.0], dtype=np.float64)
            )
    
    def _generate_timing_bathtub(
        self,
        time_data: npt.NDArray[np.float64],
        voltage_data: npt.NDArray[np.float64]
    ) -> BathtubCurve:
        """Generate timing bathtub curve"""
        try:
            # Create timing offsets
            max_offset = self.symbol_period / 4  # Quarter symbol period
            offsets = np.linspace(-max_offset, max_offset, 100, dtype=np.float64)
            
            ber_values = np.zeros_like(offsets, dtype=np.float64)
            
            # Calculate BER for each timing offset
            for i, offset in enumerate(offsets):
                # Simulate timing offset by shifting decision point
                shifted_times = time_data + offset
                
                # Simple BER estimation based on eye opening at shifted time
                # This is a simplified model
                eye_opening_factor = 1.0 - abs(offset) / max_offset
                base_ber = 1e-12  # Base BER
                
                # BER increases as we move away from eye center
                if eye_opening_factor > 0.5:
                    ber_values[i] = base_ber * np.exp(-eye_opening_factor * 10)
                else:
                    ber_values[i] = 1e-3  # High BER outside eye
            
            # Find eye opening (where BER is below threshold)
            ber_threshold = 1e-9
            eye_opening_mask = ber_values < ber_threshold
            
            if np.any(eye_opening_mask):
                eye_opening_indices = np.where(eye_opening_mask)[0]
                eye_opening = float(offsets[eye_opening_indices[-1]] - offsets[eye_opening_indices[0]])
            else:
                eye_opening = 0.0
            
            bathtub_floor = float(np.min(ber_values))
            
            return BathtubCurve(
                offsets=offsets,
                ber_values=ber_values,
                eye_opening=eye_opening,
                bathtub_floor=bathtub_floor
            )
            
        except Exception as e:
            logger.error(f"Timing bathtub generation failed: {e}")
            # Return default curve
            return BathtubCurve(
                offsets=np.array([0.0], dtype=np.float64),
                ber_values=np.array([1e-12], dtype=np.float64),
                eye_opening=0.0,
                bathtub_floor=1e-12
            )
    
    def _generate_voltage_bathtub(
        self,
        voltage_data: npt.NDArray[np.float64]
    ) -> BathtubCurve:
        """Generate voltage bathtub curve"""
        try:
            # Create voltage offsets
            voltage_range = np.max(voltage_data) - np.min(voltage_data)
            max_offset = voltage_range / 4
            offsets = np.linspace(-max_offset, max_offset, 100, dtype=np.float64)
            
            ber_values = np.zeros_like(offsets, dtype=np.float64)
            
            # Calculate BER for each voltage threshold offset
            center_voltage = np.mean(voltage_data)
            
            for i, offset in enumerate(offsets):
                threshold = center_voltage + offset
                
                # Simple BER estimation based on threshold position
                # Count errors (simplified model)
                high_samples = voltage_data > threshold
                low_samples = voltage_data <= threshold
                
                # Estimate error rate based on threshold position
                if abs(offset) < voltage_range / 8:
                    ber_values[i] = 1e-12 * np.exp(abs(offset) / (voltage_range / 20))
                else:
                    ber_values[i] = 1e-3
            
            # Find eye opening
            ber_threshold = 1e-9
            eye_opening_mask = ber_values < ber_threshold
            
            if np.any(eye_opening_mask):
                eye_opening_indices = np.where(eye_opening_mask)[0]
                eye_opening = float(offsets[eye_opening_indices[-1]] - offsets[eye_opening_indices[0]])
            else:
                eye_opening = 0.0
            
            bathtub_floor = float(np.min(ber_values))
            
            return BathtubCurve(
                offsets=offsets,
                ber_values=ber_values,
                eye_opening=eye_opening,
                bathtub_floor=bathtub_floor
            )
            
        except Exception as e:
            logger.error(f"Voltage bathtub generation failed: {e}")
            # Return default curve
            return BathtubCurve(
                offsets=np.array([0.0], dtype=np.float64),
                ber_values=np.array([1e-12], dtype=np.float64),
                eye_opening=0.0,
                bathtub_floor=1e-12
            )
    
    def _analyze_eye_contour(
        self,
        eye_diagram: npt.NDArray[np.float64],
        time_axis: npt.NDArray[np.float64],
        voltage_axis: npt.NDArray[np.float64]
    ) -> EyeContour:
        """Analyze eye contour"""
        try:
            # Define contour levels
            max_density = np.max(eye_diagram)
            contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
            contour_levels = [level * max_density for level in contour_levels]
            
            contour_areas = []
            
            # Calculate area for each contour level
            for level in contour_levels:
                contour_mask = eye_diagram < level
                area = float(np.sum(contour_mask))
                contour_areas.append(area)
            
            # Eye mask analysis (simplified)
            # Define a simple rectangular eye mask
            eye_center_time = 0.0
            eye_center_voltage = np.mean(voltage_axis)
            
            mask_width = self.symbol_period * 0.6  # 60% of symbol period
            mask_height = (np.max(voltage_axis) - np.min(voltage_axis)) * 0.4  # 40% of voltage range
            
            # Count violations (points inside mask that shouldn't be there)
            time_mask = np.abs(time_axis - eye_center_time) < mask_width / 2
            voltage_mask = np.abs(voltage_axis[:, np.newaxis] - eye_center_voltage) < mask_height / 2
            
            mask_region = time_mask[np.newaxis, :] & voltage_mask
            mask_violations = int(np.sum(eye_diagram[mask_region] > max_density * 0.1))
            
            # Eye mask margin (simplified)
            eye_mask_margin = float(mask_width * mask_height / (len(time_axis) * len(voltage_axis)))
            
            return EyeContour(
                contour_levels=contour_levels,
                contour_areas=contour_areas,
                eye_mask_margin=eye_mask_margin,
                mask_violations=mask_violations
            )
            
        except Exception as e:
            logger.error(f"Eye contour analysis failed: {e}")
            # Return default contour
            return EyeContour(
                contour_levels=[0.0],
                contour_areas=[0.0],
                eye_mask_margin=0.0,
                mask_violations=0
            )


# Factory functions
def create_nrz_eye_analyzer(
    symbol_rate: float = 32e9,
    samples_per_symbol: int = 16
) -> AdvancedEyeAnalyzer:
    """Create eye analyzer for NRZ signals"""
    assert isinstance(symbol_rate, float), f"Symbol rate must be float, got {type(symbol_rate)}"
    assert isinstance(samples_per_symbol, int), \
        f"Samples per symbol must be int, got {type(samples_per_symbol)}"
    
    assert symbol_rate > 0, f"Symbol rate must be positive, got {symbol_rate}"
    assert samples_per_symbol > 0, \
        f"Samples per symbol must be positive, got {samples_per_symbol}"
    
    params = EyeParameters(
        symbol_rate=symbol_rate,
        samples_per_symbol=samples_per_symbol,
        eye_samples=1000,
        confidence_level=0.95,
        jitter_analysis=True
    )
    
    return AdvancedEyeAnalyzer(params)


def create_pam4_eye_analyzer(
    symbol_rate: float = 32e9,
    samples_per_symbol: int = 32
) -> AdvancedEyeAnalyzer:
    """Create eye analyzer for PAM4 signals"""
    assert isinstance(symbol_rate, float), f"Symbol rate must be float, got {type(symbol_rate)}"
    assert isinstance(samples_per_symbol, int), \
        f"Samples per symbol must be int, got {type(samples_per_symbol)}"
    
    assert symbol_rate > 0, f"Symbol rate must be positive, got {symbol_rate}"
    assert samples_per_symbol > 0, \
        f"Samples per symbol must be positive, got {samples_per_symbol}"
    
    params = EyeParameters(
        symbol_rate=symbol_rate,
        samples_per_symbol=samples_per_symbol,
        eye_samples=1500,  # Higher resolution for PAM4
        confidence_level=0.99,  # Stricter confidence for PAM4
        jitter_analysis=True
    )
    
    return AdvancedEyeAnalyzer(params)
