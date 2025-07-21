"""
PCIe Analyzer Module

This module provides analysis capabilities for PCI Express 6.0 signals
with comprehensive type checking and validation.

Features:
- NRZ/PAM4 dual-mode support
- Link training analysis
- Equalization optimization
- Compliance testing
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

from .mode_switcher import ModeConfig, ModeSwitcher, SignalMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PCIeConfig:
    """PCIe configuration with validation"""
    mode: SignalMode
    sample_rate: float
    bandwidth: float
    voltage_range: float
    link_speed: float
    lane_count: int
    
    def __post_init__(self) -> None:
        """Validate PCIe configuration"""
        # Type validation
        assert isinstance(self.mode, SignalMode), \
            "Mode must be SignalMode enum"
        assert isinstance(self.sample_rate, float), \
            "Sample rate must be float"
        assert isinstance(self.bandwidth, float), \
            "Bandwidth must be float"
        assert isinstance(self.voltage_range, float), \
            "Voltage range must be float"
        assert isinstance(self.link_speed, float), \
            "Link speed must be float"
        assert isinstance(self.lane_count, int), \
            "Lane count must be integer"
            
        # Value validation
        assert self.sample_rate > 0, "Sample rate must be positive"
        assert self.bandwidth > 0, "Bandwidth must be positive"
        assert self.voltage_range > 0, "Voltage range must be positive"
        assert self.link_speed > 0, "Link speed must be positive"
        assert self.lane_count > 0, "Lane count must be positive"

@dataclass
class TrainingConfig:
    """Link training configuration"""
    preset_index: int
    adaptation_mode: str
    target_snr: float
    max_iterations: int
    
    def __post_init__(self) -> None:
        """Validate training configuration"""
        # Type validation
        assert isinstance(self.preset_index, int), \
            "Preset index must be integer"
        assert isinstance(self.adaptation_mode, str), \
            "Adaptation mode must be string"
        assert isinstance(self.target_snr, float), \
            "Target SNR must be float"
        assert isinstance(self.max_iterations, int), \
            "Max iterations must be integer"
            
        # Value validation
        assert 0 <= self.preset_index <= 10, \
            "Preset index must be between 0 and 10"
        assert self.adaptation_mode in ['fast', 'precise'], \
            "Invalid adaptation mode"
        assert self.target_snr > 0, "Target SNR must be positive"
        assert self.max_iterations > 0, "Max iterations must be positive"

@dataclass
class TrainingResults:
    """Link training results"""
    success: bool
    iterations: int
    final_snr: float
    tap_weights: List[float]
    error_history: List[float]
    
    def __post_init__(self) -> None:
        """Validate training results"""
        # Type validation
        assert isinstance(self.success, bool), \
            "Success must be boolean"
        assert isinstance(self.iterations, int), \
            "Iterations must be integer"
        assert isinstance(self.final_snr, float), \
            "Final SNR must be float"
        assert all(isinstance(x, float) for x in self.tap_weights), \
            "Tap weights must be floats"
        assert all(isinstance(x, float) for x in self.error_history), \
            "Error history must be floats"
            
        # Value validation
        assert self.iterations >= 0, "Iterations must be non-negative"
        assert self.final_snr >= 0, "Final SNR must be non-negative"
        assert len(self.tap_weights) > 0, "Tap weights cannot be empty"
        assert len(self.error_history) > 0, "Error history cannot be empty"

class PCIeAnalyzer:
    """PCIe signal analyzer with dual-mode support"""
    
    def __init__(
        self,
        config: PCIeConfig,
        mode_switcher: Optional[ModeSwitcher] = None
    ) -> None:
        """
        Initialize PCIe analyzer
        
        Args:
            config: PCIe configuration
            mode_switcher: Optional mode switcher instance
            
        Raises:
            AssertionError: If configuration is invalid
        """
        # Validate config
        assert isinstance(config, PCIeConfig), \
            "Config must be PCIeConfig"
        
        self.config = config
        self.mode_switcher = mode_switcher or ModeSwitcher(
            default_mode=config.mode,
            default_sample_rate=config.sample_rate,
            default_bandwidth=config.bandwidth
        )
        
        # Initialize analysis state
        self.current_data: Dict[str, npt.NDArray[np.float64]] = {}
        self.training_state: Optional[TrainingResults] = None
        
        logger.info(f"PCIe analyzer initialized in {config.mode.name} mode")
        
    def configure_mode(
        self,
        new_mode: SignalMode,
        mode_config: Optional[ModeConfig] = None
    ) -> bool:
        """
        Configure signal mode
        
        Args:
            new_mode: Target signal mode
            mode_config: Optional mode configuration
            
        Returns:
            True if configuration successful
            
        Raises:
            ValueError: If configuration fails
        """
        # Validate input
        assert isinstance(new_mode, SignalMode), \
            "New mode must be SignalMode enum"
            
        try:
            # Switch mode
            result = self.mode_switcher.switch_mode(new_mode, mode_config)
            
            if result.success:
                self.config.mode = new_mode
                logger.info(f"Configured for {new_mode.name} mode")
                return True
            else:
                logger.error(f"Mode configuration failed: {result.error_message}")
                return False
                
        except Exception as e:
            raise ValueError(f"Mode configuration failed: {e}")
            
    def analyze_signal(
        self,
        data: Dict[str, npt.NDArray[np.float64]]
    ) -> Dict[str, float]:
        """
        Analyze PCIe signal quality
        
        Args:
            data: Dictionary with time and voltage arrays
            
        Returns:
            Dictionary of quality metrics
            
        Raises:
            ValueError: If analysis fails
        """
        try:
            # Validate input data
            assert 'time' in data and 'voltage' in data, \
                "Data must contain 'time' and 'voltage' arrays"
            self._validate_signal_arrays(data['time'], data['voltage'])
            
            # Store data
            self.current_data = {
                'time': data['time'].astype(np.float64),
                'voltage': data['voltage'].astype(np.float64)
            }
            
            # Mode-specific analysis
            if self.config.mode == SignalMode.NRZ:
                return self._analyze_nrz_signal()
            else:  # PAM4
                return self._analyze_pam4_signal()
                
        except Exception as e:
            raise ValueError(f"Signal analysis failed: {e}")
            
    def _validate_signal_arrays(
        self,
        time_data: npt.NDArray[np.float64],
        voltage_data: npt.NDArray[np.float64]
    ) -> None:
        """
        Validate signal array properties
        
        Args:
            time_data: Time points array
            voltage_data: Voltage measurements array
            
        Raises:
            AssertionError: If validation fails
        """
        # Type validation
        assert isinstance(time_data, np.ndarray), \
            "Time data must be numpy array"
        assert isinstance(voltage_data, np.ndarray), \
            "Voltage data must be numpy array"
            
        # Data type validation
        assert np.issubdtype(time_data.dtype, np.floating), \
            "Time data must be floating-point"
        assert np.issubdtype(voltage_data.dtype, np.floating), \
            "Voltage data must be floating-point"
            
        # Array validation
        assert len(time_data) == len(voltage_data), \
            "Array length mismatch"
        assert len(time_data) > 0, "Arrays cannot be empty"
        
        # Value validation
        assert not np.any(np.isnan(time_data)), \
            "Time data contains NaN values"
        assert not np.any(np.isnan(voltage_data)), \
            "Voltage data contains NaN values"
        assert not np.any(np.isinf(time_data)), \
            "Time data contains infinite values"
        assert not np.any(np.isinf(voltage_data)), \
            "Voltage data contains infinite values"
            
    def _analyze_nrz_signal(self) -> Dict[str, float]:
        """
        Analyze NRZ signal quality
        
        Returns:
            Dictionary of quality metrics
            
        Raises:
            ValueError: If analysis fails
        """
        try:
            voltage = self.current_data['voltage']
            time = self.current_data['time']
            
            # Calculate level separation
            hist, bins = np.histogram(voltage, bins=100)
            hist = hist.astype(np.float64)  # Convert to float
            levels = self._find_nrz_levels(hist, bins)
            level_separation = float(np.abs(levels[1] - levels[0]))
            
            # Calculate SNR
            signal_power = float(np.mean(voltage**2))
            noise_power = float(np.var(voltage))
            snr = float(10 * np.log10(signal_power / noise_power))
            
            # Calculate jitter
            zero_crossings = np.where(np.diff(np.signbit(voltage)))[0]
            crossing_times = time[zero_crossings]
            jitter = float(np.std(np.diff(crossing_times)))
            
            return {
                'level_separation': level_separation,
                'snr_db': snr,
                'jitter_ps': jitter * 1e12
            }
            
        except Exception as e:
            raise ValueError(f"NRZ analysis failed: {e}")
            
    def _analyze_pam4_signal(self) -> Dict[str, float]:
        """
        Analyze PAM4 signal quality
        
        Returns:
            Dictionary of quality metrics
            
        Raises:
            ValueError: If analysis fails
        """
        try:
            voltage = self.current_data['voltage']
            time = self.current_data['time']
            
            # Find PAM4 levels
            hist, bins = np.histogram(voltage, bins=100)
            hist = hist.astype(np.float64)  # Convert to float
            levels = self._find_pam4_levels(hist, bins)
            level_separations = np.diff(np.sort(levels))
            min_separation = float(np.min(level_separations))
            
            # Calculate EVM
            ideal_levels = np.array([-3.0, -1.0, 1.0, 3.0])
            errors = voltage - ideal_levels[np.argmin(
                np.abs(voltage[:, np.newaxis] - ideal_levels),
                axis=1
            )]
            rms_evm = float(np.sqrt(np.mean(errors**2)))
            
            # Calculate SNR
            signal_power = float(np.mean(voltage**2))
            noise_power = float(np.var(errors))
            snr = float(10 * np.log10(signal_power / noise_power))
            
            return {
                'min_level_separation': min_separation,
                'rms_evm_percent': rms_evm * 100,
                'snr_db': snr
            }
            
        except Exception as e:
            raise ValueError(f"PAM4 analysis failed: {e}")
            
    def _find_nrz_levels(
        self,
        histogram: npt.NDArray[np.float64],
        bin_edges: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Find NRZ voltage levels
        
        Args:
            histogram: Signal histogram
            bin_edges: Histogram bin edges
            
        Returns:
            Array of voltage levels
            
        Raises:
            ValueError: If level detection fails
        """
        # Validate inputs
        assert isinstance(histogram, np.ndarray), "Histogram must be numpy array"
        assert isinstance(bin_edges, np.ndarray), "Bin edges must be numpy array"
        
        # Type validation
        if not np.issubdtype(histogram.dtype, np.floating):
            raise ValueError("Histogram must be floating-point")
        if not np.issubdtype(bin_edges.dtype, np.floating):
            raise ValueError("Bin edges must be floating-point")
            
        try:
            # Find peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(histogram, height=np.max(histogram)/4, distance=len(histogram)//10)
            
            # If we don't find exactly 2 peaks, use alternative method
            if len(peaks) != 2:
                # Use K-means clustering to find 2 levels
                from sklearn.cluster import KMeans
                
                # Get bin centers
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Weight by histogram values and create data points
                data_points = []
                for i, count in enumerate(histogram):
                    if count > 0:
                        data_points.extend([bin_centers[i]] * int(count))
                
                if len(data_points) < 2:
                    # Fallback: use min and max
                    return np.array([np.min(bin_centers), np.max(bin_centers)], dtype=np.float64)
                
                # K-means clustering
                kmeans = KMeans(n_clusters=2, random_state=42)
                data_array = np.array(data_points).reshape(-1, 1)
                kmeans.fit(data_array)
                
                levels = kmeans.cluster_centers_.flatten()
                return np.sort(levels).astype(np.float64)
            else:
                # Get voltage levels from peaks
                levels = bin_edges[peaks]
                return np.sort(levels).astype(np.float64)
            
        except Exception as e:
            # Final fallback: use quartiles
            logger.warning(f"NRZ level detection using fallback method: {e}")
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            q25 = np.percentile(bin_centers, 25)
            q75 = np.percentile(bin_centers, 75)
            return np.array([q25, q75], dtype=np.float64)
            
    def _find_pam4_levels(
        self,
        histogram: npt.NDArray[np.float64],
        bin_edges: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Find PAM4 voltage levels
        
        Args:
            histogram: Signal histogram
            bin_edges: Histogram bin edges
            
        Returns:
            Array of voltage levels
            
        Raises:
            ValueError: If level detection fails
        """
        # Validate inputs
        assert isinstance(histogram, np.ndarray), "Histogram must be numpy array"
        assert isinstance(bin_edges, np.ndarray), "Bin edges must be numpy array"
        
        # Type validation
        if not np.issubdtype(histogram.dtype, np.floating):
            raise ValueError("Histogram must be floating-point")
        if not np.issubdtype(bin_edges.dtype, np.floating):
            raise ValueError("Bin edges must be floating-point")
            
        try:
            # Find peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(histogram, height=np.max(histogram)/6, distance=len(histogram)//20)
            
            # If we don't find exactly 4 peaks, use alternative method
            if len(peaks) != 4:
                # Use K-means clustering to find 4 levels
                from sklearn.cluster import KMeans
                
                # Get bin centers
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Weight by histogram values and create data points
                data_points = []
                for i, count in enumerate(histogram):
                    if count > 0:
                        data_points.extend([bin_centers[i]] * int(count))
                
                if len(data_points) < 4:
                    # Fallback: use quartiles
                    q0 = np.min(bin_centers)
                    q33 = np.percentile(bin_centers, 33)
                    q67 = np.percentile(bin_centers, 67)
                    q100 = np.max(bin_centers)
                    return np.array([q0, q33, q67, q100], dtype=np.float64)
                
                # K-means clustering
                kmeans = KMeans(n_clusters=4, random_state=42)
                data_array = np.array(data_points).reshape(-1, 1)
                kmeans.fit(data_array)
                
                levels = kmeans.cluster_centers_.flatten()
                return np.sort(levels).astype(np.float64)
            else:
                # Get voltage levels from peaks
                levels = bin_edges[peaks]
                return np.sort(levels).astype(np.float64)
            
        except Exception as e:
            # Final fallback: use quartiles
            logger.warning(f"PAM4 level detection using fallback method: {e}")
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            q0 = np.percentile(bin_centers, 12.5)
            q1 = np.percentile(bin_centers, 37.5)
            q2 = np.percentile(bin_centers, 62.5)
            q3 = np.percentile(bin_centers, 87.5)
            return np.array([q0, q1, q2, q3], dtype=np.float64)
            
    def run_link_training(
        self,
        config: TrainingConfig
    ) -> TrainingResults:
        """
        Run link training sequence
        
        Args:
            config: Training configuration
            
        Returns:
            Training results
            
        Raises:
            ValueError: If training fails
        """
        try:
            # Initialize training
            tap_weights = [0.0] * 5  # 5-tap equalizer
            tap_weights[2] = 1.0  # Center tap
            error_history = []
            
            # Training loop
            for iteration in range(config.max_iterations):
                # Update equalizer
                error = self._update_equalizer(tap_weights)
                error_history.append(float(error))
                
                # Check convergence
                current_snr = self._calculate_snr()
                if current_snr >= config.target_snr:
                    return TrainingResults(
                        success=True,
                        iterations=iteration + 1,
                        final_snr=current_snr,
                        tap_weights=list(map(float, tap_weights)),
                        error_history=error_history
                    )
                    
            # Training failed to converge
            return TrainingResults(
                success=False,
                iterations=config.max_iterations,
                final_snr=current_snr,
                tap_weights=list(map(float, tap_weights)),
                error_history=error_history
            )
            
        except Exception as e:
            raise ValueError(f"Link training failed: {e}")
            
    def _update_equalizer(
        self,
        tap_weights: List[float]
    ) -> float:
        """
        Update equalizer tap weights
        
        Args:
            tap_weights: Current tap weights
            
        Returns:
            Updated error value
            
        Raises:
            ValueError: If update fails
        """
        try:
            # Validate input
            assert all(isinstance(x, float) for x in tap_weights), \
                "Tap weights must be floats"
                
            # Simple LMS update example
            voltage = self.current_data['voltage']
            error = float(np.mean(np.abs(voltage)))
            
            # Update taps
            learning_rate = 0.01
            for i in range(len(tap_weights)):
                tap_weights[i] -= learning_rate * error
                
            return error
            
        except Exception as e:
            raise ValueError(f"Equalizer update failed: {e}")
            
    def _calculate_snr(self) -> float:
        """
        Calculate signal-to-noise ratio
        
        Returns:
            SNR in dB
            
        Raises:
            ValueError: If calculation fails
        """
        try:
            voltage = self.current_data['voltage']
            
            # Calculate power
            signal_power = float(np.mean(voltage**2))
            noise_power = float(np.var(voltage))
            
            # Calculate SNR
            if noise_power > 0:
                return float(10 * np.log10(signal_power / noise_power))
            else:
                return float('inf')
                
        except Exception as e:
            raise ValueError(f"SNR calculation failed: {e}")
