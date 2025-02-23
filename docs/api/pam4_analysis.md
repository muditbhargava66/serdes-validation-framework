# PAM4 Signal Analysis API

## Overview

The PAM4 analysis module provides comprehensive tools for analyzing PAM4 signals with strict type safety and validation. Key features include:

- Level separation analysis
- Error Vector Magnitude (EVM) calculation
- Eye diagram measurements
- Jitter analysis

## Type Definitions

```python
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt

# Type aliases
FloatArray = npt.NDArray[np.float64]
SignalData = Dict[str, FloatArray]
Measurements = Dict[str, Union[float, List[float]]]
```

## Data Classes

### PAM4Levels

```python
@dataclass
class PAM4Levels:
    """PAM4 voltage level measurements"""
    level_means: FloatArray
    level_separations: FloatArray
    uniformity: float
    
    def __post_init__(self) -> None:
        """Validate PAM4 level measurements"""
        # Type validation
        assert isinstance(self.level_means, np.ndarray), \
            "Level means must be numpy array"
        assert isinstance(self.level_separations, np.ndarray), \
            "Level separations must be numpy array"
        assert isinstance(self.uniformity, float), \
            "Uniformity must be float"
        
        # Data validation
        assert len(self.level_means) == 4, \
            f"Expected 4 PAM4 levels, got {len(self.level_means)}"
        assert len(self.level_separations) == 3, \
            f"Expected 3 level separations, got {len(self.level_separations)}"
        assert 0 <= self.uniformity <= 1, \
            f"Uniformity must be between 0 and 1, got {self.uniformity}"
        
        # Type checking
        assert np.issubdtype(self.level_means.dtype, np.floating), \
            f"Level means must be floating-point, got {self.level_means.dtype}"
        assert np.issubdtype(self.level_separations.dtype, np.floating), \
            f"Level separations must be floating-point, got {self.level_separations.dtype}"
```

### EVMResults

```python
@dataclass
class EVMResults:
    """Error Vector Magnitude results"""
    rms_evm_percent: float
    peak_evm_percent: float
    errors: FloatArray
    
    def __post_init__(self) -> None:
        """Validate EVM results"""
        # Type validation
        assert isinstance(self.rms_evm_percent, float), \
            "RMS EVM must be float"
        assert isinstance(self.peak_evm_percent, float), \
            "Peak EVM must be float"
        assert isinstance(self.errors, np.ndarray), \
            "Errors must be numpy array"
        
        # Value validation
        assert 0 <= self.rms_evm_percent <= 100, \
            f"RMS EVM must be percentage, got {self.rms_evm_percent}"
        assert 0 <= self.peak_evm_percent <= 100, \
            f"Peak EVM must be percentage, got {self.peak_evm_percent}"
        assert self.peak_evm_percent >= self.rms_evm_percent, \
            "Peak EVM must be greater than or equal to RMS EVM"
        
        # Array validation
        assert np.issubdtype(self.errors.dtype, np.floating), \
            f"Errors must be floating-point, got {self.errors.dtype}"
```

### EyeResults

```python
@dataclass
class EyeResults:
    """Eye diagram measurements"""
    eye_heights: List[float]
    eye_widths: List[float]
    worst_eye_height: float = field(init=False)
    worst_eye_width: float = field(init=False)
    
    def __post_init__(self) -> None:
        """Validate and calculate eye measurements"""
        # Type validation
        assert all(isinstance(h, float) for h in self.eye_heights), \
            "Eye heights must be floats"
        assert all(isinstance(w, float) for w in self.eye_widths), \
            "Eye widths must be floats"
        
        # Length validation
        assert len(self.eye_heights) == 3, \
            f"Expected 3 eye heights, got {len(self.eye_heights)}"
        assert len(self.eye_widths) == 3, \
            f"Expected 3 eye widths, got {len(self.eye_widths)}"
        
        # Value validation
        assert all(h >= 0 for h in self.eye_heights), \
            "Eye heights must be non-negative"
        assert all(w >= 0 for w in self.eye_widths), \
            "Eye widths must be non-negative"
        
        # Calculate worst values
        self.worst_eye_height = min(self.eye_heights)
        self.worst_eye_width = min(self.eye_widths)
```

## PAM4 Analyzer

### Initialization

```python
class PAM4Analyzer:
    """PAM4 signal analyzer with type validation"""
    
    def __init__(
        self,
        data: SignalData,
        sample_rate: float = 256e9,
        symbol_rate: float = 112e9
    ) -> None:
        """
        Initialize PAM4 analyzer
        
        Args:
            data: Dictionary with 'time' and 'voltage' arrays
            sample_rate: Sampling rate in Hz
            symbol_rate: Symbol rate in Hz
        """
        # Validate input types
        assert isinstance(data, dict), "Data must be dictionary"
        assert isinstance(sample_rate, float), "Sample rate must be float"
        assert isinstance(symbol_rate, float), "Symbol rate must be float"
        
        # Validate required data
        assert 'time' in data and 'voltage' in data, \
            "Data must contain 'time' and 'voltage' arrays"
        
        # Validate array types
        self._validate_signal_arrays(data['time'], data['voltage'])
        
        # Store validated data
        self.data = {
            'time': data['time'].astype(np.float64),
            'voltage': data['voltage'].astype(np.float64)
        }
        self.sample_rate = float(sample_rate)
        self.symbol_rate = float(symbol_rate)
```

### Signal Validation

```python
def _validate_signal_arrays(
    self,
    time_data: FloatArray,
    voltage_data: FloatArray
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
        f"Time data must be numpy array, got {type(time_data)}"
    assert isinstance(voltage_data, np.ndarray), \
        f"Voltage data must be numpy array, got {type(voltage_data)}"
    
    # Data type validation
    assert np.issubdtype(time_data.dtype, np.floating), \
        f"Time data must be floating-point, got {time_data.dtype}"
    assert np.issubdtype(voltage_data.dtype, np.floating), \
        f"Voltage data must be floating-point, got {voltage_data.dtype}"
    
    # Array validation
    assert len(time_data) == len(voltage_data), \
        f"Array length mismatch: {len(time_data)} != {len(voltage_data)}"
    assert len(time_data) > 0, "Arrays cannot be empty"
    
    # Value validation
    assert not np.any(np.isnan(time_data)), "Time data contains NaN values"
    assert not np.any(np.isnan(voltage_data)), "Voltage data contains NaN values"
    assert not np.any(np.isinf(time_data)), "Time data contains infinite values"
    assert not np.any(np.isinf(voltage_data)), "Voltage data contains infinite values"
    
    # Time array validation
    assert np.all(np.diff(time_data) > 0), "Time data must be monotonically increasing"
```

### Level Analysis

```python
def analyze_level_separation(
    self,
    voltage_column: str = 'voltage',
    threshold: float = 0.1
) -> PAM4Levels:
    """
    Analyze PAM4 voltage level separation
    
    Args:
        voltage_column: Name of voltage data column
        threshold: Detection threshold
        
    Returns:
        PAM4Levels object with analysis results
    """
    # Validate inputs
    assert isinstance(voltage_column, str), "Voltage column must be string"
    assert isinstance(threshold, float), "Threshold must be float"
    assert 0 < threshold < 1, "Threshold must be between 0 and 1"
    
    try:
        # Get voltage data
        voltage_data = self.data[voltage_column]
        
        # Find levels using histogram
        hist, bins = np.histogram(voltage_data, bins=100)
        level_means = self._find_voltage_levels(hist, bins)
        
        # Calculate level separations
        level_separations = np.diff(np.sort(level_means))
        
        # Calculate uniformity
        uniformity = float(np.std(level_separations) / np.mean(level_separations))
        
        return PAM4Levels(
            level_means=level_means,
            level_separations=level_separations,
            uniformity=uniformity
        )
        
    except KeyError:
        raise ValueError(f"Column '{voltage_column}' not found in data")
    except Exception as e:
        raise AnalysisError(f"Level analysis failed: {e}")
```

### EVM Calculation

```python
def calculate_evm(
    self,
    measured_column: str = 'voltage',
    reference_column: Optional[str] = None
) -> EVMResults:
    """
    Calculate Error Vector Magnitude
    
    Args:
        measured_column: Name of measured signal column
        reference_column: Optional reference signal column
        
    Returns:
        EVMResults object with EVM calculations
    """
    # Validate inputs
    assert isinstance(measured_column, str), "Measured column must be string"
    if reference_column is not None:
        assert isinstance(reference_column, str), "Reference column must be string"
    
    try:
        # Get measured signal
        measured = self.data[measured_column]
        
        # Get or generate reference
        if reference_column is not None:
            reference = self.data[reference_column]
        else:
            reference = self._generate_ideal_pam4(len(measured))
        
        # Calculate errors
        errors = measured - reference
        
        # Calculate EVM percentages
        rms_error = float(np.sqrt(np.mean(errors**2)))
        peak_error = float(np.max(np.abs(errors)))
        reference_rms = float(np.sqrt(np.mean(reference**2)))
        
        rms_evm = (rms_error / reference_rms) * 100
        peak_evm = (peak_error / reference_rms) * 100
        
        return EVMResults(
            rms_evm_percent=rms_evm,
            peak_evm_percent=peak_evm,
            errors=errors
        )
        
    except KeyError as e:
        raise ValueError(f"Column not found: {e}")
    except Exception as e:
        raise AnalysisError(f"EVM calculation failed: {e}")
```

### Eye Analysis

```python
def analyze_eye_diagram(
    self,
    voltage_column: str = 'voltage',
    time_column: str = 'time',
    ui_period: float = 8.9e-12
) -> EyeResults:
    """
    Analyze eye diagram parameters
    
    Args:
        voltage_column: Name of voltage column
        time_column: Name of time column
        ui_period: Unit interval in seconds
        
    Returns:
        EyeResults object with eye measurements
    """
    # Validate inputs
    assert isinstance(voltage_column, str), "Voltage column must be string"
    assert isinstance(time_column, str), "Time column must be string"
    assert isinstance(ui_period, float), "UI period must be float"
    assert ui_period > 0, "UI period must be positive"
    
    try:
        # Get signal data
        voltage = self.data[voltage_column]
        time = self.data[time_column]
        
        # Calculate eye parameters
        eye_heights = self._calculate_eye_heights(voltage)
        eye_widths = self._calculate_eye_widths(voltage, time, ui_period)
        
        return EyeResults(
            eye_heights=list(map(float, eye_heights)),
            eye_widths=list(map(float, eye_widths))
        )
        
    except KeyError as e:
        raise ValueError(f"Column not found: {e}")
    except Exception as e:
        raise AnalysisError(f"Eye analysis failed: {e}")
```

## Error Handling

```python
class AnalysisError(Exception):
    """Base class for analysis errors"""
    pass

class ValidationError(AnalysisError):
    """Error in data validation"""
    pass

class MeasurementError(AnalysisError):
    """Error in measurement calculation"""
    pass
```

## Usage Example

```python
def analyze_pam4_signal(
    time_data: FloatArray,
    voltage_data: FloatArray,
    sample_rate: float = 256e9
) -> Measurements:
    """
    Analyze PAM4 signal quality
    
    Args:
        time_data: Time points array
        voltage_data: Voltage measurements array
        sample_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of measurement results
    """
    try:
        # Create analyzer
        analyzer = PAM4Analyzer({
            'time': time_data,
            'voltage': voltage_data
        }, sample_rate=sample_rate)
        
        # Analyze levels
        level_results = analyzer.analyze_level_separation()
        
        # Calculate EVM
        evm_results = analyzer.calculate_evm()
        
        # Analyze eye
        eye_results = analyzer.analyze_eye_diagram()
        
        # Compile results
        return {
            'level_uniformity': float(level_results.uniformity),
            'rms_evm': float(evm_results.rms_evm_percent),
            'peak_evm': float(evm_results.peak_evm_percent),
            'worst_eye_height': float(eye_results.worst_eye_height),
            'worst_eye_width': float(eye_results.worst_eye_width)
        }
        
    except Exception as e:
        raise AnalysisError(f"Signal analysis failed: {e}")
```

## See Also

- [224G Ethernet API](eth_224g.md)
- [PAM4 Analysis Tutorial](../tutorials/pam4_analysis.md)