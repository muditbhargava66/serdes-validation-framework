# PAM4 Signal Analysis Tutorial

## Overview

This tutorial demonstrates:
- PAM4 signal analysis techniques
- Level separation measurements
- EVM calculations
- Eye diagram analysis
- Type-safe data processing

## Basic Setup

### Required Imports

```python
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import numpy.typing as npt
from serdes_validation_framework.data_analysis.pam4_analyzer import (
    PAM4Analyzer,
    PAM4Levels,
    EVMResults,
    EyeResults
)

# Type aliases
FloatArray = npt.NDArray[np.float64]
SignalData = Dict[str, FloatArray]
```

### Data Validation

```python
def validate_signal_data(
    data: SignalData,
    required_columns: List[str]
) -> None:
    """
    Validate signal data structure
    
    Args:
        data: Signal data dictionary
        required_columns: List of required column names
        
    Raises:
        AssertionError: If validation fails
    """
    # Type validation
    assert isinstance(data, dict), \
        f"Data must be dictionary, got {type(data)}"
    
    # Check required columns
    missing = set(required_columns) - set(data.keys())
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Validate arrays
    for column in required_columns:
        array = data[column]
        
        # Type check
        assert isinstance(array, np.ndarray), \
            f"Column {column} must be numpy array, got {type(array)}"
        
        # Data type check
        assert np.issubdtype(array.dtype, np.floating), \
            f"Column {column} must be floating-point, got {array.dtype}"
        
        # Value validation
        assert len(array) > 0, f"Column {column} cannot be empty"
        assert not np.any(np.isnan(array)), \
            f"Column {column} contains NaN values"
        assert not np.any(np.isinf(array)), \
            f"Column {column} contains infinite values"
```

## Signal Generation

### Test Signal Creation

```python
def generate_test_signal(
    duration: float = 1e-6,
    sample_rate: float = 256e9,
    noise_amplitude: float = 0.05
) -> SignalData:
    """
    Generate synthetic PAM4 test signal
    
    Args:
        duration: Signal duration in seconds
        sample_rate: Sampling rate in Hz
        noise_amplitude: Noise amplitude (0-1)
        
    Returns:
        Dictionary with time and voltage arrays
        
    Raises:
        AssertionError: If parameters are invalid
    """
    # Validate inputs
    assert isinstance(duration, float), "Duration must be float"
    assert isinstance(sample_rate, float), "Sample rate must be float"
    assert isinstance(noise_amplitude, float), "Noise amplitude must be float"
    assert duration > 0, "Duration must be positive"
    assert sample_rate > 0, "Sample rate must be positive"
    assert 0 <= noise_amplitude <= 1, "Noise amplitude must be between 0 and 1"
    
    try:
        # Calculate points
        num_points = int(duration * sample_rate)
        
        # Generate time array
        time = np.arange(num_points, dtype=np.float64) / sample_rate
        
        # Generate PAM4 levels
        levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
        symbols = np.random.choice(levels, size=num_points)
        
        # Add noise
        noise = np.random.normal(0, noise_amplitude, num_points)
        voltage = (symbols + noise).astype(np.float64)
        
        return {
            'time': time,
            'voltage': voltage
        }
        
    except Exception as e:
        raise RuntimeError(f"Signal generation failed: {e}")
```

## Level Analysis

### Basic Level Analysis

```python
def analyze_signal_levels(
    voltage_data: FloatArray,
    threshold: float = 0.1
) -> Tuple[PAM4Levels, Dict[str, float]]:
    """
    Analyze PAM4 voltage levels
    
    Args:
        voltage_data: Voltage measurements array
        threshold: Detection threshold
        
    Returns:
        Tuple of (PAM4Levels, statistics)
    """
    # Validate inputs
    assert isinstance(voltage_data, np.ndarray), \
        "Voltage data must be numpy array"
    assert isinstance(threshold, float), "Threshold must be float"
    assert np.issubdtype(voltage_data.dtype, np.floating), \
        "Voltage data must be floating-point"
    assert 0 < threshold < 1, "Threshold must be between 0 and 1"
    
    try:
        # Create analyzer
        analyzer = PAM4Analyzer({'voltage': voltage_data})
        
        # Analyze levels
        levels = analyzer.analyze_level_separation('voltage')
        
        # Calculate statistics
        stats = {
            'mean_separation': float(np.mean(levels.level_separations)),
            'min_separation': float(np.min(levels.level_separations)),
            'uniformity': float(levels.uniformity)
        }
        
        return levels, stats
        
    except Exception as e:
        raise RuntimeError(f"Level analysis failed: {e}")
```

### Level Quality Checks

```python
def check_level_quality(
    levels: PAM4Levels,
    min_separation: float = 0.5,
    max_uniformity: float = 0.2
) -> Dict[str, bool]:
    """
    Check PAM4 level quality metrics
    
    Args:
        levels: PAM4 level measurements
        min_separation: Minimum level separation
        max_uniformity: Maximum uniformity metric
        
    Returns:
        Dictionary of quality checks
    """
    # Validate inputs
    assert isinstance(min_separation, float), \
        "Minimum separation must be float"
    assert isinstance(max_uniformity, float), \
        "Maximum uniformity must be float"
    assert min_separation > 0, "Minimum separation must be positive"
    assert max_uniformity > 0, "Maximum uniformity must be positive"
    
    try:
        return {
            'level_separation': all(sep >= min_separation 
                                  for sep in levels.level_separations),
            'level_uniformity': levels.uniformity <= max_uniformity,
            'monotonic_levels': np.all(np.diff(levels.level_means) > 0)
        }
        
    except Exception as e:
        raise RuntimeError(f"Quality check failed: {e}")
```

## EVM Analysis

### EVM Calculation

```python
def calculate_signal_evm(
    measured: FloatArray,
    reference: Optional[FloatArray] = None
) -> EVMResults:
    """
    Calculate Error Vector Magnitude
    
    Args:
        measured: Measured signal array
        reference: Optional reference signal array
        
    Returns:
        EVMResults object with calculations
    """
    # Validate measured signal
    assert isinstance(measured, np.ndarray), \
        "Measured signal must be numpy array"
    assert np.issubdtype(measured.dtype, np.floating), \
        "Measured signal must be floating-point"
    
    # Validate reference if provided
    if reference is not None:
        assert isinstance(reference, np.ndarray), \
            "Reference signal must be numpy array"
        assert np.issubdtype(reference.dtype, np.floating), \
            "Reference signal must be floating-point"
        assert len(reference) == len(measured), \
            "Signal lengths must match"
    
    try:
        # Create analyzer
        analyzer = PAM4Analyzer({'voltage': measured})
        
        # Calculate EVM
        if reference is not None:
            analyzer.data['reference'] = reference
            return analyzer.calculate_evm('voltage', 'reference')
        else:
            return analyzer.calculate_evm('voltage')
            
    except Exception as e:
        raise RuntimeError(f"EVM calculation failed: {e}")
```

### EVM Quality Checks

```python
def check_evm_quality(
    evm_results: EVMResults,
    max_rms_evm: float = 5.0,
    max_peak_evm: float = 10.0
) -> Dict[str, bool]:
    """
    Check EVM quality metrics
    
    Args:
        evm_results: EVM measurement results
        max_rms_evm: Maximum RMS EVM percentage
        max_peak_evm: Maximum peak EVM percentage
        
    Returns:
        Dictionary of quality checks
    """
    # Validate inputs
    assert isinstance(max_rms_evm, float), "Maximum RMS EVM must be float"
    assert isinstance(max_peak_evm, float), "Maximum peak EVM must be float"
    assert max_rms_evm > 0, "Maximum RMS EVM must be positive"
    assert max_peak_evm > 0, "Maximum peak EVM must be positive"
    assert max_peak_evm >= max_rms_evm, "Peak EVM limit must exceed RMS limit"
    
    return {
        'rms_evm': evm_results.rms_evm_percent <= max_rms_evm,
        'peak_evm': evm_results.peak_evm_percent <= max_peak_evm
    }
```

## Eye Analysis

### Eye Measurements

```python
def analyze_eye_diagram(
    time_data: FloatArray,
    voltage_data: FloatArray,
    ui_period: float = 8.9e-12
) -> EyeResults:
    """
    Analyze eye diagram parameters
    
    Args:
        time_data: Time points array
        voltage_data: Voltage measurements array
        ui_period: Unit interval in seconds
        
    Returns:
        EyeResults object with measurements
    """
    # Validate inputs
    assert isinstance(ui_period, float), "UI period must be float"
    assert ui_period > 0, "UI period must be positive"
    
    # Validate arrays
    validate_signal_data(
        {'time': time_data, 'voltage': voltage_data},
        ['time', 'voltage']
    )
    
    try:
        # Create analyzer
        analyzer = PAM4Analyzer({
            'time': time_data,
            'voltage': voltage_data
        })
        
        # Analyze eye diagram
        return analyzer.analyze_eye_diagram(
            'voltage',
            'time',
            ui_period=ui_period
        )
        
    except Exception as e:
        raise RuntimeError(f"Eye analysis failed: {e}")
```

### Eye Quality Checks

```python
def check_eye_quality(
    eye_results: EyeResults,
    min_height: float = 0.2,
    min_width: float = 0.4
) -> Dict[str, bool]:
    """
    Check eye quality metrics
    
    Args:
        eye_results: Eye measurement results
        min_height: Minimum eye height
        min_width: Minimum eye width
        
    Returns:
        Dictionary of quality checks
    """
    # Validate inputs
    assert isinstance(min_height, float), "Minimum height must be float"
    assert isinstance(min_width, float), "Minimum width must be float"
    assert min_height > 0, "Minimum height must be positive"
    assert min_width > 0, "Minimum width must be positive"
    
    return {
        'eye_height': eye_results.worst_eye_height >= min_height,
        'eye_width': eye_results.worst_eye_width >= min_width,
        'uniform_eyes': len(set(eye_results.eye_heights)) == 3
    }
```

## Complete Example

### Full Analysis Pipeline

```python
def analyze_pam4_signal() -> None:
    """Demonstrate complete PAM4 analysis pipeline"""
    try:
        # 1. Generate test signal
        signal = generate_test_signal(
            duration=1e-6,      # 1 µs
            sample_rate=256e9,  # 256 GSa/s
            noise_amplitude=0.05
        )
        
        # 2. Analyze levels
        levels, level_stats = analyze_signal_levels(
            signal['voltage']
        )
        level_quality = check_level_quality(levels)
        
        print("\nLevel Analysis:")
        print(f"Uniformity: {level_stats['uniformity']:.3f}")
        print(f"Min Separation: {level_stats['min_separation']:.3f}")
        print("Quality Checks:", level_quality)
        
        # 3. Calculate EVM
        evm = calculate_signal_evm(signal['voltage'])
        evm_quality = check_evm_quality(evm)
        
        print("\nEVM Analysis:")
        print(f"RMS EVM: {evm.rms_evm_percent:.2f}%")
        print(f"Peak EVM: {evm.peak_evm_percent:.2f}%")
        print("Quality Checks:", evm_quality)
        
        # 4. Analyze eye
        eye = analyze_eye_diagram(
            signal['time'],
            signal['voltage']
        )
        eye_quality = check_eye_quality(eye)
        
        print("\nEye Analysis:")
        print(f"Worst Height: {eye.worst_eye_height:.3f}")
        print(f"Worst Width: {eye.worst_eye_width:.3f}")
        print("Quality Checks:", eye_quality)
        
        # 5. Overall quality
        overall_quality = all([
            all(level_quality.values()),
            all(evm_quality.values()),
            all(eye_quality.values())
        ])
        
        print(f"\nOverall Quality: {'PASS' if overall_quality else 'FAIL'}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
```

## Best Practices

### 1. Array Validation

```python
def validate_array(
    array: FloatArray,
    name: str = "array"
) -> None:
    """Validate numeric array"""
    assert isinstance(array, np.ndarray), \
        f"{name} must be numpy array"
    assert np.issubdtype(array.dtype, np.floating), \
        f"{name} must be floating-point"
    assert len(array) > 0, \
        f"{name} cannot be empty"
    assert not np.any(np.isnan(array)), \
        f"{name} contains NaN values"
```

### 2. Parameter Validation

```python
def validate_parameter(
    value: float,
    name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> None:
    """Validate numeric parameter"""
    assert isinstance(value, float), \
        f"{name} must be float"
    if min_value is not None:
        assert value >= min_value, \
            f"{name} must be >= {min_value}"
    if max_value is not None:
        assert value <= max_value, \
            f"{name} must be <= {max_value}"
```

### 3. Result Validation

```python
def validate_measurements(
    results: Dict[str, float],
    limits: Dict[str, float]
) -> bool:
    """
    Validate measurement results against limits
    
    Args:
        results: Dictionary of measurements
        limits: Dictionary of measurement limits
        
    Returns:
        True if all measurements pass
        
    Raises:
        ValueError: If validation fails
    """
    try:
        for name, value in results.items():
            limit = limits.get(name)
            if limit is not None:
                if not isinstance(value, float):
                    raise ValueError(f"Measurement {name} must be float")
                if value > limit:
                    return False
        return True
        
    except Exception as e:
        raise ValueError(f"Measurement validation failed: {e}")
```

### 4. Error Handling

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

def handle_analysis_errors(func):
    """Decorator for handling analysis errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            print(f"Validation Error: {e}")
            raise
        except MeasurementError as e:
            print(f"Measurement Error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected Error: {e}")
            raise
    return wrapper
```

## Advanced Topics

### Custom Analysis Pipeline

```python
@handle_analysis_errors
def custom_analysis_pipeline(
    data: SignalData,
    config: Dict[str, float]
) -> Dict[str, Any]:
    """
    Create custom analysis pipeline
    
    Args:
        data: Signal data dictionary
        config: Analysis configuration
        
    Returns:
        Analysis results dictionary
    """
    # Validate inputs
    validate_signal_data(data, ['time', 'voltage'])
    required_config = {'min_eye_height', 'max_evm', 'min_snr'}
    missing = required_config - set(config.keys())
    if missing:
        raise ValidationError(f"Missing configuration: {missing}")
    
    try:
        # Create analyzer
        analyzer = PAM4Analyzer(data)
        
        # Step 1: Level analysis
        levels = analyzer.analyze_level_separation('voltage')
        level_quality = check_level_quality(
            levels,
            min_separation=config.get('min_separation', 0.5)
        )
        
        # Step 2: EVM calculation
        evm = analyzer.calculate_evm('voltage')
        evm_quality = check_evm_quality(
            evm,
            max_rms_evm=config['max_evm']
        )
        
        # Step 3: Eye analysis
        eye = analyzer.analyze_eye_diagram(
            'voltage',
            'time'
        )
        eye_quality = check_eye_quality(
            eye,
            min_height=config['min_eye_height']
        )
        
        # Compile results
        return {
            'levels': {
                'measurements': levels,
                'quality': level_quality
            },
            'evm': {
                'measurements': evm,
                'quality': evm_quality
            },
            'eye': {
                'measurements': eye,
                'quality': eye_quality
            },
            'overall_quality': all([
                all(level_quality.values()),
                all(evm_quality.values()),
                all(eye_quality.values())
            ])
        }
        
    except Exception as e:
        raise MeasurementError(f"Analysis pipeline failed: {e}")
```

### Signal Processing Utilities

```python
def process_signal(
    signal: FloatArray,
    sample_rate: float
) -> FloatArray:
    """
    Apply signal processing with validation
    
    Args:
        signal: Input signal array
        sample_rate: Sample rate in Hz
        
    Returns:
        Processed signal array
    """
    # Validate inputs
    validate_array(signal, "signal")
    assert isinstance(sample_rate, float), "Sample rate must be float"
    assert sample_rate > 0, "Sample rate must be positive"
    
    try:
        # 1. Remove DC offset
        signal = signal - np.mean(signal)
        
        # 2. Apply filtering
        nyq = sample_rate / 2
        b, a = scipy.signal.butter(4, 0.1)
        filtered = scipy.signal.filtfilt(b, a, signal)
        
        # 3. Normalize amplitude
        normalized = filtered / np.max(np.abs(filtered))
        
        return normalized.astype(np.float64)
        
    except Exception as e:
        raise ValueError(f"Signal processing failed: {e}")
```

### Advanced Visualization

```python
def plot_analysis_results(
    data: SignalData,
    results: Dict[str, Any]
) -> None:
    """
    Create comprehensive visualization
    
    Args:
        data: Signal data dictionary
        results: Analysis results dictionary
    """
    try:
        # Set up plotting
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Signal trace
        ax1.plot(data['time'][:1000], data['voltage'][:1000], 'b-', alpha=0.7)
        ax1.set_title('Signal Trace')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Voltage (V)')
        ax1.grid(True)
        
        # 2. Level histogram
        hist, bins = np.histogram(data['voltage'], bins=100)
        ax2.hist(data['voltage'], bins=100, density=True, alpha=0.7)
        for level in results['levels']['measurements'].level_means:
            ax2.axvline(level, color='r', linestyle='--', alpha=0.5)
        ax2.set_title('Level Distribution')
        ax2.set_xlabel('Voltage (V)')
        ax2.set_ylabel('Density')
        
        # 3. Eye diagram
        t_norm = (data['time'] % 8.9e-12) / 8.9e-12
        ax3.plot(t_norm, data['voltage'], 'b.', alpha=0.1, markersize=1)
        ax3.set_title('Eye Diagram')
        ax3.set_xlabel('UI')
        ax3.set_ylabel('Voltage (V)')
        
        # 4. Quality metrics
        metrics = [
            f"RMS EVM: {results['evm']['measurements'].rms_evm_percent:.2f}%",
            f"Eye Height: {results['eye']['measurements'].worst_eye_height:.3f}",
            f"Level Uniformity: {results['levels']['measurements'].uniformity:.3f}",
            f"Overall: {'PASS' if results['overall_quality'] else 'FAIL'}"
        ]
        ax4.text(0.1, 0.5, '\n'.join(metrics), transform=ax4.transAxes)
        ax4.set_title('Quality Metrics')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Plotting failed: {e}")
```

## See Also

- [Mock Testing Tutorial](mock_testing.md)
- [224G Validation Tutorial](224g_validation.md)
- [PAM4 Analysis API](../api/pam4_analysis.md)

## References

- PAM4 Modulation Theory
- Signal Processing Handbook
- IEEE 802.3 Specifications