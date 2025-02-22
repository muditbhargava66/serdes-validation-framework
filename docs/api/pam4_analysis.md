# PAM4 Signal Analysis API Documentation

## Overview

The PAM4 Analysis module provides comprehensive tools for analyzing PAM4 signals in high-speed serial interfaces, with particular focus on 224G Ethernet applications.

## Core Components

### PAM4Analyzer

The main class for PAM4 signal analysis.

```python
from serdes_validation_framework.data_analysis.pam4_analyzer import PAM4Analyzer

# Initialize analyzer
analyzer = PAM4Analyzer({
    'time': time_array,
    'voltage': voltage_array
})

# Analyze signal
levels = analyzer.analyze_level_separation('voltage')
evm = analyzer.calculate_evm('voltage', 'time')
eye = analyzer.analyze_eye_diagram('voltage', 'time')
```

#### Key Methods

##### analyze_level_separation
```python
def analyze_level_separation(
    self,
    voltage_column: str,
    threshold: float = 0.1
) -> PAM4Levels:
    """
    Analyze PAM4 voltage level separation
    
    Args:
        voltage_column: Name of voltage data column
        threshold: Detection threshold (0-1)
    
    Returns:
        PAM4Levels object with analysis results
    """
```

##### calculate_evm
```python
def calculate_evm(
    self,
    measured_column: str,
    timestamp_column: str
) -> EVMResults:
    """
    Calculate Error Vector Magnitude
    
    Args:
        measured_column: Name of measured signal column
        timestamp_column: Name of timestamp column
    
    Returns:
        EVMResults object with EVM calculations
    """
```

##### analyze_eye_diagram
```python
def analyze_eye_diagram(
    self,
    voltage_column: str,
    time_column: str,
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
```

### Result Types

#### PAM4Levels
```python
@dataclass
class PAM4Levels:
    level_means: np.ndarray          # Measured voltage levels
    level_separations: np.ndarray    # Level separation distances
    uniformity: float                # Level spacing uniformity
```

#### EVMResults
```python
@dataclass
class EVMResults:
    rms_evm_percent: float          # RMS EVM percentage
    peak_evm_percent: float         # Peak EVM percentage
```

#### EyeResults
```python
@dataclass
class EyeResults:
    eye_heights: List[float]        # Eye opening heights
    eye_widths: List[float]         # Eye opening widths
    worst_eye_height: float         # Minimum eye height
    worst_eye_width: float          # Minimum eye width
```

## Signal Processing

### Level Detection

The analyzer uses histogram-based level detection:

```python
def _find_voltage_levels(
    self,
    histogram: np.ndarray,
    bin_edges: np.ndarray
) -> np.ndarray:
    """Find PAM4 voltage levels from histogram"""
```

### Signal Normalization

Signals are normalized to standard PAM4 levels:

```python
def _normalize_signal(
    self,
    signal: np.ndarray
) -> np.ndarray:
    """Normalize signal to [-3, -1, 1, 3] levels"""
```

### Eye Analysis

Eye diagram analysis includes:
- Height measurement at each level
- Width measurement at crossings
- Jitter decomposition

## Type Safety

All numeric inputs are strictly validated:

```python
# Correct usage
analyzer = PAM4Analyzer({
    'time': np.array([0.0, 1.0, 2.0], dtype=np.float64),
    'voltage': np.array([-1.0, 1.0, -1.0], dtype=np.float64)
})

# Will raise AssertionError
analyzer = PAM4Analyzer({
    'time': [0, 1, 2],  # Integer list instead of float array
    'voltage': [-1, 1, -1]
})
```

## Best Practices

### Data Preparation

1. Use numpy arrays with float64 dtype:
   ```python
   data = np.array(signal, dtype=np.float64)
   ```

2. Ensure adequate sampling rate:
   ```python
   sample_rate = 256e9  # 256 GSa/s recommended
   ```

3. Remove DC offset:
   ```python
   signal = signal - np.mean(signal)
   ```

### Analysis Configuration

1. Set appropriate thresholds:
   ```python
   levels = analyzer.analyze_level_separation(
       'voltage',
       threshold=0.1  # 10% of level spacing
   )
   ```

2. Use correct UI period:
   ```python
   eye = analyzer.analyze_eye_diagram(
       'voltage',
       'time',
       ui_period=8.9e-12  # 224G PAM4
   )
   ```

### Error Handling

Always handle potential errors:

```python
try:
    results = analyzer.analyze_level_separation('voltage')
except AssertionError as e:
    print(f"Input validation failed: {e}")
except Exception as e:
    print(f"Analysis failed: {e}")
```

## Example Usage

### Basic Analysis
```python
# Create analyzer
analyzer = PAM4Analyzer({
    'time': time_data,
    'voltage': voltage_data
})

# Run analysis
level_results = analyzer.analyze_level_separation('voltage')
evm_results = analyzer.calculate_evm('voltage', 'time')
eye_results = analyzer.analyze_eye_diagram('voltage', 'time')

# Print results
print(f"Level uniformity: {level_results.uniformity:.3f}")
print(f"RMS EVM: {evm_results.rms_evm_percent:.2f}%")
print(f"Worst eye height: {eye_results.worst_eye_height:.3f}")
```

### Validation Example
```python
def validate_signal_quality(
    analyzer: PAM4Analyzer,
    max_evm: float = 5.0,
    min_eye: float = 0.3
) -> bool:
    """Validate signal quality metrics"""
    evm_results = analyzer.calculate_evm('voltage', 'time')
    eye_results = analyzer.analyze_eye_diagram('voltage', 'time')
    
    return (
        evm_results.rms_evm_percent <= max_evm and
        eye_results.worst_eye_height >= min_eye
    )
```

## See Also

- [224G Ethernet API Documentation](eth_224g.md)
- [PAM4 Analysis Tutorial](../tutorials/pam4_analysis.md)

---