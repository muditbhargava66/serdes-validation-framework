# Data Analysis Examples

This document provides comprehensive examples for data analysis using the SerDes Validation Framework v1.3.0, including advanced signal processing and PCIe integration.

## Overview

The data analysis examples demonstrate:
- Basic statistical analysis and signal processing
- Advanced PAM4 signal analysis with EVM calculation
- Signal quality analysis (SNR, RMS, crest factor)
- Eye diagram analysis with jitter decomposition
- Integration with PCIe 6.0 validation workflows
- Multiple data format support (CSV, NPY, TXT)

## Example Files

### Core Data Analysis Examples
- **`examples/data_analysis_example.py`** - Comprehensive data analysis demonstration
- **`scripts/data_analysis.py`** - Production data analysis script

### Related Examples
- **`examples/pcie_example.py`** - PCIe validation with integrated analysis
- **`examples/test_sequence_example.py`** - Test automation with analysis

## Quick Start Examples

### Basic Signal Analysis

```python
#!/usr/bin/env python3
"""Basic signal analysis example"""

import numpy as np
from serdes_validation_framework.data_analysis.analyzer import DataAnalyzer

# Generate sample data
sample_data = {
    'signal_strength': np.array([0.1, 0.5, 0.3, 0.7, 0.2, 0.4, 0.8], dtype=np.float64)
}

# Create analyzer
analyzer = DataAnalyzer(sample_data)

# Compute statistics
stats = analyzer.compute_statistics('signal_strength')
print(f"Statistics: {stats}")

# Generate histogram (if matplotlib available)
try:
    analyzer.plot_histogram('signal_strength')
    print("Histogram generated successfully")
except ImportError:
    print("Matplotlib not available for plotting")
```

### Advanced Signal Generation

```python
#!/usr/bin/env python3
"""Advanced signal generation and analysis"""

import numpy as np
from serdes_validation_framework.data_analysis.analyzer import DataAnalyzer

def generate_test_signals():
    """Generate comprehensive test signals"""
    # Parameters
    sample_rate = 100e9  # 100 GSa/s
    duration = 1e-6      # 1 microsecond
    num_samples = int(sample_rate * duration)
    
    # Time vector
    time = np.linspace(0, duration, num_samples, dtype=np.float64)
    
    # NRZ signal
    symbol_rate = 32e9
    symbols_per_sample = sample_rate / symbol_rate
    num_symbols = int(num_samples / symbols_per_sample)
    
    binary_data = np.random.choice([-1.0, 1.0], size=num_symbols)
    nrz_signal = np.repeat(binary_data, int(symbols_per_sample))[:num_samples]
    
    # Add realistic noise
    snr_db = 25.0
    signal_power = np.mean(nrz_signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
    nrz_signal += noise
    
    # PAM4 signal
    pam4_levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
    pam4_symbols = np.random.choice(pam4_levels, size=num_symbols)
    pam4_signal = np.repeat(pam4_symbols, int(symbols_per_sample))[:num_samples]
    
    # Add noise to PAM4
    pam4_signal_power = np.mean(pam4_signal**2)
    pam4_noise_power = pam4_signal_power / (10**(snr_db/10))
    pam4_noise = np.random.normal(0, np.sqrt(pam4_noise_power), num_samples)
    pam4_signal += pam4_noise
    
    return {
        'time': time,
        'nrz_signal': nrz_signal.astype(np.float64),
        'pam4_signal': pam4_signal.astype(np.float64)
    }

# Generate and analyze signals
signals = generate_test_signals()
analyzer = DataAnalyzer(signals)

# Analyze each signal type
for signal_type in ['nrz_signal', 'pam4_signal']:
    print(f"\n{signal_type.upper()} Analysis:")
    stats = analyzer.compute_statistics(signal_type)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
```

### PAM4 Signal Analysis

```python
#!/usr/bin/env python3
"""PAM4-specific signal analysis"""

import numpy as np
from serdes_validation_framework.data_analysis.pam4_analyzer import PAM4Analyzer

# Generate PAM4 test signal
def generate_pam4_signal(num_samples=10000, snr_db=25.0):
    """Generate PAM4 test signal"""
    time = np.linspace(0, num_samples/100e9, num_samples, dtype=np.float64)
    
    # PAM4 levels
    levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
    symbols = np.random.choice(levels, size=num_samples)
    
    # Add noise
    signal_power = np.mean(symbols**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
    voltage = symbols + noise
    
    return {'time': time, 'voltage': voltage.astype(np.float64)}

# Generate signal
pam4_data = generate_pam4_signal()

# Create PAM4 analyzer
analyzer = PAM4Analyzer(pam4_data)

# Level separation analysis
try:
    level_results = analyzer.analyze_level_separation()
    print(f"Level separation results: {level_results}")
except Exception as e:
    print(f"Level separation analysis failed: {e}")

# EVM calculation
try:
    evm_results = analyzer.calculate_evm()
    print(f"EVM results: {evm_results}")
except Exception as e:
    print(f"EVM calculation failed: {e}")

# Eye diagram analysis
try:
    eye_results = analyzer.analyze_eye_diagram()
    print(f"Eye diagram results: {eye_results}")
except Exception as e:
    print(f"Eye diagram analysis failed: {e}")
```

### Signal Quality Analysis

```python
#!/usr/bin/env python3
"""Signal quality analysis example"""

import numpy as np

def calculate_signal_quality(signal):
    """Calculate comprehensive signal quality metrics"""
    # Basic power measurements
    signal_power = np.mean(signal**2)
    voltage_rms = np.sqrt(signal_power)
    voltage_pp = np.max(signal) - np.min(signal)
    voltage_peak = np.max(np.abs(signal))
    
    # Crest factor
    crest_factor = voltage_peak / voltage_rms if voltage_rms > 0 else 0
    
    # SNR estimation
    try:
        from scipy import signal as scipy_signal
        
        # High-pass filter to isolate noise
        nyquist = 0.5
        high_cutoff = 0.8
        b, a = scipy_signal.butter(4, high_cutoff, btype='high')
        noise_estimate = scipy_signal.filtfilt(b, a, signal)
        noise_power = np.mean(noise_estimate**2)
        
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
    except ImportError:
        # Fallback SNR calculation
        noise_power = np.var(signal)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    return {
        'signal_power': signal_power,
        'voltage_rms': voltage_rms,
        'voltage_pp': voltage_pp,
        'voltage_peak': voltage_peak,
        'crest_factor': crest_factor,
        'snr_db': snr_db
    }

# Example usage
test_signal = np.random.normal(0, 1, 10000) + np.sin(2*np.pi*np.linspace(0, 1, 10000))
quality_metrics = calculate_signal_quality(test_signal)

print("Signal Quality Metrics:")
for metric, value in quality_metrics.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.6f}")
    else:
        print(f"  {metric}: {value}")
```

## File Format Support

### CSV File Analysis

```python
#!/usr/bin/env python3
"""CSV file analysis example"""

import pandas as pd
import numpy as np
from serdes_validation_framework.data_analysis.analyzer import DataAnalyzer

def analyze_csv_file(filepath):
    """Analyze data from CSV file"""
    try:
        # Load CSV data
        df = pd.read_csv(filepath)
        
        # Convert to analyzer format
        data = {col: df[col].values.astype(np.float64) for col in df.columns}
        
        # Create analyzer
        analyzer = DataAnalyzer(data)
        
        # Analyze each column
        results = {}
        for column in df.columns:
            print(f"\nAnalyzing column: {column}")
            stats = analyzer.compute_statistics(column)
            results[column] = stats
            
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
        
        return results
        
    except Exception as e:
        print(f"CSV analysis failed: {e}")
        return None

# Example usage
# results = analyze_csv_file('signal_data.csv')
```

### NumPy Array Analysis

```python
#!/usr/bin/env python3
"""NumPy array analysis example"""

import numpy as np
from serdes_validation_framework.data_analysis.analyzer import DataAnalyzer

def analyze_numpy_file(filepath):
    """Analyze data from NumPy file"""
    try:
        # Load NumPy data
        array_data = np.load(filepath)
        
        # Handle different array dimensions
        if array_data.ndim == 1:
            data = {'signal': array_data.astype(np.float64)}
        else:
            data = {f'channel_{i}': array_data[:, i].astype(np.float64) 
                   for i in range(array_data.shape[1])}
        
        # Create analyzer
        analyzer = DataAnalyzer(data)
        
        # Analyze each channel
        results = {}
        for channel_name in data.keys():
            print(f"\nAnalyzing {channel_name}:")
            stats = analyzer.compute_statistics(channel_name)
            results[channel_name] = stats
            
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
        
        return results
        
    except Exception as e:
        print(f"NumPy analysis failed: {e}")
        return None

# Example usage
# results = analyze_numpy_file('signal_data.npy')
```

## Advanced Analysis Examples

### Frequency Domain Analysis

```python
#!/usr/bin/env python3
"""Frequency domain analysis example"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_frequency_domain(signal, sample_rate):
    """Perform frequency domain analysis"""
    # Calculate FFT
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)
    
    # Calculate power spectral density
    psd = np.abs(fft_result)**2 / len(signal)
    
    # Find dominant frequencies
    positive_freqs = frequencies[:len(frequencies)//2]
    positive_psd = psd[:len(psd)//2]
    
    # Find peaks
    peak_indices = np.argsort(positive_psd)[-5:]  # Top 5 peaks
    dominant_freqs = positive_freqs[peak_indices]
    peak_powers = positive_psd[peak_indices]
    
    print("Frequency Domain Analysis:")
    print(f"  Sample Rate: {sample_rate/1e9:.1f} GSa/s")
    print(f"  Frequency Resolution: {frequencies[1]/1e6:.2f} MHz")
    print(f"  Dominant Frequencies:")
    
    for i, (freq, power) in enumerate(zip(dominant_freqs, peak_powers)):
        print(f"    {i+1}: {freq/1e9:.3f} GHz (Power: {10*np.log10(power):.1f} dB)")
    
    return {
        'frequencies': frequencies,
        'psd': psd,
        'dominant_frequencies': dominant_freqs,
        'peak_powers': peak_powers
    }

# Example usage
sample_rate = 100e9  # 100 GSa/s
test_signal = np.sin(2*np.pi*1e9*np.linspace(0, 1e-6, 100000))  # 1 GHz sine wave
freq_analysis = analyze_frequency_domain(test_signal, sample_rate)
```

### Statistical Analysis

```python
#!/usr/bin/env python3
"""Advanced statistical analysis example"""

import numpy as np
from scipy import stats

def advanced_statistical_analysis(signal):
    """Perform advanced statistical analysis"""
    # Basic statistics
    mean = np.mean(signal)
    std = np.std(signal)
    variance = np.var(signal)
    skewness = stats.skew(signal)
    kurtosis = stats.kurtosis(signal)
    
    # Distribution fitting
    try:
        # Fit normal distribution
        mu, sigma = stats.norm.fit(signal)
        
        # Kolmogorov-Smirnov test for normality
        ks_statistic, ks_p_value = stats.kstest(signal, 'norm', args=(mu, sigma))
        
        # Anderson-Darling test for normality
        ad_statistic, ad_critical_values, ad_significance_level = stats.anderson(signal, 'norm')
        
        print("Advanced Statistical Analysis:")
        print(f"  Mean: {mean:.6f}")
        print(f"  Standard Deviation: {std:.6f}")
        print(f"  Variance: {variance:.6f}")
        print(f"  Skewness: {skewness:.6f}")
        print(f"  Kurtosis: {kurtosis:.6f}")
        print(f"  Fitted Normal Distribution: μ={mu:.6f}, σ={sigma:.6f}")
        print(f"  KS Test p-value: {ks_p_value:.6f}")
        print(f"  AD Test statistic: {ad_statistic:.6f}")
        
        return {
            'mean': mean,
            'std': std,
            'variance': variance,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'fitted_mu': mu,
            'fitted_sigma': sigma,
            'ks_p_value': ks_p_value,
            'ad_statistic': ad_statistic
        }
        
    except ImportError:
        print("SciPy not available for advanced statistics")
        return {
            'mean': mean,
            'std': std,
            'variance': variance
        }

# Example usage
test_signal = np.random.normal(0, 1, 10000)
stats_results = advanced_statistical_analysis(test_signal)
```

## Integration with PCIe Analysis

### Combined PCIe and Data Analysis

```python
#!/usr/bin/env python3
"""Combined PCIe and data analysis example"""

from serdes_validation_framework.protocols.pcie.constants import SignalMode
from serdes_validation_framework.instrument_control.pcie_analyzer import PCIeAnalyzer, PCIeConfig
from serdes_validation_framework.data_analysis.analyzer import DataAnalyzer

def comprehensive_pcie_analysis(signal_data, mode):
    """Perform comprehensive PCIe and data analysis"""
    
    # PCIe-specific analysis
    config = PCIeConfig(
        mode=mode,
        sample_rate=200e9 if mode == SignalMode.PAM4 else 100e9,
        bandwidth=100e9 if mode == SignalMode.PAM4 else 50e9,
        voltage_range=1.2 if mode == SignalMode.PAM4 else 1.0,
        link_speed=64e9,
        lane_count=1
    )
    
    pcie_analyzer = PCIeAnalyzer(config)
    pcie_results = pcie_analyzer.analyze_signal(signal_data)
    
    # General data analysis
    data_analyzer = DataAnalyzer(signal_data)
    data_stats = data_analyzer.compute_statistics('voltage')
    
    # Signal quality analysis
    voltage = signal_data['voltage']
    quality_metrics = calculate_signal_quality(voltage)
    
    # Combined results
    combined_results = {
        'pcie_analysis': pcie_results,
        'data_statistics': data_stats,
        'signal_quality': quality_metrics
    }
    
    print(f"Comprehensive {mode.name} Analysis:")
    print(f"  PCIe SNR: {pcie_results.get('snr_db', 0):.1f} dB")
    print(f"  Data Mean: {data_stats.get('mean', 0):.6f}")
    print(f"  Signal Power: {quality_metrics.get('signal_power', 0):.6f}")
    
    return combined_results

# Example usage with generated PCIe signal
# signal_data = generate_pcie_signal(SignalMode.PAM4)
# results = comprehensive_pcie_analysis(signal_data, SignalMode.PAM4)
```

## Running the Examples

### Command Line Usage

```bash
# Run comprehensive data analysis example
python examples/data_analysis_example.py

# Run with verbose output
python examples/data_analysis_example.py --verbose

# Run production data analysis script
python scripts/data_analysis.py --verbose

# Analyze custom data file
python scripts/data_analysis.py --input data.csv --output ./results
```

### Script Parameters

```bash
# Data analysis script options
python scripts/data_analysis.py [OPTIONS]

Options:
  --input FILE     Input data file (CSV, NPY, or TXT format)
  --output DIR     Output directory for results (default: ./analysis_output)
  --verbose        Enable verbose logging
```

### Mock Mode Testing

```bash
# All examples work without hardware
python examples/data_analysis_example.py

# Generate sample data automatically
python scripts/data_analysis.py --verbose
```

## Performance Optimization

### Efficient Data Processing

```python
#!/usr/bin/env python3
"""Efficient data processing example"""

import numpy as np
from numba import jit

@jit(nopython=True)
def fast_statistics(data):
    """Fast statistics calculation using Numba"""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    minimum = np.min(data)
    maximum = np.max(data)
    
    return mean, std, minimum, maximum

def optimized_analysis(large_signal):
    """Optimized analysis for large signals"""
    # Process in chunks for memory efficiency
    chunk_size = 10000
    num_chunks = len(large_signal) // chunk_size
    
    chunk_stats = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = large_signal[start_idx:end_idx]
        
        mean, std, minimum, maximum = fast_statistics(chunk)
        chunk_stats.append({
            'mean': mean,
            'std': std,
            'min': minimum,
            'max': maximum
        })
    
    # Combine chunk statistics
    overall_mean = np.mean([stat['mean'] for stat in chunk_stats])
    overall_std = np.mean([stat['std'] for stat in chunk_stats])
    overall_min = min([stat['min'] for stat in chunk_stats])
    overall_max = max([stat['max'] for stat in chunk_stats])
    
    return {
        'mean': overall_mean,
        'std': overall_std,
        'min': overall_min,
        'max': overall_max,
        'chunks_processed': num_chunks
    }

# Example usage for large datasets
# large_signal = np.random.randn(1000000)  # 1M samples
# results = optimized_analysis(large_signal)
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Files**
   ```python
   # Process data in chunks
   def process_large_file(filepath, chunk_size=10000):
       # Implementation for chunked processing
       pass
   ```

2. **Missing Dependencies**
   ```bash
   # Install optional dependencies
   pip install matplotlib scipy pandas numba
   ```

3. **File Format Issues**
   ```python
   # Check file format before processing
   def detect_file_format(filepath):
       # Implementation for format detection
       pass
   ```

### Debug Mode

```bash
# Enable verbose logging
python examples/data_analysis_example.py --verbose

# Check for detailed error messages
python scripts/data_analysis.py --input problematic_file.csv --verbose
```

## See Also

- [Data Analysis API Reference](../api/data_analysis.md)
- [PAM4 Analysis Documentation](../api/pam4_analysis.md)
- [PCIe Examples](pcie_examples.md)
- Scripts Documentation