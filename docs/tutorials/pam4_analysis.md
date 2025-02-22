# PAM4 Signal Analysis Tutorial

## Introduction

This tutorial guides you through the process of analyzing PAM4 signals using the SerDes Validation Framework. We'll cover signal acquisition, analysis techniques, and result interpretation.

## Prerequisites

- Python 3.10 or higher
- SerDes Validation Framework installed
- NumPy and Matplotlib
- Basic understanding of PAM4 modulation

## Basic Signal Analysis

### Setting Up the Analyzer

1. Import required modules:
   ```python
   from serdes_validation_framework.data_analysis.pam4_analyzer import (
       PAM4Analyzer,
       PAM4Levels,
       EVMResults,
       EyeResults
   )
   import numpy as np
   import matplotlib.pyplot as plt
   ```

2. Create test data:
   ```python
   def generate_test_signal(
       duration: float = 1e-9,    # 1 ns
       sample_rate: float = 256e9  # 256 GSa/s
   ):
       # Generate time array
       time = np.arange(0, duration, 1/sample_rate, dtype=np.float64)
       
       # Generate PAM4 symbols
       levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
       symbols = np.random.choice(levels, size=len(time))
       
       # Add noise
       noise = np.random.normal(0, 0.1, len(time))
       voltage = symbols + noise
       
       return {
           'time': time,
           'voltage': voltage.astype(np.float64)
       }
   
   # Generate test signal
   signal_data = generate_test_signal()
   ```

3. Initialize analyzer:
   ```python
   analyzer = PAM4Analyzer(signal_data)
   ```

### Level Analysis

1. Analyze PAM4 levels:
   ```python
   # Run level analysis
   level_results = analyzer.analyze_level_separation('voltage')
   
   # Print results
   print("PAM4 Level Analysis:")
   print(f"Level means: {level_results.level_means}")
   print(f"Level separations: {level_results.level_separations}")
   print(f"Uniformity: {level_results.uniformity:.3f}")
   ```

2. Visualize levels:
   ```python
   def plot_level_histogram(voltage_data, level_means):
       plt.figure(figsize=(10, 6))
       plt.hist(voltage_data, bins=100, density=True)
       for level in level_means:
           plt.axvline(level, color='r', linestyle='--')
       plt.title('PAM4 Level Distribution')
       plt.xlabel('Voltage')
       plt.ylabel('Density')
       plt.grid(True)
       plt.show()
   
   plot_level_histogram(signal_data['voltage'], level_results.level_means)
   ```

### EVM Calculations

1. Calculate EVM:
   ```python
   # Calculate EVM
   evm_results = analyzer.calculate_evm('voltage', 'time')
   
   # Print results
   print("\nEVM Analysis:")
   print(f"RMS EVM: {evm_results.rms_evm_percent:.2f}%")
   print(f"Peak EVM: {evm_results.peak_evm_percent:.2f}%")
   ```

2. Visualize error distribution:
   ```python
   def plot_evm_distribution(analyzer, voltage_data):
       # Get ideal symbols
       ideal = analyzer._find_nearest_level(
           analyzer._normalize_signal(voltage_data)
       )
       
       # Calculate errors
       errors = voltage_data - ideal
       
       plt.figure(figsize=(10, 6))
       plt.hist(errors, bins=50, density=True)
       plt.title('Error Vector Distribution')
       plt.xlabel('Error Magnitude')
       plt.ylabel('Density')
       plt.grid(True)
       plt.show()
   
   plot_evm_distribution(analyzer, signal_data['voltage'])
   ```

### Eye Diagram Analysis

1. Analyze eye diagram:
   ```python
   # Analyze eye diagram
   eye_results = analyzer.analyze_eye_diagram(
       'voltage',
       'time',
       ui_period=8.9e-12  # 224G PAM4
   )
   
   # Print results
   print("\nEye Diagram Analysis:")
   print(f"Eye heights: {eye_results.eye_heights}")
   print(f"Eye widths: {eye_results.eye_widths}")
   print(f"Worst eye height: {eye_results.worst_eye_height:.3f}")
   print(f"Worst eye width: {eye_results.worst_eye_width:.3f}")
   ```

2. Plot eye diagram:
   ```python
   def plot_eye_diagram(time_data, voltage_data, ui_period):
       # Fold signal into eye diagram
       t_norm = (time_data % ui_period) / ui_period
       
       plt.figure(figsize=(10, 6))
       plt.plot(t_norm, voltage_data, 'b.', alpha=0.1, markersize=1)
       plt.title('Eye Diagram')
       plt.xlabel('UI')
       plt.ylabel('Voltage')
       plt.grid(True)
       plt.show()
   
   plot_eye_diagram(
       signal_data['time'],
       signal_data['voltage'],
       8.9e-12
   )
   ```

## Advanced Analysis

### Signal Quality Metrics

1. Define quality checks:
   ```python
   def check_signal_quality(
       level_results: PAM4Levels,
       evm_results: EVMResults,
       eye_results: EyeResults
   ) -> bool:
       # Check level spacing
       if level_results.uniformity > 0.2:
           return False
       
       # Check EVM
       if evm_results.rms_evm_percent > 5.0:
           return False
       
       # Check eye opening
       if eye_results.worst_eye_height < 0.2:
           return False
       
       return True
   ```

2. Use quality metrics:
   ```python
   # Check signal quality
   quality_ok = check_signal_quality(
       level_results,
       evm_results,
       eye_results
   )
   print(f"\nSignal quality check: {'PASS' if quality_ok else 'FAIL'}")
   ```

### Advanced Signal Processing

1. Custom filtering:
   ```python
   def apply_signal_filtering(voltage_data: np.ndarray) -> np.ndarray:
       """
       Apply custom filtering to voltage data
       
       Args:
           voltage_data: Raw voltage measurements
           
       Returns:
           Filtered voltage data
       """
       from scipy import signal
       
       # Design filter
       nyq = 0.5 * 256e9  # Half of sample rate
       cutoff = 112e9     # 112 GHz cutoff
       b, a = signal.butter(4, cutoff/nyq)
       
       # Apply filter
       filtered = signal.filtfilt(b, a, voltage_data)
       return filtered.astype(np.float64)
   
   # Use filtered data
   filtered_data = {
       'time': signal_data['time'],
       'voltage': apply_signal_filtering(signal_data['voltage'])
   }
   analyzer_filtered = PAM4Analyzer(filtered_data)
   ```

2. Advanced jitter analysis:
   ```python
   def analyze_jitter(time_data: np.ndarray, voltage_data: np.ndarray) -> Dict[str, float]:
       """
       Perform detailed jitter analysis
       
       Args:
           time_data: Time stamps
           voltage_data: Voltage measurements
           
       Returns:
           Dictionary of jitter components
       """
       # Find zero crossings
       zero_crossings = np.where(np.diff(np.signbit(voltage_data)))[0]
       
       # Calculate period jitter
       periods = np.diff(time_data[zero_crossings])
       
       return {
           'rj': float(np.std(periods)),
           'pj': float(np.ptp(periods)),
           'tj': float(np.std(periods) * 14 + np.ptp(periods))
       }
   
   jitter_results = analyze_jitter(signal_data['time'], signal_data['voltage'])
   print("\nJitter Analysis:")
   for jitter_type, value in jitter_results.items():
       print(f"{jitter_type.upper()}: {value*1e12:.2f} ps")
   ```

### Batch Processing

1. Create batch analyzer:
   ```python
   class BatchPAM4Analyzer:
       """Process multiple PAM4 signals in batch"""
       
       def __init__(self, signals: List[Dict[str, np.ndarray]]):
           self.signals = signals
           self.analyzers = [PAM4Analyzer(signal) for signal in signals]
       
       def analyze_batch(self) -> List[Dict[str, Any]]:
           results = []
           for analyzer in self.analyzers:
               result = {
                   'levels': analyzer.analyze_level_separation('voltage'),
                   'evm': analyzer.calculate_evm('voltage', 'time'),
                   'eye': analyzer.analyze_eye_diagram('voltage', 'time')
               }
               results.append(result)
           return results
   
   # Use batch analyzer
   test_signals = [generate_test_signal() for _ in range(5)]
   batch_analyzer = BatchPAM4Analyzer(test_signals)
   batch_results = batch_analyzer.analyze_batch()
   ```

### Statistical Analysis

1. Track signal quality over time:
   ```python
   def track_signal_metrics(results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
       """
       Track signal metrics across multiple measurements
       
       Args:
           results: List of analysis results
           
       Returns:
           Dictionary of metric histories
       """
       metrics = {
           'evm_rms': [],
           'eye_height': [],
           'level_uniformity': []
       }
       
       for result in results:
           metrics['evm_rms'].append(result['evm'].rms_evm_percent)
           metrics['eye_height'].append(result['eye'].worst_eye_height)
           metrics['level_uniformity'].append(result['levels'].uniformity)
       
       return {k: np.array(v, dtype=np.float64) for k, v in metrics.items()}
   
   # Plot metric trends
   metrics = track_signal_metrics(batch_results)
   
   plt.figure(figsize=(12, 8))
   for name, values in metrics.items():
       plt.plot(values, label=name)
   plt.title('Signal Quality Metrics Over Time')
   plt.xlabel('Measurement Index')
   plt.ylabel('Value')
   plt.legend()
   plt.grid(True)
   plt.show()
   ```

## Advanced Visualization

### Interactive Eye Diagram

1. Create interactive plot:
   ```python
   import plotly.graph_objects as go
   
   def plot_interactive_eye(time_data: np.ndarray, voltage_data: np.ndarray) -> None:
       """Create interactive eye diagram plot"""
       t_norm = (time_data % 8.9e-12) / 8.9e-12
       
       fig = go.Figure()
       fig.add_trace(go.Scatter(
           x=t_norm,
           y=voltage_data,
           mode='markers',
           marker=dict(
               size=2,
               opacity=0.5
           )
       ))
       
       fig.update_layout(
           title='Interactive Eye Diagram',
           xaxis_title='UI',
           yaxis_title='Voltage',
           showlegend=False
       )
       
       fig.show()
   ```

### Custom Analysis Dashboard

1. Create analysis dashboard:
   ```python
   def create_analysis_dashboard(analyzer: PAM4Analyzer) -> None:
       """Create comprehensive analysis dashboard"""
       fig, axes = plt.subplots(2, 2, figsize=(15, 10))
       
       # Level distribution
       axes[0, 0].hist(analyzer.data['voltage'], bins=100)
       axes[0, 0].set_title('Level Distribution')
       
       # Eye diagram
       t_norm = (analyzer.data['time'] % 8.9e-12) / 8.9e-12
       axes[0, 1].plot(t_norm, analyzer.data['voltage'], '.', alpha=0.1)
       axes[0, 1].set_title('Eye Diagram')
       
       # EVM distribution
       evm = analyzer.calculate_evm('voltage', 'time')
       axes[1, 0].hist(evm.errors, bins=50)
       axes[1, 0].set_title('EVM Distribution')
       
       # Metrics summary
       metrics = [
           f"RMS EVM: {evm.rms_evm_percent:.2f}%",
           f"Peak EVM: {evm.peak_evm_percent:.2f}%",
           f"Eye Height: {analyzer.analyze_eye_diagram('voltage', 'time').worst_eye_height:.3f}"
       ]
       axes[1, 1].text(0.1, 0.5, '\n'.join(metrics))
       axes[1, 1].set_title('Metrics Summary')
       
       plt.tight_layout()
       plt.show()
   ```

## References

- PAM4 Modulation Theory
- Signal Processing Handbook
- IEEE 802.3 Specifications

---