# Jupyter Dashboard Guide

The SerDes Validation Framework includes a comprehensive Jupyter dashboard system for interactive eye diagram visualization and waveform analysis.

## Overview

The Jupyter dashboard module provides:

- **Interactive Eye Diagram Visualization**: Professional-grade eye diagrams with real-time controls
- **Waveform Analysis**: Comprehensive signal quality metrics and pass/fail annotations
- **Multi-Protocol Support**: USB4, PCIe, and Ethernet protocol templates
- **Export Capabilities**: Save results and plots in multiple formats
- **Captured Waveform Integration**: Direct integration with SVF analyzers

## Quick Start

### 1. Check Dependencies

```python
from serdes_validation_framework.jupyter_dashboard import check_dashboard_dependencies

# Check what's available
deps = check_dashboard_dependencies()
```

### 2. Create a Basic Dashboard

```python
from serdes_validation_framework.jupyter_dashboard import create_dashboard
import numpy as np

# Generate or load your signal data
signal_data = np.random.randn(10000) * 0.4  # Example signal
sample_rate = 40e9  # 40 GSa/s

# Create dashboard
dashboard = create_dashboard(
    signal_data=signal_data,
    sample_rate=sample_rate,
    protocol="USB4"
)

# Analyze eye diagram
results = dashboard.analyze_eye_diagram()
print(f"Eye Height: {results['eye_height']:.4f}V")
print(f"Status: {'PASS' if results['passed'] else 'FAIL'}")
```

### 3. Multi-Lane Analysis

```python
# Multi-lane signal data
signal_data = {
    'lane_0': np.random.randn(10000) * 0.4,
    'lane_1': np.random.randn(10000) * 0.4,
    'lane_2': np.random.randn(10000) * 0.4,
    'lane_3': np.random.randn(10000) * 0.4
}

dashboard = create_dashboard(
    signal_data=signal_data,
    sample_rate=40e9,
    protocol="USB4"
)

# Analyze specific lane
results = dashboard.analyze_eye_diagram(lane='lane_0')
```

## Interactive Features (Jupyter Notebook)

### 1. Real-time Controls

```python
# In Jupyter notebook
dashboard.create_interactive_dashboard()
```

This creates interactive widgets for:
- Lane selection
- Time window adjustment
- Measurement threshold controls
- Protocol parameter tuning

### 2. Static Dashboard

```python
# Create static matplotlib dashboard
dashboard.create_static_dashboard(lane='lane_0')
```

## Waveform Analysis

```python
from serdes_validation_framework.jupyter_dashboard import WaveformAnalyzer

analyzer = WaveformAnalyzer(sample_rate=40e9, protocol="USB4")

# Analyze signal quality
result = analyzer.analyze_waveform(
    voltage_data=signal_data,
    lane="lane_0"
)

print(f"SNR: {result.snr_db:.2f} dB")
print(f"THD: {result.thd_percent:.2f}%")
print(f"Status: {'PASS' if result.passed else 'FAIL'}")

# Get summary report
summary = analyzer.get_summary_report()
print(summary)
```

## Configuration

```python
from serdes_validation_framework.jupyter_dashboard import DashboardConfig

# Custom configuration
config = DashboardConfig(
    figure_width=14,
    figure_height=10,
    show_measurements=True,
    show_mask=True,
    background_color='white'
)

dashboard = create_dashboard(
    signal_data=signal_data,
    sample_rate=40e9,
    protocol="USB4",
    config=config
)
```

## Export Results

```python
# Export analysis results
dashboard.export_results("eye_analysis_results.json")

# Results include:
# - Eye height/width measurements
# - SNR and Q-factor
# - Pass/fail status
# - Protocol-specific metrics
# - Timestamp and configuration
```

## Protocol Support

### USB4
- Data rate: 20 Gbps per lane
- Eye height threshold: 100 mV
- SNR threshold: 15 dB

### PCIe
- Data rate: 32 Gbps per lane  
- Eye height threshold: 150 mV
- SNR threshold: 12 dB

### Ethernet
- Data rate: 112 Gbps per lane
- Eye height threshold: 200 mV
- SNR threshold: 10 dB

## Integration with SVF

```python
# Use with SVF captured data
from serdes_validation_framework.data_capture import SignalCapture

# Capture real signals
capture = SignalCapture()
waveforms = capture.capture_waveforms(duration=1e-6, sample_rate=40e9)

# Create dashboard with captured data
dashboard = create_dashboard(
    signal_data=waveforms,
    sample_rate=40e9,
    protocol="USB4"
)
```

## Examples

See `examples/jupyter_dashboard_example.py` for comprehensive usage examples including:

- Basic dashboard creation
- Multi-protocol comparison
- Waveform analysis
- Interactive plotting
- Results export

## Jupyter Notebook

Run the interactive notebook:

```bash
jupyter notebook examples/jupyter_eye_diagram_dashboard.ipynb
```

This provides a complete interactive environment with:
- Real-time parameter controls
- Live plot updates
- Export capabilities
- Multi-lane analysis tools

## Dependencies

Required packages:
- `jupyter` - Notebook environment
- `matplotlib` - Static plotting
- `plotly` - Interactive plotting
- `ipywidgets` - Interactive controls (for full interactivity)

Install with:
```bash
pip install jupyter matplotlib plotly ipywidgets
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Plot Display**: In Jupyter, use `%matplotlib inline` or `%matplotlib widget`
3. **Interactive Widgets**: Requires `ipywidgets` and may need `jupyter labextension install @jupyter-widgets/jupyterlab-manager`

### Performance Tips

1. **Large Datasets**: Use time windowing for signals > 100k samples
2. **Memory Usage**: Process lanes individually for very large multi-lane datasets
3. **Plotting Speed**: Use static dashboards for batch processing

## API Reference

See the full API documentation for detailed parameter descriptions and advanced usage patterns.