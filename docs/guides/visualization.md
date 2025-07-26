# 🎨 Visualization Guide

The SerDes Validation Framework v1.4.0 includes a comprehensive visualization system that provides professional-grade plots, interactive dashboards, and multi-protocol analysis capabilities.

## 📊 Overview

The visualization system is built on a modular architecture that supports:
- **Advanced Eye Diagrams** with automatic measurements
- **Interactive Dashboards** using Plotly
- **Protocol-Specific Visualizations** for USB4, PCIe, and Ethernet
- **Multi-Protocol Comparison** tools
- **Signal Analysis Plots** including spectrum and time domain

## 🚀 Quick Start

### Basic Eye Diagram
```python
from serdes_validation_framework.visualization import USB4Visualizer
import numpy as np

# Create visualizer
viz = USB4Visualizer()

# Generate test signal
signal_data = {
    0: {'voltage': np.random.randn(1000), 'time': np.linspace(0, 1e-6, 1000)}
}

# Create eye diagram with measurements
result = viz.plot_eye_diagram(
    signal_data,
    show_measurements=True,
    filename='usb4_eye.png'
)
```

### Interactive Dashboard
```python
# Create interactive dashboard
test_results = {
    'test_results': {
        'eye_diagram': {'status': 'PASS'},
        'jitter': {'status': 'PASS'},
        'power_management': {'status': 'FAIL'}
    },
    'signal_quality': {
        'eye_height': 0.75,
        'snr': 22.5,
        'jitter_rms': 0.12
    }
}

dashboard_result = viz.create_interactive_dashboard(
    test_results,
    filename='usb4_dashboard.html'
)
```

## 🏗️ Architecture

### Base Visualizer
All visualizers inherit from `BaseVisualizer`, which provides:
- Automatic dependency detection (matplotlib, plotly, seaborn)
- Common plotting utilities (spectrum, time domain)
- Error handling and graceful degradation
- File management and output directory creation

### Protocol-Specific Visualizers

#### USB4Visualizer
```python
from serdes_validation_framework.visualization import USB4Visualizer

viz = USB4Visualizer()

# USB4-specific features
viz.plot_tunnel_bandwidth(bandwidth_data)
viz.plot_power_states(power_data)
viz.create_interactive_dashboard(test_results)
```

#### PCIeVisualizer
```python
from serdes_validation_framework.visualization import PCIeVisualizer

viz = PCIeVisualizer()

# PCIe-specific features
viz.plot_link_training(training_data)
viz.plot_equalization(eq_data)
```

#### EthernetVisualizer
```python
from serdes_validation_framework.visualization import EthernetVisualizer

viz = EthernetVisualizer()

# Ethernet-specific features
viz.plot_pam4_levels(signal_data)
```

## 📈 Advanced Eye Diagram Analysis

### Features
- **Automatic Measurements**: Eye height, width, SNR, jitter
- **Sampling Point Indicators**: Visual markers for optimal sampling
- **Statistical Overlays**: Mean, standard deviation indicators
- **Adaptive Rendering**: Automatic transparency adjustment for trace density

### Example
```python
from serdes_validation_framework.visualization import EyeDiagramVisualizer

eye_viz = EyeDiagramVisualizer("USB4")

result = eye_viz.plot_eye_diagram(
    signal_data,
    symbol_rate=20e9,
    samples_per_symbol=10,
    show_measurements=True,
    title="USB4 Eye Diagram Analysis"
)
```

### Measurements Calculated
- **Eye Height**: Vertical eye opening in volts
- **Eye Width**: Horizontal eye opening in UI
- **SNR**: Signal-to-noise ratio in dB
- **RMS Jitter**: Root-mean-square jitter in picoseconds
- **Crossing Percentage**: Zero-crossing statistics

## 🌐 Interactive Dashboards

### Features
- **Multi-Panel Layout**: Test results, signal quality, performance, compliance
- **Real-Time Updates**: Dynamic data visualization
- **Interactive Exploration**: Zoom, pan, hover capabilities
- **Export Options**: HTML, PNG, SVG formats

### Dashboard Components

#### Test Results Panel
```python
test_results = {
    'test_results': {
        'eye_diagram': {'status': 'PASS'},
        'jitter': {'status': 'PASS'},
        'compliance': {'status': 'FAIL'}
    }
}
```

#### Signal Quality Panel
```python
signal_quality = {
    'eye_height': 0.75,
    'eye_width': 0.85,
    'snr': 22.5,
    'jitter_rms': 0.12,
    'ber': 1e-12
}
```

#### Performance Panel
```python
performance = {
    'throughput': 38.5,  # Gbps
    'latency': 125,      # ns
    'power': 2.5         # W
}
```

## 🔌 Protocol-Specific Features

### USB4 Visualizations

#### Tunnel Bandwidth Analysis
```python
bandwidth_data = {
    'PCIe': {'utilization': 75, 'max_bandwidth': 32},
    'DisplayPort': {'utilization': 60, 'max_bandwidth': 20},
    'USB3.2': {'utilization': 45, 'max_bandwidth': 10}
}

result = usb4_viz.plot_tunnel_bandwidth(bandwidth_data)
```

#### Power State Analysis
```python
power_data = {
    'timeline': {
        'times': [0, 1, 2, 3, 4, 5],
        'states': ['U0', 'U0', 'U1', 'U2', 'U0', 'U3']
    },
    'consumption': {
        'U0': 2.5,  # Active
        'U1': 1.2,  # Standby
        'U2': 0.5,  # Sleep
        'U3': 0.1   # Deep sleep
    }
}

result = usb4_viz.plot_power_states(power_data)
```

### PCIe Visualizations

#### Link Training Analysis
```python
training_data = {
    'phases': {
        'Detect': {'duration': 12, 'success': True},
        'Polling': {'duration': 24, 'success': True},
        'Configuration': {'duration': 48, 'success': True},
        'L0': {'duration': 1000, 'success': True}
    },
    'equalization': {
        'C-1': -2,
        'C0': 15,
        'C+1': -3
    }
}

result = pcie_viz.plot_link_training(training_data)
```

### Ethernet Visualizations

#### PAM4 Level Analysis
```python
# Analyze PAM4 signal levels
result = eth_viz.plot_pam4_levels(
    pam4_signal_data,
    filename='pam4_analysis.png'
)
```

## 🔄 Multi-Protocol Comparison

### Protocol Comparison Tool
```python
from serdes_validation_framework.visualization import ProtocolComparison

comparison = ProtocolComparison()

# Compare eye diagrams
protocol_data = {
    'USB4': {
        'signal': usb4_signal,
        'params': {'symbol_rate': 20e9, 'samples_per_symbol': 10}
    },
    'PCIe': {
        'signal': pcie_signal,
        'params': {'symbol_rate': 32e9, 'samples_per_symbol': 8}
    },
    'Ethernet': {
        'signal': eth_signal,
        'params': {'symbol_rate': 112e9, 'samples_per_symbol': 4}
    }
}

result = comparison.compare_eye_diagrams(protocol_data)
```

### Performance Comparison
```python
performance_data = {
    'USB4': {'throughput': 40, 'latency': 125, 'power': 2.5},
    'PCIe': {'throughput': 64, 'latency': 100, 'power': 3.2},
    'Ethernet': {'throughput': 224, 'latency': 50, 'power': 5.8}
}

result = comparison.create_performance_comparison(performance_data)
```

## 📊 Signal Analysis Plots

### Frequency Spectrum Analysis
```python
# Plot frequency spectrum
spectrum_result = viz.plot_spectrum(
    signal_data,
    sample_rate=200e9,
    filename='signal_spectrum.png'
)
```

### Time Domain Analysis
```python
# Plot time domain signal
time_result = viz.plot_time_domain(
    signal_data,
    time_axis=time_vector,
    filename='time_domain.png'
)
```

### Comprehensive Summary Report
```python
# Create multi-panel summary
summary_result = viz.create_summary_report(
    test_results,
    filename='validation_summary.png'
)
```

## 🎨 Customization

### Custom Visualizer
```python
from serdes_validation_framework.visualization import BaseVisualizer

class CustomProtocolVisualizer(BaseVisualizer):
    def __init__(self):
        super().__init__("CustomProtocol")
    
    def plot_eye_diagram(self, signal_data, **kwargs):
        # Custom eye diagram implementation
        pass
    
    def plot_signal_quality(self, metrics, **kwargs):
        # Custom signal quality plot
        pass
    
    def plot_custom_feature(self, data, **kwargs):
        # Protocol-specific visualization
        if not self.matplotlib_available:
            return None
        
        fig, ax = self.plt.subplots(figsize=(10, 6))
        # Custom plotting code
        ax.plot(data)
        ax.set_title(f'{self.protocol_name} Custom Analysis')
        
        filename = kwargs.get('filename', 'custom_plot.png')
        filepath = self._ensure_output_dir(filename)
        self.plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plt.close()
        
        return str(filepath)
```

### Styling Options
```python
# Use seaborn styling (if available)
if viz.seaborn_available:
    viz.sns.set_style("whitegrid")
    viz.sns.set_palette("husl")

# Custom color schemes
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
```

## 🔧 Configuration

### Output Settings
```python
# High-resolution output
result = viz.plot_eye_diagram(
    signal_data,
    filename='high_res_eye.png',
    dpi=300,
    figsize=(12, 8)
)

# Interactive dashboard settings
dashboard_result = viz.create_interactive_dashboard(
    test_results,
    filename='dashboard.html',
    height=800,
    show_legend=True
)
```

### Performance Optimization
```python
# For large datasets
result = viz.plot_eye_diagram(
    large_signal_data,
    max_traces=100,  # Limit number of traces
    downsample=True,  # Enable downsampling
    alpha=0.3        # Reduce opacity for performance
)
```

## 🔍 Troubleshooting

### Common Issues

#### Missing Dependencies
```python
# Check availability
from serdes_validation_framework.visualization import MATPLOTLIB_AVAILABLE, PLOTLY_AVAILABLE

if not MATPLOTLIB_AVAILABLE:
    print("Install matplotlib: pip install matplotlib")

if not PLOTLY_AVAILABLE:
    print("Install plotly: pip install plotly")
```

#### Memory Issues with Large Datasets
```python
# Use downsampling for large datasets
result = viz.plot_eye_diagram(
    large_signal,
    max_samples=10000,  # Limit samples
    downsample_factor=10  # Downsample by factor
)
```

#### File Permission Issues
```python
# Ensure output directory exists and is writable
import os
output_dir = "visualization_output"
os.makedirs(output_dir, exist_ok=True)

result = viz.plot_eye_diagram(
    signal_data,
    filename=f"{output_dir}/eye_diagram.png"
)
```

## 📚 Examples

### Complete Visualization Workflow
```python
from serdes_validation_framework.visualization import USB4Visualizer
import numpy as np

# Initialize visualizer
viz = USB4Visualizer()

# Generate test data
signal_data = {
    0: {'voltage': np.random.randn(2000) * 0.1 + np.sin(np.linspace(0, 20*np.pi, 2000))}
}

# 1. Eye diagram analysis
eye_result = viz.plot_eye_diagram(
    signal_data,
    show_measurements=True,
    filename='usb4_eye_analysis.png'
)

# 2. Signal quality metrics
quality_metrics = {
    'eye_height': 0.75,
    'eye_width': 0.85,
    'snr': 22.5,
    'jitter_rms': 0.12
}

quality_result = viz.plot_signal_quality(
    quality_metrics,
    filename='signal_quality.png'
)

# 3. Tunnel bandwidth analysis
bandwidth_data = {
    'PCIe': {'utilization': 75, 'allocated_bandwidth': 24},
    'DisplayPort': {'utilization': 60, 'allocated_bandwidth': 12},
    'USB3.2': {'utilization': 45, 'allocated_bandwidth': 4.5}
}

bandwidth_result = viz.plot_tunnel_bandwidth(
    bandwidth_data,
    filename='tunnel_bandwidth.png'
)

# 4. Interactive dashboard
test_results = {
    'test_results': {
        'eye_diagram': {'status': 'PASS'},
        'jitter': {'status': 'PASS'},
        'tunneling': {'status': 'PASS'}
    },
    'signal_quality': quality_metrics,
    'tunnel_bandwidth': {k: v['utilization'] for k, v in bandwidth_data.items()}
}

dashboard_result = viz.create_interactive_dashboard(
    test_results,
    filename='usb4_comprehensive_dashboard.html'
)

print("Visualization complete!")
print(f"Eye diagram: {eye_result}")
print(f"Signal quality: {quality_result}")
print(f"Bandwidth analysis: {bandwidth_result}")
print(f"Interactive dashboard: {dashboard_result}")
```

## 🚀 Best Practices

### Performance
- Use downsampling for datasets > 100K samples
- Limit eye diagram traces to < 200 for optimal rendering
- Enable caching for repeated visualizations
- Use appropriate figure sizes for target output

### Quality
- Use 300 DPI for publication-ready plots
- Include measurement overlays for professional appearance
- Use consistent color schemes across related plots
- Add proper titles, labels, and legends

### Usability
- Generate both static and interactive versions
- Include hover information in interactive plots
- Use meaningful filenames with timestamps
- Organize outputs in structured directories

### Integration
- Combine multiple visualizations in summary reports
- Use consistent data formats across visualizers
- Implement error handling for missing data
- Provide fallback options for missing dependencies

---

The visualization system provides a comprehensive toolkit for creating professional-grade plots and interactive dashboards for SerDes validation. With its modular architecture and extensive customization options, it can be adapted to meet specific analysis and reporting needs.