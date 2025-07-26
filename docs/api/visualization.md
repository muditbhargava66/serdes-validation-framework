# Visualization API Reference

## Overview

The visualization module provides comprehensive plotting and analysis capabilities for SerDes validation results across all supported protocols.

## Module Structure

```
serdes_validation_framework.visualization/
├── __init__.py                 # Main module exports
├── base_visualizer.py         # Base visualizer class
├── eye_diagram.py             # Advanced eye diagram analysis
├── protocol_visualizers.py    # Protocol-specific visualizers
├── dashboard.py               # Interactive dashboards
└── comparison.py              # Multi-protocol comparison
```

## Base Classes

### BaseVisualizer

Abstract base class providing common visualization functionality.

```python
class BaseVisualizer(ABC):
    """Base class for all protocol visualizers"""
    
    def __init__(self, protocol_name: str)
    def plot_spectrum(self, signal_data: np.ndarray, sample_rate: float, **kwargs) -> Optional[str]
    def plot_time_domain(self, signal_data: np.ndarray, time_axis: Optional[np.ndarray] = None, **kwargs) -> Optional[str]
    def create_summary_report(self, test_results: Dict[str, Any], **kwargs) -> Optional[str]
    
    @abstractmethod
    def plot_eye_diagram(self, signal_data: Union[np.ndarray, Dict], **kwargs) -> Optional[str]
    
    @abstractmethod
    def plot_signal_quality(self, metrics: Dict[str, Any], **kwargs) -> Optional[str]
```

#### Properties
- `matplotlib_available: bool` - Whether matplotlib is available
- `plotly_available: bool` - Whether plotly is available
- `seaborn_available: bool` - Whether seaborn is available
- `protocol_name: str` - Name of the protocol

#### Methods

##### `plot_spectrum(signal_data, sample_rate, **kwargs)`
Plot frequency spectrum of signal data.

**Parameters:**
- `signal_data` (np.ndarray): Signal data to analyze
- `sample_rate` (float): Sample rate in Hz
- `**kwargs`: Additional parameters
  - `filename` (str): Output filename

**Returns:**
- `Optional[str]`: Path to saved plot or None if failed

##### `plot_time_domain(signal_data, time_axis=None, **kwargs)`
Plot time domain representation of signal.

**Parameters:**
- `signal_data` (np.ndarray): Signal data
- `time_axis` (Optional[np.ndarray]): Time axis values
- `**kwargs`: Additional parameters

**Returns:**
- `Optional[str]`: Path to saved plot or None if failed

##### `create_summary_report(test_results, **kwargs)`
Create comprehensive multi-panel summary report.

**Parameters:**
- `test_results` (Dict[str, Any]): Test results data
- `**kwargs`: Additional parameters

**Returns:**
- `Optional[str]`: Path to saved report or None if failed

## Eye Diagram Analysis

### EyeDiagramVisualizer

Advanced eye diagram visualization with automatic measurements.

```python
class EyeDiagramVisualizer(BaseVisualizer):
    """Advanced eye diagram visualization for SerDes signals"""
    
    def __init__(self, protocol_name: str = "SerDes")
    def plot_eye_diagram(self, signal_data: Union[np.ndarray, Dict], **kwargs) -> Optional[str]
    def plot_signal_quality(self, metrics: Dict[str, Any], **kwargs) -> Optional[str]
```

#### Methods

##### `plot_eye_diagram(signal_data, **kwargs)`
Create comprehensive eye diagram with measurements.

**Parameters:**
- `signal_data` (Union[np.ndarray, Dict]): Signal data
- `**kwargs`: Additional parameters
  - `symbol_rate` (float): Symbol rate in Hz (default: 20e9)
  - `samples_per_symbol` (int): Samples per symbol (default: 10)
  - `show_measurements` (bool): Show eye measurements (default: True)
  - `title` (str): Plot title
  - `filename` (str): Output filename

**Returns:**
- `Optional[str]`: Path to saved plot or None if failed

**Eye Measurements Calculated:**
- Eye height (V)
- Eye width (UI)
- SNR (dB)
- RMS jitter (ps)
- Crossing percentage

## Protocol-Specific Visualizers

### USB4Visualizer

Enhanced USB4 visualization with tunneling and power management support.

```python
class USB4Visualizer(BaseVisualizer):
    """Enhanced USB4 visualization"""
    
    def __init__(self)
    def plot_eye_diagram(self, signal_data: Union[np.ndarray, Dict], **kwargs) -> Optional[str]
    def plot_signal_quality(self, metrics: Dict[str, Any], **kwargs) -> Optional[str]
    def plot_tunnel_bandwidth(self, bandwidth_data: Dict[str, Any], **kwargs) -> Optional[str]
    def plot_power_states(self, power_data: Dict[str, Any], **kwargs) -> Optional[str]
    def create_interactive_dashboard(self, test_results: Dict[str, Any], **kwargs) -> Optional[str]
```

#### Methods

##### `plot_tunnel_bandwidth(bandwidth_data, **kwargs)`
Plot USB4 tunnel bandwidth utilization.

**Parameters:**
- `bandwidth_data` (Dict[str, Any]): Bandwidth data by tunnel type
  ```python
  {
      'PCIe': {'utilization': 75, 'max_bandwidth': 32, 'allocated_bandwidth': 24},
      'DisplayPort': {'utilization': 60, 'max_bandwidth': 20, 'allocated_bandwidth': 12},
      'USB3.2': {'utilization': 45, 'max_bandwidth': 10, 'allocated_bandwidth': 4.5}
  }
  ```
- `**kwargs`: Additional parameters

**Returns:**
- `Optional[str]`: Path to saved plot or None if failed

##### `plot_power_states(power_data, **kwargs)`
Plot USB4 power state transitions and consumption.

**Parameters:**
- `power_data` (Dict[str, Any]): Power state data
  ```python
  {
      'timeline': {
          'times': [0, 1, 2, 3, 4, 5],
          'states': ['U0', 'U0', 'U1', 'U2', 'U0', 'U3']
      },
      'consumption': {
          'U0': 2.5,  # Active (W)
          'U1': 1.2,  # Standby (W)
          'U2': 0.5,  # Sleep (W)
          'U3': 0.1   # Deep sleep (W)
      }
  }
  ```
- `**kwargs`: Additional parameters

**Returns:**
- `Optional[str]`: Path to saved plot or None if failed

##### `create_interactive_dashboard(test_results, **kwargs)`
Create interactive USB4 dashboard with multiple panels.

**Parameters:**
- `test_results` (Dict[str, Any]): Comprehensive test results
  ```python
  {
      'test_results': {
          'eye_diagram': {'status': 'PASS'},
          'jitter': {'status': 'PASS'},
          'tunneling': {'status': 'FAIL'}
      },
      'signal_quality': {
          'eye_height': 0.75,
          'snr': 22.5,
          'jitter_rms': 0.12
      },
      'tunnel_bandwidth': {
          'PCIe': 75,
          'DisplayPort': 60,
          'USB3.2': 45
      },
      'power_timeline': {
          'times': [0, 1, 2, 3],
          'states': ['U0', 'U1', 'U0', 'U2']
      }
  }
  ```
- `**kwargs`: Additional parameters
  - `filename` (str): Output HTML filename
  - `height` (int): Dashboard height in pixels

**Returns:**
- `Optional[str]`: Path to saved HTML dashboard or None if failed

### PCIeVisualizer

Enhanced PCIe visualization with link training and equalization support.

```python
class PCIeVisualizer(BaseVisualizer):
    """Enhanced PCIe visualization"""
    
    def __init__(self)
    def plot_eye_diagram(self, signal_data: Union[np.ndarray, Dict], **kwargs) -> Optional[str]
    def plot_signal_quality(self, metrics: Dict[str, Any], **kwargs) -> Optional[str]
    def plot_link_training(self, training_data: Dict[str, Any], **kwargs) -> Optional[str]
```

#### Methods

##### `plot_link_training(training_data, **kwargs)`
Plot PCIe link training progression and equalization.

**Parameters:**
- `training_data` (Dict[str, Any]): Link training data
  ```python
  {
      'phases': {
          'Detect': {'duration': 12, 'success': True},
          'Polling': {'duration': 24, 'success': True},
          'Configuration': {'duration': 48, 'success': True},
          'L0': {'duration': 1000, 'success': True}
      },
      'equalization': {
          'C-1': -2,  # Pre-cursor coefficient
          'C0': 15,   # Main cursor coefficient
          'C+1': -3   # Post-cursor coefficient
      }
  }
  ```
- `**kwargs`: Additional parameters

**Returns:**
- `Optional[str]`: Path to saved plot or None if failed

### EthernetVisualizer

Enhanced 224G Ethernet visualization with PAM4 support.

```python
class EthernetVisualizer(BaseVisualizer):
    """Enhanced 224G Ethernet visualization"""
    
    def __init__(self)
    def plot_eye_diagram(self, signal_data: Union[np.ndarray, Dict], **kwargs) -> Optional[str]
    def plot_signal_quality(self, metrics: Dict[str, Any], **kwargs) -> Optional[str]
    def plot_pam4_levels(self, signal_data: np.ndarray, **kwargs) -> Optional[str]
```

#### Methods

##### `plot_pam4_levels(signal_data, **kwargs)`
Plot PAM4 level distribution and analysis.

**Parameters:**
- `signal_data` (np.ndarray): PAM4 signal data
- `**kwargs`: Additional parameters

**Returns:**
- `Optional[str]`: Path to saved plot or None if failed

## Interactive Dashboards

### ValidationDashboard

Creates comprehensive interactive dashboards for validation results.

```python
class ValidationDashboard:
    """Interactive validation dashboard creator"""
    
    def __init__(self)
    def create_validation_dashboard(self, results: Dict[str, Any], **kwargs) -> Optional[str]
```

#### Methods

##### `create_validation_dashboard(results, **kwargs)`
Create comprehensive validation dashboard.

**Parameters:**
- `results` (Dict[str, Any]): Validation results data
- `**kwargs`: Additional parameters
  - `filename` (str): Output HTML filename

**Returns:**
- `Optional[str]`: Path to saved dashboard or None if failed

## Multi-Protocol Comparison

### ProtocolComparison

Multi-protocol comparison and analysis tools.

```python
class ProtocolComparison(BaseVisualizer):
    """Multi-protocol comparison visualizer"""
    
    def __init__(self)
    def compare_eye_diagrams(self, protocol_data: Dict[str, Any], **kwargs) -> Optional[str]
    def create_performance_comparison(self, performance_data: Dict[str, Dict], **kwargs) -> Optional[str]
    def plot_signal_quality(self, metrics: Dict[str, Any], **kwargs) -> Optional[str]
```

#### Methods

##### `compare_eye_diagrams(protocol_data, **kwargs)`
Compare eye diagrams across multiple protocols.

**Parameters:**
- `protocol_data` (Dict[str, Any]): Protocol data for comparison
  ```python
  {
      'USB4': {
          'signal': usb4_signal_data,
          'params': {'symbol_rate': 20e9, 'samples_per_symbol': 10}
      },
      'PCIe': {
          'signal': pcie_signal_data,
          'params': {'symbol_rate': 32e9, 'samples_per_symbol': 8}
      },
      'Ethernet': {
          'signal': eth_signal_data,
          'params': {'symbol_rate': 112e9, 'samples_per_symbol': 4}
      }
  }
  ```
- `**kwargs`: Additional parameters

**Returns:**
- `Optional[str]`: Path to saved comparison plot or None if failed

##### `create_performance_comparison(performance_data, **kwargs)`
Create performance comparison across protocols.

**Parameters:**
- `performance_data` (Dict[str, Dict]): Performance data by protocol
  ```python
  {
      'USB4': {'throughput': 40, 'latency': 125, 'power': 2.5},
      'PCIe': {'throughput': 64, 'latency': 100, 'power': 3.2},
      'Ethernet': {'throughput': 224, 'latency': 50, 'power': 5.8}
  }
  ```
- `**kwargs`: Additional parameters

**Returns:**
- `Optional[str]`: Path to saved comparison plot or None if failed

## Backward Compatibility

### Legacy USB4 Visualizer

The original USB4 visualizer has been enhanced with automatic fallback to advanced features.

```python
from serdes_validation_framework.protocols.usb4.visualization import USB4Visualizer

# Automatically uses enhanced features if available
viz = USB4Visualizer()
result = viz.plot_eye_diagram(data)  # Enhanced eye diagram with measurements
```

## Usage Examples

### Basic Eye Diagram
```python
from serdes_validation_framework.visualization import EyeDiagramVisualizer

viz = EyeDiagramVisualizer("USB4")
result = viz.plot_eye_diagram(
    signal_data,
    symbol_rate=20e9,
    show_measurements=True,
    filename='usb4_eye.png'
)
```

### Protocol-Specific Visualization
```python
from serdes_validation_framework.visualization import USB4Visualizer

viz = USB4Visualizer()

# Eye diagram
eye_result = viz.plot_eye_diagram(signal_data)

# Tunnel bandwidth
bandwidth_result = viz.plot_tunnel_bandwidth(bandwidth_data)

# Interactive dashboard
dashboard_result = viz.create_interactive_dashboard(test_results)
```

### Multi-Protocol Comparison
```python
from serdes_validation_framework.visualization import ProtocolComparison

comparison = ProtocolComparison()

# Compare eye diagrams
eye_comparison = comparison.compare_eye_diagrams(protocol_data)

# Compare performance
perf_comparison = comparison.create_performance_comparison(performance_data)
```

## Error Handling

All visualization methods include comprehensive error handling:

- **Graceful Degradation**: Falls back to basic implementations if advanced features unavailable
- **Dependency Checking**: Automatic detection of matplotlib, plotly, seaborn
- **Input Validation**: Validates signal data format and parameters
- **File Management**: Automatic output directory creation
- **Meaningful Errors**: Clear error messages for troubleshooting

## Dependencies

### Required
- `numpy>=1.26.0`

### Optional
- `matplotlib>=3.9.0` - For static plots
- `plotly>=5.17.0` - For interactive dashboards
- `seaborn>=0.13.0` - For enhanced styling

### Availability Flags
```python
from serdes_validation_framework.visualization import (
    MATPLOTLIB_AVAILABLE,
    PLOTLY_AVAILABLE,
    SEABORN_AVAILABLE
)
```

## Performance Considerations

- **Large Datasets**: Use downsampling for datasets > 100K samples
- **Eye Diagrams**: Limit traces to < 200 for optimal rendering
- **Memory Usage**: Optimized for datasets up to 1M samples
- **Rendering Speed**: ~0.5-2.0 seconds for typical eye diagrams
- **Interactive Dashboards**: ~1.0-3.0 seconds for multi-panel layouts