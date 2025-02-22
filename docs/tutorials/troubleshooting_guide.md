# 224G Ethernet Troubleshooting Guide

## Common Issues and Solutions

### Signal Quality Issues

#### Problem: High EVM Values
```python
# Example of high EVM detection
evm_results = analyzer.calculate_evm('voltage', 'time')
if evm_results.rms_evm_percent > 5.0:
    print(f"High EVM detected: {evm_results.rms_evm_percent:.2f}%")
```

**Possible Causes:**
1. Insufficient bandwidth
2. Signal reflection
3. Noise coupling

**Solutions:**
1. Check scope bandwidth:
   ```python
   if scope.default_config.bandwidth < 120e9:
       print("Warning: Insufficient scope bandwidth")
   ```

2. Check termination:
   ```python
   def check_termination(voltage_data: np.ndarray) -> None:
       reflection_ratio = np.max(np.abs(np.diff(voltage_data))) / np.max(np.abs(voltage_data))
       if reflection_ratio > 0.2:
           print(f"High reflection detected: {reflection_ratio:.2f}")
   ```

3. Visualize noise:
   ```python
   def plot_noise_spectrum(voltage_data: np.ndarray, sample_rate: float) -> None:
       from scipy import signal
       f, pxx = signal.welch(voltage_data, fs=sample_rate)
       plt.figure(figsize=(10, 6))
       plt.semilogy(f/1e9, pxx)
       plt.title('Noise Spectrum')
       plt.xlabel('Frequency (GHz)')
       plt.ylabel('Power')
       plt.grid(True)
       plt.show()
   ```

#### Problem: Poor Eye Opening
```python
# Check eye height
eye_results = analyzer.analyze_eye_diagram('voltage', 'time')
if eye_results.worst_eye_height < 0.2:
    print("Warning: Eye height below threshold")
```

**Possible Causes:**
1. ISI (Inter-Symbol Interference)
2. Jitter
3. Level mismatch

**Solutions:**
1. Analyze ISI:
   ```python
   def analyze_isi(analyzer: PAM4Analyzer) -> None:
       # Plot post-cursor ISI
       plt.figure(figsize=(12, 6))
       
       # Main eye
       t_norm = (analyzer.data['time'] % 8.9e-12) / 8.9e-12
       plt.subplot(121)
       plt.plot(t_norm, analyzer.data['voltage'], '.', alpha=0.1)
       plt.title('Eye Diagram')
       
       # Post-cursor
       plt.subplot(122)
       plt.hist2d(t_norm, analyzer.data['voltage'], bins=50)
       plt.title('ISI Heat Map')
       plt.colorbar()
       
       plt.tight_layout()
       plt.show()
   ```

2. Analyze jitter components:
   ```python
   def visualize_jitter_decomposition(
       time_data: np.ndarray,
       voltage_data: np.ndarray
   ) -> None:
       # Create bathtub curve
       edges = np.linspace(0, 1, 100)
       crossings = np.zeros_like(edges)
       
       for i, edge in enumerate(edges):
           mask = (voltage_data[:-1] <= edge) & (voltage_data[1:] > edge)
           crossings[i] = len(np.where(mask)[0])
       
       plt.figure(figsize=(10, 6))
       plt.semilogy(edges, crossings/np.max(crossings))
       plt.title('Jitter Bathtub Curve')
       plt.xlabel('UI')
       plt.ylabel('BER')
       plt.grid(True)
       plt.show()
   ```

### Equipment Setup Issues

#### Problem: VISA Communication Errors
```python
try:
    scope.configure_for_224g()
except Exception as e:
    print(f"VISA Error: {e}")
```

**Solutions:**
1. Check connections:
   ```python
   def verify_instrument_connection(resource_name: str) -> bool:
       try:
           from pyvisa import ResourceManager
           rm = ResourceManager()
           instrument = rm.open_resource(resource_name)
           response = instrument.query('*IDN?')
           print(f"Connected to: {response}")
           return True
       except Exception as e:
           print(f"Connection failed: {e}")
           return False
   ```

2. Reset instruments:
   ```python
   def reset_instruments(scope_address: str, pattern_gen_address: str) -> None:
       sequence = Ethernet224GTestSequence()
       
       try:
           # Reset scope
           sequence.instrument_controller.send_command(scope_address, '*RST')
           sequence.instrument_controller.send_command(scope_address, '*CLS')
           
           # Reset pattern generator
           sequence.instrument_controller.send_command(pattern_gen_address, '*RST')
           sequence.instrument_controller.send_command(pattern_gen_address, '*CLS')
           
           print("Instruments reset successfully")
       except Exception as e:
           print(f"Reset failed: {e}")
   ```

### Analysis Issues

#### Problem: Invalid Results
```python
def validate_analysis_results(results: Dict[str, Any]) -> bool:
    """Validate analysis results for common issues"""
    try:
        # Check EVM range
        if not 0 <= results['evm_results'].rms_evm_percent <= 100:
            return False
            
        # Check eye measurements
        if not all(0 <= height <= 1 for height in results['eye_results'].eye_heights):
            return False
            
        # Check level separation
        if not results['pam4_levels'].level_separations.all():
            return False
            
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False
```

**Solutions:**
1. Data validation:
   ```python
   def validate_input_data(
       time_data: np.ndarray,
       voltage_data: np.ndarray
   ) -> bool:
       """Validate input data quality"""
       checks = {
           'length_match': len(time_data) == len(voltage_data),
           'time_monotonic': np.all(np.diff(time_data) > 0),
           'voltage_range': np.abs(voltage_data).max() < 10,
           'no_nans': not (np.isnan(time_data).any() or np.isnan(voltage_data).any())
       }
       
       for check_name, passed in checks.items():
           if not passed:
               print(f"Validation failed: {check_name}")
               
       return all(checks.values())
   ```

2. Result visualization:
   ```python
   def visualize_analysis_results(results: Dict[str, Any]) -> None:
       """Create comprehensive visualization of results"""
       fig = plt.figure(figsize=(15, 10))
       
       # Layout
       gs = fig.add_gridspec(3, 3)
       
       # Eye diagram
       ax1 = fig.add_subplot(gs[0, :2])
       plot_eye_diagram(results['eye_results'])
       
       # Level histogram
       ax2 = fig.add_subplot(gs[1, :2])
       plot_level_histogram(results['pam4_levels'])
       
       # EVM plot
       ax3 = fig.add_subplot(gs[2, :2])
       plot_evm_distribution(results['evm_results'])
       
       # Metrics summary
       ax4 = fig.add_subplot(gs[:, 2])
       plot_metrics_summary(results)
       
       plt.tight_layout()
       plt.show()
   ```

## Advanced Visualization Examples

### Interactive Analysis Dashboard
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_dashboard(analyzer: PAM4Analyzer) -> None:
    """Create interactive analysis dashboard"""
    # Get analysis results
    level_results = analyzer.analyze_level_separation('voltage')
    evm_results = analyzer.calculate_evm('voltage', 'time')
    eye_results = analyzer.analyze_eye_diagram('voltage', 'time')
    
    # Create dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Eye Diagram',
            'Level Distribution',
            'EVM Distribution',
            'Metrics'
        )
    )
    
    # Add eye diagram
    t_norm = (analyzer.data['time'] % 8.9e-12) / 8.9e-12
    fig.add_trace(
        go.Scatter(
            x=t_norm,
            y=analyzer.data['voltage'],
            mode='markers',
            marker=dict(size=2, opacity=0.5)
        ),
        row=1, col=1
    )
    
    # Add level histogram
    fig.add_trace(
        go.Histogram(
            x=analyzer.data['voltage'],
            nbinsx=100
        ),
        row=1, col=2
    )
    
    # Add EVM distribution
    fig.add_trace(
        go.Histogram(
            x=evm_results.errors,
            nbinsx=50
        ),
        row=2, col=1
    )
    
    # Add metrics table
    metrics = [
        f"RMS EVM: {evm_results.rms_evm_percent:.2f}%",
        f"Peak EVM: {evm_results.peak_evm_percent:.2f}%",
        f"Eye Height: {eye_results.worst_eye_height:.3f}",
        f"Level Uniformity: {level_results.uniformity:.3f}"
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value']),
            cells=dict(values=[
                ['RMS EVM', 'Peak EVM', 'Eye Height', 'Level Uniformity'],
                [f"{evm_results.rms_evm_percent:.2f}%",
                 f"{evm_results.peak_evm_percent:.2f}%",
                 f"{eye_results.worst_eye_height:.3f}",
                 f"{level_results.uniformity:.3f}"]
            ])
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    fig.show()
```

### Real-Time Monitoring
```python
def create_real_time_monitor(
    analyzer: PAM4Analyzer,
    update_interval: float = 1.0
) -> None:
    """Create real-time signal quality monitor"""
    import matplotlib.animation as animation
    
    # Initialize plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Real-Time Signal Monitor')
    
    def update(frame):
        # Clear plots
        for ax in axes.flat:
            ax.clear()
        
        # Update measurements
        level_results = analyzer.analyze_level_separation('voltage')
        evm_results = analyzer.calculate_evm('voltage', 'time')
        eye_results = analyzer.analyze_eye_diagram('voltage', 'time')
        
        # Update plots
        plot_eye_diagram(eye_results, axes[0, 0])
        plot_level_histogram(level_results, axes[0, 1])
        plot_evm_distribution(evm_results, axes[1, 0])
        plot_metrics_summary(
            level_results,
            evm_results,
            eye_results,
            axes[1, 1]
        )
        
        plt.tight_layout()
    
    ani = animation.FuncAnimation(
        fig,
        update,
        interval=update_interval*1000
    )
    plt.show()
```

---