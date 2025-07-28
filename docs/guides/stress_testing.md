# 🔄 Stress Testing Guide

## Overview

The SerDes Validation Framework v1.4.1 introduces comprehensive **Loopback Stress Testing** capabilities that simulate TX → RX → back to TX signal paths and track signal degradation over multiple test cycles. This feature is essential for long-term reliability testing and signal integrity validation.

## Key Features

### 🔄 Complete Loopback Simulation
- **TX → RX → back to TX** signal path simulation
- **Progressive degradation modeling** with realistic signal effects
- **Multi-cycle testing** supporting 1000+ test cycles
- **Real-time monitoring** with live progress tracking

### 📊 Advanced Analytics
- **Eye diagram tracking** with automatic measurements
- **Jitter analysis** (RMS and peak jitter over time)
- **SNR calculation** and monitoring
- **BER estimation** per cycle
- **Degradation percentage** quantification

### 🎨 Visualization & Reporting
- **Interactive plots** showing degradation trends
- **CSV export** for external analysis
- **Waveform saving** for failed cycles
- **Comprehensive reports** with statistical summaries

## Quick Start

### Basic Usage

```python
from serdes_validation_framework.stress_testing import LoopbackStressTest, create_stress_test_config

# Create configuration
config = create_stress_test_config(
    protocol="USB4",
    num_cycles=1000,
    output_dir="usb4_stress_test"
)

# Run stress test
stress_test = LoopbackStressTest(config)
results = stress_test.run_stress_test()

# Analyze results
print(f"Success rate: {results.success_rate:.1%}")
print(f"Max degradation: {results.max_degradation:.1f}%")
```

### Command-Line Usage

```bash
# Basic stress test
python scripts/run_loopback_stress_test.py --protocol USB4 --cycles 1000

# Advanced configuration
python scripts/run_loopback_stress_test.py \
    --protocol PCIe \
    --cycles 500 \
    --duration 0.5 \
    --eye-threshold 0.15 \
    --save-waveforms \
    --output pcie_stress_results

# Quick test
python scripts/run_loopback_stress_test.py --quick
```

## Configuration

### StressTestConfig Parameters

```python
from serdes_validation_framework.stress_testing import StressTestConfig
from pathlib import Path

config = StressTestConfig(
    # Test parameters
    num_cycles=1000,           # Number of test cycles
    cycle_duration=1.0,        # Duration per cycle (seconds)
    
    # Protocol settings
    protocol="USB4",           # USB4, PCIe, or Ethernet
    data_rate=20e9,           # Data rate in bps
    voltage_swing=0.8,        # Voltage swing in V
    
    # Thresholds
    eye_height_threshold=0.1,  # 10% degradation threshold
    jitter_threshold=0.05,     # 5% jitter increase threshold
    
    # Output settings
    output_dir=Path("stress_test_results"),
    save_waveforms=False,      # Save waveforms for failed cycles
    generate_plots=True,       # Generate analysis plots
    
    # Logging
    log_level="INFO"
)
```

### Protocol-Specific Defaults

#### USB4 Configuration
```python
config = create_stress_test_config(protocol="USB4")
# Defaults: 20 Gbps, 0.8V swing, NRZ signaling
```

#### PCIe Configuration
```python
config = create_stress_test_config(protocol="PCIe")
config.data_rate = 32e9  # 32 GT/s
config.voltage_swing = 1.2  # 1.2V
# PAM4 signaling with link training effects
```

#### Ethernet Configuration
```python
config = create_stress_test_config(protocol="Ethernet")
config.data_rate = 112e9  # 112 Gbps
config.voltage_swing = 0.8  # 0.8V
# PAM4 signaling with level analysis
```

## Advanced Usage

### Multi-Protocol Comparison

```python
protocols = ["USB4", "PCIe", "Ethernet"]
results = {}

for protocol in protocols:
    config = create_stress_test_config(
        protocol=protocol,
        num_cycles=100,
        output_dir=f"{protocol.lower()}_stress"
    )
    
    stress_test = LoopbackStressTest(config)
    results[protocol] = stress_test.run_stress_test()

# Compare results
for protocol, result in results.items():
    print(f"{protocol}: {result.success_rate:.1%} success, "
          f"{result.max_degradation:.1f}% max degradation")
```

### Custom Degradation Analysis

```python
# Access detailed cycle results
for cycle_result in results.cycle_results:
    if not cycle_result.passed:
        print(f"Cycle {cycle_result.cycle_number} failed:")
        print(f"  Eye height: {cycle_result.eye_height:.4f}V")
        print(f"  Degradation: {cycle_result.degradation_percent:.1f}%")
        print(f"  Notes: {cycle_result.notes}")

# Statistical analysis
print(f"Mean eye height: {results.mean_eye_height:.4f}V ± {results.std_eye_height:.4f}V")
print(f"Mean jitter: {results.mean_jitter:.4f} ± {results.std_jitter:.4f}")
print(f"Degradation rate: {results.degradation_rate:.4f}% per cycle")
```

## Signal Degradation Modeling

### Progressive Effects

The stress testing system models realistic signal degradation effects:

1. **Amplitude Loss**: Up to 10% signal attenuation over test duration
2. **Timing Jitter**: Progressive increase in timing uncertainty
3. **Noise Increase**: Growing noise floor over cycles
4. **ISI (Inter-Symbol Interference)**: Realistic channel effects after 30% cycles

### Protocol-Specific Modeling

#### USB4 Degradation
- **NRZ signaling** with 20 Gbps data rate
- **Dual-lane effects** with lane skew simulation
- **Power state transitions** affecting signal quality

#### PCIe Degradation
- **PAM4 signaling** with 32 GT/s data rate
- **Link training effects** on signal quality
- **Equalization degradation** over time

#### Ethernet Degradation
- **PAM4 signaling** with 112 Gbps data rate
- **Level separation** degradation
- **Amplitude distribution** changes

## Output Analysis

### CSV Data Format

The stress test generates detailed CSV files with the following columns:

```csv
cycle_number,timestamp,eye_height,eye_width,rms_jitter,peak_jitter,snr,ber_estimate,passed,degradation_percent,notes
1,1640995200.0,0.1234,0.8765,0.0123,0.0456,15.67,1.23e-12,True,0.0,""
2,1640995201.0,0.1220,0.8750,0.0125,0.0460,15.45,1.45e-12,True,1.1,"Slight degradation"
```

### Interactive Plots

Generated plots include:

1. **Eye Height vs Cycle**: Shows signal degradation over time
2. **Jitter vs Cycle**: Tracks jitter increase
3. **Degradation Percentage**: Quantified degradation trends
4. **Pass/Fail Status**: Test status over time

### Statistical Summaries

```python
# Access comprehensive statistics
print(f"Test Duration: {results.duration:.1f} seconds")
print(f"Total Cycles: {results.total_cycles}")
print(f"Success Rate: {results.success_rate:.1%}")
print(f"Initial Eye Height: {results.initial_eye_height:.4f}V")
print(f"Final Eye Height: {results.final_eye_height:.4f}V")
print(f"Max Degradation: {results.max_degradation:.1f}%")
print(f"Degradation Rate: {results.degradation_rate:.4f}% per cycle")
```

## Performance Characteristics

### Execution Performance
- **Signal generation**: ~1ms per cycle for 10K samples
- **Analysis time**: ~5ms per cycle including eye diagram analysis
- **Memory usage**: Optimized for long-running tests (1000+ cycles)
- **CSV logging**: Efficient append-only logging

### Scalability
- **Cycle count**: Tested up to 10,000 cycles
- **Data rate**: Supports up to 112 Gbps (Ethernet PAM4)
- **Signal length**: Configurable from 1K to 100K samples
- **Output size**: ~1MB per 1000 cycles (CSV + logs)

## Use Cases

### 🔬 Research & Development
- **Signal integrity validation** over extended periods
- **Channel characterization** and degradation analysis
- **Protocol comparison** studies
- **Threshold optimization** research

### 🏭 Manufacturing & Production
- **Production testing** with automated pass/fail criteria
- **Burn-in testing** for reliability validation
- **Quality control** with statistical process control
- **Failure analysis** with detailed waveform capture

### 📚 Education & Training
- **Signal integrity education** with hands-on examples
- **Protocol understanding** through practical testing
- **Data analysis skills** development
- **Testing methodology** training

## Best Practices

### Test Configuration
- **Start with quick tests** (100 cycles) to validate setup
- **Use appropriate thresholds** based on protocol requirements
- **Enable waveform saving** only for debugging (storage intensive)
- **Monitor test progress** for long-running tests

### Data Analysis
- **Export CSV data** for external statistical analysis
- **Use interactive plots** for trend identification
- **Compare protocols** under similar conditions
- **Document test conditions** for reproducibility

### Performance Optimization
- **Adjust cycle duration** based on analysis requirements
- **Use appropriate signal lengths** for your protocol
- **Disable plots** for very long tests to save time
- **Monitor system resources** during extended tests

## Troubleshooting

### Common Issues

#### Test Hangs or Runs Slowly
- **Reduce cycle duration** for faster execution
- **Decrease signal length** to reduce processing time
- **Disable plot generation** for long tests
- **Check system resources** (CPU, memory)

#### High Failure Rates
- **Adjust thresholds** to appropriate levels for your protocol
- **Check signal generation** parameters
- **Verify protocol configuration** (data rate, voltage swing)
- **Review degradation model** parameters

#### Memory Issues
- **Reduce number of cycles** for memory-constrained systems
- **Disable waveform saving** to reduce memory usage
- **Use smaller signal lengths** 
- **Monitor memory usage** during tests

### Error Messages

#### "Samples per symbol must be positive"
- **Increase sample rate** in configuration
- **Reduce data rate** if necessary
- **Check protocol defaults** for reasonable values

#### "Analysis failed for cycle X"
- **Check signal generation** parameters
- **Verify eye diagram analyzer** configuration
- **Review error logs** for specific issues

## API Reference

### Core Classes

#### LoopbackStressTest
Main class for running stress tests.

```python
class LoopbackStressTest:
    def __init__(self, config: StressTestConfig)
    def run_stress_test(self) -> StressTestResults
    def _generate_loopback_signal(self, cycle: int) -> np.ndarray
    def _analyze_signal(self, signal: np.ndarray, cycle: int) -> CycleResults
```

#### StressTestConfig
Configuration for stress tests.

```python
@dataclass
class StressTestConfig:
    num_cycles: int = 1000
    cycle_duration: float = 1.0
    protocol: str = "USB4"
    data_rate: float = 20e9
    voltage_swing: float = 0.8
    # ... additional parameters
```

#### StressTestResults
Complete test results with statistics.

```python
@dataclass
class StressTestResults:
    config: StressTestConfig
    total_cycles: int
    passed_cycles: int
    failed_cycles: int
    cycle_results: List[CycleResults]
    # ... statistical properties
```

### Helper Functions

#### create_stress_test_config()
Create configuration with sensible defaults.

```python
def create_stress_test_config(
    protocol: str = "USB4",
    num_cycles: int = 1000,
    output_dir: str = "stress_test_results"
) -> StressTestConfig
```

## Examples

See the following example files for complete usage demonstrations:

- **[examples/loopback_stress_test_example.py](../examples/loopback_stress_test_example.py)** - Comprehensive usage examples
- **[scripts/run_loopback_stress_test.py](../scripts/run_loopback_stress_test.py)** - Command-line tool

## Integration

### With Existing Framework

```python
from serdes_validation_framework import create_validation_framework
from serdes_validation_framework.stress_testing import LoopbackStressTest

# Use with main framework
framework = create_validation_framework()
# ... run initial validation ...

# Follow up with stress testing
config = create_stress_test_config(protocol="USB4")
stress_test = LoopbackStressTest(config)
stress_results = stress_test.run_stress_test()
```

### With CI/CD Pipelines

```yaml
# Example GitHub Actions integration
- name: Run Stress Tests
  run: |
    python scripts/run_loopback_stress_test.py \
      --protocol USB4 \
      --cycles 100 \
      --quick \
      --output ci_stress_results
```

## Future Enhancements

### Planned Features (v1.4.2+)
- **Parallel testing** for faster execution
- **Advanced degradation models** with machine learning
- **Real-time monitoring** dashboard
- **Test automation** with scheduling

### Long-term Vision
- **Cloud integration** for distributed testing
- **Hardware integration** for real-world stress testing
- **Advanced analytics** with predictive modeling
- **Industry standard** compliance testing

---

*For more information, see the [API documentation](../api/stress_testing.md) and [examples](../examples/).*