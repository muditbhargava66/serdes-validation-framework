# PCIe 6.0 Examples

This document provides comprehensive examples for PCIe 6.0 validation using the SerDes Validation Framework v1.3.0.

## Overview

The PCIe 6.0 examples demonstrate:
- Complete PCIe 6.0 specification compliance testing
- NRZ/PAM4 dual-mode operation
- Multi-lane analysis with skew detection
- Advanced link training and equalization
- Comprehensive compliance testing
- Performance benchmarking and optimization

## Example Files

### Core PCIe Example
- **`examples/pcie_example.py`** - Complete PCIe 6.0 validation workflow

### Enhanced Examples with PCIe Integration
- **`examples/data_analysis_example.py`** - Advanced data analysis with PCIe support
- **`examples/test_sequence_example.py`** - Test automation with PCIe integration

### Production Scripts
- **`scripts/pcie_validation.py`** - Production PCIe validation suite

## Quick Start Examples

### Basic PCIe Validation

```python
#!/usr/bin/env python3
"""Basic PCIe 6.0 validation example"""

from serdes_validation_framework.protocols.pcie.constants import SignalMode
from serdes_validation_framework.instrument_control.pcie_analyzer import PCIeAnalyzer, PCIeConfig

# Configure PCIe analyzer for PAM4 mode
config = PCIeConfig(
    mode=SignalMode.PAM4,
    sample_rate=200e9,  # 200 GSa/s
    bandwidth=100e9,    # 100 GHz
    voltage_range=1.2,  # 1.2V range
    link_speed=64e9,    # 64 GT/s
    lane_count=1
)

# Create analyzer
analyzer = PCIeAnalyzer(config)

# Analyze signal (signal_data contains time and voltage arrays)
results = analyzer.analyze_signal(signal_data)
print(f"SNR: {results['snr_db']:.1f} dB")
print(f"EVM: {results['rms_evm_percent']:.2f}%")
```

### Mode Switching Example

```python
#!/usr/bin/env python3
"""PCIe dual-mode switching example"""

from serdes_validation_framework.protocols.pcie.constants import SignalMode
from serdes_validation_framework.instrument_control.mode_switcher import create_mode_switcher

# Create mode switcher
switcher = create_mode_switcher(
    default_mode=SignalMode.NRZ,
    sample_rate=100e9,
    bandwidth=50e9
)

print(f"Current mode: {switcher.get_current_mode().name}")

# Switch to PAM4
result = switcher.switch_mode(SignalMode.PAM4)
if result.success:
    print(f"Switched to PAM4 in {result.switch_time*1000:.2f} ms")
else:
    print(f"Switch failed: {result.error_message}")
```

### Link Training Example

```python
#!/usr/bin/env python3
"""PCIe link training example"""

from serdes_validation_framework.protocols.pcie.constants import SignalMode
from serdes_validation_framework.protocols.pcie.link_training import create_pam4_trainer

# Create PAM4 trainer
trainer = create_pam4_trainer(
    target_ber=1e-12,
    max_iterations=1000
)

# Run training
result = trainer.run_training(signal_data)

print(f"Training Results:")
print(f"  Success: {result.success}")
print(f"  Final BER: {result.final_ber:.2e}")
print(f"  Iterations: {result.iterations}")
print(f"  Final SNR: {result.snr_history[-1]:.1f} dB")
```

### Compliance Testing Example

```python
#!/usr/bin/env python3
"""PCIe compliance testing example"""

from serdes_validation_framework.protocols.pcie.compliance import (
    ComplianceTestSuite, ComplianceConfig, ComplianceType
)

# Create compliance test configuration
config = ComplianceConfig(
    test_pattern="PRBS31",
    sample_rate=200e9,
    record_length=100e-6,
    voltage_range=2.0,
    test_types=[ComplianceType.FULL]
)

# Create and run test suite
test_suite = ComplianceTestSuite(config)
results = test_suite.run_compliance_tests(time_data, voltage_data)

# Check overall compliance
overall_status = test_suite.get_overall_status()
print(f"Overall Compliance: {'PASS' if overall_status else 'FAIL'}")

# Display detailed results
for category, tests in results.items():
    print(f"\n{category.upper()} Tests:")
    for test_name, result in tests.items():
        status = "PASS" if result.status else "FAIL"
        print(f"  {test_name}: {status} ({result.measured_value:.3f})")
```

### Multi-Lane Analysis Example

```python
#!/usr/bin/env python3
"""PCIe multi-lane analysis example"""

from serdes_validation_framework.test_sequence.pcie_sequence import create_multi_lane_pam4_test

# Create 4-lane PAM4 test
test_sequence = create_multi_lane_pam4_test(
    num_lanes=4,
    sample_rate=200e9,
    bandwidth=100e9
)

# Prepare multi-lane data (4 lanes)
multi_lane_data = {
    0: {'time': time_array_0, 'voltage': voltage_array_0},
    1: {'time': time_array_1, 'voltage': voltage_array_1},
    2: {'time': time_array_2, 'voltage': voltage_array_2},
    3: {'time': time_array_3, 'voltage': voltage_array_3}
}

# Run complete validation
result = test_sequence.run_complete_sequence(multi_lane_data)

print(f"Multi-Lane Results:")
print(f"  Overall Status: {result.overall_status.name}")
print(f"  Duration: {result.total_duration:.2f} seconds")
print(f"  Phases: {len(result.phase_results)}")

# Lane-specific results
for lane_id, metrics in result.lane_results.items():
    score = metrics.get('performance_score', 0)
    print(f"  Lane {lane_id}: {score:.1f}% performance")
```

## Advanced Examples

### Eye Diagram Analysis

```python
#!/usr/bin/env python3
"""Advanced eye diagram analysis example"""

from serdes_validation_framework.data_analysis.eye_diagram import create_pam4_eye_analyzer

# Create PAM4 eye analyzer
eye_analyzer = create_pam4_eye_analyzer(
    symbol_rate=32e9,
    samples_per_symbol=32
)

# Analyze eye diagram
eye_result = eye_analyzer.analyze_eye_diagram(time_data, voltage_data)

print(f"Eye Analysis Results:")
print(f"  Eye Height: {eye_result.eye_height:.3f} V")
print(f"  Eye Width: {eye_result.eye_width*1e12:.1f} ps")
print(f"  Q-Factor: {eye_result.q_factor:.2f}")

# Jitter analysis
if eye_result.jitter_analysis:
    jitter = eye_result.jitter_analysis
    print(f"  Total Jitter: {jitter.total_jitter*1e12:.2f} ps")
    print(f"  Random Jitter: {jitter.random_jitter*1e12:.2f} ps")
    print(f"  Deterministic Jitter: {jitter.deterministic_jitter*1e12:.2f} ps")

# Bathtub curves
if eye_result.timing_bathtub:
    bathtub = eye_result.timing_bathtub
    print(f"  Timing Eye Opening: {bathtub.eye_opening*1e12:.1f} ps")
```

### Equalization Example

```python
#!/usr/bin/env python3
"""PCIe equalization example"""

from serdes_validation_framework.protocols.pcie.equalization import (
    create_lms_equalizer, create_rls_equalizer
)

# Create LMS equalizer
lms_eq = create_lms_equalizer(
    num_forward_taps=11,
    num_feedback_taps=5,
    step_size=0.01
)

# Create RLS equalizer
rls_eq = create_rls_equalizer(
    num_forward_taps=11,
    num_feedback_taps=5,
    forgetting_factor=0.99
)

# Test both equalizers
for name, equalizer in [("LMS", lms_eq), ("RLS", rls_eq)]:
    result = equalizer.equalize_signal(distorted_signal)
    print(f"{name} Equalizer:")
    print(f"  Converged: {result.converged}")
    print(f"  Final MSE: {result.final_mse:.6f}")
    print(f"  Iterations: {result.iterations}")
    
    # Apply equalization
    equalized = equalizer.apply_equalization(input_signal)
    print(f"  Output length: {len(equalized)} samples")
```

## Running the Examples

### Command Line Usage

```bash
# Run complete PCIe example
python examples/pcie_example.py

# Run with verbose output
python examples/pcie_example.py --verbose

# Run data analysis with PCIe integration
python examples/data_analysis_example.py

# Run test sequence with PCIe support
python examples/test_sequence_example.py
```

### Production Script Usage

```bash
# Complete PCIe validation
python scripts/pcie_validation.py --mode both --lanes 4 --verbose

# PAM4-only validation with benchmarks
python scripts/pcie_validation.py --mode pam4 --benchmark

# Multi-lane validation
python scripts/pcie_validation.py --mode both --lanes 8 --samples 20000
```

### Mock Mode Testing

```bash
# Force mock mode (no hardware required)
export SVF_MOCK_MODE=1
python examples/pcie_example.py

# Auto-detect mode (default)
unset SVF_MOCK_MODE
python examples/pcie_example.py
```

## Performance Examples

### Benchmarking Code

```python
#!/usr/bin/env python3
"""PCIe performance benchmarking example"""

import time
from serdes_validation_framework.protocols.pcie.constants import SignalMode
from serdes_validation_framework.instrument_control.pcie_analyzer import PCIeAnalyzer, PCIeConfig

def benchmark_analysis(mode, sample_counts):
    """Benchmark signal analysis performance"""
    results = {}
    
    for sample_count in sample_counts:
        # Generate test signal
        signal_data = generate_test_signal(mode, sample_count)
        
        # Configure analyzer
        config = PCIeConfig(
            mode=mode,
            sample_rate=200e9 if mode == SignalMode.PAM4 else 100e9,
            bandwidth=100e9 if mode == SignalMode.PAM4 else 50e9,
            voltage_range=1.2 if mode == SignalMode.PAM4 else 1.0,
            link_speed=64e9,
            lane_count=1
        )
        
        analyzer = PCIeAnalyzer(config)
        
        # Time the analysis
        start_time = time.time()
        analyzer.analyze_signal(signal_data)
        analysis_time = time.time() - start_time
        
        throughput = sample_count / analysis_time
        results[sample_count] = {
            'time': analysis_time,
            'throughput': throughput
        }
        
        print(f"{mode.name} - {sample_count} samples: {analysis_time:.3f}s ({throughput:.0f} samples/s)")
    
    return results

# Run benchmarks
sample_counts = [1000, 5000, 10000, 20000]
nrz_results = benchmark_analysis(SignalMode.NRZ, sample_counts)
pam4_results = benchmark_analysis(SignalMode.PAM4, sample_counts)
```

### Expected Performance

```python
# Typical performance metrics:
PERFORMANCE_TARGETS = {
    'signal_analysis': {
        'nrz_10k_samples': '<1.0s',
        'pam4_10k_samples': '<1.5s'
    },
    'mode_switching': {
        'nrz_to_pam4': '<10ms',
        'pam4_to_nrz': '<10ms'
    },
    'link_training': {
        'nrz_convergence': '<5s',
        'pam4_convergence': '<8s'
    },
    'compliance_testing': {
        'full_suite': '<3s',
        'electrical_only': '<1s'
    }
}
```

## Integration Examples

### CI/CD Integration

```yaml
# GitHub Actions example
name: PCIe Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install -r requirements.txt
    
    - name: Run PCIe validation
      run: |
        export SVF_MOCK_MODE=1
        python scripts/pcie_validation.py --mode both --lanes 4
    
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: pcie-validation-results
        path: ./results/
```

### Docker Integration

```dockerfile
# Dockerfile for PCIe validation
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .
RUN pip install -r requirements.txt

# Set mock mode for containerized testing
ENV SVF_MOCK_MODE=1

CMD ["python", "scripts/pcie_validation.py", "--mode", "both", "--lanes", "4"]
```

### Python Integration

```python
#!/usr/bin/env python3
"""Integration example for custom applications"""

import subprocess
import sys
import json

def run_pcie_validation(mode='both', lanes=4, samples=10000):
    """Run PCIe validation and return results"""
    cmd = [
        sys.executable, 'scripts/pcie_validation.py',
        '--mode', mode,
        '--lanes', str(lanes),
        '--samples', str(samples),
        '--output', './validation_results'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("PCIe validation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"PCIe validation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

# Usage
if __name__ == "__main__":
    success = run_pcie_validation(mode='pam4', lanes=2, samples=5000)
    if success:
        print("Validation passed - ready for production")
    else:
        print("Validation failed - check configuration")
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure proper installation
   pip install -e .
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Hardware Connection Issues**
   ```bash
   # Use mock mode for development
   export SVF_MOCK_MODE=1
   python examples/pcie_example.py
   ```

3. **Performance Issues**
   ```bash
   # Reduce sample count for faster testing
   python scripts/pcie_validation.py --samples 5000
   
   # Use single mode for faster validation
   python scripts/pcie_validation.py --mode nrz
   ```

### Debug Mode

```bash
# Enable verbose logging
python examples/pcie_example.py --verbose

# Check log files
tail -f pcie_validation.log
```

## See Also

- [PCIe API Reference](../api/pcie.md)
- [PCIe Tutorial](../tutorials/pcie_validation.md)
- [Scripts Documentation](../../scripts/README.md)
- [Examples Documentation](../../examples/README.md)