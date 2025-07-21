# PCIe 6.0 API Reference

The PCIe 6.0 module provides comprehensive support for PCI Express 6.0 validation with dual-mode NRZ/PAM4 capabilities.

## Overview

The PCIe 6.0 implementation includes:

- **Complete PCIe 6.0 specification compliance** (64 GT/s)
- **NRZ/PAM4 dual-mode support** with seamless switching
- **Advanced link training** with multi-phase optimization
- **Comprehensive compliance testing** for electrical and timing parameters
- **Enhanced equalization algorithms** (LMS, RLS, CMA, Decision-directed)
- **Multi-lane support** (1-16 lanes) with skew analysis
- **Advanced eye diagram analysis** with statistical modeling

## Core Modules

### Constants and Specifications

```python
from serdes_validation_framework.protocols.pcie.constants import (
    SignalMode, PCIE_SPECS, PCIeSpecs, NRZSpecs, PAM4Specs
)

# Signal modes
mode = SignalMode.PAM4  # or SignalMode.NRZ

# Access specifications
pcie_specs = PCIE_SPECS['base']  # PCIeSpecs instance
nrz_specs = PCIE_SPECS['nrz']    # NRZSpecs instance
pam4_specs = PCIE_SPECS['pam4']  # PAM4Specs instance
```

#### SignalMode Enum

- `SignalMode.NRZ`: Non-Return-to-Zero signaling
- `SignalMode.PAM4`: 4-level Pulse Amplitude Modulation

#### Specifications

- **PCIeSpecs**: Base PCIe 6.0 specifications (64 GT/s, timing parameters)
- **NRZSpecs**: NRZ-specific parameters (amplitude, eye requirements)
- **PAM4Specs**: PAM4-specific parameters (levels, EVM limits)

### Mode Switching

```python
from serdes_validation_framework.instrument_control.mode_switcher import (
    create_mode_switcher, ModeSwitcher, ModeConfig
)

# Create mode switcher
switcher = create_mode_switcher(
    default_mode=SignalMode.NRZ,
    sample_rate=100e9,
    bandwidth=50e9
)

# Switch modes
result = switcher.switch_mode(SignalMode.PAM4)
print(f"Switch successful: {result.success}")
print(f"Switch time: {result.switch_time*1000:.2f} ms")
```

#### ModeSwitcher Class

**Methods:**
- `switch_mode(target_mode, config=None)`: Switch to target signal mode
- `get_current_mode()`: Get current signal mode
- `get_mode_config(mode)`: Get configuration for specific mode
- `update_mode_config(mode, config)`: Update mode configuration

### Signal Analysis

```python
from serdes_validation_framework.instrument_control.pcie_analyzer import (
    PCIeAnalyzer, PCIeConfig
)

# Create analyzer configuration
config = PCIeConfig(
    mode=SignalMode.PAM4,
    sample_rate=200e9,
    bandwidth=100e9,
    voltage_range=1.2,
    link_speed=64e9,
    lane_count=1
)

# Create analyzer
analyzer = PCIeAnalyzer(config)

# Analyze signal
signal_data = {
    'time': time_array,
    'voltage': voltage_array
}
results = analyzer.analyze_signal(signal_data)
```

#### Analysis Results

**NRZ Mode Results:**
- `level_separation`: Voltage separation between levels
- `snr_db`: Signal-to-noise ratio in dB
- `jitter_ps`: Jitter measurement in picoseconds

**PAM4 Mode Results:**
- `min_level_separation`: Minimum separation between PAM4 levels
- `rms_evm_percent`: RMS Error Vector Magnitude percentage
- `snr_db`: Signal-to-noise ratio in dB

### Link Training

```python
from serdes_validation_framework.protocols.pcie.link_training import (
    create_nrz_trainer, create_pam4_trainer, LinkTrainer
)

# Create trainer
trainer = create_pam4_trainer(
    target_ber=1e-12,
    max_iterations=1000
)

# Run training
result = trainer.run_training(signal_data)
print(f"Training success: {result.success}")
print(f"Final BER: {result.final_ber:.2e}")
print(f"Iterations: {result.iterations}")
```

#### Training Results

- `success`: Boolean indicating training convergence
- `final_ber`: Final bit error rate achieved
- `iterations`: Number of training iterations
- `equalizer_coeffs`: Final equalizer coefficients
- `snr_history`: SNR progression during training

### Equalization

```python
from serdes_validation_framework.protocols.pcie.equalization import (
    create_lms_equalizer, create_rls_equalizer, AdaptiveEqualizer
)

# Create equalizer
equalizer = create_lms_equalizer(
    num_forward_taps=11,
    num_feedback_taps=5,
    step_size=0.01
)

# Run equalization
result = equalizer.equalize_signal(distorted_signal)
print(f"Converged: {result.converged}")
print(f"Final MSE: {result.final_mse:.6f}")

# Apply equalization
equalized_signal = equalizer.apply_equalization(input_signal)
```

#### Equalization Algorithms

- **LMS (Least Mean Squares)**: Fast convergence, good for stationary signals
- **RLS (Recursive Least Squares)**: Better tracking, higher computational cost
- **CMA (Constant Modulus Algorithm)**: Blind equalization
- **Decision-Directed**: Uses symbol decisions for adaptation

### Compliance Testing

```python
from serdes_validation_framework.protocols.pcie.compliance import (
    ComplianceTestSuite, ComplianceConfig, ComplianceType
)

# Create test configuration
config = ComplianceConfig(
    test_pattern="PRBS31",
    sample_rate=200e9,
    record_length=100e-6,
    voltage_range=2.0,
    test_types=[ComplianceType.ELECTRICAL, ComplianceType.TIMING]
)

# Create test suite
test_suite = ComplianceTestSuite(config)

# Run compliance tests
results = test_suite.run_compliance_tests(time_data, voltage_data)

# Check overall status
overall_pass = test_suite.get_overall_status()
```

#### Compliance Test Types

- `ComplianceType.ELECTRICAL`: Voltage levels, swing, common mode
- `ComplianceType.TIMING`: Unit interval, jitter, eye opening
- `ComplianceType.PROTOCOL`: Protocol-specific compliance
- `ComplianceType.FULL`: All compliance tests

### Test Sequences

```python
from serdes_validation_framework.test_sequence.pcie_sequence import (
    create_single_lane_nrz_test, create_multi_lane_pam4_test,
    create_comprehensive_pcie_test
)

# Single lane test
test_sequence = create_single_lane_nrz_test(
    lane_id=0,
    sample_rate=100e9,
    bandwidth=50e9
)

# Multi-lane test
multi_lane_test = create_multi_lane_pam4_test(
    num_lanes=4,
    sample_rate=200e9,
    bandwidth=100e9
)

# Run complete test sequence
signal_data = {0: {'time': time_array, 'voltage': voltage_array}}
result = test_sequence.run_complete_sequence(signal_data)
```

#### Test Phases

- `TestPhase.INITIALIZATION`: Signal quality verification
- `TestPhase.LINK_TRAINING`: Adaptive link training
- `TestPhase.COMPLIANCE`: Compliance testing
- `TestPhase.STRESS_TEST`: Environmental stress testing
- `TestPhase.VALIDATION`: Final validation and cross-lane analysis

### Advanced Eye Diagram Analysis

```python
from serdes_validation_framework.data_analysis.eye_diagram import (
    create_nrz_eye_analyzer, create_pam4_eye_analyzer, AdvancedEyeAnalyzer
)

# Create eye analyzer
eye_analyzer = create_pam4_eye_analyzer(
    symbol_rate=32e9,
    samples_per_symbol=32
)

# Analyze eye diagram
eye_result = eye_analyzer.analyze_eye_diagram(time_data, voltage_data)

# Access results
print(f"Eye height: {eye_result.eye_height:.3f} V")
print(f"Eye width: {eye_result.eye_width*1e12:.1f} ps")
print(f"Q-factor: {eye_result.q_factor:.2f}")
```

#### Eye Analysis Features

- **Statistical eye modeling**: Density-based eye diagram generation
- **Jitter decomposition**: Random, deterministic, periodic, data-dependent
- **Bathtub curves**: Timing and voltage bathtub curve generation
- **Eye contour analysis**: Multi-level contour analysis
- **Eye mask compliance**: Automated mask violation detection

## Usage Examples

### Basic PCIe Validation Workflow

```python
import numpy as np
from serdes_validation_framework.protocols.pcie.constants import SignalMode
from serdes_validation_framework.instrument_control.pcie_analyzer import PCIeAnalyzer, PCIeConfig
from serdes_validation_framework.protocols.pcie.link_training import create_pam4_trainer
from serdes_validation_framework.protocols.pcie.compliance import ComplianceTestSuite, ComplianceConfig, ComplianceType

# 1. Configure analyzer
config = PCIeConfig(
    mode=SignalMode.PAM4,
    sample_rate=200e9,
    bandwidth=100e9,
    voltage_range=1.2,
    link_speed=64e9,
    lane_count=1
)

analyzer = PCIeAnalyzer(config)

# 2. Analyze signal quality
signal_data = {'time': time_array, 'voltage': voltage_array}
analysis_results = analyzer.analyze_signal(signal_data)
print(f"SNR: {analysis_results['snr_db']:.1f} dB")

# 3. Run link training
trainer = create_pam4_trainer(target_ber=1e-12)
training_result = trainer.run_training(signal_data)
print(f"Training converged: {training_result.success}")

# 4. Compliance testing
compliance_config = ComplianceConfig(
    test_pattern="PRBS31",
    sample_rate=200e9,
    record_length=100e-6,
    voltage_range=2.0,
    test_types=[ComplianceType.FULL]
)

test_suite = ComplianceTestSuite(compliance_config)
compliance_results = test_suite.run_compliance_tests(
    signal_data['time'], signal_data['voltage']
)
print(f"Compliance: {'PASS' if test_suite.get_overall_status() else 'FAIL'}")
```

### Multi-Lane Analysis

```python
from serdes_validation_framework.test_sequence.pcie_sequence import create_multi_lane_pam4_test

# Create 4-lane test
test_sequence = create_multi_lane_pam4_test(num_lanes=4)

# Prepare multi-lane data
multi_lane_data = {
    0: {'time': time_array_0, 'voltage': voltage_array_0},
    1: {'time': time_array_1, 'voltage': voltage_array_1},
    2: {'time': time_array_2, 'voltage': voltage_array_2},
    3: {'time': time_array_3, 'voltage': voltage_array_3}
}

# Run complete validation
result = test_sequence.run_complete_sequence(multi_lane_data)

# Check results
print(f"Overall status: {result.overall_status.name}")
print(f"Total duration: {result.total_duration:.2f} seconds")

# Lane-specific results
for lane_id, metrics in result.lane_results.items():
    print(f"Lane {lane_id} performance score: {metrics.get('performance_score', 0):.1f}")
```

## Performance Characteristics

- **Signal Analysis**: < 1 second for 10K samples
- **Mode Switching**: < 10 milliseconds
- **Link Training**: < 5 seconds for convergence
- **Compliance Testing**: < 3 seconds for full suite
- **Eye Diagram Analysis**: < 2 seconds for complete analysis

## Error Handling

All PCIe modules include comprehensive error handling:

- **Type validation**: Runtime checking of all parameters
- **Graceful degradation**: Fallback algorithms for edge cases
- **Detailed error messages**: Clear indication of failure causes
- **Recovery mechanisms**: Automatic retry and alternative methods

## Thread Safety

The PCIe modules are designed to be thread-safe for concurrent analysis of multiple lanes or signals.

## See Also

- [PCIe Tutorial](../tutorials/pcie_validation.md)
- [PCIe Examples](../examples/pcie_examples.md)
- [Compliance Testing Guide](../guides/compliance_testing.md)
- [Link Training Guide](../guides/link_training.md)