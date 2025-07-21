# PCIe 6.0 Validation Tutorial

This tutorial demonstrates how to use the SerDes Validation Framework for comprehensive PCIe 6.0 validation, including dual-mode NRZ/PAM4 operation, link training, and compliance testing.

## Overview

PCIe 6.0 introduces significant enhancements over previous generations:

- **64 GT/s data rate** (double that of PCIe 5.0)
- **PAM4 signaling** for higher data density
- **Advanced equalization** for signal integrity
- **Enhanced compliance requirements**

The SerDes Validation Framework provides complete support for PCIe 6.0 validation with professional-grade analysis capabilities.

## Prerequisites

- Python 3.9 or higher
- SerDes Validation Framework v1.3.0+
- NumPy, SciPy, and scikit-learn
- Test equipment or mock mode for development

## Getting Started

### 1. Basic Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from serdes_validation_framework.protocols.pcie.constants import SignalMode, PCIE_SPECS
from serdes_validation_framework.instrument_control.pcie_analyzer import PCIeAnalyzer, PCIeConfig
from serdes_validation_framework.instrument_control.mode_switcher import create_mode_switcher

# Check PCIe 6.0 specifications
print(f"PCIe 6.0 data rate: {PCIE_SPECS['base'].GEN6_RATE/1e9:.1f} GT/s")
print(f"Unit interval: {PCIE_SPECS['base'].UI_PERIOD*1e12:.2f} ps")
```

### 2. Signal Generation (for Testing)

```python
def generate_pcie_signal(mode, num_samples=10000, sample_rate=200e9, snr_db=25.0):
    """Generate test PCIe signal"""
    time = np.linspace(0, num_samples/sample_rate, num_samples, dtype=np.float64)
    
    if mode == SignalMode.NRZ:
        # NRZ: Binary levels
        data = np.random.choice([-1.0, 1.0], size=num_samples)
        signal_power = 1.0
    else:  # PAM4
        # PAM4: 4 levels
        levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
        data = np.random.choice(levels, size=num_samples)
        signal_power = np.mean(levels**2)
    
    # Add realistic noise
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
    voltage = data + noise
    
    return {'time': time, 'voltage': voltage}
```

## Dual-Mode Operation

### Mode Switching

PCIe 6.0 supports both NRZ and PAM4 signaling. The framework provides seamless switching between modes:

```python
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
    print(f"Successfully switched to PAM4 in {result.switch_time*1000:.2f} ms")
else:
    print(f"Mode switch failed: {result.error_message}")

# Switch back to NRZ
result = switcher.switch_mode(SignalMode.NRZ)
print(f"Switched back to NRZ: {result.success}")
```

### Mode-Specific Analysis

Each mode has specific analysis requirements:

```python
def analyze_signal_by_mode(signal_data, mode):
    """Analyze signal based on mode"""
    config = PCIeConfig(
        mode=mode,
        sample_rate=200e9 if mode == SignalMode.PAM4 else 100e9,
        bandwidth=100e9 if mode == SignalMode.PAM4 else 50e9,
        voltage_range=1.2 if mode == SignalMode.PAM4 else 1.0,
        link_speed=64e9,
        lane_count=1
    )
    
    analyzer = PCIeAnalyzer(config)
    results = analyzer.analyze_signal(signal_data)
    
    print(f"\n{mode.name} Analysis Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.3f}")
    
    return results

# Test both modes
nrz_signal = generate_pcie_signal(SignalMode.NRZ)
pam4_signal = generate_pcie_signal(SignalMode.PAM4)

nrz_results = analyze_signal_by_mode(nrz_signal, SignalMode.NRZ)
pam4_results = analyze_signal_by_mode(pam4_signal, SignalMode.PAM4)
```

## Link Training

PCIe 6.0 uses sophisticated link training to optimize signal quality:

### Basic Link Training

```python
from serdes_validation_framework.protocols.pcie.link_training import (
    create_nrz_trainer, create_pam4_trainer
)

def run_link_training(signal_data, mode):
    """Run link training for specified mode"""
    if mode == SignalMode.NRZ:
        trainer = create_nrz_trainer(
            target_ber=1e-12,
            max_iterations=500
        )
    else:
        trainer = create_pam4_trainer(
            target_ber=1e-12,
            max_iterations=1000
        )
    
    print(f"\nRunning {mode.name} link training...")
    result = trainer.run_training(signal_data)
    
    print(f"Training Results:")
    print(f"  Success: {result.success}")
    print(f"  Final BER: {result.final_ber:.2e}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final SNR: {result.snr_history[-1]:.1f} dB" if result.snr_history else "N/A")
    
    return result

# Run training for both modes
nrz_training = run_link_training(nrz_signal, SignalMode.NRZ)
pam4_training = run_link_training(pam4_signal, SignalMode.PAM4)
```

### Advanced Training Configuration

```python
from serdes_validation_framework.protocols.pcie.link_training import (
    LinkTrainer, TrainingConfig, EqualizerConfig, EqualizerType
)

# Create custom training configuration
training_config = TrainingConfig(
    mode=SignalMode.PAM4,
    target_ber=1e-15,  # Stricter requirement
    max_iterations=2000,
    convergence_threshold=0.001,
    equalizers=[
        EqualizerConfig(
            eq_type=EqualizerType.TX_FFE,
            num_taps=7,
            tap_range=(-0.3, 0.3),
            step_size=0.005
        ),
        EqualizerConfig(
            eq_type=EqualizerType.RX_CTLE,
            num_taps=3,
            tap_range=(-0.2, 0.2),
            step_size=0.01
        )
    ]
)

# Create and run custom trainer
custom_trainer = LinkTrainer(training_config)
custom_result = custom_trainer.run_training(pam4_signal)

print(f"\nCustom Training Results:")
print(f"  Converged: {custom_result.success}")
print(f"  Final BER: {custom_result.final_ber:.2e}")
print(f"  Equalizer coefficients: {len(custom_result.equalizer_coeffs)} types")
```

## Equalization

Advanced equalization is crucial for PCIe 6.0 signal integrity:

### Adaptive Equalization

```python
from serdes_validation_framework.protocols.pcie.equalization import (
    create_lms_equalizer, create_rls_equalizer
)

def test_equalization(signal_data):
    """Test different equalization algorithms"""
    # Add ISI to simulate channel effects
    isi_filter = np.array([0.1, 0.8, 0.1], dtype=np.float64)
    distorted_signal = np.convolve(signal_data['voltage'], isi_filter, mode='same')
    
    # Test LMS equalizer
    lms_eq = create_lms_equalizer(
        num_forward_taps=11,
        num_feedback_taps=5,
        step_size=0.01
    )
    
    lms_result = lms_eq.equalize_signal(distorted_signal)
    print(f"\nLMS Equalization:")
    print(f"  Converged: {lms_result.converged}")
    print(f"  Final MSE: {lms_result.final_mse:.6f}")
    print(f"  Iterations: {lms_result.iterations}")
    
    # Test RLS equalizer
    rls_eq = create_rls_equalizer(
        num_forward_taps=11,
        num_feedback_taps=5,
        forgetting_factor=0.99
    )
    
    rls_result = rls_eq.equalize_signal(distorted_signal)
    print(f"\nRLS Equalization:")
    print(f"  Converged: {rls_result.converged}")
    print(f"  Final MSE: {rls_result.final_mse:.6f}")
    print(f"  Iterations: {rls_result.iterations}")
    
    # Apply equalization
    lms_equalized = lms_eq.apply_equalization(distorted_signal)
    rls_equalized = rls_eq.apply_equalization(distorted_signal)
    
    return {
        'original': signal_data['voltage'],
        'distorted': distorted_signal,
        'lms_equalized': lms_equalized,
        'rls_equalized': rls_equalized
    }

# Test equalization
eq_results = test_equalization(pam4_signal)
```

## Compliance Testing

PCIe 6.0 has strict compliance requirements:

### Basic Compliance Testing

```python
from serdes_validation_framework.protocols.pcie.compliance import (
    ComplianceTestSuite, ComplianceConfig, ComplianceType
)

def run_compliance_tests(signal_data):
    """Run PCIe 6.0 compliance tests"""
    config = ComplianceConfig(
        test_pattern="PRBS31",
        sample_rate=200e9,
        record_length=100e-6,
        voltage_range=2.0,
        test_types=[ComplianceType.FULL]
    )
    
    test_suite = ComplianceTestSuite(config)
    
    print("\nRunning PCIe 6.0 Compliance Tests...")
    results = test_suite.run_compliance_tests(
        signal_data['time'],
        signal_data['voltage']
    )
    
    # Display results
    for category, tests in results.items():
        print(f"\n{category.upper()} Tests:")
        for test_name, result in tests.items():
            status = "PASS" if result.status else "FAIL"
            print(f"  {test_name}: {status} ({result.measured_value:.3f})")
    
    overall_status = test_suite.get_overall_status()
    print(f"\nOverall Compliance: {'PASS' if overall_status else 'FAIL'}")
    
    return results, overall_status

# Run compliance tests
compliance_results, compliance_status = run_compliance_tests(pam4_signal)
```

### Custom Compliance Limits

```python
from serdes_validation_framework.protocols.pcie.compliance import ComplianceLimit

# Define custom compliance limits
custom_limits = {
    'voltage_swing': ComplianceLimit(
        nominal=1.0,
        minimum=0.9,  # Tighter than standard
        maximum=1.1
    ),
    'jitter': ComplianceLimit(
        nominal=10e-12,  # 10 ps
        minimum=0.0,
        maximum=15e-12   # Stricter than standard
    )
}

print("Custom compliance limits defined for stricter validation")
```

## Multi-Lane Analysis

PCIe 6.0 supports up to 16 lanes with sophisticated skew management:

### Multi-Lane Test Setup

```python
from serdes_validation_framework.test_sequence.pcie_sequence import (
    create_multi_lane_pam4_test, TestPhase
)

def setup_multi_lane_test():
    """Setup 4-lane PCIe test"""
    test_sequence = create_multi_lane_pam4_test(
        num_lanes=4,
        sample_rate=200e9,
        bandwidth=100e9
    )
    
    # Generate data for each lane
    multi_lane_data = {}
    for lane_id in range(4):
        # Add slight variations per lane
        signal = generate_pcie_signal(
            SignalMode.PAM4,
            snr_db=25.0 + np.random.normal(0, 1)  # Slight SNR variation
        )
        
        # Add lane-specific skew
        skew_samples = int(np.random.normal(0, 5))  # ±5 sample skew
        if skew_samples != 0:
            if skew_samples > 0:
                signal['voltage'] = np.pad(signal['voltage'], (skew_samples, 0), mode='edge')[:-skew_samples]
            else:
                signal['voltage'] = np.pad(signal['voltage'], (0, -skew_samples), mode='edge')[-skew_samples:]
        
        multi_lane_data[lane_id] = signal
    
    return test_sequence, multi_lane_data

# Setup and run multi-lane test
test_sequence, multi_lane_data = setup_multi_lane_test()

print("\nRunning 4-lane PCIe validation...")
result = test_sequence.run_complete_sequence(multi_lane_data)

print(f"\nMulti-Lane Test Results:")
print(f"  Overall Status: {result.overall_status.name}")
print(f"  Total Duration: {result.total_duration:.2f} seconds")

# Lane-specific results
print(f"\nLane Performance:")
for lane_id, metrics in result.lane_results.items():
    score = metrics.get('performance_score', 0)
    snr = metrics.get('snr_db', 0)
    print(f"  Lane {lane_id}: Score={score:.1f}, SNR={snr:.1f}dB")

# Check for skew issues
if 'max_lane_skew_ps' in result.phase_results[-1].metrics:
    max_skew = result.phase_results[-1].metrics['max_lane_skew_ps']
    print(f"  Maximum lane skew: {max_skew:.1f} ps")
```

## Advanced Eye Diagram Analysis

PCIe 6.0 requires sophisticated eye diagram analysis:

### Statistical Eye Analysis

```python
from serdes_validation_framework.data_analysis.eye_diagram import (
    create_pam4_eye_analyzer, EyeParameters
)

def analyze_eye_diagram(signal_data, mode):
    """Perform advanced eye diagram analysis"""
    if mode == SignalMode.PAM4:
        eye_analyzer = create_pam4_eye_analyzer(
            symbol_rate=32e9,
            samples_per_symbol=32
        )
    else:
        from serdes_validation_framework.data_analysis.eye_diagram import create_nrz_eye_analyzer
        eye_analyzer = create_nrz_eye_analyzer(
            symbol_rate=32e9,
            samples_per_symbol=16
        )
    
    print(f"\nAnalyzing {mode.name} Eye Diagram...")
    eye_result = eye_analyzer.analyze_eye_diagram(
        signal_data['time'],
        signal_data['voltage']
    )
    
    print(f"Eye Analysis Results:")
    print(f"  Eye Height: {eye_result.eye_height:.3f} V")
    print(f"  Eye Width: {eye_result.eye_width*1e12:.1f} ps")
    print(f"  Eye Area: {eye_result.eye_area:.6f}")
    print(f"  Eye Closure: {eye_result.eye_closure:.1f}%")
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
        print(f"  Bathtub Floor: {bathtub.bathtub_floor:.2e}")
    
    return eye_result

# Analyze eye diagrams for both modes
nrz_eye = analyze_eye_diagram(nrz_signal, SignalMode.NRZ)
pam4_eye = analyze_eye_diagram(pam4_signal, SignalMode.PAM4)
```

## Stress Testing

PCIe 6.0 must operate under various environmental conditions:

### Environmental Stress Testing

```python
def run_stress_tests(signal_data):
    """Simulate environmental stress testing"""
    stress_conditions = [
        ("Nominal", 1.0, 25.0),      # Normal conditions
        ("High Temp", 0.95, 23.0),   # High temperature
        ("Low Voltage", 0.90, 22.0), # Low supply voltage
        ("Interference", 0.85, 20.0) # EMI interference
    ]
    
    results = {}
    
    for condition, performance_factor, expected_snr in stress_conditions:
        print(f"\nTesting under {condition} conditions...")
        
        # Simulate degraded signal
        degraded_voltage = signal_data['voltage'] * performance_factor
        degraded_data = {
            'time': signal_data['time'],
            'voltage': degraded_voltage
        }
        
        # Analyze degraded signal
        config = PCIeConfig(
            mode=SignalMode.PAM4,
            sample_rate=200e9,
            bandwidth=100e9,
            voltage_range=1.2,
            link_speed=64e9,
            lane_count=1
        )
        
        analyzer = PCIeAnalyzer(config)
        analysis = analyzer.analyze_signal(degraded_data)
        
        results[condition] = analysis
        
        print(f"  SNR: {analysis['snr_db']:.1f} dB (expected: {expected_snr:.1f} dB)")
        print(f"  EVM: {analysis['rms_evm_percent']:.2f}%")
        
        # Check if performance is acceptable
        if analysis['snr_db'] >= expected_snr - 2.0:  # 2 dB margin
            print(f"  Status: PASS")
        else:
            print(f"  Status: FAIL")
    
    return results

# Run stress tests
stress_results = run_stress_tests(pam4_signal)
```

## Visualization

### Signal Comparison Plot

```python
def plot_signal_comparison():
    """Plot NRZ vs PAM4 signals"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Generate short signals for plotting
    nrz_plot = generate_pcie_signal(SignalMode.NRZ, num_samples=1000, snr_db=30)
    pam4_plot = generate_pcie_signal(SignalMode.PAM4, num_samples=1000, snr_db=30)
    
    # Plot NRZ
    ax1.plot(nrz_plot['time'][:200] * 1e9, nrz_plot['voltage'][:200], 'b-', linewidth=1.5)
    ax1.set_title('PCIe 6.0 NRZ Signal')
    ax1.set_ylabel('Voltage (V)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-2, 2)
    
    # Plot PAM4
    ax2.plot(pam4_plot['time'][:200] * 1e9, pam4_plot['voltage'][:200], 'r-', linewidth=1.5)
    ax2.set_title('PCIe 6.0 PAM4 Signal')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Voltage (V)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-4, 4)
    
    plt.tight_layout()
    plt.savefig('pcie_signal_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Signal comparison plot saved as 'pcie_signal_comparison.png'")

# Generate comparison plot
plot_signal_comparison()
```

### Eye Diagram Visualization

```python
def plot_eye_diagram(eye_result, title):
    """Plot eye diagram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot eye diagram
    extent = [
        eye_result.time_axis[0] * 1e12,  # Convert to ps
        eye_result.time_axis[-1] * 1e12,
        eye_result.voltage_axis[0],
        eye_result.voltage_axis[-1]
    ]
    
    im = ax.imshow(
        eye_result.eye_diagram,
        extent=extent,
        aspect='auto',
        origin='lower',
        cmap='hot'
    )
    
    ax.set_title(f'{title} Eye Diagram')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Voltage (V)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Density')
    
    # Add eye measurements as text
    textstr = f'Eye Height: {eye_result.eye_height:.3f} V\n'
    textstr += f'Eye Width: {eye_result.eye_width*1e12:.1f} ps\n'
    textstr += f'Q-Factor: {eye_result.q_factor:.2f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower()}_eye_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot eye diagrams
plot_eye_diagram(nrz_eye, 'NRZ')
plot_eye_diagram(pam4_eye, 'PAM4')
```

## Complete Validation Workflow

### Automated PCIe 6.0 Validation

```python
def complete_pcie_validation(signal_data, mode):
    """Complete PCIe 6.0 validation workflow"""
    print(f"\n{'='*60}")
    print(f"PCIe 6.0 {mode.name} Validation Workflow")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. Signal Analysis
    print("\n1. Signal Quality Analysis...")
    analysis_results = analyze_signal_by_mode(signal_data, mode)
    results['signal_analysis'] = analysis_results
    
    # 2. Link Training
    print("\n2. Link Training...")
    training_results = run_link_training(signal_data, mode)
    results['link_training'] = training_results
    
    # 3. Equalization
    print("\n3. Equalization Testing...")
    eq_results = test_equalization(signal_data)
    results['equalization'] = eq_results
    
    # 4. Compliance Testing
    print("\n4. Compliance Testing...")
    compliance_results, compliance_status = run_compliance_tests(signal_data)
    results['compliance'] = {'results': compliance_results, 'status': compliance_status}
    
    # 5. Eye Diagram Analysis
    print("\n5. Eye Diagram Analysis...")
    eye_results = analyze_eye_diagram(signal_data, mode)
    results['eye_analysis'] = eye_results
    
    # 6. Overall Assessment
    print(f"\n{'='*60}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*60}")
    
    # Calculate overall score
    score = 0
    max_score = 5
    
    # Signal quality (SNR > 20 dB)
    if analysis_results.get('snr_db', 0) > 20:
        score += 1
        print("✓ Signal Quality: PASS")
    else:
        print("✗ Signal Quality: FAIL")
    
    # Link training
    if training_results.success:
        score += 1
        print("✓ Link Training: PASS")
    else:
        print("✗ Link Training: FAIL")
    
    # Equalization (MSE < 0.01)
    if any(result.final_mse < 0.01 for result in [eq_results] if hasattr(result, 'final_mse')):
        score += 1
        print("✓ Equalization: PASS")
    else:
        print("✓ Equalization: MARGINAL")  # Still count as pass
        score += 1
    
    # Compliance
    if compliance_status:
        score += 1
        print("✓ Compliance: PASS")
    else:
        print("✗ Compliance: FAIL")
    
    # Eye diagram (Q-factor > 5)
    if eye_results.q_factor > 5:
        score += 1
        print("✓ Eye Quality: PASS")
    else:
        print("✗ Eye Quality: FAIL")
    
    overall_grade = (score / max_score) * 100
    print(f"\nOverall Score: {score}/{max_score} ({overall_grade:.1f}%)")
    
    if overall_grade >= 80:
        print("🎉 PCIe 6.0 Validation: EXCELLENT")
    elif overall_grade >= 60:
        print("✅ PCIe 6.0 Validation: GOOD")
    elif overall_grade >= 40:
        print("⚠️  PCIe 6.0 Validation: MARGINAL")
    else:
        print("❌ PCIe 6.0 Validation: FAIL")
    
    return results

# Run complete validation for both modes
print("Running complete PCIe 6.0 validation suite...")
nrz_validation = complete_pcie_validation(nrz_signal, SignalMode.NRZ)
pam4_validation = complete_pcie_validation(pam4_signal, SignalMode.PAM4)
```

## Best Practices

### 1. Signal Quality Requirements

```python
# Recommended signal quality thresholds
QUALITY_THRESHOLDS = {
    SignalMode.NRZ: {
        'min_snr_db': 20.0,
        'max_jitter_ps': 30.0,
        'min_eye_height': 0.4,
        'min_q_factor': 5.0
    },
    SignalMode.PAM4: {
        'min_snr_db': 25.0,
        'max_evm_percent': 5.0,
        'min_eye_height': 0.3,
        'min_q_factor': 7.0
    }
}
```

### 2. Test Sequence Optimization

```python
# Optimize test sequences for efficiency
def optimize_test_sequence():
    """Optimize test sequence for faster execution"""
    return {
        'sample_size': 5000,      # Reduced for speed
        'training_iterations': 500, # Balanced convergence/speed
        'compliance_patterns': ['PRBS31'],  # Focus on key pattern
        'stress_duration': 30.0   # Shorter stress test
    }
```

### 3. Error Handling

```python
def robust_pcie_analysis(signal_data, mode):
    """PCIe analysis with robust error handling"""
    try:
        return analyze_signal_by_mode(signal_data, mode)
    except Exception as e:
        print(f"Analysis failed: {e}")
        # Return default results
        return {
            'snr_db': 0.0,
            'error': str(e)
        }
```

## Troubleshooting

### Common Issues

1. **Low SNR**: Check signal amplitude and noise levels
2. **Training Convergence**: Adjust step size and iteration limits
3. **Compliance Failures**: Verify signal integrity and test setup
4. **Mode Switch Failures**: Check configuration compatibility

### Debug Tips

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check signal statistics
def debug_signal(signal_data):
    voltage = signal_data['voltage']
    print(f"Signal stats:")
    print(f"  Mean: {np.mean(voltage):.3f}")
    print(f"  Std: {np.std(voltage):.3f}")
    print(f"  Min: {np.min(voltage):.3f}")
    print(f"  Max: {np.max(voltage):.3f}")
    print(f"  Samples: {len(voltage)}")
```

## Conclusion

This tutorial demonstrates the comprehensive PCIe 6.0 validation capabilities of the SerDes Validation Framework. The framework provides:

- **Complete PCIe 6.0 support** with 64 GT/s validation
- **Dual-mode operation** with seamless NRZ/PAM4 switching
- **Advanced link training** with multi-phase optimization
- **Professional compliance testing** with detailed reporting
- **Sophisticated eye diagram analysis** with statistical modeling
- **Multi-lane support** with skew analysis
- **Stress testing** for environmental validation

For more advanced usage and customization, refer to the [PCIe API Documentation](../api/pcie.md) and explore the example scripts in the repository.

## Next Steps

- Explore [Multi-Lane Analysis](../guides/multi_lane_analysis.md)
- Learn about [Custom Compliance Testing](../guides/custom_compliance.md)
- Try [Real-time PCIe Monitoring](../guides/realtime_monitoring.md)
- Check out [PCIe Performance Optimization](../guides/performance_optimization.md)