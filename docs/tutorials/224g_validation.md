# 224G Ethernet Validation Tutorial

## Introduction

This tutorial guides you through the process of validating 224G Ethernet interfaces using the SerDes Validation Framework. We'll cover equipment setup, test sequence configuration, and result analysis.

## Prerequisites

- Python 3.10 or higher
- SerDes Validation Framework installed
- High-bandwidth oscilloscope (120+ GHz)
- Pattern generator with PAM4 support
- GPIB or equivalent instrument control interface

## Equipment Setup

### Oscilloscope Configuration

1. Connect the oscilloscope:
   ```python
   from serdes_validation_framework.instrument_control.scope_224g import (
       HighBandwidthScope,
       ScopeConfig
   )

   # Initialize scope
   scope = HighBandwidthScope("GPIB0::7::INSTR")
   
   # Configure for 224G measurements
   config = ScopeConfig(
       sampling_rate=256e9,  # 256 GSa/s
       bandwidth=120e9,      # 120 GHz
       timebase=5e-12,      # 5 ps/div
       voltage_range=0.8    # 0.8V
   )
   scope.configure_for_224g(config)
   ```

2. Verify configuration:
   ```python
   # Check settings
   print(f"Sampling rate: {scope.default_config.sampling_rate/1e9:.1f} GSa/s")
   print(f"Bandwidth: {scope.default_config.bandwidth/1e9:.1f} GHz")
   ```

### Pattern Generator Setup

1. Configure the pattern generator:
   ```python
   from serdes_validation_framework.test_sequence.eth_224g_sequence import (
       Ethernet224GTestSequence
   )
   
   sequence = Ethernet224GTestSequence()
   sequence.setup_instruments([
       "GPIB0::7::INSTR",    # Scope
       "GPIB0::10::INSTR"    # Pattern generator
   ])
   ```

## Running Tests

### Link Training Validation

1. Basic training test:
   ```python
   # Run training with default settings
   training_results = sequence.run_link_training_test(
       scope_resource="GPIB0::7::INSTR",
       pattern_gen_resource="GPIB0::10::INSTR"
   )
   
   # Check results
   print(f"Training status: {training_results.convergence_status}")
   print(f"Training time: {training_results.training_time:.2f} seconds")
   print(f"Final error: {training_results.adaptation_error:.6f}")
   ```

2. Custom training configuration:
   ```python
   from serdes_validation_framework.protocols.ethernet_224g.training import (
       TrainingConfig
   )
   
   # Configure training parameters
   config = TrainingConfig(
       adaptation_rate=0.01,
       max_iterations=1000,
       convergence_threshold=0.1,
       min_snr=20.0
   )
   
   # Run training with custom config
   training_results = sequence.run_link_training_test(
       scope_resource="GPIB0::7::INSTR",
       pattern_gen_resource="GPIB0::10::INSTR",
       config=config
   )
   ```

### Compliance Testing

1. Run full compliance suite:
   ```python
   # Run all compliance tests
   compliance_results = sequence.run_compliance_test_suite(
       scope_resource="GPIB0::7::INSTR",
       pattern_gen_resource="GPIB0::10::INSTR"
   )
   
   # Check overall status
   print(f"Compliance status: {compliance_results.test_status}")
   ```

2. Check specific measurements:
   ```python
   # PAM4 level analysis
   print("\nPAM4 Levels:")
   print(f"Level means: {compliance_results.pam4_levels.level_means}")
   print(f"Level separations: {compliance_results.pam4_levels.level_separations}")
   print(f"Uniformity: {compliance_results.pam4_levels.uniformity:.3f}")
   
   # EVM results
   print("\nEVM Measurements:")
   print(f"RMS EVM: {compliance_results.evm_results.rms_evm_percent:.2f}%")
   print(f"Peak EVM: {compliance_results.evm_results.peak_evm_percent:.2f}%")
   
   # Eye diagram analysis
   print("\nEye Diagram Analysis:")
   print(f"Eye heights: {compliance_results.eye_results.eye_heights}")
   print(f"Worst eye height: {compliance_results.eye_results.worst_eye_height:.3f}")
   
   # Jitter analysis
   print("\nJitter Components:")
   for jitter_type, value in compliance_results.jitter_results.items():
       print(f"{jitter_type.upper()}: {value:.3f} ps")
   ```

## Result Analysis

### Saving Results

1. Basic result saving:
   ```python
   from datetime import datetime
   import json
   
   # Save to JSON
   results_file = f"validation_results_{datetime.now():%Y%m%d_%H%M%S}.json"
   with open(results_file, 'w') as f:
       json.dump(compliance_results.__dict__, f, indent=4)
   ```

2. Generate report:
   ```python
   # Create summary report
   report_file = f"validation_report_{datetime.now():%Y%m%d_%H%M%S}.txt"
   with open(report_file, 'w') as f:
       f.write("224G Ethernet Validation Report\n")
       f.write("============================\n\n")
       
       f.write(f"Test Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
       f.write(f"Overall Status: {compliance_results.test_status}\n\n")
       
       # Add detailed results
       f.write("Detailed Measurements:\n")
       f.write("-----------------------\n")
       # ... add more details
   ```

### Visualization

1. Plot eye diagram:
   ```python
   import matplotlib.pyplot as plt
   
   def plot_eye_diagram(eye_results):
       plt.figure(figsize=(10, 6))
       plt.bar(range(len(eye_results.eye_heights)), 
               eye_results.eye_heights)
       plt.title('Eye Heights')
       plt.xlabel('Eye Number')
       plt.ylabel('Height')
       plt.grid(True)
       plt.show()
   
   plot_eye_diagram(compliance_results.eye_results)
   ```

2. Plot PAM4 levels:
   ```python
   def plot_pam4_levels(level_results):
       plt.figure(figsize=(10, 6))
       plt.plot(level_results.level_means, 'bo-')
       plt.title('PAM4 Level Distribution')
       plt.xlabel('Level Number')
       plt.ylabel('Voltage')
       plt.grid(True)
       plt.show()
   
   plot_pam4_levels(compliance_results.pam4_levels)
   ```

## Error Handling

1. Handle equipment errors:
   ```python
   try:
       scope.configure_for_224g()
   except Exception as e:
       print(f"Scope configuration failed: {e}")
       # Attempt recovery or cleanup
       scope.cleanup()
   ```

2. Validate measurements:
   ```python
   def validate_measurements(results):
       # Check EVM
       if results.evm_results.rms_evm_percent > 5.0:
           print("Warning: EVM exceeds maximum limit")
       
       # Check eye height
       if results.eye_results.worst_eye_height < 0.2:
           print("Warning: Eye height below minimum")
       
       # Check jitter
       if results.jitter_results['tj'] > 0.3:
           print("Warning: Total jitter exceeds limit")
   
   validate_measurements(compliance_results)
   ```

## Best Practices

1. Always cleanup after testing:
   ```python
   try:
       # Run tests
       results = sequence.run_compliance_test_suite(...)
   finally:
       # Cleanup
       scope.cleanup()
       sequence.cleanup([scope_address, pattern_gen_address])
   ```

2. Use appropriate sample sizes:
   ```python
   # Configure acquisition
   scope.configure_for_224g(ScopeConfig(
       sampling_rate=256e9,
       record_length=1e6  # 1M points
   ))
   ```

3. Monitor environmental conditions:
   ```python
   def check_test_conditions():
       # Add your environmental checks here
       temperature = get_temperature()
       if temperature > 30.0:
           print("Warning: Temperature too high")
   ```

## Next Steps

- Review the [PAM4 Analysis Tutorial](pam4_analysis.md)

## References

- IEEE 802.3 224G Ethernet Specification
- High-Speed Serial Link Design Guide
- Equipment Manuals and Specifications

---