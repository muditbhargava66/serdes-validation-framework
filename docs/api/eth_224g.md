# 224G Ethernet Protocol API Documentation

## Overview

The 224G Ethernet module provides comprehensive support for validating 224G Ethernet interfaces, including PAM4 signal analysis, link training, and compliance testing.

## Core Components

### Equipment Configuration

#### HighBandwidthScope

Class for controlling high-bandwidth oscilloscopes capable of 224G measurements.

```python
from serdes_validation_framework.instrument_control.scope_224g import HighBandwidthScope

# Initialize scope
scope = HighBandwidthScope(
    resource_name="GPIB0::7::INSTR"
)

# Configure for 224G
scope.configure_for_224g()
```

Key Methods:
- `configure_for_224g(config: Optional[ScopeConfig] = None) -> None`
- `capture_eye_diagram(duration_seconds: float = 1.0, num_ui: int = 1000) -> Dict[str, Union[WaveformData, float]]`
- `measure_jitter() -> Dict[str, float]`

#### ScopeConfig

Configuration dataclass for high-bandwidth scope settings.

```python
from serdes_validation_framework.instrument_control.scope_224g import ScopeConfig

config = ScopeConfig(
    sampling_rate=256e9,  # 256 GSa/s
    bandwidth=120e9,      # 120 GHz
    timebase=5e-12,      # 5 ps/div
    voltage_range=0.8    # 0.8V
)
```

### Test Sequence

#### Ethernet224GTestSequence

Class for running 224G Ethernet test sequences.

```python
from serdes_validation_framework.test_sequence.eth_224g_sequence import Ethernet224GTestSequence

# Initialize sequence
sequence = Ethernet224GTestSequence()

# Run link training
training_results = sequence.run_link_training_test(
    scope_resource="GPIB0::7::INSTR",
    pattern_gen_resource="GPIB0::10::INSTR"
)

# Run compliance tests
compliance_results = sequence.run_compliance_test_suite(
    scope_resource="GPIB0::7::INSTR",
    pattern_gen_resource="GPIB0::10::INSTR"
)
```

Key Methods:
- `run_link_training_test(scope_resource: str, pattern_gen_resource: str, timeout_seconds: float = 10.0) -> TrainingResults`
- `run_compliance_test_suite(scope_resource: str, pattern_gen_resource: str) -> ComplianceResults`
- `measure_pam4_levels(scope_resource: str) -> PAM4Levels`

### Result Types

#### TrainingResults

Dataclass containing link training results.

```python
@dataclass
class TrainingResults:
    training_time: float              # Training completion time
    convergence_status: str           # PASS/FAIL status
    final_eq_settings: List[float]    # Equalizer tap values
    adaptation_error: float           # Final error
```

#### ComplianceResults

Dataclass containing compliance test results.

```python
@dataclass
class ComplianceResults:
    pam4_levels: PAM4Levels          # Level measurements
    evm_results: EVMResults          # EVM calculations
    eye_results: EyeResults          # Eye measurements
    jitter_results: Dict[str, float] # Jitter components
    test_status: str                 # Overall status
```

## Protocol Constants

The `ethernet_224g.constants` module provides protocol-specific constants and specifications:

```python
from serdes_validation_framework.protocols.ethernet_224g.constants import (
    PAM4Specs,
    JitterSpecs,
    EyeSpecs,
    TrainingSpecs,
    ETHERNET_224G_SPECS
)

# Access specifications
symbol_rate = PAM4Specs.SYMBOL_RATE  # 112 GBaud
ui_period = PAM4Specs.UI_PERIOD      # 8.9ps
max_jitter = JitterSpecs.MAX_TJ      # 0.3ps
```

## Compliance Specifications

The `ethernet_224g.compliance` module provides compliance testing functionality:

```python
from serdes_validation_framework.protocols.ethernet_224g.compliance import (
    ComplianceSpecification,
    ComplianceTestConfig
)

# Initialize compliance checker
spec = ComplianceSpecification()

# Check PAM4 levels
passed, measurements = spec.check_pam4_levels(measured_levels)

# Check eye diagram
passed, measurements = spec.check_eye_diagram(eye_height, eye_width)

# Check EVM
passed, measurements = spec.check_evm(rms_evm, peak_evm)
```

## Link Training

The `ethernet_224g.training` module provides link training capabilities:

```python
from serdes_validation_framework.protocols.ethernet_224g.training import (
    LinkTraining,
    TrainingConfig,
    TrainingStatus
)

# Initialize trainer
trainer = LinkTraining(num_taps=5)

# Generate training sequence
sequence = trainer.generate_training_sequence('adapt', length=1000)

# Train equalizer
status, equalized = trainer.train_equalizer(received_signal, 'adapt')
```

## Error Handling

The framework uses custom exceptions for error handling:

```python
from serdes_validation_framework.protocols.ethernet_224g import ValidationError

try:
    # Run validation
    results = sequence.run_compliance_test_suite(...)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## Type Safety

All numeric parameters are strictly type-checked:

```python
# Correct usage
eye_height = 0.3  # float
results = spec.check_eye_diagram(eye_height, eye_width)

# Will raise AssertionError
eye_height = 3    # int
results = spec.check_eye_diagram(eye_height, eye_width)  # Error!
```

## Data Classes

The framework uses Python's dataclasses for structured data:

```python
@dataclass
class PAM4Levels:
    level_means: np.ndarray          # Level voltages
    level_separations: np.ndarray    # Gaps between levels
    uniformity: float                # Level uniformity metric
```

## Best Practices

1. Always validate input types:
   ```python
   assert isinstance(value, float), "Value must be a float"
   ```

2. Use type hints:
   ```python
   def measure_jitter(self) -> Dict[str, float]:
   ```

3. Handle cleanup properly:
   ```python
   try:
       scope.configure_for_224g()
       # ... perform measurements
   finally:
       scope.cleanup()
   ```

4. Use logging for debugging:
   ```python
   logger.debug(f"EVM measurement: {evm_result:.2f}%")
   ```

## See Also

- [PAM4 Analysis Documentation](pam4_analysis.md)
- [224G Validation Tutorial](../tutorials/224g_validation.md)
- [Getting Started Guide](../tutorials/getting_started.md)

---