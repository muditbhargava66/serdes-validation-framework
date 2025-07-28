# Stress Testing Overview

The SerDes Validation Framework provides two complementary stress testing systems, each designed for different use cases and scenarios.

## 🔄 Loopback Stress Testing

### Purpose
General-purpose loopback stress testing for any SerDes protocol, simulating TX → RX → back to TX loops with progressive degradation tracking.

### Key Features
- **Multi-Cycle Testing**: Support for 1000+ test cycles with real-time monitoring
- **Progressive Degradation**: Models realistic signal degradation over time
- **Multi-Protocol Support**: USB4, PCIe, and Ethernet protocols
- **Signal Quality Tracking**: Eye height, jitter, SNR, and BER analysis over time
- **CSV Data Export**: Cycle-by-cycle data logging for external analysis
- **Interactive Plots**: Degradation trends and pass/fail status visualization

### Usage
```python
from serdes_validation_framework.stress_testing import LoopbackStressTest, create_stress_test_config

# Create configuration
config = create_stress_test_config(
    protocol="USB4",
    num_cycles=1000,
    cycle_duration=1.0,
    data_rate=20e9
)

# Run stress test
stress_test = LoopbackStressTest(config)
results = stress_test.run_stress_test()

# Results include cycle-by-cycle data
print(f"Total cycles: {results.total_cycles}")
print(f"Pass rate: {results.pass_rate:.1%}")
```

### When to Use
- **Long-term reliability testing** of SerDes links
- **Degradation analysis** over multiple transmission cycles
- **Multi-protocol comparison** studies
- **Performance regression testing** in CI/CD pipelines
- **General stress testing** across different protocols

---

## 🔧 USB4-Specific Stress Testing

### Purpose
USB4/Thunderbolt-specific stress testing with protocol-aware features and comprehensive USB4 stress scenarios.

### Key Features
- **Thermal Stress Testing**: Temperature monitoring and thermal stress scenarios
- **Error Injection & Recovery**: Protocol-specific error injection and recovery testing
- **Power Cycling Stress**: Power state cycling and power management stress
- **Bandwidth Saturation**: Maximum bandwidth stress testing
- **Protocol State Machine Stress**: USB4 protocol state machine stress scenarios
- **Multi-Device Stress**: Multiple device stress testing scenarios
- **Real-time Monitoring**: Live monitoring with alerting capabilities

### Usage
```python
from serdes_validation_framework.protocols.usb4.stress_testing import (
    USB4StressTester, 
    USB4StressTestConfig,
    USB4StressTestType
)

# Create USB4-specific configuration
config = USB4StressTestConfig(
    test_type=USB4StressTestType.THERMAL,
    duration=3600,  # 1 hour
    signal_mode=USB4SignalMode.DUAL_LANE,
    temperature_range=(25, 85)  # °C
)

# Run USB4 stress test
stress_tester = USB4StressTester(config)
result = stress_tester.run_thermal_stress_test(
    temperature_controller=temp_controller,
    performance_monitor=perf_monitor,
    temperature_monitor=temp_monitor
)
```

### When to Use
- **USB4/Thunderbolt certification testing**
- **Thermal characterization** of USB4 devices
- **Protocol-specific stress scenarios**
- **Error recovery validation**
- **Power management testing**
- **Multi-device system validation**

---

## 📊 Comparison Matrix

| Feature | Loopback Stress Testing | USB4-Specific Stress Testing |
|---------|------------------------|------------------------------|
| **Scope** | Multi-protocol, general-purpose | USB4/Thunderbolt-specific |
| **Focus** | TX→RX→TX loopback simulation | Protocol-aware stress scenarios |
| **Protocols** | USB4, PCIe, Ethernet | USB4/Thunderbolt only |
| **Test Types** | Progressive degradation cycles | Thermal, error injection, power cycling |
| **Duration** | Multi-cycle (1000+ cycles) | Long-duration (hours) |
| **Monitoring** | Signal quality over cycles | Real-time with temperature/power |
| **Export** | CSV cycle data | Comprehensive test reports |
| **Use Case** | General reliability testing | Certification and compliance |

---

## 🎯 Choosing the Right Stress Testing

### Use **Loopback Stress Testing** when:
- Testing **multiple protocols** (USB4, PCIe, Ethernet)
- Analyzing **long-term degradation** over many cycles
- Performing **comparative analysis** across protocols
- Running **automated regression tests**
- Need **CSV data export** for external analysis

### Use **USB4-Specific Stress Testing** when:
- Working specifically with **USB4/Thunderbolt devices**
- Performing **certification testing**
- Need **thermal stress testing** with temperature monitoring
- Testing **error recovery mechanisms**
- Validating **power management** features
- Testing **multi-device scenarios**

---

## 🔗 Integration Examples

### Combined Testing Workflow
```python
# 1. Start with general loopback stress testing
from serdes_validation_framework.stress_testing import LoopbackStressTest, create_stress_test_config

loopback_config = create_stress_test_config(protocol="USB4", num_cycles=500)
loopback_test = LoopbackStressTest(loopback_config)
loopback_results = loopback_test.run_stress_test()

# 2. If USB4-specific issues found, use USB4 stress testing
if loopback_results.pass_rate < 0.95:  # Less than 95% pass rate
    from serdes_validation_framework.protocols.usb4.stress_testing import USB4StressTester
    
    usb4_config = USB4StressTestConfig(
        test_type=USB4StressTestType.ERROR_INJECTION,
        duration=1800  # 30 minutes
    )
    
    usb4_tester = USB4StressTester(usb4_config)
    usb4_results = usb4_tester.run_error_injection_test(...)
```

### Framework Integration
```python
from serdes_validation_framework import create_validation_framework

# Main validation
framework = create_validation_framework()
validation_results = framework.validate_signal(signal_data)

# Follow up with appropriate stress testing
if validation_results.protocol == "USB4":
    # Use USB4-specific stress testing for detailed analysis
    usb4_stress_results = run_usb4_stress_testing(signal_data)
else:
    # Use general loopback stress testing
    loopback_stress_results = run_loopback_stress_testing(signal_data)
```

---

## 📚 Related Documentation

- **Loopback Stress Testing**: See `docs/guides/stress_testing.md`
- **USB4 Stress Testing**: See `docs/usb4/certification/stress_testing.md`
- **API Reference**: See `docs/api/stress_testing.md`
- **Examples**: See `examples/loopback_stress_test_example.py`
- **Scripts**: See `scripts/run_loopback_stress_test.py`

---

## 🏗️ Architecture Notes

Both stress testing systems are designed to be:

- **Complementary**: They work together, not in competition
- **Modular**: Each can be used independently
- **Extensible**: New stress test types can be added to either system
- **Well-tested**: Comprehensive test coverage for both systems
- **Production-ready**: Used in real validation workflows

The dual-system approach provides both **breadth** (multi-protocol loopback testing) and **depth** (USB4-specific scenarios) for comprehensive SerDes validation.