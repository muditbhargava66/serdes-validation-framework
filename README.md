<div align="center">

# 🚀 SerDes Validation Framework

[![CI](https://github.com/muditbhargava66/serdes-validation-framework/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/muditbhargava66/serdes-validation-framework/actions/workflows/ci.yml)
[![Lint](https://github.com/muditbhargava66/serdes-validation-framework/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/muditbhargava66/serdes-validation-framework/actions/workflows/lint.yml)
[![CodeQL](https://github.com/muditbhargava66/serdes-validation-framework/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/muditbhargava66/serdes-validation-framework/actions/workflows/github-code-scanning/codeql)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/)
[![Code Coverage](https://img.shields.io/badge/coverage-98%25-green)](https://github.com/muditbhargava66/serdes-validation-framework/actions)
[![Documentation](https://img.shields.io/badge/docs-readthedocs.io-blue)](https://serdes-validation-framework.readthedocs.io/)
[![Last Commit](https://img.shields.io/github/last-commit/muditbhargava66/serdes-validation-framework)](https://github.com/muditbhargava66/serdes-validation-framework/commits/main)
[![Contributors](https://img.shields.io/github/contributors/muditbhargava66/serdes-validation-framework)](https://github.com/muditbhargava66/serdes-validation-framework/graphs/contributors)

**A comprehensive framework for validating high-speed SerDes protocols with PCIe 6.0 support, NRZ/PAM4 dual-mode capabilities, automated data collection, advanced signal analysis, and multi-vendor instrument control.**

![Banner](docs/images/serdes-framework-banner.png)

</div>

## ✨ Key Features

### Core Capabilities
- 🔄 **Automated Data Collection:** Seamless data gathering from lab instruments
- 📊 **Advanced Analysis:** Comprehensive signal processing and visualization
- 🎛️ **Universal Instrument Control:** GPIB/USB interface for multi-vendor support
- 📋 **Flexible Test Sequences:** Customizable, reusable test automation

### 🆕 New Features in v1.3.0
- 🚀 **PCIe 6.0 Complete Support:** Full 64 GT/s specification compliance with multi-lane validation
- 🔄 **NRZ/PAM4 Dual-Mode:** Seamless mode switching with <10ms transition time
- 🎯 **Advanced Link Training:** Multi-phase adaptive training (Phase 0-3) with convergence detection
- ⚡ **Enhanced Equalization:** LMS, RLS, CMA algorithms with multi-tap optimization
- 📏 **Multi-Lane Analysis:** Up to 16-lane support with lane skew analysis and compensation
- 👁️ **Advanced Eye Diagrams:** Statistical modeling with RJ/DJ/PJ jitter decomposition
- 🧪 **Comprehensive Testing:** Stress testing, compliance validation, and automated workflows
- 🔧 **Intelligent Analysis:** Robust signal detection with K-means clustering and fallback algorithms
- 📊 **Performance Optimization:** 40% faster analysis with memory-efficient operations
- 🛡️ **Type Safety:** 100% type hint coverage with runtime validation

### Previous Features
- 🔍 **Mock Testing Support:** Development and testing without physical hardware
- 📡 **224G Ethernet Support:** Complete validation suite for 224G interfaces
- 📊 **PAM4 Analysis:** Advanced PAM4 signal processing capabilities
- 🔬 **Enhanced Scope Control:** High-bandwidth oscilloscope integration

### 🎮 Intelligent Hardware/Mock Adaptation

```python
from serdes_validation_framework import get_instrument_controller

# Auto-detects available hardware
controller = get_instrument_controller()

# For development/testing without hardware:
os.environ['SVF_MOCK_MODE'] = '1'

# For specific hardware testing:
os.environ['SVF_MOCK_MODE'] = '0'

# Example usage (works in both modes):
controller.connect_instrument('GPIB::1::INSTR')
response = controller.query_instrument('GPIB::1::INSTR', '*IDN?')
print(f"Operating in {controller.get_mode()} mode")  # 'mock' or 'real'
```

### 📡 224G Ethernet Validation

```python
from serdes_validation_framework.protocols.ethernet_224g import (
    Ethernet224GTestSequence,
    ComplianceSpecification
)

# Initialize test sequence
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

print(f"Training status: {training_results.convergence_status}")
print(f"Compliance status: {compliance_results.test_status}")
```

### 🚀 PCIe 6.0 Validation

```python
from serdes_validation_framework.protocols.pcie.constants import SignalMode
from serdes_validation_framework.test_sequence.pcie_sequence import (
    PCIeTestSequence, 
    PCIeTestPhase,
    PCIeTestResult,
    PCIeTestSequenceConfig,
    LaneConfig,
    create_single_lane_nrz_test,
    create_multi_lane_pam4_test
)

# Create a PCIe 6.0 test configuration for multi-lane PAM4
config = create_multi_lane_pam4_test(
    num_lanes=4,
    sample_rate=200e9,  # 200 GSa/s
    bandwidth=100e9,    # 100 GHz
    voltage_range=1.2,
    target_ber=1e-12
)

# Create a PCIe test sequence
sequence = PCIeTestSequence(config)

# Run complete test sequence with all phases
result = sequence.run_complete_sequence(signal_data)

# Check results
if result.overall_status == PCIeTestResult.PASS:
    print("✅ PCIe 6.0 validation passed!")
    print(f"Total duration: {result.total_duration:.2f} seconds")
    
    # Check individual phase results
    for phase_result in result.phase_results:
        phase_name = phase_result.phase.name
        status = phase_result.status.name
        print(f"Phase {phase_name}: {status} ({phase_result.duration:.2f}s)")
        
        # Print key metrics
        for metric, value in phase_result.metrics.items():
            print(f"  - {metric}: {value}")
else:
    print(f"❌ PCIe 6.0 validation failed: {result.overall_status.name}")

# Access lane-specific results
for lane_id, lane_results in result.lane_results.items():
    print(f"Lane {lane_id} SNR: {lane_results.get('snr_db', 'N/A')} dB")
```

### 📊 PAM4 Signal Analysis

```python
from serdes_validation_framework.data_analysis import PAM4Analyzer

# Initialize analyzer
analyzer = PAM4Analyzer({
    'time': time_data,
    'voltage': voltage_data
})

# Analyze signal
levels = analyzer.analyze_level_separation()
evm = analyzer.calculate_evm()
eye = analyzer.analyze_eye_diagram()

print(f"RMS EVM: {evm.rms_evm_percent:.2f}%")
print(f"Worst Eye Height: {eye.worst_eye_height:.3f}")
```

### 🔄 Automatic Mode Selection

The framework intelligently adapts between hardware and mock modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| 🔍 **Auto** | Automatically detects available hardware | Default behavior |
| 🎮 **Mock** | Simulates hardware responses | Development & testing |
| 🔧 **Real** | Uses physical instruments | Production validation |

Configure via environment:
```bash
# Development without hardware
export SVF_MOCK_MODE=1

# Force hardware mode
export SVF_MOCK_MODE=0

# Auto-detect (default)
unset SVF_MOCK_MODE
```

### 📈 Real vs Mock Data Comparison

```python
# Mock mode provides realistic data simulation:
def generate_pam4_waveform():
    """Generate synthetic PAM4 data"""
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    symbols = np.random.choice(levels, size=num_points)
    noise = np.random.normal(0, 0.05, num_points)
    return symbols + noise

# Auto-switching between real/mock:
def capture_waveform(scope):
    """Capture waveform data"""
    if scope.controller.get_mode() == 'mock':
        return generate_pam4_waveform()
    else:
        return scope.query_waveform()
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (recommended 3.10)
- Git
- VISA Library (optional, for hardware control)

### Installation

#### Option 1: Install from PyPI (Recommended)
```bash
# Install latest stable version
pip install serdes-validation-framework

# Install specific version
pip install serdes-validation-framework==1.3.0
```

#### Option 2: Install from Source
```bash
# Clone repository
git clone https://github.com/muditbhargava66/serdes-validation-framework.git
cd serdes-validation-framework

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

#### Verify Installation
```bash
python -c "from serdes_validation_framework import __version__; print(f'SerDes Framework v{__version__}')"
```

### Basic Usage

```python
from serdes_validation_framework.test_sequence import PCIeTestSequencer
from serdes_validation_framework.protocols.pcie import SignalMode

# Initialize PCIe test sequencer (auto-detects mock/real mode)
sequencer = PCIeTestSequencer()

# Connect to instruments
sequencer.setup_instruments([
    'GPIB::1::INSTR',  # Oscilloscope
    'GPIB::2::INSTR'   # Pattern generator
])

# Run a simple test sequence
results = sequencer.run_sequence([
    {'command': 'configure_scope', 'params': {'bandwidth': 100e9}},
    {'command': 'capture_data', 'params': {'duration': 1.0}},
    {'command': 'analyze_signal', 'params': {'mode': SignalMode.PAM4}}
])

print(f"Test completed: {results['status']}")
```

## 🛠️ Development

### Setup Development Environment
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
python -m unittest discover -s tests

# Run linter
ruff check src tests

# Test all Python versions
tox
```

### Mock Testing
Enable mock mode for development without hardware:
```bash
# Enable mock mode
export SVF_MOCK_MODE=1

# Run examples
python examples/mock_testing_example.py
```

## 📊 Performance Benchmarks

| Operation | Performance | Improvement |
|-----------|-------------|-------------|
| � Signpal Analysis | <1s for 10K samples | 40% faster |
| � Maode Switching | <10ms NRZ↔PAM4 | Real-time |
| 🎯 Link Training | <5s convergence | Optimized |
| ✅ Compliance Testing | <3s full suite | Comprehensive |
| � ️ Eye Diagram Analysis | <2s complete | Enhanced |
| � Multii-lane Processing | Linear scaling | Up to 16 lanes |

## 🔧 Feature Comparison

| Feature | Mock Mode | Hardware Mode | PCIe 6.0 Support |
|---------|-----------|---------------|-------------------|
| 🚀 Setup Speed | Instant | Requires calibration | ✅ Full |
| 📊 Data Quality | Simulated | Real measurements | ✅ 64 GT/s |
| 🔄 Mode Support | NRZ/PAM4 | NRZ/PAM4 | ✅ Dual-mode |
| 📈 Analysis | All features | All features | ✅ Advanced |
| 🕒 Execution Time | Fast | Hardware-dependent | ✅ Optimized |
| 🔧 Requirements | None | VISA, hardware | ✅ Compatible |

## 📚 Documentation

### Getting Started
- [📖 Installation Guide](docs/INSTALL.md)
- [🎯 Usage Guide](docs/USAGE.md)
- [🔰 Quick Start Tutorial](docs/tutorials/getting_started.md)
- [🤝 Contributing Guide](docs/CONTRIBUTING.md)

### API Reference
- [🔌 Instrument Control](docs/api/instrument_control.md)
- [🧪 Mock Testing](docs/api/mock_controller.md)
- [🚀 PCIe 6.0 Validation](docs/api/pcie_validation.md)
- [📡 224G Ethernet](docs/api/eth_224g.md)
- [📊 PAM4 Analysis](docs/api/pam4_analysis.md)
- [🎯 Link Training](docs/api/link_training.md)
- [⚡ Equalization](docs/api/equalization.md)

### Guides & Tutorials
- [🔧 Hardware Setup](docs/guides/instrument_setup.md)
- [🏃 Mock Testing](docs/tutorials/mock_testing.md)
- [🚀 PCIe 6.0 Quick Start](docs/tutorials/pcie_quickstart.md)
- [🔄 NRZ/PAM4 Mode Switching](docs/tutorials/dual_mode.md)
- [🎯 Link Training Guide](docs/tutorials/link_training.md)
- [📈 Signal Analysis](docs/tutorials/pam4_analysis.md)
- [✅ Compliance Testing](docs/tutorials/compliance_testing.md)
- [📊 Multi-lane Validation](docs/tutorials/multi_lane.md)

### Development Setup

1. **Install development dependencies:**

    ```bash
    pip install -r requirements-dev.txt
    ```

2. **Run tests:**

    ```bash
    python -m unittest discover -s tests
    ```

3. **Run linter:**

    ```bash
    ruff check src tests
    ```

4. **Run Tox for testing across multiple environments:**

    ```bash
    tox
    ```

## Project Structure

```plaintext
serdes-validation-framework/
├── .github/                          # [Existing] GitHub specific files
├── docs/
│   ├── api/
│   │   ├── index.md
│   │   ├── usage.md
│   │   ├── eth_224g.md              # [New] 224G Ethernet API documentation
│   │   └── pam4_analysis.md         # [New] PAM4 analysis documentation
│   ├── images/
│   ├── tutorials/
│   │   ├── getting_started.md
│   │   ├── 224g_validation.md       # [New] 224G validation tutorial
│   │   └── pam4_analysis.md         # [New] PAM4 analysis tutorial
│   ├── CONTRIBUTING.md
│   ├── INSTALL.md
│   └── USAGE.md
├── examples/
│   ├── test_sequence_example.py
│   ├── data_analysis_example.py
│   ├── pcie_6_validation_example.py # [New] PCIe 6.0 validation example
│   ├── dual_mode_example.py         # [New] NRZ/PAM4 mode switching
│   ├── link_training_example.py     # [New] Link training example
│   ├── eth_224g_example.py          # [Existing] 224G testing example
│   └── pam4_analysis_example.py     # [Existing] PAM4 analysis example
├── scripts/
│   ├── data_collection.py
│   ├── data_analysis.py
│   ├── instrument_control.py
│   ├── test_sequence.py
│   └── eth_224g_validation.py       # [New] 224G validation script
├── src/
│   └── serdes_validation_framework/
│       ├── __init__.py
│       ├── data_collection/         # [Existing] Base data collection
│       │   ├── __init__.py
│       │   └── data_collector.py
│       ├── data_analysis/          
│       │   ├── __init__.py
│       │   ├── analyzer.py
│       │   └── pam4_analyzer.py     # [New] PAM4 signal analysis
│       ├── instrument_control/
│       │   ├── __init__.py
│       │   ├── controller.py
│       │   ├── mock_controller.py
│       │   └── scope_224g.py        # [New] High-bandwidth scope control
│       ├── test_sequence/
│       │   ├── __init__.py
│       │   ├── sequencer.py         # [Updated] PCIeTestSequencer
│       │   ├── pcie_sequence.py     # [New] PCIe 6.0 test sequences
│       │   ├── dual_mode_sequence.py # [New] Dual-mode test sequences
│       │   └── eth_224g_sequence.py # [Existing] 224G test sequences
│       └── protocols/               # [Expanded] Protocol-specific modules
│           ├── __init__.py
│           ├── pcie/                # [New] PCIe 6.0 protocol support
│           │   ├── __init__.py
│           │   ├── constants.py     # PCIe constants and enums
│           │   ├── compliance.py    # PCIe compliance testing
│           │   ├── link_training.py # Advanced link training
│           │   ├── equalization.py  # Equalization algorithms
│           │   └── dual_mode/       # NRZ/PAM4 dual-mode support
│           │       ├── __init__.py
│           │       ├── mode_control.py
│           │       ├── nrz_training.py
│           │       └── pam4_training.py
│           └── ethernet_224g/       # [Existing] 224G Ethernet
│               ├── __init__.py
│               ├── constants.py      # Protocol constants
│               ├── compliance.py     # Compliance specifications
│               └── training.py       # Link training patterns
├── tests/
│   ├── test_data_collection.py
│   ├── test_data_analysis.py
│   ├── test_instrument_control.py
│   ├── test_test_sequence.py       # [Updated] PCIeTestSequencer tests
│   ├── test_pcie_sequence.py       # [New] PCIe 6.0 sequence tests
│   ├── test_pcie_analyzer.py       # [New] PCIe analyzer tests
│   ├── test_dual_mode.py           # [New] Dual-mode tests
│   ├── test_pcie_integration.py    # [New] PCIe integration tests
│   ├── test_nrz_analyzer.py        # [New] NRZ analyzer tests
│   ├── test_pam4_analyzer.py       # [Existing] PAM4 analyzer tests
│   ├── test_eth_224g_sequence.py   # [Existing] 224G sequence tests
│   └── test_scope_224g.py         # [Existing] Scope control tests
├── .gitignore
├── LICENSE
├── README.md                       # [Update] Add 224G features
├── requirements.txt                # [Update] Add new dependencies
├── setup.py                       # [Update] Add new modules
├── CHANGELOG.md
└── tox.ini
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](docs/CONTRIBUTING.md) for details on:
- Code Style
- Development Process
- Submission Guidelines
- Testing Requirements

## Community and Support

For any questions, issues, or contributions, please open an issue on the [GitHub repository](https://github.com/muditbhargava66/serdes-validation-framework/issues). Contributions and feedback are always welcome.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

<div align="center">

## Star History

<a href="https://star-history.com/#muditbhargava66/serdes-validation-framework&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=muditbhargava66/serdes-validation-framework&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=muditbhargava66/serdes-validation-framework&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=muditbhargava66/serdes-validation-framework&type=Date" />
 </picture>
</a>

---  
  
**Enjoy using the SerDes Validation Framework?**  
⭐️ Star the repo and consider contributing!  
  
📫 **Contact**: [@muditbhargava66](https://github.com/muditbhargava66)
🐛 **Report Issues**: [Issue Tracker](https://github.com/muditbhargava66/serdes-validation-framework/issues)
  
© 2025 Mudit Bhargava. [MIT License](LICENSE)  
<!-- Copyright symbol using HTML entity for better compatibility -->
</div>