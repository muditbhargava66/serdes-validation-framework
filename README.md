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
- 🚀 **PCIe 6.0 Support:** Complete PCIe 6.0 specification compliance (64 GT/s)
- 🔄 **NRZ/PAM4 Dual-Mode:** Seamless switching between signaling modes
- 🎯 **Advanced Link Training:** Multi-phase adaptive training with convergence detection
- ⚡ **Enhanced Equalization:** LMS, RLS, CMA, and decision-directed algorithms
- 📏 **Multi-Lane Analysis:** Up to 16-lane support with skew detection
- 👁️ **Advanced Eye Diagrams:** Statistical modeling with jitter decomposition
- 🧪 **Stress Testing:** Environmental condition simulation and validation

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
from serdes_validation_framework.instrument_control.pcie_analyzer import PCIeAnalyzer, PCIeConfig
from serdes_validation_framework.protocols.pcie.link_training import create_pam4_trainer
from serdes_validation_framework.instrument_control.mode_switcher import create_mode_switcher

# Create mode switcher for NRZ/PAM4 dual-mode
switcher = create_mode_switcher(default_mode=SignalMode.PAM4)
result = switcher.switch_mode(SignalMode.PAM4)
print(f"Mode switch: {result.success} in {result.switch_time*1000:.2f}ms")

# Configure PCIe 6.0 analyzer
config = PCIeConfig(
    mode=SignalMode.PAM4,
    sample_rate=200e9,  # 200 GSa/s
    bandwidth=100e9,    # 100 GHz
    voltage_range=1.2,
    link_speed=64e9,    # 64 GT/s
    lane_count=4
)

analyzer = PCIeAnalyzer(config)
results = analyzer.analyze_signal(signal_data)
print(f"SNR: {results['snr_db']:.1f} dB, EVM: {results['rms_evm_percent']:.2f}%")

# Run link training
trainer = create_pam4_trainer(target_ber=1e-12)
training_result = trainer.run_training(signal_data)
print(f"Training converged: {training_result.success}")
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

```bash
# Clone repository
git clone https://github.com/muditbhargava66/serdes-validation-framework.git
cd serdes-validation-framework

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from serdes_validation_framework import get_instrument_controller

# Initialize controller (auto-detects mock/real mode)
controller = get_instrument_controller()

# Connect to instrument
controller.connect_instrument('GPIB::1::INSTR')

# Basic operations
controller.send_command('GPIB::1::INSTR', '*RST')
response = controller.query_instrument('GPIB::1::INSTR', '*IDN?')
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

## 📊 Feature Comparison

| Feature | Mock Mode | Hardware Mode |
|---------|-----------|---------------|
| 🚀 Setup Speed | Instant | Requires calibration |
| 📊 Data Quality | Simulated | Real measurements |
| 🔄 Automation | Full support | Full support |
| 📈 Analysis | All features | All features |
| 🕒 Execution Time | Fast | Hardware-dependent |
| 🔧 Requirements | None | VISA, hardware |

## 📚 Documentation

### Getting Started
- [📖 Installation Guide](docs/INSTALL.md)
- [🎯 Usage Guide](docs/USAGE.md)
- [🔰 Quick Start Tutorial](docs/tutorials/getting_started.md)
- [🤝 Contributing Guide](docs/CONTRIBUTING.md)

### API Reference
- [🔌 Instrument Control](docs/api/instrument_control.md)
- [🧪 Mock Testing](docs/api/mock_controller.md)
- [📡 224G Ethernet](docs/api/eth_224g.md)
- [📊 PAM4 Analysis](docs/api/pam4_analysis.md)

### Guides & Tutorials
- [🔧 Hardware Setup](docs/guides/instrument_setup.md)
- [🏃 Mock Testing](docs/tutorials/mock_testing.md)
- [📈 Signal Analysis](docs/tutorials/pam4_analysis.md)
- [✅ Validation Guide](docs/tutorials/224g_validation.md)

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
│   ├── eth_224g_example.py          # [New] 224G testing example
│   └── pam4_analysis_example.py     # [New] PAM4 analysis example
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
│       │   ├── sequencer.py
│       │   └── eth_224g_sequence.py # [New] 224G test sequences
│       └── protocols/               # [New] Protocol-specific modules
│           ├── __init__.py
│           └── ethernet_224g/
│               ├── __init__.py
│               ├── constants.py      # Protocol constants
│               ├── compliance.py     # Compliance specifications
│               └── training.py       # Link training patterns
├── tests/
│   ├── test_data_collection.py
│   ├── test_data_analysis.py
│   ├── test_instrument_control.py
│   ├── test_test_sequence.py
│   ├── test_pam4_analyzer.py       # [New] PAM4 analyzer tests
│   ├── test_eth_224g_sequence.py   # [New] 224G sequence tests
│   └── test_scope_224g.py         # [New] Scope control tests
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