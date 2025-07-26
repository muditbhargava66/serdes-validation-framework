# Installation Guide

This guide provides comprehensive instructions for installing the SerDes Validation Framework.

## System Requirements

### Operating Systems
- **Linux**: Ubuntu 20.04+, CentOS 8+, RHEL 8+
- **Windows**: Windows 10/11 (64-bit)
- **macOS**: macOS 10.15+ (Catalina or later)

### Python Requirements
- **Python Version**: 3.9 or higher
- **Recommended**: Python 3.10 or 3.11
- **Package Manager**: pip 21.0+ or conda

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8 GB
- **Storage**: 2 GB free space
- **Network**: Internet connection for installation

#### Recommended Requirements
- **CPU**: 8+ cores, 3.0+ GHz
- **RAM**: 16+ GB
- **Storage**: 10+ GB free space (for test data and reports)
- **GPU**: Optional, for accelerated signal processing

#### Hardware Instruments (Optional)
- **Oscilloscopes**: Keysight, Tektronix, Rohde & Schwarz
- **Pattern Generators**: Keysight, Tektronix
- **Power Meters**: Keysight, Rohde & Schwarz
- **USB4/Thunderbolt Devices**: For real hardware testing

## Installation Methods

### Method 1: PyPI Installation (Recommended)

#### Basic Installation
```bash
# Install the framework
pip install serdes-validation-framework

# Verify installation
python -c "import serdes_validation_framework; print('Installation successful!')"
```

#### Installation with Optional Dependencies
```bash
# Install with visualization support
pip install serdes-validation-framework[visualization]

# Install with instrument control support
pip install serdes-validation-framework[instruments]

# Install with all optional dependencies
pip install serdes-validation-framework[all]
```

#### Development Installation
```bash
# Install with development dependencies
pip install serdes-validation-framework[dev]

# Or install all dependencies including development tools
pip install serdes-validation-framework[all,dev]
```

### Method 2: Conda Installation

```bash
# Add conda-forge channel
conda config --add channels conda-forge

# Install the framework
conda install serdes-validation-framework

# Or create a new environment
conda create -n svf python=3.10 serdes-validation-framework
conda activate svf
```

### Method 3: Source Installation

#### Clone and Install
```bash
# Clone the repository
git clone https://github.com/muditbhargava66/serdes-validation-framework.git
cd serdes-validation-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install with all dependencies
pip install -e .[all,dev]
```

#### Build from Source
```bash
# Install build dependencies
pip install build wheel

# Build the package
python -m build

# Install the built package
pip install dist/serdes_validation_framework-*.whl
```

## Dependency Installation

### Core Dependencies

The framework automatically installs these core dependencies:

```bash
# Core scientific computing
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0

# Configuration and utilities
pyyaml>=6.0
click>=8.0
tqdm>=4.62.0

# Logging and monitoring
structlog>=21.1.0
```

### Optional Dependencies

#### Visualization Support
```bash
# Install matplotlib and related packages
pip install matplotlib>=3.5.0
pip install plotly>=5.0.0
pip install seaborn>=0.11.0

# Or install with the framework
pip install serdes-validation-framework[visualization]
```

#### Instrument Control Support
```bash
# Install VISA and instrument drivers
pip install pyvisa>=1.12.0
pip install pyvisa-py>=0.5.0

# Serial communication
pip install pyserial>=3.5

# USB communication
pip install pyusb>=1.2.0

# Or install with the framework
pip install serdes-validation-framework[instruments]
```

#### Documentation Support
```bash
# Install documentation dependencies
pip install sphinx>=4.0.0
pip install sphinx-rtd-theme>=1.0.0
pip install myst-parser>=0.17.0

# Or install with the framework
pip install serdes-validation-framework[docs]
```

## Platform-Specific Instructions

### Linux Installation

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install system dependencies
sudo apt install python3-dev python3-pip python3-venv
sudo apt install build-essential libffi-dev libssl-dev

# Install USB development libraries (for instrument control)
sudo apt install libusb-1.0-0-dev libudev-dev

# Install the framework
pip3 install serdes-validation-framework[all]
```

#### CentOS/RHEL/Fedora
```bash
# Install system dependencies
sudo dnf install python3-devel python3-pip
sudo dnf install gcc gcc-c++ make libffi-devel openssl-devel

# Install USB libraries
sudo dnf install libusb1-devel systemd-devel

# Install the framework
pip3 install serdes-validation-framework[all]
```

### Windows Installation

#### Using pip (Recommended)
```batch
# Open Command Prompt or PowerShell as Administrator

# Upgrade pip
python -m pip install --upgrade pip

# Install Visual C++ Build Tools (if needed)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install the framework
pip install serdes-validation-framework[all]
```

#### Using Anaconda
```batch
# Open Anaconda Prompt

# Create new environment
conda create -n svf python=3.10
conda activate svf

# Install the framework
conda install -c conda-forge serdes-validation-framework
```

### macOS Installation

#### Using pip
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python@3.10
brew install libusb

# Install the framework
pip3 install serdes-validation-framework[all]
```

#### Using conda
```bash
# Install Miniconda (if not already installed)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# Create environment and install
conda create -n svf python=3.10
conda activate svf
conda install -c conda-forge serdes-validation-framework
```

## Virtual Environment Setup

### Using venv (Recommended)
```bash
# Create virtual environment
python -m venv svf-env

# Activate environment
# Linux/macOS:
source svf-env/bin/activate
# Windows:
svf-env\Scripts\activate

# Install framework
pip install serdes-validation-framework[all]

# Deactivate when done
deactivate
```

### Using conda
```bash
# Create conda environment
conda create -n svf python=3.10 numpy scipy matplotlib

# Activate environment
conda activate svf

# Install framework
pip install serdes-validation-framework

# Deactivate when done
conda deactivate
```

## Configuration Setup

### Initial Configuration
```bash
# Create configuration directory
mkdir -p ~/.svf

# Generate default configuration
python -c "
from serdes_validation_framework.config import ConfigManager
config = ConfigManager()
config.create_default_config()
print('Default configuration created at ~/.svf/config.yaml')
"
```

### Environment Variables
```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export SVF_CONFIG_PATH=~/.svf/config.yaml
export SVF_DATA_PATH=~/.svf/data
export SVF_LOG_LEVEL=INFO

# For development/testing
export SVF_MOCK_MODE=1
```

## Verification

### Basic Verification
```python
# Test basic import
import serdes_validation_framework
print(f"Framework version: {serdes_validation_framework.__version__}")

# Test USB4 module
from serdes_validation_framework.protocols.usb4 import USB4Validator
validator = USB4Validator()
print("USB4 module loaded successfully")

# Test mock mode
import os
os.environ['SVF_MOCK_MODE'] = '1'
mock_data = validator.generate_mock_signal_data()
print(f"Mock mode working: generated {len(mock_data.lane0_data)} samples")
```

### Comprehensive Test
```python
# Run comprehensive test
from serdes_validation_framework.tests import run_installation_test

result = run_installation_test()
if result.success:
    print("✓ Installation test passed")
    print(f"  - {result.tests_passed}/{result.total_tests} tests passed")
else:
    print("✗ Installation test failed")
    for error in result.errors:
        print(f"  - {error}")
```

### Command Line Verification
```bash
# Check if command line tools are available
svf --version
svf-validate --help
svf-report --help

# Run quick test
svf-test --quick
```

## Troubleshooting

### Common Issues

#### Import Errors
```python
# Check Python path
import sys
print(sys.path)

# Check installed packages
import pkg_resources
installed_packages = [d.project_name for d in pkg_resources.working_set]
print("serdes-validation-framework" in installed_packages)
```

#### Permission Issues (Linux/macOS)
```bash
# Fix permission issues
sudo chown -R $USER:$USER ~/.svf
chmod -R 755 ~/.svf

# Use user installation
pip install --user serdes-validation-framework
```

#### Windows-Specific Issues
```batch
# Install Visual C++ redistributables
# Download from Microsoft website

# Use long path support
git config --system core.longpaths true

# Run as administrator if needed
```

#### Dependency Conflicts
```bash
# Create clean environment
python -m venv clean-env
source clean-env/bin/activate
pip install --upgrade pip
pip install serdes-validation-framework[all]
```

### Getting Help

#### Check Installation Status
```python
from serdes_validation_framework.diagnostics import check_installation

report = check_installation()
print(report.summary)

if not report.all_good:
    for issue in report.issues:
        print(f"Issue: {issue.description}")
        print(f"Solution: {issue.suggested_fix}")
```

#### Debug Information
```python
from serdes_validation_framework.diagnostics import get_debug_info

debug_info = get_debug_info()
print(debug_info.format_report())
```

## Next Steps

After successful installation:

1. **Quick Start**: Follow the [Getting Started Tutorial](../tutorials/getting_started.md)
2. **USB4 Validation**: Try the [USB4 Quick Start Guide](../usb4/quickstart.md)
3. **Configuration**: Review the [Configuration Reference](../reference/configuration.md)
4. **Examples**: Explore the examples directory

## Support

If you encounter issues during installation:

1. **Check Documentation**: Review this guide and the [troubleshooting guide](troubleshooting.md)
2. **Search Issues**: Check [GitHub Issues](https://github.com/muditbhargava66/serdes-validation-framework/issues)
3. **Ask for Help**: Open a new issue with your installation details
4. **Community Support**: Join our discussions on GitHub

## Uninstallation

### Remove Framework
```bash
# Uninstall the framework
pip uninstall serdes-validation-framework

# Remove configuration (optional)
rm -rf ~/.svf

# Remove virtual environment (if used)
rm -rf svf-env
```

### Clean Conda Environment
```bash
# Remove conda environment
conda env remove -n svf

# Clean conda cache
conda clean --all
```

Happy validating! 🚀