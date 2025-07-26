# Installation

For complete installation instructions, please see the [Installation Guide](guides/installation.md).

## Quick Install

```bash
# Install from PyPI
pip install serdes-validation-framework[all]

# Verify installation
python -c "import serdes_validation_framework; print('Installation successful!')"

# Run core tests to verify functionality (91 tests, 0.83s)
python -m pytest tests/ -v --tb=short --ignore=tests/integration --ignore=tests/performance --ignore=tests/legacy
```

## Requirements

- Python 3.9+
- 8GB RAM (16GB recommended)
- 2GB free disk space

## Next Steps

1. Follow the [Getting Started Tutorial](tutorials/getting_started.md)
2. Try the [USB4 Quick Start Guide](usb4/quickstart.md)
3. Explore the [Testing Guide](guides/testing.md) for comprehensive testing
4. Review the [Configuration Reference](reference/configuration.md)

For detailed installation instructions, troubleshooting, and platform-specific guidance, see the complete [Installation Guide](guides/installation.md).