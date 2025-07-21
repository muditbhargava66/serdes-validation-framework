# SerDes Validation Framework Examples

This directory contains comprehensive examples demonstrating the capabilities of the SerDes Validation Framework v1.3.0.

## 📁 Example Files

### Core Examples
- **`data_analysis_example.py`** - Comprehensive data analysis demonstration
- **`test_sequence_example.py`** - Advanced test sequence automation
- **`pcie_example.py`** - Complete PCIe 6.0 validation workflow

### Protocol-Specific Examples
- **`eth_224g_example.py`** - 224G Ethernet validation
- **`pam4_analysis_example.py`** - PAM4 signal analysis

## 🚀 Quick Start

### Basic Data Analysis
```bash
python examples/data_analysis_example.py
```

### PCIe 6.0 Validation
```bash
python examples/pcie_example.py
```

### Test Sequence Automation
```bash
python examples/test_sequence_example.py
```

## 📊 Example Categories

### 1. Signal Analysis Examples
- Basic statistical analysis
- Advanced signal processing
- PAM4 signal analysis
- Eye diagram generation
- SNR and EVM calculations

### 2. PCIe 6.0 Examples
- NRZ/PAM4 dual-mode operation
- Link training and equalization
- Compliance testing
- Multi-lane analysis
- Performance benchmarking

### 3. Test Automation Examples
- Instrument control workflows
- Mock testing scenarios
- Data collection and analysis
- Error handling and recovery
- Performance monitoring

### 4. Protocol Validation Examples
- 224G Ethernet validation
- Protocol compliance testing
- Advanced measurement techniques
- Automated reporting

## 🔧 Requirements

### Basic Requirements
- Python 3.9+
- NumPy
- SerDes Validation Framework v1.3.0+

### Optional Dependencies
- Matplotlib (for plotting)
- SciPy (for advanced signal processing)
- Pandas (for data handling)
- scikit-learn (for machine learning features)

### Hardware Requirements
- None (examples work in mock mode)
- Optional: GPIB/USB instruments for real hardware testing

## 📖 Usage Patterns

### Running Examples
```bash
# Basic execution
python examples/example_name.py

# With verbose output
python examples/example_name.py --verbose

# With custom parameters
python examples/pcie_example.py --mode pam4 --lanes 4
```

### Mock vs Real Hardware
```bash
# Force mock mode (no hardware required)
export SVF_MOCK_MODE=1
python examples/test_sequence_example.py

# Use real hardware (if available)
export SVF_MOCK_MODE=0
python examples/test_sequence_example.py

# Auto-detect (default)
unset SVF_MOCK_MODE
python examples/test_sequence_example.py
```

## 🎯 Learning Path

### Beginner
1. Start with `data_analysis_example.py` for basic concepts
2. Try `test_sequence_example.py` for automation basics
3. Explore mock testing capabilities

### Intermediate
1. Run `pcie_example.py` for advanced PCIe validation
2. Experiment with different signal modes and parameters
3. Try multi-lane analysis examples

### Advanced
1. Modify examples for custom protocols
2. Integrate with real hardware setups
3. Develop custom analysis algorithms
4. Create automated test suites

## 📈 Performance Examples

### Signal Analysis Performance
```python
# Typical performance metrics from examples:
# - Signal Analysis: < 1 second for 10K samples
# - Mode Switching: < 10 milliseconds
# - Link Training: < 5 seconds for convergence
# - Compliance Testing: < 3 seconds for full suite
```

### Throughput Examples
```python
# Example throughput measurements:
# - NRZ Analysis: ~50,000 samples/second
# - PAM4 Analysis: ~25,000 samples/second
# - Multi-lane Analysis: ~10,000 samples/second per lane
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure src is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Missing Dependencies**
   ```bash
   # Install optional dependencies
   pip install matplotlib scipy pandas scikit-learn
   ```

3. **Hardware Connection Issues**
   ```bash
   # Use mock mode for development
   export SVF_MOCK_MODE=1
   ```

### Debug Mode
```bash
# Enable debug logging
python examples/example_name.py --verbose
```

## 📚 Additional Resources

### Documentation
- [API Reference](../docs/api/index.md)
- [PCIe Tutorial](../docs/tutorials/pcie_validation.md)
- [User Guide](../docs/USAGE.md)

### Scripts
- [PCIe Validation Script](../scripts/pcie_validation.py)
- [Data Analysis Script](../scripts/data_analysis.py)
- [Instrument Control Script](../scripts/instrument_control.py)

## 🤝 Contributing

To contribute new examples:

1. Follow the existing code style and structure
2. Include comprehensive docstrings and comments
3. Add error handling and logging
4. Test with both mock and real hardware (if applicable)
5. Update this README with your example

## 📄 License

All examples are provided under the same MIT license as the main framework.

---

**Happy Validating!** 🚀

For questions or issues, please refer to the main project documentation or open an issue on GitHub.