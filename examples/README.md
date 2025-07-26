# SerDes Validation Framework Examples

This directory contains comprehensive examples demonstrating the capabilities of the SerDes Validation Framework v1.4.0.

## 📁 Example Files

### Navigation
- **`example_index.py`** - Interactive menu for browsing and running examples

### Core Examples
- **`data_analysis_example.py`** - Comprehensive data analysis demonstration
- **`test_sequence_example.py`** - Advanced test sequence automation
- **`pcie_example.py`** - Complete PCIe 6.0 validation workflow

### Protocol-Specific Examples
- **`eth_224g_example.py`** - 224G Ethernet validation
- **`pam4_analysis_example.py`** - PAM4 signal analysis

### USB4/Thunderbolt 4 Examples
- **`usb4_quick_start.py`** - Quick start guide for USB4 validation
- **`usb4_basic_validation_example.py`** - Comprehensive USB4 protocol validation
- **`usb4_thunderbolt_certification_example.py`** - Thunderbolt 4 certification testing
- **`usb4_tunneling_example.py`** - Multi-protocol tunneling validation
- **`usb4_jitter_analysis_demo.py`** - Advanced jitter analysis with SSC
- **`usb4_power_management_demo.py`** - Power state management validation
- **`usb4_link_recovery_demo.py`** - Link recovery and error handling

### Advanced Examples
- **`multi_protocol_comparison.py`** - Cross-protocol performance comparison
- **`framework_integration_example.py`** - Unified validation framework usage
- **`comprehensive_visualization_example.py`** - Complete visualization system demonstration

## 🚀 Quick Start

### Interactive Examples Menu
```bash
python examples/example_index.py
```
*Use the interactive menu to browse and run examples based on your experience level*

### Direct Example Execution

#### Basic Data Analysis
```bash
python examples/data_analysis_example.py
```

#### PCIe 6.0 Validation
```bash
python examples/pcie_example.py
```

#### USB4 Quick Start
```bash
python examples/usb4_quick_start.py
```

#### Comprehensive Visualization Demo
```bash
python examples/comprehensive_visualization_example.py
```

#### Test Sequence Automation
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
- USB4/Thunderbolt 4 validation
- Protocol compliance testing
- Advanced measurement techniques
- Automated reporting

### 5. USB4/Thunderbolt 4 Examples
- Dual-lane signal analysis with lane skew compensation
- Multi-protocol tunneling validation (PCIe, DisplayPort, USB 3.2)
- Link training and power state management
- Thunderbolt 4 certification and security testing
- Advanced jitter analysis with SSC support
- Comprehensive compliance testing

### 6. Advanced Framework Examples
- Multi-protocol comparison and benchmarking
- Unified validation framework integration
- Cross-protocol performance analysis
- Automatic protocol detection validation
- Custom validation workflow development

### 7. 🎨 Visualization Examples
- **Professional Eye Diagrams**: Advanced eye diagram generation with automatic measurements
- **Interactive Dashboards**: Real-time web-based analysis dashboards with Plotly
- **Protocol-Specific Visualizations**: USB4 tunneling, PCIe link training, Ethernet PAM4 analysis
- **Multi-Protocol Comparison**: Side-by-side protocol analysis and benchmarking
- **Signal Analysis Plots**: Frequency spectrum, time domain, and signal quality visualizations

## 🔧 Requirements

### Basic Requirements
- Python 3.9+
- NumPy
- SerDes Validation Framework v1.4.0+

### Optional Dependencies
- Matplotlib (for plotting and visualization)
- Plotly (for interactive dashboards)
- Seaborn (for enhanced visualization styling)
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

# USB4/Thunderbolt 4 examples
python examples/usb4_quick_start.py
python examples/usb4_basic_validation_example.py --mock
python examples/usb4_thunderbolt_certification_example.py --device-type HOST_CONTROLLER
python examples/usb4_tunneling_example.py --protocols pcie,dp
python examples/multi_protocol_comparison.py --protocols all
python examples/framework_integration_example.py --verbose
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
2. Try `usb4_basic_validation_example.py` for USB4 validation
3. Experiment with different signal modes and parameters
4. Explore multi-lane analysis examples

### Advanced
1. Run `usb4_thunderbolt_certification_example.py` for certification testing
2. Try `multi_protocol_comparison.py` for cross-protocol analysis
3. Use `framework_integration_example.py` for unified framework features
4. Modify examples for custom protocols
5. Integrate with real hardware setups
6. Develop custom analysis algorithms
7. Create automated test suites

## 📈 Performance Examples

### Signal Analysis Performance
```python
# Typical performance metrics from examples:
# - PCIe Signal Analysis: < 1 second for 10K samples
# - USB4 Dual-lane Analysis: < 800ms for 8K samples per lane
# - Mode Switching: < 10 milliseconds
# - Link Training: < 5 seconds for convergence
# - Compliance Testing: < 3 seconds for full suite
# - Multi-protocol Comparison: < 2 seconds for 3 protocols
```

### Throughput Examples
```python
# Example throughput measurements:
# - PCIe NRZ Analysis: ~50,000 samples/second
# - PCIe PAM4 Analysis: ~25,000 samples/second
# - USB4 Dual-lane Analysis: ~20,000 samples/second total
# - 224G Ethernet Analysis: ~30,000 samples/second
# - Cross-protocol Detection: ~100,000 samples/second
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