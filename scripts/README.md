# SerDes Validation Framework Scripts v1.4.1

This directory contains production-ready scripts for comprehensive SerDes validation workflows, including new API server management and stress testing capabilities.

## 📁 Script Files

### 🆕 New in v1.4.1
- **`run_loopback_stress_test.py`** - Loopback stress testing with progressive degradation
- **API Server Scripts** - Available via `run_api_server.py` in root directory
- **API Client Scripts** - Available via `test_api_client.py` in root directory

### Core Scripts
- **`pcie_validation.py`** - Complete PCIe 6.0 validation suite
- **`data_analysis.py`** - Comprehensive data analysis tool
- **`instrument_control.py`** - Instrument control and testing
- **`data_collection.py`** - Automated data collection
- **`test_sequence.py`** - Test sequence automation
- **`multi_protocol_validation.py`** - Unified multi-protocol validation framework

### Protocol-Specific Scripts
- **`eth_224g_validation.py`** - 224G Ethernet validation
- **`usb4_validation.py`** - USB4/Thunderbolt 4 comprehensive validation

## 🚀 Quick Start

### 🆕 New v1.4.1 Features

#### Loopback Stress Testing
```bash
# Run comprehensive loopback stress test
python scripts/run_loopback_stress_test.py --protocol USB4 --cycles 1000

# Multi-protocol stress testing
python scripts/run_loopback_stress_test.py --protocol PCIe --cycles 500 --output ./stress_results

# Interactive stress test with real-time monitoring
python scripts/run_loopback_stress_test.py --protocol Ethernet --cycles 2000 --interactive
```

#### API Server Management (from root directory)
```bash
# Start API server
python run_api_server.py --host 0.0.0.0 --port 8000

# Test API functionality
python test_api_client.py --no-interactive

# API server with custom configuration
python run_api_server.py --host 127.0.0.1 --port 8080 --no-reload
```

### PCIe 6.0 Validation
```bash
# Complete PCIe validation (both NRZ and PAM4)
python scripts/pcie_validation.py --mode both --lanes 4 --verbose

# PAM4-only validation with benchmarks
python scripts/pcie_validation.py --mode pam4 --benchmark

# Multi-lane validation with custom sample count
python scripts/pcie_validation.py --mode both --lanes 8 --samples 20000
```

### USB4/Thunderbolt 4 Validation
```bash
# Complete USB4/Thunderbolt 4 validation
python scripts/usb4_validation.py --mode both --tunneling all --verbose

# Thunderbolt 4 certification testing
python scripts/usb4_validation.py --mode thunderbolt --certification --benchmark

# USB4 tunneling validation
python scripts/usb4_validation.py --mode usb4 --tunneling pcie --samples 10000
```

### Multi-Protocol Validation
```bash
# Validate all protocols with cross-comparison
python scripts/multi_protocol_validation.py --protocols all --compare --benchmark

# Compare specific protocols
python scripts/multi_protocol_validation.py --protocols pcie,usb4 --compare --verbose

# Comprehensive validation with all features
python scripts/multi_protocol_validation.py --protocols all --samples 12000 --benchmark --compare
```

### Data Analysis
```bash
# Analyze sample data
python scripts/data_analysis.py --verbose

# Analyze custom data file
python scripts/data_analysis.py --input data.csv --output ./results

# Generate comprehensive analysis report
python scripts/data_analysis.py --input signal_data.npy --verbose
```

### Instrument Control
```bash
# Test instrument connectivity
python scripts/instrument_control.py

# Force mock mode for testing
SVF_MOCK_MODE=1 python scripts/instrument_control.py
```

## 📊 Script Categories

### 1. Validation Scripts
- **PCIe 6.0 Validation** - Complete protocol compliance testing
- **224G Ethernet Validation** - High-speed Ethernet testing
- **USB4/Thunderbolt 4 Validation** - Comprehensive USB4 and Thunderbolt 4 testing
- **Multi-protocol Validation** - Cross-protocol analysis

### 2. Analysis Scripts
- **Signal Analysis** - Comprehensive signal processing
- **Statistical Analysis** - Advanced statistical methods
- **Performance Analysis** - Throughput and timing analysis

### 3. Control Scripts
- **Instrument Control** - Hardware interface testing
- **Test Automation** - Automated test sequences
- **Data Collection** - Systematic data gathering

### 4. Utility Scripts
- **Configuration Management** - Setup and configuration
- **Report Generation** - Automated reporting
- **Performance Monitoring** - System performance tracking

### 5. 🆕 Stress Testing Scripts (v1.4.1)
- **Loopback Stress Testing** - Progressive signal degradation simulation
- **Multi-cycle Testing** - Long-duration stress validation
- **Real-time Monitoring** - Live stress test monitoring and reporting

## 🔧 Command-Line Options

### PCIe Validation Script
```bash
python scripts/pcie_validation.py [OPTIONS]

Options:
  --mode {nrz,pam4,both}     Signal mode to test (default: both)
  --lanes N                  Number of lanes (1-16, default: 1)
  --samples N                Samples per test (default: 10000)
  --output DIR               Output directory (default: ./results)
  --verbose                  Enable verbose logging
  --benchmark                Run performance benchmarks
```

### Data Analysis Script
```bash
python scripts/data_analysis.py [OPTIONS]

Options:
  --input FILE               Input data file (CSV/NPY/TXT)
  --output DIR               Output directory (default: ./analysis_output)
  --verbose                  Enable verbose logging
```

### Instrument Control Script
```bash
python scripts/instrument_control.py [OPTIONS]

Options:
  --resource ADDR            Instrument resource address
  --verbose                  Enable verbose logging
  --mock                     Force mock mode
```

### USB4/Thunderbolt 4 Validation Script
```bash
python scripts/usb4_validation.py [OPTIONS]

Options:
  --mode {usb4,thunderbolt,both}    Protocol mode to test (default: both)
  --lanes {1,2}                     Number of lanes (default: 2)
  --tunneling {pcie,dp,usb32,all}   Tunneling protocols (default: all)
  --samples N                       Samples per test (default: 8000)
  --output DIR                      Output directory (default: ./usb4_results)
  --verbose                         Enable verbose logging
  --benchmark                       Run performance benchmarks
  --certification                   Run Thunderbolt 4 certification tests
  --mock                           Force mock mode
```

### Multi-Protocol Validation Script
```bash
python scripts/multi_protocol_validation.py [OPTIONS]

Options:
  --protocols LIST              Protocols to test: pcie,eth224g,usb4,thunderbolt,all (default: all)
  --samples N                   Samples per test (default: 8000)
  --output DIR                  Output directory (default: ./multi_protocol_results)
  --verbose                     Enable verbose logging
  --benchmark                   Run performance benchmarks
  --compare                     Generate cross-protocol comparison
  --mock                       Force mock mode
```

### 🆕 Loopback Stress Test Script (v1.4.1)
```bash
python scripts/run_loopback_stress_test.py [OPTIONS]

Options:
  --protocol {USB4,PCIe,Ethernet}   Protocol to test (default: USB4)
  --cycles N                        Number of test cycles (default: 100)
  --output DIR                      Output directory (default: ./stress_results)
  --interactive                     Enable interactive monitoring
  --verbose                         Enable verbose logging
  --config FILE                     Custom configuration file
```

## 📈 Performance Benchmarks

### PCIe Validation Performance
```bash
# Benchmark results from pcie_validation.py --benchmark:
# - NRZ Analysis (10K samples): ~0.8 seconds
# - PAM4 Analysis (10K samples): ~1.2 seconds
# - Mode Switching: ~5-8 milliseconds
# - Link Training: ~3-5 seconds
# - Compliance Testing: ~2-3 seconds
```

### Data Analysis Throughput
```bash
# Typical throughput from data_analysis.py:
# - Basic Statistics: ~100K samples/second
# - Advanced Analysis: ~50K samples/second
# - PAM4 Analysis: ~25K samples/second
```

## 🎯 Usage Scenarios

### Production Testing
```bash
# Complete PCIe 6.0 validation for production
python scripts/pcie_validation.py \
  --mode both \
  --lanes 16 \
  --samples 50000 \
  --output ./production_results \
  --benchmark

# Complete USB4/Thunderbolt 4 validation for production
python scripts/usb4_validation.py \
  --mode both \
  --tunneling all \
  --samples 12000 \
  --certification \
  --benchmark \
  --output ./usb4_production_results

# Comprehensive multi-protocol validation for production
python scripts/multi_protocol_validation.py \
  --protocols all \
  --samples 15000 \
  --benchmark \
  --compare \
  --output ./multi_protocol_production_results
```

### Development Testing
```bash
# Quick validation during development
SVF_MOCK_MODE=1 python scripts/pcie_validation.py \
  --mode pam4 \
  --lanes 2 \
  --samples 5000 \
  --verbose

# USB4 development testing with mock mode
python scripts/usb4_validation.py \
  --mode usb4 \
  --tunneling pcie \
  --samples 4000 \
  --mock \
  --verbose
```

### Automated CI/CD
```bash
# Continuous integration testing
python scripts/pcie_validation.py \
  --mode both \
  --lanes 4 \
  --samples 10000 \
  --output ./ci_results 2>&1 | tee validation.log
```

### Research and Analysis
```bash
# Detailed analysis for research
python scripts/data_analysis.py \
  --input research_data.csv \
  --output ./research_results \
  --verbose
```

## 📊 Output Formats

### Validation Reports
- **Text Reports** - Human-readable validation summaries
- **CSV Data** - Machine-readable measurement data
- **JSON Results** - Structured result data
- **Log Files** - Detailed execution logs

### Analysis Results
- **Statistical Summaries** - Mean, std, min, max, etc.
- **Signal Quality Metrics** - SNR, EVM, BER estimates
- **Performance Metrics** - Throughput, timing, efficiency
- **Visualization Data** - Plot data and histograms

## 🔍 Error Handling

### Common Error Scenarios
1. **Hardware Connection Issues** - Automatic fallback to mock mode
2. **Invalid Parameters** - Clear error messages and suggestions
3. **Data Format Issues** - Automatic format detection and conversion
4. **Resource Limitations** - Graceful degradation and warnings

### Debug Information
```bash
# Enable maximum debug output
python scripts/pcie_validation.py --verbose 2>&1 | tee debug.log

# Check log files for detailed information
tail -f pcie_validation.log
```

## 🔧 Configuration

### Environment Variables
```bash
# Mock mode control
export SVF_MOCK_MODE=1          # Force mock mode
export SVF_MOCK_MODE=0          # Force real hardware
unset SVF_MOCK_MODE             # Auto-detect (default)

# Logging control
export SVF_LOG_LEVEL=DEBUG      # Enable debug logging
export SVF_LOG_LEVEL=INFO       # Standard logging (default)
```

### Configuration Files
- Scripts support configuration via command-line arguments
- Default settings optimized for typical use cases
- Custom configurations can be specified via parameters

## 📚 Integration Examples

### Shell Scripts
```bash
#!/bin/bash
# Automated validation pipeline
echo "Starting PCIe validation pipeline..."

# Run validation
python scripts/pcie_validation.py \
  --mode both \
  --lanes 4 \
  --output ./pipeline_results

# Check results
if [ $? -eq 0 ]; then
    echo "Validation PASSED"
    exit 0
else
    echo "Validation FAILED"
    exit 1
fi
```

### Python Integration
```python
import subprocess
import sys

# Run PCIe validation from Python
result = subprocess.run([
    sys.executable, 'scripts/pcie_validation.py',
    '--mode', 'pam4',
    '--lanes', '2',
    '--output', './python_results'
], capture_output=True, text=True)

if result.returncode == 0:
    print("Validation successful")
    print(result.stdout)
else:
    print("Validation failed")
    print(result.stderr)
```

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Run PCIe Validation
  run: |
    python scripts/pcie_validation.py \
      --mode both \
      --lanes 4 \
      --output ./ci_results
    
- name: Upload Results
  uses: actions/upload-artifact@v2
  with:
    name: validation-results
    path: ./ci_results/
```

## 🛠️ Development

### Adding New Scripts
1. Follow the existing script structure and patterns
2. Include comprehensive command-line argument parsing
3. Add proper logging and error handling
4. Include performance monitoring and reporting
5. Test with both mock and real hardware scenarios

### Script Template
```python
#!/usr/bin/env python3
"""
Script Description

Usage:
    python scripts/script_name.py [options]

Author: Your Name
Date: Current Date
Version: 1.4.0
"""

import argparse
import logging
import sys
from pathlib import Path

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Script Description")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    logger.info("Script starting...")
    
    try:
        # Script logic here
        pass
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 📄 License

All scripts are provided under the same MIT license as the main framework.

---

**Happy Validating!** 🚀

For questions or issues, please refer to the main project documentation or open an issue on GitHub.