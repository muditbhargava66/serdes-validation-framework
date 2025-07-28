# SerDes Validation Framework Documentation

Welcome to the comprehensive documentation for the SerDes Validation Framework - a powerful, extensible platform for high-speed serial interface validation and testing.

## 🚀 What's New

### Latest Features (v1.4.1+)
- **🌐 REST API Framework** - Complete web API for remote access and integration
  - FastAPI-based REST API with automatic OpenAPI documentation
  - Eye diagram analysis and waveform analysis endpoints
  - Asynchronous stress testing with real-time monitoring
  - Test fixture control and environmental monitoring
  - CLI client for command-line API interaction
  - Interactive Swagger UI documentation at `/docs`

- **🔄 Loopback Stress Testing** - Complete TX → RX → back to TX simulation with progressive degradation tracking
  - Multi-cycle testing (1000+ cycles) with real-time monitoring
  - Signal quality tracking (eye height, jitter, SNR, BER) over time
  - CSV data export and interactive trend visualization
  - Multi-protocol support (USB4, PCIe, Ethernet)

### Major Features (v1.4.0+)
- **Complete USB4/Thunderbolt 4 Support** - Full implementation with dual-lane analysis, tunneling validation, and certification testing
- **Advanced Mock Testing Infrastructure** - Comprehensive mock system enabling CI/CD testing without hardware
- **Multi-Protocol Validation** - Unified framework supporting USB4, PCIe, Thunderbolt, and 224G Ethernet
- **Intelligent Protocol Detection** - Automatic protocol detection from signal characteristics
- **Enhanced Test Framework** - 91+ core tests with comprehensive coverage and fast execution
- **🎨 Comprehensive Visualization System** - Professional-grade visualizations with interactive dashboards
  - Advanced eye diagrams with automatic measurements
  - Interactive web-based dashboards using Plotly
  - Protocol-specific visualizations (USB4 tunneling, PCIe link training, Ethernet PAM4)
  - Multi-protocol comparison and benchmarking tools
- **Automated Reporting** - Certification-ready reports with trend analysis
- **Production-Ready Testing** - Robust test infrastructure with proper error handling and isolation

## 📚 Documentation Structure

### Getting Started
- [Installation Guide](guides/installation.md) - Complete setup instructions
- [Quick Start Tutorial](tutorials/getting_started.md) - Your first validation test
- [USB4 Quick Start](usb4/quickstart.md) - USB4 validation walkthrough
- [Environment Configuration](reference/configuration.md) - Configuration options

### USB4/Thunderbolt 4 Validation
- [USB4 Documentation Hub](usb4/index.md) - Complete USB4 and Thunderbolt 4 documentation
- [USB4 Quick Start](usb4/quickstart.md) - Get started with USB4 validation
- [Thunderbolt 4 Certification](usb4/certification/thunderbolt4.md) - Complete certification guide
- [USB4 Examples](usb4/examples/basic-validation.md) - Practical validation examples
- [USB4 Best Practices](usb4/guides/best-practices.md) - Optimization and recommendations

### API & Integration
- [REST API Documentation](api/rest-api.md) - Complete REST API reference
- [API Client Examples](examples/api_examples.md) - Python and CLI usage examples
- [Integration Guide](guides/integration.md) - Integrating with external systems

### Protocol Support
- [PCIe Validation](api/pcie.md) - PCIe signal analysis and compliance testing
- [224G Ethernet](api/eth_224g.md) - High-speed Ethernet PHY validation
- Multi-Protocol Testing - Cross-protocol validation

### Testing & Development
- [Mock Testing Guide](tutorials/mock_testing.md) - Testing without hardware
- [Test Framework](guides/testing.md) - Comprehensive testing strategies
- [CI/CD Integration](guides/cicd.md) - Continuous integration setup

### Advanced Features
- Signal Analysis - Eye diagrams, jitter analysis, and more
- [🎨 Visualization System](guides/visualization.md) - Comprehensive visualization guide
- [Visualization API](api/visualization.md) - Complete visualization API reference
- [🔄 Stress Testing](guides/stress_testing.md) - Loopback stress testing guide
- [Reporting](api/reporting.md) - Automated report generation
- [Instrument Control](api/instrument_control.md) - Hardware integration

### Reference
- [API Reference](api/index.md) - Complete API documentation
- [Configuration Reference](reference/configuration.md) - All configuration options
- [Troubleshooting](guides/troubleshooting.md) - Common issues and solutions
- [Contributing](CONTRIBUTING.md) - How to contribute to the project

## 🎯 Key Capabilities

### USB4/Thunderbolt 4 Validation
- Dual-lane signal analysis with skew measurement
- Complete tunneling protocol validation (PCIe, DisplayPort, USB 3.2)
- Thunderbolt 4 security and daisy-chain testing
- Intel certification compliance testing

### Signal Integrity Analysis
- Advanced eye diagram analysis with compliance masks
- Comprehensive jitter decomposition (RJ, DJ, PJ)
- Multi-lane skew analysis and compensation
- Real-time signal quality monitoring

### Test Automation
- Automated test sequence orchestration
- Mock testing for CI/CD environments
- Performance regression testing
- Comprehensive reporting and visualization

## 🛠️ Framework Architecture

The framework is built with modularity and extensibility in mind:

```
SerDes Validation Framework
├── Protocol Modules (USB4, PCIe, Ethernet)
├── Signal Analysis Engine
├── Instrument Control Layer
├── Test Automation Framework
├── Visualization & Reporting
└── Mock Testing Infrastructure
```

## 📊 Supported Protocols

| Protocol | Status | Features |
|----------|--------|----------|
| USB4 2.0 | ✅ Complete | Dual-lane, tunneling, power management |
| Thunderbolt 4 | ✅ Complete | Security, daisy-chain, certification |
| PCIe 6.0 | ✅ Complete | NRZ/PAM4, link training, compliance |
| 224G Ethernet | ✅ Complete | PAM4 analysis, equalization |
| Multi-Protocol | ✅ Complete | Cross-protocol validation |

## 🧪 Testing Infrastructure

The framework includes a comprehensive testing infrastructure:

- **91+ Core Tests Passing** - Extensive test coverage with fast execution (0.83s)
- **Advanced Mock Testing Support** - Test without hardware with realistic signal simulation
- **Multi-Protocol Integration Tests** - Cross-protocol validation testing
- **CI/CD Ready** - Automated testing pipeline with proper test isolation
- **Performance Regression Testing** - Trend analysis and monitoring
- **Intelligent Test Organization** - Core, integration, and performance test categories
- **Conditional Import System** - Graceful degradation when dependencies are missing

## 📈 Getting Started

1. **Install the Framework**
   ```bash
   pip install serdes-validation-framework
   ```

2. **Run Your First Test**
   ```python
   from serdes_validation_framework import USB4Validator
   
   validator = USB4Validator()
   results = validator.validate_compliance(signal_data)
   ```

3. **Generate Reports**
   ```python
   from serdes_validation_framework.reporting import USB4TestReporter
   
   reporter = USB4TestReporter()
   report_path = reporter.generate_compliance_report(session_id)
   ```

## 🤝 Community & Support

- **GitHub Repository**: [SerDes Validation Framework](https://github.com/muditbhargava66/serdes-validation-framework)
- **Issue Tracker**: Report bugs and request features
- **Discussions**: Community support and questions
- **Contributing**: See our [Contributing Guide](CONTRIBUTING.md)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Ready to start validating?** Check out our [Quick Start Guide](tutorials/getting_started.md) to begin your journey with the SerDes Validation Framework.