# Documentation v1.4.1

This directory contains the comprehensive documentation for the SerDes Validation Framework, including new REST API and Jupyter Dashboard documentation.

## Building the Documentation

1. Install documentation dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the documentation:
   ```bash
   sphinx-build -b html . _build/html
   ```

3. View the documentation:
   ```bash
   python -m http.server --directory _build/html
   ```

## Directory Structure

```
docs/
├── api/                       # 🆕 Enhanced API documentation (v1.4.1)
│   ├── rest-api.md           # 🆕 Complete REST API reference
│   ├── index.md
│   ├── mock_controller.md     # Mock controller docs
│   ├── instrument_control.md  # Instrument control docs
│   ├── eth_224g.md           # 224G protocol docs
│   └── pam4_analysis.md      # PAM4 analysis docs
├── tutorials/                 # Tutorials
│   ├── getting_started.md    # Getting started guide
│   ├── mock_testing.md       # Mock testing tutorial
│   ├── 224g_validation.md    # 224G validation guide
│   └── pam4_analysis.md      # PAM4 analysis guide
├── guides/                    # 🆕 Enhanced user guides (v1.4.1)
│   ├── installation.md      # Installation guide
│   ├── testing.md          # Comprehensive testing guide (130+ tests)
│   ├── troubleshooting.md   # Troubleshooting guide
│   ├── cicd.md             # CI/CD integration guide
│   ├── jupyter_dashboard_guide.md  # 🆕 Jupyter Dashboard guide
│   └── stress_testing_overview.md  # 🆕 Stress testing guide
├── examples/                  # 🆕 Enhanced code examples (v1.4.1)
│   ├── api_examples.md       # 🆕 REST API usage examples
│   ├── mock_controller_examples.md
│   ├── testing_examples.md
│   ├── protocol_detection_examples.md
│   └── integration_testing_examples.md
├── CHANGELOG.md             # 🆕 Detailed changelog with v1.4.1 improvements
├── CONTRIBUTING.md          # Contribution guide
├── INSTALL.md              # Installation guide
└── USAGE.md                # 🆕 Updated usage guide with API examples
```

## 🆕 New Documentation in v1.4.1

### REST API Documentation
- **Complete API Reference** (`api/rest-api.md`) - All 8 endpoints documented
- **Usage Examples** (`examples/api_examples.md`) - Comprehensive API usage patterns
- **Interactive Documentation** - Available at `/docs` when API server is running

### Jupyter Dashboard Documentation
- **Dashboard Guide** (`guides/jupyter_dashboard_guide.md`) - Interactive eye diagram analysis
- **Usage Examples** - Multi-protocol dashboard examples
- **Integration Guide** - How to integrate with existing workflows

### Stress Testing Documentation
- **Stress Testing Overview** (`guides/stress_testing_overview.md`) - Dual stress testing systems
- **Loopback Testing** - Progressive degradation simulation
- **USB4-Specific Testing** - Protocol-aware stress scenarios

## Documentation Standards

1. Use Markdown syntax
2. Include type hints in code examples
3. Validate floating-point numbers
4. Provide clear error messages
5. Include example usage
6. 🆕 Include API endpoint documentation with request/response examples
7. 🆕 Provide interactive dashboard usage patterns

## Adding New Documentation

1. Create your `.md` file in the appropriate directory
2. Add it to the toctree in index.md
3. Link it from relevant pages
4. Build and test locally
5. Submit a pull request