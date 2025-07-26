# Documentation

This directory contains the documentation for the SerDes Validation Framework.

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
├── api/                       # API documentation
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
├── guides/                    # User guides
│   ├── installation.md      # Installation guide
│   ├── testing.md          # Comprehensive testing guide (91+ tests)
│   ├── troubleshooting.md   # Troubleshooting guide
│   └── cicd.md             # CI/CD integration guide
├── examples/                  # Code examples
│   ├── mock_controller_examples.md
│   ├── testing_examples.md
│   ├── protocol_detection_examples.md
│   └── integration_testing_examples.md
├── CHANGELOG.md             # Detailed changelog with v1.4.0 improvements
├── CONTRIBUTING.md          # Contribution guide
├── INSTALL.md              # Installation guide
└── USAGE.md                # Usage guide
```

## Documentation Standards

1. Use Markdown syntax
2. Include type hints in code examples
3. Validate floating-point numbers
4. Provide clear error messages
5. Include example usage

## Adding New Documentation

1. Create your `.md` file in the appropriate directory
2. Add it to the toctree in index.md
3. Link it from relevant pages
4. Build and test locally
5. Submit a pull request