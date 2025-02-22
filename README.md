# SerDes Validation Framework

![Banner](docs/images/serdes-framework-banner.png)

[![CI](https://github.com/muditbhargava66/serdes-validation-framework/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/muditbhargava66/serdes-validation-framework/actions/workflows/ci.yml)
[![Lint](https://github.com/muditbhargava66/serdes-validation-framework/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/muditbhargava66/serdes-validation-framework/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)

This project provides a comprehensive framework for validating high-speed SerDes protocols. It includes tools for data collection, data analysis, instrument control, and running test sequences.

## Features

- **Automated Data Collection:** Gather data from lab instruments seamlessly.
- **Data Analysis and Visualization:** Analyze and visualize collected data with ease.
- **Instrument Control via GPIB:** Control lab instruments using the General Purpose Interface Bus (GPIB).
- **Customizable Test Sequences:** Define and run customizable test sequences.

## Installation

### Prerequisites

- Python 3.10 or higher
- Git

### Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/muditbhargava66/serdes-validation-framework.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd serdes-validation-framework
    ```

3. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

4. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Refer to the [USAGE.md](docs/USAGE.md) for detailed usage instructions.

### Quick Start

1. **Run Data Collection Example:**

    ```bash
    python examples/data_collection_example.py
    ```

2. **Run Data Analysis Example:**

    ```bash
    python examples/data_analysis_example.py
    ```

3. **Run Instrument Control Example:**

    ```bash
    python examples/instrument_control_example.py
    ```

4. **Run Test Sequence Example:**

    ```bash
    python examples/test_sequence_example.py
    ```

## Documentation

Detailed documentation is available in the `docs/` folder:

- [API Documentation](docs/api/index.md)
- [Usage Guide](docs/USAGE.md)
- [Installation Guide](docs/INSTALL.md)
- [Getting Started Tutorial](docs/tutorials/getting_started.md)
- [Contribution Guide](docs/CONTRIBUTING.md)

## Contributing

We welcome contributions from the community. Please read our [contributing guide](docs/CONTRIBUTING.md) to get started.

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

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

## Community and Support

For any questions, issues, or contributions, please open an issue on the [GitHub repository](https://github.com/muditbhargava66/serdes-validation-framework/issues).

---

Enjoy using the SerDes Validation Framework! Contributions and feedback are always welcome.

---