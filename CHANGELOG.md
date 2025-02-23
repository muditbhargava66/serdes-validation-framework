# Changelog

All notable changes to the SerDes Validation Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2024-02-22

### Added
- Mock Controller Support
  - Added MockInstrumentController class
  - Environment variable control (SVF_MOCK_MODE)
  - Realistic mock data generation
  - Configurable error simulation
- Enhanced Testing Framework
  - Comprehensive test coverage for all modules
  - Better test isolation
  - Mock-aware test infrastructure
  - Improved test documentation
- New Validation Features
  - Resource name validation
  - Data type checking
  - Enhanced error handling
  - Better state tracking
- Improved Scripts
  - Better result formatting
  - Enhanced error reporting
  - Progress tracking
  - Cleanup validation
- ReadTheDocs Integration [2024-20-23]
  - Set up documentation hosting on ReadTheDocs
  - Configured automatic documentation builds
  - Improved navigation and search capabilities
  - Added links to hosted documentation

### Changed
- Updated TestSequencer implementation
  - Added controller injection support
  - Improved cleanup handling
  - Better state management
  - Enhanced error reporting
- Enhanced DataCollector functionality
  - Added controller support
  - Better resource validation
  - Improved connection handling
  - Enhanced data collection
- Improved error handling
  - More specific error messages
  - Better error categorization
  - Enhanced error recovery
  - Proper cleanup in error cases
- Updated documentation
  - Added mock mode documentation
  - Enhanced API documentation
  - Improved example coverage
  - Better usage guidelines

### Fixed
- Resource cleanup in error cases
- Mock response handling
- Numeric data validation
- Connection state tracking
- Test isolation issues
- Error propagation in cleanup
- Response type handling
- Instrument state management

## [1.1.0] - 2024-02-22

### Added
- 224G Ethernet PHY support
  - New test sequences for 224G validation
  - PAM4 signal analysis
  - Advanced equalization algorithms
  - High-bandwidth scope control
- Enhanced type checking and validation throughout codebase
- New protocol-specific modules under `protocols/`
- Comprehensive eye diagram analysis
- Advanced jitter measurements

### Changed
- Updated EyeResults dataclass structure
  - Renamed `worst_eye_height` to `worst_height`
  - Renamed `worst_eye_width` to `worst_width`
- Enhanced data type validation in analysis modules
- Improved error handling and logging
- Updated documentation with 224G examples

### Fixed
- Data type issues in histogram analysis
- Eye diagram measurement stability
- Signal normalization accuracy
- Mock instrument response handling

## [1.0.0] - 2024-06-26

### Added
- Initial release of SerDes Validation Framework
- Basic test sequence support
- Data collection from lab instruments
- Data analysis capabilities
- GPIB instrument control
- Basic visualization tools
- Unit tests and documentation

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

[1.2.0]: https://github.com/muditbhargava66/serdes-validation-framework/compare/v1.1.0...v1.2.0
[1.1.0]: N/A
[1.0.0]: https://github.com/muditbhargava66/serdes-validation-framework/releases/tag/v1.0.0

## Release History Quick Reference

| Version | Release Date | Major Changes |
|---------|--------------|---------------|
| 1.2.0   | 2025-02-22   | Added Mock Controller Support, Enhanced Testing Framework |
| 1.1.0   | 2025-02-21   | Added 224G Ethernet support, Enhanced type checking |
| 1.0.0   | 2024-06-26   | Initial release |

## Version Compatibility Matrix

| Version | Python Version | Key Dependencies |
|---------|---------------|------------------|
| 1.2.0   | ≥3.10        | numpy ≥1.21.0, pandas ≥1.7.0, pyvisa ≥1.11.0, pytest ≥7.1.1 |
| 1.1.0   | ≥3.10        | numpy ≥1.21.0, pandas ≥1.7.0, pyvisa ≥1.11.0 |
| 1.0.0   | ≥3.7         | numpy ≥1.19.0, pandas ≥1.3.0, pyvisa ≥1.11.0 |

## Future Plans

### [1.3.0] - Planned
- PCI Express 6.0 support
  - PCIe 6.0 compliance testing
  - NRZ/PAM4 dual-mode support
  - Advanced link training analysis
  - Enhanced equalization algorithms

### [1.4.0] - Planned
- Cloud Integration and Automation
  - Cloud data storage
  - Remote instrument control
  - Automated compliance testing
  - CI/CD pipeline integration
  - Machine learning analysis

### [1.5.0] - Planned
- Advanced Analysis Features
  - Real-time analysis capabilities
  - Advanced visualization tools
  - Automated report generation
  - Lab automation integration
  - Predictive error analysis