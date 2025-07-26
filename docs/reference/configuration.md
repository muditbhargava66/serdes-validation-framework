# Configuration Reference

This document provides comprehensive information about configuring the SerDes Validation Framework.

## Environment Variables

### Core Configuration

#### `SVF_MOCK_MODE`
- **Type**: Boolean (0/1, true/false, yes/no)
- **Default**: `0` (disabled)
- **Description**: Enables mock mode for testing without hardware
- **Example**: `export SVF_MOCK_MODE=1`

#### `SVF_LOG_LEVEL`
- **Type**: String
- **Default**: `INFO`
- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Description**: Sets the logging level for the framework
- **Example**: `export SVF_LOG_LEVEL=DEBUG`

#### `SVF_CONFIG_PATH`
- **Type**: Path
- **Default**: `~/.svf/config.yaml`
- **Description**: Path to the main configuration file
- **Example**: `export SVF_CONFIG_PATH=/path/to/config.yaml`

#### `SVF_DATA_PATH`
- **Type**: Path
- **Default**: `~/.svf/data`
- **Description**: Directory for storing test data and results
- **Example**: `export SVF_DATA_PATH=/data/svf`

#### `SVF_CACHE_PATH`
- **Type**: Path
- **Default**: `~/.svf/cache`
- **Description**: Directory for caching processed data
- **Example**: `export SVF_CACHE_PATH=/tmp/svf_cache`

### USB4/Thunderbolt Configuration

#### `SVF_USB4_DEFAULT_MODE`
- **Type**: String
- **Default**: `GEN3_X2`
- **Options**: `GEN2_X2`, `GEN3_X2`, `ASYMMETRIC`
- **Description**: Default USB4 signal mode for validation
- **Example**: `export SVF_USB4_DEFAULT_MODE=GEN2_X2`

#### `SVF_USB4_SSC_ENABLED`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enable spread spectrum clocking by default
- **Example**: `export SVF_USB4_SSC_ENABLED=false`

#### `SVF_THUNDERBOLT_SECURITY_LEVEL`
- **Type**: String
- **Default**: `HIGH`
- **Options**: `LOW`, `MEDIUM`, `HIGH`
- **Description**: Default Thunderbolt security level
- **Example**: `export SVF_THUNDERBOLT_SECURITY_LEVEL=MEDIUM`

### Instrument Configuration

#### `SVF_INSTRUMENT_TIMEOUT`
- **Type**: Integer (milliseconds)
- **Default**: `30000`
- **Description**: Default timeout for instrument communication
- **Example**: `export SVF_INSTRUMENT_TIMEOUT=60000`

#### `SVF_INSTRUMENT_RETRY_COUNT`
- **Type**: Integer
- **Default**: `3`
- **Description**: Number of retries for failed instrument operations
- **Example**: `export SVF_INSTRUMENT_RETRY_COUNT=5`

#### `SVF_SCOPE_ADDRESS`
- **Type**: String
- **Default**: `None`
- **Description**: Default oscilloscope VISA address
- **Example**: `export SVF_SCOPE_ADDRESS="TCPIP::192.168.1.100::INSTR"`

### Reporting Configuration

#### `SVF_REPORT_OUTPUT_DIR`
- **Type**: Path
- **Default**: `./reports`
- **Description**: Default directory for generated reports
- **Example**: `export SVF_REPORT_OUTPUT_DIR=/reports/usb4`

#### `SVF_REPORT_FORMAT`
- **Type**: String
- **Default**: `HTML`
- **Options**: `HTML`, `PDF`, `JSON`, `XML`
- **Description**: Default report format
- **Example**: `export SVF_REPORT_FORMAT=PDF`

#### `SVF_REPORT_INCLUDE_CHARTS`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Include charts in reports by default
- **Example**: `export SVF_REPORT_INCLUDE_CHARTS=false`

### Performance Configuration

#### `SVF_PARALLEL_PROCESSING`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enable parallel processing for analysis
- **Example**: `export SVF_PARALLEL_PROCESSING=false`

#### `SVF_MAX_WORKERS`
- **Type**: Integer
- **Default**: `4`
- **Description**: Maximum number of worker processes
- **Example**: `export SVF_MAX_WORKERS=8`

#### `SVF_MEMORY_LIMIT`
- **Type**: String
- **Default**: `4GB`
- **Description**: Memory limit for processing large datasets
- **Example**: `export SVF_MEMORY_LIMIT=8GB`

## Configuration Files

### Main Configuration File

The main configuration file is located at `~/.svf/config.yaml` by default.

```yaml
# SerDes Validation Framework Configuration

# Core settings
core:
  mock_mode: false
  log_level: INFO
  data_path: ~/.svf/data
  cache_path: ~/.svf/cache
  parallel_processing: true
  max_workers: 4

# USB4/Thunderbolt settings
usb4:
  default_mode: GEN3_X2
  ssc_enabled: true
  max_lane_skew_ps: 20
  eye_mask_margin: 0.1
  jitter_tolerance: 0.05

thunderbolt:
  security_level: HIGH
  dma_protection: true
  device_authentication: true
  max_daisy_devices: 6
  certification_mode: false

# Instrument settings
instruments:
  timeout_ms: 30000
  retry_count: 3
  auto_detect: true
  
  oscilloscope:
    default_address: null
    sample_rate: 100e9
    record_length: 1000000
    bandwidth: 50e9
  
  pattern_generator:
    default_address: null
    output_amplitude: 0.8
    rise_time: 10e-12
  
  power_meter:
    default_address: null
    measurement_range: auto

# Reporting settings
reporting:
  output_directory: ./reports
  default_format: HTML
  include_charts: true
  include_raw_data: false
  auto_open: false
  
  templates:
    compliance:
      sections: [summary, results, recommendations]
      include_charts: true
    
    certification:
      sections: [cover, summary, results, appendix]
      include_charts: true
      include_raw_data: true

# Visualization settings
visualization:
  output_directory: ./plots
  default_format: PNG
  dpi: 300
  figure_size: [12, 8]
  color_scheme: default
  interactive: false

# Performance settings
performance:
  memory_limit: 4GB
  cache_enabled: true
  cache_size: 1GB
  compression: true
  
# Development settings
development:
  debug_mode: false
  profiling: false
  test_data_path: ./test_data
  mock_data_seed: 42
```

### Protocol-Specific Configuration

#### USB4 Configuration (`~/.svf/usb4_config.yaml`)

```yaml
# USB4 Protocol Configuration

signal_modes:
  GEN2_X2:
    data_rate: 20e9  # 20 Gbps
    ui_period: 50e-12  # 50 ps
    voltage_swing: 1.2  # V
    
  GEN3_X2:
    data_rate: 40e9  # 40 Gbps
    ui_period: 25e-12  # 25 ps
    voltage_swing: 1.2  # V

compliance_limits:
  eye_height_min: 0.65
  eye_width_min: 0.65
  jitter_rms_max: 0.025
  jitter_pp_max: 0.15
  lane_skew_max: 20e-12  # 20 ps
  
spread_spectrum:
  enabled: true
  frequency: 33000  # Hz
  deviation: 0.005  # 0.5%
  
tunneling:
  pcie:
    max_bandwidth_allocation: 0.8
    latency_requirement: 1e-6  # 1 μs
  
  displayport:
    max_displays: 2
    max_resolution: "4K"
    max_refresh_rate: 60
  
  usb32:
    backward_compatibility: true
    enumeration_timeout: 5.0  # seconds
```

#### Thunderbolt Configuration (`~/.svf/thunderbolt_config.yaml`)

```yaml
# Thunderbolt 4 Configuration

certification:
  intel_requirements: true
  security_mandatory: true
  power_delivery_required: true
  
security:
  dma_protection: true
  iommu_required: true
  device_authentication: true
  user_authorization: true
  
daisy_chain:
  max_devices: 6
  bandwidth_management: dynamic
  topology_validation: true
  
power_delivery:
  max_power: 100  # W
  voltage_levels: [5, 9, 15, 20]  # V
  current_limits: [3, 3, 3, 5]   # A
  
displays:
  max_4k_displays: 2
  max_8k_displays: 1
  dp_version: "1.4a"
```

### Instrument Configuration

#### Oscilloscope Configuration (`~/.svf/instruments/scope_config.yaml`)

```yaml
# Oscilloscope Configuration

default_settings:
  sample_rate: 100e9  # 100 GSa/s
  record_length: 1000000
  bandwidth: 50e9  # 50 GHz
  input_impedance: 50  # Ohms
  coupling: DC
  
channels:
  lane0:
    channel: 1
    scale: 0.2  # V/div
    offset: 0.0  # V
    
  lane1:
    channel: 2
    scale: 0.2  # V/div
    offset: 0.0  # V
    
  clock:
    channel: 3
    scale: 0.5  # V/div
    offset: 0.0  # V

triggers:
  usb4:
    type: edge
    source: lane0
    level: 0.0  # V
    slope: positive
    
measurements:
  eye_diagram:
    persistence: 1000
    mask_test: true
    
  jitter:
    measurement_type: TIE
    reference_clock: recovered
    
supported_models:
  - "Keysight DSAZ634A"
  - "Tektronix DPO77002SX"
  - "Rohde & Schwarz RTO2064"
```

## Configuration Management

### Loading Configuration

```python
from serdes_validation_framework.config import ConfigManager

# Load default configuration
config = ConfigManager()

# Load custom configuration file
config = ConfigManager(config_path="/path/to/custom_config.yaml")

# Access configuration values
mock_mode = config.get('core.mock_mode', False)
usb4_mode = config.get('usb4.default_mode', 'GEN3_X2')
```

### Runtime Configuration

```python
# Update configuration at runtime
config.set('core.log_level', 'DEBUG')
config.set('usb4.ssc_enabled', False)

# Save configuration
config.save()

# Reload configuration
config.reload()
```

### Environment Override

Environment variables take precedence over configuration files:

```python
import os

# Environment variable overrides config file
os.environ['SVF_LOG_LEVEL'] = 'DEBUG'

# This will use DEBUG level regardless of config file
config = ConfigManager()
log_level = config.get_log_level()  # Returns 'DEBUG'
```

## Validation and Defaults

### Configuration Validation

```python
# Validate configuration
validation_result = config.validate()

if not validation_result.is_valid:
    for error in validation_result.errors:
        print(f"Configuration error: {error}")
```

### Default Values

```python
# Get value with default
timeout = config.get('instruments.timeout_ms', 30000)

# Check if value exists
if config.has('usb4.custom_setting'):
    custom_value = config.get('usb4.custom_setting')
```

## Best Practices

### Configuration Organization
- Use separate files for different protocols
- Group related settings together
- Use descriptive names for settings
- Include units in setting names where applicable

### Security Considerations
- Don't store sensitive information in config files
- Use environment variables for secrets
- Restrict file permissions on config files
- Validate all configuration inputs

### Performance Optimization
- Cache frequently accessed settings
- Use appropriate data types
- Avoid deep nesting in configuration
- Consider memory usage for large configurations

### Development vs Production
- Use different configurations for development and production
- Enable debug features only in development
- Use mock mode for CI/CD environments
- Monitor configuration changes in production

## Troubleshooting

### Common Issues

#### Configuration Not Found
```bash
# Check if config file exists
ls -la ~/.svf/config.yaml

# Create default configuration
svf-config --create-default
```

#### Invalid Configuration
```python
# Validate configuration
from serdes_validation_framework.config import validate_config

result = validate_config('/path/to/config.yaml')
if not result.valid:
    for error in result.errors:
        print(f"Error: {error}")
```

#### Environment Variable Issues
```bash
# Check environment variables
env | grep SVF_

# Clear all SVF environment variables
unset $(env | grep '^SVF_' | cut -d= -f1)
```

### Debug Configuration

```python
# Enable configuration debugging
import logging
logging.getLogger('serdes_validation_framework.config').setLevel(logging.DEBUG)

# Print current configuration
config = ConfigManager()
config.print_config()
```

For more configuration examples, see the examples directory and tutorials.