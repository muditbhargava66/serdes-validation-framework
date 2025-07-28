# REST API Documentation

The SerDes Validation Framework provides a comprehensive REST API for remote access to all framework capabilities. The API is built using FastAPI and provides automatic OpenAPI documentation.

## Quick Start

### Starting the API Server

```bash
# Using the CLI
python -m serdes_validation_framework.api.cli server --host 0.0.0.0 --port 8000

# Or programmatically
python -c "
from serdes_validation_framework.api import create_app
import uvicorn
app = create_app()
uvicorn.run(app, host='0.0.0.0', port=8000)
"
```

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## API Endpoints

### System Status

#### GET /api/v1/status
Get system status and health information.

**Response:**
```json
{
  "version": "1.4.1",
  "status": "healthy",
  "active_tests": 0,
  "total_tests_run": 42,
  "uptime_seconds": 3600,
  "memory_usage_mb": 256.5,
  "available_protocols": ["USB4", "PCIe", "Ethernet"],
  "features": {
    "eye_analysis": true,
    "stress_testing": true,
    "fixture_control": true
  }
}
```

### Eye Diagram Analysis

#### POST /api/v1/eye-diagram/analyze
Analyze eye diagram from signal data.

**Request:**
```json
{
  "signal_data": [0.1, -0.1, 0.2, -0.2],
  "sample_rate": 40000000000,
  "protocol": "USB4",
  "lane": 0,
  "show_mask": true,
  "time_data": null
}
```

**Response:**
```json
{
  "eye_height": 0.8234,
  "eye_width": 0.7,
  "eye_area": 0.5764,
  "q_factor": 12.5,
  "snr": 25.2,
  "passed": true,
  "lane": "lane_0",
  "protocol": "USB4",
  "mask_analysis": {
    "mask_passed": true,
    "violations": 0,
    "compliance_level": "pass",
    "eye_opening_percentage": 85.2
  },
  "timestamp": 1640995200.0
}
```

#### POST /api/v1/eye-diagram/mask-compliance
Check eye mask compliance without full analysis.

### Waveform Analysis

#### POST /api/v1/waveform/analyze
Analyze waveform signal quality.

**Request:**
```json
{
  "signal_data": {
    "lane_0": [0.1, -0.1, 0.2],
    "lane_1": [0.15, -0.05, 0.25]
  },
  "sample_rate": 40000000000,
  "protocol": "PCIe",
  "lane": "lane_0",
  "time_data": null
}
```

**Response:**
```json
{
  "protocol": "PCIe",
  "lane": "lane_0",
  "peak_to_peak": 0.4,
  "rms": 0.141,
  "snr": 20.5,
  "thd": 0.05,
  "passed": true,
  "failure_reasons": [],
  "timestamp": 1640995200.0
}
```

### Stress Testing

#### POST /api/v1/stress-test/start
Start an asynchronous stress test.

**Request:**
```json
{
  "protocol": "USB4",
  "num_cycles": 1000,
  "cycle_duration": 1.0,
  "enable_bert_hooks": false,
  "degradation_rate": 0.001,
  "output_dir": null
}
```

**Response:**
```json
{
  "test_id": "stress_test_abc123",
  "status": "started",
  "message": "Stress test started successfully"
}
```

#### GET /api/v1/stress-test/{test_id}/result
Get stress test results.

**Response:**
```json
{
  "test_id": "stress_test_abc123",
  "protocol": "USB4",
  "total_cycles": 1000,
  "passed_cycles": 987,
  "failed_cycles": 13,
  "success_rate": 0.987,
  "duration": 1000.5,
  "max_degradation": 15.2,
  "initial_eye_height": 0.8,
  "final_eye_height": 0.678,
  "timestamp": 1640995200.0
}
```

### Test Management

#### GET /api/v1/test/{test_id}/status
Get test status and progress.

**Response:**
```json
{
  "test_id": "stress_test_abc123",
  "status": "running",
  "progress": 0.65,
  "message": "Processing cycle 650/1000",
  "started_at": 1640995200.0,
  "estimated_completion": 1640995800.0
}
```

#### DELETE /api/v1/test/{test_id}/cancel
Cancel a running test.

### Fixture Control

#### POST /api/v1/fixture/control
Control test fixtures and equipment.

**Request:**
```json
{
  "fixture_name": "probe_station_1",
  "action": "set_voltage",
  "parameters": {
    "voltage": 3.3
  }
}
```

**Response:**
```json
{
  "fixture_name": "probe_station_1",
  "connected": true,
  "temperature": 25.5,
  "voltage": 3.3,
  "status": "ready",
  "last_calibration": "2024-01-15T10:30:00Z"
}
```

#### GET /api/v1/fixture/{fixture_name}/status
Get fixture status.

## CLI Client

The framework includes a command-line client for API interaction:

### Eye Diagram Analysis
```bash
python -m serdes_validation_framework.api.cli analyze-eye \
    --signal-file signal_data.csv \
    --protocol USB4 \
    --sample-rate 40e9 \
    --show-mask \
    --output results.json
```

### Waveform Analysis
```bash
python -m serdes_validation_framework.api.cli analyze-waveform \
    --signal-file multi_lane_data.csv \
    --protocol PCIe \
    --lane lane_0 \
    --sample-rate 80e9
```

### Stress Testing
```bash
python -m serdes_validation_framework.api.cli start-stress-test \
    --protocol USB4 \
    --cycles 1000 \
    --duration 1.0 \
    --output-dir stress_results/
```

### System Status
```bash
python -m serdes_validation_framework.api.cli status
```

## Python Client Examples

### Basic Usage
```python
import requests
import numpy as np

# API base URL
BASE_URL = "http://localhost:8000/api/v1"

# Generate test signal
signal = np.random.randn(1000) * 0.4

# Analyze eye diagram
response = requests.post(f"{BASE_URL}/eye-diagram/analyze", json={
    "signal_data": signal.tolist(),
    "sample_rate": 40e9,
    "protocol": "USB4",
    "show_mask": True
})

if response.status_code == 200:
    result = response.json()
    print(f"Eye Height: {result['eye_height']:.4f}V")
    print(f"Q-Factor: {result['q_factor']:.2f}")
    print(f"Compliance: {result['mask_analysis']['compliance_level']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Async Stress Testing
```python
import requests
import time

# Start stress test
response = requests.post(f"{BASE_URL}/stress-test/start", json={
    "protocol": "PCIe",
    "num_cycles": 500,
    "cycle_duration": 2.0
})

test_id = response.json()["test_id"]
print(f"Started test: {test_id}")

# Monitor progress
while True:
    status_response = requests.get(f"{BASE_URL}/test/{test_id}/status")
    status = status_response.json()
    
    print(f"Status: {status['status']} - Progress: {status.get('progress', 0):.1%}")
    
    if status['status'] in ['completed', 'failed', 'cancelled']:
        break
    
    time.sleep(5)

# Get results
if status['status'] == 'completed':
    result_response = requests.get(f"{BASE_URL}/stress-test/{test_id}/result")
    result = result_response.json()
    print(f"Success Rate: {result['success_rate']:.1%}")
    print(f"Max Degradation: {result['max_degradation']:.1f}%")
```

### Multi-lane Waveform Analysis
```python
# Multi-lane signal data
signal_data = {
    "lane_0": np.random.randn(1000) * 0.4,
    "lane_1": np.random.randn(1000) * 0.35,
    "lane_2": np.random.randn(1000) * 0.42,
    "lane_3": np.random.randn(1000) * 0.38
}

# Analyze each lane
results = {}
for lane, data in signal_data.items():
    response = requests.post(f"{BASE_URL}/waveform/analyze", json={
        "signal_data": {lane: data.tolist()},
        "sample_rate": 80e9,
        "protocol": "PCIe",
        "lane": lane
    })
    
    if response.status_code == 200:
        results[lane] = response.json()

# Compare results
for lane, result in results.items():
    print(f"{lane}: SNR={result['snr']:.1f}dB, "
          f"P2P={result['peak_to_peak']:.3f}V, "
          f"Pass={result['passed']}")
```

## Error Handling

The API uses standard HTTP status codes and provides detailed error messages:

### Common Error Responses

#### 400 Bad Request
```json
{
  "detail": "Invalid sample rate: must be positive",
  "error_code": "VALIDATION_ERROR"
}
```

#### 404 Not Found
```json
{
  "detail": "Test ID not found: stress_test_xyz789",
  "error_code": "TEST_NOT_FOUND"
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Eye diagram analysis failed: insufficient data points",
  "error_code": "ANALYSIS_ERROR"
}
```

### Error Handling in Python
```python
import requests

try:
    response = requests.post(f"{BASE_URL}/eye-diagram/analyze", json={
        "signal_data": [],  # Invalid: empty data
        "sample_rate": 40e9,
        "protocol": "USB4"
    })
    response.raise_for_status()
    result = response.json()
except requests.exceptions.HTTPError as e:
    error_detail = e.response.json()
    print(f"API Error: {error_detail['detail']}")
    print(f"Error Code: {error_detail.get('error_code', 'UNKNOWN')}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

## Configuration

### Environment Variables
- `SVF_API_HOST`: API server host (default: "127.0.0.1")
- `SVF_API_PORT`: API server port (default: 8000)
- `SVF_MOCK_MODE`: Enable mock mode for testing (default: "0")
- `SVF_LOG_LEVEL`: Logging level (default: "INFO")

### CORS Configuration
The API includes CORS middleware for web browser access:

```python
from serdes_validation_framework.api import create_app

app = create_app()

# CORS is pre-configured for:
# - Origins: ["http://localhost:3000", "http://localhost:8080"]
# - Methods: ["GET", "POST", "PUT", "DELETE"]
# - Headers: ["*"]
```

## Security Considerations

- The API runs in development mode by default
- For production deployment, use a proper ASGI server like Gunicorn
- Consider implementing authentication and rate limiting
- Validate all input data thoroughly
- Use HTTPS in production environments

## Performance Tips

- Use appropriate sample rates for your analysis needs
- For large datasets, consider chunking the data
- Monitor memory usage during long-running stress tests
- Use async clients for better performance with multiple requests
- Cache frequently accessed system status information