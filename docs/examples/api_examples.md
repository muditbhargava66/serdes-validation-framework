# API Client Examples

This document provides comprehensive examples for using the SerDes Validation Framework REST API.

## Server Setup

### Starting the API Server

```bash
# Method 1: Using the CLI
python -m serdes_validation_framework.api.cli server --host 0.0.0.0 --port 8000

# Method 2: Using Python
python -c "
from serdes_validation_framework.api import create_app
import uvicorn
app = create_app()
uvicorn.run(app, host='0.0.0.0', port=8000)
"

# Method 3: Using the run script
python run_api_server.py
```

### Server Configuration

```python
# Custom server configuration
from serdes_validation_framework.api import create_app
import uvicorn

app = create_app()

# Production configuration
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    workers=4,
    log_level="info",
    access_log=True
)
```

## Python Client Examples

### Basic Eye Diagram Analysis

```python
import requests
import numpy as np
import json

# API configuration
BASE_URL = "http://localhost:8000/api/v1"
headers = {"Content-Type": "application/json"}

# Generate test signal
np.random.seed(42)  # For reproducible results
signal_data = np.random.randn(1000) * 0.4

# Analyze eye diagram
response = requests.post(
    f"{BASE_URL}/eye-diagram/analyze",
    headers=headers,
    json={
        "signal_data": signal_data.tolist(),
        "sample_rate": 40e9,
        "protocol": "USB4",
        "lane": 0,
        "show_mask": True
    }
)

if response.status_code == 200:
    result = response.json()
    print("Eye Diagram Analysis Results:")
    print(f"  Eye Height: {result['eye_height']:.4f}V")
    print(f"  Eye Width: {result['eye_width']:.4f} UI")
    print(f"  Q-Factor: {result['q_factor']:.2f}")
    print(f"  SNR: {result['snr']:.2f} dB")
    print(f"  Overall Pass: {result['passed']}")
    
    # Mask compliance details
    mask = result['mask_analysis']
    print(f"  Mask Compliance: {mask['compliance_level']}")
    print(f"  Violations: {mask['violations']}")
    print(f"  Eye Opening: {mask['eye_opening_percentage']:.1f}%")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### Multi-Lane Waveform Analysis

```python
import requests
import numpy as np

# Generate multi-lane test data
lanes = ["lane_0", "lane_1", "lane_2", "lane_3"]
signal_data = {}

for i, lane in enumerate(lanes):
    # Simulate different signal qualities per lane
    noise_level = 0.3 + i * 0.05  # Increasing noise per lane
    signal_data[lane] = (np.random.randn(1000) * noise_level).tolist()

# Analyze each lane
results = {}
for lane in lanes:
    response = requests.post(
        f"{BASE_URL}/waveform/analyze",
        headers=headers,
        json={
            "signal_data": {lane: signal_data[lane]},
            "sample_rate": 80e9,
            "protocol": "PCIe",
            "lane": lane
        }
    )
    
    if response.status_code == 200:
        results[lane] = response.json()

# Display results
print("Multi-Lane Waveform Analysis:")
print(f"{'Lane':<8} {'SNR (dB)':<10} {'P2P (V)':<10} {'RMS (V)':<10} {'Pass':<6}")
print("-" * 50)

for lane, result in results.items():
    print(f"{lane:<8} {result['snr']:<10.1f} {result['peak_to_peak']:<10.3f} "
          f"{result['rms']:<10.3f} {result['passed']:<6}")
```

### Asynchronous Stress Testing

```python
import requests
import time
import json

def start_stress_test(protocol, cycles, duration):
    """Start a stress test and return test ID"""
    response = requests.post(
        f"{BASE_URL}/stress-test/start",
        headers=headers,
        json={
            "protocol": protocol,
            "num_cycles": cycles,
            "cycle_duration": duration,
            "enable_bert_hooks": False,
            "degradation_rate": 0.001
        }
    )
    
    if response.status_code == 200:
        return response.json()["test_id"]
    else:
        raise Exception(f"Failed to start test: {response.text}")

def monitor_test_progress(test_id):
    """Monitor test progress until completion"""
    print(f"Monitoring test: {test_id}")
    
    while True:
        # Get test status
        response = requests.get(f"{BASE_URL}/test/{test_id}/status")
        
        if response.status_code == 200:
            status = response.json()
            progress = status.get('progress', 0) * 100
            
            print(f"Status: {status['status']} - Progress: {progress:.1f}% - {status.get('message', '')}")
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                return status['status']
        else:
            print(f"Error getting status: {response.text}")
            break
        
        time.sleep(2)  # Poll every 2 seconds
    
    return None

def get_stress_test_results(test_id):
    """Get final stress test results"""
    response = requests.get(f"{BASE_URL}/stress-test/{test_id}/result")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting results: {response.text}")
        return None

# Example: Run USB4 stress test
try:
    print("Starting USB4 stress test...")
    test_id = start_stress_test("USB4", 100, 1.0)
    
    # Monitor progress
    final_status = monitor_test_progress(test_id)
    
    if final_status == 'completed':
        # Get results
        results = get_stress_test_results(test_id)
        if results:
            print("\nStress Test Results:")
            print(f"  Protocol: {results['protocol']}")
            print(f"  Total Cycles: {results['total_cycles']}")
            print(f"  Passed Cycles: {results['passed_cycles']}")
            print(f"  Failed Cycles: {results['failed_cycles']}")
            print(f"  Success Rate: {results['success_rate']:.1%}")
            print(f"  Duration: {results['duration']:.1f}s")
            print(f"  Max Degradation: {results['max_degradation']:.1f}%")
            print(f"  Initial Eye Height: {results['initial_eye_height']:.4f}V")
            print(f"  Final Eye Height: {results['final_eye_height']:.4f}V")
    else:
        print(f"Test ended with status: {final_status}")

except Exception as e:
    print(f"Error: {e}")
```

### System Status and Health Monitoring

```python
import requests
import time

def get_system_status():
    """Get comprehensive system status"""
    response = requests.get(f"{BASE_URL}/status")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting status: {response.text}")
        return None

def monitor_system_health(duration_minutes=5):
    """Monitor system health over time"""
    end_time = time.time() + (duration_minutes * 60)
    
    print(f"Monitoring system health for {duration_minutes} minutes...")
    print(f"{'Time':<20} {'Status':<10} {'Active Tests':<12} {'Memory (MB)':<12} {'Uptime (s)':<12}")
    print("-" * 70)
    
    while time.time() < end_time:
        status = get_system_status()
        if status:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp:<20} {status['status']:<10} {status['active_tests']:<12} "
                  f"{status['memory_usage_mb']:<12.1f} {status['uptime_seconds']:<12.1f}")
        
        time.sleep(30)  # Check every 30 seconds

# Example usage
status = get_system_status()
if status:
    print("System Status:")
    print(f"  Version: {status['version']}")
    print(f"  Status: {status['status']}")
    print(f"  Active Tests: {status['active_tests']}")
    print(f"  Total Tests Run: {status['total_tests_run']}")
    print(f"  Uptime: {status['uptime_seconds']:.1f} seconds")
    print(f"  Memory Usage: {status['memory_usage_mb']:.1f} MB")
    print(f"  Available Protocols: {', '.join(status['available_protocols'])}")
    print("  Features:")
    for feature, enabled in status['features'].items():
        print(f"    {feature}: {'✓' if enabled else '✗'}")

# Uncomment to monitor health over time
# monitor_system_health(2)  # Monitor for 2 minutes
```

### Fixture Control Examples

```python
import requests

def control_fixture(fixture_name, action, parameters=None):
    """Control a test fixture"""
    request_data = {
        "fixture_name": fixture_name,
        "action": action
    }
    
    if parameters:
        request_data["parameters"] = parameters
    
    response = requests.post(
        f"{BASE_URL}/fixture/control",
        headers=headers,
        json=request_data
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error controlling fixture: {response.text}")
        return None

def get_fixture_status(fixture_name):
    """Get fixture status"""
    response = requests.get(f"{BASE_URL}/fixture/{fixture_name}/status")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting fixture status: {response.text}")
        return None

# Example: Control probe station
fixture_name = "probe_station_1"

# Connect to fixture
print("Connecting to probe station...")
result = control_fixture(fixture_name, "connect", {
    "fixture_type": "PROBE_STATION"
})

if result:
    print(f"Connected: {result['connected']}")
    print(f"Temperature: {result['temperature']:.1f}°C")
    print(f"Status: {result['status']}")

# Set voltage
print("Setting voltage to 3.3V...")
result = control_fixture(fixture_name, "set_voltage", {
    "voltage": 3.3
})

if result:
    print(f"Voltage set to: {result['voltage']}V")

# Get status
status = get_fixture_status(fixture_name)
if status:
    print("Fixture Status:")
    print(f"  Name: {status['fixture_name']}")
    print(f"  Connected: {status['connected']}")
    print(f"  Temperature: {status['temperature']:.1f}°C")
    print(f"  Voltage: {status['voltage']}V")
    print(f"  Status: {status['status']}")
    print(f"  Last Calibration: {status['last_calibration']}")

# Disconnect
print("Disconnecting...")
result = control_fixture(fixture_name, "disconnect")
if result:
    print(f"Disconnected: {not result['connected']}")
```

## CLI Client Examples

### Eye Diagram Analysis

```bash
# Basic eye diagram analysis
python -m serdes_validation_framework.api.cli analyze-eye \
    --signal-file signal_data.csv \
    --protocol USB4 \
    --sample-rate 40e9 \
    --output results.json

# With mask compliance checking
python -m serdes_validation_framework.api.cli analyze-eye \
    --signal-file signal_data.csv \
    --protocol PCIe \
    --sample-rate 80e9 \
    --show-mask \
    --lane 2 \
    --output pcie_eye_results.json
```

### Waveform Analysis

```bash
# Single lane analysis
python -m serdes_validation_framework.api.cli analyze-waveform \
    --signal-file lane_data.csv \
    --protocol Ethernet \
    --sample-rate 224e9 \
    --lane lane_0

# Multi-lane analysis
python -m serdes_validation_framework.api.cli analyze-waveform \
    --signal-file multi_lane_data.csv \
    --protocol PCIe \
    --sample-rate 80e9 \
    --lane lane_1 \
    --output waveform_results.json
```

### Stress Testing

```bash
# Start stress test
python -m serdes_validation_framework.api.cli start-stress-test \
    --protocol USB4 \
    --cycles 1000 \
    --duration 1.5 \
    --output-dir stress_results/

# Monitor test progress
python -m serdes_validation_framework.api.cli test-status \
    --test-id stress_test_abc123

# Get test results
python -m serdes_validation_framework.api.cli test-result \
    --test-id stress_test_abc123 \
    --output final_results.json
```

### System Operations

```bash
# Get system status
python -m serdes_validation_framework.api.cli status

# Start API server
python -m serdes_validation_framework.api.cli server \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4

# Health check
python -m serdes_validation_framework.api.cli health
```

## Advanced Integration Examples

### Batch Processing

```python
import requests
import numpy as np
import concurrent.futures
import json

def process_signal_batch(signal_files, protocol, sample_rate):
    """Process multiple signal files in parallel"""
    
    def analyze_single_file(file_path):
        # Load signal data (implement based on your file format)
        signal_data = np.loadtxt(file_path).tolist()
        
        response = requests.post(
            f"{BASE_URL}/eye-diagram/analyze",
            headers=headers,
            json={
                "signal_data": signal_data,
                "sample_rate": sample_rate,
                "protocol": protocol,
                "show_mask": True
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            result['file'] = file_path
            return result
        else:
            return {"file": file_path, "error": response.text}
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(analyze_single_file, signal_files))
    
    return results

# Example usage
signal_files = ["signal1.csv", "signal2.csv", "signal3.csv"]
results = process_signal_batch(signal_files, "USB4", 40e9)

# Analyze results
passed_count = sum(1 for r in results if r.get('passed', False))
print(f"Batch Analysis Results: {passed_count}/{len(results)} passed")

for result in results:
    if 'error' in result:
        print(f"Error in {result['file']}: {result['error']}")
    else:
        print(f"{result['file']}: Eye Height={result['eye_height']:.4f}V, "
              f"Q-Factor={result['q_factor']:.2f}, Pass={result['passed']}")
```

### Real-time Monitoring Dashboard

```python
import requests
import time
import threading
from collections import deque

class APIMonitor:
    def __init__(self, base_url):
        self.base_url = base_url
        self.running = False
        self.status_history = deque(maxlen=100)
        
    def start_monitoring(self):
        """Start monitoring system status"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                response = requests.get(f"{self.base_url}/status", timeout=5)
                if response.status_code == 200:
                    status = response.json()
                    status['timestamp'] = time.time()
                    self.status_history.append(status)
                    
                    # Check for alerts
                    self._check_alerts(status)
                    
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(10)  # Check every 10 seconds
    
    def _check_alerts(self, status):
        """Check for system alerts"""
        # Memory usage alert
        if status['memory_usage_mb'] > 1000:  # 1GB threshold
            print(f"⚠️  High memory usage: {status['memory_usage_mb']:.1f}MB")
        
        # Active tests alert
        if status['active_tests'] > 10:
            print(f"⚠️  High test load: {status['active_tests']} active tests")
        
        # Health status alert
        if status['status'] != 'healthy':
            print(f"🚨 System unhealthy: {status['status']}")
    
    def get_status_summary(self):
        """Get monitoring summary"""
        if not self.status_history:
            return "No monitoring data available"
        
        latest = self.status_history[-1]
        avg_memory = sum(s['memory_usage_mb'] for s in self.status_history) / len(self.status_history)
        max_active_tests = max(s['active_tests'] for s in self.status_history)
        
        return {
            "latest_status": latest['status'],
            "current_memory_mb": latest['memory_usage_mb'],
            "average_memory_mb": avg_memory,
            "max_active_tests": max_active_tests,
            "uptime_seconds": latest['uptime_seconds'],
            "total_tests_run": latest['total_tests_run']
        }

# Example usage
monitor = APIMonitor(BASE_URL)
monitor.start_monitoring()

try:
    # Let it run for a while
    time.sleep(60)  # Monitor for 1 minute
    
    # Get summary
    summary = monitor.get_status_summary()
    print("Monitoring Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
        
finally:
    monitor.stop_monitoring()
```

### Error Handling and Retry Logic

```python
import requests
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=1.0, backoff=2.0):
    """Decorator for retrying API calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    retries += 1
                    if retries >= max_retries:
                        raise e
                    
                    print(f"Request failed (attempt {retries}/{max_retries}): {e}")
                    print(f"Retrying in {current_delay:.1f} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator

@retry_on_failure(max_retries=3, delay=1.0)
def robust_eye_analysis(signal_data, protocol, sample_rate):
    """Eye analysis with retry logic"""
    response = requests.post(
        f"{BASE_URL}/eye-diagram/analyze",
        headers=headers,
        json={
            "signal_data": signal_data,
            "sample_rate": sample_rate,
            "protocol": protocol,
            "show_mask": True
        },
        timeout=30  # 30 second timeout
    )
    
    response.raise_for_status()  # Raise exception for HTTP errors
    return response.json()

# Example with comprehensive error handling
def safe_api_call():
    try:
        signal_data = np.random.randn(1000) * 0.4
        result = robust_eye_analysis(signal_data.tolist(), "USB4", 40e9)
        
        if result:
            print("Analysis successful:")
            print(f"  Eye Height: {result['eye_height']:.4f}V")
            print(f"  Q-Factor: {result['q_factor']:.2f}")
            return result
        else:
            print("Analysis failed after all retries")
            return None
            
    except requests.exceptions.Timeout:
        print("Request timed out")
    except requests.exceptions.ConnectionError:
        print("Connection error - is the API server running?")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return None

# Run safe API call
result = safe_api_call()
```

## Best Practices

### 1. Connection Management
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure session with connection pooling and retries
session = requests.Session()

# Retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Use session for all requests
response = session.post(f"{BASE_URL}/eye-diagram/analyze", json=data)
```

### 2. Data Validation
```python
def validate_signal_data(signal_data, min_length=100):
    """Validate signal data before sending to API"""
    if not isinstance(signal_data, (list, dict)):
        raise ValueError("Signal data must be list or dict")
    
    if isinstance(signal_data, list):
        if len(signal_data) < min_length:
            raise ValueError(f"Signal data too short: {len(signal_data)} < {min_length}")
        if not all(isinstance(x, (int, float)) for x in signal_data):
            raise ValueError("Signal data must contain only numbers")
    
    elif isinstance(signal_data, dict):
        for lane, data in signal_data.items():
            if not isinstance(data, list):
                raise ValueError(f"Lane {lane} data must be a list")
            validate_signal_data(data, min_length)
    
    return True
```

### 3. Performance Optimization
```python
# Use appropriate chunk sizes for large datasets
def chunk_signal_data(signal_data, chunk_size=10000):
    """Process large signals in chunks"""
    for i in range(0, len(signal_data), chunk_size):
        yield signal_data[i:i + chunk_size]

# Process large signal
large_signal = np.random.randn(100000) * 0.4
results = []

for chunk in chunk_signal_data(large_signal.tolist()):
    result = robust_eye_analysis(chunk, "USB4", 40e9)
    if result:
        results.append(result)

# Combine results as needed
avg_eye_height = sum(r['eye_height'] for r in results) / len(results)
print(f"Average eye height across chunks: {avg_eye_height:.4f}V")
```

This comprehensive guide should help you effectively use the SerDes Validation Framework REST API in your applications and workflows.