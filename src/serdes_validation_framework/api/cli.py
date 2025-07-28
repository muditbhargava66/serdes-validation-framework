"""
API Command Line Interface

CLI tool for interacting with the SerDes Validation Framework API.
"""

import json
import sys
import time
from typing import Any, Dict, Optional

import click
import numpy as np
import requests


class APIClient:
    """Client for interacting with the SerDes API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/v1"
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to API"""
        url = f"{self.api_url}{endpoint}"
        
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            click.echo(f"API request failed: {e}", err=True)
            sys.exit(1)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return self._make_request('GET', '/status')
    
    def analyze_eye_diagram(self, signal_data, sample_rate: float, protocol: str, lane: int = 0, show_mask: bool = True) -> Dict[str, Any]:
        """Analyze eye diagram"""
        data = {
            'signal_data': signal_data,
            'sample_rate': sample_rate,
            'protocol': protocol,
            'lane': lane,
            'show_mask': show_mask
        }
        return self._make_request('POST', '/analyze/eye-diagram', json=data)
    
    def analyze_waveform(self, signal_data, sample_rate: float, protocol: str, lane: str = "lane_0") -> Dict[str, Any]:
        """Analyze waveform"""
        data = {
            'signal_data': signal_data,
            'sample_rate': sample_rate,
            'protocol': protocol,
            'lane': lane
        }
        return self._make_request('POST', '/analyze/waveform', json=data)
    
    def start_stress_test(self, protocol: str, num_cycles: int = 100, cycle_duration: float = 1.0, enable_bert_hooks: bool = False) -> str:
        """Start stress test"""
        data = {
            'protocol': protocol,
            'num_cycles': num_cycles,
            'cycle_duration': cycle_duration,
            'enable_bert_hooks': enable_bert_hooks
        }
        result = self._make_request('POST', '/test/stress', json=data)
        return result['test_id']
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get test status"""
        return self._make_request('GET', f'/test/{test_id}/status')
    
    def get_stress_test_result(self, test_id: str) -> Dict[str, Any]:
        """Get stress test result"""
        return self._make_request('GET', f'/test/stress/{test_id}')
    
    def cancel_test(self, test_id: str) -> Dict[str, Any]:
        """Cancel test"""
        return self._make_request('POST', f'/test/{test_id}/cancel')
    
    def control_fixture(self, fixture_name: str, action: str, **params) -> Dict[str, Any]:
        """Control fixture"""
        data = {
            'fixture_name': fixture_name,
            'action': action,
            'parameters': params
        }
        return self._make_request('POST', '/fixture/control', json=data)
    
    def get_protocols(self) -> Dict[str, Any]:
        """Get available protocols"""
        return self._make_request('GET', '/protocols')


@click.group()
@click.option('--api-url', default='http://localhost:8000', help='API base URL')
@click.pass_context
def cli(ctx, api_url):
    """SerDes Validation Framework API CLI"""
    ctx.ensure_object(dict)
    ctx.obj['client'] = APIClient(api_url)


@cli.command()
@click.pass_context
def status(ctx):
    """Get system status"""
    client = ctx.obj['client']
    
    try:
        status = client.get_system_status()
        
        click.echo("🖥️  System Status")
        click.echo("=" * 50)
        click.echo(f"Version: {status['version']}")
        click.echo(f"Status: {status['status']}")
        click.echo(f"Active Tests: {status['active_tests']}")
        click.echo(f"Total Tests Run: {status['total_tests_run']}")
        click.echo(f"Uptime: {status['uptime_seconds']:.1f} seconds")
        click.echo(f"Memory Usage: {status['memory_usage_mb']:.1f} MB")
        click.echo(f"CPU Usage: {status['cpu_usage_percent']:.1f}%")
        
        click.echo("\n📡 Available Protocols:")
        for protocol in status['available_protocols']:
            click.echo(f"  • {protocol}")
        
        click.echo("\n🔧 Features:")
        for feature, available in status['features'].items():
            status_icon = "✅" if available else "❌"
            click.echo(f"  {status_icon} {feature}")
        
    except Exception as e:
        click.echo(f"Failed to get status: {e}", err=True)


@cli.command()
@click.pass_context
def protocols(ctx):
    """List available protocols"""
    client = ctx.obj['client']
    
    try:
        protocols = client.get_protocols()
        
        click.echo("📡 Available Protocols")
        click.echo("=" * 50)
        
        for name, spec in protocols['protocols'].items():
            click.echo(f"\n{name}:")
            click.echo(f"  Data Rate: {spec['data_rate_gbps']} Gbps")
            click.echo(f"  Voltage Swing: {spec['voltage_swing_mv']} mV")
            click.echo(f"  Eye Height Threshold: {spec['eye_height_threshold_mv']} mV")
            click.echo(f"  SNR Threshold: {spec['snr_threshold_db']} dB")
            click.echo(f"  Description: {spec['description']}")
        
    except Exception as e:
        click.echo(f"Failed to get protocols: {e}", err=True)


@cli.command()
@click.option('--protocol', default='USB4', help='Protocol to use')
@click.option('--samples', default=1000, help='Number of signal samples')
@click.option('--sample-rate', default=40e9, help='Sample rate in Hz')
@click.option('--lane', default=0, help='Lane to analyze')
@click.pass_context
def analyze_eye(ctx, protocol, samples, sample_rate, lane):
    """Analyze eye diagram with generated signal"""
    client = ctx.obj['client']
    
    # Generate test signal
    click.echo(f"🔍 Generating {samples} samples for {protocol} analysis...")
    signal_data = (np.random.randn(samples) * 0.4).tolist()
    
    try:
        result = client.analyze_eye_diagram(signal_data, sample_rate, protocol, lane)
        
        click.echo("\n👁️  Eye Diagram Analysis Results")
        click.echo("=" * 50)
        click.echo(f"Protocol: {result['protocol']}")
        click.echo(f"Lane: {result['lane']}")
        click.echo(f"Eye Height: {result['eye_height']:.4f} V")
        click.echo(f"Eye Width: {result['eye_width']:.4f} UI")
        click.echo(f"Eye Area: {result['eye_area']:.4f}")
        click.echo(f"Q-Factor: {result['q_factor']:.2f}")
        click.echo(f"SNR: {result['snr']:.2f} dB")
        
        status_icon = "✅" if result['passed'] else "❌"
        click.echo(f"Status: {status_icon} {'PASS' if result['passed'] else 'FAIL'}")
        
        if result.get('mask_analysis'):
            mask = result['mask_analysis']
            click.echo("\n🎭 Mask Analysis:")
            click.echo(f"  Compliance: {mask['compliance_level']}")
            click.echo(f"  Violations: {mask['violations']}")
            click.echo(f"  Margin: {mask['margin_percentage']:.1f}%")
        
    except Exception as e:
        click.echo(f"Analysis failed: {e}", err=True)


@cli.command()
@click.option('--protocol', default='USB4', help='Protocol to test')
@click.option('--cycles', default=10, help='Number of test cycles')
@click.option('--bert-hooks', is_flag=True, help='Enable BERT hooks')
@click.option('--wait', is_flag=True, help='Wait for test completion')
@click.pass_context
def stress_test(ctx, protocol, cycles, bert_hooks, wait):
    """Run stress test"""
    client = ctx.obj['client']
    
    try:
        click.echo(f"🚀 Starting {protocol} stress test with {cycles} cycles...")
        test_id = client.start_stress_test(protocol, cycles, bert_hooks)
        click.echo(f"Test ID: {test_id}")
        
        if wait:
            click.echo("⏳ Waiting for test completion...")
            
            while True:
                status = client.get_test_status(test_id)
                
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    break
                
                click.echo(f"Progress: {status['progress']:.1f}% - {status['message']}")
                time.sleep(2)
            
            if status['status'] == 'completed':
                result = client.get_stress_test_result(test_id)
                
                click.echo("\n📊 Stress Test Results")
                click.echo("=" * 50)
                click.echo(f"Protocol: {result['protocol']}")
                click.echo(f"Total Cycles: {result['total_cycles']}")
                click.echo(f"Passed: {result['passed_cycles']}")
                click.echo(f"Failed: {result['failed_cycles']}")
                click.echo(f"Success Rate: {result['success_rate']:.1%}")
                click.echo(f"Duration: {result['duration']:.1f} seconds")
                click.echo(f"Max Degradation: {result['max_degradation']:.1f}%")
                
                if result.get('initial_eye_height') and result.get('final_eye_height'):
                    click.echo(f"Initial Eye Height: {result['initial_eye_height']:.4f} V")
                    click.echo(f"Final Eye Height: {result['final_eye_height']:.4f} V")
            else:
                click.echo(f"❌ Test {status['status']}: {status['message']}")
        
    except Exception as e:
        click.echo(f"Stress test failed: {e}", err=True)


@cli.command()
@click.argument('test_id')
@click.pass_context
def test_status(ctx, test_id):
    """Get test status"""
    client = ctx.obj['client']
    
    try:
        status = client.get_test_status(test_id)
        
        click.echo(f"📋 Test Status: {test_id}")
        click.echo("=" * 50)
        click.echo(f"Status: {status['status']}")
        click.echo(f"Progress: {status['progress']:.1f}%")
        click.echo(f"Message: {status['message']}")
        click.echo(f"Started: {time.ctime(status['started_at'])}")
        
        if status.get('estimated_completion'):
            click.echo(f"Estimated Completion: {time.ctime(status['estimated_completion'])}")
        
    except Exception as e:
        click.echo(f"Failed to get test status: {e}", err=True)


@cli.command()
@click.argument('test_id')
@click.pass_context
def cancel(ctx, test_id):
    """Cancel a running test"""
    client = ctx.obj['client']
    
    try:
        result = client.cancel_test(test_id)
        click.echo(f"✅ {result['message']}")
        
    except Exception as e:
        click.echo(f"Failed to cancel test: {e}", err=True)


@cli.command()
@click.option('--fixture', required=True, help='Fixture name')
@click.option('--action', required=True, help='Action to perform')
@click.option('--voltage', type=float, help='Voltage to set')
@click.option('--frequency', type=float, help='Frequency to set')
@click.pass_context
def fixture(ctx, fixture, action, voltage, frequency):
    """Control test fixture"""
    client = ctx.obj['client']
    
    params = {}
    if voltage is not None:
        params['voltage'] = voltage
    if frequency is not None:
        params['frequency'] = frequency
    
    try:
        result = client.control_fixture(fixture, action, **params)
        
        click.echo(f"🔧 Fixture: {result['fixture_name']}")
        click.echo("=" * 50)
        click.echo(f"Status: {result['status']}")
        click.echo(f"Connected: {'Yes' if result['connected'] else 'No'}")
        click.echo(f"Temperature: {result['temperature']:.1f}°C")
        click.echo(f"Voltage: {result['voltage']:.2f} V")
        click.echo(f"Current: {result['current']:.3f} A")
        click.echo(f"Uptime: {result['uptime_seconds']:.1f} seconds")
        
        if result['error_message']:
            click.echo(f"Error: {result['error_message']}")
        
    except Exception as e:
        click.echo(f"Fixture control failed: {e}", err=True)


if __name__ == '__main__':
    cli()
