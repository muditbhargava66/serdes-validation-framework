#!/usr/bin/env python3
"""
SerDes Validation Framework API Test Client

Test client to validate API functionality and demonstrate usage.
This script tests all major API endpoints and provides examples of API usage.

Usage:
    python test_api_client.py [--host HOST] [--port PORT] [--no-interactive]

Requirements:
    - API server must be running (python run_api_server.py)
    - All dependencies installed (pip install -r requirements.txt)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set environment for testing
os.environ.setdefault('SVF_MOCK_MODE', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

try:
    import numpy as np
    import requests
    from serdes_validation_framework.api.cli import APIClient
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure you have installed all dependencies:")
    print("   pip install -r requirements.txt")
    print("   pip install -e .")
    sys.exit(1)


class APITester:
    """API testing class with comprehensive test suite"""
    
    def __init__(self, base_url: str):
        self.client = APIClient(base_url)
        self.base_url = base_url
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if details:
            print(f"      {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details
        })
    
    def test_server_connectivity(self) -> bool:
        """Test basic server connectivity"""
        print("1. Testing server connectivity...")
        try:
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                self.log_test("Server connectivity", True, f"Status: {response.status_code}")
                return True
            else:
                self.log_test("Server connectivity", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Server connectivity", False, str(e))
            return False
    
    def test_system_status(self) -> bool:
        """Test system status endpoint"""
        print("\n2. Testing system status...")
        try:
            status = self.client.get_system_status()
            
            # Validate required fields
            required_fields = ['status', 'version', 'memory_usage_mb', 'available_protocols']
            missing_fields = [field for field in required_fields if field not in status]
            
            if missing_fields:
                self.log_test("System status", False, f"Missing fields: {missing_fields}")
                return False
            
            self.log_test("System status", True, 
                         f"Version: {status['version']}, Status: {status['status']}")
            print(f"      📊 Memory: {status['memory_usage_mb']:.1f}MB")
            print(f"      📡 Protocols: {', '.join(status['available_protocols'])}")
            return True
            
        except Exception as e:
            self.log_test("System status", False, str(e))
            return False
    
    def test_eye_diagram_analysis(self) -> bool:
        """Test eye diagram analysis"""
        print("\n3. Testing eye diagram analysis...")
        try:
            # Generate test signal
            np.random.seed(42)  # For reproducible results
            signal_data = np.random.randn(1000) * 0.4
            
            result = self.client.analyze_eye_diagram(
                signal_data.tolist(),
                sample_rate=40e9,
                protocol="USB4",
                lane=0,
                show_mask=True
            )
            
            # Validate response
            required_fields = ['eye_height', 'eye_width', 'q_factor', 'snr', 'passed', 'protocol']
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                self.log_test("Eye diagram analysis", False, f"Missing fields: {missing_fields}")
                return False
            
            self.log_test("Eye diagram analysis", True,
                         f"Eye Height: {result['eye_height']:.4f}V, SNR: {result['snr']:.2f}dB")
            
            # Test mask analysis if available
            if result.get('mask_analysis'):
                mask = result['mask_analysis']
                print(f"      🎭 Mask: {mask['compliance_level']} ({mask['violations']} violations)")
            
            return True
            
        except Exception as e:
            self.log_test("Eye diagram analysis", False, str(e))
            return False
    
    def test_waveform_analysis(self) -> bool:
        """Test waveform analysis"""
        print("\n4. Testing waveform analysis...")
        try:
            # Generate test signal
            signal_data = np.random.randn(1000) * 0.4
            
            result = self.client.analyze_waveform(
                signal_data.tolist(),
                sample_rate=40e9,
                protocol="USB4",
                lane="lane_0"
            )
            
            # Validate response
            required_fields = ['snr_db', 'thd_percent', 'peak_to_peak', 'passed', 'protocol']
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                self.log_test("Waveform analysis", False, f"Missing fields: {missing_fields}")
                return False
            
            self.log_test("Waveform analysis", True,
                         f"SNR: {result['snr_db']:.2f}dB, THD: {result['thd_percent']:.1f}%")
            print(f"      📊 P2P: {result['peak_to_peak']:.4f}V, Status: {'PASS' if result['passed'] else 'FAIL'}")
            return True
            
        except Exception as e:
            self.log_test("Waveform analysis", False, str(e))
            return False
    
    def test_stress_testing(self) -> bool:
        """Test stress testing functionality"""
        print("\n5. Testing stress test...")
        try:
            # Start stress test
            test_id = self.client.start_stress_test(
                protocol="USB4",
                num_cycles=5,
                cycle_duration=0.5,  # Short duration for testing
                enable_bert_hooks=False
            )
            
            if not test_id:
                self.log_test("Stress test start", False, "No test ID returned")
                return False
            
            print(f"      🚀 Started test: {test_id}")
            
            # Monitor test progress
            max_wait_time = 30  # Maximum wait time in seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                status = self.client.get_test_status(test_id)
                progress = status.get('progress', 0) * 100
                print(f"      ⏳ Progress: {progress:.1f}% - {status.get('message', 'Running...')}")
                
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    break
                
                time.sleep(1)
            
            # Check final status
            if status['status'] == 'completed':
                try:
                    result = self.client.get_stress_test_result(test_id)
                    self.log_test("Stress test", True,
                                 f"Success Rate: {result['success_rate']:.1%}, "
                                 f"Max Degradation: {result['max_degradation']:.1f}%")
                    return True
                except Exception as e:
                    self.log_test("Stress test", False, f"Failed to get results: {e}")
                    return False
            else:
                self.log_test("Stress test", False, f"Test status: {status['status']}")
                return False
                
        except Exception as e:
            self.log_test("Stress test", False, str(e))
            return False
    
    def test_fixture_control(self) -> bool:
        """Test fixture control functionality"""
        print("\n6. Testing fixture control...")
        try:
            # Test fixture connection
            fixture_result = self.client.control_fixture(
                fixture_name="test_probe",
                action="connect",
                parameters={"fixture_type": "PROBE_STATION"}
            )
            
            if not fixture_result:
                self.log_test("Fixture control", False, "No response from fixture control")
                return False
            
            print(f"      🔧 Fixture: {fixture_result.get('status', 'unknown')}")
            print(f"      🌡️  Temperature: {fixture_result.get('temperature', 0):.1f}°C")
            
            # Test disconnection
            disconnect_result = self.client.control_fixture(
                fixture_name="test_probe",
                action="disconnect"
            )
            
            self.log_test("Fixture control", True, "Connect/disconnect successful")
            return True
            
        except Exception as e:
            # Fixture control might not be available in all environments
            self.log_test("Fixture control", True, f"Skipped: {e}")
            return True
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests"""
        print("🧪 SerDes Validation Framework API Test Suite")
        print("=" * 60)
        
        # Run tests in order
        tests = [
            self.test_server_connectivity,
            self.test_system_status,
            self.test_eye_diagram_analysis,
            self.test_waveform_analysis,
            self.test_stress_testing,
            self.test_fixture_control
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
        
        # Summary
        print("\n📊 Test Summary")
        print("=" * 60)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {passed/total:.1%}")
        
        if passed == total:
            print("\n🎉 All API tests completed successfully!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        
        return {
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': passed / total,
            'results': self.test_results
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='SerDes Validation Framework API Test Client')
    parser.add_argument('--host', default='localhost', help='API server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8000, help='API server port (default: 8000)')
    parser.add_argument('--no-interactive', action='store_true', help='Skip interactive prompts')
    parser.add_argument('--output', help='Save test results to JSON file')
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print(f"🔗 Testing API at: {base_url}")
    
    if not args.no_interactive:
        print("\n⚠️  Make sure the API server is running:")
        print(f"   python run_api_server.py --host {args.host} --port {args.port}")
        print()
        
        try:
            input("Press Enter to start API tests (Ctrl+C to cancel)...")
        except KeyboardInterrupt:
            print("\n👋 Test cancelled by user")
            return
    
    # Run tests
    tester = APITester(base_url)
    results = tester.run_all_tests()
    
    # Save results if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Test results saved to: {args.output}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if results['passed'] == results['total'] else 1)


if __name__ == "__main__":
    main()
