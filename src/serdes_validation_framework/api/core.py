"""
API Core Implementation

Core API logic that wraps the SerDes Validation Framework functionality.
"""

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import numpy as np
import psutil

from .. import __version__
from ..data_analysis.eye_masks import create_eye_mask_analyzer
from ..jupyter_dashboard import DashboardConfig, WaveformAnalyzer, create_dashboard
from ..stress_testing import LoopbackStressTest, create_stress_test_config
from ..test_fixtures import FixtureType, create_environmental_monitor, create_fixture_controller
from .models import (
    EyeDiagramRequest,
    EyeDiagramResponse,
    FixtureControlRequest,
    FixtureStatusResponse,
    ProtocolType,
    StatusType,
    StressTestRequest,
    StressTestResponse,
    SystemStatusResponse,
    TestStatusResponse,
    WaveformAnalysisRequest,
    WaveformAnalysisResponse,
)

logger = logging.getLogger(__name__)


class TestManager:
    """Manages running tests and their status"""
    
    def __init__(self):
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.completed_tests: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        self.total_tests_run = 0
        self.start_time = time.time()
    
    def create_test(self, test_type: str, **kwargs) -> str:
        """Create a new test and return test ID"""
        test_id = f"{test_type}_{uuid.uuid4().hex[:8]}"
        
        with self.lock:
            self.active_tests[test_id] = {
                'id': test_id,
                'type': test_type,
                'status': StatusType.PENDING,
                'progress': 0.0,
                'message': 'Test queued',
                'started_at': time.time(),
                'kwargs': kwargs,
                'future': None
            }
        
        return test_id
    
    def start_test(self, test_id: str, test_func, *args, **kwargs):
        """Start executing a test"""
        with self.lock:
            if test_id in self.active_tests:
                future = self.executor.submit(self._run_test, test_id, test_func, *args, **kwargs)
                self.active_tests[test_id]['future'] = future
                self.active_tests[test_id]['status'] = StatusType.RUNNING
                self.active_tests[test_id]['message'] = 'Test started'
    
    def _run_test(self, test_id: str, test_func, *args, **kwargs):
        """Internal test runner"""
        try:
            result = test_func(*args, **kwargs)
            
            with self.lock:
                if test_id in self.active_tests:
                    test_info = self.active_tests.pop(test_id)
                    test_info['status'] = StatusType.COMPLETED
                    test_info['progress'] = 100.0
                    test_info['message'] = 'Test completed successfully'
                    test_info['result'] = result
                    test_info['completed_at'] = time.time()
                    self.completed_tests[test_id] = test_info
                    self.total_tests_run += 1
        
        except Exception as e:
            logger.error(f"Test {test_id} failed: {e}")
            
            with self.lock:
                if test_id in self.active_tests:
                    test_info = self.active_tests.pop(test_id)
                    test_info['status'] = StatusType.FAILED
                    test_info['message'] = f'Test failed: {str(e)}'
                    test_info['error'] = str(e)
                    test_info['completed_at'] = time.time()
                    self.completed_tests[test_id] = test_info
    
    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get test status"""
        with self.lock:
            if test_id in self.active_tests:
                return self.active_tests[test_id].copy()
            elif test_id in self.completed_tests:
                return self.completed_tests[test_id].copy()
        return None
    
    def cancel_test(self, test_id: str) -> bool:
        """Cancel a running test"""
        with self.lock:
            if test_id in self.active_tests:
                test_info = self.active_tests[test_id]
                if test_info['future']:
                    test_info['future'].cancel()
                test_info['status'] = StatusType.CANCELLED
                test_info['message'] = 'Test cancelled by user'
                return True
        return False
    
    def get_active_test_count(self) -> int:
        """Get number of active tests"""
        with self.lock:
            return len(self.active_tests)


class SerDesAPI:
    """Main API class that wraps SerDes Validation Framework"""
    
    def __init__(self):
        self.test_manager = TestManager()
        self.fixtures: Dict[str, Any] = {}
        self.environmental_monitors: Dict[str, Any] = {}
        
        logger.info("SerDes API initialized")
    
    # Eye Diagram Analysis
    def analyze_eye_diagram(self, request: EyeDiagramRequest) -> EyeDiagramResponse:
        """Analyze eye diagram from signal data"""
        try:
            # Convert signal data to numpy array if needed
            if isinstance(request.signal_data, list):
                signal_data = np.array(request.signal_data)
            elif isinstance(request.signal_data, dict):
                # Handle multi-lane data
                signal_data = {k: np.array(v) if isinstance(v, list) else v 
                              for k, v in request.signal_data.items()}
            else:
                signal_data = request.signal_data
            
            # Create dashboard
            config = DashboardConfig(show_mask=request.show_mask)
            dashboard = create_dashboard(
                signal_data=signal_data,
                sample_rate=request.sample_rate,
                protocol=request.protocol.value,
                config=config
            )
            
            # Set time data if provided
            if request.time_data:
                dashboard.time_data = np.array(request.time_data)
            
            # Analyze eye diagram
            results = dashboard.analyze_eye_diagram(lane=request.lane)
            
            # Convert to response model
            response = EyeDiagramResponse(
                eye_height=results['eye_height'],
                eye_width=results['eye_width'],
                eye_area=results['eye_area'],
                q_factor=results['q_factor'],
                snr=results['snr'],
                passed=results['passed'],
                lane=str(results['lane']),
                protocol=results['protocol'],
                mask_analysis=results.get('mask_analysis'),
                timestamp=time.time()
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Eye diagram analysis failed: {e}")
            raise
    
    # Stress Testing
    def start_stress_test(self, request: StressTestRequest) -> str:
        """Start a stress test (async)"""
        test_id = self.test_manager.create_test('stress_test', request=request)
        self.test_manager.start_test(test_id, self._run_stress_test, request)
        return test_id
    
    def _run_stress_test(self, request: StressTestRequest) -> StressTestResponse:
        """Internal stress test runner"""
        # Create configuration
        config = create_stress_test_config(
            protocol=request.protocol.value,
            num_cycles=request.num_cycles,
            enable_bert_hooks=request.enable_bert_hooks
        )
        
        # Override specific parameters if provided
        if request.data_rate:
            config.data_rate = request.data_rate
        if request.voltage_swing:
            config.voltage_swing = request.voltage_swing
        
        config.cycle_duration = request.cycle_duration
        
        # Run stress test
        stress_test = LoopbackStressTest(config)
        results = stress_test.run_stress_test()
        
        # Convert to response
        response = StressTestResponse(
            test_id="",  # Will be set by caller
            status=StatusType.COMPLETED,
            protocol=results.config.protocol,
            total_cycles=results.total_cycles,
            passed_cycles=results.passed_cycles,
            failed_cycles=results.failed_cycles,
            success_rate=results.success_rate,
            duration=results.duration,
            max_degradation=results.max_degradation,
            initial_eye_height=results.initial_eye_height,
            final_eye_height=results.final_eye_height,
            timestamp=time.time()
        )
        
        return response
    
    def get_stress_test_result(self, test_id: str) -> Optional[StressTestResponse]:
        """Get stress test result"""
        test_info = self.test_manager.get_test_status(test_id)
        if test_info and 'result' in test_info:
            result = test_info['result']
            result.test_id = test_id
            return result
        return None
    
    # Waveform Analysis
    def analyze_waveform(self, request: WaveformAnalysisRequest) -> WaveformAnalysisResponse:
        """Analyze waveform signal quality"""
        try:
            # Create analyzer
            analyzer = WaveformAnalyzer(
                sample_rate=request.sample_rate,
                protocol=request.protocol.value
            )
            
            # Get signal data for specified lane
            if isinstance(request.signal_data, dict):
                if request.lane not in request.signal_data:
                    raise ValueError(f"Lane {request.lane} not found in signal data")
                voltage_data = np.array(request.signal_data[request.lane])
            else:
                voltage_data = np.array(request.signal_data)
            
            # Generate time data if not provided
            if request.time_data:
                time_data = np.array(request.time_data)
            else:
                time_data = np.linspace(0, len(voltage_data) / request.sample_rate, len(voltage_data))
            
            # Analyze waveform
            result = analyzer.analyze_waveform(
                voltage_data=voltage_data,
                time_data=time_data,
                lane=request.lane
            )
            
            # Convert to response
            response = WaveformAnalysisResponse(
                lane=result.lane,
                protocol=result.protocol,
                mean_voltage=result.mean_voltage,
                rms_voltage=result.rms_voltage,
                peak_to_peak=result.peak_to_peak,
                snr_db=result.snr_db,
                thd_percent=result.thd_percent,
                dynamic_range=result.dynamic_range,
                passed=result.passed,
                failure_reasons=result.failure_reasons,
                timestamp=time.time()
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Waveform analysis failed: {e}")
            raise
    
    # Fixture Control
    def control_fixture(self, request: FixtureControlRequest) -> FixtureStatusResponse:
        """Control test fixture"""
        try:
            fixture_name = request.fixture_name
            action = request.action
            params = request.parameters
            
            # Get or create fixture
            if fixture_name not in self.fixtures:
                if action == 'connect':
                    # Create new fixture
                    fixture_type = FixtureType.PROBE_STATION  # Default
                    if 'fixture_type' in params:
                        fixture_type = FixtureType[params['fixture_type'].upper()]
                    
                    self.fixtures[fixture_name] = create_fixture_controller(
                        fixture_type, fixture_name
                    )
                else:
                    raise ValueError(f"Fixture {fixture_name} not found. Use 'connect' action first.")
            
            fixture = self.fixtures[fixture_name]
            
            # Perform action
            if action == 'connect':
                success = fixture.connect()
                if not success:
                    raise RuntimeError("Failed to connect to fixture")
            
            elif action == 'disconnect':
                success = fixture.disconnect()
                if not success:
                    raise RuntimeError("Failed to disconnect fixture")
            
            elif action == 'set_voltage':
                voltage = params.get('voltage', 0.0)
                success = fixture.set_voltage(voltage)
                if not success:
                    raise RuntimeError(f"Failed to set voltage to {voltage}V")
            
            elif action == 'set_frequency':
                frequency = params.get('frequency', 1e6)
                success = fixture.set_frequency(frequency)
                if not success:
                    raise RuntimeError(f"Failed to set frequency to {frequency}Hz")
            
            elif action == 'calibrate':
                success = fixture.calibrate()
                if not success:
                    raise RuntimeError("Calibration failed")
            
            # Get current status
            status = fixture.get_status()
            
            response = FixtureStatusResponse(
                fixture_name=status.fixture_name,
                status=status.status.name,
                temperature=status.temperature,
                voltage=status.voltage,
                current=status.current,
                connected=fixture.connected,
                last_calibration=status.last_calibration,
                uptime_seconds=status.uptime_seconds,
                error_message=status.error_message,
                timestamp=status.timestamp
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Fixture control failed: {e}")
            raise
    
    # Test Status
    def get_test_status(self, test_id: str) -> Optional[TestStatusResponse]:
        """Get test status"""
        test_info = self.test_manager.get_test_status(test_id)
        if test_info:
            # Calculate estimated completion
            estimated_completion = None
            if test_info['status'] == StatusType.RUNNING and test_info['progress'] > 0:
                elapsed = time.time() - test_info['started_at']
                total_estimated = elapsed / (test_info['progress'] / 100.0)
                estimated_completion = test_info['started_at'] + total_estimated
            
            return TestStatusResponse(
                test_id=test_info['id'],
                status=test_info['status'],
                progress=test_info['progress'],
                message=test_info['message'],
                started_at=test_info['started_at'],
                estimated_completion=estimated_completion
            )
        return None
    
    def cancel_test(self, test_id: str) -> bool:
        """Cancel a running test"""
        return self.test_manager.cancel_test(test_id)
    
    # System Status
    def get_system_status(self) -> SystemStatusResponse:
        """Get system status"""
        try:
            # Get system metrics
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            # Check available features
            features = {
                'eye_analysis': True,
                'stress_testing': True,
                'bert_hooks': True,
                'fixture_control': True,
                'environmental_monitoring': True,
                'mask_compliance': True
            }
            
            response = SystemStatusResponse(
                version=__version__,
                status="healthy",
                active_tests=self.test_manager.get_active_test_count(),
                total_tests_run=self.test_manager.total_tests_run,
                uptime_seconds=time.time() - self.test_manager.start_time,
                memory_usage_mb=memory_info.rss / 1024 / 1024,
                cpu_usage_percent=cpu_percent,
                available_protocols=["USB4", "PCIe", "Ethernet"],
                features=features,
                timestamp=time.time()
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            raise
    
    # Mask Compliance
    def check_mask_compliance(self, request: EyeDiagramRequest) -> Dict[str, Any]:
        """Check eye mask compliance"""
        try:
            # Create mask analyzer
            data_rate = {
                ProtocolType.USB4: 20.0,
                ProtocolType.PCIE: 32.0,
                ProtocolType.ETHERNET: 112.0
            }.get(request.protocol, 20.0)
            
            mask_analyzer = create_eye_mask_analyzer(
                request.protocol.value, data_rate
            )
            
            # Create dummy eye diagram for analysis
            # In real implementation, this would come from actual eye diagram data
            if isinstance(request.signal_data, dict):
                signal = np.array(list(request.signal_data.values())[0])
            else:
                signal = np.array(request.signal_data)
            
            # Create simple eye diagram representation
            eye_data = np.outer(signal[:100], np.ones(100))
            time_axis = np.linspace(-0.5, 0.5, 100)
            voltage_axis = np.linspace(np.min(signal), np.max(signal), 100) * 1000  # Convert to mV
            
            # Analyze against mask
            result = mask_analyzer.analyze_eye_against_mask(
                eye_data, time_axis, voltage_axis
            )
            
            return {
                'protocol': result.protocol,
                'mask_passed': result.mask_passed,
                'violations': len(result.violations),
                'margin_percentage': result.margin_percentage,
                'compliance_level': result.compliance_level,
                'eye_opening_percentage': result.eye_opening_percentage,
                'violation_details': [
                    {
                        'point_index': v.point_index,
                        'measured_voltage': v.measured_voltage,
                        'mask_voltage': v.mask_voltage,
                        'violation_margin': v.violation_margin,
                        'time_ui': v.time_ui,
                        'severity': v.severity
                    } for v in result.violations
                ],
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Mask compliance check failed: {e}")
            raise
