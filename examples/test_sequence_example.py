#!/usr/bin/env python3
"""
Test Sequence Example Script

This script demonstrates comprehensive test sequence capabilities including:
- Automated instrument control and data collection
- Mock and real hardware support
- PCIe test sequences with multi-lane support
- Advanced test workflows with error handling
- Performance monitoring and reporting

The example showcases both basic and advanced test sequence patterns
suitable for production SerDes validation.

Dependencies:
    - logging
    - unittest.mock
    - numpy
    - serdes_validation_framework
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from serdes_validation_framework.instrument_control import get_instrument_controller
from serdes_validation_framework.test_sequence.sequencer import TestSequencer

# Try to import PCIe modules
try:
    from serdes_validation_framework.protocols.pcie.constants import SignalMode
    from serdes_validation_framework.test_sequence.pcie_sequence import create_multi_lane_pam4_test, create_single_lane_nrz_test
    PCIE_AVAILABLE = True
except ImportError:
    PCIE_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_basic_sequence():
    """Demonstrate basic test sequence functionality"""
    logger.info("=== Basic Test Sequence Demo ===")
    
    @patch('pyvisa.ResourceManager')
    def run_basic_test(mock_rm):
        try:
            # Setup mock instrument
            mock_instrument = MagicMock()
            mock_rm.return_value.open_resource.return_value = mock_instrument
            
            # Create test sequencer
            ts = TestSequencer()
            instruments = ['GPIB::1::INSTR', 'GPIB::2::INSTR']
            ts.setup_instruments(instruments)
            
            # Define test sequence
            sequence = [
                {'resource': 'GPIB::1::INSTR', 'command': '*RST', 'action': 'send'},
                {'resource': 'GPIB::1::INSTR', 'command': '*IDN?', 'action': 'query'},
                {'resource': 'GPIB::2::INSTR', 'command': '*RST', 'action': 'send'},
                {'resource': 'GPIB::2::INSTR', 'command': '*IDN?', 'action': 'query'}
            ]
            
            # Configure mock responses
            mock_instrument.query.return_value = 'Mock Instrument v1.0'
            
            # Run sequence
            results = ts.run_sequence(sequence)
            logger.info(f"Sequence results: {results}")
            
            # Simulate data collection
            ts.data_collector.instruments['GPIB::2::INSTR'] = mock_instrument
            mock_instrument.query.return_value = '0.1 0.2 0.3 0.4 0.5'
            
            stats = ts.collect_and_analyze_data(
                'GPIB::2::INSTR', 
                'MEASure:VOLTage:DC?', 
                'voltage'
            )
            logger.info(f"Data statistics: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Basic sequence failed: {e}")
            return False
        finally:
            try:
                ts.cleanup(instruments)
            except Exception as cleanup_error:
                logger.error(f"Cleanup error: {cleanup_error}")
    
    return run_basic_test()


def demonstrate_advanced_sequence():
    """Demonstrate advanced test sequence with error handling"""
    logger.info("=== Advanced Test Sequence Demo ===")
    
    try:
        # Use the framework's intelligent controller
        controller = get_instrument_controller()
        logger.info(f"Using controller in {controller.get_mode()} mode")
        
        # Define advanced test sequence
        test_resources = ['GPIB::1::INSTR', 'GPIB::2::INSTR', 'GPIB::3::INSTR']
        
        # Connect to instruments
        for resource in test_resources:
            try:
                controller.connect_instrument(resource)
                logger.info(f"Connected to {resource}")
            except Exception as e:
                logger.warning(f"Failed to connect to {resource}: {e}")
        
        # Perform instrument identification
        instrument_info = {}
        for resource in test_resources:
            try:
                idn = controller.query_instrument(resource, '*IDN?')
                instrument_info[resource] = idn
                logger.info(f"{resource}: {idn}")
            except Exception as e:
                logger.warning(f"Failed to query {resource}: {e}")
        
        # Perform measurement sequence
        measurements = {}
        measurement_commands = {
            'voltage': 'MEASure:VOLTage:DC?',
            'frequency': 'MEASure:FREQuency?',
            'power': 'MEASure:POWer?'
        }
        
        for resource in test_resources:
            measurements[resource] = {}
            for param, command in measurement_commands.items():
                try:
                    result = controller.query_instrument(resource, command)
                    measurements[resource][param] = result
                    logger.info(f"{resource} {param}: {result}")
                except Exception as e:
                    logger.warning(f"Measurement {param} failed for {resource}: {e}")
                    measurements[resource][param] = f"Error: {str(e)}"
        
        # Cleanup
        for resource in test_resources:
            try:
                controller.disconnect_instrument(resource)
            except Exception as e:
                logger.warning(f"Disconnect failed for {resource}: {e}")
        
        return measurements
        
    except Exception as e:
        logger.error(f"Advanced sequence failed: {e}")
        return None


def demonstrate_pcie_sequence():
    """Demonstrate PCIe test sequence (if available)"""
    if not PCIE_AVAILABLE:
        logger.warning("PCIe modules not available, skipping PCIe sequence demo")
        return
    
    logger.info("=== PCIe Test Sequence Demo ===")
    
    try:
        # Generate mock PCIe signal data
        def generate_pcie_signal(mode, num_samples=5000):
            time = np.linspace(0, num_samples/100e9, num_samples, dtype=np.float64)
            
            if mode == SignalMode.NRZ:
                data = np.random.choice([-1.0, 1.0], size=num_samples)
            else:  # PAM4
                levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
                data = np.random.choice(levels, size=num_samples)
            
            # Add noise
            noise = np.random.normal(0, 0.1, num_samples)
            voltage = data + noise
            
            return {'time': time, 'voltage': voltage}
        
        # Single lane NRZ test
        logger.info("--- Single Lane NRZ Test ---")
        nrz_test = create_single_lane_nrz_test(
            lane_id=0,
            sample_rate=100e9,
            bandwidth=50e9
        )
        
        nrz_signal_data = {0: generate_pcie_signal(SignalMode.NRZ)}
        nrz_result = nrz_test.run_complete_sequence(nrz_signal_data)
        
        logger.info("NRZ Test Results:")
        logger.info(f"  Overall Status: {nrz_result.overall_status.name}")
        logger.info(f"  Duration: {nrz_result.total_duration:.2f} seconds")
        logger.info(f"  Phases Completed: {len(nrz_result.phase_results)}")
        
        # Multi-lane PAM4 test
        logger.info("--- Multi-Lane PAM4 Test ---")
        pam4_test = create_multi_lane_pam4_test(
            num_lanes=2,  # Reduced for demo
            sample_rate=200e9,
            bandwidth=100e9
        )
        
        pam4_signal_data = {
            0: generate_pcie_signal(SignalMode.PAM4),
            1: generate_pcie_signal(SignalMode.PAM4)
        }
        
        pam4_result = pam4_test.run_complete_sequence(pam4_signal_data)
        
        logger.info("PAM4 Test Results:")
        logger.info(f"  Overall Status: {pam4_result.overall_status.name}")
        logger.info(f"  Duration: {pam4_result.total_duration:.2f} seconds")
        logger.info(f"  Lanes Tested: {len(pam4_result.lane_results)}")
        
        # Lane-specific results
        for lane_id, metrics in pam4_result.lane_results.items():
            score = metrics.get('performance_score', 0)
            logger.info(f"  Lane {lane_id} Performance: {score:.1f}")
        
        return True
        
    except Exception as e:
        logger.error(f"PCIe sequence demo failed: {e}")
        return False


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities"""
    logger.info("=== Performance Monitoring Demo ===")
    
    try:
        import time
        
        # Simulate performance monitoring
        test_metrics = {
            'setup_time': 0.0,
            'execution_time': 0.0,
            'cleanup_time': 0.0,
            'total_measurements': 0,
            'successful_measurements': 0,
            'error_count': 0
        }
        
        # Setup phase
        start_time = time.time()
        controller = get_instrument_controller()
        test_metrics['setup_time'] = time.time() - start_time
        
        # Execution phase
        start_time = time.time()
        test_resources = ['GPIB::1::INSTR', 'GPIB::2::INSTR']
        
        for resource in test_resources:
            for i in range(5):  # 5 measurements per instrument
                test_metrics['total_measurements'] += 1
                try:
                    # Simulate measurement
                    controller.connect_instrument(resource)
                    result = controller.query_instrument(resource, f'MEASure:TEST{i}?')
                    test_metrics['successful_measurements'] += 1
                    
                    # Simulate processing time
                    time.sleep(0.01)
                    
                except Exception as e:
                    test_metrics['error_count'] += 1
                    logger.warning(f"Measurement failed: {e}")
                finally:
                    try:
                        controller.disconnect_instrument(resource)
                    except Exception:
                        pass
        
        test_metrics['execution_time'] = time.time() - start_time
        
        # Cleanup phase
        start_time = time.time()
        # Cleanup operations would go here
        test_metrics['cleanup_time'] = time.time() - start_time
        
        # Calculate performance metrics
        total_time = (test_metrics['setup_time'] + 
                     test_metrics['execution_time'] + 
                     test_metrics['cleanup_time'])
        
        success_rate = (test_metrics['successful_measurements'] / 
                       test_metrics['total_measurements'] * 100)
        
        throughput = test_metrics['total_measurements'] / total_time
        
        # Report performance
        logger.info("Performance Metrics:")
        logger.info(f"  Setup Time: {test_metrics['setup_time']:.3f} seconds")
        logger.info(f"  Execution Time: {test_metrics['execution_time']:.3f} seconds")
        logger.info(f"  Cleanup Time: {test_metrics['cleanup_time']:.3f} seconds")
        logger.info(f"  Total Time: {total_time:.3f} seconds")
        logger.info(f"  Success Rate: {success_rate:.1f}%")
        logger.info(f"  Throughput: {throughput:.1f} measurements/second")
        logger.info(f"  Error Count: {test_metrics['error_count']}")
        
        return test_metrics
        
    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")
        return None


def main():
    """Main function demonstrating various test sequence capabilities"""
    logger.info("SerDes Test Sequence Example - v1.4.0")
    logger.info("=" * 50)
    
    results = {
        'basic_sequence': False,
        'advanced_sequence': False,
        'pcie_sequence': False,
        'performance_monitoring': False
    }
    
    try:
        # Run demonstrations
        results['basic_sequence'] = demonstrate_basic_sequence()
        
        advanced_result = demonstrate_advanced_sequence()
        results['advanced_sequence'] = advanced_result is not None
        
        results['pcie_sequence'] = demonstrate_pcie_sequence()
        
        perf_result = demonstrate_performance_monitoring()
        results['performance_monitoring'] = perf_result is not None
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("Test Sequence Demonstration Summary:")
        for test_name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"  {test_name}: {status}")
        
        overall_success = all(results.values())
        logger.info(f"\nOverall Status: {'SUCCESS' if overall_success else 'PARTIAL'}")
        
        if not overall_success:
            logger.warning("Some demonstrations failed - check logs for details")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
