"""
Instrument Control Script

This script demonstrates basic instrument control functionality with support
for both real and mock hardware modes.
"""

import sys
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.instrument_control.controller import InstrumentController
from src.serdes_validation_framework.instrument_control.mock_controller import (
    get_instrument_controller,
    MockInstrumentController
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def perform_instrument_test(resource_name: str) -> Dict[str, Any]:
    """
    Perform basic instrument control test
    
    Args:
        resource_name: Instrument resource identifier
        
    Returns:
        Dictionary containing test results and status
    """
    controller = None
    try:
        # Validate input
        if not isinstance(resource_name, str) or not resource_name:
            raise ValueError("Invalid resource name")
            
        # Initialize controller
        controller = get_instrument_controller()
        
        # Connect to instrument
        controller.connect_instrument(resource_name)
        
        # Perform basic operations
        controller.send_command(resource_name, '*RST')
        id_response = controller.query_instrument(resource_name, '*IDN?')
        
        # Additional test measurements
        measurements = {}
        for cmd in ['*TST?', '*OPC?', 'MEASure:VOLTage:DC?']:
            try:
                measurements[cmd] = controller.query_instrument(resource_name, cmd)
            except Exception as e:
                logger.warning(f"Measurement {cmd} failed: {e}")
                measurements[cmd] = f"Error: {str(e)}"
        
        return {
            'status': 'success',
            'mode': 'mock' if isinstance(controller, MockInstrumentController) else 'real',
            'id': id_response,
            'measurements': measurements
        }
        
    except Exception as e:
        logger.error(f"Instrument test failed: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }
    finally:
        if controller is not None:
            try:
                controller.disconnect_instrument(resource_name)
            except Exception as cleanup_error:
                logger.warning(f"Cleanup warning: {cleanup_error}")

def main() -> None:
    """Main function with enhanced error handling"""
    try:
        # Run instrument test
        results = perform_instrument_test('GPIB::2::INSTR')
        
        # Display results
        print("\nInstrument Control Results:")
        print("-" * 50)
        print(f"Status: {results['status']}")
        
        if results['status'] == 'success':
            print(f"Mode: {results['mode']}")
            print(f"Instrument ID: {results['id']}")
            
            print("\nMeasurements:")
            for cmd, response in results['measurements'].items():
                print(f"  {cmd}: {response}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()