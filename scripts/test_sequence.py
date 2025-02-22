"""
Test Sequence Script

This script executes a test sequence with support for both real and mock hardware.
It demonstrates basic instrument control, data collection, and analysis.
"""

import sys
import os
import logging
from typing import Dict, List, Any
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.test_sequence.sequencer import TestSequencer
from src.serdes_validation_framework.instrument_control.mock_controller import get_instrument_controller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_test_sequence(instruments: List[str]) -> Dict[str, Any]:
    """
    Execute test sequence with validation
    
    Args:
        instruments: List of instrument resource names
        
    Returns:
        Dictionary containing test results and status
    """
    sequencer = None
    try:
        logger.info("Initializing test sequence...")
        # Initialize controller and sequencer
        controller = get_instrument_controller()
        sequencer = TestSequencer(controller=controller)
        
        # Set up instruments
        logger.info(f"Setting up instruments: {instruments}")
        sequencer.setup_instruments(instruments)
        
        # Define test sequence
        sequence = [
            {'resource': instruments[0], 'command': '*RST', 'action': 'send'},
            {'resource': instruments[1], 'command': '*IDN?', 'action': 'query'},
            {'resource': instruments[1], 'command': 'MEASure:VOLTage:DC?', 'action': 'query'}
        ]
        
        # Run sequence
        logger.info("Executing test sequence...")
        sequence_results = sequencer.run_sequence(sequence)
        
        # Collect measurements
        logger.info("Collecting measurements...")
        voltage_data = sequencer.collect_and_analyze_data(
            instruments[1],
            'MEASure:VOLTage:DC?',
            'voltage'
        )
        
        results = {
            'status': 'success',
            'mode': 'mock' if hasattr(controller, 'mock_responses') else 'real',
            'sequence_results': sequence_results,
            'measurements': {
                'voltage': voltage_data
            }
        }
        
        logger.info("Test sequence completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Test sequence failed: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }
    finally:
        if sequencer is not None:
            try:
                logger.info("Cleaning up instruments...")
                sequencer.cleanup(instruments)
            except Exception as cleanup_error:
                logger.warning(f"Cleanup warning: {cleanup_error}")

def display_results(results: Dict[str, Any]) -> None:
    """
    Display formatted test results
    
    Args:
        results: Dictionary containing test results
    """
    print("\nTest Sequence Results:")
    print("-" * 50)
    print(f"Status: {results['status']}")
    
    if results['status'] == 'success':
        print(f"Mode: {results['mode']}")
        
        print("\nSequence Results:")
        for cmd, response in results.get('sequence_results', {}).items():
            print(f"  {cmd}: {response}")
        
        print("\nMeasurements:")
        if 'measurements' in results:
            for name, value in results['measurements'].items():
                if isinstance(value, dict):
                    print(f"  {name}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {name}: {value}")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")

def main() -> None:
    """Main function with enhanced error handling"""
    try:
        # Define test instruments
        instruments = ['GPIB::1::INSTR', 'GPIB::2::INSTR']
        
        # Run test sequence
        logger.info("Starting test sequence execution...")
        results = run_test_sequence(instruments)
        
        # Display results
        display_results(results)
        
        # Set exit code based on results
        if results['status'] != 'success':
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()