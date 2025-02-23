"""
Data Collection Script

This script demonstrates basic data collection functionality with support for both
real and mock hardware modes.
"""

import logging
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.data_collection.data_collector import DataCollector
from src.serdes_validation_framework.instrument_control.mock_controller import get_instrument_controller

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collect_measurements(collector: DataCollector, address: str) -> Dict[str, Any]:
    """
    Collect a set of measurements from an instrument
    
    Args:
        collector: Configured DataCollector instance
        address: Instrument address
        
    Returns:
        Dictionary containing measurement results
    """
    measurements = {
        'ID Query': '*IDN?',
        'DC Voltage': 'MEASure:VOLTage:DC?',
        'Eye Height': ':MEASure:EYE:HEIGht?',
        'Eye Width': ':MEASure:EYE:WIDTh?'
    }

    results = {}
    for name, command in measurements.items():
        try:
            response = collector.collect_data(address, command)
            results[name] = response
            logger.info(f"{name}: {response}")
        except Exception as e:
            logger.error(f"Failed to collect {name}: {e}")
            results[name] = f"Error: {str(e)}"

    return results

def main() -> None:
    """Main function with enhanced error handling"""
    collector = None
    address = 'GPIB::1::INSTR'

    try:
        # Initialize controller and collector
        controller = get_instrument_controller()
        collector = DataCollector(controller)

        # Connect to instrument
        collector.connect_instrument(address)

        # Collect measurements
        results = collect_measurements(collector, address)

        # Display results
        print("\nData Collection Results:")
        print("-" * 50)
        print(f"Mode: {'mock' if 'Mock' in results.get('ID Query', '') else 'real'}")

        for name, value in results.items():
            print(f"{name}: {value}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        # Clean up
        if collector is not None:
            try:
                if address in collector.instruments:
                    collector.disconnect_instrument(address)
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")

if __name__ == "__main__":
    main()
