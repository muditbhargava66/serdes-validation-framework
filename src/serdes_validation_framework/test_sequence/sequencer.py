"""
Test Sequencer Module

Provides functionality for executing test sequences with support for both
real and mock hardware modes.
"""

import logging
from typing import Any, Dict, List, Optional

from ..data_analysis.analyzer import DataAnalyzer
from ..instrument_control.mock_controller import get_instrument_controller

logger = logging.getLogger(__name__)


class PCIeTestSequencer:
    """Test sequence execution controller"""

    def __init__(self, controller: Optional[Any] = None) -> None:
        """
        Initialize test sequencer

        Args:
            controller: Optional instrument controller (uses auto-detection if None)
        """
        self.instruments: Dict[str, Any] = {}
        self.instrument_controller = controller if controller is not None else get_instrument_controller()
        self.data_collector = None  # Will be initialized during setup
        logger.info("Test sequencer initialized")

    def setup_instruments(self, resource_names: List[str]) -> None:
        """
        Set up instruments for test sequence

        Args:
            resource_names: List of VISA resource identifiers

        Raises:
            ValueError: If resource names are invalid
        """
        try:
            # Validate inputs
            if not resource_names:
                raise ValueError("No resource names provided")

            for resource in resource_names:
                if not isinstance(resource, str) or not resource:
                    raise ValueError(f"Invalid resource name: {resource}")

                # Connect instrument
                self.instrument_controller.connect_instrument(resource)
                self.instruments[resource] = self.instrument_controller
                logger.info(f"Connected to instrument {resource}")

        except Exception as e:
            logger.error(f"Failed to set up instruments: {e}")
            self.cleanup(list(self.instruments.keys()))
            raise

    def run_sequence(self, sequence: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Execute test sequence

        Args:
            sequence: List of command dictionaries with keys:
                     - resource: Instrument identifier
                     - command: SCPI command
                     - action: 'send' or 'query'

        Returns:
            Dictionary mapping resources to responses

        Raises:
            ValueError: If sequence is invalid
        """
        results = {}
        try:
            # Validate sequence
            if not sequence:
                raise ValueError("Empty sequence")

            # Execute sequence
            for step in sequence:
                # Validate step
                if not all(k in step for k in ["resource", "command", "action"]):
                    raise ValueError(f"Invalid sequence step: {step}")

                resource = step["resource"]
                command = step["command"]
                action = step["action"]

                # Execute step
                if action == "send":
                    self.instrument_controller.send_command(resource, command)
                elif action == "query":
                    response = self.instrument_controller.query_instrument(resource, command)
                    results[resource] = response
                else:
                    raise ValueError(f"Invalid action: {action}")

            return results

        except Exception as e:
            logger.error(f"Failed to run sequence: {e}")
            raise

    def collect_and_analyze_data(self, resource: str, command: str, data_type: str) -> Dict[str, float]:
        """
        Collect and analyze measurement data

        Args:
            resource: Instrument identifier
            command: Measurement command
            data_type: Type of data being measured

        Returns:
            Dictionary of analysis results
        """
        try:
            # Collect data
            response = self.instrument_controller.query_instrument(resource, command)

            # Convert to numeric data
            try:
                value = float(response)
                data = {data_type: [value]}
            except ValueError:
                raise ValueError(f"Invalid numeric response: {response}")

            # Analyze data
            analyzer = DataAnalyzer(data)
            stats = analyzer.compute_statistics(data_type)
            return stats.to_dict()

        except Exception as e:
            logger.error(f"Failed to collect and analyze data: {e}")
            raise

    def cleanup(self, resource_names: List[str]) -> None:
        """
        Clean up instrument connections

        Args:
            resource_names: List of instrument identifiers
        """
        for resource in resource_names:
            try:
                if resource in self.instruments:
                    self.instrument_controller.disconnect_instrument(resource)
                    del self.instruments[resource]
                    logger.info(f"Disconnected from {resource}")
            except Exception as e:
                logger.warning(f"Failed to disconnect from {resource}: {e}")
