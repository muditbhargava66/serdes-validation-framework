# examples/eth_224g_example.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.test_sequence.eth_224g_sequence import (
    Ethernet224GTestSequence,
    TrainingResults,
    ComplianceResults
)
from src.serdes_validation_framework.instrument_control.scope_224g import (
    HighBandwidthScope,
    ScopeConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_instruments(
    scope_address: str = 'GPIB0::7::INSTR',
    pattern_gen_address: str = 'GPIB0::10::INSTR'
) -> Tuple[HighBandwidthScope, Ethernet224GTestSequence]:
    """
    Set up test instruments
    
    Args:
        scope_address: VISA address for oscilloscope
        pattern_gen_address: VISA address for pattern generator
        
    Returns:
        Tuple of (scope controller, test sequencer)
    """
    try:
        # Initialize scope
        logger.info(f"Connecting to scope at {scope_address}")
        scope = HighBandwidthScope(scope_address)
        
        # Initialize test sequence
        logger.info("Initializing test sequencer")
        sequencer = Ethernet224GTestSequence()
        
        # Set up instruments
        logger.info("Setting up instruments")
        sequencer.setup_instruments([scope_address, pattern_gen_address])
        
        return scope, sequencer
        
    except Exception as e:
        logger.error(f"Failed to set up instruments: {e}")
        raise

def run_link_training(
    sequencer: Ethernet224GTestSequence,
    scope_address: str,
    pattern_gen_address: str
) -> None:
    """
    Run and plot link training results
    
    Args:
        sequencer: Test sequencer instance
        scope_address: Scope address
        pattern_gen_address: Pattern generator address
    """
    try:
        logger.info("Running link training...")
        results = sequencer.run_link_training_test(
            scope_address,
            pattern_gen_address
        )
        
        # Create figure with proper sizing and DPI
        plt.figure(figsize=(12, 8), dpi=100)
        
        # Plot equalizer taps
        plt.subplot(2, 1, 1)
        x_taps = np.arange(len(results.final_eq_settings))
        plt.bar(x_taps, results.final_eq_settings, color='blue', alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.title('Final Equalizer Taps', pad=10)
        plt.xlabel('Tap Number')
        plt.ylabel('Tap Weight')
        
        # Plot error history
        plt.subplot(2, 1, 2)
        x_error = np.arange(len(results.error_history))
        plt.plot(x_error, results.error_history, 'r-', linewidth=1, alpha=0.8)
        plt.grid(True, alpha=0.3)
        plt.title('Adaptation Error History', pad=10)
        plt.xlabel('Block Number')
        plt.ylabel('Average Error')
        
        plt.tight_layout()
        plt.show()
        
        # Log results
        logger.info(f"Training completed: {results.convergence_status}")
        logger.info(f"Training time: {results.training_time:.2f} seconds")
        logger.info(f"Final adaptation error: {results.adaptation_error:.3e}")
        
    except Exception as e:
        logger.error(f"Link training failed: {e}")
        raise

def run_compliance_tests(
    sequencer: Ethernet224GTestSequence,
    scope_address: str,
    pattern_gen_address: str
) -> None:
    """
    Run and display compliance test results
    
    Args:
        sequencer: Test sequencer instance
        scope_address: Scope address
        pattern_gen_address: Pattern generator address
    """
    try:
        logger.info("Running compliance tests...")
        results = sequencer.run_compliance_test_suite(
            scope_address,
            pattern_gen_address
        )
        
        # Create results summary
        logger.info("\n=== Compliance Test Results ===")
        logger.info(f"Overall Status: {results.test_status}")
        
        logger.info("\nPAM4 Levels:")
        logger.info(f"- Level means: {results.pam4_levels.level_means}")
        logger.info(f"- Level separations: {results.pam4_levels.level_separations}")
        logger.info(f"- Uniformity: {results.pam4_levels.uniformity:.3f}")
        
        logger.info("\nEVM Results:")
        logger.info(f"- RMS EVM: {results.evm_results.rms_evm_percent:.2f}%")
        logger.info(f"- Peak EVM: {results.evm_results.peak_evm_percent:.2f}%")
        
        logger.info("\nEye Diagram Results:")
        logger.info(f"- Eye heights: {results.eye_results.eye_heights}")
        logger.info(f"- Eye widths: {results.eye_results.eye_widths}")
        logger.info(f"- Worst eye height: {results.eye_results.worst_eye_height:.3f}")
        logger.info(f"- Worst eye width: {results.eye_results.worst_eye_width:.3f}")
        
        logger.info("\nJitter Results:")
        for jitter_type, value in results.jitter_results.items():
            logger.info(f"- {jitter_type.upper()}: {value:.3f} ps")
        
    except Exception as e:
        logger.error(f"Compliance tests failed: {e}")
        raise

def main() -> None:
    """Main example function"""
    try:
        # For example purposes, we'll use mock instruments
        with patch('pyvisa.ResourceManager') as mock_rm:
            # Setup mock responses
            mock_instrument = MagicMock()
            mock_rm.return_value.open_resource.return_value = mock_instrument
            
            # Configure mock to return realistic PAM4 data
            def mock_query(query: str) -> str:
                if query == ":WAVeform:DATA?":
                    # Generate synthetic PAM4 data
                    num_points = 1000000
                    base_levels = [-3, -1, 1, 3]
                    symbols = np.random.choice(base_levels, num_points)
                    noise = np.random.normal(0, 0.1, num_points)
                    data = symbols + noise
                    return ','.join(map(str, data))
                elif query == "*IDN?":
                    return "Mock Scope"
                else:
                    return "0.0"
                    
            mock_instrument.query = mock_query
            
            # Set up instruments
            scope_address = 'GPIB0::7::INSTR'
            pattern_gen_address = 'GPIB0::10::INSTR'
            scope, sequencer = setup_instruments(scope_address, pattern_gen_address)
            
            # Configure scope
            logger.info("Configuring scope for 224G...")
            scope.configure_for_224g()
            
            # Run link training
            run_link_training(sequencer, scope_address, pattern_gen_address)
            
            # Run compliance tests
            run_compliance_tests(sequencer, scope_address, pattern_gen_address)
            
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
    finally:
        # Clean up
        try:
            scope.cleanup()
            sequencer.cleanup([scope_address, pattern_gen_address])
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed: {cleanup_error}")

if __name__ == "__main__":
    main()