"""
224G Ethernet Validation Example Script

A comprehensive example demonstrating 224G Ethernet testing capabilities using
the SerDes validation framework. Features include:
- Link training simulation
- PAM4 signal generation
- Compliance testing
- Advanced visualization
- Mock instrument control

The script provides realistic mock responses for:
- Eye diagram measurements
- Jitter analysis
- PAM4 level measurements
- Link training sequences

Key Features:
    - Automated test sequencing
    - Real-time data visualization
    - Comprehensive error handling
    - Test results storage and analysis
    - Mock instrument simulation

Author: Mudit Bhargava
Date: February 2025
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.instrument_control.mock_controller import (
    MockInstrumentController,
    get_instrument_controller,
    get_instrument_mode,
)
from src.serdes_validation_framework.instrument_control.scope_224g import HighBandwidthScope
from src.serdes_validation_framework.test_sequence.eth_224g_sequence import Ethernet224GTestSequence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_instruments(
    scope_address: str = 'GPIB0::7::INSTR',
    pattern_gen_address: str = 'GPIB0::10::INSTR'
) -> Tuple[HighBandwidthScope, Ethernet224GTestSequence]:
    """
    Set up test instruments with mock controller support
    
    Args:
        scope_address: VISA address for oscilloscope
        pattern_gen_address: VISA address for pattern generator
        
    Returns:
        Tuple of (scope controller, test sequencer)
    """
    try:
        # Get controller (mock or real based on environment)
        controller = get_instrument_controller()
        mode = get_instrument_mode()
        
        # Initialize scope with controller
        logger.info(f"Connecting to scope at {scope_address}")
        scope = HighBandwidthScope(scope_address, controller=controller)

        # Configure mock responses for PAM4 measurements if using mock controller
        if isinstance(controller, MockInstrumentController):
            configure_mock_responses(controller)

        # Initialize test sequencer
        logger.info("Initializing test sequencer")
        sequencer = Ethernet224GTestSequence()

        # Set up instruments
        logger.info("Setting up instruments")
        sequencer.setup_instruments([scope_address, pattern_gen_address])

        return scope, sequencer

    except Exception as e:
        logger.error(f"Failed to set up instruments: {e}")
        raise

def configure_mock_responses(controller: MockInstrumentController) -> None:
    """
    Configure mock controller with realistic PAM4 responses
    
    Args:
        controller: Mock controller instance to configure
    """
    try:
        # Update waveform generation parameters
        controller.add_mock_response(
            ':WAVeform:DATA?',
            lambda: generate_pam4_waveform(),
            delay=0.1
        )
        
        # Add mock responses for scope measurements
        controller.add_mock_response(
            ':MEASure:EYE:HEIGht?',
            lambda: f"{np.random.normal(0.6, 0.05):.6f}",
            delay=0.1
        )
        controller.add_mock_response(
            ':MEASure:EYE:WIDTh?',
            lambda: f"{np.random.normal(0.7, 0.05):.6f}",
            delay=0.1
        )
        controller.add_mock_response(
            ':MEASure:JITTer:TJ?',
            lambda: f"{np.random.normal(2e-12, 1e-13):.3e}",
            delay=0.1
        )
        
    except Exception as e:
        logger.error(f"Failed to configure mock responses: {e}")
        raise

def generate_pam4_waveform() -> str:
    """
    Generate synthetic PAM4 waveform data with enhanced level separation
    
    Returns:
        Comma-separated waveform data string with well-defined PAM4 levels
    """
    try:
        # Use fixed parameters for stable output
        num_points = 1000000
        samples_per_symbol = 32  # Reduced for clearer level transitions
        num_symbols = num_points // samples_per_symbol
        
        # Define exact PAM4 levels with wider separation
        levels = np.array([-0.9, -0.3, 0.3, 0.9])  # Increased separation
        
        # Generate balanced symbol pattern
        symbols = np.zeros(num_symbols)
        for i in range(0, num_symbols, 4):
            pattern = np.array([0, 1, 2, 3])  # All four levels
            np.random.shuffle(pattern)  # Randomize order but keep all levels
            if i + 4 <= num_symbols:
                symbols[i:i+4] = pattern
            else:
                symbols[i:] = pattern[:num_symbols-i]
                
        # Map to voltage levels and repeat for sample rate
        voltage_levels = levels[symbols.astype(int)]
        waveform = np.repeat(voltage_levels, samples_per_symbol)
        
        # Add minimal noise
        noise_amplitude = 0.01  # Very small noise
        noise = np.random.normal(0, noise_amplitude, len(waveform))
        
        # Apply raised cosine filter for smooth transitions
        t = np.linspace(-1, 1, 16)
        h = 0.5 * (1 + np.cos(np.pi * t))
        filtered = np.convolve(waveform, h/np.sum(h), mode='same')
        
        # Add very small jitter
        t = np.linspace(0, num_points/256e9, num_points)
        jitter = 0.005 * np.sin(2 * np.pi * 1e9 * t)
        
        # Combine components
        final_waveform = filtered + noise + jitter
        
        # Ensure perfect centering and scaling
        final_waveform = final_waveform - np.mean(final_waveform)
        scale = 0.9 / np.max(np.abs(final_waveform))
        final_waveform = final_waveform * scale
        
        # Return formatted string with high precision
        return ','.join(f"{x:.8f}" for x in final_waveform)
        
    except Exception as e:
        logger.error(f"Failed to generate PAM4 waveform: {e}")
        return ','.join(['0.0'] * 1000)  # Fallback data

def run_link_training(
    sequencer: Ethernet224GTestSequence,
    scope_address: str,
    pattern_gen_address: str,
    output_dir: str = 'validation_results'
) -> None:
    """
    Run and plot link training results
    
    Args:
        sequencer: Test sequencer instance
        scope_address: Scope address
        pattern_gen_address: Pattern generator address
        output_dir: Directory to save results (default: validation_results)
    """
    try:
        # Create output directory if it doesn't exist
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(output_dir) / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Running link training...")
        results = sequencer.run_link_training_test(
            scope_address,
            pattern_gen_address
        )

        # Plot results if available and valid
        if hasattr(plt, 'figure') and len(results.final_eq_settings) > 0:
            # Close any existing plots
            plt.close('all')
            
            # Create figure with proper sizing and DPI
            fig = plt.figure(figsize=(12, 8), dpi=100)
            
            # Plot equalizer taps with improved formatting
            ax1 = plt.subplot(2, 1, 1)
            x_taps = np.arange(len(results.final_eq_settings))
            ax1.bar(x_taps, results.final_eq_settings, color='blue', alpha=0.7)
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Final Equalizer Taps', pad=10)
            ax1.set_xlabel('Tap Number')
            ax1.set_ylabel('Tap Weight')
            ax1.set_ylim([-0.5, 0.5])  # Set reasonable y-axis limits
            
            # Plot error history with improvements
            ax2 = plt.subplot(2, 1, 2)
            if len(results.error_history) > 0:
                x_error = np.arange(len(results.error_history))
                ax2.plot(x_error, results.error_history, 'r-', linewidth=1, alpha=0.8)
                ax2.grid(True, alpha=0.3)
                ax2.set_title('Adaptation Error History', pad=10)
                ax2.set_xlabel('Block Number')
                ax2.set_ylabel('Average Error')
                # Set reasonable y-axis limits
                error_mean = np.mean(results.error_history)
                error_std = np.std(results.error_history)
                ax2.set_ylim([error_mean - 3*error_std, error_mean + 3*error_std])
            
            plt.tight_layout()
            
            # Save plots
            plot_file = save_dir / 'link_training_results.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {plot_file}")
            
            # Show plot with longer display time
            plt.show(block=False)
            plt.pause(10)  # Keep plot open for 10 seconds
            plt.close(fig)

        # Save training results to file
        results_file = save_dir / 'link_training_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'status': results.convergence_status,
                'training_time': results.training_time,
                'adaptation_error': results.adaptation_error,
                'final_eq_settings': list(map(float, results.final_eq_settings)),
                'error_history': list(map(float, results.error_history))
            }, f, indent=4)

        # Log results
        logger.info(f"Training completed: {results.convergence_status}")
        logger.info(f"Training time: {results.training_time:.2f} seconds")
        logger.info(f"Final adaptation error: {results.adaptation_error:.3e}")
        logger.info(f"Results saved to {save_dir}")

    except Exception as e:
        logger.error(f"Link training failed: {e}")
        raise

def run_compliance_tests(
    sequencer: Ethernet224GTestSequence,
    scope_address: str,
    pattern_gen_address: str
) -> None:
    """Run and display compliance test results"""
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
    # Force mock mode for example
    os.environ['SVF_MOCK_MODE'] = '1'
    
    try:
        # Set up instruments with mock controller
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
