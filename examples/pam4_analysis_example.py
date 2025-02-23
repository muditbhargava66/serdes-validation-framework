#!/usr/bin/env python3
"""
PAM4 Signal Analysis Example Script

This script demonstrates the analysis of PAM4 signals using the SerDes validation framework.
It includes signal generation, analysis, and visualization of PAM4 characteristics including:
- Level separation analysis
- Error Vector Magnitude (EVM) calculations
- Eye diagram measurements
- Performance metrics visualization

The script handles proper data types, includes comprehensive error checking,
and demonstrates best practices for signal analysis.

Dependencies:
    - numpy
    - matplotlib
    - logging
    - serdes_validation_framework

Author: Mudit Bhargava
Date: February 2025
"""

import logging
import os
import sys
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.data_analysis.pam4_analyzer import EVMResults, EyeResults, PAM4Analyzer, PAM4Levels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type aliases for improved readability
TimeVoltageDict = Dict[str, npt.NDArray[np.float64]]
SignalParameters = Dict[str, Union[float, int]]

# Constants for signal generation and analysis
SIGNAL_PARAMS: SignalParameters = {
    'DEFAULT_DURATION': 1e-9,      # 1 ns
    'DEFAULT_SAMPLE_RATE': 256e9,  # 256 GSa/s
    'SYMBOL_RATE': 112e9,         # 112 Gbaud
    'NOISE_AMPLITUDE': 0.1,       # Relative noise amplitude
    'PAM4_LEVELS': [-3.0, -1.0, 1.0, 3.0]  # Normalized PAM4 levels
}

def validate_signal_parameters(
    duration: float,
    sample_rate: float,
    noise_level: Optional[float] = None
) -> None:
    """
    Validate parameters for signal generation.
    
    Args:
        duration: Signal duration in seconds
        sample_rate: Sampling rate in Hz
        noise_level: Optional noise amplitude (if None, uses default)
        
    Raises:
        AssertionError: If parameters are invalid
        ValueError: If parameters are out of reasonable ranges
    """
    # Type checking
    assert isinstance(duration, float), f"Duration must be float, got {type(duration)}"
    assert isinstance(sample_rate, float), f"Sample rate must be float, got {type(sample_rate)}"
    if noise_level is not None:
        assert isinstance(noise_level, float), f"Noise level must be float, got {type(noise_level)}"

    # Value range checking
    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration}")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    if noise_level is not None and noise_level < 0:
        raise ValueError(f"Noise level must be non-negative, got {noise_level}")

    # Reasonable range checking
    if duration > 1e-6:  # More than 1 µs
        raise ValueError(f"Duration {duration} seems unreasonably long")
    if sample_rate < 1e9:  # Less than 1 GSa/s
        raise ValueError(f"Sample rate {sample_rate} seems unreasonably low")
    if sample_rate > 1e12:  # More than 1 TSa/s
        raise ValueError(f"Sample rate {sample_rate} seems unreasonably high")

def generate_test_signal(
    duration: float = SIGNAL_PARAMS['DEFAULT_DURATION'],
    sample_rate: float = SIGNAL_PARAMS['DEFAULT_SAMPLE_RATE'],
    noise_amplitude: float = SIGNAL_PARAMS['NOISE_AMPLITUDE']
) -> TimeVoltageDict:
    """
    Generate synthetic PAM4 test signal with realistic characteristics.
    
    Args:
        duration: Signal duration in seconds
        sample_rate: Sampling rate in Hz
        noise_amplitude: Amplitude of additive Gaussian noise
        
    Returns:
        Dictionary containing:
            - 'time': Time points array (float64)
            - 'voltage': Voltage values array (float64)
        
    Raises:
        AssertionError: If input parameters are invalid
        ValueError: If signal generation fails
    """
    try:
        # Validate input parameters
        validate_signal_parameters(duration, sample_rate, noise_amplitude)

        # Calculate timing parameters
        samples_per_symbol = int(sample_rate / SIGNAL_PARAMS['SYMBOL_RATE'])
        total_samples = int(duration * sample_rate)
        num_symbols = total_samples // samples_per_symbol

        # Generate time array
        time = np.arange(total_samples, dtype=np.float64) / sample_rate

        # Generate PAM4 symbols
        levels = np.array(SIGNAL_PARAMS['PAM4_LEVELS'], dtype=np.float64)
        raw_symbols = np.random.choice(levels, size=num_symbols)

        # Create full signal with symbol repetition
        symbols = np.repeat(raw_symbols, samples_per_symbol)
        if len(symbols) < total_samples:
            pad_length = total_samples - len(symbols)
            symbols = np.pad(symbols, (0, pad_length), 'edge')

        # Add Gaussian noise
        noise = np.random.normal(0, noise_amplitude, total_samples).astype(np.float64)
        voltage = (symbols + noise).astype(np.float64)

        # Validate output arrays
        assert_valid_signal_arrays(time, voltage)

        return {
            'time': time,
            'voltage': voltage
        }

    except Exception as e:
        logger.error(f"Failed to generate test signal: {e}")
        raise

def assert_valid_signal_arrays(
    time: npt.NDArray[np.float64],
    voltage: npt.NDArray[np.float64]
) -> None:
    """
    Validate signal arrays for proper types and characteristics.
    
    Args:
        time: Time points array
        voltage: Voltage values array
        
    Raises:
        AssertionError: If arrays are invalid
    """
    # Check array types
    assert isinstance(time, np.ndarray), f"Time must be numpy array, got {type(time)}"
    assert isinstance(voltage, np.ndarray), f"Voltage must be numpy array, got {type(voltage)}"

    # Check data types
    assert np.issubdtype(time.dtype, np.floating), \
        f"Time array must be floating-point, got {time.dtype}"
    assert np.issubdtype(voltage.dtype, np.floating), \
        f"Voltage array must be floating-point, got {voltage.dtype}"

    # Check array properties
    assert len(time) == len(voltage), \
        f"Time and voltage arrays must have same length: {len(time)} != {len(voltage)}"
    assert len(time) > 0, "Arrays cannot be empty"

    # Check for invalid values
    assert not np.any(np.isnan(time)), "Time array contains NaN values"
    assert not np.any(np.isnan(voltage)), "Voltage array contains NaN values"
    assert not np.any(np.isinf(time)), "Time array contains infinite values"
    assert not np.any(np.isinf(voltage)), "Voltage array contains infinite values"

    # Check time array properties
    assert np.all(np.diff(time) > 0), "Time array must be strictly increasing"

def validate_plot_inputs(
    voltage: npt.NDArray[np.float64],
    pam4_levels: PAM4Levels,
    evm_results: EVMResults,
    eye_results: EyeResults
) -> None:
    """
    Validate inputs for plotting function.
    
    Args:
        voltage: Voltage values array
        pam4_levels: PAM4 level analysis results
        evm_results: EVM measurement results
        eye_results: Eye diagram measurements
        
    Raises:
        AssertionError: If inputs are invalid
    """
    # Check types
    assert isinstance(voltage, np.ndarray), f"Voltage must be numpy array, got {type(voltage)}"
    assert isinstance(pam4_levels, PAM4Levels), \
        f"pam4_levels must be PAM4Levels object, got {type(pam4_levels)}"
    assert isinstance(evm_results, EVMResults), \
        f"evm_results must be EVMResults object, got {type(evm_results)}"
    assert isinstance(eye_results, EyeResults), \
        f"eye_results must be EyeResults object, got {type(eye_results)}"

    # Check array properties
    assert len(voltage) > 0, "Voltage array cannot be empty"
    assert len(pam4_levels.level_means) == 4, "PAM4 must have exactly 4 levels"
    assert len(eye_results.eye_heights) == 3, "PAM4 must have exactly 3 eyes"

    # Check value ranges
    assert 0 <= evm_results.rms_evm_percent <= 100, \
        f"RMS EVM must be percentage, got {evm_results.rms_evm_percent}"
    assert 0 <= evm_results.peak_evm_percent <= 100, \
        f"Peak EVM must be percentage, got {evm_results.peak_evm_percent}"
    assert all(h > 0 for h in eye_results.eye_heights), "Eye heights must be positive"
    assert all(w > 0 for w in eye_results.eye_widths), "Eye widths must be positive"

def plot_results(
    voltage: npt.NDArray[np.float64],
    pam4_levels: PAM4Levels,
    evm_results: EVMResults,
    eye_results: EyeResults,
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive visualization of PAM4 analysis results.
    
    Args:
        voltage: Signal voltage array
        pam4_levels: PAM4 level analysis results
        evm_results: EVM measurement results
        eye_results: Eye diagram measurements
        save_path: Optional path to save the plot
        
    Raises:
        ValueError: If plotting fails
    """
    try:
        # Validate inputs
        validate_plot_inputs(voltage, pam4_levels, evm_results, eye_results)

        # Set up plotting style
        plt.style.use('seaborn')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # 1. Signal Histogram with Level Markers
        hist_kwargs = {
            'bins': 100,
            'density': True,
            'alpha': 0.7,
            'color': 'blue'
        }
        ax1.hist(voltage, **hist_kwargs)
        ax1.set_title('PAM4 Level Distribution')
        ax1.set_xlabel('Voltage (normalized)')
        ax1.set_ylabel('Density')

        # Add level markers
        for level in pam4_levels.level_means:
            ax1.axvline(
                level,
                color='red',
                linestyle='--',
                alpha=0.5,
                label='Detected Levels'
            )
        ax1.legend()

        # 2. Signal Trace
        trace_length = min(1000, len(voltage))
        ax2.plot(
            np.arange(trace_length),
            voltage[:trace_length],
            'b-',
            alpha=0.7,
            linewidth=1
        )
        ax2.set_title('PAM4 Signal Trace')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Voltage (normalized)')
        ax2.grid(True, alpha=0.3)

        # 3. Eye Heights
        bar_colors = ['#2ecc71', '#3498db', '#e74c3c']
        ax3.bar(
            range(len(eye_results.eye_heights)),
            eye_results.eye_heights,
            color=bar_colors,
            alpha=0.7
        )
        ax3.set_title('Eye Diagram Heights')
        ax3.set_xlabel('Eye Level')
        ax3.set_ylabel('Height (normalized)')
        ax3.set_xticks(range(len(eye_results.eye_heights)))
        ax3.set_xticklabels(['Lower', 'Middle', 'Upper'])

        # 4. Performance Metrics
        metrics = {
            'RMS EVM (%)': evm_results.rms_evm_percent,
            'Peak EVM (%)': evm_results.peak_evm_percent,
            'Worst Eye\nHeight': eye_results.worst_eye_height,
            'Worst Eye\nWidth': eye_results.worst_eye_width
        }

        metric_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
        ax4.bar(
            range(len(metrics)),
            list(metrics.values()),
            color=metric_colors,
            alpha=0.7
        )
        ax4.set_title('Performance Metrics')
        ax4.set_xticks(range(len(metrics)))
        ax4.set_xticklabels(metrics.keys(), rotation=45)

        # Adjust layout and display
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Failed to plot results: {e}")
        raise

def main() -> None:
    """
    Main function demonstrating PAM4 signal analysis workflow.
    """
    try:
        # 1. Generate test signal
        logger.info("Generating synthetic PAM4 test signal...")
        signal_data = generate_test_signal()

        # 2. Initialize analyzer
        logger.info("Creating PAM4 analyzer...")
        analyzer = PAM4Analyzer(signal_data)

        # 3. Analyze signal levels
        logger.info("Analyzing PAM4 voltage levels...")
        pam4_levels = analyzer.analyze_level_separation('voltage')
        logger.info(
            f"Level analysis results:\n"
            f"  - Means: {pam4_levels.level_means}\n"
            f"  - Separations: {pam4_levels.level_separations}\n"
            f"  - Uniformity: {pam4_levels.uniformity:.3f}"
        )

        # 4. Calculate EVM
        logger.info("Calculating Error Vector Magnitude...")
        evm_results = analyzer.calculate_evm('voltage', 'time')
        logger.info(
            f"EVM measurement results:\n"
            f"  - RMS EVM: {evm_results.rms_evm_percent:.2f}%\n"
            f"  - Peak EVM: {evm_results.peak_evm_percent:.2f}%"
        )

        # 5. Analyze eye diagram
        logger.info("Performing eye diagram analysis...")
        eye_results = analyzer.analyze_eye_diagram('voltage', 'time')
        logger.info(
            f"Eye diagram measurements:\n"
            f"  - Eye heights: {eye_results.eye_heights}\n"
            f"  - Eye widths: {eye_results.eye_widths}\n"
            f"  - Worst eye height: {eye_results.worst_eye_height:.3f}\n"
            f"  - Worst eye width: {eye_results.worst_eye_width:.3f}"
        )

        # 6. Visualize results
        logger.info("Generating visualization...")
        plot_results(
            signal_data['voltage'],
            pam4_levels,
            evm_results,
            eye_results,
            save_path="pam4_analysis_results.png"
        )

        # 7. Print summary
        print_analysis_summary(pam4_levels, evm_results, eye_results)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

def print_analysis_summary(
    pam4_levels: PAM4Levels,
    evm_results: EVMResults,
    eye_results: EyeResults
) -> None:
    """
    Print a comprehensive summary of the PAM4 signal analysis results.
    
    Args:
        pam4_levels: PAM4 level analysis results
        evm_results: EVM measurement results
        eye_results: Eye diagram measurements
    """
    try:
        # Define pass/fail criteria
        CRITERIA = {
            'max_level_uniformity': 0.2,
            'max_rms_evm': 5.0,
            'min_eye_height': 0.4,
            'min_eye_width': 0.6
        }

        # Create summary
        print("\n" + "="*50)
        print("PAM4 Signal Analysis Summary")
        print("="*50)

        # Level Analysis
        print("\nPAM4 Level Analysis:")
        print(f"  Level Means: {[f'{x:.3f}' for x in pam4_levels.level_means]}")
        print(f"  Level Separations: {[f'{x:.3f}' for x in pam4_levels.level_separations]}")
        print(
            f"  Uniformity: {pam4_levels.uniformity:.3f} "
            f"({'PASS' if pam4_levels.uniformity < CRITERIA['max_level_uniformity'] else 'FAIL'})"
        )

        # EVM Results
        print("\nError Vector Magnitude:")
        print(
            f"  RMS EVM: {evm_results.rms_evm_percent:.2f}% "
            f"({'PASS' if evm_results.rms_evm_percent < CRITERIA['max_rms_evm'] else 'FAIL'})"
        )
        print(f"  Peak EVM: {evm_results.peak_evm_percent:.2f}%")

        # Eye Measurements
        print("\nEye Diagram Analysis:")
        print("  Eye Heights:")
        for i, height in enumerate(eye_results.eye_heights):
            print(f"    Eye {i+1}: {height:.3f}")
        print("  Eye Widths:")
        for i, width in enumerate(eye_results.eye_widths):
            print(f"    Eye {i+1}: {width:.3f}")
        print(
            f"  Worst Eye Height: {eye_results.worst_eye_height:.3f} "
            f"({'PASS' if eye_results.worst_eye_height > CRITERIA['min_eye_height'] else 'FAIL'})"
        )
        print(
            f"  Worst Eye Width: {eye_results.worst_eye_width:.3f} "
            f"({'PASS' if eye_results.worst_eye_width > CRITERIA['min_eye_width'] else 'FAIL'})"
        )

        # Overall Status
        overall_status = all([
            pam4_levels.uniformity < CRITERIA['max_level_uniformity'],
            evm_results.rms_evm_percent < CRITERIA['max_rms_evm'],
            eye_results.worst_eye_height > CRITERIA['min_eye_height'],
            eye_results.worst_eye_width > CRITERIA['min_eye_width']
        ])

        print("\n" + "="*50)
        print(f"Overall Status: {'PASS' if overall_status else 'FAIL'}")
        print("="*50 + "\n")

    except Exception as e:
        logger.error(f"Failed to print analysis summary: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
    else:
        logger.info("Analysis completed successfully")
        sys.exit(0)
