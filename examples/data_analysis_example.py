#!/usr/bin/env python3
"""
SerDes Data Analysis Example Script

This script demonstrates comprehensive data analysis capabilities of the SerDes 
validation framework including:
- Basic signal analysis with statistical measures
- PAM4 signal analysis with EVM calculation
- Eye diagram analysis
- Advanced signal processing

The example shows both basic and advanced analysis techniques suitable for
high-speed SerDes validation.

Dependencies:
    - numpy
    - matplotlib (optional, for plotting)
    - serdes_validation_framework
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from serdes_validation_framework.data_analysis.analyzer import DataAnalyzer
from serdes_validation_framework.data_analysis.pam4_analyzer import PAM4Analyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_signals():
    """Generate test signals for analysis demonstration"""
    # Generate time vector
    sample_rate = 100e9  # 100 GSa/s
    duration = 1e-6      # 1 microsecond
    num_samples = int(sample_rate * duration)
    time = np.linspace(0, duration, num_samples, dtype=np.float64)
    
    # Generate NRZ signal
    symbol_rate = 32e9  # 32 Gbps
    symbols_per_sample = sample_rate / symbol_rate
    num_symbols = int(num_samples / symbols_per_sample)
    
    # Random binary data
    binary_data = np.random.choice([-1.0, 1.0], size=num_symbols)
    nrz_signal = np.repeat(binary_data, int(symbols_per_sample))[:num_samples]
    
    # Add noise
    snr_db = 25.0
    signal_power = np.mean(nrz_signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
    nrz_signal += noise
    
    # Generate PAM4 signal
    pam4_levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
    pam4_symbols = np.random.choice(pam4_levels, size=num_symbols)
    pam4_signal = np.repeat(pam4_symbols, int(symbols_per_sample))[:num_samples]
    
    # Add noise to PAM4
    pam4_signal_power = np.mean(pam4_signal**2)
    pam4_noise_power = pam4_signal_power / (10**(snr_db/10))
    pam4_noise = np.random.normal(0, np.sqrt(pam4_noise_power), num_samples)
    pam4_signal += pam4_noise
    
    return {
        'time': time,
        'nrz_signal': nrz_signal.astype(np.float64),
        'pam4_signal': pam4_signal.astype(np.float64)
    }


def demonstrate_basic_analysis():
    """Demonstrate basic signal analysis"""
    logger.info("=== Basic Signal Analysis Demo ===")
    
    # Simple test data
    sample_data = {
        'signal_strength': np.array([0.1, 0.5, 0.3, 0.7, 0.2, 0.4, 0.8], dtype=np.float64)
    }
    
    try:
        analyzer = DataAnalyzer(sample_data)
        stats = analyzer.compute_statistics('signal_strength')
        logger.info(f"Basic statistics: {stats}")
        
        # Try to plot histogram if matplotlib is available
        try:
            analyzer.plot_histogram('signal_strength')
            logger.info("Histogram plot generated successfully")
        except ImportError:
            logger.warning("Matplotlib not available, skipping histogram plot")
        except Exception as e:
            logger.warning(f"Histogram plot failed: {e}")
            
    except Exception as e:
        logger.error(f"Basic analysis failed: {e}")


def demonstrate_advanced_analysis():
    """Demonstrate advanced signal analysis with generated signals"""
    logger.info("=== Advanced Signal Analysis Demo ===")
    
    try:
        # Generate test signals
        signals = generate_test_signals()
        logger.info(f"Generated signals with {len(signals['time'])} samples")
        
        # Analyze NRZ signal
        logger.info("\n--- NRZ Signal Analysis ---")
        nrz_data = {
            'time': signals['time'],
            'voltage': signals['nrz_signal']
        }
        
        nrz_analyzer = DataAnalyzer(nrz_data)
        nrz_stats = nrz_analyzer.compute_statistics('voltage')
        logger.info("NRZ signal statistics:")
        for key, value in nrz_stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Analyze PAM4 signal
        logger.info("\n--- PAM4 Signal Analysis ---")
        pam4_data = {
            'time': signals['time'],
            'voltage': signals['pam4_signal']
        }
        
        try:
            pam4_analyzer = PAM4Analyzer(pam4_data)
            
            # Level separation analysis
            level_results = pam4_analyzer.analyze_level_separation()
            logger.info(f"PAM4 level separation: {level_results}")
            
            # EVM calculation
            evm_results = pam4_analyzer.calculate_evm()
            logger.info(f"PAM4 EVM results: {evm_results}")
            
            # Eye diagram analysis
            try:
                eye_results = pam4_analyzer.analyze_eye_diagram()
                logger.info(f"PAM4 eye diagram results: {eye_results}")
            except Exception as e:
                logger.warning(f"Eye diagram analysis failed: {e}")
                
        except Exception as e:
            logger.error(f"PAM4 analysis failed: {e}")
        
        # Signal quality comparison
        logger.info("\n--- Signal Quality Comparison ---")
        nrz_snr = calculate_snr(signals['nrz_signal'])
        pam4_snr = calculate_snr(signals['pam4_signal'])
        
        logger.info(f"NRZ SNR: {nrz_snr:.2f} dB")
        logger.info(f"PAM4 SNR: {pam4_snr:.2f} dB")
        
        # Signal power analysis
        nrz_power = np.mean(signals['nrz_signal']**2)
        pam4_power = np.mean(signals['pam4_signal']**2)
        
        logger.info(f"NRZ signal power: {nrz_power:.6f}")
        logger.info(f"PAM4 signal power: {pam4_power:.6f}")
        
    except Exception as e:
        logger.error(f"Advanced analysis failed: {e}")


def calculate_snr(signal):
    """Calculate signal-to-noise ratio"""
    try:
        # Simple SNR estimation
        signal_power = np.mean(signal**2)
        
        # Estimate noise from high-frequency components
        from scipy import signal as scipy_signal
        
        # High-pass filter to isolate noise
        nyquist = 0.5
        high_cutoff = 0.8
        b, a = scipy_signal.butter(4, high_cutoff, btype='high')
        noise_estimate = scipy_signal.filtfilt(b, a, signal)
        noise_power = np.mean(noise_estimate**2)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = float('inf')
            
        return float(snr_db)
        
    except ImportError:
        logger.warning("SciPy not available, using simplified SNR calculation")
        # Fallback: use signal variance as noise estimate
        signal_power = np.mean(signal**2)
        noise_power = np.var(signal)
        return float(10 * np.log10(signal_power / noise_power)) if noise_power > 0 else float('inf')
    except Exception as e:
        logger.error(f"SNR calculation failed: {e}")
        return 0.0


def demonstrate_plotting():
    """Demonstrate signal plotting capabilities"""
    logger.info("=== Signal Plotting Demo ===")
    
    try:
        import matplotlib.pyplot as plt
        
        # Generate short signals for plotting
        signals = generate_test_signals()
        
        # Plot first 1000 samples
        plot_samples = 1000
        time_plot = signals['time'][:plot_samples] * 1e9  # Convert to ns
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot NRZ signal
        ax1.plot(time_plot, signals['nrz_signal'][:plot_samples], 'b-', linewidth=1)
        ax1.set_title('NRZ Signal')
        ax1.set_ylabel('Voltage (V)')
        ax1.grid(True, alpha=0.3)
        
        # Plot PAM4 signal
        ax2.plot(time_plot, signals['pam4_signal'][:plot_samples], 'r-', linewidth=1)
        ax2.set_title('PAM4 Signal')
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Voltage (V)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('signal_analysis_example.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Signal plots saved as 'signal_analysis_example.png'")
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping plotting demo")
    except Exception as e:
        logger.error(f"Plotting demo failed: {e}")


def main():
    """Main function demonstrating various analysis capabilities"""
    logger.info("SerDes Data Analysis Example - v1.3.0")
    logger.info("=" * 50)
    
    try:
        # Run demonstrations
        demonstrate_basic_analysis()
        demonstrate_advanced_analysis()
        demonstrate_plotting()
        
        logger.info("\n" + "=" * 50)
        logger.info("Data analysis demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
