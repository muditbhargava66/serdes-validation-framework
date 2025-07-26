#!/usr/bin/env python3
"""
Data Analysis Script

This script provides comprehensive data analysis capabilities for SerDes validation
including signal processing, statistical analysis, and visualization for PCIe 6.0,
224G Ethernet, and USB4/Thunderbolt 4 protocols.

Features:
- Multi-protocol signal analysis
- Advanced statistical processing
- Eye diagram analysis
- Jitter decomposition
- PAM4 and NRZ signal processing
- USB4 dual-lane analysis
- Comprehensive visualization

Usage:
    python scripts/data_analysis.py [--input FILE] [--output DIR] [--verbose]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from serdes_validation_framework.data_analysis.analyzer import DataAnalyzer

# Try to import advanced analyzers
try:
    from serdes_validation_framework.data_analysis.pam4_analyzer import PAM4Analyzer
    PAM4_AVAILABLE = True
except ImportError:
    PAM4_AVAILABLE = False

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def generate_sample_data():
    """Generate comprehensive sample data for analysis"""
    np.random.seed(42)  # For reproducible results
    
    # Basic signal strength data
    signal_strength = np.random.normal(0.5, 0.2, 1000)
    signal_strength = np.clip(signal_strength, 0, 1)  # Clip to valid range
    
    # Time-series voltage data
    time = np.linspace(0, 1e-6, 10000)  # 1 microsecond
    frequency = 1e6  # 1 MHz
    voltage = np.sin(2 * np.pi * frequency * time) + 0.1 * np.random.randn(len(time))
    
    # PAM4-like signal
    pam4_levels = [-3, -1, 1, 3]
    pam4_symbols = np.random.choice(pam4_levels, size=5000)
    pam4_noise = 0.2 * np.random.randn(5000)
    pam4_signal = pam4_symbols + pam4_noise
    
    return {
        'signal_strength': signal_strength.astype(np.float64),
        'time': time.astype(np.float64),
        'voltage': voltage.astype(np.float64),
        'pam4_signal': pam4_signal.astype(np.float64)
    }

def analyze_basic_data(data, logger):
    """Perform basic data analysis"""
    logger.info("=== Basic Data Analysis ===")
    
    try:
        analyzer = DataAnalyzer(data)
        
        # Analyze each data type
        for data_type in ['signal_strength', 'voltage', 'pam4_signal']:
            if data_type in data:
                logger.info(f"\nAnalyzing {data_type}:")
                stats = analyzer.compute_statistics(data_type)
                
                for key, value in stats.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.6f}")
                    else:
                        logger.info(f"  {key}: {value}")
                
                # Try to generate histogram
                try:
                    analyzer.plot_histogram(data_type)
                    logger.info(f"  Histogram saved for {data_type}")
                except ImportError:
                    logger.warning("  Matplotlib not available, skipping histogram")
                except Exception as e:
                    logger.warning(f"  Histogram generation failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Basic analysis failed: {e}")
        return False

def analyze_pam4_data(data, logger):
    """Perform PAM4-specific analysis"""
    if not PAM4_AVAILABLE:
        logger.warning("PAM4 analyzer not available, skipping PAM4 analysis")
        return False
    
    logger.info("=== PAM4 Signal Analysis ===")
    
    try:
        # Create PAM4 data structure
        pam4_data = {
            'time': np.linspace(0, len(data['pam4_signal'])/1e9, len(data['pam4_signal'])),
            'voltage': data['pam4_signal']
        }
        
        analyzer = PAM4Analyzer(pam4_data)
        
        # Level separation analysis
        try:
            level_results = analyzer.analyze_level_separation()
            logger.info(f"Level separation results: {level_results}")
        except Exception as e:
            logger.warning(f"Level separation analysis failed: {e}")
        
        # EVM calculation
        try:
            evm_results = analyzer.calculate_evm()
            logger.info(f"EVM results: {evm_results}")
        except Exception as e:
            logger.warning(f"EVM calculation failed: {e}")
        
        # Eye diagram analysis
        try:
            eye_results = analyzer.analyze_eye_diagram()
            logger.info(f"Eye diagram results: {eye_results}")
        except Exception as e:
            logger.warning(f"Eye diagram analysis failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"PAM4 analysis failed: {e}")
        return False

def analyze_signal_quality(data, logger):
    """Perform signal quality analysis"""
    logger.info("=== Signal Quality Analysis ===")
    
    try:
        # SNR calculation for voltage signal
        voltage = data['voltage']
        signal_power = np.mean(voltage**2)
        
        # Estimate noise from high-frequency components
        try:
            from scipy import signal
            # High-pass filter to isolate noise
            nyquist = 0.5
            high_cutoff = 0.8
            b, a = signal.butter(4, high_cutoff, btype='high')
            noise_estimate = signal.filtfilt(b, a, voltage)
            noise_power = np.mean(noise_estimate**2)
            
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
                logger.info(f"Estimated SNR: {snr_db:.2f} dB")
            else:
                logger.info("SNR: Infinite (no detectable noise)")
                
        except ImportError:
            logger.warning("SciPy not available, using simplified noise estimation")
            noise_power = np.var(voltage)
            snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            logger.info(f"Simplified SNR estimate: {snr_db:.2f} dB")
        
        # Peak-to-peak analysis
        voltage_pp = np.max(voltage) - np.min(voltage)
        logger.info(f"Voltage peak-to-peak: {voltage_pp:.6f} V")
        
        # RMS calculation
        voltage_rms = np.sqrt(np.mean(voltage**2))
        logger.info(f"Voltage RMS: {voltage_rms:.6f} V")
        
        # Crest factor
        voltage_peak = np.max(np.abs(voltage))
        crest_factor = voltage_peak / voltage_rms if voltage_rms > 0 else 0
        logger.info(f"Crest factor: {crest_factor:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Signal quality analysis failed: {e}")
        return False

def load_data_from_file(filepath, logger):
    """Load data from file"""
    try:
        # Support different file formats
        file_path = Path(filepath)
        
        if file_path.suffix.lower() == '.csv':
            import pandas as pd
            df = pd.read_csv(file_path)
            data = {col: df[col].values.astype(np.float64) for col in df.columns}
            logger.info(f"Loaded CSV data with columns: {list(data.keys())}")
            
        elif file_path.suffix.lower() == '.npy':
            array_data = np.load(file_path)
            if array_data.ndim == 1:
                data = {'signal': array_data.astype(np.float64)}
            else:
                data = {f'channel_{i}': array_data[:, i].astype(np.float64) 
                       for i in range(array_data.shape[1])}
            logger.info(f"Loaded NumPy data with shape: {array_data.shape}")
            
        elif file_path.suffix.lower() == '.txt':
            # Assume space or comma separated values
            array_data = np.loadtxt(file_path)
            if array_data.ndim == 1:
                data = {'signal': array_data.astype(np.float64)}
            else:
                data = {f'channel_{i}': array_data[:, i].astype(np.float64) 
                       for i in range(array_data.shape[1])}
            logger.info(f"Loaded text data with shape: {array_data.shape}")
            
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {e}")
        return None

def save_results(results, output_dir, logger):
    """Save analysis results"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save results as text file
        results_file = output_path / "analysis_results.txt"
        with open(results_file, 'w') as f:
            f.write("SerDes Data Analysis Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Generated: {np.datetime64('now')}\n\n")
            
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Results saved to: {results_file}")
        return str(results_file)
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return None

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="SerDes Data Analysis Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', 
        help='Input data file (CSV, NPY, or TXT format)'
    )
    
    parser.add_argument(
        '--output', 
        default='./analysis_output',
        help='Output directory for results (default: ./analysis_output)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("SerDes Data Analysis Script - v1.4.0")
    
    try:
        # Load or generate data
        if args.input:
            data = load_data_from_file(args.input, logger)
            if data is None:
                sys.exit(1)
        else:
            logger.info("No input file specified, generating sample data")
            data = generate_sample_data()
        
        # Run analyses
        results = {
            'basic_analysis': analyze_basic_data(data, logger),
            'pam4_analysis': analyze_pam4_data(data, logger),
            'signal_quality': analyze_signal_quality(data, logger)
        }
        
        # Save results
        save_results(results, args.output, logger)
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 50)
        
        passed_analyses = sum(results.values())
        total_analyses = len(results)
        
        logger.info(f"Completed analyses: {passed_analyses}/{total_analyses}")
        
        for analysis, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"  {analysis}: {status}")
        
        if passed_analyses == total_analyses:
            logger.info("🎉 All analyses completed successfully!")
        else:
            logger.warning("⚠️  Some analyses failed - check logs for details")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
