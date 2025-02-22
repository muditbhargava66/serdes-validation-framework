import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging
from src.serdes_validation_framework.data_analysis.analyzer import DataAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    sample_data = {
        'signal_strength': [0.1, 0.5, 0.3, 0.7, 0.2, 0.4, 0.8]
    }

    try:
        analyzer = DataAnalyzer(sample_data)
        stats = analyzer.compute_statistics('signal_strength')
        print(f"Statistics: {stats}")
        analyzer.plot_histogram('signal_strength')
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
