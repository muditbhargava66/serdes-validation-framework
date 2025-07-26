import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnalyzer:
    def __init__(self, data):
        self.data = pd.DataFrame(data)
        logger.info("DataAnalyzer initialized")

    def plot_histogram(self, column, bins=10):
        if column not in self.data.columns:
            logger.error(f"Column {column} not found in data")
            raise ValueError(f"Column {column} not found in data")
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[column], bins=bins, kde=True)
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()
            logger.info(f"Histogram of {column} plotted")
        except Exception as e:
            logger.error(f"Failed to plot histogram of {column}: {e}")
            raise

    def compute_statistics(self, column):
        if column not in self.data.columns:
            logger.error(f"Column {column} not found in data")
            raise ValueError(f"Column {column} not found in data")
        try:
            stats = self.data[column].describe()
            logger.info(f"Statistics for {column}: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to compute statistics for {column}: {e}")
            raise


if __name__ == "__main__":
    sample_data = {"signal_strength": [0.1, 0.5, 0.3, 0.7, 0.2, 0.4, 0.8]}
    analyzer = DataAnalyzer(sample_data)
    analyzer.plot_histogram("signal_strength")
    stats = analyzer.compute_statistics("signal_strength")
    print(stats)
