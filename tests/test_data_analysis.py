import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.serdes_validation_framework.data_analysis.analyzer import DataAnalyzer


class TestDataAnalyzer(unittest.TestCase):

    def setUp(self):
        self.sample_data = {'signal_strength': [0.1, 0.5, 0.3, 0.7, 0.2, 0.4, 0.8]}
        self.analyzer = DataAnalyzer(self.sample_data)

    def test_compute_statistics(self):
        stats = self.analyzer.compute_statistics('signal_strength')
        expected_stats = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8], name='signal_strength').describe()
        pd.testing.assert_series_equal(stats, expected_stats)

    def test_compute_statistics_invalid_column(self):
        with self.assertRaises(ValueError):
            self.analyzer.compute_statistics('invalid_column')

    def test_plot_histogram(self):
        try:
            self.analyzer.plot_histogram('signal_strength')
        except Exception as e:
            self.fail(f"plot_histogram raised an exception: {e}")

    def test_plot_histogram_invalid_column(self):
        with self.assertRaises(ValueError):
            self.analyzer.plot_histogram('invalid_column')

if __name__ == '__main__':
    unittest.main()
