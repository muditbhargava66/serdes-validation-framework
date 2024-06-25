import logging
from src.serdes_validation_framework.data_collection.data_collector import DataCollector
from src.serdes_validation_framework.data_analysis.analyzer import DataAnalyzer
from src.serdes_validation_framework.instrument_control.controller import InstrumentController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSequencer:
    def __init__(self):
        self.data_collector = DataCollector()
        self.instrument_controller = InstrumentController()

    def setup_instruments(self, resource_names):
        for resource_name in resource_names:
            self.instrument_controller.connect_instrument(resource_name)

    def run_sequence(self, sequence):
        results = {}
        for step in sequence:
            resource = step['resource']
            command = step['command']
            action = step['action']
            if action == 'send':
                self.instrument_controller.send_command(resource, command)
            elif action == 'query':
                response = self.instrument_controller.query_instrument(resource, command)
                results[resource] = response
        return results

    def collect_and_analyze_data(self, resource_name, query, column):
        data = self.data_collector.collect_data(resource_name, query)
        analyzer = DataAnalyzer({column: data.split()})
        stats = analyzer.compute_statistics(column)
        analyzer.plot_histogram(column)
        return stats

    def cleanup(self, resource_names):
        for resource_name in resource_names:
            self.instrument_controller.disconnect_instrument(resource_name)

if __name__ == "__main__":
    ts = TestSequencer()
    instruments = ['GPIB::1::INSTR', 'GPIB::2::INSTR']
    ts.setup_instruments(instruments)
    sequence = [
        {'resource': 'GPIB::1::INSTR', 'command': '*RST', 'action': 'send'},
        {'resource': 'GPIB::2::INSTR', 'command': '*IDN?', 'action': 'query'}
    ]
    results = ts.run_sequence(sequence)
    print(results)
    stats = ts.collect_and_analyze_data('GPIB::2::INSTR', 'MEASure:VOLTage:DC?', 'voltage')
    print(stats)
    ts.cleanup(instruments)
