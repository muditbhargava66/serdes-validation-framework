import logging
from src.serdes_validation_framework.test_sequence.sequencer import TestSequencer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        ts = TestSequencer()
        instruments = ['GPIB::1::INSTR', 'GPIB::2::INSTR']
        ts.setup_instruments(instruments)

        sequence = [
            {'resource': 'GPIB::1::INSTR', 'command': '*RST', 'action': 'send'},
            {'resource': 'GPIB::2::INSTR', 'command': '*IDN?', 'action': 'query'}
        ]
        results = ts.run_sequence(sequence)
        print(f"Sequence results: {results}")

        stats = ts.collect_and_analyze_data('GPIB::2::INSTR', 'MEASure:VOLTage:DC?', 'voltage')
        print(f"Data statistics: {stats}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        ts.cleanup(instruments)

if __name__ == "__main__":
    main()
