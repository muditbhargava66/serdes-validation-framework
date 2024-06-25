# Usage Guide

This guide provides instructions and examples for using the SerDes Validation Framework.

## Data Collection

### Collecting Data from an Instrument

```python
from src.serdes_validation_framework.data_collection.data_collector import DataCollector

dc = DataCollector()
dc.connect_instrument('GPIB::1::INSTR')
data = dc.collect_data('GPIB::1::INSTR', '*IDN?')
print(f"Data collected: {data}")
dc.disconnect_instrument('GPIB::1::INSTR')
```

## Data Analysis

### Analyzing Collected Data

```python
from src.serdes_validation_framework.data_analysis.analyzer import DataAnalyzer

sample_data = {
    'signal_strength': [0.1, 0.5, 0.3, 0.7, 0.2, 0.4, 0.8]
}

analyzer = DataAnalyzer(sample_data)
stats = analyzer.compute_statistics('signal_strength')
print(f"Statistics: {stats}")
analyzer.plot_histogram('signal_strength')
```

## Instrument Control

### Sending Commands to an Instrument

```python
from src.serdes_validation_framework.instrument_control.controller import InstrumentController

ic = InstrumentController()
ic.connect_instrument('GPIB::2::INSTR')
ic.send_command('GPIB::2::INSTR', '*RST')
response = ic.query_instrument('GPIB::2::INSTR', '*IDN?')
print(f"Instrument response: {response}")
ic.disconnect_instrument('GPIB::2::INSTR')
```

## Test Sequence

### Running a Test Sequence

```python
from src.serdes_validation_framework.test_sequence.sequencer import TestSequencer

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
ts.cleanup(instruments)
```

For more detailed examples, refer to the [examples directory](../examples/).

---