# Getting Started

This tutorial will guide you through the process of setting up and using the SerDes Validation Framework.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/muditbhargava66/serdes-validation-framework.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd serdes-validation-framework
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Basic Usage

### Data Collection

1. **Create a data collector instance:**

    ```python
    from src.serdes_validation_framework.data_collection.data_collector import DataCollector

    dc = DataCollector()
    dc.connect_instrument('GPIB::1::INSTR')
    data = dc.collect_data('GPIB::1::INSTR', '*IDN?')
    print(f"Data collected: {data}")
    dc.disconnect_instrument('GPIB::1::INSTR')
    ```

### Data Analysis

1. **Create a data analyzer instance:**

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

### Instrument Control

1. **Create an instrument controller instance:**

    ```python
    from src.serdes_validation_framework.instrument_control.controller import InstrumentController

    ic = InstrumentController()
    ic.connect_instrument('GPIB::2::INSTR')
    ic.send_command('GPIB::2::INSTR', '*RST')
    response = ic.query_instrument('GPIB::2::INSTR', '*IDN?')
    print(f"Instrument response: {response}")
    ic.disconnect_instrument('GPIB::2::INSTR')
    ```

### Test Sequence

1. **Create a test sequencer instance:**

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

Refer to the [installation](../INSTALL.md) and [usage](../USAGE.md) documentation for more details.

---