import logging
from src.serdes_validation_framework.instrument_control.controller import InstrumentController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        ic = InstrumentController()
        ic.connect_instrument('GPIB::2::INSTR')
        ic.send_command('GPIB::2::INSTR', '*RST')
        response = ic.query_instrument('GPIB::2::INSTR', '*IDN?')
        print(f"Instrument response: {response}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        ic.disconnect_instrument('GPIB::2::INSTR')

if __name__ == "__main__":
    main()
