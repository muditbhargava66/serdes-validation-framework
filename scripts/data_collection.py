import logging
from src.serdes_validation_framework.data_collection.data_collector import DataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        dc = DataCollector()
        dc.connect_instrument('GPIB::1::INSTR')
        data = dc.collect_data('GPIB::1::INSTR', '*IDN?')
        print(f"Data collected: {data}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        dc.disconnect_instrument('GPIB::1::INSTR')

if __name__ == "__main__":
    main()
