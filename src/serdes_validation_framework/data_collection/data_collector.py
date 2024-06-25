import pyvisa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.rm = pyvisa.ResourceManager('@py')
        self.instruments = {}

    def connect_instrument(self, resource_name):
        try:
            instrument = self.rm.open_resource(resource_name)
            self.instruments[resource_name] = instrument
            logger.info(f"Connected to {resource_name}")
        except Exception as e:
            logger.error(f"Failed to connect to {resource_name}: {e}")
            raise

    def collect_data(self, resource_name, command):
        if resource_name not in self.instruments:
            logger.error(f"Instrument {resource_name} not connected")
            raise ValueError(f"Instrument {resource_name} not connected")
        try:
            instrument = self.instruments[resource_name]
            response = instrument.query(command)
            logger.info(f"Data collected from {resource_name}: {response}")
            return response
        except Exception as e:
            logger.error(f"Failed to collect data from {resource_name}: {e}")
            raise

    def disconnect_instrument(self, resource_name):
        if resource_name in self.instruments:
            try:
                self.instruments[resource_name].close()
                del self.instruments[resource_name]
                logger.info(f"Disconnected from {resource_name}")
            except Exception as e:
                logger.error(f"Failed to disconnect from {resource_name}: {e}")
                raise
        else:
            raise ValueError(f"Instrument {resource_name} not connected")

if __name__ == "__main__":
    dc = DataCollector()
    dc.connect_instrument('GPIB::1::INSTR')
    data = dc.collect_data('GPIB::1::INSTR', '*IDN?')
    print(data)
    dc.disconnect_instrument('GPIB::1::INSTR')
