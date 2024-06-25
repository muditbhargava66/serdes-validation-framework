import pyvisa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InstrumentController:
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

    def send_command(self, resource_name, command):
        if resource_name not in self.instruments:
            logger.error(f"Instrument {resource_name} not connected")
            raise ValueError(f"Instrument {resource_name} not connected")
        try:
            instrument = self.instruments[resource_name]
            instrument.write(command)
            logger.info(f"Command sent to {resource_name}: {command}")
        except Exception as e:
            logger.error(f"Failed to send command to {resource_name}: {e}")
            raise

    def query_instrument(self, resource_name, query):
        if resource_name not in self.instruments:
            logger.error(f"Instrument {resource_name} not connected")
            raise ValueError(f"Instrument {resource_name} not connected")
        try:
            instrument = self.instruments[resource_name]
            response = instrument.query(query)
            logger.info(f"Query sent to {resource_name}: {query}")
            logger.info(f"Response from {resource_name}: {response}")
            return response
        except Exception as e:
            logger.error(f"Failed to query {resource_name}: {e}")
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
    ic = InstrumentController()
    ic.connect_instrument('GPIB::2::INSTR')
    ic.send_command('GPIB::2::INSTR', '*RST')
    response = ic.query_instrument('GPIB::2::INSTR', '*IDN?')
    print(response)
    ic.disconnect_instrument('GPIB::2::INSTR')
