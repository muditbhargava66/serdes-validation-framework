"""
USB4 Instrument Synchronization and Timing Control Module

This module provides synchronization and timing control for coordinated USB4 test
instrument operation with precise timing alignment and trigger coordination.

Features:
- Multi-instrument synchronization
- Precision timing control
- Trigger coordination
- Clock distribution
- Measurement alignment
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Union

from .controller import InstrumentController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USB4SyncMode(Enum):
    """USB4 synchronization modes"""

    MASTER_SLAVE = auto()
    DISTRIBUTED = auto()
    EXTERNAL_CLOCK = auto()


class USB4TriggerType(Enum):
    """USB4 trigger types"""

    SOFTWARE = auto()
    HARDWARE = auto()
    PATTERN = auto()
    TIMER = auto()


@dataclass
class USB4SyncConfig:
    """USB4 synchronization configuration"""

    sync_mode: USB4SyncMode
    master_instrument: str
    slave_instruments: List[str]
    trigger_type: USB4TriggerType
    clock_frequency: float
    trigger_delay: float = 0.0
    skew_compensation: bool = True

    def __post_init__(self) -> None:
        """Validate sync configuration"""
        assert isinstance(self.sync_mode, USB4SyncMode), f"Sync mode must be USB4SyncMode, got {type(self.sync_mode)}"
        assert isinstance(self.master_instrument, str), f"Master instrument must be string, got {type(self.master_instrument)}"
        assert isinstance(self.slave_instruments, list), f"Slave instruments must be list, got {type(self.slave_instruments)}"
        assert all(isinstance(s, str) for s in self.slave_instruments), "All slave instruments must be strings"
        assert isinstance(
            self.trigger_type, USB4TriggerType
        ), f"Trigger type must be USB4TriggerType, got {type(self.trigger_type)}"
        assert isinstance(self.clock_frequency, float), f"Clock frequency must be float, got {type(self.clock_frequency)}"
        assert isinstance(self.trigger_delay, float), f"Trigger delay must be float, got {type(self.trigger_delay)}"

        assert self.clock_frequency > 0, f"Clock frequency must be positive, got {self.clock_frequency}"
        assert self.trigger_delay >= 0, f"Trigger delay must be non-negative, got {self.trigger_delay}"


@dataclass
class USB4TimingMeasurement:
    """USB4 timing measurement result"""

    instrument_id: str
    trigger_time: float
    measurement_time: float
    skew_compensation: float
    synchronized: bool
    metadata: Dict[str, Union[str, float, bool]]

    def __post_init__(self) -> None:
        """Validate timing measurement"""
        assert isinstance(self.instrument_id, str), f"Instrument ID must be string, got {type(self.instrument_id)}"
        assert isinstance(self.trigger_time, float), f"Trigger time must be float, got {type(self.trigger_time)}"
        assert isinstance(self.measurement_time, float), f"Measurement time must be float, got {type(self.measurement_time)}"
        assert isinstance(self.skew_compensation, float), f"Skew compensation must be float, got {type(self.skew_compensation)}"
        assert isinstance(self.synchronized, bool), f"Synchronized must be bool, got {type(self.synchronized)}"
        assert isinstance(self.metadata, dict), f"Metadata must be dict, got {type(self.metadata)}"


class USB4SyncController:
    """Synchronization controller for USB4 test instruments"""

    def __init__(self, config: USB4SyncConfig) -> None:
        """
        Initialize USB4 sync controller

        Args:
            config: Synchronization configuration

        Raises:
            ValueError: If initialization fails
        """
        assert isinstance(config, USB4SyncConfig), f"Config must be USB4SyncConfig, got {type(config)}"

        self.config = config

        # Instrument controllers
        self.instruments: Dict[str, InstrumentController] = {}
        self.connected_instruments: Dict[str, bool] = {}

        # Timing state
        self.sync_established = False
        self.timing_measurements: List[USB4TimingMeasurement] = []
        self.skew_corrections: Dict[str, float] = {}

        # Initialize master controller
        self.master_controller = InstrumentController()

        logger.info(f"USB4 sync controller initialized with {len(config.slave_instruments)} slaves")

    def connect_instruments(self) -> bool:
        """
        Connect to all instruments in sync group

        Returns:
            True if all connections successful

        Raises:
            ValueError: If connection fails
        """
        try:
            # Connect to master instrument
            self.master_controller.connect_instrument(self.config.master_instrument)
            self.connected_instruments[self.config.master_instrument] = True
            logger.info(f"Connected to master instrument: {self.config.master_instrument}")

            # Connect to slave instruments
            for slave_resource in self.config.slave_instruments:
                controller = InstrumentController()
                controller.connect_instrument(slave_resource)
                self.instruments[slave_resource] = controller
                self.connected_instruments[slave_resource] = True
                logger.info(f"Connected to slave instrument: {slave_resource}")

            # Initialize synchronization
            self._initialize_synchronization()

            return True

        except Exception as e:
            logger.error(f"Failed to connect instruments: {e}")
            raise ValueError(f"Instrument connection failed: {e}")

    def disconnect_instruments(self) -> None:
        """Disconnect from all instruments"""
        try:
            # Disconnect master
            if self.config.master_instrument in self.connected_instruments:
                self.master_controller.disconnect_instrument(self.config.master_instrument)
                self.connected_instruments[self.config.master_instrument] = False

            # Disconnect slaves
            for slave_resource, controller in self.instruments.items():
                if slave_resource in self.connected_instruments:
                    controller.disconnect_instrument(slave_resource)
                    self.connected_instruments[slave_resource] = False

            self.sync_established = False
            logger.info("Disconnected from all instruments")

        except Exception as e:
            logger.error(f"Failed to disconnect instruments: {e}")

    def _initialize_synchronization(self) -> None:
        """Initialize instrument synchronization"""
        try:
            if self.config.sync_mode == USB4SyncMode.MASTER_SLAVE:
                self._setup_master_slave_sync()
            elif self.config.sync_mode == USB4SyncMode.DISTRIBUTED:
                self._setup_distributed_sync()
            elif self.config.sync_mode == USB4SyncMode.EXTERNAL_CLOCK:
                self._setup_external_clock_sync()

            # Configure trigger system
            self._configure_trigger_system()

            # Measure and compensate for skew if enabled
            if self.config.skew_compensation:
                self._measure_and_compensate_skew()

            self.sync_established = True
            logger.info("Instrument synchronization established")

        except Exception as e:
            raise ValueError(f"Synchronization initialization failed: {e}")

    def _setup_master_slave_sync(self) -> None:
        """Setup master-slave synchronization"""
        try:
            # Configure master instrument
            master_resource = self.config.master_instrument

            # Set master as clock source
            self.master_controller.send_command(master_resource, ":ROSCillator:SOURce INTernal")
            self.master_controller.send_command(master_resource, f":ROSCillator:FREQuency {self.config.clock_frequency}")

            # Enable clock output
            self.master_controller.send_command(master_resource, ":ROSCillator:OUTPut:STATe ON")

            # Configure trigger output
            self.master_controller.send_command(master_resource, ":TRIGger:OUTPut:STATe ON")

            # Configure slave instruments
            for slave_resource in self.config.slave_instruments:
                controller = self.instruments[slave_resource]

                # Set slave to external clock
                controller.send_command(slave_resource, ":ROSCillator:SOURce EXTernal")

                # Set trigger input
                controller.send_command(slave_resource, ":TRIGger:SOURce EXTernal")

                # Configure trigger slope
                controller.send_command(slave_resource, ":TRIGger:SLOPe POSitive")

        except Exception as e:
            raise ValueError(f"Master-slave sync setup failed: {e}")

    def _setup_distributed_sync(self) -> None:
        """Setup distributed synchronization"""
        try:
            # In distributed mode, all instruments use the same external reference
            reference_frequency = self.config.clock_frequency

            # Configure master
            self.master_controller.send_command(self.config.master_instrument, ":ROSCillator:SOURce EXTernal")
            self.master_controller.send_command(self.config.master_instrument, f":ROSCillator:FREQuency {reference_frequency}")

            # Configure slaves
            for slave_resource in self.config.slave_instruments:
                controller = self.instruments[slave_resource]
                controller.send_command(slave_resource, ":ROSCillator:SOURce EXTernal")
                controller.send_command(slave_resource, f":ROSCillator:FREQuency {reference_frequency}")

            # Setup distributed trigger
            self._setup_distributed_trigger()

        except Exception as e:
            raise ValueError(f"Distributed sync setup failed: {e}")

    def _setup_external_clock_sync(self) -> None:
        """Setup external clock synchronization"""
        try:
            # All instruments use external clock reference
            for instrument_resource in [self.config.master_instrument] + self.config.slave_instruments:
                if instrument_resource == self.config.master_instrument:
                    controller = self.master_controller
                else:
                    controller = self.instruments[instrument_resource]

                # Set external clock source
                controller.send_command(instrument_resource, ":ROSCillator:SOURce EXTernal")
                controller.send_command(instrument_resource, f":ROSCillator:FREQuency {self.config.clock_frequency}")

                # Configure for external trigger
                controller.send_command(instrument_resource, ":TRIGger:SOURce EXTernal")

        except Exception as e:
            raise ValueError(f"External clock sync setup failed: {e}")

    def _setup_distributed_trigger(self) -> None:
        """Setup distributed trigger system"""
        try:
            # Configure trigger distribution
            # Master generates trigger
            self.master_controller.send_command(self.config.master_instrument, ":TRIGger:OUTPut:STATe ON")

            # Slaves receive trigger
            for slave_resource in self.config.slave_instruments:
                controller = self.instruments[slave_resource]
                controller.send_command(slave_resource, ":TRIGger:SOURce EXTernal")
                controller.send_command(slave_resource, ":TRIGger:SLOPe POSitive")

        except Exception as e:
            raise ValueError(f"Distributed trigger setup failed: {e}")

    def _configure_trigger_system(self) -> None:
        """Configure trigger system based on trigger type"""
        try:
            if self.config.trigger_type == USB4TriggerType.SOFTWARE:
                # Software trigger - all instruments triggered by software command
                for instrument_resource in [self.config.master_instrument] + self.config.slave_instruments:
                    if instrument_resource == self.config.master_instrument:
                        controller = self.master_controller
                    else:
                        controller = self.instruments[instrument_resource]

                    controller.send_command(instrument_resource, ":TRIGger:SOURce BUS")

            elif self.config.trigger_type == USB4TriggerType.HARDWARE:
                # Hardware trigger already configured in sync setup
                pass

            elif self.config.trigger_type == USB4TriggerType.PATTERN:
                # Pattern trigger - trigger on specific USB4 patterns
                self._configure_pattern_trigger()

            elif self.config.trigger_type == USB4TriggerType.TIMER:
                # Timer trigger - periodic triggering
                self._configure_timer_trigger()

            # Set trigger delay if specified
            if self.config.trigger_delay > 0:
                self._configure_trigger_delay()

        except Exception as e:
            raise ValueError(f"Trigger system configuration failed: {e}")

    def _configure_pattern_trigger(self) -> None:
        """Configure pattern-based triggering"""
        try:
            # USB4 training sequence pattern
            usb4_pattern = "HLHLHLHLHLHLHLHL"  # Training sequence pattern

            # Configure master for pattern trigger
            self.master_controller.send_command(self.config.master_instrument, ":TRIGger:MODE PATTern")
            self.master_controller.send_command(self.config.master_instrument, f":TRIGger:PATTern:PATTern '{usb4_pattern}'")

            # Slaves will receive trigger from master
            for slave_resource in self.config.slave_instruments:
                controller = self.instruments[slave_resource]
                controller.send_command(slave_resource, ":TRIGger:SOURce EXTernal")

        except Exception as e:
            raise ValueError(f"Pattern trigger configuration failed: {e}")

    def _configure_timer_trigger(self) -> None:
        """Configure timer-based triggering"""
        try:
            # Calculate timer period for desired trigger rate
            trigger_period = 1.0 / 1000.0  # 1 kHz default

            # Configure master for timer trigger
            self.master_controller.send_command(self.config.master_instrument, ":TRIGger:SOURce TIMer")
            self.master_controller.send_command(self.config.master_instrument, f":TRIGger:TIMer {trigger_period}")

            # Slaves receive trigger from master
            for slave_resource in self.config.slave_instruments:
                controller = self.instruments[slave_resource]
                controller.send_command(slave_resource, ":TRIGger:SOURce EXTernal")

        except Exception as e:
            raise ValueError(f"Timer trigger configuration failed: {e}")

    def _configure_trigger_delay(self) -> None:
        """Configure trigger delay compensation"""
        try:
            # Apply delay to slave instruments to compensate for propagation
            for slave_resource in self.config.slave_instruments:
                controller = self.instruments[slave_resource]
                controller.send_command(slave_resource, f":TRIGger:DELay {self.config.trigger_delay}")

        except Exception as e:
            raise ValueError(f"Trigger delay configuration failed: {e}")

    def _measure_and_compensate_skew(self) -> None:
        """Measure inter-instrument skew and apply compensation"""
        try:
            logger.info("Measuring inter-instrument skew...")

            # Generate test signal on master
            self.master_controller.send_command(self.config.master_instrument, ":OUTPut:TEST:STATe ON")

            # Measure timing on all instruments
            reference_time = None

            for instrument_resource in [self.config.master_instrument] + self.config.slave_instruments:
                if instrument_resource == self.config.master_instrument:
                    controller = self.master_controller
                else:
                    controller = self.instruments[instrument_resource]

                # Trigger measurement
                controller.send_command(instrument_resource, ":TRIGger:FORCe")

                # Wait for trigger
                time.sleep(0.1)

                # Get trigger timestamp
                trigger_time_str = controller.query_instrument(instrument_resource, ":TRIGger:TIME?")
                trigger_time = float(trigger_time_str)

                if reference_time is None:
                    reference_time = trigger_time

                # Calculate skew
                skew = trigger_time - reference_time
                self.skew_corrections[instrument_resource] = -skew  # Negative to compensate

                logger.info(f"Instrument {instrument_resource}: skew = {skew*1e9:.2f} ns")

            # Apply skew corrections
            for instrument_resource, correction in self.skew_corrections.items():
                if instrument_resource != self.config.master_instrument:
                    controller = self.instruments[instrument_resource]
                    controller.send_command(instrument_resource, f":TRIGger:DELay {correction}")

            # Turn off test signal
            self.master_controller.send_command(self.config.master_instrument, ":OUTPut:TEST:STATe OFF")

            logger.info("Skew compensation applied")

        except Exception as e:
            logger.warning(f"Skew compensation failed: {e}")

    def synchronize_measurement(self, measurement_duration: float, pre_trigger_delay: float = 0.0) -> List[USB4TimingMeasurement]:
        """
        Perform synchronized measurement across all instruments

        Args:
            measurement_duration: Duration of measurement in seconds
            pre_trigger_delay: Delay before triggering in seconds

        Returns:
            List of timing measurements from all instruments

        Raises:
            ValueError: If synchronized measurement fails
        """
        assert isinstance(measurement_duration, float), f"Measurement duration must be float, got {type(measurement_duration)}"
        assert isinstance(pre_trigger_delay, float), f"Pre-trigger delay must be float, got {type(pre_trigger_delay)}"
        assert measurement_duration > 0, f"Measurement duration must be positive, got {measurement_duration}"
        assert pre_trigger_delay >= 0, f"Pre-trigger delay must be non-negative, got {pre_trigger_delay}"

        if not self.sync_established:
            raise ValueError("Synchronization not established")

        try:
            measurements = []

            # Pre-trigger delay
            if pre_trigger_delay > 0:
                time.sleep(pre_trigger_delay)

            # Arm all instruments
            self._arm_all_instruments()

            # Generate trigger
            trigger_time = self._generate_synchronized_trigger()

            # Wait for measurement completion
            time.sleep(measurement_duration)

            # Collect measurements from all instruments
            for instrument_resource in [self.config.master_instrument] + self.config.slave_instruments:
                if instrument_resource == self.config.master_instrument:
                    controller = self.master_controller
                else:
                    controller = self.instruments[instrument_resource]

                # Get measurement timestamp
                measurement_time_str = controller.query_instrument(instrument_resource, ":SYSTem:TIME?")
                measurement_time = float(measurement_time_str)

                # Get skew compensation
                skew_compensation = self.skew_corrections.get(instrument_resource, 0.0)

                # Check synchronization status
                sync_status = controller.query_instrument(instrument_resource, ":TRIGger:STATus?")
                synchronized = "TRIGGERED" in sync_status.upper()

                # Create measurement record
                metadata = {
                    "sync_mode": self.config.sync_mode.name,
                    "trigger_type": self.config.trigger_type.name,
                    "measurement_duration": measurement_duration,
                    "pre_trigger_delay": pre_trigger_delay,
                }

                measurement = USB4TimingMeasurement(
                    instrument_id=instrument_resource,
                    trigger_time=trigger_time,
                    measurement_time=measurement_time,
                    skew_compensation=skew_compensation,
                    synchronized=synchronized,
                    metadata=metadata,
                )

                measurements.append(measurement)

            self.timing_measurements.extend(measurements)
            logger.info(f"Synchronized measurement completed with {len(measurements)} instruments")

            return measurements

        except Exception as e:
            logger.error(f"Synchronized measurement failed: {e}")
            raise ValueError(f"Synchronized measurement failed: {e}")

    def _arm_all_instruments(self) -> None:
        """Arm all instruments for synchronized measurement"""
        try:
            # Arm master
            self.master_controller.send_command(self.config.master_instrument, ":INITiate")

            # Arm slaves
            for slave_resource in self.config.slave_instruments:
                controller = self.instruments[slave_resource]
                controller.send_command(slave_resource, ":INITiate")

            # Wait for all instruments to be ready
            time.sleep(0.1)

        except Exception as e:
            raise ValueError(f"Instrument arming failed: {e}")

    def _generate_synchronized_trigger(self) -> float:
        """Generate synchronized trigger signal"""
        try:
            trigger_time = time.time()

            if self.config.trigger_type == USB4TriggerType.SOFTWARE:
                # Software trigger to all instruments
                self.master_controller.send_command(self.config.master_instrument, "*TRG")

                for slave_resource in self.config.slave_instruments:
                    controller = self.instruments[slave_resource]
                    controller.send_command(slave_resource, "*TRG")

            elif self.config.trigger_type == USB4TriggerType.HARDWARE:
                # Hardware trigger generated by master
                self.master_controller.send_command(self.config.master_instrument, ":TRIGger:FORCe")

            elif self.config.trigger_type == USB4TriggerType.PATTERN:
                # Pattern trigger - wait for pattern detection
                # Master will trigger when pattern is detected
                pass

            elif self.config.trigger_type == USB4TriggerType.TIMER:
                # Timer trigger - automatic triggering
                pass

            return trigger_time

        except Exception as e:
            raise ValueError(f"Trigger generation failed: {e}")

    def get_synchronization_status(self) -> Dict[str, Union[bool, float, str]]:
        """
        Get synchronization status

        Returns:
            Dictionary with synchronization status information

        Raises:
            ValueError: If status query fails
        """
        try:
            status = {
                "sync_established": self.sync_established,
                "sync_mode": self.config.sync_mode.name,
                "trigger_type": self.config.trigger_type.name,
                "master_instrument": self.config.master_instrument,
                "slave_count": len(self.config.slave_instruments),
                "clock_frequency": self.config.clock_frequency,
                "trigger_delay": self.config.trigger_delay,
                "skew_compensation_enabled": self.config.skew_compensation,
            }

            # Instrument connection status
            connected_count = sum(self.connected_instruments.values())
            total_instruments = len(self.connected_instruments)
            status["instruments_connected"] = f"{connected_count}/{total_instruments}"

            # Skew correction values
            if self.skew_corrections:
                max_skew = max(abs(skew) for skew in self.skew_corrections.values())
                status["max_skew_ns"] = max_skew * 1e9
                status["skew_corrections"] = {k: v * 1e9 for k, v in self.skew_corrections.items()}

            # Timing measurement statistics
            if self.timing_measurements:
                status["total_measurements"] = len(self.timing_measurements)
                sync_success_rate = sum(1 for m in self.timing_measurements if m.synchronized) / len(self.timing_measurements)
                status["sync_success_rate"] = sync_success_rate

            return status

        except Exception as e:
            logger.error(f"Synchronization status query failed: {e}")
            raise ValueError(f"Status query failed: {e}")

    def reset_synchronization(self) -> bool:
        """
        Reset synchronization system

        Returns:
            True if reset successful

        Raises:
            ValueError: If reset fails
        """
        try:
            # Stop all measurements
            for instrument_resource in [self.config.master_instrument] + self.config.slave_instruments:
                if instrument_resource == self.config.master_instrument:
                    controller = self.master_controller
                else:
                    controller = self.instruments[instrument_resource]

                controller.send_command(instrument_resource, ":ABORt")

            # Clear timing measurements
            self.timing_measurements = []
            self.skew_corrections = {}

            # Re-initialize synchronization
            self._initialize_synchronization()

            logger.info("Synchronization system reset")
            return True

        except Exception as e:
            logger.error(f"Synchronization reset failed: {e}")
            raise ValueError(f"Synchronization reset failed: {e}")


__all__ = [
    "USB4SyncMode",
    "USB4TriggerType",
    "USB4SyncConfig",
    "USB4TimingMeasurement",
    "USB4SyncController",
]
