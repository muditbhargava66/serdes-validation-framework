"""
USB4 Link Training and State Management

This module implements USB4 link training sequences, state management, and error recovery
mechanisms according to the USB4 v2.0 specification.

Features:
- Complete USB4 link training state machine
- Power state transition monitoring (U0-U3)
- Speed and width negotiation validation
- Error recovery mechanism testing
- Link training sequence validation
- Comprehensive error logging and analysis
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .base import (
    USB4Config,
    USB4LinkTrainer,
)
from .constants import (
    USB4ErrorType,
    USB4LinkState,
    USB4SignalMode,
    USB4Specs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USB4TrainingState(Enum):
    """USB4 link training state machine states"""

    RESET = auto()  # Initial reset state
    DETECT = auto()  # Link detection
    POLLING = auto()  # Polling for link partner
    CONFIGURATION = auto()  # Link configuration
    RECOVERY = auto()  # Error recovery
    L0 = auto()  # Active state (equivalent to U0)
    L1 = auto()  # Standby state (equivalent to U1)
    L2 = auto()  # Sleep state (equivalent to U2)
    L3 = auto()  # Suspend state (equivalent to U3)
    DISABLED = auto()  # Link disabled
    LOOPBACK = auto()  # Loopback test mode
    HOT_RESET = auto()  # Hot reset state


class USB4TrainingEvent(Enum):
    """USB4 link training events"""

    LINK_UP = auto()
    LINK_DOWN = auto()
    SPEED_CHANGE = auto()
    WIDTH_CHANGE = auto()
    ERROR_DETECTED = auto()
    RECOVERY_SUCCESS = auto()
    RECOVERY_FAILURE = auto()
    POWER_STATE_CHANGE = auto()
    TIMEOUT = auto()
    USER_REQUEST = auto()


@dataclass
class USB4TrainingConfig(USB4Config):
    """USB4 link training configuration"""

    max_training_time: float = 100.0e-3  # Maximum training time (100 ms)
    training_timeout: float = 1.0  # Training timeout (s)
    max_retries: int = 3  # Maximum training retries
    enable_recovery: bool = True  # Enable error recovery
    enable_power_management: bool = True  # Enable power state management
    target_mode: USB4SignalMode = USB4SignalMode.GEN2X2
    enable_loopback: bool = False  # Enable loopback testing
    monitor_power_states: bool = True  # Monitor power state transitions


@dataclass
class USB4StateTransition:
    """USB4 state transition record"""

    from_state: USB4TrainingState
    to_state: USB4TrainingState
    event: USB4TrainingEvent
    timestamp: float
    duration: float
    success: bool
    error_info: Optional[str] = None


@dataclass
class USB4TrainingResults:
    """USB4 link training results"""

    training_time: float
    final_state: USB4TrainingState
    final_link_state: USB4LinkState
    negotiated_mode: USB4SignalMode
    negotiated_lanes: int
    error_count: int
    retry_count: int
    convergence_status: bool
    state_transitions: List[USB4StateTransition] = field(default_factory=list)
    power_measurements: Dict[USB4LinkState, float] = field(default_factory=dict)
    training_sequence_valid: bool = True
    recovery_attempts: int = 0
    recovery_success_rate: float = 0.0


@dataclass
class USB4NegotiationResults:
    """USB4 speed and width negotiation results"""

    requested_mode: USB4SignalMode
    negotiated_mode: USB4SignalMode
    requested_lanes: int
    negotiated_lanes: int
    negotiation_time: float
    fallback_occurred: bool
    fallback_reason: Optional[str] = None


@dataclass
class USB4RecoveryResults:
    """USB4 error recovery results"""

    error_type: USB4ErrorType
    recovery_time: float
    recovery_success: bool
    recovery_attempts: int
    final_state: USB4TrainingState
    error_details: Dict[str, Any] = field(default_factory=dict)


class USB4LinkTraining(USB4LinkTrainer):
    """USB4 link training state machine implementation"""

    def __init__(self, config: USB4TrainingConfig):
        """
        Initialize USB4 link training

        Args:
            config: USB4 training configuration
        """
        super().__init__(config)
        self.config: USB4TrainingConfig = config
        self.specs = USB4Specs()

        # State machine
        self.current_state = USB4TrainingState.RESET
        self.previous_state = USB4TrainingState.RESET
        self.current_link_state = USB4LinkState.U3

        # Training state
        self.training_start_time = 0.0
        self.state_transitions: List[USB4StateTransition] = []
        self.error_count = 0
        self.retry_count = 0

        # Negotiation state
        self.negotiated_mode = USB4SignalMode.GEN2X2
        self.negotiated_lanes = 2

        # Power monitoring
        self.power_measurements: Dict[USB4LinkState, float] = {}
        self.power_transition_times: Dict[Tuple[USB4LinkState, USB4LinkState], float] = {}

        # Recovery state
        self.recovery_attempts = 0
        self.recovery_successes = 0

    def initialize(self) -> bool:
        """
        Initialize link training component

        Returns:
            True if initialization successful
        """
        try:
            self.validate_config()
            self.specs.validate_all()
            self._reset_state_machine()
            self._initialized = True
            self.logger.info("USB4 link training initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize USB4 link training: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up link training resources"""
        self._reset_state_machine()
        self.state_transitions.clear()
        self.power_measurements.clear()
        self.power_transition_times.clear()
        self._initialized = False
        self.logger.info("USB4 link training cleaned up")

    def execute_training(self) -> USB4TrainingResults:
        """
        Execute complete USB4 link training sequence

        Returns:
            Training results
        """
        if not self.is_initialized:
            raise RuntimeError("Link training not initialized")

        self.logger.info("Starting USB4 link training sequence")
        self.training_start_time = time.time()
        self._reset_state_machine()

        try:
            # Execute state machine
            success = self._run_training_state_machine()

            # Calculate results
            training_time = time.time() - self.training_start_time

            results = USB4TrainingResults(
                training_time=training_time,
                final_state=self.current_state,
                final_link_state=self.current_link_state,
                negotiated_mode=self.negotiated_mode,
                negotiated_lanes=self.negotiated_lanes,
                error_count=self.error_count,
                retry_count=self.retry_count,
                convergence_status=success,
                state_transitions=self.state_transitions.copy(),
                power_measurements=self.power_measurements.copy(),
                recovery_attempts=self.recovery_attempts,
                recovery_success_rate=self._calculate_recovery_success_rate(),
            )

            self.logger.info(f"USB4 link training completed in {training_time:.3f}s, " f"final state: {self.current_state.name}")
            return results

        except Exception as e:
            self.logger.error(f"USB4 link training failed: {e}")
            raise

    def monitor_link_state(self) -> USB4LinkState:
        """
        Monitor current USB4 link state

        Returns:
            Current link state
        """
        return self.current_link_state

    def validate_training_sequence(self, sequence_data: npt.NDArray) -> bool:
        """
        Validate link training sequence data

        Args:
            sequence_data: Training sequence data to validate

        Returns:
            True if sequence is valid
        """
        if sequence_data.size == 0:
            self.logger.warning("Empty training sequence data")
            return False

        # Validate sequence patterns (simplified implementation)
        # In real implementation, this would check for specific USB4 training patterns
        try:
            # Check for minimum sequence length
            min_length = int(self.specs.MAX_TRAINING_TIME * 1e6)  # Convert to samples
            if sequence_data.size < min_length:
                self.logger.warning(f"Training sequence too short: {sequence_data.size} < {min_length}")
                return False

            # Check for valid signal levels
            if np.max(np.abs(sequence_data)) > 2.0:  # Reasonable signal level check
                self.logger.warning("Training sequence signal levels out of range")
                return False

            # Check for signal activity
            if np.std(sequence_data) < 0.1:  # Minimum signal activity
                self.logger.warning("Training sequence appears to be inactive")
                return False

            self.logger.info("Training sequence validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Training sequence validation failed: {e}")
            return False

    def monitor_state_transitions(self, duration: float) -> List[USB4StateTransition]:
        """
        Monitor USB4 power state transitions for specified duration

        Args:
            duration: Monitoring duration in seconds

        Returns:
            List of state transitions observed
        """
        if not self.config.monitor_power_states:
            self.logger.warning("Power state monitoring disabled")
            return []

        self.logger.info(f"Monitoring USB4 state transitions for {duration:.3f}s")

        transitions = []
        start_time = time.time()
        last_state = self.current_link_state

        while (time.time() - start_time) < duration:
            current_state = self._simulate_power_state_monitoring()

            if current_state != last_state:
                transition = USB4StateTransition(
                    from_state=self._link_state_to_training_state(last_state),
                    to_state=self._link_state_to_training_state(current_state),
                    event=USB4TrainingEvent.POWER_STATE_CHANGE,
                    timestamp=time.time(),
                    duration=0.0,  # Will be calculated
                    success=True,
                )
                transitions.append(transition)
                last_state = current_state

            time.sleep(0.01)  # 10ms monitoring interval

        self.logger.info(f"Monitored {len(transitions)} state transitions")
        return transitions

    def validate_speed_negotiation(self, requested_mode: USB4SignalMode) -> USB4NegotiationResults:
        """
        Validate USB4 speed negotiation process

        Args:
            requested_mode: Requested USB4 signal mode

        Returns:
            Negotiation results
        """
        self.logger.info(f"Validating speed negotiation for mode: {requested_mode.name}")

        start_time = time.time()
        negotiated_mode = requested_mode
        fallback_occurred = False
        fallback_reason = None

        # Simulate negotiation process
        if requested_mode == USB4SignalMode.GEN3X2:
            # Simulate potential fallback to Gen2
            if np.random.random() < 0.1:  # 10% chance of fallback
                negotiated_mode = USB4SignalMode.GEN2X2
                fallback_occurred = True
                fallback_reason = "Signal integrity insufficient for Gen3"

        negotiation_time = time.time() - start_time

        # Update internal state
        self.negotiated_mode = negotiated_mode

        results = USB4NegotiationResults(
            requested_mode=requested_mode,
            negotiated_mode=negotiated_mode,
            requested_lanes=2,
            negotiated_lanes=2,  # USB4 always uses 2 lanes
            negotiation_time=negotiation_time,
            fallback_occurred=fallback_occurred,
            fallback_reason=fallback_reason,
        )

        self.logger.info(f"Speed negotiation completed: {requested_mode.name} -> {negotiated_mode.name}")
        return results

    def test_error_recovery(self, error_type: USB4ErrorType) -> USB4RecoveryResults:
        """
        Test USB4 error recovery mechanisms

        Args:
            error_type: Type of error to inject and recover from

        Returns:
            Recovery test results
        """
        if not self.config.enable_recovery:
            raise RuntimeError("Error recovery testing disabled")

        self.logger.info(f"Testing error recovery for: {error_type.name}")

        start_time = time.time()
        recovery_success = False
        attempts = 0
        max_attempts = self.config.max_retries

        # Inject error and attempt recovery
        self._inject_error(error_type)

        while attempts < max_attempts and not recovery_success:
            attempts += 1
            recovery_success = self._attempt_recovery(error_type)

            if not recovery_success:
                time.sleep(0.1)  # Brief delay between attempts

        recovery_time = time.time() - start_time

        # Update recovery statistics
        self.recovery_attempts += attempts
        if recovery_success:
            self.recovery_successes += 1

        final_state = USB4TrainingState.L0 if recovery_success else USB4TrainingState.RECOVERY

        results = USB4RecoveryResults(
            error_type=error_type,
            recovery_time=recovery_time,
            recovery_success=recovery_success,
            recovery_attempts=attempts,
            final_state=final_state,
            error_details={
                "error_injected_at": start_time,
                "recovery_method": self._get_recovery_method(error_type),
                "total_recovery_attempts": self.recovery_attempts,
                "total_recovery_successes": self.recovery_successes,
            },
        )

        self.logger.info(
            f"Error recovery test completed: success={recovery_success}, " f"attempts={attempts}, time={recovery_time:.3f}s"
        )
        return results

    def _reset_state_machine(self) -> None:
        """Reset the link training state machine"""
        self.current_state = USB4TrainingState.RESET
        self.previous_state = USB4TrainingState.RESET
        self.current_link_state = USB4LinkState.U3
        self.state_transitions.clear()
        self.error_count = 0
        self.retry_count = 0

    def _run_training_state_machine(self) -> bool:
        """
        Run the complete USB4 link training state machine

        Returns:
            True if training completed successfully
        """
        timeout = self.config.training_timeout
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            if self._process_current_state():
                if self.current_state == USB4TrainingState.L0:
                    return True  # Training completed successfully

            time.sleep(0.001)  # 1ms state machine cycle

        self.logger.error("USB4 link training timed out")
        return False

    def _process_current_state(self) -> bool:
        """
        Process current state machine state

        Returns:
            True if state processing successful
        """
        state_start_time = time.time()
        success = True
        next_state = self.current_state

        try:
            if self.current_state == USB4TrainingState.RESET:
                next_state = self._process_reset_state()
            elif self.current_state == USB4TrainingState.DETECT:
                next_state = self._process_detect_state()
            elif self.current_state == USB4TrainingState.POLLING:
                next_state = self._process_polling_state()
            elif self.current_state == USB4TrainingState.CONFIGURATION:
                next_state = self._process_configuration_state()
            elif self.current_state == USB4TrainingState.RECOVERY:
                next_state = self._process_recovery_state()
            elif self.current_state == USB4TrainingState.L0:
                next_state = self._process_l0_state()
            else:
                # Handle other states as needed
                pass

        except Exception as e:
            self.logger.error(f"Error processing state {self.current_state.name}: {e}")
            success = False
            next_state = USB4TrainingState.RECOVERY
            self.error_count += 1

        # Record state transition
        if next_state != self.current_state:
            duration = time.time() - state_start_time
            transition = USB4StateTransition(
                from_state=self.current_state,
                to_state=next_state,
                event=USB4TrainingEvent.LINK_UP if success else USB4TrainingEvent.ERROR_DETECTED,
                timestamp=time.time(),
                duration=duration,
                success=success,
            )
            self.state_transitions.append(transition)

            self.previous_state = self.current_state
            self.current_state = next_state

        return success

    def _process_reset_state(self) -> USB4TrainingState:
        """Process RESET state"""
        self.logger.debug("Processing RESET state")
        # Initialize hardware, reset counters
        time.sleep(0.001)  # Simulate reset time
        return USB4TrainingState.DETECT

    def _process_detect_state(self) -> USB4TrainingState:
        """Process DETECT state"""
        self.logger.debug("Processing DETECT state")
        # Detect link partner presence
        time.sleep(0.005)  # Simulate detection time
        # Assume link partner detected
        return USB4TrainingState.POLLING

    def _process_polling_state(self) -> USB4TrainingState:
        """Process POLLING state"""
        self.logger.debug("Processing POLLING state")
        # Poll for link partner response
        time.sleep(0.010)  # Simulate polling time
        # Assume successful polling
        return USB4TrainingState.CONFIGURATION

    def _process_configuration_state(self) -> USB4TrainingState:
        """Process CONFIGURATION state"""
        self.logger.debug("Processing CONFIGURATION state")
        # Configure link parameters
        time.sleep(0.020)  # Simulate configuration time

        # Perform speed and width negotiation
        negotiation_result = self.validate_speed_negotiation(self.config.target_mode)
        self.negotiated_mode = negotiation_result.negotiated_mode
        self.negotiated_lanes = negotiation_result.negotiated_lanes

        # Update link state to active
        self.current_link_state = USB4LinkState.U0

        return USB4TrainingState.L0

    def _process_recovery_state(self) -> USB4TrainingState:
        """Process RECOVERY state"""
        self.logger.debug("Processing RECOVERY state")

        if self.retry_count >= self.config.max_retries:
            self.logger.error("Maximum recovery retries exceeded")
            return USB4TrainingState.DISABLED

        self.retry_count += 1
        time.sleep(0.050)  # Simulate recovery time

        # Attempt recovery
        if np.random.random() < 0.8:  # 80% recovery success rate
            self.logger.info(f"Recovery successful (attempt {self.retry_count})")
            return USB4TrainingState.CONFIGURATION
        else:
            self.logger.warning(f"Recovery failed (attempt {self.retry_count})")
            return USB4TrainingState.RECOVERY

    def _process_l0_state(self) -> USB4TrainingState:
        """Process L0 (active) state"""
        # Monitor for power state changes or errors
        if self.config.enable_power_management:
            # Simulate occasional power state transitions
            if np.random.random() < 0.01:  # 1% chance of power state change
                new_link_state = np.random.choice(list(USB4LinkState))
                if new_link_state != self.current_link_state:
                    self._transition_power_state(new_link_state)

        return USB4TrainingState.L0  # Stay in L0

    def _transition_power_state(self, new_state: USB4LinkState) -> None:
        """
        Transition to new power state

        Args:
            new_state: Target power state
        """
        old_state = self.current_link_state
        transition_key = (old_state, new_state)

        # Simulate transition time
        transition_time = self._get_power_transition_time(old_state, new_state)
        time.sleep(transition_time)

        self.current_link_state = new_state
        self.power_transition_times[transition_key] = transition_time

        # Measure power consumption
        power_consumption = self._measure_power_consumption(new_state)
        self.power_measurements[new_state] = power_consumption

        self.logger.info(
            f"Power state transition: {old_state.name} -> {new_state.name} "
            f"({transition_time*1000:.1f}ms, {power_consumption:.2f}W)"
        )

    def _get_power_transition_time(self, from_state: USB4LinkState, to_state: USB4LinkState) -> float:
        """
        Get expected power state transition time

        Args:
            from_state: Source power state
            to_state: Target power state

        Returns:
            Transition time in seconds
        """
        # Transition time matrix (simplified)
        transition_times = {
            (USB4LinkState.U0, USB4LinkState.U1): 0.001,  # 1ms
            (USB4LinkState.U0, USB4LinkState.U2): 0.010,  # 10ms
            (USB4LinkState.U0, USB4LinkState.U3): 0.100,  # 100ms
            (USB4LinkState.U1, USB4LinkState.U0): 0.001,  # 1ms
            (USB4LinkState.U1, USB4LinkState.U2): 0.005,  # 5ms
            (USB4LinkState.U1, USB4LinkState.U3): 0.050,  # 50ms
            (USB4LinkState.U2, USB4LinkState.U0): 0.010,  # 10ms
            (USB4LinkState.U2, USB4LinkState.U1): 0.005,  # 5ms
            (USB4LinkState.U2, USB4LinkState.U3): 0.020,  # 20ms
            (USB4LinkState.U3, USB4LinkState.U0): 0.100,  # 100ms
            (USB4LinkState.U3, USB4LinkState.U1): 0.050,  # 50ms
            (USB4LinkState.U3, USB4LinkState.U2): 0.020,  # 20ms
        }

        return transition_times.get((from_state, to_state), 0.001)

    def _measure_power_consumption(self, state: USB4LinkState) -> float:
        """
        Measure power consumption for given state

        Args:
            state: USB4 link state

        Returns:
            Power consumption in watts
        """
        # Use specifications with some measurement noise
        base_power = {
            USB4LinkState.U0: self.specs.IDLE_POWER_U0,
            USB4LinkState.U1: self.specs.IDLE_POWER_U1,
            USB4LinkState.U2: self.specs.IDLE_POWER_U2,
            USB4LinkState.U3: self.specs.IDLE_POWER_U3,
        }[state]

        # Add 5% measurement noise
        noise = np.random.normal(0, 0.05 * base_power)
        return max(0.0, base_power + noise)

    def _simulate_power_state_monitoring(self) -> USB4LinkState:
        """
        Simulate power state monitoring

        Returns:
            Current power state
        """
        # Simple simulation - mostly stay in current state
        if np.random.random() < 0.95:
            return self.current_link_state
        else:
            # Occasional state change
            return np.random.choice(list(USB4LinkState))

    def _link_state_to_training_state(self, link_state: USB4LinkState) -> USB4TrainingState:
        """
        Convert USB4 link state to training state

        Args:
            link_state: USB4 link state

        Returns:
            Corresponding training state
        """
        mapping = {
            USB4LinkState.U0: USB4TrainingState.L0,
            USB4LinkState.U1: USB4TrainingState.L1,
            USB4LinkState.U2: USB4TrainingState.L2,
            USB4LinkState.U3: USB4TrainingState.L3,
        }
        return mapping[link_state]

    def _inject_error(self, error_type: USB4ErrorType) -> None:
        """
        Inject error for recovery testing

        Args:
            error_type: Type of error to inject
        """
        self.logger.info(f"Injecting error: {error_type.name}")
        self.error_count += 1

        # Simulate error effects
        if error_type == USB4ErrorType.SIGNAL_INTEGRITY:
            # Simulate signal degradation
            pass
        elif error_type == USB4ErrorType.LINK_TRAINING:
            # Force training failure
            self.current_state = USB4TrainingState.RECOVERY
        elif error_type == USB4ErrorType.PROTOCOL:
            # Simulate protocol violation
            pass
        elif error_type == USB4ErrorType.POWER_MANAGEMENT:
            # Force power state error
            self.current_link_state = USB4LinkState.U3

    def _attempt_recovery(self, error_type: USB4ErrorType) -> bool:
        """
        Attempt error recovery

        Args:
            error_type: Type of error to recover from

        Returns:
            True if recovery successful
        """
        self.logger.debug(f"Attempting recovery from {error_type.name}")

        # Simulate recovery success rates based on error type
        success_rates = {
            USB4ErrorType.SIGNAL_INTEGRITY: 0.7,
            USB4ErrorType.LINK_TRAINING: 0.8,
            USB4ErrorType.PROTOCOL: 0.6,
            USB4ErrorType.POWER_MANAGEMENT: 0.9,
        }

        success_rate = success_rates.get(error_type, 0.5)
        return np.random.random() < success_rate

    def _get_recovery_method(self, error_type: USB4ErrorType) -> str:
        """
        Get recovery method for error type

        Args:
            error_type: Type of error

        Returns:
            Recovery method description
        """
        methods = {
            USB4ErrorType.SIGNAL_INTEGRITY: "Signal equalization adjustment",
            USB4ErrorType.LINK_TRAINING: "Link training restart",
            USB4ErrorType.PROTOCOL: "Protocol state reset",
            USB4ErrorType.POWER_MANAGEMENT: "Power state recovery",
        }
        return methods.get(error_type, "Generic recovery")

    def _calculate_recovery_success_rate(self) -> float:
        """
        Calculate overall recovery success rate

        Returns:
            Recovery success rate (0.0 to 1.0)
        """
        if self.recovery_attempts == 0:
            return 0.0
        return self.recovery_successes / self.recovery_attempts


__all__ = [
    # Enums
    "USB4TrainingState",
    "USB4TrainingEvent",
    # Data structures
    "USB4TrainingConfig",
    "USB4StateTransition",
    "USB4TrainingResults",
    "USB4NegotiationResults",
    "USB4RecoveryResults",
    # Main class
    "USB4LinkTraining",
]
