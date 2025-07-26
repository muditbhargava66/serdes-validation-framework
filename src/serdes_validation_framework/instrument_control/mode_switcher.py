"""
Mode Switcher Module

This module provides dual-mode switching capabilities for PCIe signals,
supporting both NRZ and PAM4 modes with comprehensive type checking.

Features:
- Seamless mode switching
- Configuration validation
- Performance optimization
- Error handling
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

# Import SignalMode from constants to avoid duplication
from ..protocols.pcie.constants import SignalMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModeConfig:
    """Configuration for mode switching with validation"""

    sample_rate: float
    bandwidth: float
    voltage_range: float

    def __post_init__(self) -> None:
        """
        Validate mode configuration

        Raises:
            AssertionError: If configuration is invalid
        """
        # Type validation
        assert isinstance(self.sample_rate, float), f"Sample rate must be float, got {type(self.sample_rate)}"
        assert isinstance(self.bandwidth, float), f"Bandwidth must be float, got {type(self.bandwidth)}"
        assert isinstance(self.voltage_range, float), f"Voltage range must be float, got {type(self.voltage_range)}"

        # Value validation
        assert self.sample_rate > 0, f"Sample rate must be positive, got {self.sample_rate}"
        assert self.bandwidth > 0, f"Bandwidth must be positive, got {self.bandwidth}"
        assert self.voltage_range > 0, f"Voltage range must be positive, got {self.voltage_range}"


@dataclass
class SwitchResult:
    """Result of mode switching operation"""

    success: bool
    previous_mode: SignalMode
    new_mode: SignalMode
    switch_time: float
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Validate switch result

        Raises:
            AssertionError: If result is invalid
        """
        # Type validation
        assert isinstance(self.success, bool), f"Success must be bool, got {type(self.success)}"
        assert isinstance(self.previous_mode, SignalMode), f"Previous mode must be SignalMode, got {type(self.previous_mode)}"
        assert isinstance(self.new_mode, SignalMode), f"New mode must be SignalMode, got {type(self.new_mode)}"
        assert isinstance(self.switch_time, float), f"Switch time must be float, got {type(self.switch_time)}"

        # Value validation
        assert self.switch_time >= 0, f"Switch time must be non-negative, got {self.switch_time}"

        if self.error_message is not None:
            assert isinstance(self.error_message, str), f"Error message must be string, got {type(self.error_message)}"


class ModeSwitcher:
    """PCIe dual-mode switcher with type safety"""

    def __init__(self, default_mode: SignalMode, default_sample_rate: float, default_bandwidth: float) -> None:
        """
        Initialize mode switcher

        Args:
            default_mode: Default signal mode
            default_sample_rate: Default sample rate in Hz
            default_bandwidth: Default bandwidth in Hz

        Raises:
            AssertionError: If parameters are invalid
        """
        # Validate inputs
        assert isinstance(default_mode, SignalMode), f"Default mode must be SignalMode, got {type(default_mode)}"
        assert isinstance(default_sample_rate, float), f"Default sample rate must be float, got {type(default_sample_rate)}"
        assert isinstance(default_bandwidth, float), f"Default bandwidth must be float, got {type(default_bandwidth)}"

        # Value validation
        assert default_sample_rate > 0, f"Default sample rate must be positive, got {default_sample_rate}"
        assert default_bandwidth > 0, f"Default bandwidth must be positive, got {default_bandwidth}"

        self.current_mode = default_mode
        self.default_config = ModeConfig(
            sample_rate=default_sample_rate,
            bandwidth=default_bandwidth,
            voltage_range=1.0,  # Default voltage range
        )

        # Mode-specific configurations
        self.mode_configs: Dict[SignalMode, ModeConfig] = {
            SignalMode.NRZ: ModeConfig(sample_rate=default_sample_rate, bandwidth=default_bandwidth, voltage_range=0.8),
            SignalMode.PAM4: ModeConfig(
                sample_rate=default_sample_rate * 2,  # Higher sample rate for PAM4
                bandwidth=default_bandwidth * 1.5,  # Higher bandwidth for PAM4
                voltage_range=1.2,
            ),
        }

        logger.info(f"Mode switcher initialized in {default_mode.name} mode")

    def switch_mode(self, target_mode: SignalMode, config: Optional[ModeConfig] = None) -> SwitchResult:
        """
        Switch to target signal mode

        Args:
            target_mode: Target signal mode
            config: Optional custom configuration

        Returns:
            Switch operation result

        Raises:
            AssertionError: If parameters are invalid
        """
        # Validate input
        assert isinstance(target_mode, SignalMode), f"Target mode must be SignalMode, got {type(target_mode)}"

        if config is not None:
            assert isinstance(config, ModeConfig), f"Config must be ModeConfig, got {type(config)}"

        previous_mode = self.current_mode

        try:
            # Use provided config or default for mode
            switch_config = config or self.mode_configs[target_mode]

            # Simulate mode switching time
            import time

            start_time = time.time()

            # Perform mode switch
            success = self._perform_mode_switch(target_mode, switch_config)

            switch_time = float(time.time() - start_time)

            if success:
                self.current_mode = target_mode
                logger.info(f"Switched from {previous_mode.name} to {target_mode.name}")

                return SwitchResult(success=True, previous_mode=previous_mode, new_mode=target_mode, switch_time=switch_time)
            else:
                return SwitchResult(
                    success=False,
                    previous_mode=previous_mode,
                    new_mode=previous_mode,  # Stay in previous mode
                    switch_time=switch_time,
                    error_message="Mode switch failed",
                )

        except Exception as e:
            logger.error(f"Mode switch failed: {e}")
            return SwitchResult(
                success=False, previous_mode=previous_mode, new_mode=previous_mode, switch_time=0.0, error_message=str(e)
            )

    def _perform_mode_switch(self, target_mode: SignalMode, config: ModeConfig) -> bool:
        """
        Perform the actual mode switching

        Args:
            target_mode: Target signal mode
            config: Mode configuration

        Returns:
            True if switch successful

        Raises:
            ValueError: If switch fails
        """
        try:
            # Validate configuration for target mode
            if not self._validate_mode_config(target_mode, config):
                return False

            # Update mode configuration
            self.mode_configs[target_mode] = config

            # Simulate hardware reconfiguration
            if target_mode == SignalMode.PAM4:
                # PAM4 requires higher sample rate and bandwidth
                if config.sample_rate < 40e9:  # 40 GSa/s minimum for PAM4
                    logger.warning("Sample rate may be too low for PAM4")

            elif target_mode == SignalMode.NRZ:
                # NRZ has lower requirements
                if config.sample_rate < 20e9:  # 20 GSa/s minimum for NRZ
                    logger.warning("Sample rate may be too low for NRZ")

            return True

        except Exception as e:
            logger.error(f"Mode switch operation failed: {e}")
            return False

    def _validate_mode_config(self, mode: SignalMode, config: ModeConfig) -> bool:
        """
        Validate configuration for specific mode

        Args:
            mode: Signal mode
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        try:
            if mode == SignalMode.PAM4:
                # PAM4 requirements
                if config.sample_rate < 20e9:
                    logger.error("PAM4 requires minimum 20 GSa/s sample rate")
                    return False
                if config.bandwidth < 30e9:
                    logger.error("PAM4 requires minimum 30 GHz bandwidth")
                    return False

            elif mode == SignalMode.NRZ:
                # NRZ requirements
                if config.sample_rate < 10e9:
                    logger.error("NRZ requires minimum 10 GSa/s sample rate")
                    return False
                if config.bandwidth < 15e9:
                    logger.error("NRZ requires minimum 15 GHz bandwidth")
                    return False

            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_current_mode(self) -> SignalMode:
        """
        Get current signal mode

        Returns:
            Current signal mode
        """
        return self.current_mode

    def get_mode_config(self, mode: SignalMode) -> ModeConfig:
        """
        Get configuration for specific mode

        Args:
            mode: Signal mode

        Returns:
            Mode configuration

        Raises:
            AssertionError: If mode is invalid
        """
        assert isinstance(mode, SignalMode), f"Mode must be SignalMode, got {type(mode)}"

        return self.mode_configs[mode]

    def update_mode_config(self, mode: SignalMode, config: ModeConfig) -> bool:
        """
        Update configuration for specific mode

        Args:
            mode: Signal mode
            config: New configuration

        Returns:
            True if update successful

        Raises:
            AssertionError: If parameters are invalid
        """
        # Validate inputs
        assert isinstance(mode, SignalMode), f"Mode must be SignalMode, got {type(mode)}"
        assert isinstance(config, ModeConfig), f"Config must be ModeConfig, got {type(config)}"

        try:
            # Validate configuration
            if self._validate_mode_config(mode, config):
                self.mode_configs[mode] = config
                logger.info(f"Updated {mode.name} configuration")
                return True
            else:
                logger.error(f"Invalid configuration for {mode.name}")
                return False

        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return False

    def get_supported_modes(self) -> list[SignalMode]:
        """
        Get list of supported signal modes

        Returns:
            List of supported modes
        """
        return list(SignalMode)

    def is_mode_available(self, mode: SignalMode) -> bool:
        """
        Check if mode is available

        Args:
            mode: Signal mode to check

        Returns:
            True if mode is available

        Raises:
            AssertionError: If mode is invalid
        """
        assert isinstance(mode, SignalMode), f"Mode must be SignalMode, got {type(mode)}"

        return mode in self.mode_configs


# Example usage and factory functions
def create_mode_switcher(
    default_mode: SignalMode = SignalMode.NRZ, sample_rate: float = 50e9, bandwidth: float = 25e9
) -> ModeSwitcher:
    """
    Create mode switcher with default configuration

    Args:
        default_mode: Default signal mode
        sample_rate: Default sample rate in Hz
        bandwidth: Default bandwidth in Hz

    Returns:
        Configured mode switcher

    Raises:
        AssertionError: If parameters are invalid
    """
    # Validate inputs
    assert isinstance(default_mode, SignalMode), f"Default mode must be SignalMode, got {type(default_mode)}"
    assert isinstance(sample_rate, float), f"Sample rate must be float, got {type(sample_rate)}"
    assert isinstance(bandwidth, float), f"Bandwidth must be float, got {type(bandwidth)}"

    return ModeSwitcher(default_mode=default_mode, default_sample_rate=sample_rate, default_bandwidth=bandwidth)
