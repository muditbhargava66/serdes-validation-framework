#!/usr/bin/env python3
"""
Test Suite for PCIe Dual-Mode Operation

This module provides comprehensive testing for the PCIe dual-mode (NRZ/PAM4) functionality.
It includes tests for:
- Mode switching operations
- NRZ training and configuration
- PAM4 training and configuration
- Mode transition validity
- Type checking and validation for all parameters

The test suite uses realistic signal parameters and proper data types throughout.
"""

import logging
import os
import sys
import unittest
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from serdes_validation_framework.protocols.pcie.dual_mode.mode_control import ModeConfig, ModeController, SignalMode
    from serdes_validation_framework.protocols.pcie.dual_mode.nrz_training import (
        NRZTrainer,
        NRZTrainingConfig,
        NRZTrainingResults,
        TrainingPhase,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False

    # Mock classes for testing
    from dataclasses import dataclass
    from enum import Enum, auto

    class SignalMode(Enum):
        NRZ = auto()
        PAM4 = auto()

    class TrainingPhase(Enum):
        INITIALIZATION = auto()
        TRAINING = auto()
        VALIDATION = auto()

    @dataclass
    class ModeConfig:
        mode: SignalMode
        sample_rate: float = 100e9
        voltage_range: float = 2.0

    @dataclass
    class NRZTrainingConfig:
        mode: SignalMode = SignalMode.NRZ
        sample_rate: float = 100e9

    @dataclass
    class NRZTrainingResults:
        success: bool = True
        phase: TrainingPhase = TrainingPhase.VALIDATION

    class ModeController:
        def __init__(self, config):
            self.config = config

        def switch_mode(self, mode):
            return {"status": "success", "mode": mode}

    class NRZTrainer:
        def __init__(self, config):
            self.config = config

        def train(self, signal_data):
            return NRZTrainingResults()


from src.serdes_validation_framework.protocols.pcie.dual_mode.pam4_training import (
    PAM4Trainer,
    PAM4TrainingConfig,
    PAM4TrainingPhase,
    PAM4TrainingResults,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@unittest.skip("Dual mode tests need API updates")
class TestDualModeOperation(unittest.TestCase):
    """
    Test cases for PCIe dual-mode operation.

    This test suite verifies the correctness of PCIe dual-mode functionality including:
    - Mode configuration and switching
    - Training operations for both NRZ and PAM4
    - Type validation for all parameters
    """

    def setUp(self) -> None:
        """Set up test fixtures for each test."""
        # Create NRZ configuration
        self.nrz_config = ModeConfig(
            mode=SignalMode.NRZ, sample_rate=16.0, bandwidth=8.0, eye_height_threshold=0.3, equalization_taps=5
        )

        # Create PAM4 configuration
        self.pam4_config = ModeConfig(
            mode=SignalMode.PAM4, sample_rate=32.0, bandwidth=16.0, eye_height_threshold=0.2, equalization_taps=7
        )

        # Initialize mode controller
        self.controller = ModeController(SignalMode.NRZ)
        self.controller.register_mode_config(self.nrz_config)
        self.controller.register_mode_config(self.pam4_config)

        # Create NRZ training configuration
        self.nrz_training_config = NRZTrainingConfig.default_config()

        # Create PAM4 training configuration
        self.pam4_training_config = PAM4TrainingConfig.default_config()

        # Initialize trainers
        self.nrz_trainer = NRZTrainer(self.nrz_training_config)
        self.pam4_trainer = PAM4Trainer(self.pam4_training_config)

    def test_mode_config_initialization(self) -> None:
        """Test ModeConfig initialization with parameter validation."""
        # Test valid initialization
        config = ModeConfig(mode=SignalMode.NRZ, sample_rate=16.0, bandwidth=8.0, eye_height_threshold=0.3, equalization_taps=5)
        self.assertIsInstance(config, ModeConfig)

        # Test with invalid mode type
        with self.assertRaises(AssertionError):
            ModeConfig(
                mode="NRZ",  # type: ignore
                sample_rate=16.0,
                bandwidth=8.0,
                eye_height_threshold=0.3,
                equalization_taps=5,
            )

        # Test with invalid sample_rate type
        with self.assertRaises(AssertionError):
            ModeConfig(
                mode=SignalMode.NRZ,
                sample_rate="16.0",  # type: ignore
                bandwidth=8.0,
                eye_height_threshold=0.3,
                equalization_taps=5,
            )

        # Test with invalid pre_cursor type
        with self.assertRaises(AssertionError):
            ModeConfig(
                mode=SignalMode.NRZ,
                voltage_swing=0.8,
                pre_cursor="0.1",  # type: ignore
                post_cursor=0.2,
                sample_rate=16.0,
                bit_rate=32.0,
            )

        # Test with invalid post_cursor type
        with self.assertRaises(AssertionError):
            ModeConfig(
                mode=SignalMode.NRZ,
                voltage_swing=0.8,
                pre_cursor=0.1,
                post_cursor="0.2",  # type: ignore
                sample_rate=16.0,
                bit_rate=32.0,
            )

        # Test with invalid sample_rate type
        with self.assertRaises(AssertionError):
            ModeConfig(
                mode=SignalMode.NRZ,
                voltage_swing=0.8,
                pre_cursor=0.1,
                post_cursor=0.2,
                sample_rate="16.0",  # type: ignore
                bit_rate=32.0,
            )

        # Test with invalid bit_rate type
        with self.assertRaises(AssertionError):
            ModeConfig(
                mode=SignalMode.NRZ,
                voltage_swing=0.8,
                pre_cursor=0.1,
                post_cursor=0.2,
                sample_rate=16.0,
                bit_rate="32.0",  # type: ignore
            )

        # Test with invalid voltage_swing value
        with self.assertRaises(AssertionError):
            ModeConfig(
                mode=SignalMode.NRZ,
                voltage_swing=-0.8,  # Negative value
                pre_cursor=0.1,
                post_cursor=0.2,
                sample_rate=16.0,
                bit_rate=32.0,
            )

        # Test with invalid pre_cursor value
        with self.assertRaises(AssertionError):
            ModeConfig(
                mode=SignalMode.NRZ,
                voltage_swing=0.8,
                pre_cursor=1.5,  # Out of range [-1, 1]
                post_cursor=0.2,
                sample_rate=16.0,
                bit_rate=32.0,
            )

        # Test with invalid post_cursor value
        with self.assertRaises(AssertionError):
            ModeConfig(
                mode=SignalMode.NRZ,
                voltage_swing=0.8,
                pre_cursor=0.1,
                post_cursor=1.5,  # Out of range [-1, 1]
                sample_rate=16.0,
                bit_rate=32.0,
            )

        # Test with invalid sample_rate value
        with self.assertRaises(AssertionError):
            ModeConfig(
                mode=SignalMode.NRZ,
                voltage_swing=0.8,
                pre_cursor=0.1,
                post_cursor=0.2,
                sample_rate=0.0,  # Zero value
                bit_rate=32.0,
            )

        # Test with invalid bit_rate value
        with self.assertRaises(AssertionError):
            ModeConfig(
                mode=SignalMode.NRZ,
                voltage_swing=0.8,
                pre_cursor=0.1,
                post_cursor=0.2,
                sample_rate=16.0,
                bit_rate=0.0,  # Zero value
            )

    def test_mode_config_serialization(self) -> None:
        """Test ModeConfig serialization and deserialization with type validation."""
        # Convert to dictionary
        config_dict = self.nrz_config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["mode"], "NRZ")
        self.assertIsInstance(config_dict["voltage_swing"], float)

        # Convert to JSON and back
        json_str = self.nrz_config.to_json()
        self.assertIsInstance(json_str, str)

        # Deserialize from JSON
        restored_config = ModeConfig.from_json(json_str)
        self.assertIsInstance(restored_config, ModeConfig)
        self.assertEqual(restored_config.mode, self.nrz_config.mode)
        self.assertEqual(restored_config.voltage_swing, self.nrz_config.voltage_swing)

        # Test from_dict with missing keys
        invalid_dict = {"mode": "NRZ"}  # Missing required keys
        with self.assertRaises(AssertionError):
            ModeConfig.from_dict(invalid_dict)

        # Test from_dict with invalid types
        invalid_dict = {
            "mode": "NRZ",
            "voltage_swing": "0.8",  # String instead of float
            "pre_cursor": 0.1,
            "post_cursor": 0.2,
            "sample_rate": 16.0,
            "bit_rate": 32.0,
        }
        # This should convert string to float successfully in from_dict
        config = ModeConfig.from_dict(invalid_dict)
        self.assertIsInstance(config.voltage_swing, float)

        # Test with invalid mode string
        invalid_dict = {
            "mode": "INVALID",  # Invalid mode name
            "voltage_swing": 0.8,
            "pre_cursor": 0.1,
            "post_cursor": 0.2,
            "sample_rate": 16.0,
            "bit_rate": 32.0,
        }
        with self.assertRaises(AssertionError):
            ModeConfig.from_dict(invalid_dict)

    def test_mode_controller_initialization(self) -> None:
        """Test ModeController initialization with parameter validation."""
        # Test valid initialization
        controller = ModeController(self.nrz_config)
        self.assertIsInstance(controller, ModeController)

        # Test with invalid config type
        with self.assertRaises(AssertionError):
            ModeController("invalid_config")  # type: ignore

        # Verify initial state
        self.assertEqual(controller.current_mode, SignalMode.NRZ)
        self.assertEqual(len(controller.mode_history), 1)

    def test_mode_controller_switch_mode(self) -> None:
        """Test mode switching with type validation."""
        # Switch to PAM4
        result = self.controller.switch_mode(self.pam4_config)
        self.assertTrue(result)
        self.assertEqual(self.controller.current_mode, SignalMode.PAM4)
        self.assertEqual(len(self.controller.mode_history), 2)

        # Switch back to NRZ
        result = self.controller.switch_mode(self.nrz_config)
        self.assertTrue(result)
        self.assertEqual(self.controller.current_mode, SignalMode.NRZ)
        self.assertEqual(len(self.controller.mode_history), 3)

        # Test with invalid config type
        with self.assertRaises(AssertionError):
            self.controller.switch_mode("invalid_config")  # type: ignore

        # Test switching to same mode (should succeed)
        result = self.controller.switch_mode(self.nrz_config)
        self.assertTrue(result)
        self.assertEqual(len(self.controller.mode_history), 4)

    def test_mode_controller_optimal_config(self) -> None:
        """Test optimal config generation with type validation."""
        # Get optimal NRZ config
        nrz_config = self.controller.get_optimal_config(SignalMode.NRZ)
        self.assertIsInstance(nrz_config, ModeConfig)
        self.assertEqual(nrz_config.mode, SignalMode.NRZ)

        # Get optimal PAM4 config
        pam4_config = self.controller.get_optimal_config(SignalMode.PAM4)
        self.assertIsInstance(pam4_config, ModeConfig)
        self.assertEqual(pam4_config.mode, SignalMode.PAM4)

        # Test with invalid mode type
        with self.assertRaises(AssertionError):
            self.controller.get_optimal_config("NRZ")  # type: ignore

    def test_nrz_training_config(self) -> None:
        """Test NRZ training configuration with type validation."""
        # Test default config
        config = NRZTrainingConfig.default_config()
        self.assertIsInstance(config, NRZTrainingConfig)

        # Test valid initialization
        config = NRZTrainingConfig(
            voltage_swing=0.8,
            pre_emphasis=0.2,
            de_emphasis=0.1,
            eq_boost=1.0,
            training_pattern=2,
            max_iterations=50,
            convergence_threshold=0.01,
            timeout_seconds=60.0,
        )
        self.assertIsInstance(config, NRZTrainingConfig)

        # Test with invalid voltage_swing type
        with self.assertRaises(AssertionError):
            NRZTrainingConfig(
                voltage_swing="0.8",  # type: ignore
                pre_emphasis=0.2,
                de_emphasis=0.1,
                eq_boost=1.0,
                training_pattern=2,
                max_iterations=50,
                convergence_threshold=0.01,
                timeout_seconds=60.0,
            )

        # Test with invalid pre_emphasis type
        with self.assertRaises(AssertionError):
            NRZTrainingConfig(
                voltage_swing=0.8,
                pre_emphasis="0.2",  # type: ignore
                de_emphasis=0.1,
                eq_boost=1.0,
                training_pattern=2,
                max_iterations=50,
                convergence_threshold=0.01,
                timeout_seconds=60.0,
            )

        # Test with invalid de_emphasis type
        with self.assertRaises(AssertionError):
            NRZTrainingConfig(
                voltage_swing=0.8,
                pre_emphasis=0.2,
                de_emphasis="0.1",  # type: ignore
                eq_boost=1.0,
                training_pattern=2,
                max_iterations=50,
                convergence_threshold=0.01,
                timeout_seconds=60.0,
            )

        # Test with invalid eq_boost type
        with self.assertRaises(AssertionError):
            NRZTrainingConfig(
                voltage_swing=0.8,
                pre_emphasis=0.2,
                de_emphasis=0.1,
                eq_boost="1.0",  # type: ignore
                training_pattern=2,
                max_iterations=50,
                convergence_threshold=0.01,
                timeout_seconds=60.0,
            )

        # Test with invalid training_pattern type
        with self.assertRaises(AssertionError):
            NRZTrainingConfig(
                voltage_swing=0.8,
                pre_emphasis=0.2,
                de_emphasis=0.1,
                eq_boost=1.0,
                training_pattern="2",  # type: ignore
                max_iterations=50,
                convergence_threshold=0.01,
                timeout_seconds=60.0,
            )

        # Test with invalid max_iterations type
        with self.assertRaises(AssertionError):
            NRZTrainingConfig(
                voltage_swing=0.8,
                pre_emphasis=0.2,
                de_emphasis=0.1,
                eq_boost=1.0,
                training_pattern=2,
                max_iterations="50",  # type: ignore
                convergence_threshold=0.01,
                timeout_seconds=60.0,
            )

        # Test with invalid convergence_threshold type
        with self.assertRaises(AssertionError):
            NRZTrainingConfig(
                voltage_swing=0.8,
                pre_emphasis=0.2,
                de_emphasis=0.1,
                eq_boost=1.0,
                training_pattern=2,
                max_iterations=50,
                convergence_threshold="0.01",  # type: ignore
                timeout_seconds=60.0,
            )

        # Test with invalid timeout_seconds type
        with self.assertRaises(AssertionError):
            NRZTrainingConfig(
                voltage_swing=0.8,
                pre_emphasis=0.2,
                de_emphasis=0.1,
                eq_boost=1.0,
                training_pattern=2,
                max_iterations=50,
                convergence_threshold=0.01,
                timeout_seconds="60.0",  # type: ignore
            )

        # Test with invalid pre_emphasis range
        with self.assertRaises(AssertionError):
            NRZTrainingConfig(
                voltage_swing=0.8,
                pre_emphasis=1.5,  # Out of range [0, 1]
                de_emphasis=0.1,
                eq_boost=1.0,
                training_pattern=2,
                max_iterations=50,
                convergence_threshold=0.01,
                timeout_seconds=60.0,
            )

        # Test serialization to dictionary
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)

        # Test deserialization from dictionary
        restored_config = NRZTrainingConfig.from_dict(config_dict)
        self.assertIsInstance(restored_config, NRZTrainingConfig)

        # Test with missing keys
        invalid_dict = {"voltage_swing": 0.8}  # Missing required keys
        with self.assertRaises(AssertionError):
            NRZTrainingConfig.from_dict(invalid_dict)

    def test_nrz_training_results(self) -> None:
        """Test NRZ training results with type validation."""
        # Test default initialization
        results = NRZTrainingResults()
        self.assertIsInstance(results, NRZTrainingResults)

        # Test with explicit values
        results = NRZTrainingResults(
            success=True,
            phase_reached=TrainingPhase.COMPLETE,
            iterations=25,
            final_eye_height=0.5,
            final_eye_width=0.7,
            ber_estimate=1e-12,
            training_time=15.5,
            voltage_adjustments=[0.1, 0.2, 0.3],
            emphasis_adjustments=[0.05, 0.1, 0.15],
        )
        self.assertIsInstance(results, NRZTrainingResults)

        # Test with invalid success type
        with self.assertRaises(AssertionError):
            NRZTrainingResults(
                success="True",  # type: ignore
                phase_reached=TrainingPhase.COMPLETE,
                iterations=25,
                final_eye_height=0.5,
                final_eye_width=0.7,
                ber_estimate=1e-12,
                training_time=15.5,
            )

        # Test with invalid phase_reached type
        with self.assertRaises(AssertionError):
            NRZTrainingResults(
                success=True,
                phase_reached="COMPLETE",  # type: ignore
                iterations=25,
                final_eye_height=0.5,
                final_eye_width=0.7,
                ber_estimate=1e-12,
                training_time=15.5,
            )

        # Test with invalid iterations type
        with self.assertRaises(AssertionError):
            NRZTrainingResults(
                success=True,
                phase_reached=TrainingPhase.COMPLETE,
                iterations="25",  # type: ignore
                final_eye_height=0.5,
                final_eye_width=0.7,
                ber_estimate=1e-12,
                training_time=15.5,
            )

        # Test with invalid final_eye_height type
        with self.assertRaises(AssertionError):
            NRZTrainingResults(
                success=True,
                phase_reached=TrainingPhase.COMPLETE,
                iterations=25,
                final_eye_height="0.5",  # type: ignore
                final_eye_width=0.7,
                ber_estimate=1e-12,
                training_time=15.5,
            )

        # Test with invalid final_eye_width type
        with self.assertRaises(AssertionError):
            NRZTrainingResults(
                success=True,
                phase_reached=TrainingPhase.COMPLETE,
                iterations=25,
                final_eye_height=0.5,
                final_eye_width="0.7",  # type: ignore
                ber_estimate=1e-12,
                training_time=15.5,
            )

        # Test with invalid ber_estimate type
        with self.assertRaises(AssertionError):
            NRZTrainingResults(
                success=True,
                phase_reached=TrainingPhase.COMPLETE,
                iterations=25,
                final_eye_height=0.5,
                final_eye_width=0.7,
                ber_estimate="1e-12",  # type: ignore
                training_time=15.5,
            )

        # Test with invalid training_time type
        with self.assertRaises(AssertionError):
            NRZTrainingResults(
                success=True,
                phase_reached=TrainingPhase.COMPLETE,
                iterations=25,
                final_eye_height=0.5,
                final_eye_width=0.7,
                ber_estimate=1e-12,
                training_time="15.5",  # type: ignore
            )

        # Test with invalid voltage_adjustments type
        with self.assertRaises(AssertionError):
            NRZTrainingResults(
                success=True,
                phase_reached=TrainingPhase.COMPLETE,
                iterations=25,
                final_eye_height=0.5,
                final_eye_width=0.7,
                ber_estimate=1e-12,
                training_time=15.5,
                voltage_adjustments="invalid",  # type: ignore
            )

        # Test with invalid voltage_adjustments elements
        with self.assertRaises(AssertionError):
            NRZTrainingResults(
                success=True,
                phase_reached=TrainingPhase.COMPLETE,
                iterations=25,
                final_eye_height=0.5,
                final_eye_width=0.7,
                ber_estimate=1e-12,
                training_time=15.5,
                voltage_adjustments=[0.1, "0.2", 0.3],  # type: ignore
            )

        # Test with invalid ber_estimate range
        with self.assertRaises(AssertionError):
            NRZTrainingResults(
                success=True,
                phase_reached=TrainingPhase.COMPLETE,
                iterations=25,
                final_eye_height=0.5,
                final_eye_width=0.7,
                ber_estimate=1.5,  # Out of range [0, 1]
                training_time=15.5,
            )

        # Test serialization to dictionary
        results_dict = results.to_dict()
        self.assertIsInstance(results_dict, dict)

        # Test deserialization from dictionary
        restored_results = NRZTrainingResults.from_dict(results_dict)
        self.assertIsInstance(restored_results, NRZTrainingResults)

        # Test with invalid phase in dictionary
        invalid_dict = results_dict.copy()
        invalid_dict["phase_reached"] = "INVALID_PHASE"
        with self.assertRaises(AssertionError):
            NRZTrainingResults.from_dict(invalid_dict)

    def test_nrz_trainer(self) -> None:
        """Test NRZ trainer with type validation."""
        # Test initialization with default config
        trainer = NRZTrainer()
        self.assertIsInstance(trainer, NRZTrainer)

        # Test initialization with explicit config
        trainer = NRZTrainer(self.nrz_training_config)
        self.assertIsInstance(trainer, NRZTrainer)

        # Test with invalid config type
        with self.assertRaises(AssertionError):
            NRZTrainer("invalid_config")  # type: ignore

        # Test training execution
        results = trainer.train()
        self.assertIsInstance(results, NRZTrainingResults)

        # Verify numeric result types
        self.assertIsInstance(results.final_eye_height, float)
        self.assertIsInstance(results.final_eye_width, float)
        self.assertIsInstance(results.ber_estimate, float)
        self.assertIsInstance(results.training_time, float)

        # Verify list types
        self.assertIsInstance(results.voltage_adjustments, list)
        self.assertIsInstance(results.emphasis_adjustments, list)

        # Verify all elements in lists are floats
        self.assertTrue(all(isinstance(v, float) for v in results.voltage_adjustments))
        self.assertTrue(all(isinstance(e, float) for e in results.emphasis_adjustments))

    def test_pam4_training_config(self) -> None:
        """Test PAM4 training configuration with type validation."""
        # Test default config
        config = PAM4TrainingConfig.default_config()
        self.assertIsInstance(config, PAM4TrainingConfig)

        # Test valid initialization
        config = PAM4TrainingConfig(
            voltage_levels=[0.2, 0.4, 0.6],
            pre_cursor_taps=[-0.1, 0.0],
            post_cursor_taps=[-0.2, -0.1, -0.05],
            level_separation_mse_threshold=0.005,
            eye_height_threshold=0.05,
            training_pattern=4,
            max_iterations=100,
            timeout_seconds=120.0,
        )
        self.assertIsInstance(config, PAM4TrainingConfig)

        # Test with invalid voltage_levels type
        with self.assertRaises(AssertionError):
            PAM4TrainingConfig(
                voltage_levels="invalid",  # type: ignore
                pre_cursor_taps=[-0.1, 0.0],
                post_cursor_taps=[-0.2, -0.1, -0.05],
                level_separation_mse_threshold=0.005,
                eye_height_threshold=0.05,
                training_pattern=4,
                max_iterations=100,
                timeout_seconds=120.0,
            )

        # Test with invalid voltage_levels elements
        with self.assertRaises(AssertionError):
            PAM4TrainingConfig(
                voltage_levels=[0.2, "0.4", 0.6],  # type: ignore
                pre_cursor_taps=[-0.1, 0.0],
                post_cursor_taps=[-0.2, -0.1, -0.05],
                level_separation_mse_threshold=0.005,
                eye_height_threshold=0.05,
                training_pattern=4,
                max_iterations=100,
                timeout_seconds=120.0,
            )

        # Test with invalid voltage_levels length
        with self.assertRaises(AssertionError):
            PAM4TrainingConfig(
                voltage_levels=[0.2, 0.4],  # Need exactly 3 elements
                pre_cursor_taps=[-0.1, 0.0],
                post_cursor_taps=[-0.2, -0.1, -0.05],
                level_separation_mse_threshold=0.005,
                eye_height_threshold=0.05,
                training_pattern=4,
                max_iterations=100,
                timeout_seconds=120.0,
            )

        # Test with invalid voltage_levels ordering
        with self.assertRaises(AssertionError):
            PAM4TrainingConfig(
                voltage_levels=[0.6, 0.4, 0.2],  # Not in ascending order
                pre_cursor_taps=[-0.1, 0.0],
                post_cursor_taps=[-0.2, -0.1, -0.05],
                level_separation_mse_threshold=0.005,
                eye_height_threshold=0.05,
                training_pattern=4,
                max_iterations=100,
                timeout_seconds=120.0,
            )

        # Test with invalid pre_cursor_taps type
        with self.assertRaises(AssertionError):
            PAM4TrainingConfig(
                voltage_levels=[0.2, 0.4, 0.6],
                pre_cursor_taps="invalid",  # type: ignore
                post_cursor_taps=[-0.2, -0.1, -0.05],
                level_separation_mse_threshold=0.005,
                eye_height_threshold=0.05,
                training_pattern=4,
                max_iterations=100,
                timeout_seconds=120.0,
            )

        # Test with invalid pre_cursor_taps elements
        with self.assertRaises(AssertionError):
            PAM4TrainingConfig(
                voltage_levels=[0.2, 0.4, 0.6],
                pre_cursor_taps=[-0.1, "0.0"],  # type: ignore
                post_cursor_taps=[-0.2, -0.1, -0.05],
                level_separation_mse_threshold=0.005,
                eye_height_threshold=0.05,
                training_pattern=4,
                max_iterations=100,
                timeout_seconds=120.0,
            )

        # Test with invalid level_separation_mse_threshold type
        with self.assertRaises(AssertionError):
            PAM4TrainingConfig(
                voltage_levels=[0.2, 0.4, 0.6],
                pre_cursor_taps=[-0.1, 0.0],
                post_cursor_taps=[-0.2, -0.1, -0.05],
                level_separation_mse_threshold="0.005",  # type: ignore
                eye_height_threshold=0.05,
                training_pattern=4,
                max_iterations=100,
                timeout_seconds=120.0,
            )

        # Test serialization to dictionary
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)

        # Test deserialization from dictionary
        restored_config = PAM4TrainingConfig.from_dict(config_dict)
        self.assertIsInstance(restored_config, PAM4TrainingConfig)

        # Test with missing keys
        invalid_dict = {"voltage_levels": [0.2, 0.4, 0.6]}  # Missing required keys
        with self.assertRaises(AssertionError):
            PAM4TrainingConfig.from_dict(invalid_dict)

    def test_pam4_training_results(self) -> None:
        """Test PAM4 training results with type validation."""
        # Test default initialization
        results = PAM4TrainingResults()
        self.assertIsInstance(results, PAM4TrainingResults)

        # Test with explicit values
        results = PAM4TrainingResults(
            success=True,
            phase_reached=PAM4TrainingPhase.COMPLETE,
            iterations=25,
            final_voltage_levels=[0.2, 0.4, 0.6],
            final_eye_heights=[0.1, 0.2, 0.1],
            final_snr=30.0,
            level_mse=0.002,
            training_time=20.5,
            voltage_adjustments_history=[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]],
            tap_adjustments_history=[[-0.1, 1.0, -0.2], [-0.05, 1.0, -0.15]],
        )
        self.assertIsInstance(results, PAM4TrainingResults)

        # Test with invalid success type
        with self.assertRaises(AssertionError):
            PAM4TrainingResults(
                success="True",  # type: ignore
                phase_reached=PAM4TrainingPhase.COMPLETE,
                iterations=25,
                final_voltage_levels=[0.2, 0.4, 0.6],
                final_eye_heights=[0.1, 0.2, 0.1],
                final_snr=30.0,
                level_mse=0.002,
                training_time=20.5,
            )

        # Test with invalid phase_reached type
        with self.assertRaises(AssertionError):
            PAM4TrainingResults(
                success=True,
                phase_reached="COMPLETE",  # type: ignore
                iterations=25,
                final_voltage_levels=[0.2, 0.4, 0.6],
                final_eye_heights=[0.1, 0.2, 0.1],
                final_snr=30.0,
                level_mse=0.002,
                training_time=20.5,
            )

        # Test with invalid final_voltage_levels type
        with self.assertRaises(AssertionError):
            PAM4TrainingResults(
                success=True,
                phase_reached=PAM4TrainingPhase.COMPLETE,
                iterations=25,
                final_voltage_levels="invalid",  # type: ignore
                final_eye_heights=[0.1, 0.2, 0.1],
                final_snr=30.0,
                level_mse=0.002,
                training_time=20.5,
            )

        # Test with invalid final_voltage_levels length
        with self.assertRaises(AssertionError):
            PAM4TrainingResults(
                success=True,
                phase_reached=PAM4TrainingPhase.COMPLETE,
                iterations=25,
                final_voltage_levels=[0.2, 0.4],  # Need exactly 3 elements
                final_eye_heights=[0.1, 0.2, 0.1],
                final_snr=30.0,
                level_mse=0.002,
                training_time=20.5,
            )

        # Test with invalid final_eye_heights type
        with self.assertRaises(AssertionError):
            PAM4TrainingResults(
                success=True,
                phase_reached=PAM4TrainingPhase.COMPLETE,
                iterations=25,
                final_voltage_levels=[0.2, 0.4, 0.6],
                final_eye_heights="invalid",  # type: ignore
                final_snr=30.0,
                level_mse=0.002,
                training_time=20.5,
            )

        # Test with invalid final_eye_heights length
        with self.assertRaises(AssertionError):
            PAM4TrainingResults(
                success=True,
                phase_reached=PAM4TrainingPhase.COMPLETE,
                iterations=25,
                final_voltage_levels=[0.2, 0.4, 0.6],
                final_eye_heights=[0.1, 0.2],  # Need exactly 3 elements
                final_snr=30.0,
                level_mse=0.002,
                training_time=20.5,
            )

        # Test with invalid voltage_adjustments_history type
        with self.assertRaises(AssertionError):
            PAM4TrainingResults(
                success=True,
                phase_reached=PAM4TrainingPhase.COMPLETE,
                iterations=25,
                final_voltage_levels=[0.2, 0.4, 0.6],
                final_eye_heights=[0.1, 0.2, 0.1],
                final_snr=30.0,
                level_mse=0.002,
                training_time=20.5,
                voltage_adjustments_history="invalid",  # type: ignore
            )

        # Test serialization to dictionary
        results_dict = results.to_dict()
        self.assertIsInstance(results_dict, dict)

        # Test deserialization from dictionary
        restored_results = PAM4TrainingResults.from_dict(results_dict)
        self.assertIsInstance(restored_results, PAM4TrainingResults)

        # Test with invalid phase in dictionary
        invalid_dict = results_dict.copy()
        invalid_dict["phase_reached"] = "INVALID_PHASE"
        with self.assertRaises(AssertionError):
            PAM4TrainingResults.from_dict(invalid_dict)

    def test_pam4_trainer(self) -> None:
        """Test PAM4 trainer with type validation."""
        # Test initialization with default config
        trainer = PAM4Trainer()
        self.assertIsInstance(trainer, PAM4Trainer)

        # Test initialization with explicit config
        trainer = PAM4Trainer(self.pam4_training_config)
        self.assertIsInstance(trainer, PAM4Trainer)

        # Test with invalid config type
        with self.assertRaises(AssertionError):
            PAM4Trainer("invalid_config")  # type: ignore

        # Test training execution
        results = trainer.train()
        self.assertIsInstance(results, PAM4TrainingResults)

        # Verify numeric result types
        self.assertIsInstance(results.final_snr, float)
        self.assertIsInstance(results.level_mse, float)
        self.assertIsInstance(results.training_time, float)

        # Verify array types
        self.assertIsInstance(results.final_voltage_levels, list)
        self.assertIsInstance(results.final_eye_heights, list)

        # Verify history types
        self.assertIsInstance(results.voltage_adjustments_history, list)
        self.assertIsInstance(results.tap_adjustments_history, list)

        # Test _adjust_voltage_levels
        voltage_levels = trainer._adjust_voltage_levels(step_scale=1.0)
        self.assertIsInstance(voltage_levels, list)
        self.assertEqual(len(voltage_levels), 3)
        self.assertTrue(all(isinstance(v, float) for v in voltage_levels))

        # Test with invalid step_scale type
        with self.assertRaises(AssertionError):
            trainer._adjust_voltage_levels(step_scale="1.0")  # type: ignore

        # Test _adjust_ffe_taps
        taps = trainer._adjust_ffe_taps(step_scale=1.0)
        self.assertIsInstance(taps, list)
        self.assertTrue(all(isinstance(t, float) for t in taps))

        # Test with invalid step_scale type
        with self.assertRaises(AssertionError):
            trainer._adjust_ffe_taps(step_scale="1.0")  # type: ignore

        # Test _measure_eye_heights
        eye_heights = trainer._measure_eye_heights()
        self.assertIsInstance(eye_heights, list)
        self.assertEqual(len(eye_heights), 3)
        self.assertTrue(all(isinstance(h, float) for h in eye_heights))

        # Test _calculate_snr
        snr = trainer._calculate_snr()
        self.assertIsInstance(snr, float)
        self.assertGreaterEqual(snr, 0.0)

        # Test _calculate_level_separation_mse
        mse = trainer._calculate_level_separation_mse()
        self.assertIsInstance(mse, float)
        self.assertGreaterEqual(mse, 0.0)


if __name__ == "__main__":
    unittest.main()
