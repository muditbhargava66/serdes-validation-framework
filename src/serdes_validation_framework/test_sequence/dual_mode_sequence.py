"""
PCIe Dual-Mode Test Sequence Module

This module provides functionality for executing dual-mode (NRZ/PAM4) test sequences
with robust type checking and validation for floating-point parameters.

It supports:
- Mode transition testing (NRZ to PAM4 and vice versa)
- Dual-mode compliance verification
- Combined NRZ/PAM4 margin analysis
- Equalization optimization for both modes
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .pcie_sequence import PCIeGen, PCIeTestSequence, SignalMode, TestParameters, TestResult, TestType

logger = logging.getLogger(__name__)


class TransitionMode(Enum):
    """Transition mode enumeration."""
    NRZ_TO_PAM4 = auto()  # Transition from NRZ to PAM4
    PAM4_TO_NRZ = auto()  # Transition from PAM4 to NRZ
    BIDIRECTIONAL = auto()  # Test both transitions


class EqualizationMode(Enum):
    """Equalization optimization mode."""
    NRZ_ONLY = auto()     # Optimize only for NRZ
    PAM4_ONLY = auto()    # Optimize only for PAM4
    NRZ_PRIORITY = auto()  # Optimize for both with NRZ priority
    PAM4_PRIORITY = auto()  # Optimize for both with PAM4 priority
    BALANCED = auto()     # Equal weight to both modes


@dataclass
class DualModeTestParameters:
    """
    Test parameters for dual-mode test sequences.
    
    Attributes:
        nrz_parameters (TestParameters): Parameters for NRZ testing.
        pam4_parameters (TestParameters): Parameters for PAM4 testing.
        transition_duration (float): Duration for mode transition in seconds.
        stability_duration (float): Duration to verify stability after transition in seconds.
        max_transitions (int): Maximum number of transitions to test.
        eq_convergence_threshold (float): Threshold for equalization convergence.
        transition_method (str): Method used for transition ('fast', 'stable', 'gradual').
    """
    
    nrz_parameters: TestParameters
    pam4_parameters: TestParameters
    transition_duration: float
    stability_duration: float
    max_transitions: int
    eq_convergence_threshold: float
    transition_method: str
    
    def __post_init__(self) -> None:
        """
        Validate the dual-mode test parameters.
        
        Raises:
            AssertionError: If any parameter has an invalid type or value.
        """
        # Type validation
        assert isinstance(self.nrz_parameters, TestParameters), "nrz_parameters must be a TestParameters instance"
        assert isinstance(self.pam4_parameters, TestParameters), "pam4_parameters must be a TestParameters instance"
        assert isinstance(self.transition_duration, float), "transition_duration must be a floating-point number"
        assert isinstance(self.stability_duration, float), "stability_duration must be a floating-point number"
        assert isinstance(self.max_transitions, int), "max_transitions must be an integer"
        assert isinstance(self.eq_convergence_threshold, float), "eq_convergence_threshold must be a floating-point number"
        assert isinstance(self.transition_method, str), "transition_method must be a string"
        
        # Value range validation
        assert self.transition_duration > 0.0, "transition_duration must be positive"
        assert self.stability_duration > 0.0, "stability_duration must be positive"
        assert self.max_transitions > 0, "max_transitions must be positive"
        assert self.eq_convergence_threshold > 0.0, "eq_convergence_threshold must be positive"
        assert self.transition_method in ['fast', 'stable', 'gradual'], \
            "transition_method must be one of: 'fast', 'stable', 'gradual'"
    
    @classmethod
    def create_default_parameters(cls, pcie_gen: PCIeGen) -> 'DualModeTestParameters':
        """
        Create default dual-mode test parameters based on PCIe generation.
        
        Args:
            pcie_gen (PCIeGen): PCIe generation.
        
        Returns:
            DualModeTestParameters: Default test parameters.
            
        Raises:
            AssertionError: If input is invalid.
            ValueError: If the PCIe generation does not support dual-mode.
        """
        assert isinstance(pcie_gen, PCIeGen), "pcie_gen must be a PCIeGen enum value"
        
        # Only Gen5 and Gen6 support dual-mode
        if pcie_gen not in [PCIeGen.GEN5, PCIeGen.GEN6]:
            raise ValueError(f"Dual-mode operation is not supported for {pcie_gen.name}")
        
        # Create default NRZ and PAM4 parameters
        nrz_parameters = TestParameters.create_default_parameters(pcie_gen, SignalMode.NRZ)
        pam4_parameters = TestParameters.create_default_parameters(pcie_gen, SignalMode.PAM4)
        
        # Set dual-mode specific parameters
        transition_duration = 5.0  # 5 seconds for mode transition
        stability_duration = 10.0  # 10 seconds to verify stability
        max_transitions = 5  # Test up to 5 transitions
        eq_convergence_threshold = 0.01
        transition_method = 'stable'  # Default to stable transition
        
        return cls(
            nrz_parameters=nrz_parameters,
            pam4_parameters=pam4_parameters,
            transition_duration=transition_duration,
            stability_duration=stability_duration,
            max_transitions=max_transitions,
            eq_convergence_threshold=eq_convergence_threshold,
            transition_method=transition_method
        )


@dataclass
class TransitionTestResult:
    """
    Results of a dual-mode transition test.
    
    Attributes:
        successful_transitions (int): Number of successful transitions.
        failed_transitions (int): Number of failed transitions.
        average_transition_time (float): Average time for successful transitions in seconds.
        nrz_to_pam4_success_rate (float): Success rate for NRZ to PAM4 transitions (0.0-1.0).
        pam4_to_nrz_success_rate (float): Success rate for PAM4 to NRZ transitions (0.0-1.0).
        nrz_performance (TestResult): Performance in NRZ mode after transitions.
        pam4_performance (TestResult): Performance in PAM4 mode after transitions.
        execution_time (float): Total test execution time in seconds.
        error_details (List[str]): Details of any errors encountered.
    """
    
    successful_transitions: int = 0
    failed_transitions: int = 0
    average_transition_time: float = 0.0
    nrz_to_pam4_success_rate: float = 0.0
    pam4_to_nrz_success_rate: float = 0.0
    nrz_performance: Optional[TestResult] = None
    pam4_performance: Optional[TestResult] = None
    execution_time: float = 0.0
    error_details: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """
        Validate the transition test results.
        
        Raises:
            AssertionError: If any parameter has an invalid type or value.
        """
        # Type validation
        assert isinstance(self.successful_transitions, int), "successful_transitions must be an integer"
        assert isinstance(self.failed_transitions, int), "failed_transitions must be an integer"
        assert isinstance(self.average_transition_time, float), "average_transition_time must be a floating-point number"
        assert isinstance(self.nrz_to_pam4_success_rate, float), "nrz_to_pam4_success_rate must be a floating-point number"
        assert isinstance(self.pam4_to_nrz_success_rate, float), "pam4_to_nrz_success_rate must be a floating-point number"
        assert self.nrz_performance is None or isinstance(self.nrz_performance, TestResult), \
            "nrz_performance must be a TestResult instance or None"
        assert self.pam4_performance is None or isinstance(self.pam4_performance, TestResult), \
            "pam4_performance must be a TestResult instance or None"
        assert isinstance(self.execution_time, float), "execution_time must be a floating-point number"
        assert isinstance(self.error_details, list), "error_details must be a list"
        
        # Value validation
        assert self.successful_transitions >= 0, "successful_transitions must be non-negative"
        assert self.failed_transitions >= 0, "failed_transitions must be non-negative"
        assert self.average_transition_time >= 0.0, "average_transition_time must be non-negative"
        assert 0.0 <= self.nrz_to_pam4_success_rate <= 1.0, "nrz_to_pam4_success_rate must be between 0.0 and 1.0"
        assert 0.0 <= self.pam4_to_nrz_success_rate <= 1.0, "pam4_to_nrz_success_rate must be between 0.0 and 1.0"
        assert self.execution_time >= 0.0, "execution_time must be non-negative"
        
        # Validate error_details elements
        for i, detail in enumerate(self.error_details):
            assert isinstance(detail, str), f"error_details[{i}] must be a string"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the transition test result to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the transition test result.
        """
        return {
            'successful_transitions': self.successful_transitions,
            'failed_transitions': self.failed_transitions,
            'average_transition_time': self.average_transition_time,
            'nrz_to_pam4_success_rate': self.nrz_to_pam4_success_rate,
            'pam4_to_nrz_success_rate': self.pam4_to_nrz_success_rate,
            'nrz_performance': self.nrz_performance.to_dict() if self.nrz_performance else None,
            'pam4_performance': self.pam4_performance.to_dict() if self.pam4_performance else None,
            'execution_time': self.execution_time,
            'error_details': self.error_details.copy()
        }
    
    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'TransitionTestResult':
        """
        Create a TransitionTestResult instance from a dictionary.
        
        Args:
            result_dict (Dict[str, Any]): Dictionary containing transition test result values.
            
        Returns:
            TransitionTestResult: A new TransitionTestResult instance.
            
        Raises:
            AssertionError: If any required key is missing or has an invalid type.
        """
        # Ensure all required keys are present
        required_keys = [
            'successful_transitions', 'failed_transitions', 'average_transition_time',
            'nrz_to_pam4_success_rate', 'pam4_to_nrz_success_rate',
            'execution_time', 'error_details'
        ]
        for key in required_keys:
            assert key in result_dict, f"{key} key is missing in result_dict"
        
        # Convert TestResult dictionaries if present
        nrz_performance = None
        if result_dict.get('nrz_performance'):
            nrz_performance = TestResult.from_dict(result_dict['nrz_performance'])
        
        pam4_performance = None
        if result_dict.get('pam4_performance'):
            pam4_performance = TestResult.from_dict(result_dict['pam4_performance'])
        
        return cls(
            successful_transitions=int(result_dict['successful_transitions']),
            failed_transitions=int(result_dict['failed_transitions']),
            average_transition_time=float(result_dict['average_transition_time']),
            nrz_to_pam4_success_rate=float(result_dict['nrz_to_pam4_success_rate']),
            pam4_to_nrz_success_rate=float(result_dict['pam4_to_nrz_success_rate']),
            nrz_performance=nrz_performance,
            pam4_performance=pam4_performance,
            execution_time=float(result_dict['execution_time']),
            error_details=result_dict['error_details']
        )


@dataclass
class EqualizationResult:
    """
    Results of equalization optimization for dual-mode operation.
    
    Attributes:
        converged (bool): Whether equalization converged successfully.
        iterations (int): Number of iterations performed.
        nrz_taps (List[float]): Optimized FFE tap values for NRZ mode.
        pam4_taps (List[float]): Optimized FFE tap values for PAM4 mode.
        compromise_taps (List[float]): Compromise FFE taps for both modes.
        nrz_performance (float): Performance metric for NRZ mode (0.0-1.0).
        pam4_performance (float): Performance metric for PAM4 mode (0.0-1.0).
        combined_performance (float): Combined performance metric (0.0-1.0).
        execution_time (float): Total optimization time in seconds.
    """
    
    converged: bool = False
    iterations: int = 0
    nrz_taps: List[float] = field(default_factory=list)
    pam4_taps: List[float] = field(default_factory=list)
    compromise_taps: List[float] = field(default_factory=list)
    nrz_performance: float = 0.0
    pam4_performance: float = 0.0
    combined_performance: float = 0.0
    execution_time: float = 0.0
    
    def __post_init__(self) -> None:
        """
        Validate the equalization results.
        
        Raises:
            AssertionError: If any parameter has an invalid type or value.
        """
        # Type validation
        assert isinstance(self.converged, bool), "converged must be a boolean"
        assert isinstance(self.iterations, int), "iterations must be an integer"
        assert isinstance(self.nrz_taps, list), "nrz_taps must be a list"
        assert isinstance(self.pam4_taps, list), "pam4_taps must be a list"
        assert isinstance(self.compromise_taps, list), "compromise_taps must be a list"
        assert isinstance(self.nrz_performance, float), "nrz_performance must be a floating-point number"
        assert isinstance(self.pam4_performance, float), "pam4_performance must be a floating-point number"
        assert isinstance(self.combined_performance, float), "combined_performance must be a floating-point number"
        assert isinstance(self.execution_time, float), "execution_time must be a floating-point number"
        
        # Value validation
        assert self.iterations >= 0, "iterations must be non-negative"
        assert 0.0 <= self.nrz_performance <= 1.0, "nrz_performance must be between 0.0 and 1.0"
        assert 0.0 <= self.pam4_performance <= 1.0, "pam4_performance must be between 0.0 and 1.0"
        assert 0.0 <= self.combined_performance <= 1.0, "combined_performance must be between 0.0 and 1.0"
        assert self.execution_time >= 0.0, "execution_time must be non-negative"
        
        # Validate tap lists elements
        for i, tap in enumerate(self.nrz_taps):
            assert isinstance(tap, float), f"nrz_taps[{i}] must be a floating-point number"
        
        for i, tap in enumerate(self.pam4_taps):
            assert isinstance(tap, float), f"pam4_taps[{i}] must be a floating-point number"
        
        for i, tap in enumerate(self.compromise_taps):
            assert isinstance(tap, float), f"compromise_taps[{i}] must be a floating-point number"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the equalization result to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the equalization result.
        """
        return {
            'converged': self.converged,
            'iterations': self.iterations,
            'nrz_taps': self.nrz_taps.copy(),
            'pam4_taps': self.pam4_taps.copy(),
            'compromise_taps': self.compromise_taps.copy(),
            'nrz_performance': self.nrz_performance,
            'pam4_performance': self.pam4_performance,
            'combined_performance': self.combined_performance,
            'execution_time': self.execution_time
        }
    
    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'EqualizationResult':
        """
        Create an EqualizationResult instance from a dictionary.
        
        Args:
            result_dict (Dict[str, Any]): Dictionary containing equalization result values.
            
        Returns:
            EqualizationResult: A new EqualizationResult instance.
            
        Raises:
            AssertionError: If any required key is missing or has an invalid type.
        """
        # Ensure all required keys are present
        required_keys = [
            'converged', 'iterations', 'nrz_taps', 'pam4_taps', 'compromise_taps',
            'nrz_performance', 'pam4_performance', 'combined_performance', 'execution_time'
        ]
        for key in required_keys:
            assert key in result_dict, f"{key} key is missing in result_dict"
        
        # Convert tap lists to ensure float values
        nrz_taps = [float(tap) for tap in result_dict['nrz_taps']]
        pam4_taps = [float(tap) for tap in result_dict['pam4_taps']]
        compromise_taps = [float(tap) for tap in result_dict['compromise_taps']]
        
        return cls(
            converged=bool(result_dict['converged']),
            iterations=int(result_dict['iterations']),
            nrz_taps=nrz_taps,
            pam4_taps=pam4_taps,
            compromise_taps=compromise_taps,
            nrz_performance=float(result_dict['nrz_performance']),
            pam4_performance=float(result_dict['pam4_performance']),
            combined_performance=float(result_dict['combined_performance']),
            execution_time=float(result_dict['execution_time'])
        )


class DualModeTestSequence:
    """
    Class for executing dual-mode test sequences.
    
    This class provides methods for testing transitions between NRZ and PAM4 modes
    and optimizing equalization for dual-mode operation.
    """
    
    def __init__(
        self,
        pcie_gen: PCIeGen,
        parameters: Optional[DualModeTestParameters] = None
    ) -> None:
        """
        Initialize a new DualModeTestSequence.
        
        Args:
            pcie_gen (PCIeGen): PCIe generation.
            parameters (Optional[DualModeTestParameters]): Test parameters.
                If None, default parameters for the specified PCIe gen are used.
                
        Raises:
            AssertionError: If pcie_gen is invalid.
            ValueError: If the PCIe generation does not support dual-mode.
        """
        assert isinstance(pcie_gen, PCIeGen), "pcie_gen must be a PCIeGen enum value"
        
        # Only Gen5 and Gen6 support dual-mode
        if pcie_gen not in [PCIeGen.GEN5, PCIeGen.GEN6]:
            raise ValueError(f"Dual-mode operation is not supported for {pcie_gen.name}")
        
        self.pcie_gen = pcie_gen
        
        if parameters is None:
            self.parameters = DualModeTestParameters.create_default_parameters(pcie_gen)
        else:
            assert isinstance(parameters, DualModeTestParameters), "parameters must be a DualModeTestParameters instance"
            self.parameters = parameters
        
        # Initialize NRZ and PAM4 test sequences
        self.nrz_sequence = PCIeTestSequence(
            pcie_gen=pcie_gen,
            signal_mode=SignalMode.NRZ,
            parameters=self.parameters.nrz_parameters
        )
        
        self.pam4_sequence = PCIeTestSequence(
            pcie_gen=pcie_gen,
            signal_mode=SignalMode.PAM4,
            parameters=self.parameters.pam4_parameters
        )
        
        self._instrument_controller = None  # Will be set externally
        self._mode_controller = None  # Will be set externally
        self._last_transition_result = None
        self._last_eq_result = None
        self._start_time = 0.0
    
    def test_mode_transition(self, transition_mode: TransitionMode) -> TransitionTestResult:
        """
        Test transitions between NRZ and PAM4 modes.
        
        Args:
            transition_mode (TransitionMode): Type of transition to test.
            
        Returns:
            TransitionTestResult: Transition test results.
            
        Raises:
            AssertionError: If transition_mode is invalid.
            RuntimeError: If an error occurs during test execution.
        """
        assert isinstance(transition_mode, TransitionMode), "transition_mode must be a TransitionMode enum value"
        
        self._start_time = time.time()
        successful_transitions = 0
        failed_transitions = 0
        transition_times = []
        error_details = []
        
        try:
            # Configure test based on transition mode
            if transition_mode == TransitionMode.NRZ_TO_PAM4:
                transitions = [(SignalMode.NRZ, SignalMode.PAM4)]
            elif transition_mode == TransitionMode.PAM4_TO_NRZ:
                transitions = [(SignalMode.PAM4, SignalMode.NRZ)]
            else:  # TransitionMode.BIDIRECTIONAL
                transitions = [
                    (SignalMode.NRZ, SignalMode.PAM4),
                    (SignalMode.PAM4, SignalMode.NRZ)
                ]
            
            # Test the specified number of transitions
            for _ in range(self.parameters.max_transitions):
                for from_mode, to_mode in transitions:
                    # Perform the transition
                    transition_start = time.time()
                    success, error = self._transition_mode(from_mode, to_mode)
                    transition_time = time.time() - transition_start
                    
                    # Record results
                    if success:
                        successful_transitions += 1
                        transition_times.append(transition_time)
                    else:
                        failed_transitions += 1
                        error_details.append(f"Failed {from_mode.name} to {to_mode.name}: {error}")
                    
                    # Verify stability in new mode
                    time.sleep(min(1.0, self.parameters.stability_duration / 10.0))  # Reduced for simulation
            
            # Calculate success rates
            nrz_to_pam4_attempts = 0
            nrz_to_pam4_successes = 0
            pam4_to_nrz_attempts = 0
            pam4_to_nrz_successes = 0
            
            for i, (from_mode, to_mode) in enumerate(transitions * self.parameters.max_transitions):
                if from_mode == SignalMode.NRZ and to_mode == SignalMode.PAM4:
                    nrz_to_pam4_attempts += 1
                    if i < successful_transitions:
                        nrz_to_pam4_successes += 1
                elif from_mode == SignalMode.PAM4 and to_mode == SignalMode.NRZ:
                    pam4_to_nrz_attempts += 1
                    if i < successful_transitions:
                        pam4_to_nrz_successes += 1
            
            nrz_to_pam4_success_rate = float(nrz_to_pam4_successes / max(1, nrz_to_pam4_attempts))
            pam4_to_nrz_success_rate = float(pam4_to_nrz_successes / max(1, pam4_to_nrz_attempts))
            
            # Measure performance in each mode
            nrz_performance = None
            pam4_performance = None
            
            if transition_mode in [TransitionMode.PAM4_TO_NRZ, TransitionMode.BIDIRECTIONAL]:
                # Ensure we're in NRZ mode
                self._transition_mode(SignalMode.PAM4, SignalMode.NRZ)
                # Measure NRZ performance
                nrz_performance = self.nrz_sequence.run_test(TestType.COMPLIANCE)
            
            if transition_mode in [TransitionMode.NRZ_TO_PAM4, TransitionMode.BIDIRECTIONAL]:
                # Ensure we're in PAM4 mode
                self._transition_mode(SignalMode.NRZ, SignalMode.PAM4)
                # Measure PAM4 performance
                pam4_performance = self.pam4_sequence.run_test(TestType.COMPLIANCE)
            
            # Calculate average transition time
            avg_transition_time = float(sum(transition_times) / max(len(transition_times), 1))
            
            # Create and return result
            result = TransitionTestResult(
                successful_transitions=successful_transitions,
                failed_transitions=failed_transitions,
                average_transition_time=avg_transition_time,
                nrz_to_pam4_success_rate=nrz_to_pam4_success_rate,
                pam4_to_nrz_success_rate=pam4_to_nrz_success_rate,
                nrz_performance=nrz_performance,
                pam4_performance=pam4_performance,
                execution_time=time.time() - self._start_time,
                error_details=error_details
            )
            
            self._last_transition_result = result
            return result
            
        except Exception as e:
            # Create failure result
            result = TransitionTestResult(
                successful_transitions=successful_transitions,
                failed_transitions=failed_transitions + 1,
                execution_time=time.time() - self._start_time,
                error_details=[f"Test execution error: {str(e)}"]
            )
            
            self._last_transition_result = result
            return result
    
    def optimize_equalization(self, eq_mode: EqualizationMode) -> EqualizationResult:
        """
        Optimize equalization for dual-mode operation.
        
        Args:
            eq_mode (EqualizationMode): Equalization optimization mode.
            
        Returns:
            EqualizationResult: Equalization optimization results.
            
        Raises:
            AssertionError: If eq_mode is invalid.
            RuntimeError: If an error occurs during optimization.
        """
        assert isinstance(eq_mode, EqualizationMode), "eq_mode must be a EqualizationMode enum value"
        
        self._start_time = time.time()
        iterations = 0
        
        try:
            # Initialize tap values
            nrz_taps = self._initialize_taps(mode=SignalMode.NRZ)
            pam4_taps = self._initialize_taps(mode=SignalMode.PAM4)
            compromise_taps = self._initialize_compromise_taps(nrz_taps, pam4_taps)
            
            # Set initial performance metrics
            nrz_performance = self._evaluate_taps(nrz_taps, mode=SignalMode.NRZ)
            pam4_performance = self._evaluate_taps(pam4_taps, mode=SignalMode.PAM4)
            compromise_nrz_perf = self._evaluate_taps(compromise_taps, mode=SignalMode.NRZ)
            compromise_pam4_perf = self._evaluate_taps(compromise_taps, mode=SignalMode.PAM4)
            
            # Apply weights based on eq_mode
            nrz_weight, pam4_weight = self._get_mode_weights(eq_mode)
            
            combined_performance = self._calculate_combined_performance(
                compromise_nrz_perf, compromise_pam4_perf, nrz_weight, pam4_weight
            )
            
            # Optimization loop
            max_iterations = 50  # Limit iterations
            converged = False
            
            for i in range(max_iterations):
                iterations = i + 1
                
                # Choose optimization strategy based on mode
                if eq_mode == EqualizationMode.NRZ_ONLY:
                    nrz_taps = self._optimize_taps(nrz_taps, mode=SignalMode.NRZ)
                    compromise_taps = nrz_taps.copy()
                elif eq_mode == EqualizationMode.PAM4_ONLY:
                    pam4_taps = self._optimize_taps(pam4_taps, mode=SignalMode.PAM4)
                    compromise_taps = pam4_taps.copy()
                else:
                    # For balanced and priority modes
                    nrz_taps = self._optimize_taps(nrz_taps, mode=SignalMode.NRZ)
                    pam4_taps = self._optimize_taps(pam4_taps, mode=SignalMode.PAM4)
                    compromise_taps = self._optimize_compromise_taps(
                        nrz_taps, pam4_taps, nrz_weight, pam4_weight
                    )
                
                # Evaluate new performance
                nrz_performance = self._evaluate_taps(nrz_taps, mode=SignalMode.NRZ)
                pam4_performance = self._evaluate_taps(pam4_taps, mode=SignalMode.PAM4)
                compromise_nrz_perf = self._evaluate_taps(compromise_taps, mode=SignalMode.NRZ)
                compromise_pam4_perf = self._evaluate_taps(compromise_taps, mode=SignalMode.PAM4)
                
                new_combined_performance = self._calculate_combined_performance(
                    compromise_nrz_perf, compromise_pam4_perf, nrz_weight, pam4_weight
                )
                
                # Check convergence
                if abs(new_combined_performance - combined_performance) < self.parameters.eq_convergence_threshold:
                    converged = True
                    break
                
                combined_performance = new_combined_performance
                
                # Add a small delay for simulation
                time.sleep(0.1)
            
            # Create and return result
            result = EqualizationResult(
                converged=converged,
                iterations=iterations,
                nrz_taps=nrz_taps,
                pam4_taps=pam4_taps,
                compromise_taps=compromise_taps,
                nrz_performance=nrz_performance,
                pam4_performance=pam4_performance,
                combined_performance=combined_performance,
                execution_time=time.time() - self._start_time
            )
            
            self._last_eq_result = result
            return result
            
        except Exception as e:
            # Create failure result
            result = EqualizationResult(
                converged=False,
                iterations=iterations,
                execution_time=time.time() - self._start_time
            )
            
            self._last_eq_result = result
            return result
    
    def get_last_transition_result(self) -> Optional[TransitionTestResult]:
        """
        Get the result of the last transition test.
        
        Returns:
            Optional[TransitionTestResult]: Result of the last transition test, or None if no test has been run.
        """
        return self._last_transition_result
    
    def get_last_eq_result(self) -> Optional[EqualizationResult]:
        """
        Get the result of the last equalization optimization.
        
        Returns:
            Optional[EqualizationResult]: Result of the last optimization, or None if no optimization has been run.
        """
        return self._last_eq_result
    
    def _transition_mode(self, from_mode: SignalMode, to_mode: SignalMode) -> Tuple[bool, str]:
        """
        Perform a mode transition.
        
        Args:
            from_mode (SignalMode): Starting signal mode.
            to_mode (SignalMode): Target signal mode.
            
        Returns:
            Tuple[bool, str]: Success flag and error message (if any).
            
        Raises:
            RuntimeError: If an error occurs during transition.
        """
        # Verify types
        assert isinstance(from_mode, SignalMode), "from_mode must be a SignalMode enum value"
        assert isinstance(to_mode, SignalMode), "to_mode must be a SignalMode enum value"
        
        # Skip if already in the target mode
        if from_mode == to_mode:
            return True, ""
        
        try:
            # In a real implementation, this would use the mode_controller
            # Here we just simulate the transition with a delay
            transition_method = self.parameters.transition_method
            
            if transition_method == 'fast':
                # Fast transition prioritizes speed over stability
                time.sleep(0.5)
            elif transition_method == 'stable':
                # Stable transition includes more validation
                time.sleep(1.0)
            else:  # 'gradual'
                # Gradual transition is slowest but most reliable
                time.sleep(1.5)
            
            # Simulate a small chance of failure
            import random
            if random.random() < 0.05:  # 5% failure rate
                return False, "Transition failed due to timing issues"
            
            return True, ""
            
        except Exception as e:
            return False, str(e)
    
    def _initialize_taps(self, mode: SignalMode) -> List[float]:
        """
        Initialize FFE tap values for the specified mode.
        
        Args:
            mode (SignalMode): Signal mode (NRZ or PAM4).
            
        Returns:
            List[float]: Initial tap values.
            
        Raises:
            AssertionError: If mode is invalid.
        """
        assert isinstance(mode, SignalMode), "mode must be a SignalMode enum value"
        
        # Create initial tap values based on mode
        if mode == SignalMode.NRZ:
            # Typical 5-tap FFE for NRZ
            return [-0.05, -0.1, 1.0, -0.3, -0.1]
        else:  # PAM4
            # Typical 7-tap FFE for PAM4 (more taps for complex signal)
            return [-0.02, -0.05, -0.1, 1.0, -0.25, -0.1, -0.05]
    
    def _initialize_compromise_taps(self, nrz_taps: List[float], pam4_taps: List[float]) -> List[float]:
        """
        Initialize compromise tap values that work for both modes.
        
        Args:
            nrz_taps (List[float]): NRZ-optimized tap values.
            pam4_taps (List[float]): PAM4-optimized tap values.
            
        Returns:
            List[float]: Compromise tap values.
            
        Raises:
            AssertionError: If inputs have invalid types.
        """
        # Validate input types
        assert isinstance(nrz_taps, list), "nrz_taps must be a list"
        assert isinstance(pam4_taps, list), "pam4_taps must be a list"
        assert all(isinstance(t, float) for t in nrz_taps), "all elements in nrz_taps must be floating-point numbers"
        assert all(isinstance(t, float) for t in pam4_taps), "all elements in pam4_taps must be floating-point numbers"
        
        # For compromise, we need a common length
        # Typically use the longer of the two (PAM4 usually)
        if len(nrz_taps) >= len(pam4_taps):
            base_taps = nrz_taps.copy()
        else:
            base_taps = pam4_taps.copy()
        
        # Create a simple weighted average of the available taps
        compromise_taps = base_taps.copy()
        
        # Weighted average of the overlapping part
        min_len = min(len(nrz_taps), len(pam4_taps))
        for i in range(min_len):
            compromise_taps[i] = (nrz_taps[i] + pam4_taps[i]) / 2.0
        
        return compromise_taps
    
    def _get_mode_weights(self, eq_mode: EqualizationMode) -> Tuple[float, float]:
        """
        Get weighting factors for NRZ and PAM4 based on equalization mode.
        
        Args:
            eq_mode (EqualizationMode): Equalization optimization mode.
            
        Returns:
            Tuple[float, float]: NRZ weight and PAM4 weight.
            
        Raises:
            AssertionError: If eq_mode is invalid.
        """
        assert isinstance(eq_mode, EqualizationMode), "eq_mode must be a EqualizationMode enum value"
        
        if eq_mode == EqualizationMode.NRZ_ONLY:
            return 1.0, 0.0
        elif eq_mode == EqualizationMode.PAM4_ONLY:
            return 0.0, 1.0
        elif eq_mode == EqualizationMode.NRZ_PRIORITY:
            return 0.7, 0.3
        elif eq_mode == EqualizationMode.PAM4_PRIORITY:
            return 0.3, 0.7
        else:  # EqualizationMode.BALANCED
            return 0.5, 0.5
    
    def _calculate_combined_performance(
        self, 
        nrz_perf: float, 
        pam4_perf: float, 
        nrz_weight: float, 
        pam4_weight: float
    ) -> float:
        """
        Calculate combined performance metric from individual mode metrics.
        
        Args:
            nrz_perf (float): NRZ performance metric (0.0-1.0).
            pam4_perf (float): PAM4 performance metric (0.0-1.0).
            nrz_weight (float): Weight for NRZ performance (0.0-1.0).
            pam4_weight (float): Weight for PAM4 performance (0.0-1.0).
            
        Returns:
            float: Combined performance metric (0.0-1.0).
            
        Raises:
            AssertionError: If inputs have invalid types or values.
        """
        # Validate input types
        assert isinstance(nrz_perf, float), "nrz_perf must be a floating-point number"
        assert isinstance(pam4_perf, float), "pam4_perf must be a floating-point number"
        assert isinstance(nrz_weight, float), "nrz_weight must be a floating-point number"
        assert isinstance(pam4_weight, float), "pam4_weight must be a floating-point number"
        
        # Validate input ranges
        assert 0.0 <= nrz_perf <= 1.0, "nrz_perf must be between 0.0 and 1.0"
        assert 0.0 <= pam4_perf <= 1.0, "pam4_perf must be between 0.0 and 1.0"
        assert 0.0 <= nrz_weight <= 1.0, "nrz_weight must be between 0.0 and 1.0"
        assert 0.0 <= pam4_weight <= 1.0, "pam4_weight must be between 0.0 and 1.0"
        assert abs(nrz_weight + pam4_weight - 1.0) < 1e-6, "weights must sum to 1.0"
        
        # Calculate weighted sum
        return nrz_perf * nrz_weight + pam4_perf * pam4_weight
    
    def _evaluate_taps(self, taps: List[float], mode: SignalMode) -> float:
        """
        Evaluate performance of tap values for a specific mode.
        
        Args:
            taps (List[float]): FFE tap values to evaluate.
            mode (SignalMode): Signal mode (NRZ or PAM4).
            
        Returns:
            float: Performance metric (0.0-1.0).
            
        Raises:
            AssertionError: If inputs have invalid types.
        """
        # Validate input types
        assert isinstance(taps, list), "taps must be a list"
        assert all(isinstance(t, float) for t in taps), "all elements in taps must be floating-point numbers"
        assert isinstance(mode, SignalMode), "mode must be a SignalMode enum value"
        
        # Check for empty taps list
        if not taps:
            return 0.0
        
        # Simulate evaluation with a simple model
        # In reality, this would involve measuring eye height, width, etc.
        
        # Normalize tap values
        main_tap_idx = len(taps) // 2
        if main_tap_idx >= len(taps) or taps[main_tap_idx] == 0.0:
            return 0.0  # Invalid tap configuration
            
        normalized_taps = [t / taps[main_tap_idx] for t in taps]
        
        # Different performance models based on mode
        if mode == SignalMode.NRZ:
            # For NRZ, more pre-cursor emphasis is generally better
            pre_cursor_sum = sum(abs(normalized_taps[i]) for i in range(main_tap_idx))
            post_cursor_sum = sum(abs(normalized_taps[i]) for i in range(main_tap_idx + 1, len(normalized_taps)))
            
            # Handle division by zero
            if post_cursor_sum < 1e-10:
                post_cursor_sum = 1e-10
                
            # Ideal pre/post ratio for NRZ is typically around 1:3
            ideal_ratio = 1.0 / 3.0
            actual_ratio = pre_cursor_sum / post_cursor_sum
            ratio_score = 1.0 - min(1.0, abs(ideal_ratio - actual_ratio) / ideal_ratio)
            
            # Performance is a function of how close the ratio is to ideal
            return 0.5 + 0.5 * ratio_score
            
        else:  # PAM4
            # For PAM4, more balanced pre/post is typically better
            pre_cursor_sum = sum(abs(normalized_taps[i]) for i in range(main_tap_idx))
            post_cursor_sum = sum(abs(normalized_taps[i]) for i in range(main_tap_idx + 1, len(normalized_taps)))
            
            # Handle division by zero
            if post_cursor_sum < 1e-10:
                post_cursor_sum = 1e-10
                
            # Ideal pre/post ratio for PAM4 is typically around 1:2
            ideal_ratio = 1.0 / 2.0
            actual_ratio = pre_cursor_sum / post_cursor_sum
            ratio_score = 1.0 - min(1.0, abs(ideal_ratio - actual_ratio) / ideal_ratio)
            
            # Also consider total number of active taps
            active_taps = sum(1 for t in normalized_taps if abs(t) > 0.05)
            tap_score = min(1.0, active_taps / 5.0)  # PAM4 benefits from more active taps
            
            # Performance is a weighted combination
            return 0.3 + 0.4 * ratio_score + 0.3 * tap_score
    
    def _optimize_taps(self, current_taps: List[float], mode: SignalMode) -> List[float]:
        """
        Optimize tap values for a specific mode.
        
        Args:
            current_taps (List[float]): Current FFE tap values.
            mode (SignalMode): Signal mode (NRZ or PAM4).
            
        Returns:
            List[float]: Optimized tap values.
            
        Raises:
            AssertionError: If inputs have invalid types.
        """
        # Validate input types
        assert isinstance(current_taps, list), "current_taps must be a list"
        assert all(isinstance(t, float) for t in current_taps), "all elements in current_taps must be floating-point numbers"
        assert isinstance(mode, SignalMode), "mode must be a SignalMode enum value"
        
        # Check for empty taps list
        if not current_taps:
            # Return a default set of taps if empty
            return self._initialize_taps(mode)
        
        # For simulation, make small random adjustments to tap values
        # In a real implementation, this would use a proper optimization algorithm
        import random
        
        main_tap_idx = len(current_taps) // 2
        if main_tap_idx >= len(current_taps):
            main_tap_idx = len(current_taps) - 1
            
        new_taps = current_taps.copy()
        
        # Adjust non-main taps
        for i in range(len(new_taps)):
            if i != main_tap_idx:
                # Small random adjustments
                adjustment = random.uniform(-0.02, 0.02)
                new_taps[i] = new_taps[i] + adjustment
        
        # Ensure main tap is non-zero
        if new_taps[main_tap_idx] == 0.0:
            new_taps[main_tap_idx] = 1.0
        
        # Normalize to keep main tap at 1.0
        main_tap_value = new_taps[main_tap_idx]
        new_taps = [t / main_tap_value for t in new_taps]
        
        # Evaluate both tap sets
        current_perf = self._evaluate_taps(current_taps, mode)
        new_perf = self._evaluate_taps(new_taps, mode)
        
        # Keep the better set
        if new_perf > current_perf:
            return new_taps
        else:
            return current_taps
    
    def _optimize_compromise_taps(
        self, 
        nrz_taps: List[float], 
        pam4_taps: List[float],
        nrz_weight: float,
        pam4_weight: float
    ) -> List[float]:
        """
        Optimize tap values that work well for both modes.
        
        Args:
            nrz_taps (List[float]): NRZ-optimized tap values.
            pam4_taps (List[float]): PAM4-optimized tap values.
            nrz_weight (float): Weight for NRZ performance (0.0-1.0).
            pam4_weight (float): Weight for PAM4 performance (0.0-1.0).
            
        Returns:
            List[float]: Optimized compromise tap values.
            
        Raises:
            AssertionError: If inputs have invalid types.
        """
        # Validate input types
        assert isinstance(nrz_taps, list), "nrz_taps must be a list"
        assert isinstance(pam4_taps, list), "pam4_taps must be a list"
        assert all(isinstance(t, float) for t in nrz_taps), "all elements in nrz_taps must be floating-point numbers"
        assert all(isinstance(t, float) for t in pam4_taps), "all elements in pam4_taps must be floating-point numbers"
        assert isinstance(nrz_weight, float), "nrz_weight must be a floating-point number"
        assert isinstance(pam4_weight, float), "pam4_weight must be a floating-point number"
        
        # Handle empty lists
        if not nrz_taps or not pam4_taps:
            return self._initialize_compromise_taps(
                self._initialize_taps(SignalMode.NRZ),
                self._initialize_taps(SignalMode.PAM4)
            )
        
        # For compromise, we need a common length
        # Typically use the longer of the two (PAM4 usually)
        if len(nrz_taps) >= len(pam4_taps):
            base_taps = nrz_taps.copy()
            short_taps = pam4_taps
        else:
            base_taps = pam4_taps.copy()
            short_taps = nrz_taps
        
        # Create weighted average of the available taps
        compromise_taps = base_taps.copy()
        main_tap_idx = len(compromise_taps) // 2
        
        # Weighted average of the overlapping part
        min_len = min(len(nrz_taps), len(pam4_taps))
        
        # Apply weights based on priority
        for i in range(len(compromise_taps)):
            if i == main_tap_idx:
                # Main tap is always 1.0
                compromise_taps[i] = 1.0
            elif i < min_len:
                # Apply weighted average for overlapping taps
                nrz_tap = nrz_taps[i]
                pam4_tap = short_taps[i]
                compromise_taps[i] = nrz_tap * nrz_weight + pam4_tap * pam4_weight
        
        # Ensure main tap is non-zero
        if compromise_taps[main_tap_idx] == 0.0:
            compromise_taps[main_tap_idx] = 1.0
            
        # Normalize to keep main tap at 1.0
        main_tap_value = compromise_taps[main_tap_idx]
        compromise_taps = [t / main_tap_value for t in compromise_taps]
        
        # Evaluate compromise taps performance
        nrz_perf = self._evaluate_taps(compromise_taps, SignalMode.NRZ)
        pam4_perf = self._evaluate_taps(compromise_taps, SignalMode.PAM4)
        
        # Fine-tune based on performance
        if nrz_perf < 0.3:  # Too poor for NRZ
            # Adjust towards NRZ
            for i in range(min_len):
                if i != main_tap_idx:
                    nrz_tap = nrz_taps[i] if i < len(nrz_taps) else 0.0
                    compromise_taps[i] = (compromise_taps[i] + nrz_tap) / 2.0
        
        if pam4_perf < 0.3:  # Too poor for PAM4
            # Adjust towards PAM4
            for i in range(min_len):
                if i != main_tap_idx:
                    pam4_tap = pam4_taps[i] if i < len(pam4_taps) else 0.0
                    compromise_taps[i] = (compromise_taps[i] + pam4_tap) / 2.0
        
        # Ensure main tap is still 1.0 after adjustments
        compromise_taps[main_tap_idx] = 1.0
        
        return compromise_taps
