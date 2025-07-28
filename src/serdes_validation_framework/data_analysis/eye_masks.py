"""
Eye Mask Analysis Module

This module provides comprehensive eye mask validation with visual overlays
for different SerDes protocols including USB4, PCIe, and Ethernet.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class MaskType(Enum):
    """Types of eye masks"""
    RECTANGULAR = "rectangular"
    DIAMOND = "diamond"
    HEXAGONAL = "hexagonal"
    CUSTOM = "custom"


@dataclass
class EyeMaskPoint:
    """Single point in an eye mask"""
    time_ui: float  # Time in UI (Unit Intervals)
    voltage_mv: float  # Voltage in mV
    
    def __post_init__(self):
        """Validate mask point"""
        assert -1.0 <= self.time_ui <= 1.0, f"Time UI must be between -1.0 and 1.0, got {self.time_ui}"


@dataclass
class EyeMask:
    """Eye mask definition for protocol compliance"""
    protocol: str
    data_rate_gbps: float
    mask_type: MaskType
    points: List[EyeMaskPoint]
    voltage_swing_mv: float
    description: str = ""
    
    def __post_init__(self):
        """Validate eye mask"""
        assert len(self.points) >= 3, "Eye mask must have at least 3 points"
        assert self.data_rate_gbps > 0, "Data rate must be positive"
        assert self.voltage_swing_mv > 0, "Voltage swing must be positive"


@dataclass
class MaskViolation:
    """Eye mask violation details"""
    point_index: int
    measured_voltage: float
    mask_voltage: float
    violation_margin: float
    time_ui: float
    severity: str  # "minor", "major", "critical"


@dataclass
class EyeMaskResult:
    """Result of eye mask analysis"""
    protocol: str
    mask_passed: bool
    violations: List[MaskViolation]
    margin_percentage: float
    worst_violation: Optional[MaskViolation]
    eye_opening_percentage: float
    compliance_level: str  # "pass", "marginal", "fail"


class EyeMaskGenerator:
    """Generate standard eye masks for different protocols"""
    
    @staticmethod
    def create_usb4_mask(data_rate_gbps: float = 20.0) -> EyeMask:
        """Create USB4 eye mask"""
        # USB4 specification eye mask points (simplified)
        points = [
            EyeMaskPoint(-0.5, 100),   # Left side, top
            EyeMaskPoint(-0.35, 50),   # Left inner, mid-top
            EyeMaskPoint(-0.15, 25),   # Center left, low
            EyeMaskPoint(0.0, 20),     # Center, minimum
            EyeMaskPoint(0.15, 25),    # Center right, low
            EyeMaskPoint(0.35, 50),    # Right inner, mid-top
            EyeMaskPoint(0.5, 100),    # Right side, top
            EyeMaskPoint(0.35, -50),   # Right inner, mid-bottom
            EyeMaskPoint(0.15, -25),   # Center right, low bottom
            EyeMaskPoint(0.0, -20),    # Center, minimum bottom
            EyeMaskPoint(-0.15, -25),  # Center left, low bottom
            EyeMaskPoint(-0.35, -50),  # Left inner, mid-bottom
        ]
        
        return EyeMask(
            protocol="USB4",
            data_rate_gbps=data_rate_gbps,
            mask_type=MaskType.HEXAGONAL,
            points=points,
            voltage_swing_mv=800,
            description=f"USB4 {data_rate_gbps} Gbps eye mask per specification"
        )
    
    @staticmethod
    def create_pcie_mask(data_rate_gbps: float = 32.0) -> EyeMask:
        """Create PCIe eye mask"""
        # PCIe 6.0 specification eye mask points
        points = [
            EyeMaskPoint(-0.45, 150),  # Left side, top
            EyeMaskPoint(-0.25, 75),   # Left inner, mid
            EyeMaskPoint(-0.1, 40),    # Center left, low
            EyeMaskPoint(0.0, 35),     # Center, minimum
            EyeMaskPoint(0.1, 40),     # Center right, low
            EyeMaskPoint(0.25, 75),    # Right inner, mid
            EyeMaskPoint(0.45, 150),   # Right side, top
            EyeMaskPoint(0.25, -75),   # Right inner, mid-bottom
            EyeMaskPoint(0.1, -40),    # Center right, low bottom
            EyeMaskPoint(0.0, -35),    # Center, minimum bottom
            EyeMaskPoint(-0.1, -40),   # Center left, low bottom
            EyeMaskPoint(-0.25, -75),  # Left inner, mid-bottom
        ]
        
        return EyeMask(
            protocol="PCIe",
            data_rate_gbps=data_rate_gbps,
            mask_type=MaskType.DIAMOND,
            points=points,
            voltage_swing_mv=1200,
            description=f"PCIe {data_rate_gbps} Gbps eye mask per specification"
        )
    
    @staticmethod
    def create_ethernet_mask(data_rate_gbps: float = 112.0) -> EyeMask:
        """Create Ethernet eye mask"""
        # 224G Ethernet specification eye mask points
        points = [
            EyeMaskPoint(-0.4, 200),   # Left side, top
            EyeMaskPoint(-0.2, 120),   # Left inner, mid
            EyeMaskPoint(-0.05, 80),   # Center left, low
            EyeMaskPoint(0.0, 75),     # Center, minimum
            EyeMaskPoint(0.05, 80),    # Center right, low
            EyeMaskPoint(0.2, 120),    # Right inner, mid
            EyeMaskPoint(0.4, 200),    # Right side, top
            EyeMaskPoint(0.2, -120),   # Right inner, mid-bottom
            EyeMaskPoint(0.05, -80),   # Center right, low bottom
            EyeMaskPoint(0.0, -75),    # Center, minimum bottom
            EyeMaskPoint(-0.05, -80),  # Center left, low bottom
            EyeMaskPoint(-0.2, -120),  # Left inner, mid-bottom
        ]
        
        return EyeMask(
            protocol="Ethernet",
            data_rate_gbps=data_rate_gbps,
            mask_type=MaskType.HEXAGONAL,
            points=points,
            voltage_swing_mv=1600,
            description=f"Ethernet {data_rate_gbps} Gbps eye mask per specification"
        )


class EyeMaskAnalyzer:
    """Analyze eye diagrams against protocol-specific masks"""
    
    def __init__(self, protocol: str = "USB4", data_rate_gbps: float = 20.0):
        """Initialize eye mask analyzer"""
        self.protocol = protocol
        self.data_rate_gbps = data_rate_gbps
        self.mask_generator = EyeMaskGenerator()
        
        # Load appropriate mask
        if protocol.upper() == "USB4":
            self.mask = self.mask_generator.create_usb4_mask(data_rate_gbps)
        elif protocol.upper() == "PCIE":
            self.mask = self.mask_generator.create_pcie_mask(data_rate_gbps)
        elif protocol.upper() == "ETHERNET":
            self.mask = self.mask_generator.create_ethernet_mask(data_rate_gbps)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
        
        logger.info(f"Eye mask analyzer initialized for {protocol} at {data_rate_gbps} Gbps")
    
    def analyze_eye_against_mask(self, 
                                eye_data: np.ndarray,
                                time_axis: np.ndarray,
                                voltage_axis: np.ndarray) -> EyeMaskResult:
        """
        Analyze eye diagram against protocol mask
        
        Args:
            eye_data: 2D array of eye diagram data
            time_axis: Time axis in UI (Unit Intervals)
            voltage_axis: Voltage axis in mV
            
        Returns:
            EyeMaskResult with compliance analysis
        """
        violations = []
        
        try:
            # Convert mask points to data coordinates
            mask_violations = self._check_mask_violations(eye_data, time_axis, voltage_axis)
            violations.extend(mask_violations)
            
            # Calculate eye opening percentage
            eye_opening = self._calculate_eye_opening(eye_data, time_axis, voltage_axis)
            
            # Determine compliance level
            compliance_level = self._determine_compliance_level(violations, eye_opening)
            
            # Find worst violation
            worst_violation = None
            if violations:
                worst_violation = max(violations, key=lambda v: abs(v.violation_margin))
            
            # Calculate overall margin
            margin_percentage = self._calculate_margin_percentage(violations, eye_opening)
            
            result = EyeMaskResult(
                protocol=self.protocol,
                mask_passed=len(violations) == 0,
                violations=violations,
                margin_percentage=margin_percentage,
                worst_violation=worst_violation,
                eye_opening_percentage=eye_opening,
                compliance_level=compliance_level
            )
            
            logger.info(f"Eye mask analysis complete: {compliance_level}, {len(violations)} violations")
            return result
            
        except Exception as e:
            logger.error(f"Eye mask analysis failed: {e}")
            return EyeMaskResult(
                protocol=self.protocol,
                mask_passed=False,
                violations=[],
                margin_percentage=0.0,
                worst_violation=None,
                eye_opening_percentage=0.0,
                compliance_level="fail"
            )
    
    def _check_mask_violations(self, 
                              eye_data: np.ndarray,
                              time_axis: np.ndarray,
                              voltage_axis: np.ndarray) -> List[MaskViolation]:
        """Check for mask violations"""
        violations = []
        
        # Create mask polygon for violation checking
        mask_polygon = self._create_mask_polygon()
        
        # Sample points around the mask boundary
        for i, point in enumerate(self.mask.points):
            # Find corresponding data point
            time_idx = self._find_nearest_index(time_axis, point.time_ui)
            voltage_idx = self._find_nearest_index(voltage_axis, point.voltage_mv)
            
            if 0 <= time_idx < len(time_axis) and 0 <= voltage_idx < len(voltage_axis):
                measured_value = eye_data[voltage_idx, time_idx]
                
                # Check if measurement violates mask
                if self._point_violates_mask(point, measured_value):
                    violation_margin = abs(measured_value - point.voltage_mv)
                    severity = self._determine_violation_severity(violation_margin)
                    
                    violation = MaskViolation(
                        point_index=i,
                        measured_voltage=measured_value,
                        mask_voltage=point.voltage_mv,
                        violation_margin=violation_margin,
                        time_ui=point.time_ui,
                        severity=severity
                    )
                    violations.append(violation)
        
        return violations
    
    def _create_mask_polygon(self) -> List[Tuple[float, float]]:
        """Create polygon representation of mask"""
        return [(point.time_ui, point.voltage_mv) for point in self.mask.points]
    
    def _find_nearest_index(self, array: np.ndarray, value: float) -> int:
        """Find nearest index in array"""
        return int(np.argmin(np.abs(array - value)))
    
    def _point_violates_mask(self, mask_point: EyeMaskPoint, measured_value: float) -> bool:
        """Check if measured point violates mask"""
        # For positive voltages, measured should be above mask
        if mask_point.voltage_mv > 0:
            return measured_value < mask_point.voltage_mv
        # For negative voltages, measured should be below mask
        else:
            return measured_value > mask_point.voltage_mv
    
    def _determine_violation_severity(self, margin: float) -> str:
        """Determine violation severity"""
        if margin < 10:  # mV
            return "minor"
        elif margin < 50:
            return "major"
        else:
            return "critical"
    
    def _calculate_eye_opening(self, 
                              eye_data: np.ndarray,
                              time_axis: np.ndarray,
                              voltage_axis: np.ndarray) -> float:
        """Calculate eye opening percentage"""
        try:
            # Find eye center
            center_time_idx = len(time_axis) // 2
            center_voltage_idx = len(voltage_axis) // 2
            
            # Calculate eye height and width
            eye_height = np.max(eye_data) - np.min(eye_data)
            eye_width = time_axis[-1] - time_axis[0]
            
            # Calculate opening as percentage of total area
            total_area = eye_height * eye_width
            mask_area = self._calculate_mask_area()
            
            if total_area > 0:
                opening_percentage = max(0, (total_area - mask_area) / total_area * 100)
            else:
                opening_percentage = 0
            
            return opening_percentage
            
        except Exception as e:
            logger.warning(f"Eye opening calculation failed: {e}")
            return 0.0
    
    def _calculate_mask_area(self) -> float:
        """Calculate mask area using shoelace formula"""
        try:
            points = [(p.time_ui, p.voltage_mv) for p in self.mask.points]
            n = len(points)
            area = 0.0
            
            for i in range(n):
                j = (i + 1) % n
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            
            return abs(area) / 2.0
            
        except Exception:
            return 0.0
    
    def _determine_compliance_level(self, violations: List[MaskViolation], eye_opening: float) -> str:
        """Determine overall compliance level"""
        if not violations and eye_opening > 70:
            return "pass"
        elif len(violations) <= 2 and eye_opening > 50:
            return "marginal"
        else:
            return "fail"
    
    def _calculate_margin_percentage(self, violations: List[MaskViolation], eye_opening: float) -> float:
        """Calculate overall margin percentage"""
        if not violations:
            return min(100.0, eye_opening)
        
        # Calculate margin based on worst violation
        worst_margin = max(v.violation_margin for v in violations)
        margin_percentage = max(0, 100 - (worst_margin / self.mask.voltage_swing_mv * 100))
        
        return margin_percentage


def create_eye_mask_analyzer(protocol: str, data_rate_gbps: float = None) -> EyeMaskAnalyzer:
    """
    Factory function to create eye mask analyzer
    
    Args:
        protocol: Protocol name (USB4, PCIe, Ethernet)
        data_rate_gbps: Data rate in Gbps (uses default if None)
        
    Returns:
        EyeMaskAnalyzer instance
    """
    # Set default data rates
    default_rates = {
        "USB4": 20.0,
        "PCIE": 32.0,
        "ETHERNET": 112.0
    }
    
    if data_rate_gbps is None:
        data_rate_gbps = default_rates.get(protocol.upper(), 20.0)
    
    return EyeMaskAnalyzer(protocol, data_rate_gbps)


# Export main classes and functions
__all__ = [
    'EyeMask',
    'EyeMaskPoint',
    'EyeMaskResult',
    'MaskViolation',
    'EyeMaskAnalyzer',
    'EyeMaskGenerator',
    'MaskType',
    'create_eye_mask_analyzer'
]
