"""
API Data Models

Pydantic models for request/response validation in the REST API.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, validator


class ProtocolType(str, Enum):
    """Supported protocols"""
    USB4 = "USB4"
    PCIE = "PCIe"
    ETHERNET = "Ethernet"


class TestType(str, Enum):
    """Types of tests"""
    EYE_DIAGRAM = "eye_diagram"
    STRESS_TEST = "stress_test"
    WAVEFORM_ANALYSIS = "waveform_analysis"
    MASK_COMPLIANCE = "mask_compliance"
    FIXTURE_CONTROL = "fixture_control"


class StatusType(str, Enum):
    """Test status types"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Request Models
class SignalDataRequest(BaseModel):
    """Request model for signal data"""
    signal_data: Union[List[float], Dict[str, List[float]]] = Field(
        ..., description="Signal data as array or multi-lane dictionary"
    )
    sample_rate: float = Field(40e9, description="Sample rate in Hz")
    protocol: ProtocolType = Field(ProtocolType.USB4, description="Protocol type")
    time_data: Optional[List[float]] = Field(None, description="Optional time axis data")
    
    @validator('sample_rate')
    def validate_sample_rate(cls, v):
        if v <= 0:
            raise ValueError('Sample rate must be positive')
        return v


class EyeDiagramRequest(SignalDataRequest):
    """Request model for eye diagram analysis"""
    lane: Union[int, str] = Field(0, description="Lane to analyze")
    show_mask: bool = Field(True, description="Include mask analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "signal_data": [0.1, -0.1, 0.2, -0.2] * 250,
                "sample_rate": 40e9,
                "protocol": "USB4",
                "lane": 0,
                "show_mask": True
            }
        }


class StressTestRequest(BaseModel):
    """Request model for stress testing"""
    protocol: ProtocolType = Field(ProtocolType.USB4, description="Protocol to test")
    num_cycles: int = Field(100, description="Number of test cycles")
    cycle_duration: float = Field(1.0, description="Duration per cycle in seconds")
    data_rate: Optional[float] = Field(None, description="Data rate in bps")
    voltage_swing: Optional[float] = Field(None, description="Voltage swing in V")
    enable_bert_hooks: bool = Field(False, description="Enable BERT hooks")
    
    @validator('num_cycles')
    def validate_cycles(cls, v):
        if v <= 0 or v > 10000:
            raise ValueError('Number of cycles must be between 1 and 10000')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "protocol": "USB4",
                "num_cycles": 100,
                "cycle_duration": 1.0,
                "enable_bert_hooks": False
            }
        }


class WaveformAnalysisRequest(SignalDataRequest):
    """Request model for waveform analysis"""
    lane: str = Field("lane_0", description="Lane identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "signal_data": {"lane_0": [0.1, -0.1, 0.2, -0.2] * 250},
                "sample_rate": 40e9,
                "protocol": "USB4",
                "lane": "lane_0"
            }
        }


class FixtureControlRequest(BaseModel):
    """Request model for fixture control"""
    fixture_name: str = Field(..., description="Fixture identifier")
    action: str = Field(..., description="Action to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    
    @validator('action')
    def validate_action(cls, v):
        allowed_actions = ['connect', 'disconnect', 'set_voltage', 'set_frequency', 'calibrate', 'get_status']
        if v not in allowed_actions:
            raise ValueError(f'Action must be one of: {allowed_actions}')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "fixture_name": "probe_station_1",
                "action": "set_voltage",
                "parameters": {"voltage": 3.3}
            }
        }


# Response Models
class EyeDiagramResponse(BaseModel):
    """Response model for eye diagram analysis"""
    eye_height: float = Field(..., description="Eye height in V")
    eye_width: float = Field(..., description="Eye width in UI")
    eye_area: float = Field(..., description="Eye area")
    q_factor: float = Field(..., description="Q-factor")
    snr: float = Field(..., description="SNR in dB")
    passed: bool = Field(..., description="Pass/fail status")
    lane: str = Field(..., description="Analyzed lane")
    protocol: str = Field(..., description="Protocol used")
    mask_analysis: Optional[Dict[str, Any]] = Field(None, description="Mask compliance analysis")
    timestamp: float = Field(..., description="Analysis timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "eye_height": 0.8,
                "eye_width": 0.7,
                "eye_area": 0.56,
                "q_factor": 7.2,
                "snr": 17.1,
                "passed": True,
                "lane": "lane_0",
                "protocol": "USB4",
                "timestamp": 1640995200.0
            }
        }


class StressTestResponse(BaseModel):
    """Response model for stress test results"""
    test_id: str = Field(..., description="Unique test identifier")
    status: StatusType = Field(..., description="Test status")
    protocol: str = Field(..., description="Protocol tested")
    total_cycles: int = Field(..., description="Total test cycles")
    passed_cycles: int = Field(..., description="Passed cycles")
    failed_cycles: int = Field(..., description="Failed cycles")
    success_rate: float = Field(..., description="Success rate (0-1)")
    duration: float = Field(..., description="Test duration in seconds")
    max_degradation: float = Field(..., description="Maximum degradation percentage")
    initial_eye_height: Optional[float] = Field(None, description="Initial eye height")
    final_eye_height: Optional[float] = Field(None, description="Final eye height")
    timestamp: float = Field(..., description="Test timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "test_id": "stress_test_123",
                "status": "completed",
                "protocol": "USB4",
                "total_cycles": 100,
                "passed_cycles": 95,
                "failed_cycles": 5,
                "success_rate": 0.95,
                "duration": 100.0,
                "max_degradation": 5.2,
                "timestamp": 1640995200.0
            }
        }


class WaveformAnalysisResponse(BaseModel):
    """Response model for waveform analysis"""
    lane: str = Field(..., description="Analyzed lane")
    protocol: str = Field(..., description="Protocol used")
    mean_voltage: float = Field(..., description="Mean voltage in V")
    rms_voltage: float = Field(..., description="RMS voltage in V")
    peak_to_peak: float = Field(..., description="Peak-to-peak voltage in V")
    snr_db: float = Field(..., description="SNR in dB")
    thd_percent: float = Field(..., description="THD percentage")
    dynamic_range: float = Field(..., description="Dynamic range in dB")
    passed: bool = Field(..., description="Pass/fail status")
    failure_reasons: List[str] = Field(default_factory=list, description="Failure reasons")
    timestamp: float = Field(..., description="Analysis timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "lane": "lane_0",
                "protocol": "USB4",
                "mean_voltage": 0.0,
                "rms_voltage": 0.4,
                "peak_to_peak": 1.6,
                "snr_db": 20.5,
                "thd_percent": 2.1,
                "dynamic_range": 45.0,
                "passed": True,
                "failure_reasons": [],
                "timestamp": 1640995200.0
            }
        }


class FixtureStatusResponse(BaseModel):
    """Response model for fixture status"""
    fixture_name: str = Field(..., description="Fixture identifier")
    status: str = Field(..., description="Current status")
    temperature: float = Field(..., description="Temperature in Celsius")
    voltage: float = Field(..., description="Current voltage in V")
    current: float = Field(..., description="Current in A")
    connected: bool = Field(..., description="Connection status")
    last_calibration: Optional[float] = Field(None, description="Last calibration timestamp")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    error_message: str = Field("", description="Error message if any")
    timestamp: float = Field(..., description="Status timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "fixture_name": "probe_station_1",
                "status": "READY",
                "temperature": 25.0,
                "voltage": 3.3,
                "current": 0.1,
                "connected": True,
                "uptime_seconds": 3600.0,
                "error_message": "",
                "timestamp": 1640995200.0
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: float = Field(..., description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid signal data format",
                "details": {"field": "signal_data", "issue": "must be array or dict"},
                "timestamp": 1640995200.0
            }
        }


class TestStatusResponse(BaseModel):
    """Test status response model"""
    test_id: str = Field(..., description="Test identifier")
    status: StatusType = Field(..., description="Current status")
    progress: float = Field(..., description="Progress percentage (0-100)")
    message: str = Field("", description="Status message")
    started_at: float = Field(..., description="Start timestamp")
    estimated_completion: Optional[float] = Field(None, description="Estimated completion timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "test_id": "stress_test_123",
                "status": "running",
                "progress": 45.0,
                "message": "Running cycle 45/100",
                "started_at": 1640995200.0,
                "estimated_completion": 1640995260.0
            }
        }


class SystemStatusResponse(BaseModel):
    """System status response model"""
    version: str = Field(..., description="Framework version")
    status: str = Field(..., description="System status")
    active_tests: int = Field(..., description="Number of active tests")
    total_tests_run: int = Field(..., description="Total tests run")
    uptime_seconds: float = Field(..., description="System uptime")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    available_protocols: List[str] = Field(..., description="Available protocols")
    features: Dict[str, bool] = Field(..., description="Available features")
    timestamp: float = Field(..., description="Status timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "version": "1.4.1",
                "status": "healthy",
                "active_tests": 2,
                "total_tests_run": 150,
                "uptime_seconds": 86400.0,
                "memory_usage_mb": 256.0,
                "cpu_usage_percent": 15.5,
                "available_protocols": ["USB4", "PCIe", "Ethernet"],
                "features": {
                    "eye_analysis": True,
                    "stress_testing": True,
                    "bert_hooks": True,
                    "fixture_control": True
                },
                "timestamp": 1640995200.0
            }
        }
