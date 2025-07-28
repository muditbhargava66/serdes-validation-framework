"""
API Routes

FastAPI route definitions for the SerDes Validation Framework REST API.
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse

from .core import SerDesAPI
from .models import (
    ErrorResponse,
    EyeDiagramRequest,
    EyeDiagramResponse,
    FixtureControlRequest,
    FixtureStatusResponse,
    ProtocolType,
    StatusType,
    StressTestRequest,
    StressTestResponse,
    SystemStatusResponse,
    TestStatusResponse,
    WaveformAnalysisRequest,
    WaveformAnalysisResponse,
)

logger = logging.getLogger(__name__)

# Create API instance
api = SerDesAPI()

# Create router
router = APIRouter()


def get_api() -> SerDesAPI:
    """Dependency to get API instance"""
    return api


# Create the dependency once to avoid B008 warning
api_dependency = Depends(get_api)


@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "name": "SerDes Validation Framework API",
        "version": "1.4.1",
        "description": "REST API for SerDes validation and testing",
        "documentation": "/docs"
    }


@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": str(time.time())
    }


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(api: SerDesAPI = api_dependency):
    """Get system status"""
    try:
        return api.get_system_status()
    except Exception as e:
        logger.error(f"System status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Eye Diagram Analysis
@router.post("/analyze/eye-diagram", response_model=EyeDiagramResponse)
async def analyze_eye_diagram(
    request: EyeDiagramRequest,
    api: SerDesAPI = api_dependency
):
    """Analyze eye diagram from signal data"""
    try:
        return api.analyze_eye_diagram(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Eye diagram analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Waveform Analysis
@router.post("/analyze/waveform", response_model=WaveformAnalysisResponse)
async def analyze_waveform(
    request: WaveformAnalysisRequest,
    api: SerDesAPI = api_dependency
):
    """Analyze waveform signal quality"""
    try:
        return api.analyze_waveform(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Waveform analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mask Compliance
@router.post("/analyze/mask-compliance", response_model=Dict[str, Any])
async def check_mask_compliance(
    request: EyeDiagramRequest,
    api: SerDesAPI = api_dependency
):
    """Check eye mask compliance"""
    try:
        return api.check_mask_compliance(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Mask compliance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Stress Testing
@router.post("/test/stress", response_model=Dict[str, str])
async def start_stress_test(
    request: StressTestRequest,
    api: SerDesAPI = api_dependency
):
    """Start a stress test (async)"""
    try:
        test_id = api.start_stress_test(request)
        return {
            "test_id": test_id,
            "status": "started",
            "message": "Stress test started successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Stress test start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/stress/{test_id}", response_model=StressTestResponse)
async def get_stress_test_result(
    test_id: str,
    api: SerDesAPI = api_dependency
):
    """Get stress test result"""
    result = api.get_stress_test_result(test_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Test not found")
    return result


# Test Status Management
@router.get("/test/{test_id}/status", response_model=TestStatusResponse)
async def get_test_status(
    test_id: str,
    api: SerDesAPI = api_dependency
):
    """Get test status"""
    status = api.get_test_status(test_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Test not found")
    return status


@router.post("/test/{test_id}/cancel", response_model=Dict[str, str])
async def cancel_test(
    test_id: str,
    api: SerDesAPI = api_dependency
):
    """Cancel a running test"""
    success = api.cancel_test(test_id)
    if not success:
        raise HTTPException(status_code=404, detail="Test not found or cannot be cancelled")
    
    return {
        "test_id": test_id,
        "status": "cancelled",
        "message": "Test cancelled successfully"
    }


# Fixture Control
@router.post("/fixture/control", response_model=FixtureStatusResponse)
async def control_fixture(
    request: FixtureControlRequest,
    api: SerDesAPI = api_dependency
):
    """Control test fixture"""
    try:
        return api.control_fixture(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Fixture control error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fixture/{fixture_name}/status", response_model=FixtureStatusResponse)
async def get_fixture_status(
    fixture_name: str,
    api: SerDesAPI = api_dependency
):
    """Get fixture status"""
    try:
        request = FixtureControlRequest(
            fixture_name=fixture_name,
            action="get_status"
        )
        return api.control_fixture(request)
    except Exception as e:
        logger.error(f"Get fixture status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Protocol Information
@router.get("/protocols", response_model=Dict[str, Any])
async def get_protocols():
    """Get available protocols and their specifications"""
    return {
        "protocols": {
            "USB4": {
                "data_rate_gbps": 20.0,
                "voltage_swing_mv": 800,
                "eye_height_threshold_mv": 100,
                "snr_threshold_db": 15.0,
                "description": "USB4 2.0 specification"
            },
            "PCIe": {
                "data_rate_gbps": 32.0,
                "voltage_swing_mv": 1200,
                "eye_height_threshold_mv": 150,
                "snr_threshold_db": 12.0,
                "description": "PCIe 6.0 specification"
            },
            "Ethernet": {
                "data_rate_gbps": 112.0,
                "voltage_swing_mv": 1600,
                "eye_height_threshold_mv": 200,
                "snr_threshold_db": 10.0,
                "description": "224G Ethernet specification"
            }
        },
        "timestamp": time.time()
    }


# Note: Exception handlers are defined in app.py for the main FastAPI app
