"""
SerDes Validation Framework REST API

This module provides a comprehensive REST API wrapper around the
SerDes Validation Framework for web-based and remote access.
"""

from .app import SerDesAPI, create_app
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
    TestType,
    WaveformAnalysisRequest,
    WaveformAnalysisResponse,
)
from .routes import router

__all__ = [
    'create_app',
    'SerDesAPI'
]
