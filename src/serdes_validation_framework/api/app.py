"""
FastAPI Application

Main FastAPI application for the SerDes Validation Framework REST API.
"""

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .core import SerDesAPI
from .models import ErrorResponse
from .routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting SerDes Validation Framework API")
    yield
    # Shutdown
    logger.info("Shutting down SerDes Validation Framework API")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="SerDes Validation Framework API",
        description="""
        REST API for the SerDes Validation Framework
        
        This API provides comprehensive SerDes validation capabilities including:
        - Eye diagram analysis
        - Waveform signal quality analysis  
        - Stress testing with progressive degradation
        - Test fixture control and environmental monitoring
        - Eye mask compliance checking
        - BERT script hooks integration
        
        ## Supported Protocols
        - USB4 2.0 (20 Gbps)
        - PCIe 6.0 (32 Gbps) 
        - 224G Ethernet (112 Gbps per lane)
        
        ## Features
        - Asynchronous test execution
        - Real-time test status monitoring
        - Comprehensive error handling
        - Protocol-specific analysis
        - Professional-grade visualizations
        """,
        version="1.4.1",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        return response
    
    # Include routes
    app.include_router(router, prefix="/api/v1")
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An internal server error occurred",
                timestamp=time.time()
            ).dict()
        )
    
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server"""
    app = create_app()
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


# For direct execution
if __name__ == "__main__":
    run_server(reload=True)
