#!/usr/bin/env python3
"""
SerDes Validation Framework API Server

Development server script for the SerDes Validation Framework REST API.
This script starts the FastAPI server with development settings.

Usage:
    python run_api_server.py [--host HOST] [--port PORT] [--no-reload]

For production deployment, use:
    uvicorn serdes_validation_framework.api.app:create_app --host 0.0.0.0 --port 8000
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set environment variables for development
os.environ.setdefault('SVF_MOCK_MODE', '1')  # Enable mock mode for development
os.environ.setdefault('MPLBACKEND', 'Agg')   # Use non-GUI matplotlib backend

def main():
    """Main function to start the API server"""
    parser = argparse.ArgumentParser(description='SerDes Validation Framework API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--no-reload', action='store_true', help='Disable auto-reload')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'],
                       help='Log level (default: info)')
    
    args = parser.parse_args()
    
    try:
        from serdes_validation_framework.api.app import run_server
        
        print("🚀 Starting SerDes Validation Framework API Server")
        print(f"📍 Server: http://{args.host}:{args.port}")
        print(f"📖 API Documentation: http://{args.host}:{args.port}/docs")
        print(f"🔄 Interactive API: http://{args.host}:{args.port}/redoc")
        print(f"⚡ Health Check: http://{args.host}:{args.port}/api/v1/health")
        print(f"📊 System Status: http://{args.host}:{args.port}/api/v1/status")
        print()
        print("💡 Press Ctrl+C to stop the server")
        print()
        
        # Start the server
        run_server(
            host=args.host,
            port=args.port,
            reload=not args.no_reload
        )
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you have installed the framework:")
        print("   pip install -e .")
        print("   or")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
