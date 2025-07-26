#!/bin/bash

# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the docs directory
cd "$SCRIPT_DIR"

# Set mock mode for documentation builds
export SVF_MOCK_MODE=1

# Install documentation dependencies
echo "Installing documentation dependencies..."
pip install -r requirements.txt

# Clean previous build
echo "Cleaning previous build..."
rm -rf _build

# Build documentation
echo "Building documentation with Sphinx..."
sphinx-build -b html . _build/html

# Show build result
echo ""
echo "✅ Documentation built successfully!"
echo "📁 Output directory: $SCRIPT_DIR/_build/html"
echo "🌐 To view locally, run:"
echo "   cd $SCRIPT_DIR/_build/html && python -m http.server 8000"
echo "   Then open: http://localhost:8000"