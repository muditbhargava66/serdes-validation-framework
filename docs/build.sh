#!/bin/bash

# Exit on error
set -e

# Install dependencies
pip install -r requirements.txt

# Clean previous build
rm -rf _build

# Build documentation
sphinx-build -b html . _build/html

# Show build result
echo "Documentation built in _build/html"
echo "To view, run: python -m http.server --directory _build/html"