#!/bin/bash
# Quick start script for the Portfolio Optimization Engine

echo "============================================"
echo "  GPU-Accelerated Portfolio Optimizer"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+"
    exit 1
fi

# Install dependencies
echo "[1/3] Installing Python dependencies..."
pip install -r "$(dirname "$0")/requirements.txt" 2>/dev/null || \
pip install -r "$(dirname "$0")/requirements.txt" --break-system-packages 2>/dev/null

# Optional: PyTorch (for GPU acceleration)
echo ""
echo "[2/3] Checking PyTorch..."
if python3 -c "import torch" 2>/dev/null; then
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "  ✓ PyTorch with CUDA detected!"
    else
        echo "  ✓ PyTorch (CPU mode) detected"
    fi
else
    echo "  ⚠ PyTorch not installed. Running in NumPy mode."
    echo "  For GPU acceleration, install: pip install torch"
fi

echo ""
echo "[3/3] Starting server..."
echo ""
echo "  → Dashboard: http://localhost:5001"
echo "  → API:       http://localhost:5001/api/optimize"
echo ""
echo "  Press Ctrl+C to stop."
echo "============================================"
echo ""

cd "$(dirname "$0")"
python3 backend/server.py
