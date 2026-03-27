#!/bin/bash
# Build the C++/OpenMP asset scorer
# Usage: ./build.sh [standalone|python|both]

MODE=${1:-both}
echo "=== Building Portfolio Asset Scorer ==="
echo "Mode: $MODE"
echo ""

# Detect platform and set compiler flags
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    # macOS: use clang++ with Homebrew libomp
    CXX="clang++"
    if brew --prefix libomp &>/dev/null; then
        OMP_PREFIX="$(brew --prefix libomp)"
        OMP_FLAGS="-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include -L${OMP_PREFIX}/lib -lomp"
    else
        echo "ERROR: libomp not found. Install with: brew install libomp"
        exit 1
    fi
else
    # Linux: use g++ with -fopenmp
    CXX="g++"
    OMP_FLAGS="-fopenmp"
    if ! $CXX $OMP_FLAGS -x c++ -E - < /dev/null &>/dev/null; then
        echo "ERROR: g++ with OpenMP support not found."
        echo "Install with: sudo apt install g++ libomp-dev"
        exit 1
    fi
fi

if [ "$MODE" = "standalone" ] || [ "$MODE" = "both" ]; then
    echo "Building standalone executable..."
    $CXX -O3 $OMP_FLAGS -std=c++17 -o scorer_standalone scorer.cpp
    echo "  Built: ./scorer_standalone"
fi

if [ "$MODE" = "python" ] || [ "$MODE" = "both" ]; then
    if python3 -c "import pybind11" 2>/dev/null; then
        EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
        echo "Building Python module..."
        if [ "$OS" = "Darwin" ]; then
            $CXX -O3 $OMP_FLAGS -std=c++17 -fPIC -dynamiclib -undefined dynamic_lookup \
                -DUSE_PYBIND11 \
                $(python3 -m pybind11 --includes) \
                -o scorer${EXT_SUFFIX} scorer.cpp
        else
            $CXX -O3 $OMP_FLAGS -shared -std=c++17 -fPIC \
                -DUSE_PYBIND11 \
                $(python3 -m pybind11 --includes) \
                -o scorer${EXT_SUFFIX} scorer.cpp
        fi
        echo "  Built: scorer${EXT_SUFFIX}"
    else
        echo "SKIP: pybind11 not installed (pip install pybind11)"
    fi
fi

echo ""
echo "Done! Run ./scorer_standalone to test."
