#!/bin/bash
# Script to upgrade dependencies for fixing kernel crashes

echo "Upgrading PyZMQ in the marlins-ds-gpu environment..."
conda run -n marlins-ds-gpu conda install -c conda-forge 'pyzmq>=26.0.3' -y
conda run -n marlins-ds-gpu conda clean -afy

echo "Checking for duplicate CUDA libraries in LD_LIBRARY_PATH..."
conda run -n marlins-ds-gpu bash -c 'ldconfig -v 2>/dev/null | grep -E "cuda|cudnn"'

echo "Verifying ZMQ library configuration..."
conda run -n marlins-ds-gpu python - <<'PY'
import ctypes, pkg_resources, sys
print("Using:", ctypes.util.find_library('zmq'), pkg_resources.get_distribution('pyzmq'))
PY

echo "Done! Now restart your container with: docker compose up datascience" 