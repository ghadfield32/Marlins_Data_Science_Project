#!/bin/bash
# Script to fix cuDNN version compatibility with JAX

set -e  # Exit on any error

echo "==== JAX cuDNN Compatibility Fix ===="
echo "This script will fix the cuDNN version mismatch with JAX"

# 1. Check current versions
echo -e "\n==== Current Versions ===="
echo "JAX version:"
python -c "import jax; print(jax.__version__)"
echo "JAXlib version:"
python -c "import jaxlib; print(getattr(jaxlib, '__version__', 'unknown'))"
echo "cuDNN version (from library):"
ldconfig -v 2>/dev/null | grep -E "libcudnn.so.[0-9]+ ->" | head -1

# 2. Install appropriate cuDNN version for JAX 0.5.x
echo -e "\n==== Installing Compatible cuDNN 9.1.x ===="
# The specific commands depend on your environment - we're using apt since it's Ubuntu
apt-get update
apt-get install -y --allow-downgrades --allow-change-held-packages libcudnn9=9.1.0* libcudnn9-dev=9.1.0*

# 3. Clean up CUDA cache
echo -e "\n==== Clearing CUDA cache ===="
XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "import jax; jax.clear_caches(); jax.clear_backends()"

# 4. Verify fix
echo -e "\n==== Verifying Fix ===="
echo "Updated cuDNN version:"
ldconfig -v 2>/dev/null | grep -E "libcudnn.so.[0-9]+ ->" | head -1

# 5. Test JAX with new cuDNN
echo -e "\n==== Testing JAX with new cuDNN ===="
python - <<EOF
import jax
import jax.numpy as jnp
import time

print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Small test
x = jnp.ones((10, 10))
y = jnp.dot(x, x)
y.block_until_ready()
print("✓ Small matrix multiplication successful")

# Larger test
start = time.time()
large_x = jnp.ones((1000, 1000))
large_y = jnp.dot(large_x, large_x)
large_y.block_until_ready()
duration = time.time() - start
print(f"✓ Large matrix multiplication completed in {duration:.4f} seconds")

print("All JAX tests passed successfully!")
EOF

echo -e "\n==== Fix Complete ===="
echo "If all tests passed, your JAX environment should now be working correctly."
echo "You can now restart your Jupyter kernel to use the fixed environment." 