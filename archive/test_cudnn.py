#!/usr/bin/env python
import ctypes
import os
import jaxlib
import sys
import platform

print("Python version:", platform.python_version())
print("jaxlib path:", getattr(jaxlib, "__file__", "unknown"))
print("jaxlib version:", getattr(jaxlib, "__version__", "unknown"))
print("CUDA_PATH:", os.environ.get("CUDA_PATH", "unset"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "unset"))

try:
    cdll = ctypes.CDLL("libcudnn.so.9")
    print("✅ cuDNN loaded successfully")
except Exception as e:
    print(f"❌ cuDNN load error: {e}")

# Also test JAX itself
try:
    import jax
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
except Exception as e:
    print(f"❌ JAX import error: {e}") 