FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set up environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install JAX with CUDA support
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir "jax[cuda12]>=0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Test script
COPY <<EOF /test_gpu.py
import os
import time
import jax
import jax.numpy as jnp
from jax import random, jit

# Print JAX info
print("JAX version:", jax.__version__)
print("Available devices:", jax.devices())
print("Available GPU count:", jax.device_count("gpu"))
print("Default backend:", jax.default_backend())

# Test if GPU is working
if jax.device_count("gpu") > 0:
    print("GPU detected! Running matrix multiplication benchmark...")
    
    # Create random matrices
    key = random.key(0)
    x = random.normal(key, (5000, 5000), dtype=jnp.float32)
    
    # JIT-compile the matrix multiplication
    @jit
    def matmul(x):
        return jnp.dot(x, x.T)
    
    # Warmup run (to compile)
    result = matmul(x)
    result.block_until_ready()
    
    # Timed run
    start_time = time.time()
    result = matmul(x)
    result.block_until_ready()  # Wait for the computation to complete
    elapsed = time.time() - start_time
    
    print(f"Matrix multiplication time: {elapsed:.4f} seconds")
    print(f"Matrix shape: {result.shape}")
else:
    print("No GPU detected by JAX.")
EOF

# Run the test script
CMD ["python3", "/test_gpu.py"] 