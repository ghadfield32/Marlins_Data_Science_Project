"""
Simple test script to verify JAX GPU functionality.
Run this to confirm GPU access is properly set up.
"""
import os
import sys
import time

# Configure JAX environment variables
os.environ["JAX_PLATFORMS"] = "cpu,cuda"

try:
    import jax
    import jax.numpy as jnp
    from jax import random, jit
except ImportError:
    print("JAX not installed. Please install with: pip install jax[cuda]")
    sys.exit(1)

def print_device_info():
    """Print detailed JAX device and platform information."""
    print("\n=== JAX Environment ===")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"GPU Count: {jax.device_count('gpu')}")
    
    # Check for GPU using device_count instead of device platform
    has_gpu = jax.device_count('gpu') > 0
    if has_gpu:
        print("✅ GPU detected by JAX")
        # Try to get GPU devices even if they don't show up in jax.devices()
        try:
            gpu_devices = jax.devices('gpu')
            for i, device in enumerate(gpu_devices):
                print(f"  GPU {i}: {device}")
        except:
            print("  GPU devices not directly accessible via jax.devices('gpu')")
    else:
        print("❌ No GPU detected by JAX")
        print("   If you expect a GPU to be available, check:")
        print("   - Docker is running with GPU access (--gpus all flag)")
        print("   - NVIDIA drivers are installed and working (nvidia-smi)")
        print("   - JAX CUDA version matches your NVIDIA driver version")

def benchmark_matrix_multiply(size=4000):
    """Run a simple matrix multiplication benchmark to test GPU performance."""
    print(f"\n=== Running JAX Matrix Multiplication Benchmark (size={size}) ===")
    
    # Create random matrices
    key = random.key(0)
    x = random.normal(key, (size, size), dtype=jnp.float32)
    
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
    
    # On a GPU, this should be much faster than on CPU
    if elapsed < 1.0:
        print("✅ Computation speed suggests GPU acceleration is working")
    elif elapsed < 5.0:
        print("⚠️ Computation speed is moderate - GPU may not be optimal")
    else:
        print("❌ Computation was slow - GPU acceleration may not be working")
        
    return elapsed

if __name__ == "__main__":
    print_device_info()
    
    # Only run benchmark if GPU is detected via device_count
    if jax.device_count('gpu') > 0:
        try:
            benchmark_matrix_multiply()
        except Exception as e:
            print(f"Error during benchmark: {e}")
    else:
        print("\nSkipping GPU benchmark as no GPU was detected.")
        if os.environ.get("REQUIRE_GPU", "").lower() in ("1", "true", "yes"):
            print("ERROR: GPU access required but not available. Exiting.")
            sys.exit(1) 