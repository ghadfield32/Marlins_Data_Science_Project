#!/usr/bin/env python
"""
Comprehensive diagnostic tool for JAX/cuDNN compatibility issues
"""
import os
import sys
import platform
import subprocess
import json
import ctypes
import logging
from pathlib import Path
import faulthandler
import datetime

# Enable faulthandler to debug segfaults
faulthandler.enable()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('jax-diag')

# Create diagnostic output directory
DIAG_DIR = Path('/tmp/jax_diagnostics')
DIAG_DIR.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_FILE = DIAG_DIR / f"jax_diag_report_{timestamp}.txt"

def log_to_file(message):
    """Log to both console and diagnostic file"""
    logger.info(message)
    with open(REPORT_FILE, 'a') as f:
        f.write(f"{message}\n")

def run_cmd(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            shell=isinstance(cmd, str)
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error running {cmd}: {e}"

def check_system_info():
    """Collect system information"""
    log_to_file("\n=== SYSTEM INFORMATION ===")
    log_to_file(f"Hostname: {platform.node()}")
    log_to_file(f"Platform: {platform.platform()}")
    log_to_file(f"Python: {platform.python_version()} ({platform.python_implementation()})")
    log_to_file(f"Processor: {platform.processor()}")
    
    # Check if running in a container
    cgroup_content = run_cmd("cat /proc/1/cgroup 2>/dev/null")
    is_container = "docker" in cgroup_content or "kubepods" in cgroup_content
    log_to_file(f"Running in container: {is_container}")

def check_cuda_env():
    """Check CUDA environment variables and libraries"""
    log_to_file("\n=== CUDA ENVIRONMENT ===")
    for var in [
        'CUDA_VISIBLE_DEVICES', 'CUDA_PATH', 'LD_LIBRARY_PATH',
        'JAX_PLATFORM_NAME', 'XLA_PYTHON_CLIENT_PREALLOCATE',
        'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_ALLOCATOR'
    ]:
        log_to_file(f"{var}={os.environ.get(var, 'not set')}")
    
    # Check nvidia-smi
    nvidia_smi = run_cmd("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_mode --format=csv,noheader")
    log_to_file(f"\nnvidia-smi GPU info:\n{nvidia_smi}")
    
    # Check CUDA version
    nvcc_version = run_cmd("nvcc --version 2>/dev/null")
    log_to_file(f"\nnvcc version:\n{nvcc_version}")
    
    # Find libcudnn.so
    cudnn_paths = run_cmd("find /usr -name 'libcudnn.so*' 2>/dev/null")
    log_to_file(f"\nFound cuDNN libraries:\n{cudnn_paths}")

def check_library_paths():
    """Check library search paths and duplicate libraries"""
    log_to_file("\n=== LIBRARY PATHS ===")
    
    # Check for duplicate cuDNN/CUDA in LD_LIBRARY_PATH
    ld_output = run_cmd("ldconfig -v 2>/dev/null | grep -E 'cuda|cudnn'")
    log_to_file(f"CUDA/cuDNN libraries in ldconfig cache:\n{ld_output}")
    
    # Check where libcudnn.so points to
    libcudnn_path = run_cmd("ls -la /usr/lib/x86_64-linux-gnu/libcudnn.so 2>/dev/null")
    log_to_file(f"\nlibcudnn.so path: {libcudnn_path}")

def try_load_cudnn():
    """Attempt to load cuDNN library via ctypes"""
    log_to_file("\n=== CUDNN LIBRARY TEST ===")
    try:
        cudnn = ctypes.CDLL("libcudnn.so")
        log_to_file("✅ Successfully loaded libcudnn.so")
        return True
    except Exception as e:
        log_to_file(f"❌ Failed to load libcudnn.so: {e}")
        return False

def check_jax_installation():
    """Check JAX installation and version"""
    log_to_file("\n=== JAX INSTALLATION ===")
    try:
        import jaxlib
        log_to_file(f"jaxlib path: {getattr(jaxlib, '__file__', 'unknown')}")
        log_to_file(f"jaxlib version: {getattr(jaxlib, '__version__', 'unknown')}")
        
        try:
            import jax
            log_to_file(f"jax path: {getattr(jax, '__file__', 'unknown')}")
            log_to_file(f"jax version: {jax.__version__}")
            
            # Check JAX configuration
            log_to_file(f"JAX default backend: {jax.default_backend()}")
            log_to_file(f"JAX devices: {jax.devices()}")
            log_to_file(f"Available memory: {jax.devices()[0].memory_stats() if jax.devices() else 'N/A'}")
            
            # Look for compilation flags related to cuDNN
            if hasattr(jaxlib, 'xla_extension'):
                version_info = getattr(jaxlib.xla_extension, 'get_build_version_info', lambda: {})()
                log_to_file(f"Build version info: {version_info}")
            
            return True
        except Exception as e:
            log_to_file(f"❌ Failed to import jax: {e}")
            return False
    except Exception as e:
        log_to_file(f"❌ Failed to import jaxlib: {e}")
        return False

def test_basic_operation():
    """Test basic JAX operations"""
    log_to_file("\n=== JAX OPERATIONS TEST ===")
    try:
        import jax
        import jax.numpy as jnp
        import time
        
        # Simple operation
        log_to_file("Testing simple array creation...")
        x = jnp.ones((10, 10))
        log_to_file(f"Created array shape: {x.shape}")
        
        # Matrix multiplication
        log_to_file("Testing matrix multiplication...")
        start = time.time()
        y = jnp.dot(x, x)
        y.block_until_ready()  # Force computation
        duration = time.time() - start
        log_to_file(f"Matrix multiplication completed in {duration:.4f} seconds")
        
        # Try a larger operation
        log_to_file("Testing larger matrix operation...")
        try:
            large_x = jnp.ones((1000, 1000))
            start = time.time()
            large_y = jnp.dot(large_x, large_x)
            large_y.block_until_ready()
            duration = time.time() - start
            log_to_file(f"Large matrix multiplication completed in {duration:.4f} seconds")
            log_to_file("✅ JAX operations test successful")
            return True
        except Exception as e:
            log_to_file(f"❌ Large matrix operation failed: {e}")
            return False
    except Exception as e:
        log_to_file(f"❌ JAX operations test failed: {e}")
        return False

def main():
    log_to_file(f"=== JAX DIAGNOSTICS REPORT ({timestamp}) ===")
    log_to_file(f"Report file: {REPORT_FILE}")
    
    # Run diagnostic checks
    check_system_info()
    check_cuda_env()
    check_library_paths()
    cudnn_loaded = try_load_cudnn()
    jax_installed = check_jax_installation()
    
    if jax_installed:
        operations_test = test_basic_operation()
    else:
        operations_test = False
    
    # Final status summary
    log_to_file("\n=== DIAGNOSTIC SUMMARY ===")
    log_to_file(f"cuDNN library load: {'✓ SUCCESS' if cudnn_loaded else '✗ FAILED'}")
    log_to_file(f"JAX installation: {'✓ SUCCESS' if jax_installed else '✗ FAILED'}")
    log_to_file(f"JAX operations test: {'✓ SUCCESS' if operations_test else '✗ FAILED'}")
    
    log_to_file("\n=== ISSUE INDICATION ===")
    
    if not cudnn_loaded:
        log_to_file("ISSUE: cuDNN library not loading - check libcudnn.so installation and version compatibility")
    
    if not jax_installed:
        log_to_file("ISSUE: JAX installation issues - check JAX/jaxlib versions and CUDA compatibility")
    
    if cudnn_loaded and jax_installed and not operations_test:
        log_to_file("ISSUE: JAX operations failing - likely cuDNN version mismatch with jaxlib or CUDA driver issues")
    
    if cudnn_loaded and jax_installed and operations_test:
        log_to_file("No immediate issues detected. If problems persist, check for specific operations that cause crashes.")
    
    log_to_file(f"\nDiagnostic report saved to: {REPORT_FILE}")
    print(f"\nDiagnostic report saved to: {REPORT_FILE}")

    # Try to load our custom GPU utils if available
    try:
        import site
        site.addsitedir("/workspace/src")
        from src.utils.jax_gpu_utils import gpu_snapshot
        snapshot = gpu_snapshot()
        log_to_file("\n=== GPU SNAPSHOT ===")
        log_to_file(json.dumps(snapshot, indent=2))
    except Exception as e:
        log_to_file(f"Failed to load gpu_snapshot: {e}")

if __name__ == "__main__":
    main() 