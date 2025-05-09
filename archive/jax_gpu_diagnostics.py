#!/usr/bin/env python
"""
JAX/CUDA/cuDNN Environment Diagnostic Tool

This script performs a series of tests to verify the JAX GPU environment
and help diagnose issues with CUDA, cuDNN, and JAX integration.
"""
import os
import sys
import subprocess
import platform
import traceback
import json
from pathlib import Path
from datetime import datetime

# Enable Python fault handler for better crash tracing
os.environ["PYTHONFAULTHANDLER"] = "1"

# Create a directory for logs
log_dir = Path("jax_diagnostics_logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"diagnostics_{timestamp}.log"

def log(message, also_print=True):
    """Log message to file and optionally print to console"""
    with open(log_file, "a") as f:
        f.write(f"{message}\n")
    if also_print:
        print(message)

def run_command(cmd, shell=False):
    """Run a shell command and return stdout/stderr"""
    try:
        result = subprocess.run(
            cmd, 
            shell=shell, 
            text=True, 
            capture_output=True, 
            check=False
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Error executing command: {str(e)}",
            "returncode": -1
        }

def section_header(title):
    """Print and log a section header"""
    header = f"\n{'=' * 40}\n{title}\n{'=' * 40}"
    log(header)

def run_and_log(cmd, description, shell=False):
    """Run a command, log the output, and return the result"""
    log(f"\n>> {description}")
    log(f"   Command: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    
    result = run_command(cmd, shell=shell)
    
    log(f"   Return code: {result['returncode']}")
    if result['stdout']:
        log(f"   stdout:\n{result['stdout'].strip()}")
    if result['stderr']:
        log(f"   stderr:\n{result['stderr'].strip()}")
        
    return result

def test_jax_imports():
    """Test importing JAX and check versions"""
    section_header("JAX Version Check")
    try:
        import jax
        import jaxlib
        
        log(f"JAX version: {jax.__version__}")
        log(f"JAXlib version: {jaxlib.__version__}")
        
        # Get build info
        log("JAX build information:")
        try:
            log(json.dumps(jax._src.lib.version_info(), indent=2))
        except:
            log("Could not retrieve JAX build information")
        
        return True
    except ImportError as e:
        log(f"Error importing JAX: {str(e)}")
        return False

def check_system_info():
    """Check system information"""
    section_header("System Information")
    
    log(f"Python version: {sys.version}")
    log(f"Platform: {platform.platform()}")
    log(f"Processor: {platform.processor()}")
    
    # Check environment variables
    env_vars = [
        "LD_LIBRARY_PATH", 
        "PYTHONPATH", 
        "CUDA_HOME", 
        "CUDNN_HOME",
        "JAX_PLATFORM_NAME",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_FLAGS"
    ]
    
    log("\nRelevant environment variables:")
    for var in env_vars:
        log(f"  {var}={os.environ.get(var, 'Not set')}")

def check_cuda_installation():
    """Check CUDA toolkit installation"""
    section_header("CUDA Installation")
    
    # Check NVCC version
    run_and_log("nvcc --version", "NVCC version")
    
    # Check NVIDIA drivers
    run_and_log("nvidia-smi", "NVIDIA Driver information")
    
    # Check cuDNN
    run_and_log("ldconfig -p | grep libcudnn", "cuDNN libraries in ldconfig", shell=True)
    
    cudnn_paths = [
        "/usr/lib/x86_64-linux-gnu/libcudnn.so*",
        "/usr/local/cuda/lib64/libcudnn.so*",
        "/opt/conda/envs/marlins-ds-gpu/lib/libcudnn.so*"
    ]
    
    for cudnn_path in cudnn_paths:
        run_and_log(f"ls -la {cudnn_path} 2>/dev/null", f"Check cuDNN at {cudnn_path}", shell=True)
    
    # Check if we can find a cuDNN .so file
    found_cudnn = None
    for cudnn_ver in range(7, 10):  # Check cuDNN 7-9
        result = run_command(f"find /usr -name 'libcudnn.so.{cudnn_ver}*' 2>/dev/null", shell=True)
        if result['stdout'].strip():
            found_cudnn = result['stdout'].strip().split('\n')[0]
            break
    
    if found_cudnn:
        run_and_log(f"strings {found_cudnn} | grep CUDNN_MAJOR -A2", 
                   f"cuDNN version in {found_cudnn}", shell=True)

def check_library_dependencies():
    """Check library dependencies"""
    section_header("Library Dependencies")
    
    try:
        import jaxlib
        jaxlib_path = jaxlib.__file__
        log(f"JAXlib path: {jaxlib_path}")
        
        # Check library dependencies of JAXlib
        run_and_log(f"ldd {jaxlib_path}", "JAXlib dependencies")
        
        # Find and check the XLA module
        xla_path = None
        jaxlib_dir = os.path.dirname(jaxlib_path)
        
        for root, dirs, files in os.walk(jaxlib_dir):
            for file in files:
                if file.endswith('.so') and 'xla_extension' in file:
                    xla_path = os.path.join(root, file)
                    break
            if xla_path:
                break
        
        if xla_path:
            log(f"Found XLA extension at: {xla_path}")
            run_and_log(f"ldd {xla_path}", "XLA extension dependencies")
            run_and_log(f"ls -la {xla_path}", "XLA extension file info")
        else:
            log("Could not find XLA extension module")
    
    except ImportError:
        log("Could not import jaxlib")

def run_jax_gpu_test():
    """Run a minimal JAX GPU test"""
    section_header("JAX GPU Minimal Test")
    
    log("Testing basic JAX GPU operations")
    try:
        import jax
        import jax.numpy as jnp
        
        log(f"Available devices: {jax.devices()}")
        
        log("Creating test arrays")
        x = jnp.ones((10, 10))
        
        log("Performing matrix multiplication")
        y = jnp.dot(x, x)
        
        log("Ensuring computation is complete")
        y.block_until_ready()
        
        log("Test successful!")
        return True
    
    except Exception:
        log(f"JAX GPU test failed with exception:\n{traceback.format_exc()}")
        return False

def run_stress_test():
    """Run a more intensive JAX GPU test"""
    section_header("JAX GPU Stress Test")
    
    try:
        import jax
        import jax.numpy as jnp
        from jax import random
        
        log("Running larger matrix operations")
        
        # Generate random key
        key = random.key(0)
        
        # Create larger matrices (2000x2000)
        log("Creating 2000x2000 random matrices")
        a = random.normal(key, (2000, 2000))
        b = random.normal(key, (2000, 2000))
        
        log("Performing matrix multiplication")
        c = jnp.dot(a, b)
        
        log("Ensuring computation is complete")
        c.block_until_ready()
        
        log("Matrix multiplication successful")
        
        # Try some more operations
        log("Testing additional operations")
        d = jnp.sin(c)
        e = jnp.exp(jnp.abs(d) * 0.01)
        e.block_until_ready()
        
        log("All operations completed successfully")
        return True
    
    except Exception:
        log(f"JAX GPU stress test failed with exception:\n{traceback.format_exc()}")
        return False

def check_xla_flags():
    """Set XLA_FLAGS and run a basic test to capture dumps"""
    section_header("XLA Debug Dump Test")
    
    # Create a directory for XLA dumps
    xla_dump_dir = log_dir / "xla_dump"
    xla_dump_dir.mkdir(exist_ok=True)
    
    # Set XLA_FLAGS for debugging
    os.environ["XLA_FLAGS"] = f"--xla_dump_to={xla_dump_dir} --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*"
    
    log(f"Set XLA_FLAGS to: {os.environ['XLA_FLAGS']}")
    log(f"XLA dumps will be written to: {xla_dump_dir}")
    
    # Run a simple JAX operation to generate dumps
    try:
        import jax
        import jax.numpy as jnp
        
        log("Running simple operation to generate XLA dumps")
        x = jnp.ones((5, 5))
        y = jnp.dot(x, x)
        y.block_until_ready()
        
        # Check if dumps were created
        dump_files = list(xla_dump_dir.glob("*"))
        log(f"Generated {len(dump_files)} XLA dump files")
        
        # List the first few dump files
        if dump_files:
            log("Sample dump files:")
            for file in dump_files[:5]:
                log(f"  {file.name}")
        
        return True
    
    except Exception:
        log(f"XLA dump generation failed with exception:\n{traceback.format_exc()}")
        return False

def test_cudnn_load():
    """Test loading cuDNN directly"""
    section_header("Direct cuDNN Loading Test")
    
    try:
        import ctypes
        
        log("Attempting to load libcudnn.so via ctypes")
        
        # Try different cuDNN versions
        for version in [9, 8, 7]:
            try:
                log(f"Trying libcudnn.so.{version}")
                ctypes.CDLL(f"libcudnn.so.{version}", mode=ctypes.RTLD_GLOBAL)
                log(f"Successfully loaded libcudnn.so.{version}")
                return True
            except Exception as e:
                log(f"Failed to load libcudnn.so.{version}: {str(e)}")
        
        log("Could not load any cuDNN library")
        return False
    
    except Exception:
        log(f"cuDNN loading test failed with exception:\n{traceback.format_exc()}")
        return False

def generate_report():
    """Generate a summary report"""
    section_header("Diagnostic Summary")
    
    log("JAX/CUDA/cuDNN Environment Diagnostic Report")
    log(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Log file: {log_file}")
    
    # Provide recommendations based on findings
    recommendations = [
        "1. If JAX imports but GPU operations fail, check cuDNN compatibility",
        "2. If ldd shows missing libraries, install the required versions",
        "3. Verify LD_LIBRARY_PATH includes the directories with cuDNN libraries",
        "4. If XLA dumps show errors, check the HLO optimization logs",
        "5. Consider testing with JAX CPU backend (set JAX_PLATFORM_NAME=cpu) to isolate issues",
        "6. Check for version mismatches between CUDA, cuDNN, and JAX"
    ]
    
    log("\nRecommendations:")
    for rec in recommendations:
        log(f"  {rec}")
    
    log("\nNext steps:")
    log("  1. Review the full log file for detailed diagnostics")
    log("  2. Check XLA dump files for compilation/optimization errors")
    log("  3. Consider running with CPU-only mode to verify JAX functionality")
    log("  4. Use docker 'nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04' to verify drivers")
    
    log("\nDiagnostic complete! Results saved to {log_file}")

def main():
    """Main function to run all diagnostic tests"""
    section_header("JAX/CUDA/cuDNN Environment Diagnostics")
    log(f"Starting diagnostics at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run the tests
    check_system_info()
    jax_imported = test_jax_imports()
    check_cuda_installation()
    check_library_dependencies()
    
    if jax_imported:
        run_jax_gpu_test()
        run_stress_test()
        test_cudnn_load()
        check_xla_flags()
    
    # Generate the summary report
    generate_report()
    
    log("\nDiagnostic script completed successfully.")
    log(f"Full diagnostic log saved to: {log_file}")
    print(f"\nDiagnostic complete! Full results saved to {log_file}")

if __name__ == "__main__":
    main() 