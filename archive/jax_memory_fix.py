#!/usr/bin/env python
"""
JAX Memory Allocation Fix for Docker Environments

location: jax_memory_fix.py
This script addresses GPU memory allocation issues in JAX when running in Docker containers.
"""
import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
import re  # Add missing import

def parse_args():
    parser = argparse.ArgumentParser(description="Fix JAX memory allocation in Docker")
    parser.add_argument("--memory-fraction", type=float, default=0.9,
                        help="Fraction of GPU memory to allocate (default: 0.9)")
    parser.add_argument("--env-file", type=str, default=".env.dev",
                        help="Environment file to update (default: .env.dev)")
    parser.add_argument("--apply", action="store_true",
                        help="Apply changes to environment file")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run diagnostic tests and provide detailed information")
    parser.add_argument("--fix", action="store_true",
                        help="Apply all fixes automatically")
    return parser.parse_args()

def generate_env_vars(mem_fraction):
    """Generate environment variables for JAX memory configuration"""
    env_vars = {
        # Memory management settings
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",  # Changed to true for better preallocation
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem_fraction),
        
        # GPU platform
        "JAX_PLATFORM_NAME": "gpu",
        
        # Additional XLA flags
        "XLA_FLAGS": "--xla_gpu_deterministic_reductions --xla_force_host_platform_device_count=1",
        
        # Performance settings
        "JAX_DISABLE_JIT": "false",
        "JAX_ENABLE_X64": "false",  # Keep 32-bit for better performance

        # New settings to force memory allocation
        "TF_FORCE_GPU_ALLOW_GROWTH": "false",  # Force full allocation in TensorFlow
    }
    return env_vars

def update_env_file(filename, env_vars):
    """Update environment file with JAX memory settings"""
    if not os.path.exists(filename):
        print(f"Creating new environment file: {filename}")
        with open(filename, "w") as f:
            f.write("# JAX Memory Configuration\n")
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        return True
    
    # Read existing file
    with open(filename, "r") as f:
        lines = f.readlines()
    
    # Process lines
    new_lines = []
    updated_keys = set()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            new_lines.append(line)
            continue
        
        # Check if this line sets any of our variables
        parts = line.split("=", 1)
        if len(parts) != 2:
            new_lines.append(line)
            continue
        
        key, _ = parts
        if key in env_vars:
            new_lines.append(f"{key}={env_vars[key]}")
            updated_keys.add(key)
        else:
            new_lines.append(line)
    
    # Add any missing variables
    if updated_keys != set(env_vars.keys()):
        new_lines.append("\n# JAX Memory Configuration")
        for key, value in env_vars.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={value}")
    
    # Write back to file
    with open(filename, "w") as f:
        f.write("\n".join(new_lines) + "\n")
    
    return True

def update_docker_compose(filename="docker-compose.yml", env_vars=None):
    """Update docker-compose.yml with JAX memory settings"""
    if env_vars is None:
        env_vars = generate_env_vars(0.9)
    
    if not os.path.exists(filename):
        print(f"docker-compose file {filename} not found. Skipping.")
        return False
    
    try:
        # Read the docker-compose.yml file
        with open(filename, "r") as f:
            content = f.read()
        
        # Check if environment section exists
        if "environment:" not in content:
            print("No environment section found in docker-compose.yml. Manual update needed.")
            return False
        
        # Build environment string for replacement
        env_section = "    environment:\n"
        for key, value in env_vars.items():
            env_section += f"      - {key}={value}\n"
        
        # TODO: This is a simplified approach. For a real production script,
        # consider using a proper YAML parser to modify the file.
        # For now, print instructions
        print(f"Add the following to your {filename} file's environment section:")
        print(env_section)
        return True
    
    except Exception as e:
        print(f"Error updating docker-compose.yml: {str(e)}")
        return False

def check_gpu_memory():
    """Check current GPU memory usage"""
    try:
        # Try to use nvidia-smi
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"], 
            text=True
        ).strip()
        
        print("GPU Memory Status:")
        print("------------------")
        
        lines = output.splitlines()
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 3:
                total, free, used = map(int, parts)
                print(f"GPU {i}: {used} MiB used / {total} MiB total ({used/total:.1%})")
        
        return True
    except Exception as e:
        print(f"Error checking GPU memory: {str(e)}")
        return False

def diagnose_jax_installation():
    """Diagnose JAX installation and memory issues"""
    print("\nJAX Installation Diagnosis")
    print("-------------------------")
    
    try:
        # Check if JAX is installed
        result = subprocess.run(
            [sys.executable, "-c", "import jax; print(jax.__version__)"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            jax_version = result.stdout.strip()
            print(f"✓ JAX is installed (version {jax_version})")
        else:
            print("✗ JAX is not properly installed")
            print(f"Error: {result.stderr}")
            return False
        
        # Check CUDA availability
        result = subprocess.run(
            [sys.executable, "-c", "import jax; print(jax.devices())"],
            capture_output=True, text=True
        )
        
        if "CudaDevice" in result.stdout:
            print("✓ CUDA devices are available to JAX")
            devices = result.stdout.strip()
            print(f"  Devices: {devices}")
        else:
            print("✗ No CUDA devices available to JAX")
            print(f"  Output: {result.stdout}")
            print(f"  Error: {result.stderr}")
        
        # Check current environment variables
        print("\nEnvironment Variables:")
        for var in ["XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_ALLOCATOR", 
                   "XLA_PYTHON_CLIENT_MEM_FRACTION", "JAX_PLATFORM_NAME", "XLA_FLAGS"]:
            value = os.environ.get(var, "Not set")
            print(f"  {var}={value}")
        
        # Check memory allocation behavior
        print("\nMemory Allocation Test:")
        test_script = """
import os
import time
import json
import jax
import jax.numpy as jnp

# Get initial memory info
from jax.lib import xla_client as xc
if hasattr(xc, "get_gpu_memory_info"):
    free1, total1 = xc.get_gpu_memory_info(0)
    used1 = total1 - free1
    
    # Run an allocation test
    x = jnp.ones((8000, 8000), dtype=jnp.float32)
    y = jnp.dot(x, x)
    y.block_until_ready()
    time.sleep(1)
    
    # Get final memory info
    free2, total2 = xc.get_gpu_memory_info(0)
    used2 = total2 - free2
    
    print(json.dumps({
        "initial_used_percent": used1/total1,
        "final_used_percent": used2/total2,
        "memory_increase_gb": (used2-used1)/1e9,
        "total_memory_gb": total1/1e9
    }))
else:
    print(json.dumps({"error": "xc.get_gpu_memory_info not available"}))
"""
        
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True, text=True
        )
        
        try:
            data = json.loads(result.stdout.strip())
            if "error" in data:
                print(f"✗ Memory test failed: {data['error']}")
            else:
                print(f"  Initial GPU memory usage: {data['initial_used_percent']:.1%}")
                print(f"  After allocation: {data['final_used_percent']:.1%}")
                print(f"  Memory increase: {data['memory_increase_gb']:.2f} GB")
                print(f"  Total GPU memory: {data['total_memory_gb']:.2f} GB")
                
                if data['final_used_percent'] < 0.1:
                    print("✗ JAX is not allocating enough GPU memory")
                else:
                    print("✓ JAX memory allocation seems to be working")
        except json.JSONDecodeError:
            print("✗ Memory test failed to execute properly")
            print(f"  Output: {result.stdout}")
            print(f"  Error: {result.stderr}")
        
        return True
        
    except Exception as e:
        print(f"Diagnosis failed: {str(e)}")
        return False

def run_memory_enforcement_test():
    """Test script to force JAX to allocate memory"""
    print("\nRunning memory enforcement test...")
    
    # Create test script
    test_script_path = "jax_memory_test.py"
    with open(test_script_path, "w") as f:
        f.write("""
import os
import time
import json
import subprocess
import re  # Add missing import

# Force environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["JAX_PLATFORM_NAME"] = "gpu"

# Now import JAX
import jax
import jax.numpy as jnp
from jax.lib import xla_client as xc

# Define function to check GPU memory
def check_gpu_memory():
    if hasattr(xc, "get_gpu_memory_info"):
        free, total = xc.get_gpu_memory_info(0)
    else:
        # Fall back to nvidia-smi
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"], text=True).splitlines()[0]
        total, free = (int(s.strip()) for s in re.split(r",\\s*", output, maxsplit=1))
        free *= 1_048_576  # MiB to bytes
        total *= 1_048_576  # MiB to bytes
    
    used = total - free
    used_percent = used / total
    return free, total, used, used_percent

# Check memory before operations
free1, total1, used1, percent1 = check_gpu_memory()

# Force memory allocation by creating large tensors
print("Creating tensors to force memory allocation...")
tensors = []
for i in range(10):
    # Create a 1GB tensor each iteration
    x = jnp.ones((8192, 8192), dtype=jnp.float32)
    y = jnp.matmul(x, x)
    y.block_until_ready()
    tensors.append(y)  # Keep reference to prevent garbage collection
    
    # Check memory
    free_now, total_now, used_now, percent_now = check_gpu_memory()
    print(f"Iteration {i+1}: Using {used_now/1e9:.2f} GB ({percent_now:.2%})")
    
    # Break if we've allocated enough
    if percent_now >= 0.4:
        print(f"Successfully allocated {percent_now:.2%} of GPU memory")
        break

# Final memory check
free2, total2, used2, percent2 = check_gpu_memory()

# Output results
result = {
    "initial_percent": percent1,
    "final_percent": percent2,
    "success": percent2 >= 0.4
}
print(json.dumps(result))
""")
    
    # Run the test
    print("Executing memory enforcement test...")
    result = subprocess.run(
        [sys.executable, test_script_path],
        capture_output=True, text=True
    )
    
    print("\nTest output:")
    print(result.stdout)
    
    if result.stderr:
        print("\nErrors:")
        print(result.stderr)
    
    # Clean up
    try:
        os.remove(test_script_path)
    except:
        pass

def check_jax_version_compatibility():
    """Check if JAX version is compatible with CUDA/cuDNN"""
    print("\nChecking JAX/CUDA compatibility...")
    
    try:
        # Get JAX version
        jax_version = subprocess.check_output(
            [sys.executable, "-c", "import jax; print(jax.__version__)"],
            text=True
        ).strip()
        
        # Get CUDA version
        cuda_version = subprocess.check_output(
            ["nvcc", "--version"], text=True
        )
        cuda_version = cuda_version.split("release ")[-1].split(",")[0]
        
        # Get cuDNN version if possible
        try:
            cudnn_version = subprocess.check_output(
                ["bash", "-c", "strings /usr/lib/x86_64-linux-gnu/libcudnn.so | grep CUDNN_MAJOR -A2"],
                text=True
            ).strip()
        except:
            cudnn_version = "Unknown"
        
        print(f"JAX version: {jax_version}")
        print(f"CUDA version: {cuda_version}")
        print(f"cuDNN version: {cudnn_version}")
        
        # Compatibility checks
        if jax_version.startswith("0.5") and float(cuda_version) >= 12.0:
            print("\n⚠️ Warning: JAX 0.5.x works best with CUDA 12.x, but needs the correct cuDNN version")
            print("   For CUDA 12.3, cuDNN 9.x is recommended")
        
        return True
    except Exception as e:
        print(f"Error checking compatibility: {str(e)}")
        return False

def apply_fixes():
    """Apply all fixes automatically"""
    print("Applying fixes to JAX memory allocation...")
    
    # 1. Set environment variables in current process
    env_vars = generate_env_vars(0.9)
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    # 2. Update .env.dev file
    update_env_file(".env.dev", env_vars)
    print("Updated .env.dev file")
    
    # 3. Update dev.env file if it exists
    if os.path.exists("dev.env"):
        update_env_file("dev.env", env_vars)
        print("Updated dev.env file")
    
    # 4. Create a fix script for the container startup
    fix_script = Path("fix_jax_memory.py")
    with open(fix_script, "w") as f:
        f.write("""
#!/usr/bin/env python
\"\"\"
JAX Memory Fix Script - Run on container startup
\"\"\"
import os

# Force JAX memory settings
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions --xla_force_host_platform_device_count=1"

# Print memory fix message
print("JAX memory settings applied")

# Run any additional commands passed in
import sys
if len(sys.argv) > 1:
    import subprocess
    subprocess.run(sys.argv[1:])
""")
    
    fix_script.chmod(0o755)  # Make executable
    print(f"Created {fix_script} startup fix script")
    
    # 5. Add fix instructions for docker-compose
    update_docker_compose()
    
    print("\nFixes applied successfully!")
    print("\nTo run Python with fixes, use:")
    print(f"python {fix_script} your_script.py")
    
    return True

def print_recommendations():
    """Print recommendations for Docker and JAX memory settings"""
    print("\nRecommendations for JAX/GPU memory in Docker:")
    print("1. Add these environment variables to your docker-compose.yml:")
    print("   environment:")
    print("     - XLA_PYTHON_CLIENT_PREALLOCATE=true")
    print("     - XLA_PYTHON_CLIENT_ALLOCATOR=platform")
    print("     - XLA_PYTHON_CLIENT_MEM_FRACTION=0.9")
    print("     - JAX_PLATFORM_NAME=gpu")
    print("     - XLA_FLAGS=--xla_gpu_deterministic_reductions --xla_force_host_platform_device_count=1")
    print()
    print("2. Ensure your container has full GPU access:")
    print("   gpus: all")
    print("   runtime: nvidia")
    print()
    print("3. Include these settings in your Python code:")
    print("   import os")
    print("   os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'")
    print("   os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'")
    print("   os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'")
    print("   # Import JAX after setting environment variables")
    print("   import jax")
    print()
    print("4. Use mixed precision (float16/bfloat16) for most operations:")
    print("   from jax import lax")
    print("   import jax.numpy as jnp")
    print("   # Use bfloat16 or float16 for matrices")
    print("   x = jnp.ones((1000, 1000), dtype=jnp.bfloat16)")
    print()
    print("5. Fix BAR1 memory allocation by adding to docker run command:")
    print("   --gpus all,capabilities=utility,compute,graphics,display")
    print()
    print("6. If you need to run with a specific JAX/cuDNN version:")
    print("   Update your container to use CUDA 12.3 + cuDNN 9.x for JAX 0.5.x:")
    print("   FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04")

def main():
    """Main function"""
    args = parse_args()
    
    print("JAX Memory Allocation Fixer for Docker")
    
    if args.diagnose:
        check_gpu_memory()
        diagnose_jax_installation()
        check_jax_version_compatibility()
        run_memory_enforcement_test()
        return
    
    if args.fix:
        apply_fixes()
        return
    
    print(f"Memory fraction: {args.memory_fraction}")
    
    # Generate environment variables
    env_vars = generate_env_vars(args.memory_fraction)
    
    # Print environment variables
    print("\nGenerated environment variables:")
    for key, value in env_vars.items():
        print(f"{key}={value}")
    
    # Update environment file if requested
    if args.apply:
        print(f"\nUpdating environment file: {args.env_file}")
        update_env_file(args.env_file, env_vars)
        print("Environment file updated successfully")
    else:
        print("\nDry run mode. Use --apply to update environment file.")
    
    # Print recommendations
    print_recommendations()

if __name__ == "__main__":
    main() 