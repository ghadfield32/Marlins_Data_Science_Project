import os, subprocess, json, logging, platform, jax
import shutil
from jax.lib import xla_bridge
from typing import Dict, List, Any, Optional, Tuple, Union

def gpu_diagnostics():
    info = {
        "backend":      xla_bridge.get_backend().platform,
        "devices":      [str(d) for d in jax.devices()],
        "python":       platform.python_version(),
        "ld_library_path": os.getenv("LD_LIBRARY_PATH","<unset>"),
    }
    if shutil.which("nvidia-smi"):
        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader,nounits"]
            )
            info["nvidia-smi"] = smi.decode().strip()
        except Exception as exc:
            info["nvidia-smi-error"] = repr(exc)
    return info

def log_gpu_diagnostics(level=logging.INFO):
    logger = logging.getLogger(__name__)
    # Make sure logging is configured
    if not logger.handlers and not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    
    logger.log(
        level,
        "GPU-diag: %s",
        json.dumps(gpu_diagnostics(), indent=2)
    )

def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get GPU memory information using nvidia-smi command.
    
    Returns:
        Dict containing GPU memory information from nvidia-smi
    """
    result = {"nvidia_smi": []}
    
    if not shutil.which("nvidia-smi"):
        return result
        
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip().split('\n')
        
        for line in output:
            values = [x.strip() for x in line.split(',')]
            if len(values) >= 5:
                gpu_info = {
                    "index": int(values[0]),
                    "total_memory_mb": int(values[1]),
                    "free_memory_mb": int(values[2]),
                    "used_memory_mb": int(values[3]),
                    "utilization": int(values[4]),
                }
                result["nvidia_smi"].append(gpu_info)
    except Exception as e:
        result["error"] = str(e)
        
    return result

def check_jax_gpu_memory() -> Dict[str, Any]:
    """
    Check JAX GPU memory status and return diagnostics with recommendations.
    
    Returns:
        Dict with memory information and recommendations
    """
    mem_info = get_gpu_memory_info()
    recommendations = []
    
    # Determine status
    status = "ok"
    
    # Add sample recommendations
    if not mem_info.get("nvidia_smi"):
        recommendations.append("No NVIDIA GPU detected or nvidia-smi unavailable")
        status = "error"
    else:
        for gpu in mem_info["nvidia_smi"]:
            util = gpu.get("utilization", 0)
            if util < 5:
                recommendations.append(f"GPU {gpu['index']} has low utilization ({util}%)")
                if status == "ok":
                    status = "warning"
    
    mem_info["recommendations"] = recommendations
    mem_info["status"] = status
    
    return mem_info

def force_gpu_memory_allocation(size_mb=1000) -> bool:
    """
    Force GPU memory allocation using JAX.
    
    Args:
        size_mb: Size in MB to allocate
        
    Returns:
        bool: Success status
    """
    try:
        # Import here to avoid circular imports
        from src.utils.jax_memory_fix_module import force_memory_allocation
        return force_memory_allocation(size_mb=size_mb, verbose=False)
    except Exception:
        return False
