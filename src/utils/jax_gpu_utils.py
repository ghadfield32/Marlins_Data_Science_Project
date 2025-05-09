
import os, subprocess, json, logging, platform, jax
from jax.lib import xla_bridge

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
    logging.getLogger(__name__).log(
        level,
        "GPU-diag: %s",
        json.dumps(gpu_diagnostics(), indent=2)
    )
