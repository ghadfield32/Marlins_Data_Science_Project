#!/usr/bin/env python
import os
import signal
import subprocess
import time
import datetime

def find_kernel_pids():
    try:
        # Find all ipykernel processes
        result = subprocess.run(
            ["pgrep", "-f", "ipykernel_launcher"],
            capture_output=True, 
            text=True
        )
        pids = result.stdout.strip().split('\n')
        return [pid for pid in pids if pid]
    except Exception as e:
        print(f"Error finding kernel PIDs: {e}")
        return []

def send_sigusr2(pid):
    try:
        os.kill(int(pid), signal.SIGUSR2)
        print(f"Sent SIGUSR2 to kernel PID {pid}")
        return True
    except Exception as e:
        print(f"Error sending SIGUSR2 to PID {pid}: {e}")
        return False

def find_crash_files():
    try:
        # Look in typical places for crash files
        locations = [
            "/tmp",
            os.path.expanduser("~/.local/share/jupyter/logs"),
            "."
        ]
        
        crash_files = []
        for loc in locations:
            if os.path.exists(loc):
                files = [os.path.join(loc, f) for f in os.listdir(loc) 
                         if "crash" in f or "kernel" in f and f.endswith(".txt")]
                crash_files.extend(files)
        
        return crash_files
    except Exception as e:
        print(f"Error finding crash files: {e}")
        return []

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"===== Kernel Stack Dump Utility ({timestamp}) =====")
    
    # 1. Find running kernels
    kernel_pids = find_kernel_pids()
    if not kernel_pids:
        print("No running kernel processes found")
    else:
        print(f"Found {len(kernel_pids)} running kernel process(es): {', '.join(kernel_pids)}")
        
        # 2. Send SIGUSR2 to each kernel
        for pid in kernel_pids:
            send_sigusr2(pid)
    
    # Wait a moment for dumps to be written
    time.sleep(1)
    
    # 3. Look for crash dump files
    print("\nLooking for crash dump files...")
    crash_files = find_crash_files()
    
    if not crash_files:
        print("No crash dump files found")
    else:
        print(f"Found {len(crash_files)} crash/dump files:")
        for f in crash_files:
            print(f"  - {f} ({os.path.getsize(f)} bytes)")
            
            # Preview the file if it exists and has content
            if os.path.exists(f) and os.path.getsize(f) > 0:
                try:
                    with open(f, 'r') as file:
                        lines = file.readlines()
                        if len(lines) > 10:
                            print("\nPreview (first 5 lines):")
                            for line in lines[:5]:
                                print(f"    {line.strip()}")
                            print("\n    ...\n")
                            print("Preview (last 5 lines):")
                            for line in lines[-5:]:
                                print(f"    {line.strip()}")
                        else:
                            print("\nFile contents:")
                            for line in lines:
                                print(f"    {line.strip()}")
                except Exception as e:
                    print(f"    Error reading file: {e}")
    
    # 4. Run jax_gpu_utils.snapshot if available
    print("\nAttempting to capture GPU snapshot...")
    try:
        import site
        site.addsitedir("/workspace/src")
        from src.utils.jax_gpu_utils import gpu_snapshot
        snapshot = gpu_snapshot()
        print("\nGPU Snapshot:")
        import json
        print(json.dumps(snapshot, indent=2))
    except Exception as e:
        print(f"Error capturing GPU snapshot: {e}")
        try:
            # Try alternate approach with nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )
            print("\nnvidia-smi output:")
            print(result.stdout.strip())
        except Exception as e2:
            print(f"Error running nvidia-smi: {e2}")

if __name__ == "__main__":
    main() 