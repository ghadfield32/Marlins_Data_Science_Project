"""
JAX Memory Monitor - Utilities for monitoring JAX memory usage during model training

This module provides tools to monitor and troubleshoot JAX GPU memory usage during
training of hierarchical models.
"""
import time
import logging
import json
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global memory snapshots storage
_memory_snapshots = []

def get_memory_usage() -> Dict[str, Any]:
    """Get current GPU memory usage information.
    
    Returns:
        dict: Memory usage information
    """
    from src.utils.jax_gpu_utils import get_gpu_memory_info
    
    memory_info = get_gpu_memory_info()
    if not memory_info:
        return {"error": "Could not get GPU memory info"}
        
    return memory_info

def take_memory_snapshot(label: str) -> Dict[str, Any]:
    """Take a snapshot of current GPU memory usage with a label.
    
    Args:
        label (str): A descriptive label for this memory snapshot
        
    Returns:
        dict: Memory snapshot with timestamp and label
    """
    global _memory_snapshots
    
    timestamp = time.time()
    memory_info = get_memory_usage()
    
    snapshot = {
        "timestamp": timestamp,
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
        "label": label,
        "memory": memory_info
    }
    
    _memory_snapshots.append(snapshot)
    return snapshot

def get_memory_snapshots() -> List[Dict[str, Any]]:
    """Get all collected memory snapshots.
    
    Returns:
        list: List of memory snapshots
    """
    global _memory_snapshots
    return _memory_snapshots

def clear_memory_snapshots():
    """Clear all collected memory snapshots."""
    global _memory_snapshots
    _memory_snapshots = []

def print_memory_snapshot(snapshot: Dict[str, Any]):
    """Print a formatted memory snapshot.
    
    Args:
        snapshot (dict): Memory snapshot to print
    """
    print(f"=== Memory Snapshot: {snapshot['label']} ({snapshot['time_str']}) ===")
    
    if "error" in snapshot:
        print(f"Error: {snapshot['error']}")
        return
        
    if "memory" in snapshot and "nvidia_smi" in snapshot["memory"]:
        for gpu in snapshot["memory"]["nvidia_smi"]:
            print(f"GPU {gpu['gpu_id']}: {gpu['used_mb']}MB / {gpu['total_mb']}MB ({gpu['utilization']}%)")
    else:
        print("No GPU memory information available")

@contextmanager
def monitor_memory_usage(operation_name: str, verbose: bool = True):
    """Context manager to monitor GPU memory usage before and after an operation.
    
    Args:
        operation_name (str): Name of the operation being monitored
        verbose (bool): Whether to print memory information
        
    Yields:
        None
    """
    # Take snapshot before operation
    before_snapshot = take_memory_snapshot(f"{operation_name} - Before")
    if verbose:
        print_memory_snapshot(before_snapshot)
    
    start_time = time.time()
    
    try:
        # Yield control back to the context block
        yield
    finally:
        # Take snapshot after operation
        elapsed = time.time() - start_time
        after_snapshot = take_memory_snapshot(f"{operation_name} - After ({elapsed:.2f}s)")
        
        if verbose:
            print_memory_snapshot(after_snapshot)
            
            # Calculate and print memory difference
            if ("memory" in before_snapshot and "nvidia_smi" in before_snapshot["memory"] and
                "memory" in after_snapshot and "nvidia_smi" in after_snapshot["memory"]):
                
                before_gpu = before_snapshot["memory"]["nvidia_smi"][0]
                after_gpu = after_snapshot["memory"]["nvidia_smi"][0]
                
                memory_diff = after_gpu["used_mb"] - before_gpu["used_mb"]
                util_diff = after_gpu["utilization"] - before_gpu["utilization"]
                
                print(f"Memory change: {memory_diff:+}MB ({util_diff:+.2f}%)")

def force_allocation_if_needed(target_fraction: float = 0.8, 
                              current_usage_threshold: float = 0.4,
                              step_size_mb: int = 1000,
                              max_steps: int = 10,
                              verbose: bool = True) -> Dict[str, Any]:
    """Force memory allocation if current usage is below threshold.
    
    Args:
        target_fraction (float): Target memory utilization (0.0 to 1.0)
        current_usage_threshold (float): Only allocate if current usage below this threshold
        step_size_mb (int): Size of each allocation step in MB
        max_steps (int): Maximum number of allocation steps
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Results of allocation attempt
    """
    # Get current memory usage
    memory_info = get_memory_usage()
    if not memory_info or "nvidia_smi" not in memory_info:
        logger.warning("Could not get GPU memory info")
        return {"success": False, "error": "Could not get GPU memory info"}
    
    # Calculate current utilization
    current_util = memory_info["nvidia_smi"][0]["utilization"] / 100.0
    
    if verbose:
        print(f"Current GPU memory utilization: {current_util:.2%}")
        print(f"Target utilization: {target_fraction:.2%}")
    
    # Skip if already above threshold
    if current_util >= current_usage_threshold:
        if verbose:
            print(f"Current utilization {current_util:.2%} already above threshold {current_usage_threshold:.2%}, skipping allocation")
        return {"success": True, "action": "skipped", "current_utilization": current_util}
    
    # Import here to avoid circular imports
    from src.utils.jax_memory_fix_module import allocate_memory_incremental
    
    # Perform allocation
    result = allocate_memory_incremental(
        target_fraction=target_fraction,
        step_size_mb=step_size_mb,
        max_steps=max_steps,
        check_interval=0.5,
        verbose=verbose
    )
    
    return result

def generate_memory_report(output_file: Optional[str] = None) -> Dict[str, Any]:
    """Generate a report of memory usage based on collected snapshots.
    
    Args:
        output_file (str, optional): Path to save the report as JSON
        
    Returns:
        dict: Memory usage report
    """
    global _memory_snapshots
    
    # Skip if no snapshots
    if not _memory_snapshots:
        return {"status": "No memory snapshots collected"}
    
    # Prepare report structure
    report = {
        "snapshots_count": len(_memory_snapshots),
        "time_range": {
            "start": _memory_snapshots[0]["time_str"],
            "end": _memory_snapshots[-1]["time_str"],
            "duration_seconds": _memory_snapshots[-1]["timestamp"] - _memory_snapshots[0]["timestamp"]
        },
        "snapshots": _memory_snapshots,
        "summary": {}
    }
    
    # Extract GPU usage patterns if available
    gpu_utils = []
    
    for snapshot in _memory_snapshots:
        if "memory" in snapshot and "nvidia_smi" in snapshot["memory"]:
            for gpu in snapshot["memory"]["nvidia_smi"]:
                gpu_utils.append(gpu["utilization"])
    
    if gpu_utils:
        report["summary"]["gpu_utilization"] = {
            "min": min(gpu_utils),
            "max": max(gpu_utils),
            "avg": sum(gpu_utils) / len(gpu_utils)
        }
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Memory report saved to {output_file}")
    
    return report

def monitor_training_session(func: Callable, *args, **kwargs) -> Any:
    """Decorator to monitor memory usage during an entire training session.
    
    Args:
        func (callable): The training function to monitor
        *args, **kwargs: Arguments to pass to the training function
        
    Returns:
        Any: Result of the training function
    """
    # Take snapshot before training
    take_memory_snapshot("Training - Start")
    
    # Force allocation if needed
    force_result = force_allocation_if_needed(
        target_fraction=0.8,
        current_usage_threshold=0.3,
        verbose=True
    )
    
    logger.info(f"Pre-training allocation: {json.dumps(force_result, indent=2)}")
    
    # Run training function
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time
    
    # Take snapshot after training
    take_memory_snapshot(f"Training - Complete ({elapsed:.2f}s)")
    
    # Generate report
    report = generate_memory_report("memory_training_report.json")
    
    # Print summary
    if "summary" in report and "gpu_utilization" in report["summary"]:
        util = report["summary"]["gpu_utilization"]
        print(f"\nTraining Complete in {elapsed:.2f}s")
        print(f"GPU Utilization - Min: {util['min']:.2f}% Max: {util['max']:.2f}% Avg: {util['avg']:.2f}%")
    
    return result 