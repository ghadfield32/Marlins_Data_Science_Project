#!/usr/bin/env python
"""
Monitor GPU Memory Usage

Simple utility script to monitor GPU memory usage in real-time.
Run this in a separate terminal while running your JAX code to see memory utilization.

Usage:
    python bin/monitor_gpu.py [interval] [count]
    
    interval: Time in seconds between checks (default: 2)
    count: Number of checks to run (default: -1, run forever)

Example:
    python bin/monitor_gpu.py 1 10  # Check every second for 10 times
"""
import os
import sys
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import memory monitoring utilities
from src.utils.jax_gpu_utils import print_gpu_memory_summary

def main():
    parser = argparse.ArgumentParser(description='Monitor GPU memory usage')
    parser.add_argument('interval', nargs='?', type=float, default=2.0,
                        help='Time in seconds between checks (default: 2)')
    parser.add_argument('count', nargs='?', type=int, default=-1,
                        help='Number of checks to run (default: -1, run forever)')
    
    args = parser.parse_args()
    
    print(f"Monitoring GPU memory every {args.interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        i = 0
        while args.count == -1 or i < args.count:
            print(f"\n--- Check {i+1} ---")
            print_gpu_memory_summary()
            
            time.sleep(args.interval)
            i += 1
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
if __name__ == '__main__':
    main() 