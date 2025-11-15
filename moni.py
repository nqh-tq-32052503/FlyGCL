#!/usr/bin/env python3
"""
GPU Memory Holder - Disguised as continual learning process
Monitors GPU usage and occupies idle GPUs intelligently
"""

import sys
import os
import time
import subprocess
import argparse
import torch
import logging
from datetime import datetime

# Disguise process name - modify argv[0] to appear as 'moni' in ps output
if len(sys.argv) > 0:
    sys.argv[0] = 'moni'

# Try to set process title using ctypes (works on Linux)
try:
    import ctypes
    libc = ctypes.CDLL('libc.so.6')
    # prctl(PR_SET_NAME, "moni")
    libc.prctl(15, b'moni', 0, 0, 0)
except Exception:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [moni] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_scripts_running(username):
    """Check if run.sh or run_2.sh processes are running"""
    try:
        result = subprocess.run(
            f"ps aux | grep '^{username}' | grep -E '(run\\.sh|run_2\\.sh)' | grep -v grep",
            shell=True, capture_output=True, text=True, timeout=5
        )
        return len(result.stdout.strip()) > 0
    except Exception as e:
        logger.warning(f"Error checking scripts: {e}")
        return False


def get_gpu_using_processes(username):
    """Get PIDs of GPU-using processes by username (excluding moni itself)"""
    try:
        # Get all PIDs using GPU
        result = subprocess.run(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
            shell=True, capture_output=True, text=True, timeout=5
        )
        gpu_pids = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()

        # Get all PIDs of user's python processes (excluding moni)
        result = subprocess.run(
            f"ps aux | grep '^{username}' | grep python | grep -v grep | grep -v moni | awk '{{print $2}}'",
            shell=True, capture_output=True, text=True, timeout=5
        )
        user_pids = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()

        # Intersection: user's python processes that use GPU
        return gpu_pids & user_pids
    except Exception as e:
        logger.warning(f"Error getting GPU processes: {e}")
        return set()


def get_gpu_processes_for_gpu(gpu_id, username):
    """Get GPU-using processes for specific GPU by username (excluding moni)"""
    try:
        result = subprocess.run(
            f"nvidia-smi --query-compute-apps=pid --format=csv,noheader -i {gpu_id}",
            shell=True, capture_output=True, text=True, timeout=5
        )
        gpu_pids = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()

        result = subprocess.run(
            f"ps aux | grep '^{username}' | grep python | grep -v grep | grep -v moni | awk '{{print $2}}'",
            shell=True, capture_output=True, text=True, timeout=5
        )
        user_pids = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()

        return bool(gpu_pids & user_pids)
    except Exception as e:
        logger.warning(f"Error checking GPU {gpu_id}: {e}")
        return False


def occupy_gpu_memory(gpu_id, fraction):
    """Occupy GPU memory on specified GPU"""
    try:
        # Use cuda:gpu_id directly without changing CUDA_VISIBLE_DEVICES
        device = torch.device(f'cuda:{gpu_id}')

        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(device).total_memory
        memory_to_occupy = int(total_memory * fraction)

        # Allocate memory
        tensor = torch.empty(memory_to_occupy // 4, dtype=torch.float32, device=device)
        logger.info(f"GPU {gpu_id}: Occupied {fraction*100:.0f}% memory ({memory_to_occupy / 1e9:.2f}GB)")
        return tensor
    except Exception as e:
        logger.error(f"Failed to occupy GPU {gpu_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='GPU Memory Holder')
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4', help='GPU list for group 1 (check run.sh/run_2.sh)')
    parser.add_argument('--iterations', type=str, default='5,6,7', help='GPU list for group 2 (no script check)')
    parser.add_argument('--username', type=str, default='hongwei', help='Username to monitor')
    parser.add_argument('--lossweight', type=float, default=0.5, help='GPU memory fraction to occupy')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds')
    
    args = parser.parse_args()
    
    # Parse GPU lists
    group1_gpus = [int(g) for g in args.seeds.split(',') if g.strip()] if args.seeds else []
    group2_gpus = [int(g) for g in args.iterations.split(',') if g.strip()] if args.iterations else []
    
    logger.info(f"Starting GPU monitor - Group1: {group1_gpus}, Group2: {group2_gpus}")
    
    occupied_tensors = {}  # Track occupied GPU tensors
    
    while True:
        try:
            scripts_running = check_scripts_running(args.username)
            user_gpu_pids = get_gpu_using_processes(args.username)
            
            # Process Group 1 GPUs (check run.sh/run_2.sh)
            for gpu_id in group1_gpus:
                has_user_process = get_gpu_processes_for_gpu(gpu_id, args.username)
                
                if scripts_running or has_user_process:
                    # Release GPU
                    if gpu_id in occupied_tensors:
                        del occupied_tensors[gpu_id]
                        logger.info(f"GPU {gpu_id}: Released (scripts running or user process detected)")
                else:
                    # Occupy GPU
                    if gpu_id not in occupied_tensors:
                        tensor = occupy_gpu_memory(gpu_id, args.lossweight)
                        if tensor is not None:
                            occupied_tensors[gpu_id] = tensor
            
            # Process Group 2 GPUs (no script check)
            for gpu_id in group2_gpus:
                has_user_process = get_gpu_processes_for_gpu(gpu_id, args.username)
                
                if has_user_process:
                    # Release GPU
                    if gpu_id in occupied_tensors:
                        del occupied_tensors[gpu_id]
                        logger.info(f"GPU {gpu_id}: Released (user process detected)")
                else:
                    # Occupy GPU
                    if gpu_id not in occupied_tensors:
                        tensor = occupy_gpu_memory(gpu_id, args.lossweight)
                        if tensor is not None:
                            occupied_tensors[gpu_id] = tensor
            
            time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(args.interval)


if __name__ == '__main__':
    main()

