"""
Utilities for distributed training.

This module contains functions to set up and manage distributed training
across multiple GPUs.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict, Callable, Any, Tuple


def setup_distributed(rank: int, world_size: int) -> None:
    """
    Setup for distributed training.
    
    Args:
        rank: Index of the current GPU
        world_size: Total number of GPUs
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    """
    Clean up distributed training resources.
    """
    dist.destroy_process_group()


def run_distributed(
    fn: Callable, 
    world_size: int, 
    args: Tuple = (), 
    kwargs: Dict = None
) -> None:
    """
    Run a function in a distributed manner across multiple GPUs.
    
    Args:
        fn: Function to run
        world_size: Number of GPUs to use
        args: Positional arguments to pass to fn
        kwargs: Keyword arguments to pass to fn
    """
    if kwargs is None:
        kwargs = {}
    
    if world_size > 1:
        mp.spawn(
            _distributed_worker,
            args=(fn, world_size, args, kwargs),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU mode - run directly
        fn(0, world_size, *args, **kwargs)


def _distributed_worker(
    rank: int, 
    fn: Callable, 
    world_size: int, 
    args: Tuple, 
    kwargs: Dict
) -> None:
    """
    Worker function that runs on each GPU.
    
    Args:
        rank: Index of the current GPU
        fn: Function to run
        world_size: Total number of GPUs
        args: Positional arguments to pass to fn
        kwargs: Keyword arguments to pass to fn
    """
    # Set up the process group
    setup_distributed(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    try:
        # Run the function
        fn(rank, world_size, *args, **{**kwargs, 'device': device})
    finally:
        # Clean up
        cleanup_distributed()


def all_reduce_dict(input_dict: Dict, device: torch.device) -> Dict:
    """
    Perform all-reduce operation on a dictionary of tensors.
    
    Args:
        input_dict: Dictionary of tensors
        device: Device where tensors reside
        
    Returns:
        Dictionary with reduced values
    """
    result_dict = {}
    
    for k, v in input_dict.items():
        tensor = v.clone().to(device)
        dist.all_reduce(tensor)
        result_dict[k] = tensor / dist.get_world_size()
    
    return result_dict


def broadcast_value(value: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast a value from one process to all others.
    
    Args:
        value: Value to broadcast
        src: Source process rank
        
    Returns:
        Broadcasted value
    """
    dist.broadcast(value, src=src)
    return value