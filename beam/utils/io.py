"""
I/O utilities for handling files and data storage.

This module provides functions for loading and saving various data formats
used in the BEAM project.
"""

import os
import pickle
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data to a pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save the file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int =0,
    train_loss: Optional[List[float]] = None,
    valid_loss: Optional[List[float]] = None,
    additional_data: Optional[Dict] = None,
    filepath: str = "checkpoint.pth"
) -> None:
    """
    Save a model checkpoint.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        train_loss: Training loss history
        valid_loss: Validation loss history
        additional_data: Any additional data to save
        filepath: Path to save the checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Get model state dict, handling DistributedDataParallel
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
    }
    
    # Add optional components
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if train_loss is not None:
        checkpoint['train_loss'] = train_loss
    if valid_loss is not None:
        checkpoint['valid_loss'] = valid_loss
    if additional_data is not None:
        checkpoint.update(additional_data)
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load a model checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into
        device: Device to load model on
        
    Returns:
        Dictionary containing the checkpoint data
    """
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state if provided
    if model is not None:
        # Handle DistributedDataParallel
        target_model = model.module if hasattr(model, 'module') else model
        target_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def save_generated_images(
    images: torch.Tensor,
    filepath_pattern: str,
    start_idx: int = 0
) -> List[str]:
    """
    Save generated images.
    
    Args:
        images: Tensor of images [N, C, H, W]
        filepath_pattern: Pattern for filenames (e.g., "sample_{}.png")
        start_idx: Starting index for filenames
        
    Returns:
        List of saved file paths
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(os.path.dirname(filepath_pattern), exist_ok=True)
    
    paths = []
    for i, img in enumerate(images):
        # Convert to numpy and reshape if needed
        if img.dim() == 3:  # [C, H, W]
            img_np = img[0].cpu().detach().numpy()  # Take first channel
        else:  # [H, W]
            img_np = img.cpu().detach().numpy()
        
        # Create filename
        filepath = filepath_pattern.format(i + start_idx)
        
        # Save image
        plt.figure(figsize=(5, 5))
        plt.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        paths.append(filepath)
    
    return paths


def save_loss_history(
    train_loss: List[float],
    valid_loss: List[float],
    filepath: str,
    time_history: Optional[List[float]] = None
) -> None:
    """
    Save loss history to a text file.
    
    Args:
        train_loss: Training loss history
        valid_loss: Validation loss history
        filepath: Path to save the file
        time_history: Optional time elapsed per epoch
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        if time_history is not None:
            f.write("Epoch\tTraining Loss\tValidation Loss\tTime (hrs)\n")
            for i in range(len(train_loss)):
                f.write(f"{i}\t{train_loss[i]:.6e}\t{valid_loss[i]:.6e}\t{time_history[i]:.3f}\n")
        else:
            f.write("Epoch\tTraining Loss\tValidation Loss\n")
            for i in range(len(train_loss)):
                f.write(f"{i}\t{train_loss[i]:.6e}\t{valid_loss[i]:.6e}\n")
