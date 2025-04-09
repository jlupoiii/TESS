"""
Visualization utilities for TESS images and model outputs.

This module contains functions for visualizing TESS images, model predictions,
and training metrics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader

from beam.models.diffusion import DDPM


def plot_samples(
    model: DDPM,
    dataloader: DataLoader,
    device: torch.device,
    n_sample: int = 3,
    n_datapoint: int = 3,
    image_shape: Tuple[int, int] = (64, 64),
    title: str = "Model Samples",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate and plot samples from the model compared to real data.
    
    Args:
        model: The diffusion model
        dataloader: DataLoader containing real data
        device: Device to generate on
        n_sample: Number of samples per conditioning vector
        n_datapoint: Number of datapoints to visualize
        image_shape: Shape of the images
        title: Plot title
        save_path: Path to save the figure (None = don't save)
        
    Returns:
        Matplotlib Figure object
    """
    # Get a batch of data
    data_batch = next(iter(dataloader))
    
    # Extract data (limited to n_datapoint)
    x_real = data_batch['y'][:n_datapoint].to(device)
    c_real = data_batch['x'][:n_datapoint].to(device)
    ffi_nums = data_batch['ffi_num'][:n_datapoint]
    orbits = data_batch['orbit'][:n_datapoint]
    
    # Generate samples
    with torch.no_grad():
        x_gen, x_gen_store = model.sample_c(
            c_real, 
            n_sample, 
            (1, image_shape[0], image_shape[1]), 
            device
        )
    
    # Create figure with original images and generated samples
    fig, axes = plt.subplots(n_datapoint, n_sample + 1, figsize=(15, 3 * n_datapoint))
    plt.subplots_adjust(top=0.9)
  
    # Handle case when n_datapoint = 1
    if n_datapoint == 1:
        axes = [axes]
    
    # Combine originals and generated samples for display
    for i in range(n_datapoint):
        # Display original
        axes[i][0].imshow(
            x_real[i][0].cpu().detach().numpy(), 
            cmap='gray', 
            vmin=0, 
            vmax=1
        )
        axes[i][0].axis('off')
        
        # Row title with FFI info
        axes[i][0].set_title(
            f"Orbit {orbits[i]}, FFI {ffi_nums[i]}", 
            fontsize=10, 
            loc='left'
        )
        
        # Display generated samples
        for j in range(n_sample):
            axes[i][j+1].imshow(
                x_gen[i*n_sample + j][0].cpu().detach().numpy(), 
                cmap='gray', 
                vmin=0, 
                vmax=1
            )
            axes[i][j+1].axis('off')
    
    # Set column titles
    axes[0][0].set_title("Original", fontsize=12)
    for j in range(n_sample):
        axes[0][j+1].set_title(f"Sample {j+1}", fontsize=12)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Save figure if requested
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path)
        print(f'Saved samples to {save_path}')
    plt.close(fig)

    return fig


def plot_loss_history(
    train_losses: List[float],
    valid_losses: List[float],
    save_dir: Optional[str] = None,
    show_last_n: Optional[int] = None
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plot training and validation loss history.
    
    Args:
        train_losses: List of training losses
        valid_losses: List of validation losses
        save_dir: Directory to save plots (None = don't save)
        show_last_n: Only plot the last N epochs (None = all epochs)
        
    Returns:
        Tuple of (full history figure, last N figure)
    """
    fig_full = plt.figure(figsize=(10, 6))
    plt.plot(valid_losses, label="Validation Loss")
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save full history plot if requested
    if save_dir:
        fig_full.savefig(os.path.join(save_dir, 'loss_graph_full.png'))
    
    # Create last N epochs plot if requested
    if show_last_n and show_last_n < len(train_losses):
        fig_last_n = plt.figure(figsize=(10, 6))
        plt.plot(
            range(len(valid_losses) - show_last_n, len(valid_losses)), 
            valid_losses[-show_last_n:], 
            label="Validation Loss"
        )
        plt.plot(
            range(len(train_losses) - show_last_n, len(train_losses)), 
            train_losses[-show_last_n:], 
            label="Training Loss"
        )
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'Training and Validation MSE Loss (Last {show_last_n} Epochs)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_dir:
            fig_last_n.savefig(os.path.join(save_dir, f'loss_graph_last{show_last_n}.png'))
    else:
        fig_last_n = None
    
    return fig_full, fig_last_n


def plot_generation_process(
    x_gen_store: np.ndarray,
    sample_idx: int = 0,
    num_timesteps: int = 8,
    title: str = "Diffusion Generation Process",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the generation process for a single sample across multiple timesteps.
    
    Args:
        x_gen_store: Array of intermediate generations (timesteps × batch × channels × height × width)
        sample_idx: Index of the sample to visualize
        num_timesteps: Number of timesteps to display
        title: Plot title
        save_path: Path to save the figure (None = don't save)
        
    Returns:
        Matplotlib Figure object
    """
    # Select timesteps to display
    if len(x_gen_store) <= num_timesteps:
        timesteps = range(len(x_gen_store))
    else:
        # Evenly spaced timesteps
        timesteps = np.linspace(0, len(x_gen_store) - 1, num_timesteps, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(1, len(timesteps), figsize=(2*len(timesteps), 3))
    
    # Display each timestep
    for i, t in enumerate(timesteps):
        axes[i].imshow(
            x_gen_store[t, sample_idx, 0], 
            cmap='gray', 
            vmin=0, 
            vmax=1
        )
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title("Start (noise)", fontsize=10)
        elif i == len(timesteps) - 1:
            axes[i].set_title("Final", fontsize=10)
        else:
            axes[i].set_title(f"Step {t}", fontsize=10)
    
    # Set overall title
    fig.suptitle(title, fontsize=14)
    
    # Save figure if requested
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path)
        print(f'Saved generation process to {save_path}')
    
    return fig
