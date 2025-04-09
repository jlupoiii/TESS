"""
BEAM: Background Elimination with Advanced Machine learning - Model Implementation

This script defines a conditional diffusion model for image training and generation.

The code is adapted from:
https://github.com/TeaPearce/Conditional_Diffusion_MNIST
which was originally modified from:
https://github.com/cloneofsimo/minDiffusion

Based on research from:
- DDPM: https://arxiv.org/abs/2006.11239
- Classifier-Free Diffusion Guidance: https://arxiv.org/abs/2207.12598
- ImageGen: https://arxiv.org/abs/2205.11487

"""

# Standard imports
from typing import Dict, Tuple, Optional, Union, List
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Deep learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset

# Image processing imports
from PIL import Image
import pickle
import torchvision.transforms as transforms
from astropy.io import fits

# For distributed training
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


class ResidualConvBlock(nn.Module):
    """
    Standard ResNet-style convolutional block with residual connections.
    """
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.is_res = is_res
        self.same_channels = in_channels == out_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_res:
            return self.conv2(self.conv1(x))
            
        # Handle residual connection
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        # Add correct residual connection
        if self.same_channels:
            out = x + x2
        else:
            out = x1 + x2
            
        # Normalize by sqrt(2) to maintain variance
        return out / 1.414


class UnetDown(nn.Module):
    """
    Downsampling block for U-Net architecture.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels), 
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetUp(nn.Module):
    """
    Upsampling block for U-Net architecture.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)  # Concatenate skip connection
        return self.model(x)


class EmbedFC(nn.Module):
    """
    Fully connected embedding network.
    """
    def __init__(self, input_dim: int, emb_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    """
    U-Net architecture with context embeddings for conditional diffusion.
    """
    def __init__(self, in_channels: int, in_dim: int, n_feat: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat

        # Initial convolution
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        
        # Downsampling path
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        
        # Bottleneck
        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.GELU())

        # Time and context embeddings
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(in_dim, 2 * n_feat)
        self.contextembed2 = EmbedFC(in_dim, n_feat)

        # Initial upsampling
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        
        # Upsampling path
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        
        # Final convolution
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Input image (noisy)
            c: Context conditioning
            t: Diffusion Timestep
            context_mask: Mask for classifier-free guidance
            
        Returns:
            Predicted noise
        """
        # Process input through initial convolution and downsample
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hidden = self.to_vec(down2)

        # Reshape context for processing
        c = c.reshape((c.shape[0], c.shape[2]))
        
        # Apply context mask for classifier-free guidance
        context_mask = context_mask.reshape((x.shape[0], c.shape[1]))
        context_mask = -1 * (1 - context_mask)  # Flip 0 <-> 1
        c = c * context_mask

        # Create embeddings for context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # Upsampling path with skip connections
        up1 = self.up0(hidden)
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # Add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        
        # Final convolution with skip connection to initial input
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Compute pre-defined schedules for DDPM training and sampling.
    
    Args:
        beta1: Starting beta value
        beta2: Ending beta value
        T: Number of diffusion steps
        
    Returns:
        Dictionary of DDPM schedules
    """
    assert 0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # Linear noise schedule
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    
    # Alpha values
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    # Other useful values for DDPM
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model implementation.
    """
    def __init__(
        self, 
        nn_model: nn.Module, 
        betas: Tuple[float, float], 
        n_T: int, 
        device: torch.device, 
        drop_prob: float = 0.1
    ):
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

        # Register DDPM schedule buffers
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            x: Input images
            c: Conditioning information
            
        Returns:
            MSE loss between predicted and actual noise
        """
        # Sample timesteps uniformly
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)
        
        # Sample noise
        noise = torch.randn_like(x)

        # Ensure values are on correct device
        self.sqrtab = self.sqrtab.to(self.device)
        self.sqrtmab = self.sqrtmab.to(self.device)

        # Create noisy samples at timestep t
        x_t = (
            self.sqrtab[_ts, None].reshape((x.shape[0], 1, 1, 1)) * x
            + self.sqrtmab[_ts, None].reshape((x.shape[0], 1, 1, 1)) * noise
        )
        
        # Context dropout for classifier-free guidance
        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob)
        
        # Compute loss between predicted and actual noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(
        self, 
        n_sample: int, 
        size: Tuple[int, ...], 
        device: torch.device, 
        guide_w: float = 0.0
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Sample from the model using classifier-free guidance.
        
        Args:
            n_sample: Number of samples to generate
            size: Size of the samples
            device: Device to generate on
            guide_w: Guidance scale (0 = no guidance)
            
        Returns:
            Tuple of (final samples, intermediate samples)
        """
        # Start with random noise
        x_i = torch.randn(n_sample, *size).to(device)  
        
        # Random context parameters
        c_i = torch.rand((n_sample, 1, 12)).to(device)
        
        # No context dropout during sampling
        context_mask = torch.zeros_like(c_i).to(device)

        # Double batch for classifier-free guidance
        c_i = c_i.repeat(2, 1, 1)
        context_mask = context_mask.repeat(2, 1, 1)
        context_mask[n_sample:] = 1.  # Second half with no context

        x_i_store = []  # Store intermediate generations
        
        print()
        # Reverse diffusion process
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # Double batch for guidance
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            # Sample random noise (only if not final step)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # Predict noise and apply classifier-free guidance
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]  # Conditioned
            eps2 = eps[n_sample:]  # Unconditional
            
            # Apply guidance
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            
            # Get only first half of doubled batch
            x_i = x_i[:n_sample]
            
            # Update sample with predicted noise
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            
            # Store intermediate samples periodically
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

    def sample_c(
        self, 
        c_i: torch.Tensor, 
        n_sample: int, 
        size: Tuple[int, ...], 
        device: torch.device
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Sample from the model with specific conditioning.
        
        Args:
            c_i: Conditioning information
            n_sample: Number of samples per conditioning vector
            size: Size of samples
            device: Device to generate on
            
        Returns:
            Tuple of (final samples, intermediate samples)
        """
        n_datapoint = c_i.shape[0]

        # Start with random noise
        x_i = torch.randn(n_datapoint * n_sample, *size).to(device)
        
        # Repeat each conditioning vector n_sample times
        c_i = torch.cat([c_i[idx:idx+1].repeat(n_sample, 1, 1) for idx in range(n_datapoint)]).to(device)
        
        # No context dropout during sampling
        context_mask = torch.zeros_like(c_i).to(device)

        x_i_store = []  # Store intermediate generations
        
        print()
        # Reverse diffusion process
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_datapoint * n_sample, 1, 1, 1)

            # Sample random noise (only if not final step)
            z = torch.randn(n_datapoint * n_sample, *size).to(device) if i > 1 else 0

            # Predict noise
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            
            # Update sample with predicted noise
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            
            # Store intermediate samples periodically
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


class TESS_ImageDataset:
    """
    Base class for TESS image datasets.
    """
    def __len__(self) -> int:
        raise NotImplementedError("Subclasses must implement __len__")
    
    def __getitem__(self, idx) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement __getitem__")


class TESS_4096_original_images(TESS_ImageDataset):
    """
    Dataset for raw original 4096x4096 TESS images.
    """
    def __init__(self):
        self.ffi_to_fits_filepath = {}
        
        # Build index of fits files
        fits_folder_paths = [f"/pdo/users/roland/SL_data/O{i}_data/" for i in range(9, 63)]

        for fits_folder_path in fits_folder_paths:
            orbit = fits_folder_path[27:29] if fits_folder_path[29] == '_' else fits_folder_path[27:28]
            for fits_filename in os.listdir(fits_folder_path):
                # Only include camera 3 FITS files
                if (len(fits_filename) > 40 and 
                    fits_filename[-7:] == 'fits.gz' and 
                    fits_filename[27] == '3'):
                    ffi_num = fits_filename[18:26]
                    self.ffi_to_fits_filepath[ffi_num] = os.path.join(fits_folder_path, fits_filename)

    def __len__(self) -> int:
        return len(self.ffi_to_fits_filepath)

    def __getitem__(self, ffi_num: str) -> torch.Tensor:
        # Rows and columns to remove from the image (black areas)
        rows_to_delete = range(2048, 2108)
        columns_to_delete = list(range(0, 44)) + list(range(2092, 2180)) + list(range(4228, 4272))
        
        # Load and process image
        image = fits.getdata(self.ffi_to_fits_filepath[ffi_num], ext=0)
        image = np.delete(image, rows_to_delete, axis=0)
        image = np.delete(image, columns_to_delete, axis=1)
        image = image.astype(np.float64)
        image *= 1/633118  # Scale values to reasonable range
        
        return torch.tensor(image)


class TESS_4096_processed_images(TESS_ImageDataset):
    """
    Dataset for pre-processed 4096x4096 TESS images.
    """
    def __init__(self):
        self.ffi_to_pkl_filepath = {}
        
        pkl_folder_path = "/pdo/users/jlupoiii/TESS/data/processed_images_im4096x4096/"
        for pkl_filename in os.listdir(pkl_folder_path):
            ffi_num = pkl_filename[18:26]
            self.ffi_to_pkl_filepath[ffi_num] = os.path.join(pkl_folder_path, pkl_filename)

    def __len__(self) -> int:
        return len(self.ffi_to_pkl_filepath)

    def __getitem__(self, ffi_num: str) -> torch.Tensor:
        with open(self.ffi_to_pkl_filepath[ffi_num], 'rb') as file:
            return pickle.load(file)