"""
Diffusion model implementation.

This module contains the implementation of the Denoising Diffusion Probabilistic Model (DDPM).
"""

from typing import Dict, Tuple, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn


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
