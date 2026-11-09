
"""
Defining ODE and SDE classes and their corresponding simulators.
"""


import torch
import torch.nn.functional as F
from beam.models.unet import ContextUnet
from beam.models.probabilitypath import GaussianProbabilityPath, VPBeta, OTAlpha
from tqdm import tqdm
from typing import Optional, Callable
from abc import ABC, abstractmethod
from torchdiffeq import odeint


# Helper functions for converting between flow and score models for gaussian paths

def noise2score(noise_model: ContextUnet, path: GaussianProbabilityPath) -> Callable:
    """
    Args:
        noise_model: ContextUnet
        path: GaussianProbabilityPath
    Returns:
        score: Callable
    Given a score model trained to output noise, return the score
    """
    def score(x: torch.Tensor, c: Optional[torch.Tensor], t: torch.Tensor, context_mask: torch.Tensor):
        b_t = path.beta(t).clamp_min(1e-5)
        return -noise_model(x, c, t, context_mask)/b_t
    return score

def flow2score(flow_model: ContextUnet, path: GaussianProbabilityPath) -> Callable:
    """
    Args:
        flow_model: ContextUnet
        guidance_value: float
        alpha: Alpha
        beta: Beta
    Returns:
        score: Callable
    Given a flow model trained to learn a gaussian path, the model can be rewritten using the schedulers to output a score model.
    """
    assert isinstance(path, GaussianProbabilityPath), "Only GaussianProbabilityPath is supported"
    alpha = path.alpha
    beta = path.beta

    def score(x: torch.Tensor, c: Optional[torch.Tensor], t: torch.Tensor, context_mask: torch.Tensor):
        alpha_t = alpha(t)
        beta_t = beta(t)
        dt_alpha_t = alpha.derivative(t)
        dt_beta_t = beta.derivative(t)


        num = alpha_t * flow_model(x, c, t, context_mask) - dt_alpha_t * x
        den = beta_t ** 2 * dt_alpha_t - alpha_t * dt_beta_t * beta_t

        return num / den
    return score

def score2flow(score_model: ContextUnet, path: GaussianProbabilityPath) -> Callable:
    """
    Args:
        score_model: ContextUnet
        guidance_value: float
        alpha: Alpha
        beta: Beta  
    Returns:
        flow: Callable
    Given a score model trained to learn a gaussian path, the model can be rewritten using the schedulers to output a flow model.
    """
    assert isinstance(path, GaussianProbabilityPath), "Only GaussianProbabilityPath is supported"
    alpha = path.alpha
    beta = path.beta
    @torch.no_grad()
    def flow(x: torch.Tensor, c: Optional[torch.Tensor], t: torch.Tensor, context_mask: torch.Tensor):
        score = score_model(x, c, t, context_mask)
        if isinstance(beta, VPBeta):

            # Clamp t to avoid division by zero
            t_safe = t.clamp(1e-3, 1.0 - 1e-3)
            return 1/t_safe*(score + x)
        else:
            t_safe = t.clamp(1e-3, 1.0 - 1e-3)
            alpha_t = alpha(t_safe)
            da = alpha.derivative(t_safe)
            db = beta.derivative(t_safe)
            beta_t = beta(t_safe)
            return (beta_t**2 * da/alpha_t - db*beta_t) * score + da/alpha_t*x
    return flow

def noise2flow(noise_model: ContextUnet, path: GaussianProbabilityPath) -> Callable:
    """
    Args:
        noise_model: ContextUnet trained to predict epsilon (the standard-normal noise x0)
        path: GaussianProbabilityPath
    Returns:
        flow: Callable returning the probability-flow velocity field
    """
    assert isinstance(path, GaussianProbabilityPath), "Only GaussianProbabilityPath is supported"
    alpha = path.alpha
    beta = path.beta
    @torch.no_grad()
    def flow(x: torch.Tensor, c: Optional[torch.Tensor], t: torch.Tensor, context_mask: torch.Tensor):
        eps_hat = noise_model(x, c, t, context_mask)
        t_safe = t.clamp(1e-3, 1.0 - 1e-3)
        a = alpha(t_safe).clamp_min(1e-12)
        b = beta(t_safe).clamp_min(1e-12)
        da = alpha.derivative(t_safe)
        db = beta.derivative(t_safe)
        return (da / a) * x - (da * b / a - db) * eps_hat
    return flow

def v2flow(v_model: ContextUnet, path: GaussianProbabilityPath) -> Callable:
    """
    Args:
        v_model: ContextUnet trained to predict v = a eps - b x1
        path: GaussianProbabilityPath
    Returns:
        flow: Callable returning the probability-flow velocity field

    """
    assert isinstance(path, GaussianProbabilityPath), "Only GaussianProbabilityPath is supported"
    alpha = path.alpha
    beta = path.beta
    @torch.no_grad()
    def flow(x: torch.Tensor, c: Optional[torch.Tensor], t: torch.Tensor, context_mask: torch.Tensor):
        v = v_model(x, c, t, context_mask)
        t_safe = t.clamp(1e-3, 1.0 - 1e-3)
        a = alpha(t_safe)
        b = beta(t_safe)
        da = alpha.derivative(t_safe)
        db = beta.derivative(t_safe)
        denom = (a ** 2 + b ** 2).clamp_min(1e-12)
        return ((da * a + db * b) * x + (db * a - da * b) * v) / denom
    return flow

def guided_output(
    nn_model: ContextUnet,
    x: torch.Tensor,
    c: torch.Tensor,
    t: torch.Tensor,
    guidance_value: float,
    cam: Optional[torch.Tensor] = None,
):
    """
    One classifier-free-guided evaluation of the network.
    """
    context_mask = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
    c_in = torch.cat([c, c], dim=0)
    x_in = torch.cat([x, x], dim=0)
    t_in = torch.cat([t, t], dim=0)

    if cam is None:
        out = nn_model(x_in, c_in, t_in, context_mask)
    else:
        out = nn_model(x_in, c_in, t_in, context_mask, torch.cat([cam, cam], dim=0))

    conditional, unconditional = out.chunk(2, dim=0)
    return (1 - guidance_value) * unconditional + guidance_value * conditional

class ODE():
    """
    Wrapper for U-net to be used as an ODE which can be solved using odeint.
    """
    def __init__(self, nn_model: ContextUnet, guidance_value: float,
                 cam: Optional[torch.Tensor] = None):
        self.nn_model = nn_model
        self.guidance_value = guidance_value
        self.cam = cam

    def velocity_field(self,x: torch.Tensor, c: Optional[torch.Tensor], t: torch.Tensor,):
        """
        Args:
            x: torch.Tensor
            c: torch.Tensor
            t: torch.Tensor
        Returns:
            gfc_drift: torch.Tensor
        Returns the drift of the ODE according to classifier-free guidance
        """
        # Clamp time to avoid numerical issues
        t_clamped = t.clamp(1e-3, 1.0 - 1e-3)
        return guided_output(
            self.nn_model, x, c, t_clamped, self.guidance_value, self.cam
        )

class SDE():
    """
    Reverse time generative SDE for a Gaussian probability path.
    """
    def __init__(self, nn_model: ContextUnet, probability_path: GaussianProbabilityPath,
                 guidance_value: float, sigma=None, architecture: str = "v",
                 cam: Optional[torch.Tensor] = None):
        self.nn_model = nn_model
        self.path = probability_path
        self.alpha = probability_path.alpha
        self.beta = probability_path.beta
        self.guidance_value = guidance_value
        # Default diffusion coefficient g(t) = beta(t)
        self.sigma = sigma if sigma is not None else probability_path.beta
        self.architecture = architecture
        self.cam = cam

    def _predict(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single classifier free guided network evaluation."""
        return guided_output(self.nn_model, x, c, t, self.guidance_value, self.cam)

    def _x1_and_eps(self, out: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        """
        Convert a guided network output at (x, t) into the implied clean image
        x1_hat and noise eps_hat, using the parametrization the model was
        trained with.
        """
        alpha = self.alpha(t)
        beta = self.beta(t)
        arch = self.architecture

        if arch == "v":
            # v = alpha*eps - beta*x1; inverting the 2x2 system with x_t needs
            # no division by alpha or beta, so this stays stable at both ends.
            norm = (alpha**2 + beta**2).clamp_min(1e-12)
            x1 = (alpha * x - beta * out) / norm
            eps = (beta * x + alpha * out) / norm
            return x1, eps

        if arch == "flow":
            # out is the velocity itself: alpha_dot*x1 + beta_dot*eps.
            da = self.alpha.derivative(t)
            db = self.beta.derivative(t)
            det = alpha * db - da * beta
            det = torch.where(det.abs() < 1e-12, torch.full_like(det, -1e-12), det)
            eps = (alpha * out - da * x) / det
            x1 = (beta * out - db * x) / (-det)
            return x1, eps

        if arch == "noise":
            eps = out
        else:  # 'score': out = grad_x log p_t(x) = -eps / beta
            eps = -beta.clamp_min(1e-6) * out

    
        x1 = (x - beta * eps) / alpha.clamp_min(1e-6)
        return x1, eps

    def _velocity_and_score(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
        """Convert the network output to (velocity field u, score s)."""
        t0 = t.clamp(1e-3, 1.0 - 1e-3)
        a = self.alpha(t0)
        b = self.beta(t0).clamp_min(1e-12)
        da = self.alpha.derivative(t0)
        db = self.beta.derivative(t0)
        out = self._predict(x, c, t0)

        if self.architecture == "noise":
            eps_hat = out
            score = -eps_hat / b
            velocity = (da / a) * x - (da * b / a - db) * eps_hat
        elif self.architecture == "score":
            score = out
            velocity = (da / a) * x + (b ** 2 * da / a - db * b) * score
        elif self.architecture == "v":
            denom = (a ** 2 + b ** 2).clamp_min(1e-12)
            eps_hat = (b * x + a * out) / denom
            score = -eps_hat / b
            velocity = ((da * a + db * b) * x + (db * a - da * b) * out) / denom
        else:
            raise ValueError(
                f"SDE architecture {self.architecture} must be one of 'noise', 'score', 'v'"
            )
        return velocity, score

    def drift(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Generative-SDE drift: u + 0.5 g(t)^2 s, with classifier-free guidance."""
        sigma = self.sigma if isinstance(self.sigma, float) else self.sigma(t)
        velocity, score = self._velocity_and_score(x, c, t)
        return velocity + 0.5 * (sigma ** 2) * score

    def velocity_field(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Probability-flow velocity field component of the drift."""
        velocity, _ = self._velocity_and_score(x, c, t)
        return velocity

    def score(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score component of the drift."""
        _, score = self._velocity_and_score(x, c, t)
        return score

    def diffusion_coeff(self, x, t):
        sigma = self.sigma if isinstance(self.sigma, float) else self.sigma(t)
        return sigma * torch.ones_like(x)

    

class Simulator(ABC):
    """
    Abstract class for all simulators.
    """

    @abstractmethod
    @torch.no_grad()
    def simulate(self, *args, **kwargs):
        pass

class Euler(Simulator):
    """
    Basic Integrator used for debugging. All final ODE generations should come from ODEIntegrator.
    """
    def __init__(self, ode: ODE, t: torch.Tensor):
        self.ode = ode
        self.num_steps = t.shape[1]
        self.t = t  #shape (1, num_steps)

    @torch.no_grad()
    def simulate(self, x0: torch.Tensor, c: torch.Tensor):
        ts = self.t
        x = x0.clone()
        for k in tqdm(range(self.num_steps - 1)):
            t = ts[:, k]
            h  = ts[:, k+1] - ts[:, k]
            h = h.view(-1, 1, 1, 1)
            x = x + self.ode.velocity_field(x, c, t[..., None, None, None]) * h
        return x

class ODEIntegrator(Simulator):
    """
    Wrapper for odeint, found at https://github.com/rtqichen/torchdiffeq, to make compatible with defined ODE class
    Odeint returns a full trajectory
    """
    def __init__(self, ode: ODE, t: torch.Tensor, method: str = "dopri5", atol: float = 1e-5, rtol: float = 1e-5, num_save: int = 1):
        self.ode = ode
        self.num_steps = t.shape[1]
        self.t = t  #shape (1, num_steps)   
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.num_save = num_save
    @torch.no_grad()    
    def simulate(self, x0: torch.Tensor, c: torch.Tensor):
        ts = self.t[0] #odeint expects timem as 1d tensor
        bs = x0.shape[0]
        x = x0.clone()
        def drift_helper(t, x):
            return self.ode.velocity_field(x, c, t.expand(bs, 1, 1, 1)) 
        x = odeint(drift_helper, x, ts, method=self.method, atol=self.atol, rtol=self.rtol, )
        if self.num_save == 1:
            ret_idx = torch.tensor([self.num_steps - 1], device=x.device, dtype=torch.long)
        else:
            ret_idx = torch.linspace(0, self.num_steps-1, steps = self.num_save, device=x.device, dtype = torch.long)
        return x[ret_idx, ...], ts[ret_idx]
    
class EulerMaruyama(Simulator):
    """
    Stochastic Euler Integrator for SDEs.
    """
    def __init__(self, sde: SDE, t: torch.Tensor, num_save: int = 1, num_corrections: int=0, corr_step_size: float=0.3):
        self.sde = sde
        self.num_steps = t.shape[1]
        self.t = t   #shape (1, num_steps)
        self.num_save = num_save

    @torch.no_grad()
    def simulate(self, x0: torch.Tensor, c: torch.Tensor):
        B, C, H, W = x0.shape
        ts = self.t.expand(B, -1)
        x = x0.clone()
        if self.num_save == 1:
            ret_idx = torch.tensor([self.num_steps - 1], device=x.device, dtype=torch.long)
        else:
            ret_idx = torch.linspace(0, self.num_steps-1, steps = self.num_save, device=x.device, dtype = torch.long)
        ret_idx.clamp_(0, self.num_steps-1)
        xs_saved = torch.empty(
        self.num_save, B, C, H, W,
        device=x.device,
        dtype=x.dtype, 
        )
        sv_ptr = 0
        next_save_idx = ret_idx[sv_ptr]
        for k in tqdm(range(self.num_steps - 1)): 
            t = ts[:, k]
            if k == next_save_idx:
                xs_saved[sv_ptr] = x
                sv_ptr += 1
                if sv_ptr < self.num_save:
                    next_save_idx = ret_idx[sv_ptr].item()
            #update x
            h  = ts[:, k+1] - ts[:, k]
            h = h.view(-1, 1, 1, 1)
            x = x + self.sde.drift(x, c, t[..., None, None, None]) * h + self.sde.diffusion_coeff(x, t[..., None, None, None]) * torch.randn_like(x) * torch.sqrt(h)

        if (self.num_steps - 1) == next_save_idx:
            xs_saved[-1] = x
        return xs_saved, self.t[0, ret_idx]

        
