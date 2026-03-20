import torch
from tqdm import tqdm
import numpy as np

def get_named_beta_schedule(schedule_name="linear", num_diffusion_timesteps=1000):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

class GibbsSampler:
    def __init__(self, Y, sigma, operator, sampler, model, device, 
                 N_MC=23, N_bi=20, rho=0.1, rho_decay_rate=0.8):
        self.N_MC = N_MC
        self.N_bi = N_bi
        self.rho = rho
        self.rho_decay_rate = rho_decay_rate
        self.device = device
        self.Y = Y
        self.sigma = sigma
        self.operator = operator
        self.sampler = sampler
        self.model = model

        # Get alphas using the beta schedule
        betas = get_named_beta_schedule('linear', 1000)
        self.alphas = np.cumsum(betas) / np.max(np.cumsum(betas))

        # Initialize matrices to store iterates
        self.X_MC = torch.zeros(size=(3, 256, 256, N_MC+1), device=self.device)
        self.Z_MC = torch.zeros(size=(3, 256, 256, N_MC+1), device=self.device)

        # Initialize X_MC and Z_MC with random values
        self.X_MC[:,:,:,0] = torch.randn((3, 256, 256), device=self.device)
        self.Z_MC[:,:,:,0] = torch.randn((3, 256, 256), device=self.device)

    def estimate_time(self, value, array=None):
        if array is None:
            array = self.alphas
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def compute_last_diff_step(self, t_start, t):
        if t < self.N_bi:
            t_stop = int(t_start * 0.7)
        else:
            t_stop = 0
        return t_stop

    def run(self):
        for t in tqdm(range(self.N_MC)):
            # Likelihood step
            self.X_MC[:,:,:,t+1] = self.operator.proximal_generator(
                self.Z_MC[:,:,:,t], self.Y, self.sigma, rho=self.rho
            )

            # Update rho and time step
            rho_iter = self.rho * (self.rho_decay_rate ** t)
            t_start = self.estimate_time(rho_iter)
            t_stop = self.compute_last_diff_step(t_start, t)

            # Prior step
            self.Z_MC[:,:,:,t+1] = self.sampler.diffuse_back(
                x=self.X_MC[:,:,:,t+1].unsqueeze(0), 
                model=self.model, 
                t_start=1000 - t_start, 
                t_end=1000 - t_stop
            ).squeeze(0)

        return self.X_MC, self.Z_MC
