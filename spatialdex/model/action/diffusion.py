# Modified from OpenAI improved-diffusion: https://github.com/openai/improved-diffusion (MIT License)

import math

import numpy as np
import torch


def cosine_betas(num_steps, max_beta=0.999):
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    return np.array(
        [min(1 - alpha_bar((i + 1) / num_steps) / alpha_bar(i / num_steps), max_beta) for i in range(num_steps)]
    )


def extract(values, t, shape):
    out = torch.from_numpy(values).to(device=t.device)[t].float()
    while len(out.shape) < len(shape):
        out = out[..., None]
    return out + torch.zeros(shape, device=t.device)


def spaced_alphas_cumprod(kept, train_steps=100):
    base = np.cumprod(1.0 - np.array(cosine_betas(train_steps), dtype=np.float64), axis=0)
    betas, timestep_map, last = [], [], 1.0
    for i, alpha_cumprod in enumerate(base):
        if i in kept:
            betas.append(1 - alpha_cumprod / last)
            last = alpha_cumprod
            timestep_map.append(i)
    return np.cumprod(1.0 - np.array(betas, dtype=np.float64), axis=0), timestep_map


class TrainingDiffusion:
    def __init__(self, train_steps=100):
        self.num_timesteps = train_steps
        alphas_cumprod, _ = spaced_alphas_cumprod(set(range(train_steps)), train_steps)
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


class DDIMSampler:
    def __init__(self, num_steps, train_steps=100):
        stride = next(i for i in range(1, train_steps) if len(range(0, train_steps, i)) == num_steps)
        self.alphas_cumprod, self.timestep_map = spaced_alphas_cumprod(set(range(0, train_steps, stride)), train_steps)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

    @torch.no_grad()
    def sample(self, model, noise, **model_kwargs):
        x = noise
        for i in reversed(range(len(self.timestep_map))):
            t = torch.tensor([i] * x.shape[0], device=x.device)
            timesteps = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)[t]
            eps = model(x, timesteps, **model_kwargs)
            recip = extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
            recipm1 = extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
            pred_xstart = recip * x - recipm1 * eps
            eps = (extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - pred_xstart) / extract(
                self.sqrt_recipm1_alphas_cumprod, t, x.shape
            )
            alpha_bar = extract(self.alphas_cumprod, t, x.shape)
            alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
            sigma = (
                0.0 * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            noise_t = torch.randn_like(x)
            mean = pred_xstart * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
            nonzero = (t != 0).float().view(-1, *[1] * (len(x.shape) - 1))
            x = mean + nonzero * sigma * noise_t
        return x
