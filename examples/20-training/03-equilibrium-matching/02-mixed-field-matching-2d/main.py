"""Mixed-field matching in 2D: mode proportions with and without the time input.

Mixed-field matching (Ghaderi et al., 2026) trains one field f(x, tau) on the
time input tau = min(t, a): a flow-matching transport field below the
threshold a, and one time-independent refinement field above it. Equilibrium
matching (Wang and Du, 2025) regresses the same truncated target with a
constant time input. As in the paper's two-dimensional example (Sec. 4.5,
App. D): five points on a regular pentagon of radius 0.5 with probabilities
0.50, 0.20, 0.15, 0.10 and 0.05, a = 0.8, multiplier 4, and an endpoint
penalty of weight 0.1 for both models, which differ only in the time input.
Each model is an MLP with three hidden layers of 128 units on the point and
five time features, trained with Adam at learning rate 1e-3 and batch 1024 for
20,000 updates. The endpoint penalty is in-batch, with the share of clean
rows of the ImageNet models, 4 in every 64, so 64 of the 1024 rows are clean
examples; the paper's two-dimensional runs use a separate batch. Mixed-field
matching samples with 128 Heun transport steps and 1,024 refinement steps at
rate 1/4; equilibrium matching takes 1,024 steps of the same size,
(1 - a) / 16.

Samples are scored by the total variation distance between the fraction of
samples nearest to each point and its probability, and by the root-mean-square
distance to the nearest point.
"""

import math
import os

import torch
from torch import nn

from torchebm.interpolants import LinearInterpolant
from torchebm.losses import MixedFieldMatchingLoss
from torchebm.samplers import GradientDescentSampler, MixedFieldSampler

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 20000
BATCH_SIZE = 1024
N_CLEAN = 64
N_SAMPLES = 512 if SMOKE else 16384
N_TRANSPORT_STEPS = 8 if SMOKE else 128
N_REFINEMENT_STEPS = 16 if SMOKE else 1024
THRESHOLD = 0.8

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
angles = torch.arange(5, device=device) * (2 * math.pi / 5)
points = 0.5 * torch.stack([angles.cos(), angles.sin()], dim=1)
probabilities = torch.tensor([0.50, 0.20, 0.15, 0.10, 0.05], device=device)


def pentagon(n):
    return points[torch.multinomial(probabilities, n, replacement=True)]


class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t=None):
        t = x.new_zeros(x.shape[0]) if t is None else t
        w = math.pi * t[:, None]
        return self.net(
            torch.cat(
                [x, t[:, None], w.sin(), w.cos(), (2 * w).sin(), (2 * w).cos()], dim=1
            )
        )


def train(time_input):
    torch.manual_seed(0)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = LinearInterpolant()
    loss_fn = MixedFieldMatchingLoss(
        ct_threshold=THRESHOLD, endpoint_penalty="in_batch", n_clean=N_CLEAN
    )
    for _ in range(N_STEPS):
        x1 = pentagon(BATCH_SIZE)
        t = loss_fn.sample_times(BATCH_SIZE, device=device)
        x_t, u_t = path(torch.randn_like(x1), x1, t)
        loss = loss_fn(model(x_t, time_input(t)), u_t, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def scores(x):
    """Total variation distance of the mode proportions and RMS distance to the nearest point."""
    distance, nearest = torch.cdist(x, points).min(dim=1)
    proportions = torch.bincount(nearest, minlength=5) / len(x)
    return (
        0.5 * (proportions - probabilities).abs().sum().item(),
        distance.square().mean().sqrt().item(),
    )


mixed_field = train(lambda t: t.clamp(max=THRESHOLD))
equilibrium = train(torch.zeros_like)

torch.manual_seed(1)
x0 = torch.randn(N_SAMPLES, 2, device=device)
samples = {
    "mixed-field matching": MixedFieldSampler(
        ct_threshold=THRESHOLD, refinement_rate=0.25, integrator="heun"
    )(
        mixed_field,
        x0,
        n_steps=N_TRANSPORT_STEPS,
        n_refinement_steps=N_REFINEMENT_STEPS,
    ),
    "equilibrium matching": GradientDescentSampler(
        step_size=(1 - THRESHOLD) / 16, prediction_type="velocity"
    )(equilibrium, x0, n_steps=N_REFINEMENT_STEPS),
}
for name, x in samples.items():
    tv, rms = scores(x)
    print(f"{name:>20}: total variation {tv:.4f}, RMS distance {rms:.4f}")
