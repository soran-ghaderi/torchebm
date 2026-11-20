"""Drifting models in 2D: a one-step generator trained with the drifting loss.

Drifting models (Deng et al., 2026) train a generator x = f(eps) so that its
samples stop drifting: the loss regresses each generated sample onto its
detached value moved by the drifting field V = V+ - V-, which attracts it to
the data and repels it from the other generated samples (Eqs. 6 to 12).
Generation is one call of the generator, with no time input. As in the
reference toy notebook (lambertae.github.io, projects/drifting/notebooks/
drifting_model_demo.ipynb): a checkerboard, an MLP from 32-dimensional
Gaussian noise with three hidden layers of 256 units and SiLU, 2048 data and
2048 generated samples per step, Adam with learning rate 1e-3 and 2000
steps. The loss is the one of the ImageNet models, with the feature and drift
normalization and the temperatures 0.02, 0.05 and 0.2 (App. A.6); the
notebook uses the field without normalization at temperature 0.05. Each step
splits the batch into 32 groups of 64 generated and 64 data samples, one
field each, the group size of the ImageNet models (Table 8).

Samples are scored by the share that lands on the eight occupied squares and
by the smallest and largest share of one square, next to the same scores of
the data.
"""

import os

import torch
from torch import nn

from torchebm.losses import DriftingLoss

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 2000
BATCH_SIZE = 256 if SMOKE else 2048
GROUP_SIZE = 64
N_GEN = 1000 if SMOKE else 20000
NOISE_DIM = 32

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


def checkerboard(n):
    b = torch.randint(0, 2, (n,), device=device)
    i = 2 * torch.randint(0, 2, (n,), device=device) + b
    j = 2 * torch.randint(0, 2, (n,), device=device) + b
    x = (torch.stack([i, j], dim=1) + torch.rand(n, 2, device=device) - 2) / 2
    return x + 0.05 * torch.randn(n, 2, device=device)


class MLP(nn.Module):
    def __init__(self, in_dim=NOISE_DIM, hidden=256, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


model = MLP().to(device)
loss_fn = DriftingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(N_STEPS):
    x = model(torch.randn(BATCH_SIZE, NOISE_DIM, device=device))
    loss = loss_fn(x.unflatten(0, (-1, GROUP_SIZE)), checkerboard(BATCH_SIZE).unflatten(0, (-1, GROUP_SIZE)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def scores(x):
    """Share on the occupied squares, and the smallest and largest share of one square."""
    cell = ((x + 1) * 2).floor().long()
    occupied = ((cell >= 0) & (cell < 4)).all(dim=1) & (cell.sum(dim=1) % 2 == 0)
    counts = torch.bincount(cell[occupied, 0] * 4 + cell[occupied, 1], minlength=16)
    shares = counts[[k for k in range(16) if (k // 4 + k % 4) % 2 == 0]] / len(x)
    return occupied.float().mean().item(), shares.min().item(), shares.max().item()


torch.manual_seed(1)
for name, samples in (("data", checkerboard(N_GEN)), ("one step", model(torch.randn(N_GEN, NOISE_DIM, device=device)))):
    inside, low, high = scores(samples.detach())
    print(f"{name:>8}: {inside:.1%} on the occupied squares, {low:.1%} to {high:.1%} per square")
