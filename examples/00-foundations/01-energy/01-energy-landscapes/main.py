"""Energy functions: an energy E(x) defines a density p(x) proportional to exp(-E(x)).

Each model maps points of shape (N, 2) to energies (N,); the negative gradient
-grad E is the score of the model density (physicists call it the force), and it
is the direction every gradient-based sampler follows. Lower energy means higher
probability. Built with prediction_type=("energy", "score"), one call returns
both; the analytic models have closed-form scores.
"""

import torch

from torchebm.core import GaussianModel, DoubleWellModel, RosenbrockModel

models = {
    "gaussian": GaussianModel(
        mean=torch.zeros(2), cov=torch.eye(2), prediction_type=("energy", "score")
    ),
    "double_well": DoubleWellModel(
        barrier_height=2.0, prediction_type=("energy", "score")
    ),
    "rosenbrock": RosenbrockModel(prediction_type=("energy", "score")),
}

xs = torch.linspace(-3.0, 3.0, 25)
grid = torch.stack(torch.meshgrid(xs, xs, indexing="xy"), dim=-1).reshape(-1, 2)

for name, model in models.items():
    energy, score = model(grid)
    print(f"{name:12s} E in [{energy.min():.2f}, {energy.max():.2f}]   "
          f"mean |score| = {score.norm(dim=1).mean():.2f}")
