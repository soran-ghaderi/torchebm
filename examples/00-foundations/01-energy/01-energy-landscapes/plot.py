"""Optional visualization for energy-landscapes (needs: pip install matplotlib).

Renders each energy as a filled contour with its force field -grad E overlaid.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from torchebm.core import GaussianModel, DoubleWellModel, RosenbrockModel

models = {
    "Gaussian": GaussianModel(mean=torch.zeros(2), cov=torch.eye(2)),
    "Double well": DoubleWellModel(barrier_height=2.0),
    "Rosenbrock": RosenbrockModel(),
}

xs = torch.linspace(-3.0, 3.0, 200)
gx, gy = torch.meshgrid(xs, xs, indexing="xy")
grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

qs = torch.linspace(-3.0, 3.0, 16)
qx, qy = torch.meshgrid(qs, qs, indexing="xy")
qgrid = torch.stack([qx.reshape(-1), qy.reshape(-1)], dim=-1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=130)
for ax, (name, model) in zip(axes, models.items()):
    energy = torch.log1p(model(grid) - model(grid).min()).reshape(200, 200).detach()
    force = -model.gradient(qgrid).detach()
    ax.contourf(gx, gy, energy, levels=40, cmap="magma")
    ax.quiver(qx.reshape(-1), qy.reshape(-1), force[:, 0], force[:, 1],
              color="white", alpha=0.6, width=0.004)
    ax.set_title(name)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
out = Path(__file__).parent / "assets" / "output.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
