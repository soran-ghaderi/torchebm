"""Optional visualization for langevin-101 (needs: pip install matplotlib)."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from torchebm.core import GaussianModel
from torchebm.samplers import LangevinDynamics

model = GaussianModel(mean=torch.zeros(2), cov=torch.tensor([[1.0, 0.8], [0.8, 1.0]]))
samples = LangevinDynamics(model=model, step_size=0.02, noise_scale=1.0).sample(
    dim=2, n_samples=2000, n_steps=500
)

s = samples.numpy()
plt.figure(figsize=(4, 4), dpi=130)
plt.scatter(s[:, 0], s[:, 1], s=5, alpha=0.4, c="#C7FF00")
plt.gca().set_aspect("equal")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title("Langevin samples")
plt.xticks([])
plt.yticks([])

out = Path(__file__).parent / "assets" / "output.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
