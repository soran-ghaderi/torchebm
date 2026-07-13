---
title: examples
draft: true
date: 2023-03-07
authors:
  - soran-ghaderi
categories:
  - Examples
---

# Code Examples 

Content coming soon.

Langevin dynamics sampling
<!-- more -->

```py title="Langevin dynamics sampling" linenums="1"
import torch
from torchebm.core import GaussianModel
from torchebm.samplers import LangevinDynamics

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GaussianModel(mean=torch.zeros(10), cov=torch.eye(10), device=device)

sampler = LangevinDynamics(model=model, step_size=5e-3, device=device)

# 10,000 parallel chains, 500 steps each
final_x = sampler.sample(dim=10, n_steps=500, n_samples=10_000)
print(final_x.shape)  # (10000, 10)

# Trajectories and diagnostics (dict of per-step tensors)
samples, diagnostics = sampler.sample(
    dim=10, n_steps=500, n_samples=250,
    return_trajectory=True, return_diagnostics=True,
)
print(samples.shape)              # (250, 500, 10)
print(diagnostics["energy"].shape)  # (500,)

# Custom initial state
x_init = torch.randn(250, 10, device=device)
samples = sampler.sample(x=x_init, n_steps=100)
```

The full, always-tested versions of these snippets live in the
[Examples gallery](../../examples/index.md).