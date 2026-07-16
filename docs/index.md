---
title: Simulation-Free Generative Modeling in PyTorch
social:
  cards_layout_options:
    title: Documentation
hide:
  - navigation
  - toc
icon: octicons/home-fill-16
---

<style>
  .md-content h1:first-child { display: none; }
</style>

<div class="home-top">
  <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" class="home-nabla" aria-hidden="true">
    <polygon points="27.9,26.9 18.8,26.9 54.6,89.0 95.0,19.0 90.4,11.1 54.6,73.1" fill="#4DE8A0" stroke="#0a2818" stroke-width="1" stroke-linejoin="bevel"/>
    <polygon points="72.1,26.9 76.7,19.0 5.0,19.0 45.4,89.0 54.6,89.0 18.8,26.9" fill="#C7FF00" stroke="#0a2818" stroke-width="1" stroke-linejoin="bevel"/>
    <polygon points="50.0,65.1 54.6,73.1 90.4,11.0 9.6,11.1 5.0,19.0 76.7,19.0" fill="#1A7848" stroke="#0a2818" stroke-width="1" stroke-linejoin="bevel"/>
  </svg>

  <div class="home-title">
    <span class="home-torch">Torch</span><span class="home-ebm">EBM</span><span class="home-berry" aria-hidden="true">&#127827;</span>
  </div>

  <p class="home-tagline">
    Simulation-free, GPU-first generative modeling in PyTorch<br>Composable primitives for scalable, stable training of modern <b>EBMs</b>, <b>diffusion</b>, <b>flow matching</b>, and <b>Schrödinger bridges</b>.
  </p>

  <div class="home-actions">
    <a href="getting_started/" class="md-button md-button--primary">
      Get Started
    </a>
    <a href="https://github.com/soran-ghaderi/torchebm" class="md-button home-star-btn" target="_blank" rel="noopener">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor" style="vertical-align: -3px; margin-right: 4px;"><path d="M12 .587l3.668 7.568L24 9.306l-6 5.848 1.416 8.259L12 19.446l-7.416 3.967L6 15.154 0 9.306l8.332-1.151z"/></svg>
      Star on GitHub
      <img src="https://img.shields.io/github/stars/soran-ghaderi/torchebm?style=social&label=" alt="" class="home-star-count">
    </a>
  </div>
    <style>
    .md-content h1:first-child { display: none; }
    .home-top {
      text-align: center;
      padding: 0 0 20px;
      margin-bottom: 20px;
    }
    .home-nabla {
      display: block;
      margin: 0 auto 20px;
      width: 220px;
      height: 220px;
      max-width: 60vw;
    }
    .home-title {
      font-size: 3.2em;
      font-weight: 700;
      font-family: Raleway, sans-serif;
      letter-spacing: 0.5px;
      line-height: 1.1;
      margin-bottom: 16px;
    }
    .home-torch {
      background: linear-gradient(90deg, #C7FF00, #C7FF4C);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      text-shadow: none;
      filter: drop-shadow(0 0 1px #000) drop-shadow(0 0 1px #000);
    }
    .home-ebm {
      background: linear-gradient(90deg, #00c6ff, #0072ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      text-shadow: none;
      filter: drop-shadow(0 0 1px #000) drop-shadow(0 0 1px #000);
    }
    .home-berry {
      font-size: 0.5em;
      vertical-align: super;
      -webkit-text-fill-color: initial;
    }
    .home-tagline {
      font-size: 1.2em;
      max-width: 700px;
      margin: 0 auto 24px;
      line-height: 1.6;
      color: var(--md-default-fg-color--light);
    }
    .home-actions {
      display: flex;
      justify-content: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    /* Badges */
    .home-badges {
      text-align: center;
      margin: 8px 0 0;
      line-height: 2;
    }
    .home-badges a { text-decoration: none; }
    .home-badges img { vertical-align: middle; margin: 2px 1px; }

    /* Demo animations */
    .home-demos {
      display: flex;
      justify-content: center;
      gap: 20px;
      flex-wrap: wrap;
      margin: 24px 0 8px;
    }
    .home-demo {
      text-align: center;
      margin: 0;
      flex: 1;
      min-width: 280px;
      max-width: 50%;
    }
    .home-demo img {
      width: 100%;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .home-demo figcaption {
      margin-top: 10px;
      font-size: 0.9em;
      color: var(--md-default-fg-color--light);
    }
    .home-demo-caption {
      text-align: center;
      font-size: 0.85em;
      color: var(--md-default-fg-color--light);
      margin-top: 0;
    }

    /* Hero star button */
    .home-star-btn {
      position: relative;
    }
    .home-star-count {
      height: 20px;
      vertical-align: middle;
      margin-left: 6px;
      opacity: 0.7;
    }

    /* Post-quickstart star callout */
    .home-star-callout {
      text-align: center;
      margin: 32px auto;
      padding: 24px 32px;
      max-width: 600px;
      border-radius: 12px;
      border: 1px solid var(--md-default-fg-color--lightest);
      background: var(--md-code-bg-color);
    }
    .home-star-callout p {
      margin: 0 0 16px;
      font-size: 1.05em;
      line-height: 1.5;
    }
    .home-star-callout-btn {
      font-size: 0.95em;
    }

    /* Footer links */
    .home-footer {
      text-align: center;
      font-size: 0.9em;
      color: var(--md-default-fg-color--light);
      margin-top: 32px;
      padding-top: 16px;
      border-top: 1px solid var(--md-default-fg-color--lightest);
    }

    @media (max-width: 600px) {
      .home-nabla { width: 160px; height: 160px; }
      .home-title { font-size: 2.4em; }
      .home-tagline { font-size: 1em; padding: 0 16px; }
      .home-demo { min-width: 100%; max-width: 100%; }
    }
  </style>

</div>



<div class="home-badges">
  <a href="https://pypi.org/project/torchebm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/torchebm?style=flat-square&color=blue"></a>
  <a href="https://github.com/soran-ghaderi/torchebm/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/soran-ghaderi/torchebm?style=flat-square&color=brightgreen"></a>
  <a href="https://github.com/soran-ghaderi/torchebm"><img alt="Stars" src="https://img.shields.io/github/stars/soran-ghaderi/torchebm?style=social"></a>
  <a href="https://deepwiki.com/soran-ghaderi/torchebm"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
  <a href="https://github.com/soran-ghaderi/torchebm/actions"><img alt="Build" src="https://img.shields.io/github/actions/workflow/status/soran-ghaderi/torchebm/tag-release.yml?branch=master&style=flat-square&label=build"></a>
  <a href="https://github.com/soran-ghaderi/torchebm/actions"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/soran-ghaderi/torchebm/docs_ci.yml?branch=master&style=flat-square&label=docs"></a>
  <a href="https://pepy.tech/project/torchebm"><img alt="Downloads" src="https://static.pepy.tech/badge/torchebm?style=flat-square"></a>
  <a href="https://pypi.org/project/torchebm/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/torchebm?style=flat-square"></a>
</div>


---

## Overview

TorchEBM is a PyTorch library for simulation-free, GPU-first generative
modeling: scalable, stable training of modern energy-based models, diffusion,
flow matching, and Schrödinger bridges. Energy-based models define probability
distributions through a scalar energy function, and the formulation is general
enough that much of modern generative modeling, from MCMC sampling and score
matching to simulation-free transport along probability paths, factors into
the same components, i.e. fields, probability paths, couplings, objectives,
and integrators. TorchEBM implements these components as composable,
high-throughput PyTorch primitives.

The [Design and Scope](concepts/design.md) page states this framing precisely
and places each method family within it.

---

## In Action

<div class="home-demos">
  <figure class="home-demo">
    <img src="assets/animations/8gaussians_flow.gif" alt="Equilibrium matching on eight gaussians" loading="lazy">
    <figcaption>Eight-gaussians distribution</figcaption>
  </figure>
  <figure class="home-demo">
    <img src="assets/animations/circles_flow.gif" alt="Equilibrium matching on circles" loading="lazy">
    <figcaption>Circles distribution</figcaption>
  </figure>
</div>

<p class="home-demo-caption">Equilibrium matching with different interpolants transporting noise onto structured distributions.</p>

---

## Core Components

<!-- torchebm:cards components -->

---

## Quick Start

```bash
pip install torchebm
```

Train a generative model with **equilibrium matching**, then sample the same
network both as an ODE flow and as a scalar energy:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.samplers import FlowSampler, NesterovSampler
from torchebm.core import BaseModel
from torchebm.datasets import EightGaussiansDataset

dataset = EightGaussiansDataset(n_samples=8192)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Any nn.Module with forward(x, t) works
class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2))
    def forward(self, x, t, **kwargs):
        return self.net(x)

model = VelocityNet()

loss_fn = EquilibriumMatchingLoss(
    model=model, interpolant="linear", energy_type="dot",
)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(50):
    for x in loader:
        loss = loss_fn(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Sample via ODE flow
flow = FlowSampler(model=model, interpolant="linear", negate_velocity=True)
flow_samples = flow.sample(x=torch.randn(1000, 2), n_steps=100)

# Same model as a scalar energy: g(x) = x · f(x)
class LearnedEnergy(BaseModel):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        t = torch.zeros(x.shape[0], device=x.device)
        return (x * self.net(x, t)).sum(-1)

nesterov = NesterovSampler(LearnedEnergy(model), step_size=0.01, momentum=0.9)
energy_samples = nesterov.sample(n_samples=1000, dim=2, n_steps=200)
```

One model, two views: the flow view generates in a fixed number of steps, the
energy view supports gradient-based refinement. That interchangeability is the
library's central design property.

See the [concepts](concepts/) and [examples](examples/) for the theory behind
each component and a runnable, CI-tested curriculum.

<div class="home-star-callout">
  <p>
    <strong>Enjoying TorchEBM?</strong> A GitHub star helps others discover the project and motivates continued development.
  </p>
  <a href="https://github.com/soran-ghaderi/torchebm" class="md-button md-button--primary home-star-callout-btn" target="_blank" rel="noopener">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor" style="vertical-align: -2px; margin-right: 4px;"><path d="M12 .587l3.668 7.568L24 9.306l-6 5.848 1.416 8.259L12 19.446l-7.416 3.967L6 15.154 0 9.306l8.332-1.151z"/></svg>
    Star on GitHub
  </a>
</div>

---

## Citation

If TorchEBM is useful in your research, please cite it:

```bibtex
@misc{torchebm_library_2025,
  author       = {Ghaderi, Soran and Contributors},
  title        = {{TorchEBM}: Simulation-Free, {GPU}-First Generative Modeling in {PyTorch}},
  year         = {2025},
  url          = {https://github.com/soran-ghaderi/torchebm},
}
```

<p class="home-footer">
  MIT License &middot;
  <a href="https://github.com/soran-ghaderi/torchebm/issues">Issues</a> &middot;
  <a href="developer_guide/">Contributing</a> &middot;
  <a href="https://github.com/soran-ghaderi/torchebm">GitHub</a>
</p>
