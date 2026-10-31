<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/logo_with_text.svg" alt="TorchEBM Logo" width="350">
</p>

<p align="center" style="margin-bottom: 20px;">
    <a href="https://pypi.org/project/torchebm/" target="_blank" title="PyPI version">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/torchebm?style=flat-square&color=blue">
    </a>
    <a href="https://github.com/soran-ghaderi/torchebm/blob/master/LICENSE" target="_blank" title="License">
        <img alt="License" src="https://img.shields.io/github/license/soran-ghaderi/torchebm?style=flat-square&color=brightgreen">
    </a>
    <a href="https://github.com/soran-ghaderi/torchebm" target="_blank" title="GitHub Repo Stars">
        <img alt="GitHub Stars" src="https://img.shields.io/github/stars/soran-ghaderi/torchebm?style=social">
    </a>
    <a href="https://deepwiki.com/soran-ghaderi/torchebm"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
    <a href="https://github.com/soran-ghaderi/torchebm/actions" target="_blank" title="Build Status">
      <img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/soran-ghaderi/torchebm/tag-release.yml?branch=master&style=flat-square&label=build">
    </a>
    <a href="https://github.com/soran-ghaderi/torchebm/actions" target="_blank" title="Documentation">
      <img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/soran-ghaderi/torchebm/docs_ci.yml?branch=master&style=flat-square&label=docs">
    </a>
    <a href="https://pepy.tech/project/torchebm" target="_blank" title="Downloads">
        <img alt="Downloads" src="https://static.pepy.tech/badge/torchebm?style=flat-square">
    </a>
    <a href="https://pypi.org/project/torchebm/" target="_blank" title="Python Versions">
        <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/torchebm?style=flat-square">
    </a>
</p>

<h1 align="center">⚡ TorchEBM: simulation-free, GPU-first generative modeling in PyTorch</h1>
<h3 align="center">Composable primitives for scalable, stable training of EBMs, diffusion, flow matching, and equilibrium and energy matching.</h3>

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/animations/ebm_training_animation.gif" alt="An energy landscape being learned from data"/>
  <br>
  <em>An energy landscape learning to place its low-energy basins on the data.</em>
</p>

## What is ∇ TorchEBM 🍓?

A PyTorch library of composable, GPU-first primitives for energy-based models, diffusion, flow matching, and equilibrium and energy matching. Every method is a composition of the same parts:

| Component | Role | Package |
|---|---|---|
| **Model** | the energy or vector field | `torchebm.core`, `torchebm.models` |
| **Interpolant** | the path from noise to data | `torchebm.interpolants` |
| **Coupling** | which noise pairs with which datum | `torchebm.couplings` |
| **Objective** | how the model fits the data | `torchebm.losses` |
| **Sampler** | the dynamics that generate | `torchebm.samplers` |
| **Integrator** | how the dynamics are discretized | `torchebm.integrators` |

📚 [Documentation](https://soran-ghaderi.github.io/torchebm/) · 🧭 [Design and scope](https://soran-ghaderi.github.io/torchebm/latest/concepts/design/) · ⚡ [Benchmarks](https://soran-ghaderi.github.io/torchebm-benchmarks/)

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/animations/8gaussians_flow.gif" alt="Equilibrium matching transporting noise onto eight gaussians" width="700"/>
  <br>
  <em>Equilibrium matching on eight gaussians.</em>
</p>

## What's inside

### Training objectives (`torchebm.losses`)

| Objective | Idea | Inner sampling |
|---|---|---|
| `ContrastiveDivergenceLoss` | push energy down on data, up on short-run MCMC negatives drawn by any sampler; a `ReplayBuffer` gives persistent CD | yes |
| `ScoreMatchingLoss` | match the model score to the data score | no |
| `DenoisingScoreMatchingLoss` | match the score of noise-perturbed data | no |
| `SlicedScoreMatchingLoss` | estimate the Hessian trace by random projections | no |
| `FlowMatchingLoss` | regress a time-conditioned velocity onto the conditional velocity of a path | no |
| `EquilibriumMatchingLoss` | regress a time-invariant field (implicit) or the gradient of a potential (EqM-E) onto the scaled path velocity; generate by ODE integration or fixed-point iteration | no |
| `EnergyMatchingLoss` | one time-independent potential: optimal-transport flow warm-up, then contrastive sharpening of its Boltzmann density | phase two only |

### Samplers (`torchebm.samplers`)

| Sampler | Dynamics |
|---|---|
| `LangevinDynamics` | overdamped Langevin: steps along the score with Gaussian noise; samples the Boltzmann density of an energy |
| `HamiltonianMonteCarlo` | Hamiltonian trajectories with a symplectic integrator and a Metropolis test |
| `RiemannianManifoldHMC` | HMC on a position-dependent metric, for ill-conditioned geometry |
| `GradientDescentSampler` | noise-free steps along the model's field: descent on an energy, fixed-point iteration on a general field |
| `NesterovSampler` | the same steps with Nesterov momentum |
| `FlowSampler` | continuous-time generation: `mode="ode"` (probability flow) or `mode="sde"` (diffusion), from velocity, score, or noise predictions |

### Numerical integrators (`torchebm.integrators`)

| Family | Integrators |
|---|---|
| Explicit Runge-Kutta | `MidpointIntegrator`, `HeunIntegrator`, `Bosh3Integrator`, `RK4Integrator`, `RK438Integrator` |
| Adaptive step size | `AdaptiveHeunIntegrator`, `Dopri5Integrator`, `Dopri8Integrator` |
| Stochastic (SDE) | `EulerMaruyamaIntegrator`, `BackwardEulerMaruyamaIntegrator` (implicit) |
| Symplectic (Hamiltonian) | `LeapfrogIntegrator`, `GeneralisedLeapfrogIntegrator` (non-separable) |

### Paths and pairings (`torchebm.interpolants`, `torchebm.couplings`)

| Interpolants | Couplings |
|---|---|
| `LinearInterpolant` (flow matching) | `IndependentCoupling` (the classical pairing) |
| `CosineInterpolant` | `GreedyCoupling` (nearest unmatched) |
| `VariancePreservingInterpolant`, `VarianceExplodingInterpolant` (diffusion) | `SinkhornCoupling` (entropic OT), `ExactOTCoupling` |
| | `UnbalancedSinkhornCoupling` (per-pair weights) |

### And the rest

Analytic energies (`GaussianModel`, `DoubleWellModel`, ...), `DiT` presets, `TimeConditionedMLP`, `ClassifierFreeGuidance`, `InteractionModel`, `ReplayBuffer`, `BaseTrainer`, schedulers, 2D datasets, CUDA kernels and mixed precision.

## Install

```bash
pip install torchebm
```

## Usage

Each block implements one published method. Runnable versions are in the [examples](https://soran-ghaderi.github.io/torchebm/latest/examples/).

### Equilibrium matching

<a href="https://arxiv.org/abs/2510.02300">Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models</a>, Wang & Du 2025. One time-invariant field that vanishes on the data, sampled by descending it.

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/equilibrium_matching_field.png" alt="Flow matching's field changes with t; equilibrium matching learns a single time-invariant field" width="820"/>
  <br>
  <em>Fig. 1 of <a href="https://arxiv.org/abs/2510.02300">Wang & Du (2025)</a>.</em>
</p>

EqM-XL/2 on ImageNet 256, as in the paper:

```python
import torch
from diffusers.models import AutoencoderKL
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchebm.interpolants import LinearInterpolant
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.models import ClassifierFreeGuidance, dit_xl_2
from torchebm.samplers import NesterovSampler

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
dataset = datasets.ImageFolder("imagenet/train", transform)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8, drop_last=True)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").cuda()

model = dit_xl_2(
    input_size=32, in_channels=4, num_classes=1000, class_dropout_prob=0.1,
    time_invariant=True, prediction_type="score",
).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9999))
path, loss_fn = LinearInterpolant(), EquilibriumMatchingLoss()

for epoch in range(1400):
    for images, y in loader:
        images, y = images.cuda(), y.cuda()
        with torch.no_grad():
            x1 = vae.encode(images).latent_dist.sample() * 0.18215
        x0, t = torch.randn_like(x1), torch.rand(len(x1), device="cuda")
        x_t, u_t = path(x0, x1, t)
        loss = loss_fn(model(x_t, y=y), u_t, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update_parameters(model)

guided = ClassifierFreeGuidance(
    ema.module.eval(), guidance_scale=1.5, null_condition=1000, guide_channels=3
)
sampler = NesterovSampler(step_size=0.0017, momentum=0.3, lookahead_only=True)
y = torch.randint(1000, (64,), device="cuda")
noise = torch.randn(64, 4, 32, 32, device="cuda")
latents = sampler(guided, noise, n_steps=250, model_kwargs={"y": y})
with torch.no_grad():
    samples = vae.decode(latents / 0.18215).sample
```

### Energy matching

<a href="https://arxiv.org/abs/2504.10612">Energy Matching: Unifying Flow Matching and Energy-Based Models</a>, Balcerak et al. 2025. One time-independent potential: an OT flow warm-up, then contrastive sharpening near the data.

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/energy_matching_transport.png" alt="Action matching, OT flow matching, EBM and energy matching transporting eight gaussians onto two moons" width="820"/>
  <br>
  <em>Fig. 1 of <a href="https://arxiv.org/abs/2504.10612">Balcerak et al. (2025)</a>.</em>
</p>

```python
from torchebm.core import TemperatureScheduler
from torchebm.couplings import SinkhornCoupling
from torchebm.interpolants import LinearInterpolant
from torchebm.losses import EnergyMatchingLoss
from torchebm.samplers import LangevinDynamics

model = EnergyMLP(prediction_type=("energy", "score"))
coupling, path = SinkhornCoupling(reg=0.01), LinearInterpolant()
loss_fn = EnergyMatchingLoss(lambda_cd=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
langevin_sweep = LangevinDynamics(
    step_size=0.01, noise_scale=TemperatureScheduler(epsilon_max=0.15, tau_star=0.8, n_steps=200)
)
langevin_hold = LangevinDynamics(step_size=0.01, noise_scale=0.15 ** 0.5)

for step in range(22_000):
    x1 = data[torch.randint(len(data), (128,))]
    x0, x1 = coupling(torch.randn_like(x1), x1)
    t = torch.rand(len(x1))
    x_t, u_t = path(x0, x1, t)
    _, score = model(x_t)
    if step < 20_000:
        loss = loss_fn(score, u_t, t)
    else:
        negatives = torch.cat([langevin_sweep(model, x0[:64], n_steps=200),
                               langevin_hold(model, x1[64:], n_steps=200)])
        loss = loss_fn(score, u_t, t, model(x1)[0], model(negatives)[0])
    optimizer.zero_grad(); loss.backward(); optimizer.step()

temperature = TemperatureScheduler(epsilon_max=0.15, tau_star=0.8, n_steps=200, t_end=1.0)
sampler = LangevinDynamics(step_size=0.01, noise_scale=temperature)
samples = sampler(model, torch.randn(4000, 2), n_steps=200)
```

### Flow matching

<a href="https://arxiv.org/abs/2210.02747">Flow Matching for Generative Modeling</a>, Lipman et al. 2023. Regress a time-conditioned velocity field onto the conditional velocity of a probability path.

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/flow_matching_diffusion_vs_ot.png" alt="Diffusion path conditional score field versus optimal-transport path conditional vector field" width="820"/>
  <br>
  <em>Fig. 2 of <a href="https://arxiv.org/abs/2210.02747">Lipman et al. (2023)</a>.</em>
</p>

```python
from torchebm.interpolants import LinearInterpolant
from torchebm.losses import FlowMatchingLoss
from torchebm.models import TimeConditionedMLP
from torchebm.samplers import FlowSampler

model = TimeConditionedMLP(in_dim=2)
path, loss_fn = LinearInterpolant(), FlowMatchingLoss()

x1 = data[torch.randint(len(data), (256,))]
x0 = torch.randn_like(x1)
t = torch.rand(len(x1))
x_t, u_t = path(x0, x1, t)
loss = loss_fn(model(x_t, t), u_t)

sampler = FlowSampler(interpolant="linear", integrator="heun")
samples = sampler(model, torch.randn(1000, 2), n_steps=50)
```

With `interpolant="vp"` and `mode="sde"`, the same sampler runs score-based diffusion (<a href="https://arxiv.org/abs/2011.13456">Song et al. 2021</a>):

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/score_sde_forward_reverse.png" alt="Forward SDE to a prior, reverse SDE and probability flow ODE back to data" width="820"/>
  <br>
  <em>Fig. 2 of <a href="https://arxiv.org/abs/2011.13456">Song et al. (2021)</a>.</em>
</p>

```python
sampler = FlowSampler(mode="sde", interpolant="vp", prediction_type="epsilon")
samples = sampler(model, torch.randn(1000, 2), n_steps=250)
```

### Optimal-transport couplings

<a href="https://arxiv.org/abs/2302.00482">Improving and generalizing flow-based generative models with minibatch optimal transport</a>, Tong et al. 2024. Minibatch OT pairing straightens the flow.

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/flow_matching_probability_paths.png" alt="Flow matching, conditional flow matching and OT conditional flow matching probability paths" width="820"/>
  <br>
  <em>Fig. 1 of <a href="https://arxiv.org/abs/2302.00482">Tong et al. (2024)</a>.</em>
</p>

```python
from torchebm.couplings import SinkhornCoupling, ExactOTCoupling

coupling = SinkhornCoupling(reg=0.05)
x0, x1 = coupling(x0, x1)
```

### Rectified flow

<a href="https://arxiv.org/abs/2209.03003">Flow Straight and Fast</a>, Liu et al. 2023. Retrain on the flow's own (noise, sample) pairs for straight, few-step paths.

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/rectified_flow_straightening.png" alt="Linear interpolation crosses; the rectified flow induced by its own pairs is straight" width="820"/>
  <br>
  <em>Fig. 2 of <a href="https://arxiv.org/abs/2209.03003">Liu et al. (2023)</a>.</em>
</p>

```python
x0 = torch.randn(256, 2)
x1 = FlowSampler(integrator="heun")(model, x0, n_steps=50)
```

### Contrastive divergence

<a href="https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf">Training products of experts by minimizing contrastive divergence</a>, Hinton 2002, with the deep-EBM recipe of <a href="https://arxiv.org/abs/1903.08689">Du & Mordatch 2019</a>. Push the energy down on data and up on short-run MCMC negatives.

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/contrastive_divergence_replay_buffer.png" alt="Replay buffer, Langevin negatives, and the CD objective" width="520"/>
  <br>
  <em>Fig. 1 of <a href="https://arxiv.org/abs/1903.08689">Du & Mordatch (2019)</a>.</em>
</p>

```python
import torch
from torch import nn
from torchebm.core import BaseModel
from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import ContrastiveDivergenceLoss
from torchebm.samplers import LangevinDynamics

class EnergyMLP(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

data = TwoMoonsDataset(n_samples=3000, noise=0.05).get_data()
model = EnergyMLP()
sampler = LangevinDynamics(step_size=0.1, noise_scale=1.0)
loss_fn = ContrastiveDivergenceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(1000):
    batch = data[torch.randint(len(data), (256,))]
    negatives = sampler(model, batch, n_steps=10)
    loss = loss_fn(model(batch), model(negatives))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

Persistent CD (<a href="https://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf">Tieleman 2008</a>) keeps the chains in a replay buffer:

```python
from torchebm.utils import ReplayBuffer

buffer = ReplayBuffer(capacity=8192, sample_shape=(2,))
negatives = sampler(model, buffer.sample(256), n_steps=10)
buffer.add(negatives)
```

### Score matching

<a href="https://jmlr.org/papers/v6/hyvarinen05a.html">Estimation of non-normalized statistical models by score matching</a>, Hyvärinen 2005, with denoising (<a href="https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf">Vincent 2011</a>) and sliced (<a href="https://arxiv.org/abs/1905.07088">Song et al. 2019</a>) variants.

```python
from torchebm.losses import DenoisingScoreMatchingLoss

model = EnergyMLP(prediction_type="score")
x = data[torch.randint(len(data), (256,))]
noise = torch.randn_like(x)
loss = DenoisingScoreMatchingLoss()(model(x + 0.1 * noise), -noise / 0.1)
```

### Sampling an energy

Chains are a batch dimension, and the model is an argument of the call.

```python
import torch
from torchebm.core import GaussianModel
from torchebm.samplers import LangevinDynamics

model = GaussianModel(mean=torch.zeros(2), cov=torch.tensor([[1., .8], [.8, 1.]]))
sampler = LangevinDynamics(step_size=0.02, noise_scale=1.0)

samples = sampler(model, torch.randn(10_000, 2), n_steps=500)
```

### Hamiltonian Monte Carlo

<a href="https://arxiv.org/abs/1206.1901">MCMC using Hamiltonian dynamics</a>, Neal 2011.

```python
from torchebm.samplers import HamiltonianMonteCarlo

sampler = HamiltonianMonteCarlo(step_size=0.1, n_leapfrog_steps=10)
samples = sampler(model, torch.randn(1000, 2), n_steps=500)
```

### Riemannian manifold HMC

<a href="https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00765.x">Riemann manifold Langevin and Hamiltonian Monte Carlo</a>, Girolami & Calderhead 2011. `metric_fn` maps `(b, d)` points to `(b, d, d)` SPD matrices.

```python
from torchebm.samplers import RiemannianManifoldHMC

def metric_fn(x):
    scale = 1.0 + x.pow(2).sum(-1)[:, None, None]
    return torch.eye(2).expand(x.shape[0], 2, 2) * scale

sampler = RiemannianManifoldHMC(metric_fn=metric_fn, step_size=0.05, n_leapfrog_steps=5)
samples = sampler(model, torch.randn(1000, 2), n_steps=200)
```

## Examples

Runnable and tested in CI: [`examples/`](https://github.com/soran-ghaderi/torchebm/tree/master/examples), also on the [website](https://soran-ghaderi.github.io/torchebm/latest/examples/).

| Tier | You learn to |
|---|---|
| [`00-foundations/`](https://github.com/soran-ghaderi/torchebm/tree/master/examples/00-foundations) | work with energies, datasets, schedulers, and interpolants |
| [`10-sampling/`](https://github.com/soran-ghaderi/torchebm/tree/master/examples/10-sampling) | sample a fixed target with MCMC, parallel chains, integrators, and flows |
| [`20-training/`](https://github.com/soran-ghaderi/torchebm/tree/master/examples/20-training) | learn a target: CD, persistent CD, score matching, equilibrium and energy matching, couplings |
| [`90-showcase/`](https://github.com/soran-ghaderi/torchebm/tree/master/examples/90-showcase) | study end-to-end demos that push the components |

```bash
python examples/10-sampling/01-mcmc/01-langevin-101/main.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Developer Guide](https://soran-ghaderi.github.io/torchebm/latest/developer_guide/).

## Built with TorchEBM

| Year | Paper | |
|---|---|---|
| 2026 | **Mixed-Field Matching** | Preprint soon |

## Citation

If TorchEBM helps your research, please ⭐️ the repository and cite it:

```bibtex
@misc{ghaderi2024torchebm,
  author       = {Ghaderi, Soran},
  title        = {{TorchEBM}: A Composable {PyTorch} Library for Energy-Based and Transport-Based Generative Modeling},
  year         = {2024},
  howpublished = {\url{https://github.com/soran-ghaderi/torchebm}},
}
```

## Changelog

See [CHANGELOG](https://github.com/soran-ghaderi/torchebm/blob/master/CHANGELOG.md) for version history.

## License

MIT License. See [LICENSE](https://github.com/soran-ghaderi/torchebm/blob/master/LICENSE) for details.

## Research collaboration

Interested in research on energy-based, flow or diffusion models? Reach out.
