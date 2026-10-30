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

TorchEBM is a PyTorch library for simulation-free, GPU-first generative modeling: scalable, stable training of energy-based models, diffusion, flow matching, and equilibrium and energy matching, with Schrödinger bridges and flow maps on the roadmap. Simulation-free objectives (flow matching, equilibrium matching, denoising score matching) fit the model with no sampler in the training loop, so training scales like supervised learning; sampling-based objectives (contrastive divergence) stay available where a calibrated energy is worth the MCMC. One set of composable primitives covers both.

Every model in TorchEBM defines a dynamical system on data space that carries samples to the data distribution, and describes it in one of two ways. A **scalar potential** $E_\theta(x)$ gives the drift $-\nabla_x E_\theta$ of Langevin dynamics, whose stationary density at temperature $\varepsilon$ is the Boltzmann density

$$p_\theta(x) \;\propto\; e^{-E_\theta(x)/\varepsilon}.$$

A **vector field** (a velocity, a score, or a noise prediction, conditioned on time or autonomous) gives the drift directly and need not be the gradient of any potential. Generation either transports a source distribution over a fixed time horizon (ODE or SDE integration, as in diffusion and flow matching) or refines samples toward the field's attractors or its stationary distribution (MCMC, fixed-point iteration). Energy-based models are the case of a potential with refinement; the library takes its name from them and covers all of the cases above.

TorchEBM implements the components of these systems once:

| Component | The question it answers | Package |
|---|---|---|
| **Model** | which potential or field defines the dynamics? | `torchebm.core`, `torchebm.models` |
| **Interpolant** | along which path do the source and the data connect? | `torchebm.interpolants` |
| **Coupling** | which source sample is paired with which datum? | `torchebm.couplings` |
| **Objective** | how is the model fit to data? | `torchebm.losses` |
| **Sampler** | which dynamics produce samples? | `torchebm.samplers` |
| **Integrator** | how are those dynamics discretized? | `torchebm.integrators` |

Flow matching with optimal-transport couplings, contrastive divergence with Langevin negatives, and equilibrium matching sampled by ODE integration or fixed-point iteration are three compositions of the same pieces. The [Design and Scope](https://soran-ghaderi.github.io/torchebm/latest/concepts/design/) page places every method family along these axes, including the ones on the roadmap, with references.

📚 Full documentation: [**the TorchEBM website**](https://soran-ghaderi.github.io/torchebm/) · ⚡ [**Benchmarks dashboard**](https://soran-ghaderi.github.io/torchebm-benchmarks/)

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/animations/8gaussians_flow.gif" alt="Equilibrium matching transporting noise onto eight gaussians" width="700"/>
  <br>
  <em><b>Equilibrium matching</b>: a time-invariant field trained on eight gaussians, then integrated from noise with <code>FlowSampler</code>.</em>
</p>

## What's inside

### Training objectives (`torchebm.losses`)

Sampling-based objectives buy a calibrated energy at the cost of MCMC in the loop; transport-based objectives are simulation-free, replacing that inner loop with regression along a path so training scales like supervised learning.

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

Every sampler takes `integrator=` as a registry name or an instance, so the numerics are swappable without touching the model.

| Family | Integrators |
|---|---|
| Explicit Runge-Kutta | `MidpointIntegrator`, `HeunIntegrator`, `Bosh3Integrator`, `RK4Integrator`, `RK438Integrator` |
| Adaptive step size | `AdaptiveHeunIntegrator`, `Dopri5Integrator`, `Dopri8Integrator` |
| Stochastic (SDE) | `EulerMaruyamaIntegrator`, `BackwardEulerMaruyamaIntegrator` (implicit) |
| Symplectic (Hamiltonian) | `LeapfrogIntegrator`, `GeneralisedLeapfrogIntegrator` (non-separable) |

### Paths and pairings (`torchebm.interpolants`, `torchebm.couplings`)

The path decides how the source becomes data, and the coupling decides which source sample becomes which datum; OT pairings straighten the learned flow, so generation needs fewer steps.

| Interpolants | Couplings |
|---|---|
| `LinearInterpolant` (flow matching) | `IndependentCoupling` (the classical pairing) |
| `CosineInterpolant` | `GreedyCoupling` (nearest unmatched) |
| `VariancePreservingInterpolant`, `VarianceExplodingInterpolant` (diffusion) | `SinkhornCoupling` (entropic OT), `ExactOTCoupling` |
| | `UnbalancedSinkhornCoupling` (per-pair weights) |

### And the rest

**Energies**: `AnalyticModel` and nine closed-form landscapes (`GaussianModel`, `DoubleWellModel`, `HarmonicModel`, `RastriginModel`, `AckleyModel`, `RosenbrockModel`, ...) with known statistics and analytic scores, so sampler behavior is measurable. **Architectures**: `DiT` with the standard size presets, `TimeConditionedMLP`, `ClassifierFreeGuidance`, and `InteractionModel` for diverse sampling. **Training utilities**: `ReplayBuffer` for persistent chains, `EMA`, and `BaseTrainer`. **Schedulers** (`CosineScheduler`, `WarmupScheduler`, `TemperatureScheduler`, ...) for any parameter that should anneal. **Datasets**: eight synthetic 2D benchmarks. **CUDA** acceleration and mixed precision throughout.

## Install

```bash
pip install torchebm
```

## Usage

Each block below implements one published method. The blocks are sketches that share `data` and a few small networks; complete, runnable versions live in the [examples gallery](https://soran-ghaderi.github.io/torchebm/latest/examples/), which CI executes. Simulation-free training comes first, sampling-based training after it, and the MCMC samplers close the section.

### Equilibrium matching

<a href="https://arxiv.org/abs/2510.02300">Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models</a>, Wang & Du 2025. Drop time conditioning: learn one time-invariant field, trained to vanish on the data, and sample it by integrating it as an ODE or by iterating it toward its attractors. The paper reads the field as the gradient of an implicit energy; the network is not constrained to be a gradient, and the explicit variant (EqM-E) parameterizes the energy directly.

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/equilibrium_matching_field.png" alt="Flow matching's field changes with t; equilibrium matching learns a single time-invariant field" width="820"/>
  <br>
  <em>Fig. 1 of <a href="https://arxiv.org/abs/2510.02300">Wang & Du (2025)</a>: flow matching needs a different field at every $t$ (left); EqM learns one time-invariant field (right) and samples by iterating it toward the data.</em>
</p>

```python
import torch
from torch import nn
from torchebm.datasets import TwoMoonsDataset
from torchebm.interpolants import LinearInterpolant
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.samplers import FlowSampler, GradientDescentSampler

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2))
    def forward(self, x, t=None, **kwargs):
        return self.net(x)

data = TwoMoonsDataset(n_samples=3000, noise=0.05).get_data()
model = MLP()
path = LinearInterpolant()
loss_fn = EquilibriumMatchingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(3000):
    x1 = data[torch.randint(len(data), (256,))]
    x0 = torch.randn_like(x1)
    t = torch.rand(len(x1))
    x_t, u_t = path(x0, x1, t)
    loss = loss_fn(model(x_t), u_t, t)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

x0 = torch.randn(1000, 2)
samples = FlowSampler(integrator="euler")(model, x0, n_steps=100)
modes = GradientDescentSampler(step_size=0.05)(model, x0, n_steps=200)
```

Both samplers apply the update $x \leftarrow x + \eta f(x)$ of the learned field: `FlowSampler` integrates it over a fixed time horizon, and `GradientDescentSampler` iterates it for longer toward the field's attractors.

### Energy matching

<a href="https://arxiv.org/abs/2504.10612">Energy Matching: Unifying Flow Matching and Energy-Based Models</a>, Balcerak et al. 2025. One time-independent potential does both jobs: an optimal-transport flow warm-up shapes it far from the data, then a contrastive phase sharpens its Boltzmann density near the data. Generation is a single temperature-scheduled Langevin sweep.

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/energy_matching_transport.png" alt="Action matching, OT flow matching, EBM and energy matching transporting eight gaussians onto two moons" width="820"/>
  <br>
  <em>Fig. 1 of <a href="https://arxiv.org/abs/2504.10612">Balcerak et al. (2025)</a>: a time-independent potential $V_\theta(x)$ (right) both transports and equilibrates, where flow matching needs $v_\theta(x, t)$ and a plain EBM needs long MCMC. Reproduced in <a href="https://github.com/soran-ghaderi/torchebm/tree/master/examples/20-training/04-energy-matching">the examples</a>.</em>
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

`EnergyMLP` is the scalar energy net of the contrastive-divergence block below; built with `prediction_type=("energy", "score")`, one call returns $V(x)$ and $-\nabla V(x)$. Phase one is pure OT flow matching; from step 20 000 the contrastive term enters with the paper's negatives, half swept from noise along the temperature profile and half held at $\varepsilon_{\max}$ from data.

### Flow matching

<a href="https://arxiv.org/abs/2210.02747">Flow Matching for Generative Modeling</a>, Lipman et al. 2023. Regress a time-conditioned velocity field onto the conditional velocity of a probability path.

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/flow_matching_diffusion_vs_ot.png" alt="Diffusion path conditional score field versus optimal-transport path conditional vector field" width="820"/>
  <br>
  <em>Fig. 2 of <a href="https://arxiv.org/abs/2210.02747">Lipman et al. (2023)</a>: the diffusion path's conditional score (left) against the OT path's conditional vector field (right). The <code>interpolant</code> selects between them.</em>
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

Swap `interpolant="vp"` and `mode="sde"` and the same sampler runs score-based diffusion (<a href="https://arxiv.org/abs/2011.13456">Song et al. 2021</a>); `prediction_type=` names whether the network emits velocity, score, or noise (`"epsilon"`).

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/score_sde_forward_reverse.png" alt="Forward SDE to a prior, reverse SDE and probability flow ODE back to data" width="820"/>
  <br>
  <em>Fig. 2 of <a href="https://arxiv.org/abs/2011.13456">Song et al. (2021)</a>: the reverse SDE and its probability-flow ODE are the two modes of one sampler (<code>mode="sde"</code> and <code>mode="ode"</code>).</em>
</p>

```python
sampler = FlowSampler(mode="sde", interpolant="vp", prediction_type="epsilon")
samples = sampler(model, torch.randn(1000, 2), n_steps=250)
```

### Optimal-transport couplings

<a href="https://arxiv.org/abs/2302.00482">Improving and generalizing flow-based generative models with minibatch optimal transport</a>, Tong et al. 2024. Pairing noise with data by minibatch optimal transport reduces path crossings, so the learned flow is straighter and generation needs fewer steps.

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/flow_matching_probability_paths.png" alt="Flow matching, conditional flow matching and OT conditional flow matching probability paths" width="820"/>
  <br>
  <em>Fig. 1 of <a href="https://arxiv.org/abs/2302.00482">Tong et al. (2024)</a>: an OT coupling turns crossing, curved transport into straight transport. The objective stays the same; only the coupling applied to each batch changes.</em>
</p>

```python
from torchebm.couplings import SinkhornCoupling, ExactOTCoupling

coupling = SinkhornCoupling(reg=0.05)
x0, x1 = coupling(x0, x1)
```

`SinkhornCoupling` is entropic OT and `ExactOTCoupling` the assignment optimum; the re-paired batch then goes through the interpolant as above.

### Rectified flow

<a href="https://arxiv.org/abs/2209.03003">Flow Straight and Fast</a>, Liu et al. 2023. Retrain on the flow's own (noise, generation) pairs and the trajectories become straight enough for few-step sampling.

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/rectified_flow_straightening.png" alt="Linear interpolation crosses; the rectified flow induced by its own pairs is straight" width="820"/>
  <br>
  <em>Fig. 2 of <a href="https://arxiv.org/abs/2209.03003">Liu et al. (2023)</a>: paths that cross (left) are rewired by the flow's own pairing into straight ones (right).</em>
</p>

```python
x0 = torch.randn(256, 2)
x1 = FlowSampler(integrator="heun")(model, x0, n_steps=50)
```

The ODE of the trained flow maps the noise to $x_1 = \Phi(x_0)$, the model's own output, so no data is needed; the pairs then go through the interpolant as above.


### Contrastive divergence

<a href="https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf">Training products of experts by minimizing contrastive divergence</a>, Hinton 2002, with the deep-EBM recipe of <a href="https://arxiv.org/abs/1903.08689">Du & Mordatch 2019</a>. Push the energy down on data and up on short-run MCMC negatives.

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/contrastive_divergence_replay_buffer.png" alt="Replay buffer, Langevin negatives, and the CD objective" width="520"/>
  <br>
  <em>Fig. 1 of <a href="https://arxiv.org/abs/1903.08689">Du & Mordatch (2019)</a>: Langevin draws the negatives, the buffer persists the chains, the objective separates them from data. That is <code>ContrastiveDivergenceLoss</code> on the energies of data and negatives, with a <code>ReplayBuffer</code> persisting the chains.</em>
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

The energy maps `(b, 2)` to `(b,)`, the sampler runs ten Langevin steps from the data batch (CD-10), and the loss compares the two energies. Persistent CD (<a href="https://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf">Tieleman 2008</a>) starts the chains from a replay buffer that persists across updates:

```python
from torchebm.utils import ReplayBuffer

buffer = ReplayBuffer(capacity=8192, sample_shape=(2,))
negatives = sampler(model, buffer.sample(256), n_steps=10)
buffer.add(negatives)
```

### Score matching

<a href="https://jmlr.org/papers/v6/hyvarinen05a.html">Estimation of non-normalized statistical models by score matching</a>, Hyvärinen 2005. Fit $\nabla_x \log p_\theta$ directly, with no MCMC. Denoising (<a href="https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf">Vincent 2011</a>) and sliced (<a href="https://arxiv.org/abs/1905.07088">Song et al. 2019</a>) variants avoid the Hessian trace.

```python
from torchebm.losses import DenoisingScoreMatchingLoss

model = EnergyMLP(prediction_type="score")
x = data[torch.randint(len(data), (256,))]
noise = torch.randn_like(x)
loss = DenoisingScoreMatchingLoss()(model(x + 0.1 * noise), -noise / 0.1)
```

Built with `prediction_type="score"`, the model's call returns $-\nabla_x E$ with its graph, so the regression trains the energy. `ScoreMatchingLoss` takes the score and its divergence, `SlicedScoreMatchingLoss` the score and the input it was evaluated at; none needs a sampler.

### Sampling an energy

Langevin dynamics takes gradient steps on the energy with injected Gaussian noise. Chains are a batch dimension: the call below advances 10,000 of them as one batch. The model is an argument of the call, as in `torchdiffeq`, and the same samplers that generate from a trained energy also draw the negatives for contrastive divergence.

```python
import torch
from torchebm.core import GaussianModel
from torchebm.samplers import LangevinDynamics

model = GaussianModel(mean=torch.zeros(2), cov=torch.tensor([[1., .8], [.8, 1.]]))
sampler = LangevinDynamics(step_size=0.02, noise_scale=1.0)

samples = sampler(model, torch.randn(10_000, 2), n_steps=500)
```

### Hamiltonian Monte Carlo

<a href="https://arxiv.org/abs/1206.1901">MCMC using Hamiltonian dynamics</a>, Neal 2011. Momentum plus a symplectic integrator buys long, decorrelated proposals.

```python
from torchebm.samplers import HamiltonianMonteCarlo

sampler = HamiltonianMonteCarlo(step_size=0.1, n_leapfrog_steps=10)
samples = sampler(model, torch.randn(1000, 2), n_steps=500)
```

### Riemannian manifold HMC

<a href="https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00765.x">Riemann manifold Langevin and Hamiltonian Monte Carlo</a>, Girolami & Calderhead 2011. A position-dependent metric makes the geometry local; the non-separable Hamiltonian is solved by an implicit generalised leapfrog. `metric_fn` maps points `(b, d)` to SPD matrices `(b, d, d)`.

```python
from torchebm.samplers import RiemannianManifoldHMC

def metric_fn(x):
    scale = 1.0 + x.pow(2).sum(-1)[:, None, None]
    return torch.eye(2).expand(x.shape[0], 2, 2) * scale

sampler = RiemannianManifoldHMC(metric_fn=metric_fn, step_size=0.05, n_leapfrog_steps=5)
samples = sampler(model, torch.randn(1000, 2), n_steps=200)
```

### Image-scale backbones

`DiT` (<a href="https://arxiv.org/abs/2212.09748">Peebles & Xie 2023</a>) for energies and fields on images, with adaLN-Zero blocks and the standard size presets (`dit_s_2` through `dit_xl_8`). `ClassifierFreeGuidance` wraps any label-conditioned model `model(x, t, y=...)` for classifier-free guidance (<a href="https://arxiv.org/abs/2207.12598">Ho & Salimans 2022</a>).

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/images/papers/dit_adaln_zero.png" alt="The DiT architecture and its adaLN-Zero conditioning block" width="820"/>
  <br>
  <em>Fig. 3 of <a href="https://arxiv.org/abs/2212.09748">Peebles & Xie (2023)</a>: the DiT block with adaLN-Zero conditioning, shipped as <code>DiT</code> and <code>AdaLNZeroBlock</code>.</em>
</p>

```python
from torchebm.models import ClassifierFreeGuidance, DiT

model = DiT(input_size=32, in_channels=3, patch_size=4, embed_dim=384, depth=12,
            num_heads=6, num_classes=10)
guided = ClassifierFreeGuidance(model, guidance_scale=2.0, null_condition=10)
out = guided(torch.randn(8, 3, 32, 32), torch.rand(8), y=torch.randint(0, 10, (8,)))
```

<p align="center">
  <img src="https://raw.githubusercontent.com/soran-ghaderi/torchebm/master/docs/assets/animations/circles_flow.gif" alt="Linear, VP and cosine interpolants transporting noise onto concentric circles" width="700"/>
  <br>
  <em>One objective trained with three <b>interpolants</b> (linear, variance-preserving, cosine).</em>
</p>

## Examples

The [`examples/`](https://github.com/soran-ghaderi/torchebm/tree/master/examples) tree is a tiered, runnable curriculum, executed in CI so it cannot drift from the library. Browse it on the [website](https://soran-ghaderi.github.io/torchebm/latest/examples/).

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

Contributions are welcome. Please read the [CONTRIBUTING.md](CONTRIBUTING.md) guide before getting started. The [Developer Guide](https://soran-ghaderi.github.io/torchebm/latest/developer_guide/) covers setup, architecture, and the PR workflow; adding an example is a good first contribution. Check the [issues](https://github.com/soran-ghaderi/torchebm/issues) for open tasks, several labelled `good first issue`.

## Show your support for ∇ TorchEBM 🍓

If TorchEBM helps your research, please ⭐️ the repository and spread the word. It helps others find the project.

## Citation

If TorchEBM is useful in your research, please cite it:

```bibtex
@misc{ghaderi2024torchebm,
  author       = {Ghaderi, Soran},
  title        = {{TorchEBM}: A Composable {PyTorch} Library for Energy-Based and Transport-Based Generative Models},
  year         = {2024},
  howpublished = {\url{https://github.com/soran-ghaderi/torchebm}},
}
```

## References

```bibtex
@article{hinton2002training,
  title   = {Training products of experts by minimizing contrastive divergence},
  author  = {Hinton, Geoffrey E.},
  journal = {Neural Computation},
  volume  = {14},
  number  = {8},
  year    = {2002},
}
```

```bibtex
@article{hyvarinen2005estimation,
  title   = {Estimation of non-normalized statistical models by score matching},
  author  = {Hyv{\"a}rinen, Aapo},
  journal = {Journal of Machine Learning Research},
  volume  = {6},
  year    = {2005},
}
```

```bibtex
@article{vincent2011connection,
  title   = {A connection between score matching and denoising autoencoders},
  author  = {Vincent, Pascal},
  journal = {Neural Computation},
  volume  = {23},
  number  = {7},
  year    = {2011},
}
```

```bibtex
@article{girolami2011riemann,
  title   = {Riemann manifold {L}angevin and {H}amiltonian {M}onte {C}arlo methods},
  author  = {Girolami, Mark and Calderhead, Ben},
  journal = {Journal of the Royal Statistical Society: Series B},
  volume  = {73},
  number  = {2},
  year    = {2011},
}
```

```bibtex
@inproceedings{song2019sliced,
  title   = {Sliced Score Matching: A Scalable Approach to Density and Score Estimation},
  author  = {Song, Yang and Garg, Sahaj and Shi, Jiaxin and Ermon, Stefano},
  year    = {2019},
  eprint  = {1905.07088},
  url     = {https://arxiv.org/abs/1905.07088},
}
```

```bibtex
@inproceedings{du2019implicit,
  title   = {Implicit Generation and Generalization in Energy-Based Models},
  author  = {Du, Yilun and Mordatch, Igor},
  year    = {2019},
  eprint  = {1903.08689},
  url     = {https://arxiv.org/abs/1903.08689},
}
```

```bibtex
@inproceedings{song2021scorebased,
  title   = {Score-Based Generative Modeling through Stochastic Differential Equations},
  author  = {Song, Yang and Sohl-Dickstein, Jascha and Kingma, Diederik P. and Kumar, Abhishek and Ermon, Stefano and Poole, Ben},
  year    = {2021},
  eprint  = {2011.13456},
  url     = {https://arxiv.org/abs/2011.13456},
}
```

```bibtex
@inproceedings{lipman2023flow,
  title   = {Flow Matching for Generative Modeling},
  author  = {Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matt},
  year    = {2023},
  eprint  = {2210.02747},
  url     = {https://arxiv.org/abs/2210.02747},
}
```

```bibtex
@inproceedings{liu2023flow,
  title   = {Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow},
  author  = {Liu, Xingchao and Gong, Chengyue and Liu, Qiang},
  year    = {2023},
  eprint  = {2209.03003},
  url     = {https://arxiv.org/abs/2209.03003},
}
```

```bibtex
@article{albergo2023stochastic,
  title   = {Stochastic Interpolants: A Unifying Framework for Flows and Diffusions},
  author  = {Albergo, Michael S. and Boffi, Nicholas M. and Vanden-Eijnden, Eric},
  year    = {2023},
  eprint  = {2303.08797},
  url     = {https://arxiv.org/abs/2303.08797},
}
```

```bibtex
@article{tong2024improving,
  title   = {Improving and generalizing flow-based generative models with minibatch optimal transport},
  author  = {Tong, Alexander and Fatras, Kilian and Malkin, Nikolay and Huguet, Guillaume and Zhang, Yanlei and Rector-Brooks, Jarrid and Wolf, Guy and Bengio, Yoshua},
  journal = {Transactions on Machine Learning Research},
  year    = {2024},
  eprint  = {2302.00482},
  url     = {https://arxiv.org/abs/2302.00482},
}
```

```bibtex
@article{wang2025equilibrium,
  title   = {Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models},
  author  = {Wang, Runqian and Du, Yilun},
  year    = {2025},
  eprint  = {2510.02300},
  url     = {https://arxiv.org/abs/2510.02300},
}
```

```bibtex
@article{balcerak2025energy,
  title   = {Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling},
  author  = {Balcerak, Michal and Amiranashvili, Tamaz and Terpin, Antonio and Shit, Suprosanna and Bogensperger, Lea and Kaltenbach, Sebastian and Koumoutsakos, Petros and Menze, Bjoern},
  year    = {2025},
  eprint  = {2504.10612},
  url     = {https://arxiv.org/abs/2504.10612},
}
```

```bibtex
@inproceedings{peebles2023scalable,
  title   = {Scalable Diffusion Models with Transformers},
  author  = {Peebles, William and Xie, Saining},
  year    = {2023},
  eprint  = {2212.09748},
  url     = {https://arxiv.org/abs/2212.09748},
}
```

```bibtex
@article{debortoli2021diffusion,
  title   = {Diffusion {S}chr{\"o}dinger Bridge with Applications to Score-Based Generative Modeling},
  author  = {De Bortoli, Valentin and Thornton, James and Heng, Jeremy and Doucet, Arnaud},
  year    = {2021},
  eprint  = {2106.01357},
  url     = {https://arxiv.org/abs/2106.01357},
}
```

## Changelog

See [CHANGELOG](https://github.com/soran-ghaderi/torchebm/blob/master/CHANGELOG.md) for version history.

## License

MIT License. See [LICENSE](https://github.com/soran-ghaderi/torchebm/blob/master/LICENSE) for details.

## Research collaboration

If you are interested in collaborating on research around energy-based, flow-based, or diffusion models, feel free to reach out. Contributions to TorchEBM 🍓 and research discussions are welcome.
