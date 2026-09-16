# Interpolants and Couplings

Transport-based training regresses a model onto a path connecting noise to data. Two independent choices define that path: the **interpolant** (the shape of the path in time) and the **coupling** (which noise point is paired with which data point). TorchEBM keeps them orthogonal, so any loss that consumes pairs composes with any of either.

## Interpolants: the path

An interpolant defines

[ x_t = \\alpha(t), x_1 + \\sigma(t), x_0, \\qquad u_t = \\dot{\\alpha}(t), x_1 + \\dot{\\sigma}(t), x_0, ]

with (x_0 \\sim \\mathcal{N}(0, I)) and (x_1 \\sim p\_{\\text{data}}); the conditional velocity (u_t) is the regression target of flow matching, and the same coefficients convert between velocity, score, and noise predictions.

Common schedules illustrate the design space: a linear path ((\\alpha = t), (\\sigma = 1 - t)) gives the straight lines of flow matching; a cosine path smooths the endpoints; a variance-preserving schedule reproduces the diffusion setting where score and noise prediction are natural. The shipped schedules, generated from the installed package:

```
graph TD
    BaseInterpolant(["BaseInterpolant"])
    BaseInterpolant --> CosineInterpolant
    BaseInterpolant --> LinearInterpolant
    BaseInterpolant --> VariancePreservingInterpolant
```

```python
from torchebm.interpolants import LinearInterpolant
xt, ut = LinearInterpolant().interpolate(x0, x1, t)   # path point + target velocity
```

Beyond `interpolate`, the base class carries the conversion algebra (`velocity_to_score`, `score_to_velocity`, `velocity_to_noise`, `compute_drift`, `compute_diffusion`), which is what lets `FlowSampler` accept any prediction type over any path.

## Couplings: the pairing

Under an independent pairing, conditional targets are noisy: many ((x_0, x_1)) pairs cross, and the marginal velocity field the model must average over is curved. A coupling re-pairs each batch before interpolation. The families range from keeping the incoming pairing (independent), through cost-based pairings of increasing fidelity (greedy nearest-neighbor, entropic Sinkhorn, exact OT, and unbalanced variants that emit per-pair weights), to model-induced couplings that pair (x_0) with the model's own output (e.g. reflow). The shipped set, generated from the installed package:

```
graph TD
    BaseCostCoupling(["BaseCostCoupling"])
    BaseCoupling(["BaseCoupling"])
    BaseModelCoupling(["BaseModelCoupling"])
    BaseCostCoupling --> ExactOTCoupling
    BaseCoupling --> BaseCostCoupling
    BaseCostCoupling --> GreedyCoupling
    BaseCoupling --> IndependentCoupling
    BaseModelCoupling --> ReflowCoupling
    BaseCoupling --> BaseModelCoupling
    BaseCostCoupling --> SinkhornCoupling
    BaseCostCoupling --> UnbalancedSinkhornCoupling
```

```python
from torchebm.couplings import SinkhornCoupling
x0, x1 = SinkhornCoupling(reg=0.05)(x0, x1)   # CouplingResult unpacks as the pair
```

Results are `CouplingResult` objects: they unpack as `(x0, x1)`, and extras ride as attributes so the contract never breaks. Today the extra is `weights`, per-pair importance weights produced by unbalanced OT; weight-aware losses such as `EnergyMatchingLoss` consume them automatically. Couplings are also addressable by registry string via `get_coupling`.

Minibatch OT couplings straighten the conditional paths[1](#fn:tong), which lowers the variance of the regression and cuts the integration steps generation needs[2](#fn:liu); `ReflowCoupling` is the model-induced version of the same idea, re-pairing noise with the model's current outputs to iteratively rectify the flow.

## How losses consume them

Transport losses take both objects as constructor arguments and hide the mechanics:

```python
EnergyMatchingLoss(model=potential, coupling=SinkhornCoupling(reg=0.01), ...)
EquilibriumMatchingLoss(model=field, interpolant="linear", ...)
```

Internally each training step is: couple the batch, draw (t), interpolate, regress. Everything upstream (which energy, which network) and downstream (which sampler, which integrator) is untouched by the choice.

## Runnable counterparts

- [Interpolant Anatomy](https://soran-ghaderi.github.io/torchebm/latest/examples/00-foundations/04-interpolants/01-interpolant-anatomy/index.md)
- [Coupling Comparison](https://soran-ghaderi.github.io/torchebm/latest/examples/20-training/05-couplings/01-coupling-comparison/index.md)
- [Energy Matching in 2D](https://soran-ghaderi.github.io/torchebm/latest/examples/20-training/04-energy-matching/01-energy-matching-2d/index.md)

______________________________________________________________________

1. A. Tong et al. Improving and generalizing flow-based generative models with minibatch optimal transport. *TMLR*, 2024. [↩](#fnref:tong "Jump back to footnote 1 in the text")
1. X. Liu, C. Gong, and Q. Liu. Flow straight and fast: learning to generate and transfer data with rectified flow. *ICLR*, 2023. [↩](#fnref:liu "Jump back to footnote 2 in the text")
