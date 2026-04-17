---
sidebar_position: 8
title: Performance
description: Writing fast TorchEBM code. vectorization, memory, and GPU patterns
icon: material/speedometer
---

# Performance

Patterns we follow when writing performance-sensitive code. For *measuring* performance, see [Benchmarking](benchmarking.md); for *debugging* a specific regression, see [Profiling](profiling.md).

---

## The four hot spots

1. **Sampler steps**: iterative, run 100–1000×, dominate wall time.
2. **Score / energy gradients**: `autograd.grad` calls are frequent and stack.
3. **Loss forward + backward**: called every training batch.
4. **Host ↔ device traffic**: `.item()`, `.cpu()`, repeated `.to()` stall the GPU.

Optimise in that order. Everything else is noise.

---

## Vectorise, don't loop

Work in batch dimensions; avoid Python-level iteration over samples.

```python
# bad
energies = torch.stack([energy_fn(x[i]) for i in range(batch)])

# good
energies = energy_fn(x)                       # (batch,)
```

Sample many chains in parallel by putting the chain index in the leading dim:

```python
x = torch.randn(n_chains, dim, device=device)
x, _ = sampler.step(x)                        # all chains advance together
```

---

## Stay on device

Keep tensors on the same device and dtype for the whole pipeline. Use `DeviceMixin`'s `self.device` / `self.dtype` inside the library; never hard-code `cuda`.

!!! warning "Do not sync unnecessarily"
    `.item()`, `.cpu()`, `.tolist()`, and Python `if tensor > 0:` all trigger a full GPU sync. Defer them until after the hot loop, ideally until logging at the epoch boundary.

```python
# bad: syncs every step
for step in range(n_steps):
    loss = loss_fn(x)
    if loss.item() < threshold:               # GPU stall
        break

# good: one sync per epoch
losses = []
for step in range(n_steps):
    losses.append(loss_fn(x).detach())
avg = torch.stack(losses).mean().item()       # single sync
```

---

## Reuse memory

Pre-allocate buffers once, reuse in the loop:

```python
# bad: allocates every call
def drift(x, t):
    t_batch = torch.ones(x.shape[0], device=x.device) * t

# good: allocate once, fill in place
t_batch = torch.empty(batch, device=device)
for step in range(n_steps):
    t_batch.fill_(t_value)
```

For trajectories, write into a pre-allocated tensor instead of appending to a list and stacking at the end:

```python
traj = torch.empty(n_steps + 1, *x.shape, device=device, dtype=x.dtype)
traj[0] = x
for i in range(n_steps):
    x = sampler.step(x)
    traj[i + 1] = x
```

In-place ops (`x.add_`, `x.mul_`) are safe outside of autograd-tracked paths.

---

## Mixed precision and compilation

Both are opt-in at the benchmark / application layer via the same entry point the profiler uses:

```python
from benchmarks.registry import apply_mode
fn = apply_mode(fn, mode="amp",     device=device)   # float16 autocast
fn = apply_mode(fn, mode="compile", device=device)   # torch.compile
```

Inside the library, wrap large matmul blocks with `self.autocast_context()` (provided by `DeviceMixin`) rather than calling `torch.autocast` directly. this honours the user's configured dtype.

---

## Sampler-specific tips

=== "Langevin dynamics"
    Rough scaling: step size \( \sim d^{-1/3} \), noise scale \( \sigma = \sqrt{2\eta} \).

    ```python
    step_size = min(0.01, 0.1 * dim ** (-1 / 3))
    noise_scale = (2 * step_size) ** 0.5
    ```

=== "HMC"
    Rough scaling: leapfrog steps \( \sim \sqrt{d} \), step size \( \sim d^{-1/4} \). Target acceptance 0.6–0.8.

    ```python
    n_leapfrog = max(5, int(dim ** 0.5))
    step_size  = min(0.01, 0.05 * dim ** (-1 / 4))
    ```

=== "Flow / diffusion"
    Prefer adaptive integrators (`DOPRI5`, `Heun`) for generation; fixed-step for training. Keep the ODE function allocation-free. see the pre-allocation pattern above.

---

## Common pitfalls

- **Implicit host ↔ device copies**: `torch.tensor(x_numpy, device=…)` inside a loop.
- **Redundant `.to()` calls**: `BaseLoss.__call__` already moves inputs; subclass `forward()` should not move them again.
- **Missing `torch.no_grad()`**: interpolation targets, momentum init, and random projection generation don't need grad tracking.
- **Tiny batches on GPU**: under-utilises SMs; prefer one big step over many small ones.
- **Python-level `isinstance` inside the inner loop**: resolve once before the loop.

---

## Next steps

- Evidence first: [Benchmarking](benchmarking.md) tells you *how fast*; [Profiling](profiling.md) tells you *where the time goes*.
- CUDA kernels: planned. See [cuRBLAS](https://github.com/soran-ghaderi/cuRBLAS) for background.
- Multi-GPU scaling: planned.
