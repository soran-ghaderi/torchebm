r"""Triton fused Langevin kernels (proof of concept for `torchebm.cuda`).

Two fusion levels for the Langevin update

\[
x_{t+1} = x_t - \eta \nabla_x U(x_t) + \sigma\sqrt{2\eta}\,\epsilon_t
\]

1. `fused_langevin_step`: model-agnostic. The gradient comes from any
   `BaseModel.gradient` call; noise generation (in-kernel Philox),
   the update arithmetic, and the optional clamp fuse into ONE kernel.
   Replaces the ~6 kernels + 1 allocation per step of the eager path
   (`randn_like`, `sqrt`, einsum-combine, mul, add, `clamp_`).

2. `doublewell_langevin_chain`: model-specific ceiling. For elementwise
   energies (DoubleWell, Harmonic) the analytic gradient is local, so the
   ENTIRE n-step chain runs register-resident in a single kernel launch:
   no autograd, no per-step launches, no intermediate memory traffic.

Run the built-in correctness check + benchmark:

    python -m torchebm.cuda.fused_langevin
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _fused_langevin_step_kernel(
    x_ptr,
    grad_ptr,
    out_ptr,
    noise_ptr,
    n_elements,
    step_size,
    noise_coef,  # noise_scale * sqrt(2 * step_size)
    seed,
    clamp_min,
    clamp_max,
    HAS_CLAMP: tl.constexpr,
    NOISE_FROM_PTR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    g = tl.load(grad_ptr + offs, mask=mask)
    if NOISE_FROM_PTR:
        eps = tl.load(noise_ptr + offs, mask=mask)
    else:
        eps = tl.randn(seed, offs)  # Philox, no global-memory round trip
    x = x - step_size * g + noise_coef * eps
    if HAS_CLAMP:
        x = tl.minimum(tl.maximum(x, clamp_min), clamp_max)
    tl.store(out_ptr + offs, x, mask=mask)


@triton.jit
def _doublewell_langevin_chain_kernel(
    x_ptr,
    n_elements,
    barrier_height,
    b_sq,
    step_size,
    noise_coef,
    seed,
    n_steps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    for i in range(n_steps):
        # Analytic elementwise gradient of E(x) = h * sum((x^2 - b^2)^2):
        # dE/dx_j = 4 h x_j (x_j^2 - b^2). No autograd, no memory traffic.
        g = 4.0 * barrier_height * x * (x * x - b_sq)
        eps = tl.randn(seed + i, offs)
        x = x - step_size * g + noise_coef * eps
    tl.store(x_ptr + offs, x, mask=mask)


def fused_langevin_step(
    x: torch.Tensor,
    grad: torch.Tensor,
    step_size: float,
    noise_scale: float = 1.0,
    clamp: Optional[Tuple[float, float]] = None,
    noise: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""One fused Langevin update: RNG + update + clamp in a single kernel.

    Args:
        x: Current state, contiguous CUDA float32 tensor of any shape.
        grad: \(\nabla_x U(x)\), same shape as `x` (from `model.gradient`).
        step_size: Step size \(\eta\).
        noise_scale: Noise scale \(\sigma\).
        clamp: Optional `(min, max)` bounds applied after the update.
        noise: Optional pre-sampled standard normal noise. When given, it is
            loaded instead of in-kernel RNG (exact-parity testing).
        seed: Philox seed for in-kernel RNG. Drawn from torch RNG when `None`.
        out: Output tensor. Pass `x` for an in-place update. New tensor
            when `None`.

    Returns:
        Updated state tensor.
    """
    assert x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
    if out is None:
        out = torch.empty_like(x)
    n = x.numel()
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    has_clamp = clamp is not None
    cmin, cmax = clamp if has_clamp else (0.0, 0.0)
    grid = (triton.cdiv(n, _BLOCK),)
    _fused_langevin_step_kernel[grid](
        x, grad, out,
        noise if noise is not None else x,  # dummy ptr when unused
        n,
        step_size,
        noise_scale * (2.0 * step_size) ** 0.5,
        seed,
        cmin, cmax,
        HAS_CLAMP=has_clamp,
        NOISE_FROM_PTR=noise is not None,
        BLOCK=_BLOCK,
    )
    return out


def doublewell_langevin_chain(
    x: torch.Tensor,
    n_steps: int,
    step_size: float,
    noise_scale: float = 1.0,
    barrier_height: float = 2.0,
    b: float = 1.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    r"""Run a full Langevin chain on the double-well energy in ONE kernel launch.

    In-place on `x`. Each element's chain is independent (the energy is
    elementwise-separable), so the whole trajectory stays in registers.

    Args:
        x: Initial state, contiguous CUDA float32 tensor, modified in place.
        n_steps: Number of Langevin steps.
        step_size: Step size \(\eta\).
        noise_scale: Noise scale \(\sigma\).
        barrier_height: Double-well barrier height \(h\).
        b: Well position (\(\pm b\)).
        seed: Base Philox seed (incremented per step).

    Returns:
        `x`, after `n_steps` in-place updates.
    """
    assert x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
    n = x.numel()
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    grid = (triton.cdiv(n, _BLOCK),)
    _doublewell_langevin_chain_kernel[grid](
        x, n,
        barrier_height, b * b,
        step_size,
        noise_scale * (2.0 * step_size) ** 0.5,
        seed, n_steps,
        BLOCK=_BLOCK,
    )
    return x


def _bench(fn, n_iters: int = 10) -> float:
    """Median wall time (ms) over `n_iters` after 3 warmup calls."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _main():
    from torchebm.core import DoubleWellModel
    from torchebm.samplers import LangevinDynamics

    device = torch.device("cuda")
    n_samples, dim, n_steps = 4096, 32, 1000
    step_size, noise_scale, bh, b = 0.01, 1.0, 2.0, 1.0
    shape = (n_samples, dim)

    # Correctness: fused step vs eager update with identical noise.
    x = torch.randn(shape, device=device)
    g = 4.0 * bh * x * (x * x - b * b)
    eps = torch.randn_like(x)
    ref = x - step_size * g + noise_scale * (2 * step_size) ** 0.5 * eps
    got = fused_langevin_step(x, g, step_size, noise_scale, noise=eps)
    err = (got - ref).abs().max().item()
    print(f"fused step vs eager (same noise): max abs err = {err:.2e}")
    assert err < 1e-6

    x0 = torch.randn(shape, device=device)

    def run_library():
        model = DoubleWellModel(barrier_height=bh, b=b, device=device)
        sampler = LangevinDynamics(model, step_size, noise_scale, device=device)
        return sampler.sample(x=x0.clone(), n_steps=n_steps)

    def run_eager_analytic():
        x = x0.clone()
        coef = noise_scale * (2 * step_size) ** 0.5
        for _ in range(n_steps):
            g = 4.0 * bh * x * (x * x - b * b)
            x = x - step_size * g + coef * torch.randn_like(x)
        return x

    def run_fused_step():
        x = x0.clone()
        for i in range(n_steps):
            g = 4.0 * bh * x * (x * x - b * b)
            fused_langevin_step(x, g, step_size, noise_scale, seed=i, out=x)
        return x

    def run_fused_chain():
        return doublewell_langevin_chain(
            x0.clone(), n_steps, step_size, noise_scale, bh, b, seed=0
        )

    # Stationary-distribution sanity: chains should concentrate near +/- b.
    mean_abs = run_fused_chain().abs().mean().item()
    ref_abs = run_eager_analytic().abs().mean().item()
    print(f"E|x| fused chain = {mean_abs:.3f}, eager = {ref_abs:.3f} (wells at {b})")

    print(f"\nbenchmark: {n_samples}x{dim}, {n_steps} steps, median of 10")
    t_lib = _bench(run_library)
    t_eager = _bench(run_eager_analytic)
    t_step = _bench(run_fused_step)
    t_chain = _bench(run_fused_chain)
    print(f"  LangevinDynamics.sample (autograd grad) : {t_lib:9.2f} ms   1.0x")
    print(f"  eager loop, analytic grad               : {t_eager:9.2f} ms  {t_lib / t_eager:4.1f}x")
    print(f"  analytic grad + fused step kernel       : {t_step:9.2f} ms  {t_lib / t_step:4.1f}x")
    print(f"  whole chain in one kernel               : {t_chain:9.2f} ms  {t_lib / t_chain:4.1f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    _main()
