r"""A/B benchmark: pre-allocated trajectory vs list-append + torch.stack.

Usage:
    python benchmarks/bench_trajectory_alloc.py [--device cuda] [--n-runs 50]
"""
import argparse
import time
import torch
from torchebm.core import DoubleWellEnergy
from torchebm.samplers import GradientDescentSampler, NesterovSampler


def _time_fn(fn, n_warmup=5, n_runs=50, device="cpu"):
    for _ in range(n_warmup):
        fn()
    if device != "cpu":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device != "cpu":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    t = torch.tensor(times)
    return t.mean().item(), t.std().item(), t.median().item()


def _sample_list_append(sampler, x0, n_steps, dim):
    """Old path: list-append + torch.stack."""
    x = x0.clone()
    sampler.reset_schedulers()
    trajectory = [x.clone()]
    with sampler.autocast_context():
        for _ in range(n_steps):
            sampler.step_schedulers()
            eta = sampler.get_scheduled_value("step_size")
            grad = sampler.model.gradient(x)
            x = x - eta * grad
            trajectory.append(x.clone())
    return torch.stack(trajectory, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-runs", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    configs = [
        {"batch_size": 64, "dim": 8, "n_steps": 50, "label": "small"},
        {"batch_size": 256, "dim": 32, "n_steps": 100, "label": "medium"},
        {"batch_size": 1024, "dim": 128, "n_steps": 200, "label": "large"},
    ]

    model = DoubleWellEnergy().to(device)
    sampler = GradientDescentSampler(model, step_size=0.01, device=device)

    print(f"Device: {device}")
    if device != "cpu":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Runs: {args.n_runs}\n")
    print(f"{'Scale':<10} {'Method':<25} {'Mean (ms)':>10} {'Std (ms)':>10} {'Median (ms)':>12}")
    print("-" * 70)

    for cfg in configs:
        bs, dim, n_steps = cfg["batch_size"], cfg["dim"], cfg["n_steps"]
        x0 = torch.randn(bs, dim, device=device, dtype=torch.float32)

        # Pre-allocated (current code)
        fn_new = lambda: sampler.sample(x=x0.clone(), n_steps=n_steps, return_trajectory=True)
        mean, std, med = _time_fn(fn_new, n_runs=args.n_runs, device=device)
        print(f"{cfg['label']:<10} {'pre-alloc':<25} {mean*1000:>10.3f} {std*1000:>10.3f} {med*1000:>12.3f}")

        # List-append (old code)
        fn_old = lambda: _sample_list_append(sampler, x0, n_steps, dim)
        mean2, std2, med2 = _time_fn(fn_old, n_runs=args.n_runs, device=device)
        print(f"{'':<10} {'list-append+stack':<25} {mean2*1000:>10.3f} {std2*1000:>10.3f} {med2*1000:>12.3f}")

        speedup = med2 / med if med > 0 else float("inf")
        print(f"{'':<10} {'speedup':<25} {'':<10} {'':<10} {speedup:>11.2f}x")
        print()


if __name__ == "__main__":
    main()
