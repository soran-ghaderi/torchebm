r"""FSDP2 training-step validation benchmark for multi-GPU NCCL runs.

Times full training steps (negatives/noise, loss forward, backward, optimizer)
for contrastive divergence and functional denoising score matching with the
energy net sharded via ``fully_shard``, at a model size where collective
ordering, memory, and dtype behavior differ from the CPU/gloo test suite.

Launch with torchrun, one process per GPU:

```bash
torchrun --standalone --nproc-per-node=4 benchmarks/distributed_fsdp2.py \
    --objective cd --steps 20
torchrun --standalone --nproc-per-node=4 benchmarks/distributed_fsdp2.py \
    --objective dsm --steps 20
torchrun --standalone --nproc-per-node=4 benchmarks/distributed_fsdp2.py \
    --objective cd --bf16 --steps 20
```

Notes:

- The bf16 ``MixedPrecisionPolicy`` affects the hook path (CD) only; the
  functional score path reads the fp32 sharded parameters directly, so DSM
  compute stays fp32 regardless of the policy.
- Exit code is non-zero if any step produces a non-finite loss.
"""

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from torchebm.core import BaseModel
from torchebm.losses import ContrastiveDivergence, DenoisingScoreMatching
from torchebm.samplers import LangevinDynamics


class MLPEnergy(BaseModel):
    def __init__(self, dim: int, hidden: int, layers: int):
        super().__init__()
        blocks = [nn.Linear(dim, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            blocks += [nn.Linear(hidden, hidden), nn.SiLU()]
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def shard(model: MLPEnergy, mesh, policy=None):
    kwargs = {"mesh": mesh}
    if policy is not None:
        kwargs["mp_policy"] = policy
    for m in model.net:
        if isinstance(m, nn.Linear):
            fully_shard(m, **kwargs)
    fully_shard(model.net, **kwargs)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=("cd", "dsm"), default="cd")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch", type=int, default=256, help="per-rank batch")
    parser.add_argument("--k-steps", type=int, default=10, help="cd sampler steps")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (world,))

    torch.manual_seed(args.seed)
    model = MLPEnergy(args.dim, args.hidden, args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16) if args.bf16 else None
    shard(model, mesh, policy)

    if args.objective == "cd":
        loss_fn = ContrastiveDivergence(
            model=model,
            sampler=LangevinDynamics(model=model, step_size=0.01),
            k_steps=args.k_steps,
            persistent=False,
            device=device,
        )
    else:
        loss_fn = DenoisingScoreMatching(
            model=model,
            noise_scale=0.1,
            use_autograd=False,
            functional_model=MLPEnergy(args.dim, args.hidden, args.layers),
        )

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed * 1000 + rank)

    times, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.warmup + args.steps):
        x = torch.randn(args.batch, args.dim, device=device, generator=generator)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if args.objective == "cd":
            loss, _ = loss_fn(x, generator=generator)
        else:
            loss = loss_fn(x)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        if step >= args.warmup:
            times.append((time.perf_counter() - t0) * 1000)
            losses.append(loss.detach())

    finite = bool(torch.isfinite(torch.stack(losses)).all())
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    dist.barrier()
    if rank == 0:
        print(
            f"objective={args.objective} world={world} params={n_params / 1e6:.1f}M "
            f"batch/rank={args.batch} bf16={args.bf16} steps={args.steps} | "
            f"step_ms mean={statistics.mean(times):.1f} "
            f"median={statistics.median(times):.1f} min={min(times):.1f} | "
            f"peak_mem_gb={peak_gb:.2f} finite={finite} "
            f"final_loss={losses[-1].item():.4f}"
        )
    dist.destroy_process_group()
    raise SystemExit(0 if finite else 1)


if __name__ == "__main__":
    main()
