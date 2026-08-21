---
title: Distributed Training
description: FSDP2 and DTensor recipes for multi-GPU training with TorchEBM components
icon: material/server-network
---

# Distributed Training

TorchEBM is distributed-transparent rather than distributed-aware: components
never require an initialized process group, no default `forward()` or
`sample()` path issues a collective, and the standard PyTorch wrappers do the
parallelism. You wrap the energy network, use your own launcher and data
sharding, and the components behave identically. Collectives exist only
behind explicit opt-ins (`process_group=` arguments and explicit methods)
where the math is batch-global.

FSDP2 (`fully_shard` with DTensor parameters) is the primary target and the
path validated on multi-GPU NCCL hardware. DDP also runs, but drives no
design decisions.

## The shard-inside pattern

Shard the network *inside* your `BaseModel` subclass and keep the model
object as the user-facing type. Samplers and losses stay unaware of sharding;
every first-order path (`gradient()`, Langevin and HMC chains, CD, implicit
EqM) runs through FSDP2's hooks unchanged.

```python
import torch
from torch import nn
from torch.distributed.fsdp import fully_shard
from torchebm.core import BaseModel

class MLPEnergy(BaseModel):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = MLPEnergy(dim=512, hidden=8192).cuda()
for m in model.net:
    if isinstance(m, nn.Linear):
        fully_shard(m)
fully_shard(model.net)          # root call; parameters become sharded DTensors
```

Never wrap samplers or losses; they hold the model and route through its
`forward`. `TorchEBMModule` resolves `model.device` and `model.dtype`
correctly from DTensor parameters.

## Training loop (torchrun)

```python
# torchrun --standalone --nproc-per-node=4 train.py
import os
import torch
import torch.distributed as dist
from torchebm.losses import ContrastiveDivergence
from torchebm.samplers import LangevinDynamics

rank = int(os.environ["RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")

model = build_sharded_energy()                     # shard-inside, as above
device = torch.device("cuda", torch.cuda.current_device())
loss_fn = ContrastiveDivergence(
    model=model,
    sampler=LangevinDynamics(model=model, step_size=0.01),
    k_steps=10,
    persistent=True,
    device=device,
)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# per-rank generator: decorrelates MCMC chains and noise across ranks
generator = torch.Generator(device=device).manual_seed(base_seed + rank)

for x in loader:                                   # DistributedSampler-sharded
    opt.zero_grad(set_to_none=True)
    loss, _ = loss_fn(x.to(device), generator=generator)
    loss.backward()
    opt.step()
```

Three library-specific points, everything else is a standard FSDP2 loop:

1. **Pass a per-rank `generator`.** The library holds no hidden RNG state;
   with a shared seed every rank runs identical chains and you pay for
   \(N\) GPUs to get one GPU of sample diversity. Generators are
   device-bound: create them on the compute device.
2. **Reduction semantics.** Losses reduce with means over the local batch;
   with gradient averaging this equals the global-batch mean exactly when
   per-rank batch sizes are equal (keep `drop_last=True`).
3. **k-step MCMC and resharding.** With default reshard-after-forward, every
   Langevin step re-gathers the parameters. For sampling-only loops (not
   inside a training forward/backward), hold them gathered:

    ```python
    from torchebm.utils.distributed import unsharded

    with unsharded(model.net):
        samples = sampler.sample(x=x0, n_steps=500, generator=generator)
    ```

## Score matching: the functional path

FSDP2 hooks cannot run the double backward score matching needs: the
post-backward hook reshards parameter storage that the second-order graph
still references, independent of `reshard_after_forward`. Score-matching
losses therefore take a hook-free functional path under sharding, selected at
construction:

```python
from torchebm.losses import DenoisingScoreMatching

loss_fn = DenoisingScoreMatching(
    model=sharded_model,
    noise_scale=0.1,
    use_autograd=False,                 # functional path
    functional_model=MLPEnergy(dim, hidden),   # unwrapped template, never trained
)
```

The template supplies the module structure for `torch.func.functional_call`;
the sharded DTensor parameters are injected per call and the double backward
runs through DTensor's differentiable collectives. Gradients match an
unsharded reference with global-batch-mean semantics and land with the
parameter's own placement (the loss reduce-scatters the per-rank
contributions at accumulation time), so standard optimizers step them
directly. The default autograd path fails fast with an actionable error when
it sees DTensor parameters.

One dtype caveat: a bf16 `MixedPrecisionPolicy` affects the hook path only.
The functional path reads the fp32 sharded parameters directly, so its
compute stays fp32 regardless of the policy.

## Objectives that cannot shard (yet)

**Energy Matching training** and **explicit EqM energies**
(`energy_type='dot'/'l2'/'mean'`) backpropagate through an input-gradient
built with `create_graph=True`, the same second-order pattern that breaks
under FSDP2 hooks, and no functional rewrite exists for them yet. In training
mode with DTensor parameters they raise immediately with the alternatives:

- train with **DDP** (full replica per GPU; the pattern works there because
  the single `loss.backward()` fires the reducer once), or
- train unsharded and shard only for evaluation/sampling, which is
  first-order and works.

Implicit EqM (the default `energy_type='none'`) is a plain regression and
shards fine.

## PCD replay buffers across ranks

Persistent CD buffers are **rank-local by design**: each rank keeps
independent chains, so the world size multiplies chain diversity at zero
cost, and no collective touches the buffer in `forward`. To occasionally
exchange chains between ranks, call the explicit collective between steps:

```python
loss_fn.mix_buffer_across_ranks()      # all ranks must call together
```

It gathers the pooled chains, applies one shared permutation (broadcast from
rank 0, so per-rank generators cannot desynchronize it), and keeps the local
shard: no chain is duplicated or lost.

## Global-batch OT coupling

Minibatch OT couplings are biased toward the batch size. With a process
group, `SinkhornCoupling` solves on the pooled global batch instead, shrinking
that bias at fixed per-rank batch size:

```python
import torch.distributed as dist
from torchebm.couplings import SinkhornCoupling

coupling = SinkhornCoupling(reg=0.05, process_group=dist.group.WORLD)
x0, x1 = coupling(x0_local, x1_local, generator=generator)
```

Every rank gathers both batches, solves the identical pooled problem, and
keeps its own rows; the row-conditional draw is broadcast from rank 0. Budget
for the \((\text{world\_size} \times \text{batch})^2\) cost matrix on every
rank. Exact and greedy couplings reject a process group (assignment solvers
scale quadratically with the pooled batch); `couple` becomes a collective all
ranks must enter together.

## EMA

`update_ema` works on sharded parameters when the EMA model and the training
model are sharded identically (same mesh, same policy); each local shard
updates in place with no collective.

```python
ema = copy.deepcopy(model)      # before sharding, then shard both identically
shard(model); shard(ema)
update_ema(ema, model, decay=0.999)
```

## Checkpointing

Sharded state and rank-local state take different routes:

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, set_model_state_dict,
)

# sharded model: distributed checkpoint (one collective save)
dcp.save({"model": get_model_state_dict(model)}, checkpoint_id=ckpt_dir)

# rank-local loss state (PCD chains differ per rank): one plain file per rank
torch.save(
    {"replay_buffer": loss_fn.replay_buffer, "buffer_ptr": loss_fn.buffer_ptr},
    f"{ckpt_dir}/loss_rank{rank}.pt",
)

# restore
state = {"model": get_model_state_dict(model)}
dcp.load(state, checkpoint_id=ckpt_dir)
set_model_state_dict(model, state["model"])
loss_fn.initialize_buffer(data_shape)          # buffer must exist before load
loss_fn.load_state_dict(
    torch.load(f"{ckpt_dir}/loss_rank{rank}.pt"), strict=False
)
```

Do not put rank-local buffers into the DCP state dict: its planner
deduplicates non-sharded tensors as replicated and silently keeps rank 0's
chains. Restore rank-local files with the same world size, or simply
re-initialize the buffer from noise (chains re-warm within a few hundred
steps).

## Large-scale recipe

For models too large to materialize on one device, compose the standard
FSDP2 ingredients; every piece is validated with TorchEBM components:

```python
with torch.device("meta"):
    model = MLPEnergy(dim, hidden)              # no memory allocated
policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
for m in model.net:
    if isinstance(m, nn.Linear):
        fully_shard(m, mp_policy=policy)
fully_shard(model.net, mp_policy=policy)
model.to_empty(device="cuda")                   # materialize shards only
torch.manual_seed(seed)                         # same init on every rank
for m in model.net:
    if isinstance(m, nn.Linear):
        m.reset_parameters()
```

With a bf16 policy, `gradient()` returns gradients in the input's dtype; set
`model.force_fp32_gradient = True` if a low-precision model needs
fp32-precision sampler gradients. Combine with the DCP checkpointing above.

## Other frameworks

The shard-inside pattern is framework-agnostic: anything that wraps an
`nn.Module` composes the same way.

- **DDP**: `model.net = DistributedDataParallel(model.net)`. Everything runs,
  including the second-order objectives that FSDP2 rejects.
- **DeepSpeed ZeRO-3 / HF Accelerate / Lightning Fabric**: wrap or `prepare`
  the inner network, keep the `BaseModel` facade outside. The FSDP2 caveats
  carry over wherever parameters are partitioned: second-order objectives
  need a replica-style engine (ZeRO-1/2, DDP) and score matching needs the
  functional path under partitioning.
- **Tensor parallel (DTensor)**: the functional score path requires a 1-D
  device mesh; TP meshes are untested territory.

## Validation status

The distributed test suite (`pytest tests/distributed`) runs 2-process
gloo/CPU in CI and validates the contract; the same suite runs under NCCL
with one process per GPU via `TORCHEBM_DIST_DEVICE=cuda`, and
`benchmarks/distributed_fsdp2.py` exercises full CD and functional-DSM
training steps at hundred-million-parameter scale on multi-GPU hardware.
