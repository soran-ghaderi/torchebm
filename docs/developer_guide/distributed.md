---
title: Distributed Training
description: FSDP2 and DTensor recipes for multi-GPU training with TorchEBM components
icon: material/server-network
---

# Distributed Training

TorchEBM leaves parallelism to the standard PyTorch wrappers: components
never require an initialized process group, and no default `forward()` path
of a model, loss or sampler issues a collective. You wrap the energy network,
use your own launcher and data sharding, and the components behave
identically. Collectives exist only
behind explicit opt-ins (`process_group=` arguments and explicit methods)
where the math is batch-global.

FSDP2 (`fully_shard` with DTensor parameters) is the primary target and the
path validated on multi-GPU NCCL hardware. DDP also runs, but drives no
design decisions.

## The shard-inside pattern

Shard the network *inside* your `BaseModel` subclass and keep the model
object as the user-facing type. Samplers and losses stay unaware of sharding;
every first-order path (energy calls, Langevin and HMC chains, CD, implicit
EqM) runs through FSDP2's hooks unchanged.

```python
import torch
from torch import nn
from torch.distributed.fsdp import fully_shard
from torchebm.core import BaseModel

class EnergyMLP(BaseModel):
    def __init__(self, dim, hidden, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = EnergyMLP(dim=512, hidden=8192).cuda()
for m in model.net:
    if isinstance(m, nn.Linear):
        fully_shard(m)
fully_shard(model.net)
```

The root call turns the parameters into sharded DTensors. Never wrap
samplers or losses: losses see only model outputs and samplers call the
model you pass them. `TorchEBMModule` resolves `model.device` and
`model.dtype` correctly from DTensor parameters.

## Training loop (torchrun)

Persistent CD on a `DistributedSampler`-sharded loader, launched with
`torchrun --standalone --nproc-per-node=4 train.py`; the model is built
shard-inside as above:

```python
import os
import torch
import torch.distributed as dist
from torchebm.losses import ContrastiveDivergenceLoss
from torchebm.samplers import LangevinDynamics
from torchebm.utils import ReplayBuffer

rank = int(os.environ["RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
device = torch.device("cuda", torch.cuda.current_device())

model = build_sharded_energy()
sampler = LangevinDynamics(step_size=0.01)
loss_fn = ContrastiveDivergenceLoss()
buffer = ReplayBuffer(capacity=8192, sample_shape=(512,), device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
generator = torch.Generator(device=device).manual_seed(base_seed + rank)

for x in loader:
    x = x.to(device)
    x0 = buffer.sample(len(x), generator=generator)
    negatives = sampler(model, x0, n_steps=10, generator=generator)
    buffer.add(negatives)
    loss = loss_fn(model(x), model(negatives))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

Three library-specific points, everything else is a standard FSDP2 loop:

1. **Pass a per-rank `generator`.** The library holds no hidden RNG state;
   with a shared seed every rank draws identical start points and runs
   identical chains, and you pay for \(N\) GPUs to get one GPU of sample
   diversity. Generators are device-bound: create them on the compute
   device.
2. **Reduction semantics.** Losses reduce with means over the local batch;
   with gradient averaging this equals the global-batch mean exactly when
   per-rank batch sizes are equal (keep `drop_last=True`).
3. **k-step MCMC and resharding.** With default reshard-after-forward, every
   Langevin step re-gathers the parameters. For sampling-only loops (not
   inside a training forward/backward), hold them gathered:

    ```python
    from torchebm.utils.distributed import unsharded

    with unsharded(model.net):
        samples = sampler(model, x0, n_steps=500, generator=generator)
    ```

## Training through the score

FSDP2 hooks cannot run the double backward that training through the score
needs (score matching, explicit EqM energies, energy matching): the
post-backward hook reshards parameter storage that the second-order graph
still references, independent of `reshard_after_forward`. A
`prediction_type="score"` or `("energy", "score")` model with DTensor
parameters therefore takes a hook-free functional path, selected by giving it
a template at construction:

```python
from torchebm.losses import DenoisingScoreMatchingLoss

with torch.device("meta"):
    template = EnergyMLP(dim, hidden)
model = EnergyMLP(dim, hidden, prediction_type="score", functional_model=template)
shard(model)

loss_fn = DenoisingScoreMatchingLoss()
noise = torch.randn(x.shape, device=device, generator=generator)
loss = loss_fn(model(x + 0.1 * noise), -noise / 0.1)
```

The template supplies the module structure for `torch.func.functional_call`
and its own parameters are never used, so it lives on the meta device. The
sharded DTensor parameters are injected per call and the double backward
runs through DTensor's differentiable collectives. Gradients match an
unsharded reference with global-batch-mean semantics and land with the
parameter's own placement (the per-rank contributions are reduce-scattered at
accumulation time), so standard optimizers step them directly. Without a
template, a score call under grad mode fails fast with an actionable error
when it sees DTensor parameters. Energy matching and EqM-E train through the
same call, so they shard the same way.

One dtype caveat: a bf16 `MixedPrecisionPolicy` affects the hook path only.
The functional path reads the fp32 sharded parameters directly, so its
compute stays fp32 regardless of the policy.

Under DDP the hook path handles the double backward; build the energy head
without an output bias (the score does not depend on it) so every parameter
receives a gradient with the default `static_graph=False`.

## PCD replay buffers across ranks

Persistent CD buffers are **rank-local by design**: each rank's
`ReplayBuffer` keeps independent chains, so the world size multiplies chain
diversity at zero cost, and no collective touches the buffer in the training
step. To occasionally exchange chains between ranks, call the explicit
collective between steps, on all ranks together:

```python
buffer.mix_across_ranks()
```

It gathers the pooled chains, applies one shared permutation (drawn on rank
0 and broadcast, so per-rank generators cannot desynchronize it), and keeps
the local shard: no chain is duplicated or lost.

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
keeps its own targets, paired with sources from any rank; the per-target draw
is broadcast from rank 0. Budget
for the \((\text{world\_size} \times \text{batch})^2\) cost matrix on every
rank. Exact and greedy couplings reject a process group (assignment solvers
scale quadratically with the pooled batch); `couple` becomes a collective all
ranks must enter together.

## EMA

`update_ema` works on sharded parameters when the EMA model and the training
model are sharded identically (same mesh, same policy); each local shard
updates in place with no collective. Copy before sharding, then shard both
the same way:

```python
ema = copy.deepcopy(model)
shard(model); shard(ema)
update_ema(ema, model, decay=0.999)
```

## Checkpointing

Sharded state takes the distributed checkpoint (one collective save); the
replay buffer is rank-local state (PCD chains differ per rank) and goes to
one plain file per rank:

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, set_model_state_dict,
)

dcp.save({"model": get_model_state_dict(model)}, checkpoint_id=ckpt_dir)
torch.save(buffer.state_dict(), f"{ckpt_dir}/buffer_rank{rank}.pt")

state = {"model": get_model_state_dict(model)}
dcp.load(state, checkpoint_id=ckpt_dir)
set_model_state_dict(model, state["model"])
buffer.load_state_dict(torch.load(f"{ckpt_dir}/buffer_rank{rank}.pt"))
```

Do not put rank-local buffers into the DCP state dict: its planner
deduplicates non-sharded tensors as replicated and silently keeps rank 0's
chains. Restore rank-local files with the same world size, or re-initialize
the buffer from noise (chains re-warm within a few hundred steps).

## Large-scale recipe

For models too large to materialize on one device, compose the standard
FSDP2 ingredients; every piece is validated with TorchEBM components:

```python
with torch.device("meta"):
    model = EnergyMLP(dim, hidden)
policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
for m in model.net:
    if isinstance(m, nn.Linear):
        fully_shard(m, mp_policy=policy)
fully_shard(model.net, mp_policy=policy)
model.to_empty(device="cuda")
torch.manual_seed(seed)
for m in model.net:
    if isinstance(m, nn.Linear):
        m.reset_parameters()
```

The meta-device construction allocates no memory, `to_empty` materializes
the local shards only, and the shared seed gives the same initialization on
every rank. With a bf16 policy the score follows the input's dtype; set
`model.force_fp32_gradient = True` if a low-precision model needs
fp32-precision sampler gradients. Combine with the DCP checkpointing above.

## Other frameworks

The shard-inside pattern is framework-agnostic: anything that wraps an
`nn.Module` composes the same way.

- **DDP**: `model.net = DistributedDataParallel(model.net)`. Everything runs,
  including training through the score, through the hooks.
- **DeepSpeed ZeRO-3 / HF Accelerate / Lightning Fabric**: wrap or `prepare`
  the inner network, keep the `BaseModel` facade outside. The FSDP2 caveat
  carries over wherever parameters are partitioned: training through the
  score needs the functional path, which supports DTensor only, or a
  replica-style engine (ZeRO-1/2, DDP).
- **Tensor parallel (DTensor)**: the functional score path requires a 1-D
  device mesh; TP meshes are untested territory.

## Validation status

The distributed test suite (`pytest tests/distributed`) runs 2-process
gloo/CPU in CI and validates the contract; the same suite runs under NCCL
with one process per GPU via `TORCHEBM_DIST_DEVICE=cuda`, and
`benchmarks/distributed_fsdp2.py` exercises full CD and functional-DSM
training steps at hundred-million-parameter scale on multi-GPU hardware.
