TorchEBM🍓

TorchEBM: Simulation-free, GPU-first generative modeling in PyTorch\
Composable primitives for scalable, stable training of **EBMs**, **diffusion**, **flow matching**, and **equilibrium and energy matching**.

[Get Started](https://soran-ghaderi.github.io/torchebm/latest/getting_started/index.md)

______________________________________________________________________

## Overview

A PyTorch library of composable, GPU-first primitives for energy-based models, diffusion, flow matching, and equilibrium and energy matching. Every method is a composition of the same parts: models, paths, couplings, objectives, samplers and integrators. See [Design and Scope](https://soran-ghaderi.github.io/torchebm/latest/concepts/design/index.md).

______________________________________________________________________

## In Action

Eight-gaussians distribution

Circles distribution

Equilibrium matching with different interpolants transporting noise onto structured distributions.

______________________________________________________________________

## Core Components

- [**Core**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/core/index.md)
- [**Samplers**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/samplers/index.md)
- [**Losses**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/losses/index.md)
- [**Interpolants**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/interpolants/index.md)
- [**Couplings**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/couplings/index.md)
- [**Integrators**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/integrators/index.md)
- [**Models**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/models/index.md)
- [**Datasets**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/datasets/index.md)
- [**Utils**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/utils/index.md)
- [**CUDA**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/cuda/index.md)

______________________________________________________________________

## Quick Start

```bash
pip install torchebm
```

Train **EqM-XL/2** with [equilibrium matching](https://arxiv.org/abs/2510.02300) on ImageNet 256, as in the paper:

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

More in the [concepts](https://soran-ghaderi.github.io/torchebm/latest/concepts/index.md) and [examples](https://soran-ghaderi.github.io/torchebm/latest/examples/index.md).

**Enjoying TorchEBM?** A GitHub star helps others discover the project and motivates continued development.

[Star on GitHub](https://github.com/soran-ghaderi/torchebm)

______________________________________________________________________

## Built with TorchEBM

2026 Mixed-Field Matching Preprint soon

______________________________________________________________________

## Citation

If TorchEBM is useful in your research, please cite it:

```bibtex
@misc{ghaderi2024torchebm,
  author       = {Ghaderi, Soran},
  title        = {{TorchEBM}: A Composable {PyTorch} Library for Energy-Based and Transport-Based Generative Modeling},
  year         = {2024},
  howpublished = {\url{https://github.com/soran-ghaderi/torchebm}},
}
```

MIT License · [Issues](https://github.com/soran-ghaderi/torchebm/issues) · [Contributing](https://soran-ghaderi.github.io/torchebm/latest/developer_guide/index.md) · [GitHub](https://github.com/soran-ghaderi/torchebm)
