# `torchebm.models`

Model namespace.

TorchEBM is designed for plug-and-play experimentation:

- try different losses with the same backbone
- try different backbones with the same loss
- use samplers as long as the model signature matches

This package therefore exposes *reusable building blocks* under `torchebm.models.components` and a small set of generic backbones/wrappers.
