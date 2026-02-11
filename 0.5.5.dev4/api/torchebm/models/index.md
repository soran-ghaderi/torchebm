# Torchebm > Models

## Contents

### Subpackages

- [Components](https://soran-ghaderi.github.io/torchebm/0.5.5.dev4/api/torchebm/models/components/index.md)

### Modules

- [Conditional_transformer_2d](https://soran-ghaderi.github.io/torchebm/0.5.5.dev4/api/torchebm/models/conditional_transformer_2d/index.md)
- [Wrappers](https://soran-ghaderi.github.io/torchebm/0.5.5.dev4/api/torchebm/models/wrappers/index.md)

## API Reference

### torchebm.models

Model namespace.

TorchEBM is designed for plug-and-play experimentation:

- try different losses with the same backbone
- try different backbones with the same loss
- use samplers as long as the model signature matches

This package therefore exposes *reusable building blocks* under `torchebm.models.components` and a small set of generic backbones/wrappers.
