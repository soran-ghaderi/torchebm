# Showcase

Cross-cutting demos that push the components end to end: annealed Langevin
transport of an image, numerical integrators raced against an exact solution, and
an interactive in-browser explainer. The techniques behind them are taught step by
step in the earlier tiers.

These are heavier than the rest of the curriculum. They may need extra
dependencies (each page says which), they are excluded from the CI smoke run via
`ci: skip` in their `meta.yaml`, and any figures or animations they produce are
written to a local, gitignored `assets/` folder.

Theory: [Design and Scope](../../concepts/design.md), which places every component
these demos compose.
