# `torchebm.integrators`

Integrators for solving differential equations in energy-based models.

Integrators are lazy-loaded to avoid importing all 8 submodules at package import time. Direct imports still work: `from torchebm.integrators import LeapfrogIntegrator`.
