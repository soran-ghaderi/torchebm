# `torchebm.couplings`

Minibatch couplings for pairing source and target samples.

A coupling reorders (or resamples) a minibatch of source samples (x_0) against targets (x_1) before interpolation, so that transport happens along efficient pairs (OT-CFM, rectified flow, Energy Matching warm-up).

Couplings are lazy-loaded to avoid importing every submodule at package import time. Direct imports still work: `from torchebm.couplings import SinkhornCoupling`.
