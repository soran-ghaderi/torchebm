---
title: examples
draft: true
date: 2023-03-07
authors:
  - soran-ghaderi
categories:
  - Examples
---

# Code Examples 

Content coming soon.

Langevin dynamics sampling
<!-- more -->


```py title="Langevin dynamics sampling" linenums="1"
def langevin_gaussain_sampling():

    energy_fn = GaussianEnergy(mean=torch.zeros(10), cov=torch.eye(10))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize Langevin dynamics model
    langevin_sampler = LangevinDynamics(
        energy_function=energy_fn, step_size=5e-3, device=device
    ).to(device)

    # Initial state: batch of 100 samples, 10-dimensional space
    ts = time.time()
    # Run Langevin sampling for 500 steps
    final_x = langevin_sampler.sample_chain(
        dim=10, n_steps=500, n_samples=10000, return_trajectory=False
    )

    print(final_x.shape)  # Output: (100, 10)  (final state)
    # print(xs.shape)  # Output: (500, 100, 10)  (history of all states)
    print("Time taken: ", time.time() - ts)

    n_samples = 250
    n_steps = 500
    dim = 10
    final_samples, diagnostics = langevin_sampler.sample_chain(
        n_samples=n_samples,
        n_steps=n_steps,
        dim=dim,
        return_trajectory=True,
        return_diagnostics=True,
    )
    print(final_samples.shape)  # Output: (100, 10)  (final state)
    print(diagnostics.shape)  # (500, 3, 100, 10) -> All diagnostics

    x_init = torch.randn(n_samples, dim, dtype=torch.float32, device="cuda")
    samples = langevin_sampler.sample(x=x_init, n_steps=100)
    print(samples.shape)  # Output: (100, 10)  (final state)
```