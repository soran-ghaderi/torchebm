---
#template: home.html
title: TorchEBM Docs
social:
  cards_layout_options:
    title: Documentation
hide:
  - navigation
  - toc
---
# ![TorchEBM Logo](assets/images/logo_with_text.svg){ width="280" } { .animate__animated .animate__fadeIn }

<p class="lead" markdown>
⚡ Energy-Based Modeling library for PyTorch, offering tools for 🔬 sampling, 🧠 inference, and 📊 learning in complex distributions.
</p>

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    Start using TorchEBM in minutes with our quick installation and setup guide.

    [:octicons-arrow-right-24: Getting Started](getting_started.md)

-   :fontawesome-solid-book:{ .lg .middle } __Introduction__

    ---

    Learn about Energy-Based Models and how TorchEBM can help you work with them.

    [:octicons-arrow-right-24: Introduction](introduction.md)

-   :material-tune:{ .lg .middle } __Guides__

    ---

    Explore in-depth guides for energy functions, samplers, and more.

    [:octicons-arrow-right-24: Guides](guides/index.md)

-   :material-code-tags:{ .lg .middle } __Examples__

    ---

    Practical examples to help you apply TorchEBM to your projects.

    [:octicons-arrow-right-24: Examples](examples/index.md)
    
-   :material-post:{ .lg .middle } __Blog__

    ---

    Stay updated with the latest news, tutorials, and insights about TorchEBM.

    [:octicons-arrow-right-24: Blog](blog/index.md)

</div>

## Quick Installation

```bash
pip install torchebm
```

## Example Analytical Energy Landscapes

!!! note "Toy Examples"
    
    These are some TorchEBM's built-in toy analytical energy landscapes for functionality and performance testing purposes.

=== "Gaussian Energy"

    <div class="energy-grid" markdown>
    <div class="energy-main" markdown>
    ![Gaussian Energy](assets/images/e_functions/gaussian.png){ .spotlight }
    
    <div class="energy-caption energy-caption-bottom">
    **Gaussian Energy**
    
    $E(x) = \frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)$
    </div>
    </div>
    <div class="energy-others" markdown>
    ![Double Well Energy](assets/images/e_functions/double_well.png)
    ![Rastrigin Energy](assets/images/e_functions/rastrigin.png)
    ![Rosenbrock Energy](assets/images/e_functions/rosenbrock.png)
    </div>
    </div>
    
    ```python
    from torchebm.core import GaussianEnergy
    import torch
    
    energy_fn = GaussianEnergy(
        mean=torch.zeros(2),
        cov=torch.eye(2)
    )
    ```

=== "Double Well Energy"

    <div class="energy-grid" markdown>
    <div class="energy-main" markdown>
    ![Double Well Energy](assets/images/e_functions/double_well.png){ .spotlight }
    
    <div class="energy-caption energy-caption-bottom">
    **Double Well Energy**
    
    $E(x) = h \sum_{i=1}^n \left[(x_i^2 - 1)^2\right]$
    </div>
    </div>
    <div class="energy-others" markdown>
    ![Gaussian Energy](assets/images/e_functions/gaussian.png)
    ![Rastrigin Energy](assets/images/e_functions/rastrigin.png)
    ![Rosenbrock Energy](assets/images/e_functions/rosenbrock.png)
    </div>
    </div>
    
    ```python
    from torchebm.core import DoubleWellEnergy
    
    energy_fn = DoubleWellEnergy(
        barrier_height=2.0
    )
    ```

=== "Rastrigin Energy"

    <div class="energy-grid" markdown>
    <div class="energy-main" markdown>
    ![Rastrigin Energy](assets/images/e_functions/rastrigin.png){ .spotlight }
    
    <div class="energy-caption energy-caption-bottom">
    **Rastrigin Energy**
    
    $E(x) = an + \sum_{i=1}^n \left[ x_i^2 - a\cos(2\pi x_i) \right]$
    </div>
    </div>
    <div class="energy-others" markdown>
    ![Gaussian Energy](assets/images/e_functions/gaussian.png)
    ![Double Well Energy](assets/images/e_functions/double_well.png)
    ![Rosenbrock Energy](assets/images/e_functions/rosenbrock.png)
    </div>
    </div>
    
    ```python
    from torchebm.core import RastriginEnergy
    
    energy_fn = RastriginEnergy(
        a=10.0
    )
    ```

=== "Rosenbrock Energy"

    <div class="energy-grid" markdown>
    <div class="energy-main" markdown>
    ![Rosenbrock Energy](assets/images/e_functions/rosenbrock.png){ .spotlight }
    
    <div class="energy-caption energy-caption-bottom">
    **Rosenbrock Energy**
    
    $E(x) = \sum_{i=1}^{n-1} \left[ a(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \right]$
    </div>
    </div>
    <div class="energy-others" markdown>
    ![Gaussian Energy](assets/images/e_functions/gaussian.png)
    ![Double Well Energy](assets/images/e_functions/double_well.png)
    ![Rastrigin Energy](assets/images/e_functions/rastrigin.png)
    </div>
    </div>
    
    ```python
    from torchebm.core import RosenbrockEnergy
    
    energy_fn = RosenbrockEnergy(
        a=1.0, 
        b=100.0
    )
    ```

## Quick Example

<div class="grid cards" markdown>

-   __Create and Sample from Energy Models__

    ---
    
    ```python
    import torch
    from torchebm.core import GaussianEnergy
    from torchebm.samplers.langevin_dynamics import LangevinDynamics
    
    # Create an energy function
    energy_fn = GaussianEnergy(
        mean=torch.zeros(2),
        cov=torch.eye(2)
    )
    
    # Create a sampler
    sampler = LangevinDynamics(
        energy_function=energy_fn,
        step_size=0.01
    )
    
    # Generate samples
    samples = sampler.sample_chain(
        dim=2, n_steps=100, n_samples=1000
    )
    ```

</div>

!!! info "Latest Release"

    TorchEBM is currently in early development. Check our [GitHub repository](https://github.com/soran-ghaderi/torchebm) for the latest updates and features.

## Features & Roadmap

Our goal is to create a comprehensive library for energy-based modeling in PyTorch.

Status indicators:

- :white_check_mark: - Completed
- :construction: - Work in progress
- :warning: - Needs improvement
- :sparkles: - Planned feature

### Core Infrastructure
<div class="grid" markdown>
<div markdown>
- [x] CUDA-accelerated implementations :white_check_mark:
- [x] Seamless integration with PyTorch :white_check_mark:
- [x] Energy function base class :white_check_mark:
- [x] Sampler base class :white_check_mark:
</div>
</div>

=== "Energy Functions"

    - [x] Gaussian :white_check_mark:
    - [x] Double well :white_check_mark:
    - [x] Rastrigin :white_check_mark:
    - [x] Rosenbrock :white_check_mark:
    - [x] Ackley :white_check_mark:

=== "Implemented Samplers"

    - [x] Langevin Dynamics :white_check_mark:
    - [x] Hamiltonian Monte Carlo (HMC) :construction:
    - [ ] Metropolis-Hastings :warning:

=== "Diffusion-based Samplers"

    - [ ] Denoising Diffusion Probabilistic Models (DDPM) :sparkles:
    - [ ] Denoising Diffusion Implicit Models (DDIM) :sparkles:
    - [ ] Generalized Gaussian Diffusion Models (GGDM) :sparkles:
    - [ ] Differentiable Diffusion Sampler Search (DDSS) :sparkles:
    - [ ] Euler Method :sparkles:
    - [ ] Heun's Method :sparkles:
    - [ ] PLMS (Pseudo Likelihood Multistep) :sparkles:
    - [ ] DPM (Diffusion Probabilistic Models) :sparkles:

=== "MCMC Samplers"

    - [ ] Gibbs Sampling :sparkles:
    - [ ] No-U-Turn Sampler (NUTS) :sparkles:
    - [ ] Slice Sampling :sparkles:
    - [ ] Reversible Jump MCMC :sparkles:
    - [ ] Particle Filters (Sequential Monte Carlo) :sparkles:
    - [ ] Adaptive Metropolis :sparkles:
    - [ ] Parallel Tempering (Replica Exchange) :sparkles:
    - [ ] Stochastic Gradient Langevin Dynamics (SGLD) :sparkles:
    - [ ] Stein Variational Gradient Descent (SVGD) :sparkles:
    - [ ] Metropolis-Adjusted Langevin Algorithm (MALA) :sparkles:
    - [ ] Unadjusted Langevin Algorithm (ULA) :sparkles:
    - [ ] Bouncy Particle Sampler :sparkles:
    - [ ] Zigzag Sampler :sparkles:
    - [ ] Annealed Importance Sampling (AIS) :sparkles:
    - [ ] Sequential Monte Carlo (SMC) Samplers :sparkles:
    - [ ] Elliptical Slice Sampling :sparkles:

=== "BaseLoss Functions"

    - [ ] Contrastive Divergence Methods :construction:
      * [ ] Contrastive Divergence (CD-k) :construction:
      * [ ] Persistent Contrastive Divergence (PCD) :sparkles:
      * [ ] Fast Persistent Contrastive Divergence (FPCD) :sparkles:
      * [ ] Parallel Tempering Contrastive Divergence (PTCD) :sparkles:
    - [ ] Score Matching Techniques :sparkles:
      * [ ] Standard Score Matching :sparkles:
      * [ ] Denoising Score Matching :sparkles:
      * [ ] Sliced Score Matching :sparkles:
    - [ ] Maximum Likelihood Estimation (MLE) :sparkles:
    - [ ] Margin BaseLoss :sparkles:
    - [ ] Noise Contrastive Estimation (NCE) :sparkles:
    - [ ] Ratio Matching :sparkles:
    - [ ] Minimum Probability Flow :sparkles:
    - [ ] Adversarial Training BaseLoss :sparkles:
    - [ ] Kullback-Leibler (KL) Divergence BaseLoss :sparkles:
    - [ ] Fisher Divergence :sparkles:
    - [ ] Hinge Embedding BaseLoss :sparkles:
    - [ ] Cross-Entropy BaseLoss (for discrete outputs) :sparkles:
    - [ ] Mean Squared Error (MSE) BaseLoss (for continuous outputs) :sparkles:
    - [ ] Improved Contrastive Divergence BaseLoss :sparkles:

=== "Other Modules"

    - [ ] Testing Framework :construction:
    - [ ] Visualization Tools :construction:
    - [ ] Performance Benchmarking :sparkles:
    - [ ] Neural Network Integration :sparkles:
    - [ ] Hyperparameter Optimization :sparkles:
    - [ ] Distribution Diagnostics :sparkles:

## License

TorchEBM is released under the [MIT License](https://github.com/soran-ghaderi/torchebm/blob/master/LICENSE), which is a permissive license that allows for reuse with few restrictions.

## Contributing

We welcome contributions! If you're interested in improving TorchEBM or adding new features, please check our [contributing guidelines](developer_guide/contributing.md).

Our project follows specific [commit message conventions](developer_guide/contributing.md#commit-message-conventions) to maintain a clear project history and generate meaningful changelogs.

<div class="grid" markdown>

<div markdown>
[:material-github: GitHub](https://github.com/soran-ghaderi/torchebm){ .md-button .md-button--primary target="_blank"}
</div>

<div markdown>
[:material-file-document: API Reference](api/index.md){ .md-button }
</div>

<div markdown>
[:material-frequently-asked-questions: FAQ](faq.md){ .md-button }
</div>

<div markdown>
[:material-tools: Development](developer_guide/contributing.md){ .md-button }
</div>

</div>

