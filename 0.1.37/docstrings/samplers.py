MCMC_MODULE = r"""
Hamiltonian Monte Carlo Sampler Module.

This module provides an implementation of the Hamiltonian Monte Carlo (HMC) sampling algorithm.
HMC is a Markov Chain Monte Carlo (MCMC) method that uses Hamiltonian dynamics to propose
new states in the sampling process, which can improve the efficiency of sampling from complex
distributions.

Classes:
    HamiltonianMonteCarlo: Implements the Hamiltonian Monte Carlo sampler.

Functions:
    visualize_sampling_trajectory: Visualize the sampling trajectory and diagnostics of the HMC sampler.
    plot_hmc_diagnostics: Plot detailed diagnostics for HMC sampling.

!!! example "Examples:"
    ```python
    from torchebm.samplers.mcmc import HamiltonianMonteCarlo
    from torchebm.core.energy_function import GaussianEnergy
    import torch
    energy_function = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))
    hmc = HamiltonianMonteCarlo(energy_function, step_size=0.1, n_leapfrog_steps=10)
    initial_state = torch.randn(10, 2)
    samples, diagnostics = hmc.sample(initial_state, n_steps=100, return_diagnostics=True)
    print(samples)
    print(diagnostics)
    ```

!!! info "Mathematical Background:"

    Hamiltonian Monte Carlo (HMC) is a method that uses Hamiltonian dynamics to propose new states in the sampling process. The Hamiltonian is defined as:

    $$
    H(x, p) = U(x) + K(p)
    $$

    where \( U(x) \) is the potential energy and \( K(p) \) is the kinetic energy. The potential energy is related to the target distribution \( \pi(x) \) by:

    $$
    U(x) = -\log \pi(x)
    $$

    The kinetic energy is typically defined as:

    $$
    K(p) = \frac{1}{2} p^T M^{-1} p
    $$

    where \( p \) is the momentum and \( M \) is the mass matrix.

    **Leapfrog Integration**
    !!! note ""

        The leapfrog integration method is used to simulate the Hamiltonian dynamics. It consists of the following steps:

        1. **Half-step update for momentum:**

            $$
            p_{t + \frac{1}{2}} = p_t - \frac{\epsilon}{2} \nabla U(x_t)
            $$

        2. **Full-step update for position:**

            $$
            x_{t + 1} = x_t + \epsilon M^{-1} p_{t + \frac{1}{2}}
            $$

        3. **Another half-step update for momentum:**

            $$
            p_{t + 1} = p_{t + \frac{1}{2}} - \frac{\epsilon}{2} \nabla U(x_{t + 1})
            $$

    **Acceptance Probability:**

    !!! note ""

        After proposing a new state using the leapfrog integration, the acceptance probability is computed as:

        $$
        \alpha = \min \left(1, \exp \left( H(x_t, p_t) - H(x_{t + 1}, p_{t + 1}) \right) \right)
        $$

        The new state is accepted with probability \( \alpha \).

??? info "References"

    - Implements the HMC based on https://faculty.washington.edu/yenchic/19A_stat535/Lec9_HMC.pdf

"""
