from typing import Optional, Union, Tuple

import torch

from torchebm.core import Sampler
from torchebm.core.energy_function import EnergyFunction


class HamiltonianMonteCarlo(Sampler):
    """Hamiltonian Monte Carlo sampler implementation.

    References:
        - Implements the HMC based on https://faculty.washington.edu/yenchic/19A_stat535/Lec9_HMC.pdf.
    """

    def __init__(
        self,
        energy_function: EnergyFunction,
        step_size: float = 1e-3,
        n_leapfrog_steps: int = 10,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        mass_matrix: Optional[torch.Tensor] = None,
    ):
        """
        Initialize Hamiltonian Monte Carlo sampler.


        Args:
            energy_function: The energy function to sample from
            step_size: The step size for leapfrog updates
            n_leapfrog_steps: Number of leapfrog steps per sample
            dtype: Tensor dtype to use
            device: Device to run on
        """
        super().__init__(energy_function, dtype, device)
        self.step_size = step_size
        self.n_leapfrog_steps = n_leapfrog_steps
        self.mass_matrix = mass_matrix

    def _compute_log_prob(self, state: torch.Tensor) -> torch.Tensor:
        """Computes the log-probability (up to a constant) for a given state (position)."""

        return -self.energy_function(state)

    def _sample_initial_momentum(
        self, batch_size: int, state_shape: tuple
    ) -> torch.Tensor:
        """Samples the initial momentum for a given state (position): ω0 ~ N(0, M^(-1))."""
        if self.mass_matrix is None:
            return torch.randn(
                (batch_size, *state_shape), device=self.device, dtype=self.dtype
            )
        return torch.matmul(
            self.mass_matrix,
            torch.randn(
                (batch_size, *state_shape), device=self.device, dtype=self.dtype
            ),
        )

    def _kinetic_energy(self, momentum: torch.Tensor) -> torch.Tensor:
        """Compute kinetic energy K(ω) = 1/2 ω^T M^(-1) ω."""

        if self.mass_matrix is None:
            return torch.sum(0.5 * momentum**2, dim=tuple(range(1, momentum.dim())))
        mass_matrix_inverse = torch.inverse(self.mass_matrix)
        return torch.sum(
            0.5 * torch.matmul(momentum, mass_matrix_inverse) * momentum,
            dim=tuple(range(1, momentum.dim())),
        )

    def _compute_hamiltonian(
        self, position: torch.Tensor, momentum: torch.Tensor
    ) -> torch.Tensor:
        """Compute H(x,ω) = -log r(x) + K(ω)."""
        return -self._compute_log_prob(position) + self._kinetic_energy(momentum)

    @torch.enable_grad()
    def sample(
        self,
        initial_state: torch.Tensor,
        n_steps: int,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Generate samples using HMC following the specified steps."""
        device = initial_state.device

        diagnostics = self._setup_diagnostics() if return_diagnostics else None
        current_position = initial_state.clone().to(device).requires_grad_(True)
        batch_size = current_position.shape[0] if len(current_position.shape) > 1 else 1
        state_shape = (
            current_position.shape[1:]
            if len(current_position.shape) > 1
            else current_position.shape
        )

        for step in range(n_steps):
            # 1. generate initial momentum
            initial_momentum = self._sample_initial_momentum(
                batch_size, state_shape
            ).to(device)

            # 2. x(0) = x0
            position = current_position.clone().to(device).requires_grad_(True)

            # 3. half-step update - momentum
            log_prob = self._compute_log_prob(position)
            grad_log_prob = torch.autograd.grad(
                log_prob.sum(), position, create_graph=False, retain_graph=True
            )[0].to(device)
            momentum = initial_momentum - 0.5 * self.step_size * grad_log_prob

            # 4. main leapfrog steps
            for l in range(self.n_leapfrog_steps - 1):
                # (a) update position
                position = (position + self.step_size * momentum).to(device)
                position.requires_grad_(True)

                # (b) update momentum
                log_prob = self._compute_log_prob(position)
                grad_log_prob = torch.autograd.grad(
                    log_prob.sum(), position, create_graph=False, retain_graph=True
                )[0].to(device)
                momentum = momentum - self.step_size * grad_log_prob

            # 5. last position update
            position = (position + self.step_size * momentum).to(device)
            position.requires_grad_(True)

            # 6. last half-step momentum update
            log_prob = self._compute_log_prob(position)
            grad_log_prob = torch.autograd.grad(
                log_prob.sum(), position, create_graph=False, retain_graph=True
            )[0].to(device)
            momentum = momentum - 0.5 * self.step_size * grad_log_prob

            # 7. compute acceptance probability
            initial_hamiltonian = self._compute_hamiltonian(
                current_position, initial_momentum
            ).to(device)
            proposed_hamiltonian = self._compute_hamiltonian(position, momentum).to(
                device
            )
            energy_diff = proposed_hamiltonian - initial_hamiltonian
            acceptance_prob = torch.exp(-energy_diff)

            # 8. accept/reject step
            accepted = torch.rand_like(acceptance_prob).to(device) < torch.minimum(
                torch.ones_like(acceptance_prob), acceptance_prob
            )

            # Update state
            current_position = torch.where(
                accepted.unsqueeze(-1), position, current_position
            )

            if return_diagnostics:
                diagnostics["energies"].append(
                    initial_hamiltonian.detach().mean().item()
                )
                diagnostics["acceptance_rate"] = (
                    diagnostics["acceptance_rate"] * step
                    + accepted.float().mean().item()
                ) / (step + 1)

        # Step 9: Return new state
        if return_diagnostics:
            return current_position, diagnostics
        return current_position

    @torch.no_grad()
    def sample_parallel(
        self,
        initial_states: torch.Tensor,
        n_steps: int,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Implementation of parallel Hamiltonian Monte Carlo sampling."""
        current_states = initial_states.to(device=self.device, dtype=self.dtype)
        diagnostics = (
            {"mean_energies": [], "acceptance_rates": []}
            if return_diagnostics
            else None
        )

        batch_size = current_states.shape[0]
        state_shape = current_states.shape[1:]

        for _ in range(n_steps):
            # Sample initial momentum
            momenta = self._sample_initial_momentum(batch_size, state_shape)

            # Perform leapfrog integration
            new_states = current_states.clone().requires_grad_(True)
            new_momenta = momenta.clone()

            for _ in range(self.n_leapfrog_steps):
                # Half-step momentum update
                log_prob = self._compute_log_prob(new_states)
                grad_log_prob = torch.autograd.grad(
                    log_prob.sum(), new_states, create_graph=False, retain_graph=True
                )[0]
                new_momenta -= 0.5 * self.step_size * grad_log_prob

                # Full-step position update
                new_states = new_states + self.step_size * new_momenta
                new_states.requires_grad_(True)

                # Full-step momentum update
                log_prob = self._compute_log_prob(new_states)
                grad_log_prob = torch.autograd.grad(
                    log_prob.sum(), new_states, create_graph=False, retain_graph=True
                )[0]
                new_momenta -= self.step_size * grad_log_prob

            # Final half-step momentum update
            log_prob = self._compute_log_prob(new_states)
            grad_log_prob = torch.autograd.grad(
                log_prob.sum(), new_states, create_graph=False, retain_graph=True
            )[0]
            new_momenta -= 0.5 * self.step_size * grad_log_prob

            # Compute Hamiltonian
            initial_hamiltonian = self._compute_hamiltonian(current_states, momenta)
            proposed_hamiltonian = self._compute_hamiltonian(new_states, new_momenta)
            energy_diff = proposed_hamiltonian - initial_hamiltonian
            acceptance_prob = torch.exp(-energy_diff)

            # Accept/reject step
            accept = torch.rand(batch_size, device=self.device) < acceptance_prob
            current_states = torch.where(
                accept.unsqueeze(-1), new_states, current_states
            )

            if return_diagnostics:
                diagnostics["mean_energies"].append(initial_hamiltonian.mean().item())
                diagnostics["acceptance_rates"].append(accept.float().mean().item())

        if return_diagnostics:
            diagnostics["mean_energies"] = torch.tensor(diagnostics["mean_energies"])
            diagnostics["acceptance_rates"] = torch.tensor(
                diagnostics["acceptance_rates"]
            )
            return current_states, diagnostics
        return current_states


import torch

from scipy.constants import physical_constants
import matplotlib.pyplot as plt
import scipy.special as sp
import seaborn as sns
import numpy as np


def radial_function(n, l, r, a0):
    """Compute the normalized radial part of the wavefunction using
    Laguerre polynomials and an exponential decay factor.
    Args:
        n (int): principal quantum number
        l (int): azimuthal quantum number
        r (numpy.ndarray): radial coordinate
        a0 (float): scaled Bohr radius
    Returns:
        numpy.ndarray: wavefunction radial component
    """

    laguerre = sp.genlaguerre(n - l - 1, 2 * l + 1)
    p = 2 * r / (n * a0)

    constant_factor = np.sqrt(
        ((2 / n * a0) ** 3 * (sp.factorial(n - l - 1)))
        / (2 * n * (sp.factorial(n + l)))
    )
    return constant_factor * np.exp(-p / 2) * (p**l) * laguerre(p)


def angular_function(m, l, theta, phi):
    """Compute the normalized angular part of the wavefunction using
    Legendre polynomials and a phase-shifting exponential factor.
    Args:
        m (int): magnetic quantum number
        l (int): azimuthal quantum number
        theta (numpy.ndarray): polar angle
        phi (int): azimuthal angle
    Returns:
        numpy.ndarray: wavefunction angular component
    """

    legendre = sp.lpmv(m, l, np.cos(theta))

    constant_factor = ((-1) ** m) * np.sqrt(
        ((2 * l + 1) * sp.factorial(l - np.abs(m)))
        / (4 * np.pi * sp.factorial(l + np.abs(m)))
    )
    return constant_factor * legendre * np.real(np.exp(1.0j * m * phi))


def compute_wavefunction(n, l, m, a0_scale_factor):
    """Compute the normalized wavefunction as a product
    of its radial and angular components.
    Args:
        n (int): principal quantum number
        l (int): azimuthal quantum number
        m (int): magnetic quantum number
        a0_scale_factor (float): Bohr radius scale factor
    Returns:
        numpy.ndarray: wavefunction
    """

    # Scale Bohr radius for effective visualization
    a0 = a0_scale_factor * physical_constants["Bohr radius"][0] * 1e12

    # x-y grid to represent electron spatial distribution
    grid_extent = 480
    grid_resolution = 680
    z = x = np.linspace(-grid_extent, grid_extent, grid_resolution)
    z, x = np.meshgrid(z, x)

    # Use epsilon to avoid division by zero during angle calculations
    eps = np.finfo(float).eps

    # Ψnlm(r,θ,φ) = Rnl(r).Ylm(θ,φ)
    psi = radial_function(n, l, np.sqrt((x**2 + z**2)), a0) * angular_function(
        m, l, np.arctan(x / (z + eps)), 0
    )
    return psi


def compute_probability_density(psi):
    """Compute the probability density of a given wavefunction.
    Args:
        psi (numpy.ndarray): wavefunction
    Returns:
        numpy.ndarray: wavefunction probability density
    """
    return np.abs(psi) ** 2


def plot_wf_probability_density(
    n, l, m, a0_scale_factor, dark_theme=False, colormap="rocket"
):
    """Plot the probability density of the hydrogen
    atom's wavefunction for a given quantum state (n,l,m).
    Args:
        n (int): principal quantum number, determines the energy level and size of the orbital
        l (int): azimuthal quantum number, defines the shape of the orbital
        m (int): magnetic quantum number, defines the orientation of the orbital
        a0_scale_factor (float): Bohr radius scale factor
        dark_theme (bool): If True, uses a dark background for the plot, defaults to False
        colormap (str): Seaborn plot colormap, defaults to 'rocket'
    """

    # Quantum numbers validation
    if not isinstance(n, int) or n < 1:
        raise ValueError("n should be an integer satisfying the condition: n >= 1")
    if not isinstance(l, int) or not (0 <= l < n):
        raise ValueError("l should be an integer satisfying the condition: 0 <= l < n")
    if not isinstance(m, int) or not (-l <= m <= l):
        raise ValueError(
            "m should be an integer satisfying the condition: -l <= m <= l"
        )

    # Colormap validation
    try:
        sns.color_palette(colormap)
    except ValueError:
        raise ValueError(f"{colormap} is not a recognized Seaborn colormap.")

    # Configure plot aesthetics using matplotlib rcParams settings
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["xtick.major.width"] = 4
    plt.rcParams["ytick.major.width"] = 4
    plt.rcParams["xtick.major.size"] = 15
    plt.rcParams["ytick.major.size"] = 15
    plt.rcParams["xtick.labelsize"] = 30
    plt.rcParams["ytick.labelsize"] = 30
    plt.rcParams["axes.linewidth"] = 4

    fig, ax = plt.subplots(figsize=(16, 16.5))
    plt.subplots_adjust(top=0.82)
    plt.subplots_adjust(right=0.905)
    plt.subplots_adjust(left=-0.1)

    # Compute and visualize the wavefunction probability density
    psi = compute_wavefunction(n, l, m, a0_scale_factor)
    prob_density = compute_probability_density(psi)

    # Here we transpose the array to align the calculated z-x plane with Matplotlib's y-x imshow display
    im = ax.imshow(
        np.sqrt(prob_density).T, cmap=sns.color_palette(colormap, as_cmap=True)
    )

    cbar = plt.colorbar(im, fraction=0.046, pad=0.03)
    cbar.set_ticks([])

    # Apply dark theme parameters
    if dark_theme:
        theme = "dt"
        background_color = sorted(
            sns.color_palette(colormap, n_colors=100),
            key=lambda color: 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2],
        )[0]
        plt.rcParams["text.color"] = "#dfdfdf"
        title_color = "#dfdfdf"
        fig.patch.set_facecolor(background_color)
        cbar.outline.set_visible(False)
        ax.tick_params(axis="x", colors="#c4c4c4")
        ax.tick_params(axis="y", colors="#c4c4c4")
        for spine in ax.spines.values():
            spine.set_color("#c4c4c4")

    else:  # Apply light theme parameters
        theme = "lt"
        plt.rcParams["text.color"] = "#000000"
        title_color = "#000000"
        ax.tick_params(axis="x", colors="#000000")
        ax.tick_params(axis="y", colors="#000000")

    ax.set_title(
        "Hydrogen Atom - Wavefunction Electron Density",
        pad=130,
        fontsize=44,
        loc="left",
        color=title_color,
    )
    ax.text(
        0,
        722,
        (
            r"$|\psi_{n \ell m}(r, \theta, \varphi)|^{2} ="
            r" |R_{n\ell}(r) Y_{\ell}^{m}(\theta, \varphi)|^2$"
        ),
        fontsize=36,
    )
    ax.text(30, 615, r"$({0}, {1}, {2})$".format(n, l, m), color="#dfdfdf", fontsize=42)
    ax.text(
        770, 140, "Electron probability distribution", rotation="vertical", fontsize=40
    )
    ax.text(705, 700, "Higher\nprobability", fontsize=24)
    ax.text(705, -60, "Lower\nprobability", fontsize=24)
    ax.text(775, 590, "+", fontsize=34)
    ax.text(769, 82, "−", fontsize=34, rotation="vertical")
    ax.invert_yaxis()

    # Save and display the plot
    plt.savefig(f"({n},{l},{m})[{theme}].png")
    plt.show()


# plot_wf_probability_density(3, 2, 1, 0.3, True)
#
# plot_wf_probability_density(4, 3, 0, 0.2, dark_theme=True, colormap="magma")
#
# plot_wf_probability_density(4, 3, 1, 0.2, dark_theme=True, colormap="mako")
#
# plot_wf_probability_density(20, 10, 5, 0.01, True, colormap="mako")
