"""Integrators: order of accuracy, measured against an exact ODE solution.

    dx/dt = A x with A = [[0, 1], [-1, 0]] is a harmonic oscillator whose exact
    solution is a rotation; halving the step should shrink the error by 2^order.
"""

import math

import torch

from torchebm.integrators import get_integrator

A = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
T = 2 * math.pi                      # one full revolution: x(T) should equal x(0)
x0 = torch.randn(256, 2)


def drift(x, t):
    return x @ A.T


def exact(t):
    c, s = math.cos(t), math.sin(t)
    return x0 @ torch.tensor([[c, -s], [s, c]]).T


# The registry constructs integrators by name; samplers accept the same names.
for name, order in (("euler_maruyama", 1), ("heun", 2), ("rk4", 4)):
    errors = []
    for n_steps in (100, 200):
        integ = get_integrator(name)
        t = torch.linspace(0.0, T, n_steps + 1)
        xT = integ.integrate(
            state={"x": x0.clone()},
            step_size=t[1] - t[0],
            n_steps=n_steps,
            drift=drift,
            t=t,
        )["x"]
        errors.append((xT - exact(T)).norm(dim=1).mean().item())
    ratio = errors[0] / errors[1]
    print(f"{name:15s} error {errors[0]:.2e} -> {errors[1]:.2e} on halved step"
          f"   ratio {ratio:5.1f}  (theory ~{2**order})")

# Euler drifts off the circle, Heun gains two orders, RK4 four; the same
# integrators plug into FlowSampler and the MCMC samplers via integrator="...".
