# Energy Matching: the paper's 2D suite

Reproduces the 2D experiments of [Energy Matching (arXiv:2504.10612)](https://arxiv.org/abs/2504.10612)
on the paper's own toy datasets: a single time-independent potential
\(V(x)\) transports an 8-Gaussian ring onto two moons, then samples its
Boltzmann density near the data. `main.py` trains with live terminal
progress and prints the quantitative checks; `plot.py` renders the figures.

`plot.py` renders four figures locally (they are not committed):

- **Sample evolution along the SDE sweep**: deterministic transport below
  \(\tau^*\), Langevin ramp after.
- **The two-regime schedule**: temperature \(\epsilon(t)\) and the flow-loss
  time gate \(w(t)\).
- **Local intrinsic dimension**: the Hessian of \(V\) at data points has one
  flat (tangent) and one stiff (normal) direction, so
  \(\mathrm{LID} = d - \mathrm{rank}(\nabla^2 V) \approx 1\) on the moons.
- **Diverse generation**: 64 Langevin chains started from a single point,
  without and with the paper's repulsive interaction energy \(W\).
