r"""Compute layer for the "Integrators in practice" explainer.

TorchEBM is the compute engine: this script runs every experiment and writes a
single ``data.json`` that the Three.js front-end (index.html / app.js) animates,
plus ``data.js`` carrying the same payload as a JS global so index.html opens
by double-click (no fetch, no CORS). It also renders a static ``preview.png``
contact sheet so the physics can be sanity-checked before building the
interactive artifact.

Three chapters:
    1. ODE: eccentric two-body (Kepler) orbit, Euler vs RK4 vs symplectic
       Leapfrog; symplectic stays a closed ellipse, the rest drift.
    2. SDE: overdamped Langevin on a double well, Euler-Maruyama vs Heun vs
       Backward-Euler-Maruyama recovering the Boltzmann density; at large dt
       the explicit method destabilises while the implicit one holds.
    3. S^2: geodesic motion on a sphere; a naive ambient Euler step drifts off
       the manifold, an exponential-map (geodesic) step stays exact.

Run:  python examples/90-showcase/03-integrator-story/compute.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import time

from torchebm.integrators import (
    EulerMaruyamaIntegrator,
    HeunIntegrator,
    BackwardEulerMaruyamaIntegrator,
    RK4Integrator,
    LeapfrogIntegrator,
)
from torchebm.core import BaseModel
from torchebm.samplers import LangevinDynamics

HERE = Path(__file__).resolve().parent
torch.manual_seed(0)

ORANGE = "#E69F00"
BLUE = "#56B4E9"
GREEN = "#00D49A"
TRUTH = "#e8e8ee"


def _r(a, nd=4):
    """Round a numpy array to a JSON-friendly nested list."""
    return np.asarray(a, dtype=np.float64).round(nd).tolist()


GM = 1.0


def kepler_accel(pos: torch.Tensor) -> torch.Tensor:
    """Newtonian gravity a = -GM r / |r|^3 for pos shape (..., 2)."""
    r = torch.linalg.vector_norm(pos, dim=-1, keepdim=True).clamp_min(1e-6)
    return -GM * pos / r ** 3


def kepler_full_drift(z, t):
    """First-order field (vx, vy, ax, ay) for the RK/Euler integrators."""
    pos, vel = z[..., :2], z[..., 2:]
    return torch.cat([vel, kepler_accel(pos)], dim=-1)


def kepler_energy(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(pos, axis=-1)
    return 0.5 * np.sum(vel ** 2, axis=-1) - GM / np.clip(r, 1e-6, None)


def run_kepler(h: float, n: int):
    """Kepler orbit from aphelion with eccentricity ~0.7 (RK4 visibly decays):
    Euler and RK4 on the full first-order system (Euler-Maruyama without noise
    is forward Euler), then symplectic Leapfrog on (x=position, p=velocity,
    mass=1)."""
    p0 = torch.tensor([[1.4, 0.0]])
    v0 = torch.tensor([[0.0, 0.45]])
    t = np.arange(n + 1) * h

    methods = []

    for name, color, integ in [
        ("Forward Euler", ORANGE, EulerMaruyamaIntegrator()),
        ("RK4", BLUE, RK4Integrator()),
    ]:
        z = torch.cat([p0, v0], dim=-1)
        state = {"x": z}
        pos = [p0.numpy().copy()[0]]
        vel = [v0.numpy().copy()[0]]
        for _ in range(n):
            state = integ.step(state, h, drift=kepler_full_drift)
            zc = state["x"][0]
            pos.append(zc[:2].numpy().copy())
            vel.append(zc[2:].numpy().copy())
        pos = np.asarray(pos); vel = np.asarray(vel)
        methods.append({"name": name, "color": color,
                        "xy": _r(pos), "energy": _r(kepler_energy(pos, vel), 5)})

    lf = LeapfrogIntegrator()
    state = {"x": p0.clone(), "p": v0.clone()}
    pos = [p0.numpy().copy()[0]]; vel = [v0.numpy().copy()[0]]
    for _ in range(n):
        state = lf.step(state, h, drift=lambda x, t: kepler_accel(x))
        pos.append(state["x"][0].numpy().copy())
        vel.append(state["p"][0].numpy().copy())
    pos = np.asarray(pos); vel = np.asarray(vel)
    methods.append({"name": "Leapfrog (symplectic)", "color": GREEN,
                    "xy": _r(pos), "energy": _r(kepler_energy(pos, vel), 5)})

    e0 = float(kepler_energy(p0.numpy(), v0.numpy())[0])
    print("  [Kepler]  energy drift |E_end-E0|:")
    for m in methods:
        drift = abs(m["energy"][-1] - e0)
        rmax = float(np.max(np.linalg.norm(np.asarray(m["xy"]), axis=-1)))
        print(f"    {m['name']:<22} {drift:9.4f}   max r={rmax:.2f}")
    return {"t": _r(t, 4), "GM": GM, "energy0": round(e0, 5), "methods": methods}


TB_X = [0.97000436, -0.24308753, -0.97000436, 0.24308753, 0.0, 0.0]
TB_V = [0.46620369, 0.43236573, 0.46620369, 0.43236573, -0.93240737, -0.86473146]
TB_PERIOD = 6.3259


def tb_accel(pos: torch.Tensor) -> torch.Tensor:
    """Pairwise gravity for 3 bodies; pos (..., 6) = [x1,y1,x2,y2,x3,y3]."""
    P = pos.view(*pos.shape[:-1], 3, 2)
    a = torch.zeros_like(P)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            d = P[..., j, :] - P[..., i, :]
            r = d.norm(dim=-1, keepdim=True).clamp_min(1e-4)
            a[..., i, :] = a[..., i, :] + d / r ** 3
    return a.reshape(pos.shape)


def tb_full_drift(z, t):
    pos, vel = z[..., :6], z[..., 6:]
    return torch.cat([vel, tb_accel(pos)], dim=-1)


def tb_energy(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    P = pos.reshape(-1, 3, 2)
    ke = 0.5 * np.sum(vel ** 2, axis=-1)
    pe = np.zeros(P.shape[0])
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        pe -= 1.0 / np.linalg.norm(P[:, j] - P[:, i], axis=-1).clip(1e-4)
    return ke + pe


def run_threebody(h: float, periods: float, keep: int):
    """The Chenciner-Montgomery figure-eight choreography (G=1, equal unit masses)."""
    p0 = torch.tensor([TB_X]); v0 = torch.tensor([TB_V])
    n = int(round(periods * TB_PERIOD / h))
    every = max(1, n // keep)

    def drive_rk(integ):
        z = torch.cat([p0, v0], -1); state = {"x": z}; P = [TB_X[:]]; V = [TB_V[:]]
        for i in range(n):
            state = integ.step(state, h, drift=tb_full_drift)
            if (i + 1) % every == 0:
                zc = state["x"][0]; P.append(zc[:6].tolist()); V.append(zc[6:].tolist())
        return np.array(P), np.array(V)

    def drive_lf():
        lf = LeapfrogIntegrator(); state = {"x": p0.clone(), "p": v0.clone()}; P = [TB_X[:]]; V = [TB_V[:]]
        for i in range(n):
            state = lf.step(state, h, drift=lambda x, t: tb_accel(x))
            if (i + 1) % every == 0:
                P.append(state["x"][0].tolist()); V.append(state["p"][0].tolist())
        return np.array(P), np.array(V)

    runs = {
        "Forward Euler": (ORANGE,) + drive_rk(EulerMaruyamaIntegrator()),
        "RK4": (BLUE,) + drive_rk(RK4Integrator()),
        "Leapfrog": (GREEN,) + drive_lf(),
    }
    e0 = float(tb_energy(np.array([TB_X]), np.array([TB_V]))[0])
    methods = []
    print(f"\n  [3-body figure-8]  h={h}  periods={periods}  e0={e0:.3f}")
    for name, (color, P, V) in runs.items():
        E = tb_energy(P, V)
        bodies = P.reshape(P.shape[0], 3, 2)
        methods.append({"name": name, "color": color,
                        "bodies": _r(bodies, 4), "energy": _r(E, 4)})
        print(f"    {name:<16} |E_end-e0|={abs(E[-1]-e0):8.3f}  "
              f"max excursion={np.abs(bodies).max():.2f}")
    return {"period": TB_PERIOD, "energy0": round(e0, 4), "methods": methods}


BARRIER = 1.0
TEMP = 0.5


def U(x):
    return BARRIER * (x ** 2 - 1.0) ** 2


def dU(x):
    return BARRIER * 4.0 * x * (x ** 2 - 1.0)


def langevin_drift(x, t):
    return -dU(x)


def run_langevin(dt: float, n_steps: int, n_walkers: int, keep: int, label_blowup=False):
    """Overdamped Langevin on U(x) = BARRIER (x^2 - 1)^2 at temperature TEMP
    (diffusion D = TEMP), walkers started near the barrier; return per-method
    walker frames and the final histogram."""
    xgrid = np.linspace(-2.2, 2.2, 220)
    true_pdf = np.exp(-U(xgrid) / TEMP)
    true_pdf /= true_pdf.sum() * (xgrid[1] - xgrid[0])

    D = torch.tensor(float(TEMP))
    specs = [
        ("Euler-Maruyama", ORANGE, EulerMaruyamaIntegrator()),
        ("Heun", BLUE, HeunIntegrator()),
        ("Backward-Euler-Maruyama", GREEN, BackwardEulerMaruyamaIntegrator()),
    ]
    methods = []
    for name, color, integ in specs:
        torch.manual_seed(0)
        x = (torch.rand(n_walkers, 1) * 0.4 - 0.2)
        frames = [x[:, 0].numpy().copy()]
        every = max(1, n_steps // keep)
        blew = False
        for i in range(n_steps):
            x = integ.step({"x": x}, dt, drift=langevin_drift, diffusion=D)["x"]
            if not torch.isfinite(x).all() or x.abs().max() > 50:
                blew = True
                x = torch.nan_to_num(x, nan=0.0).clamp(-50, 50)
            if (i + 1) % every == 0:
                frames.append(x[:, 0].numpy().copy())
        frames = np.asarray(frames)
        hist, edges = np.histogram(frames[-1], bins=60, range=(-2.2, 2.2), density=True)
        methods.append({"name": name, "color": color, "blew_up": bool(blew),
                        "frames": _r(frames, 3),
                        "hist": _r(hist, 4)})
    out = {"dt": dt, "n_walkers": n_walkers,
           "xgrid": _r(xgrid, 4), "U": _r(U(xgrid), 4), "true_pdf": _r(true_pdf, 5),
           "hist_edges": _r(np.linspace(-2.2, 2.2, 61), 4),
           "temp": TEMP, "barrier": BARRIER, "methods": methods}
    tag = "large dt" if label_blowup else "small dt"
    print(f"  [Langevin {tag}] dt={dt}:")
    for m in methods:
        fr = np.asarray(m["frames"][-1])
        print(f"    {m['name']:<26} blew_up={m['blew_up']!s:<5} "
              f"frac|x|>0.5={np.mean(np.abs(fr) > 0.5):.2f}  range=[{fr.min():.2f},{fr.max():.2f}]")
    return out


def run_sde_line(theta=1.5, sigma=1.0, x0=2.0, dt=0.3, n=42, stride=24):
    r"""Drive Euler-Maruyama / Heun / Backward-EM on dX = -theta X dt + sigma dW,
    all sharing ONE Brownian path, against a fine-step reference.

    The integrators add ``sqrt(2*diffusion)*noise*sqrt(dt)``, so diffusion =
    sigma^2/2 and ``noise`` is a standard normal. The coarse normal for step i is
    the standardised sum of the ``stride`` fine increments inside that step, so the
    coarse methods and the fine reference see the identical Brownian motion.
    """
    torch.manual_seed(0)
    D = torch.tensor(sigma ** 2 / 2.0)
    drift = lambda x, t: -theta * x
    dt_fine = dt / stride
    zf = torch.randn(n * stride, 1)

    em = EulerMaruyamaIntegrator()
    x = torch.tensor([[x0]]); ref = [x0]
    for i in range(n * stride):
        x = em.step({"x": x}, dt_fine, drift=drift, diffusion=D, noise=zf[i:i + 1])["x"]
        if (i + 1) % stride == 0:
            ref.append(float(x.item()))
    ref = np.asarray(ref)

    zc = zf.view(n, stride, 1).sum(dim=1) / math.sqrt(stride)

    specs = [("Euler-Maruyama", ORANGE, EulerMaruyamaIntegrator()),
             ("Heun", BLUE, HeunIntegrator()),
             ("Backward-Euler-Maruyama", GREEN, BackwardEulerMaruyamaIntegrator())]
    methods = []
    for name, color, integ in specs:
        x = torch.tensor([[x0]]); path = [x0]
        for i in range(n):
            x = integ.step({"x": x}, dt, drift=drift, diffusion=D, noise=zc[i:i + 1])["x"]
            path.append(float(x.item()))
        path = np.asarray(path)
        methods.append({"name": name, "color": color, "path": _r(path, 4),
                        "err": round(float(np.max(np.abs(path - ref))), 4)})

    t = np.arange(n + 1) * dt
    print(f"\n  [SDE line]  OU theta={theta} dt={dt}  max|path-ref|:")
    for m in methods:
        print(f"    {m['name']:<26} {m['err']:.3f}   end={m['path'][-1]:+.3f}  (ref end {ref[-1]:+.3f})")
    return {"theta": theta, "sigma": sigma, "dt": dt, "t": _r(t, 4),
            "ref": _r(ref, 4), "methods": methods}


def run_ot(n=620, F=64):
    r"""Flow-Matching / optimal-transport demo: a Gaussian source is carried to a
    structured two-moons target. A greedy OT coupling pairs the points, and
    TorchEBM's interpolants set the path: LinearInterpolant -> straight OT
    geodesics, CosineInterpolant -> curved paths.  x_t = alpha(t) x1 + sigma(t) x0."""
    from torchebm.interpolants.linear import LinearInterpolant
    from torchebm.interpolants.cosine import CosineInterpolant
    torch.manual_seed(2)

    x0 = torch.randn(n, 2) * 0.42 + torch.tensor([-2.7, 0.0])

    h1 = n // 2
    a1 = math.pi * torch.rand(h1)
    a2 = math.pi * torch.rand(n - h1)
    moons = torch.cat([torch.stack([torch.cos(a1), torch.sin(a1)], 1),
                       torch.stack([1 - torch.cos(a2), 0.5 - torch.sin(a2)], 1)], 0)
    moons = (moons - moons.mean(0)) * 1.45 + torch.randn(n, 2) * 0.045
    x1 = moons + torch.tensor([2.7, 0.0])

    C = ((x0[:, None, :] - x1[None, :, :]) ** 2).sum(-1).numpy()
    order = np.argsort(C, axis=None)
    u0 = np.zeros(n, bool); u1 = np.zeros(n, bool); pair = np.full(n, -1); cnt = 0
    for idx in order:
        i, j = int(idx // n), int(idx % n)
        if not u0[i] and not u1[j]:
            pair[i] = j; u0[i] = u1[j] = True; cnt += 1
            if cnt == n:
                break
    x1 = x1[torch.from_numpy(pair)]

    tg = torch.linspace(0, 1, F)
    sched = {}
    for name, interp in [("linear", LinearInterpolant()), ("cosine", CosineInterpolant())]:
        al = interp.compute_alpha_t(tg)[0].reshape(-1)
        sg = interp.compute_sigma_t(tg)[0].reshape(-1)
        sched[name] = {"alpha": _r(al.numpy(), 4), "sigma": _r(sg.numpy(), 4)}
    print(f"\n  [OT]  n={n}  source=Gaussian  target=two-moons  coupling=greedy-OT  "
          f"interpolants={list(sched)}")
    return {"x0": _r(x0.numpy(), 3), "x1": _r(x1.numpy(), 3),
            "t": _r(tg.numpy(), 4), "sched": sched}


def run_sphere(h: float, n: int):
    """A great-circle geodesic from the north pole along a tilted unit-speed
    tangent: the exact exponential-map step (a rotation in the (x, v) plane)
    against naive ambient Euler on x'' = -x with no re-projection, whose
    energy drift lets |x| grow and the path spiral off the sphere."""
    x0 = np.array([0.0, 0.0, 1.0])
    v0 = np.array([1.0, 0.35, 0.0]); v0 = v0 - np.dot(v0, x0) * x0
    v0 = v0 / np.linalg.norm(v0) * 1.0

    t = np.arange(n + 1) * h
    true = (np.cos(t)[:, None] * x0 + np.sin(t)[:, None] * v0)

    geo = []
    x, v = x0.copy(), v0.copy()
    for _ in range(n + 1):
        geo.append(x.copy())
        w = np.linalg.norm(v)
        xn = math.cos(w * h) * x + math.sin(w * h) * (v / w)
        vn = -w * math.sin(w * h) * x + math.cos(w * h) * v
        x, v = xn, vn
    geo = np.asarray(geo)

    naive = []
    x, v = x0.copy(), v0.copy()
    for _ in range(n + 1):
        naive.append(x.copy())
        a = -x
        x = x + h * v
        v = v + h * a
    naive = np.asarray(naive)

    rad = lambda a: np.linalg.norm(a, axis=-1)
    print("  [Sphere]  radius drift |r_end-1|:")
    print(f"    geodesic (exp-map)   {abs(rad(geo)[-1]-1):.4f}")
    print(f"    naive ambient Euler  {abs(rad(naive)[-1]-1):.4f}")
    return {
        "t": _r(t, 4),
        "true_path": _r(true, 4),
        "methods": [
            {"name": "Geodesic (exp-map)", "color": GREEN,
             "path": _r(geo, 4), "radius": _r(rad(geo), 4)},
            {"name": "Naive ambient Euler", "color": ORANGE,
             "path": _r(naive, 4), "radius": _r(rad(naive), 4)},
        ],
    }


class RingEnergy(BaseModel):
    """A vortex-ring energy: low energy on the circle r=R."""
    def forward(self, x):
        r = torch.linalg.vector_norm(x, dim=-1)
        return 6.0 * (r - 1.2) ** 2


def run_fluid(disp=1000, nsteps=180, keep=2):
    """Tracers in the steady Taylor-Green vortices u = (cos x sin y, -sin x cos y),
    a Karman street of staggered point vortices in a freestream, and vorticity
    as a density: Langevin walkers on the vortex-ring energy."""
    rk = RK4Integrator()
    out = {}

    def tg(z, t):
        x, y = z[..., 0:1], z[..., 1:2]
        return torch.cat([torch.cos(x) * torch.sin(y), -torch.sin(x) * torch.cos(y)], -1)

    torch.manual_seed(3)
    P = torch.rand(disp, 2) * (2 * math.pi)
    st = {"x": P.clone()}; fr = [P.numpy().copy()]
    for i in range(nsteps):
        st = rk.step(st, 0.05, drift=tg)
        if (i + 1) % keep == 0:
            fr.append(st["x"].numpy().copy())
    Nb = 60000; sb = {"x": torch.rand(Nb, 2) * (2 * math.pi)}
    t0 = time.time()
    for _ in range(400):
        sb = rk.step(sb, 0.05, drift=tg)
    el = time.time() - t0
    out["taylor"] = {"tracers": _r(np.stack(fr, 0), 3), "L": round(2 * math.pi, 4),
                     "timing": f"{Nb // 1000}k tracers · 400 RK4 steps · {el * 1000:.0f} ms"
                               f"  ({Nb * 400 / el / 1e6:.0f}M particle-steps/s)"}
    print(f"  [Fluid] Taylor-Green: {out['taylor']['timing']}")

    sp, hgt, G, U = 1.7, 0.6, 2.6, 0.9
    vlist = []
    for k in range(-3, 4):
        vlist.append((k * sp, hgt, -G))
        vlist.append((k * sp + sp / 2, -hgt, G))
    VX = torch.tensor([[a, b] for a, b, _ in vlist]); GAM = torch.tensor([c for _, _, c in vlist])
    core = 0.10

    def karman(z, t):
        d = z[:, None, :] - VX[None, :, :]
        r2 = (d ** 2).sum(-1) + core
        ux = (-GAM / (2 * math.pi) * d[..., 1] / r2).sum(-1) + U
        uy = (GAM / (2 * math.pi) * d[..., 0] / r2).sum(-1)
        return torch.stack([ux, uy], -1)

    torch.manual_seed(4)
    Pk = torch.stack([torch.rand(disp) * 10 - 7.0, torch.rand(disp) * 3.4 - 1.7], -1)
    stk = {"x": Pk.clone()}; frk = [Pk.numpy().copy()]
    for i in range(nsteps):
        stk = rk.step(stk, 0.04, drift=karman)
        if (i + 1) % keep == 0:
            frk.append(stk["x"].numpy().copy())
    out["karman"] = {"tracers": _r(np.stack(frk, 0), 3),
                     "vortices": _r(VX.numpy(), 3), "gam": GAM.tolist()}

    torch.manual_seed(5)
    sampler = LangevinDynamics(step_size=5e-3, noise_scale=0.25)
    x0 = torch.randn(900, 2) * 0.25
    traj = sampler(RingEnergy(), x0, n_steps=900, return_trajectory=True)[:, 11::12]
    vort = traj.permute(1, 0, 2).numpy()
    out["vorticity"] = {"walkers": _r(vort, 3), "R": 1.2}
    print(f"  [Fluid] vorticity walkers {vort.shape}")
    return out


def preview(data, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": "#0b0b0f", "savefig.facecolor": "#0b0b0f",
                         "text.color": "#f5f5f7", "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(15, 5), dpi=110)

    ax = fig.add_subplot(1, 3, 1)
    ax.set_facecolor("#0b0b0f"); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_title("Ch.1 · Kepler orbit (ODE)", color="#9aa0aa")
    ax.plot(0, 0, "o", color="#ffd27f", ms=10)
    for m in data["ode"]["methods"]:
        xy = np.asarray(m["xy"])
        ax.plot(xy[:, 0], xy[:, 1], color=m["color"], lw=1.3, alpha=0.9, label=m["name"])
    ax.legend(loc="upper right", fontsize=7, frameon=False, labelcolor="#f5f5f7")

    ax = fig.add_subplot(1, 3, 2)
    ax.set_facecolor("#0b0b0f")
    ax.set_title("Ch.2 · Langevin double-well (SDE)", color="#9aa0aa")
    s = data["sde_small"]
    xg = np.asarray(s["xgrid"])
    ax.plot(xg, np.asarray(s["true_pdf"]), color=TRUTH, lw=2, ls="--", label="true p(x)")
    edges = np.asarray(s["hist_edges"]); ctr = 0.5 * (edges[1:] + edges[:-1])
    for m in s["methods"]:
        ax.plot(ctr, np.asarray(m["hist"]), color=m["color"], lw=1.6, alpha=0.9, label=m["name"])
    ax.plot(xg, 0.15 * np.asarray(s["U"]), color="#555", lw=1, alpha=0.6)
    ax.set_xlim(-2.2, 2.2); ax.tick_params(colors="#9aa0aa")
    ax.legend(loc="upper center", fontsize=7, frameon=False, labelcolor="#f5f5f7")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.set_facecolor("#0b0b0f")
    ax.set_title("Ch.3 · Geodesic on S² (3D finale)", color="#9aa0aa")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color="#333", linewidth=0.3)
    for m in data["sphere"]["methods"]:
        p = np.asarray(m["path"])
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=m["color"], lw=2, label=m["name"])
    ax.legend(loc="upper center", fontsize=7, labelcolor="#f5f5f7")
    ax.set_box_aspect((1, 1, 1)); ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0, 0, 0, 0)); a.line.set_color((0, 0, 0, 0))
        a.set_ticklabels([])

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"\nwrote {path}")


def main():
    print("Computing chapters...\n")
    data = {
        "ode": run_kepler(h=0.08, n=460),
        "threebody": run_threebody(h=0.024, periods=6.0, keep=600),
        "sde_small": run_langevin(dt=0.01, n_steps=2400, n_walkers=2000, keep=80),
        "sphere": run_sphere(h=0.06, n=210),
    }
    out = HERE / "data.json"
    payload = json.dumps(data, separators=(",", ":"))
    out.write_text(payload)
    (HERE / "data.js").write_text("window.STORY_DATA=" + payload + ";\n")
    mb = out.stat().st_size / 1e6
    print(f"\nwrote {out} and data.js  ({mb:.1f} MB)")
    preview(data, HERE / "preview.png")


if __name__ == "__main__":
    main()
