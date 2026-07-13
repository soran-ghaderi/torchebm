r"""Integrators vs. the exact solution — TorchEBM v0.6.0 announcement animations.

Two NeurIPS-style time-series scenes that race TorchEBM's numerical integrators
against a ground-truth solution, rendered on the dark brand theme.

scene "oscillator"  (default ground-truth = analytic cos t)
    The undamped harmonic oscillator  q'' = -q  (q(0)=1, p(0)=0, exact q=cos t).
    Forward Euler (1st), Heun (2nd) and RK4 (4th) are driven step-by-step; Euler
    does not conserve energy and visibly drifts while the higher-order methods
    stay on the exact curve — the classic "order of accuracy" story.

scene "riemannian"  (ground-truth = an independent fine-step RK4 reference)
    A *non-separable* Hamiltonian with a position-dependent metric M(x)=1+x^2
    (the Riemann-manifold-HMC setting, Girolami & Calderhead 2011). The standard
    LeapfrogIntegrator assumes a separable Hamiltonian, so it ignores the metric
    and integrates the wrong dynamics; the new GeneralisedLeapfrogIntegrator
    solves the implicit, geometry-aware updates and tracks the reference. This is
    why TorchEBM 0.6.0 adds the generalised integrator.

Every plotted curve is the genuine per-step output of the integrator (the
``integrate`` methods return only the final state, so we drive ``step`` directly).

Run:
    python examples/90-showcase/02-integrator-ode/main.py                    # both scenes, mp4+gif
    python examples/90-showcase/02-integrator-ode/main.py --scene riemannian --still
    python examples/90-showcase/02-integrator-ode/main.py --scene oscillator --no-gif
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter

from torchebm.integrators import (
    EulerMaruyamaIntegrator,
    HeunIntegrator,
    RK4Integrator,
    LeapfrogIntegrator,
    GeneralisedLeapfrogIntegrator,
)

# ----------------------------------------------------------------------------- #
# Paths
# ----------------------------------------------------------------------------- #

# Outputs land next to this script (assets/ is gitignored for examples).
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# Brand wordmark, rasterized from the repo's SVG when available; the scene
# degrades to a text title outside a repo checkout.
LOGO_SVG = ROOT / "docs" / "assets" / "images" / "logo_with_text.svg"
LOGO_PNG = OUT / "_targets" / "logo_with_text.png"

# ----------------------------------------------------------------------------- #
# Brand / NeurIPS palette (colorblind-safe Wong colors on the dark brand bg)
# ----------------------------------------------------------------------------- #

BG = "#0b0b0f"            # brand background
FG = "#f5f5f7"            # near-white text
MUTED = "#9aa0aa"         # subtitle / ticks
LIME = "#C7FF00"          # brand accent
TRUTH = "#e8e8ee"         # ground-truth reference (light, dashed)
ORANGE = "#E69F00"        # the method that goes wrong
BLUE = "#56B4E9"
GREEN = "#00D49A"         # the method that stays correct


# ----------------------------------------------------------------------------- #
# Scene 1: harmonic oscillator  z=(q, p),  dz/dt = (p, -q)  (RK integrators)
# ----------------------------------------------------------------------------- #

def osc_drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Autonomous drift f(z) = (p, -q) for the unit harmonic oscillator."""
    q, p = x[..., 0], x[..., 1]
    return torch.stack([p, -q], dim=-1)


def osc_run(integrator, x0: torch.Tensor, h: float, n_steps: int) -> np.ndarray:
    """Advance one RK integrator step-by-step; return the q-trajectory."""
    state = {"x": x0.clone()}
    qs = [float(x0[..., 0].item())]
    for i in range(n_steps):
        state = integrator.step(state, h, drift=osc_drift, t=torch.full((1,), i * h))
        qs.append(float(state["x"][..., 0].item()))
    return np.asarray(qs, dtype=np.float64)


def build_oscillator(h: float, periods: float) -> dict:
    torch.manual_seed(0)
    x0 = torch.tensor([[1.0, 0.0]])                  # q=1, p=0  ->  exact q(t)=cos(t)
    n = int(round(periods * 2 * math.pi / h))
    t = np.arange(n + 1, dtype=np.float64) * h
    exact = np.cos(t)

    methods = [
        ("Euler · 1st order", ORANGE, EulerMaruyamaIntegrator, 3.2, 4, True),
        ("Heun · 2nd order", BLUE, HeunIntegrator, 3.4, 6, True),
        ("RK4 · 4th order", GREEN, RK4Integrator, 1.9, 7, False),
    ]
    series = []
    rows = []
    for label, color, builder, lw, z, glow in methods:
        q = osc_run(builder(), x0, h, n)
        series.append({"label": label, "color": color, "traj": q,
                       "lw": lw, "z": z, "glow": glow})
        rows.append((label, abs(q[-1]), float(np.max(np.abs(q - exact)))))
    _print_table("harmonic oscillator  q''=-q", h, n, t[-1], t[-1] / (2 * math.pi), rows)

    return {
        "t": t, "ref": exact, "ref_label": "exact  cos(t)", "series": series,
        "ylim": 2.9, "ylabel": "position  q(t)",
        "subtitle": r"Numerical integrators vs. the exact solution  ·  $\ddot{q}=-q$",
        "out": "integrator_ode",
    }


# ----------------------------------------------------------------------------- #
# Scene 2: non-separable Hamiltonian, metric M(x)=1+x^2, U(x)=x^2/2
#   force    = -dH/dx   (depends on x AND p)
#   velocity =  dH/dp = p / M(x)
# Standard Leapfrog can only see drift=-dU/dx and constant mass -> wrong dynamics;
# GeneralisedLeapfrog takes (force, velocity) and is exact for the metric.
# ----------------------------------------------------------------------------- #

RM_PERIOD = 8.0   # measured fundamental period of the reference orbit (x0=1.6, p0=0)


def rm_force(x, p, t):
    invM = 1.0 / (1.0 + x ** 2)
    return -(x + 0.5 * p ** 2 * (-2.0 * x * invM ** 2) + 0.5 * (2.0 * x * invM))


def rm_velocity(x, p, t):
    return p / (1.0 + x ** 2)


def rm_full_drift(z, t):
    """First-order field (velocity, force) for the independent RK4 reference."""
    x, p = z[..., 0:1], z[..., 1:2]
    return torch.cat([rm_velocity(x, p, t), rm_force(x, p, t)], dim=-1)


def _rm_x0(x0, p0):
    return torch.tensor([[x0]]), torch.tensor([[p0]])


def rm_generalised(x0, p0, h, n) -> np.ndarray:
    integ = GeneralisedLeapfrogIntegrator()
    x, p = _rm_x0(x0, p0)
    state = {"x": x, "p": p}
    xs = [float(x0)]
    for _ in range(n):
        state = integ.step(state, h, force=rm_force, velocity=rm_velocity)
        xs.append(float(state["x"].item()))
    return np.asarray(xs, dtype=np.float64)


def rm_leapfrog(x0, p0, h, n) -> np.ndarray:
    integ = LeapfrogIntegrator()
    x, p = _rm_x0(x0, p0)
    state = {"x": x, "p": p}
    xs = [float(x0)]
    for _ in range(n):                                   # separable approx: drift=-dU/dx, mass=1
        state = integ.step(state, h, drift=lambda x, t: -x)
        xs.append(float(state["x"].item()))
    return np.asarray(xs, dtype=np.float64)


def rm_reference(x0, p0, h, n) -> np.ndarray:
    integ = RK4Integrator()
    state = {"x": torch.tensor([[x0, p0]])}
    xs = [float(x0)]
    for _ in range(n):
        state = integ.step(state, h, drift=rm_full_drift)
        xs.append(float(state["x"][..., 0].item()))
    return np.asarray(xs, dtype=np.float64)


def build_riemannian(h: float, periods: float) -> dict:
    x0, p0 = 1.6, 0.0
    n = int(round(periods * RM_PERIOD / h))
    t = np.arange(n + 1, dtype=np.float64) * h

    sub = 20
    ref = rm_reference(x0, p0, h / sub, n * sub)[::sub]   # independent fine-step truth
    glf = rm_generalised(x0, p0, h, n)
    lf = rm_leapfrog(x0, p0, h, n)

    rows = [("GeneralisedLeapfrog", abs(glf[-1]), float(np.max(np.abs(glf - ref)))),
            ("Leapfrog", abs(lf[-1]), float(np.max(np.abs(lf - ref))))]
    _print_table("non-separable H  ·  M(x)=1+x^2", h, n, t[-1], t[-1] / RM_PERIOD, rows,
                 errlabel="max err vs ref")

    series = [
        {"label": "GeneralisedLeapfrog · respects M(x)", "color": GREEN, "traj": glf,
         "lw": 2.6, "z": 7, "glow": True},
        {"label": "Leapfrog · ignores the metric", "color": ORANGE, "traj": lf,
         "lw": 2.8, "z": 5, "glow": True},
    ]
    return {
        "t": t, "ref": ref, "ref_label": "reference  ·  fine RK4", "series": series,
        "ylim": 1.95, "ylabel": "position  x(t)",
        "subtitle": r"Geometry-aware integration  ·  non-separable $H$,  $M(x)=1+x^2$",
        "out": "integrator_rmhmc",
    }


def _print_table(title, h, n, T, periods, rows, errlabel="max abs err"):
    print(f"\n  {title}   h={h}  n_steps={n}  T={T:.2f}  ({periods:.2f} periods)")
    print(f"  {'method':<24}{'end |x|':>10}{errlabel:>16}")
    print("  " + "-" * 50)
    for name, end, err in rows:
        print(f"  {name:<24}{end:>10.3f}{err:>16.3e}")
    print()


# ----------------------------------------------------------------------------- #
# Branding helpers
# ----------------------------------------------------------------------------- #

def _glow(lw: float, color: str, alpha: float = 0.35):
    """Soft outer glow so semi-thick lines pop on the dark background."""
    return [pe.Stroke(linewidth=lw + 3.5, foreground=color, alpha=alpha), pe.Normal()]


def load_logo(width_px: int = 1400):
    """Return the brand wordmark as a trimmed RGBA array (or None).

    Rasterizes the canonical ``logo_with_text.svg`` with cairosvg (cached under
    ``_targets/``) so the logo is always the real one. Falls back to the cached
    PNG, then to ``None`` (caller draws a text title instead).
    """
    try:
        import cairosvg
        LOGO_PNG.parent.mkdir(parents=True, exist_ok=True)
        cairosvg.svg2png(url=str(LOGO_SVG), write_to=str(LOGO_PNG), output_width=width_px)
    except Exception as exc:  # noqa: BLE001
        print(f"  (cairosvg unavailable: {exc}; using cached logo if present)")
    if not LOGO_PNG.exists():
        return None
    from PIL import Image
    im = Image.open(LOGO_PNG).convert("RGBA")
    bbox = im.getbbox()
    return np.asarray(im.crop(bbox) if bbox else im)


def _draw_header(fig, subtitle: str):
    """The SVG wordmark with the version snug to its right, plus a subtitle."""
    lx, lcy, logo_h = 0.088, 0.945, 0.056
    logo = load_logo()
    if logo is not None:
        aspect = logo.shape[1] / logo.shape[0]
        logo_w = logo_h * aspect                      # square figure -> aspect preserved
        logo_ax = fig.add_axes([lx, lcy - logo_h / 2, logo_w, logo_h], zorder=6)
        logo_ax.imshow(logo, aspect="auto", interpolation="antialiased")
        logo_ax.axis("off")
        logo_ax.patch.set_visible(False)
        ver_x = lx + logo_w + 0.018
    else:
        fig.text(lx, lcy, "∇ TorchEBM", color=LIME, fontsize=23,
                 fontweight="bold", ha="left", va="center")
        ver_x = 0.50
    fig.text(ver_x, lcy, "0.6.0", color=FG, fontsize=22, fontweight="bold",
             ha="left", va="center", alpha=0.9)
    fig.text(lx, lcy - 0.052, subtitle, color=MUTED, fontsize=12.5,
             ha="left", va="center")


# ----------------------------------------------------------------------------- #
# Generic render: dark theme, semi-thick glowing lines, integration-front markers
# ----------------------------------------------------------------------------- #

def render(scene: dict, *, fps: int, hold: int,
           make_mp4: bool, make_gif: bool, still: bool):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.edgecolor": MUTED,
    })

    t, ref, ylim = scene["t"], scene["ref"], scene["ylim"]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=150)   # -> 1080x1080
    fig.subplots_adjust(left=0.105, right=0.965, top=0.85, bottom=0.115)
    ax.set_facecolor(BG)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel("time  t", color=MUTED, fontsize=12)
    ax.set_ylabel(scene["ylabel"], color=MUTED, fontsize=12)
    ax.tick_params(colors=MUTED, labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_alpha(0.4)
    ax.grid(True, color=MUTED, alpha=0.12, linewidth=0.8)
    ax.axhline(0.0, color=MUTED, alpha=0.18, linewidth=0.8)

    # Ground truth: soft wide guide + crisp dashed centerline (animated, on top).
    ax.plot(t, ref, color=TRUTH, lw=7.0, alpha=0.10, solid_capstyle="round", zorder=1)
    (ref_line,) = ax.plot([], [], color=TRUTH, lw=2.0, ls=(0, (5, 4)), alpha=0.9,
                          label=scene["ref_label"], zorder=8)

    lines, dots = [], []
    for spec in scene["series"]:
        (ln,) = ax.plot([], [], color=spec["color"], lw=spec["lw"],
                        solid_capstyle="round", label=spec["label"], zorder=spec["z"])
        if spec["glow"]:
            ln.set_path_effects(_glow(spec["lw"], spec["color"]))
        (dot,) = ax.plot([], [], "o", color=spec["color"], ms=6.0,
                         zorder=spec["z"] + 10, markeredgecolor=BG, markeredgewidth=1.0)
        lines.append((ln, spec["traj"]))
        dots.append(dot)

    ax.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor="none",
              framealpha=0.6, fontsize=11, labelcolor=FG, handlelength=1.8,
              borderaxespad=0.6)
    _draw_header(fig, scene["subtitle"])
    fig.text(0.965, 0.04, "pip install torchebm", color=MUTED, fontsize=10,
             ha="right", va="center", alpha=0.8)

    n = len(t)

    def update(i):
        k = min(i, n - 1) + 1
        ref_line.set_data(t[:k], ref[:k])
        arts = [ref_line]
        for (ln, traj), dot in zip(lines, dots):
            ln.set_data(t[:k], traj[:k])
            dot.set_data([t[k - 1]], [traj[k - 1]])
            arts += [ln, dot]
        return arts

    total = n + hold
    anim = FuncAnimation(fig, update, frames=total, interval=1000 / fps, blit=False)
    base = OUT / scene["out"]

    if make_mp4:
        try:
            import imageio_ffmpeg
            plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                                  extra_args=["-pix_fmt", "yuv420p", "-crf", "18",
                                              "-preset", "slow"])
            anim.save(base.with_suffix(".mp4"), writer=writer, dpi=150)
            print(f"wrote {base.with_suffix('.mp4')}  ({total} frames @ {fps}fps)")
        except Exception as exc:  # noqa: BLE001
            print(f"!! MP4 export failed ({exc}); install imageio-ffmpeg. Falling back to GIF.")
            make_gif = True

    if make_gif:
        anim.save(base.with_suffix(".gif"), writer=PillowWriter(fps=min(fps, 25)), dpi=84)
        print(f"wrote {base.with_suffix('.gif')}  ({total} frames)")

    if still:
        update(n - 1)
        fig.savefig(base.with_suffix(".png"), dpi=150)
        print(f"wrote {base.with_suffix('.png')}")

    plt.close(fig)


# ----------------------------------------------------------------------------- #

SCENES = {"oscillator": build_oscillator, "riemannian": build_riemannian}
DEFAULT_H = {"oscillator": 0.13, "riemannian": 0.09}
DEFAULT_PERIODS = {"oscillator": 2.5, "riemannian": 3.0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=["oscillator", "riemannian", "both"], default="both")
    p.add_argument("--h", type=float, default=None, help="integrator step size (per-scene default)")
    p.add_argument("--periods", type=float, default=None, help="periods to show (per-scene default)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--hold", type=int, default=20, help="freeze frames at the end")
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--still", action="store_true", help="also save a final-frame PNG")
    args = p.parse_args()

    names = ["oscillator", "riemannian"] if args.scene == "both" else [args.scene]
    for name in names:
        h = args.h if args.h is not None else DEFAULT_H[name]
        periods = args.periods if args.periods is not None else DEFAULT_PERIODS[name]
        scene = SCENES[name](h, periods)
        render(scene, fps=args.fps, hold=args.hold,
               make_mp4=not args.no_mp4, make_gif=not args.no_gif, still=args.still)


if __name__ == "__main__":
    main()
