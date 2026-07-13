# Integrators vs. the Exact Solution

Two time-series scenes that race TorchEBM's numerical integrators against a
ground-truth solution, on the dark brand theme.

## What it teaches

- **Order of accuracy** (`oscillator` scene): on the undamped oscillator
  \(\ddot q = -q\), forward `EulerMaruyamaIntegrator` does not conserve energy and
  visibly drifts, while `HeunIntegrator` (2nd) and `RK4Integrator` (4th) stay on
  the exact \(\cos t\) curve.
- **Geometry-aware integration** (`riemannian` scene): on a non-separable
  Hamiltonian with metric \(M(x)=1+x^2\), the standard `LeapfrogIntegrator`
  ignores the metric and integrates the wrong dynamics, while the new
  `GeneralisedLeapfrogIntegrator` solves the implicit, geometry-aware updates and
  tracks the reference. This is the rm-HMC setting that motivated the generalised
  integrator in v0.6.0.

## Requires

Runs on TorchEBM alone. For the polished outputs, MP4 export uses
`imageio-ffmpeg` and the brand logo uses `cairosvg`
(`pip install imageio-ffmpeg cairosvg`); both degrade gracefully (GIF fallback,
text title) when absent.

## Expected output

`integrator_ode.{mp4,gif}` (oscillator) and `integrator_rmhmc.{mp4,gif}`
(riemannian), written to the `assets/` folder next to the script; add
`--still` for a final-frame PNG.
