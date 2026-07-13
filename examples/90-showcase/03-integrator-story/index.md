# Integrators in practice — a TorchEBM explainer

A short, narrated web animation comparing numerical integrators on an ODE, an
SDE, and a curved manifold. **TorchEBM is the compute engine**; the browser only
renders precomputed trajectories.

```
compute.py   → runs every experiment with TorchEBM → data.js (+ data.json, preview.png)
index.html   → the page
app.js       → renderer: Canvas2D (Ch.1–2) + Three.js (Ch.3) + narration timeline + recorder
vendor/      → three.min.js (r128, vendored so it works offline)
```

## Chapters
1. **ODE — Kepler orbit.** Forward Euler spirals out, RK4 slowly decays, symplectic
   **Leapfrog** holds a stable ellipse; an energy meter shows the drift.
2. **SDE — Langevin double-well.** Walkers under `dX = −∇U dt + √(2T) dW` build a
   histogram that matches the true Boltzmann `p ∝ e^(−U/T)`; all three SDE
   integrators agree.
3. **S² — geodesic.** A naive flat-space Euler step drifts off the sphere while an
   exponential-map (geodesic / Riemannian) step stays exact — RM-HMC / `GeneralisedLeapfrog`.

## Run
Just open `index.html` (double-click works offline — data is inlined in `data.js`).
Or serve it: `python -m http.server` in this folder, then visit the printed URL.

- **Play / Pause / Restart / scrubber** at the bottom.
- **Record .webm** captures the canvas to a video file. Convert to mp4 with the
  bundled ffmpeg:
  ```
  python -c "import imageio_ffmpeg as f;print(f.get_ffmpeg_exe())"   # path to ffmpeg
  <ffmpeg> -i torchebm_integrators.webm -pix_fmt yuv420p torchebm_integrators.mp4
  ```
- `index.html?t=<seconds>` freezes a single frame (used for screenshot tests).

## Regenerate the data
```
python compute.py
```
Edits to the experiments (step sizes, eccentricity, temperature, …) live in
`compute.py`; rerun it to refresh `data.js`.
