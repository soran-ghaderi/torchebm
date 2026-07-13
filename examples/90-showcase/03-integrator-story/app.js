/* TorchEBM — "Integrators in practice"
 * A narrated explainer. TorchEBM (compute.py) produced window.STORY_DATA; this
 * file renders it: Canvas2D for the ODE/SDE chapters, Three.js for the sphere,
 * all composited onto one recordable 2D canvas.
 */
"use strict";

const D = window.STORY_DATA;
const stage = document.getElementById("stage");
const ctx = stage.getContext("2d");
const W = stage.width, H = stage.height;

// ---- palette --------------------------------------------------------------
const BG = "#0b0b0f", FG = "#f5f5f7", MUTED = "#9aa0aa", LIME = "#C7FF00";
const ORANGE = "#E69F00", BLUE = "#56B4E9", GREEN = "#00D49A", TRUTH = "#e8e8ee";

// ---- helpers --------------------------------------------------------------
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const easeIO = t => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
const rand1 = i => { const x = Math.sin((i + 1) * 12.9898) * 43758.5453; return x - Math.floor(x); };

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
function rrect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
function text(s, x, y, { size = 20, color = FG, font = "sans-serif", weight = "400",
  align = "left", baseline = "alphabetic", alpha = 1 } = {}) {
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.fillStyle = color;
  ctx.textAlign = align; ctx.textBaseline = baseline;
  ctx.font = `${weight} ${size}px ${font}`;
  ctx.fillText(s, x, y);
  ctx.restore();
}
// Glow via layered translucent strokes (NOT ctx.shadowBlur, which flickers in
// headless capture). Draws the path a few times: wide+faint -> core.
function glowLine(pts, color, width, glow) {
  if (pts.length < 2) return;
  ctx.save();
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  const stroke = () => {
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
  };
  if (glow) {
    ctx.strokeStyle = hexA(color, 0.10); ctx.lineWidth = width + glow * 1.1; stroke();
    ctx.strokeStyle = hexA(color, 0.20); ctx.lineWidth = width + glow * 0.45; stroke();
  }
  ctx.strokeStyle = color; ctx.lineWidth = width; stroke();
  ctx.restore();
}
function dot(x, y, r, color, glow = 0) {
  ctx.save();
  if (glow) {
    ctx.fillStyle = hexA(color, 0.16); ctx.beginPath(); ctx.arc(x, y, r + glow * 0.55, 0, 7); ctx.fill();
    ctx.fillStyle = hexA(color, 0.30); ctx.beginPath(); ctx.arc(x, y, r + glow * 0.22, 0, 7); ctx.fill();
  }
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
  ctx.restore();
}

// ---- logo (real brand SVG, inlined so the canvas stays untainted) ---------
const LOGO_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80" viewBox="0 0 300 80" fill="none">
<defs>
<linearGradient id="ng" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#C7FF00"/><stop offset="50%" stop-color="#7affF2"/><stop offset="100%" stop-color="#2ECC71"/></linearGradient>
<linearGradient id="tg" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#C7FF00"/><stop offset="100%" stop-color="#C7FF4C"/></linearGradient>
<linearGradient id="eg" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#00c6ff"/><stop offset="100%" stop-color="#0072ff"/></linearGradient>
</defs>
<g transform="translate(5,18) scale(0.55)">
<polygon points="27.9,26.9 18.8,26.9 54.6,89.0 95.0,19.0 90.4,11.1 54.6,73.1" fill="#4DE8A0" stroke="#0a2818" stroke-width="1"/>
<polygon points="72.1,26.9 76.7,19.0 5.0,19.0 45.4,89.0 54.6,89.0 18.8,26.9" fill="#C7FF00" stroke="#0a2818" stroke-width="1"/>
<polygon points="50.0,65.1 54.6,73.1 90.4,11.0 9.6,11.1 5.0,19.0 76.7,19.0" fill="#1A7848" stroke="#0a2818" stroke-width="1"/>
</g>
<g transform="translate(80,50)">
<text x="0" y="15" font-family="Raleway,Arial,sans-serif" font-size="40" font-weight="600" fill="url(#tg)">Torch</text>
<text x="115" y="15" font-family="Raleway,Arial,sans-serif" font-size="40" font-weight="700" fill="url(#eg)">EBM</text>
</g></svg>`;
const logoImg = new Image();
let logoReady = false;
logoImg.onload = () => { logoReady = true; };
logoImg.src = "data:image/svg+xml," + encodeURIComponent(LOGO_SVG);

function drawLogo(x, y, h, withVersion) {
  const w = h * (300 / 80);
  if (logoReady) ctx.drawImage(logoImg, x, y, w, h);
  else text("∇ TorchEBM", x, y + h * 0.78, { size: h * 0.8, weight: "700", color: LIME });
  if (withVersion) text("0.6.x", x + w + 10, y + h * 0.72, { size: h * 0.62, weight: "700", color: FG, alpha: 0.9 });
  return w;
}

// Big stacked brand logo (Penrose nabla on top, "TorchEBM 🍓" below) — matches the website hero.
const NABLA_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 440" fill="none"><polygon points="163.4,145 127.4,145 268,388.5 426.6,113.8 408.6,82.7 268,326.2" fill="#4DE8A0" stroke="#0a2818" stroke-width="2" stroke-linejoin="bevel"/><polygon points="336.6,145 354.6,113.8 73.4,113.8 232,388.5 268,388.5 127.4,145" fill="#C7FF00" stroke="#0a2818" stroke-width="2" stroke-linejoin="bevel"/><polygon points="250,295 268,326.2 408.6,82.6 91.4,82.7 73.4,113.8 354.6,113.8" fill="#1A7848" stroke="#0a2818" stroke-width="2" stroke-linejoin="bevel"/></svg>`;
const nablaImg = new Image();
let nablaReady = false;
nablaImg.onload = () => { nablaReady = true; };
nablaImg.src = "data:image/svg+xml," + encodeURIComponent(NABLA_SVG);
const EBM_BLUE = "#1f9cff";

function brandWordmark(cx, y, size, alpha) {
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  ctx.font = `700 ${size}px sans-serif`;
  const wT = ctx.measureText("Torch").width, wE = ctx.measureText("EBM").width, wS = ctx.measureText("  🍓").width;
  let x = cx - (wT + wE + wS) / 2;
  ctx.fillStyle = LIME; ctx.fillText("Torch", x, y); x += wT;
  ctx.fillStyle = EBM_BLUE; ctx.fillText("EBM", x, y); x += wE;
  ctx.fillText("  🍓", x, y);
  ctx.restore();
}

function bigLogo(cx, topY, nablaW, alpha) {
  const nh = nablaW * (440 / 500);
  if (nablaReady) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.drawImage(nablaImg, cx - nablaW / 2, topY, nablaW, nh);
    ctx.restore();
  }
  brandWordmark(cx, topY + nh + 66, 54, alpha);
  return topY + nh + 90;
}

// =====================================================================
// Three.js sphere (Chapter 3) — rendered offscreen, blitted onto stage
// =====================================================================
const Sphere = (() => {
  let renderer, scene, camera, group, geoLine, naiveLine, geoHead, naiveHead, ok = false;
  const SZ = 700;
  function init() {
    if (typeof THREE === "undefined") return;
    try {
      const cv = document.createElement("canvas"); cv.width = cv.height = SZ;
      renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: true, preserveDrawingBuffer: true });
      renderer.setSize(SZ, SZ, false);
      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
      camera.position.set(1.8, 1.45, 2.95); camera.lookAt(0, 0, 0);
      group = new THREE.Group(); scene.add(group);

      // translucent sphere + wireframe shell
      const sg = new THREE.SphereGeometry(1, 48, 32);
      group.add(new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: 0x0e2230, transparent: true, opacity: 0.42 })));
      group.add(new THREE.LineSegments(new THREE.WireframeGeometry(new THREE.SphereGeometry(1, 24, 16)),
        new THREE.LineBasicMaterial({ color: 0x35525e, transparent: true, opacity: 0.6 })));

      naiveLine = mkLine(D.sphere.methods[1].path, ORANGE); group.add(naiveLine);
      geoLine = mkLine(D.sphere.methods[0].path, GREEN); group.add(geoLine);
      naiveHead = mkHead(ORANGE); group.add(naiveHead);
      geoHead = mkHead(GREEN); group.add(geoHead);
      ok = true;
    } catch (e) { ok = false; }
  }
  function mkLine(path, color) {
    const g = new THREE.BufferGeometry();
    const a = new Float32Array(path.length * 3);
    for (let i = 0; i < path.length; i++) { a[i * 3] = path[i][0]; a[i * 3 + 1] = path[i][1]; a[i * 3 + 2] = path[i][2]; }
    g.setAttribute("position", new THREE.BufferAttribute(a, 3));
    g.setDrawRange(0, 1);
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color }));
  }
  function mkHead(color) {
    return new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 16), new THREE.MeshBasicMaterial({ color }));
  }
  function setHead(mesh, path, k) { const p = path[Math.min(k, path.length - 1)]; mesh.position.set(p[0], p[1], p[2]); }
  function render(p, rot) {
    if (!ok) return null;
    const gp = D.sphere.methods[0].path, np = D.sphere.methods[1].path;
    const kg = Math.max(1, Math.floor(p * (gp.length - 1)) + 1);
    const kn = Math.max(1, Math.floor(p * (np.length - 1)) + 1);
    geoLine.geometry.setDrawRange(0, kg); naiveLine.geometry.setDrawRange(0, kn);
    setHead(geoHead, gp, kg - 1); setHead(naiveHead, np, kn - 1);
    group.rotation.y = rot;
    renderer.render(scene, camera);
    return renderer.domElement;
  }
  return { init, render, ready: () => ok };
})();

// =====================================================================
// Chapter draws
// =====================================================================
function panelTitle() {}   // titles are now rendered as KaTeX in the #mtitle overlay
const lin = x => clamp(x, 0, 1);   // linear time mapping (constant speed, no ease)

// ---- KaTeX overlays (proper math typography for titles + cards) ------------
const _mt = document.getElementById("mtitle"), _mc = document.getElementById("mcard");
let _mtLast = null, _mcLast = null;
function _krender(el, latex) {
  try { katex.render(latex, el, { throwOnError: false, displayMode: false }); }
  catch (e) { el.textContent = latex; }
}
function setOverlayTitle(latex) {
  if (latex !== _mtLast) { _mtLast = latex; if (latex) _krender(_mt, latex); else _mt.innerHTML = ""; }
  _mt.style.opacity = latex ? "1" : "0";
}
function setOverlayCard(latex, alpha) {
  if (latex !== _mcLast) { _mcLast = latex; if (latex) _krender(_mc, latex); else _mc.innerHTML = ""; }
  _mc.style.opacity = (latex ? alpha : 0).toFixed(3);
}

// ---- Ch.1: Kepler ---------------------------------------------------------
function drawKepler(t, p) {
  panelTitle("Two-body orbit");
  const R = 2.2, S = 470, cx = 560, cy = 330;
  const map = (x, y) => [cx + (x / R) * (S / 2), cy - (y / R) * (S / 2)];
  // star
  dot(...map(0, 0), 7, "#ffd27f", 26);
  const methods = D.ode.methods, N = methods[0].xy.length;
  const pa = lin(p);
  const k = Math.max(1, Math.floor(pa * (N - 1)));
  ctx.save(); rrect(cx - S / 2, cy - S / 2, S, S, 14); ctx.clip();
  for (const m of methods) {
    const pts = [];
    for (let i = 0; i <= k; i++) pts.push(map(m.xy[i][0], m.xy[i][1]));
    glowLine(pts, hexA(m.color, 0.85), 2, 8);
    dot(...pts[pts.length - 1], 4.5, m.color, 12);
  }
  ctx.restore();
  // legend
  let ly = 150;
  for (const m of methods) {
    dot(cx + S / 2 - 200, ly - 5, 5, m.color, 8);
    text(m.name, cx + S / 2 - 185, ly, { size: 15, color: FG }); ly += 26;
  }
  // energy meter
  const ex = 92, ey = 410, ew = 300, eh = 120, e0 = D.ode.energy0;
  const elo = -1.2, ehi = 0.8;
  ctx.save(); ctx.globalAlpha = 0.95;
  rrect(ex, ey, ew, eh, 10); ctx.fillStyle = "rgba(255,255,255,.03)"; ctx.fill();
  text("energy  E(t)", ex + 12, ey + 22, { size: 13, color: MUTED });
  const e0y = ey + eh - (e0 - elo) / (ehi - elo) * eh;
  ctx.strokeStyle = hexA(TRUTH, 0.25); ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(ex, e0y); ctx.lineTo(ex + ew, e0y); ctx.stroke(); ctx.setLineDash([]);
  for (const m of methods) {
    const pts = [];
    for (let i = 0; i <= k; i += 2) {
      const e = clamp(m.energy[i], elo, ehi);
      pts.push([ex + (i / (N - 1)) * ew, ey + eh - (e - elo) / (ehi - elo) * eh]);
    }
    glowLine(pts, m.color, 1.8, 0);
  }
  ctx.restore();
}

// ---- Three-body figure-eight ----------------------------------------------
const TB_BODY = ["#ffd27f", "#7ec8ff", "#ff9ec7"];
function drawThreebody(t, p) {
  panelTitle("The three-body problem");
  const M = D.threebody, e0 = M.energy0;
  const euler = M.methods[0], leap = M.methods[2];
  const F = leap.bodies.length;
  const k = Math.max(1, Math.floor(lin(p) * (F - 1)));
  const TRAIL = 175, R = 1.5, S = 330, cyP = 270;   // >= 1 orbital period -> the figure-8 stays lit (no strobe)

  function panel(method, cx, label, sub, good) {
    const map = (x, y) => [cx + x / R * (S / 2), cyP - y / R * (S / 2)];
    ctx.save(); rrect(cx - S / 2, cyP - S / 2, S, S, 14);
    ctx.strokeStyle = hexA(good ? GREEN : ORANGE, 0.22); ctx.lineWidth = 1; ctx.stroke(); ctx.clip();
    for (let b = 0; b < 3; b++) {
      const col = TB_BODY[b], pts = [];
      for (let i = Math.max(0, k - TRAIL); i <= k; i++) pts.push(map(method.bodies[i][b][0], method.bodies[i][b][1]));
      ctx.save(); ctx.globalAlpha = 0.55; glowLine(pts, col, 2, 6); ctx.restore();
      const hd = map(method.bodies[k][b][0], method.bodies[k][b][1]);
      dot(hd[0], hd[1], 4.5, col, 14);
    }
    ctx.restore();
    text(label, cx, cyP + S / 2 + 26, { size: 16, weight: "700", color: good ? GREEN : ORANGE, align: "center" });
    text(sub, cx, cyP + S / 2 + 45, { size: 13, color: MUTED, align: "center" });
  }
  panel(euler, 820, "Forward Euler", "energy grows, flies apart", false);
  panel(leap, 358, "Leapfrog (symplectic)", "energy conserved, holds", true);

  // energy meter + live numeric ranking
  const mx = 388, my = 508, mw = 470, mh = 58, elo = -1.5, ehi = -0.1;
  text("energy  E(t)", mx, my - 6, { size: 12, color: MUTED });
  const e0y = my + mh - (e0 - elo) / (ehi - elo) * mh;
  ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = hexA(TRUTH, 0.25); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(mx, e0y); ctx.lineTo(mx + mw, e0y); ctx.stroke(); ctx.restore();
  for (const m of M.methods) {
    const pts = [];
    for (let i = 0; i <= k; i += 2) pts.push([mx + (i / (F - 1)) * mw, my + mh - (clamp(m.energy[i], elo, ehi) - elo) / (ehi - elo) * mh]);
    glowLine(pts, m.color, 1.8, 0);
  }
  let lx = mx + mw + 26;
  text("|ΔE| so far", lx, my - 6, { size: 12, color: MUTED });
  let ly = my + 13;
  for (const m of M.methods) {
    dot(lx + 4, ly - 4, 4, m.color, 6);
    text(`${m.name}: ${Math.abs(m.energy[k] - e0).toFixed(3)}`, lx + 14, ly, { size: 12.5, color: FG });
    ly += 19;
  }
}

// ---- Ch.2: Langevin -------------------------------------------------------
function drawLangevin(t, p) {
  panelTitle("Langevin sampling a double well");
  const s = D.sde_small, xlo = -2.2, xhi = 2.2;
  const px0 = 200, px1 = 1080, midY = 350, denTop = 120, potBot = 560;
  const X = x => px0 + (x - xlo) / (xhi - xlo) * (px1 - px0);
  const xg = s.xgrid;
  // potential well (bottom)
  const Umax = 3.0, potH = potBot - midY - 12;
  const potY = u => potBot - clamp(u, 0, Umax) / Umax * potH;
  const upts = xg.map((x, i) => [X(x), potY(s.U[i])]);
  glowLine(upts, hexA(MUTED, 0.6), 2, 0);
  text("U(x)", X(0) - 16, potBot + 4, { size: 14, color: MUTED });
  // walkers (method 0 = Euler-Maruyama) nestled in the potential
  const m0 = s.methods[0], F = m0.frames.length;
  const pa = lin(p);
  const f = Math.min(F - 1, Math.floor(pa * (F - 1)));
  const cur = m0.frames[f];
  ctx.save(); ctx.globalAlpha = 0.5;
  for (let i = 0; i < cur.length; i += 2) {            // subsample for speed
    const x = clamp(cur[i], xlo, xhi);
    const uy = potY(interp(xg, s.U, x)) - 4 - rand1(i) * 16;
    dot(X(x), uy, 1.6, ORANGE, 0);
  }
  ctx.restore();
  // density (top): live histogram of current frame + true pdf
  const denH = midY - denTop;
  const dScale = denH / 1.0;                            // density 1.0 -> denH px
  const denY = d => midY - clamp(d, 0, 1.0) * dScale;
  // baseline
  ctx.strokeStyle = hexA(MUTED, 0.25); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px0, midY); ctx.lineTo(px1, midY); ctx.stroke();
  // live histogram (filled)
  const nb = 64, hist = new Float64Array(nb), bw = (xhi - xlo) / nb;
  for (let i = 0; i < cur.length; i++) { const b = Math.floor((clamp(cur[i], xlo, xhi) - xlo) / bw); if (b >= 0 && b < nb) hist[b]++; }
  for (let b = 0; b < nb; b++) hist[b] /= cur.length * bw;
  ctx.save(); ctx.beginPath(); ctx.moveTo(X(xlo), midY);
  for (let b = 0; b < nb; b++) { const x = xlo + (b + 0.5) * bw; ctx.lineTo(X(x), denY(hist[b])); }
  ctx.lineTo(X(xhi), midY); ctx.closePath();
  ctx.fillStyle = hexA(ORANGE, 0.30); ctx.fill();
  ctx.strokeStyle = ORANGE; ctx.lineWidth = 1.5; ctx.stroke(); ctx.restore();
  // true pdf (dashed)
  ctx.save(); ctx.setLineDash([7, 5]); ctx.strokeStyle = TRUTH; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < xg.length; i++) { const X_ = X(xg[i]), Y_ = denY(s.true_pdf[i]); i ? ctx.lineTo(X_, Y_) : ctx.moveTo(X_, Y_); }
  ctx.stroke(); ctx.restore();
  text("true density", px1 - 130, denTop + 8, { size: 14, color: TRUTH });
  // near the end, show all three methods agree
  if (p > 0.7) {
    const a = clamp((p - 0.7) / 0.2, 0, 1);
    for (let mi = 1; mi < s.methods.length; mi++) {
      const m = s.methods[mi], edges = s.hist_edges;
      ctx.save(); ctx.globalAlpha = a * 0.9; ctx.strokeStyle = m.color; ctx.lineWidth = 1.6; ctx.beginPath();
      for (let b = 0; b < m.hist.length; b++) { const x = 0.5 * (edges[b] + edges[b + 1]); const X_ = X(x), Y_ = denY(m.hist[b]); b ? ctx.lineTo(X_, Y_) : ctx.moveTo(X_, Y_); }
      ctx.stroke(); ctx.restore();
    }
    let ly = denTop + 30;
    for (const m of s.methods) { text("·  " + m.name, px0 + 10, ly, { size: 13, color: m.color, alpha: a }); ly += 20; }
  }
}
function interp(xs, ys, x) {
  const t = (x - xs[0]) / (xs[xs.length - 1] - xs[0]) * (xs.length - 1);
  const i = clamp(Math.floor(t), 0, xs.length - 2); const f = t - i;
  return lerp(ys[i], ys[i + 1], f);
}

// ---- Ch.3: Sphere ---------------------------------------------------------
function drawSphere(t, p) {
  panelTitle("Geodesic on a sphere");
  const pa = lin(p);
  const cv = Sphere.render(pa, 0.4 + t * 0.22);
  const S = 480, cx = 640 - S / 2, cy = 325 - S / 2;
  if (cv) ctx.drawImage(cv, cx, cy, S, S);
  else { text("(WebGL unavailable: open in a browser with WebGL enabled)", 640, 325, { size: 18, color: MUTED, align: "center" }); return; }
  // legend
  const items = [[GREEN, "Geodesic step: stays on the sphere"], [ORANGE, "Naive Euler: drifts off the manifold"]];
  let ly = 150;
  for (const [c, s] of items) { dot(900, ly - 5, 5, c, 8); text(s, 915, ly, { size: 15, color: FG }); ly += 26; }
}

// ---- title / outro --------------------------------------------------------
function drawTitle(t, p) {
  const a = easeIO(clamp(p / 0.4, 0, 1));
  bigLogo(640, 120, 210, a);
  text("Integrators in practice", 640, 478, { size: 40, weight: "700", color: FG, align: "center", alpha: a });
}
function drawOutro(t, p) {
  const a = easeIO(clamp(p / 0.4, 0, 1));
  bigLogo(640, 104, 182, a);
  text("Energy-based modeling and sampling in PyTorch", 640, 442, { size: 18, color: MUTED, align: "center", alpha: a });
  text("pip install torchebm", 640, 484, { size: 27, weight: "700", color: FG, align: "center", alpha: a });
  text("github.com/soran-ghaderi/torchebm", 640, 522, { size: 18, color: LIME, align: "center", alpha: a });
}

// =====================================================================
// Scenes + narration
// =====================================================================
// ===================== Deck: Optimal transport (Gaussian OT / Flow Matching) =
function lerpColor(c1, c2, t) {
  const a = parseInt(c1.slice(1), 16), b = parseInt(c2.slice(1), 16);
  const r = Math.round(lerp((a >> 16) & 255, (b >> 16) & 255, t));
  const g = Math.round(lerp((a >> 8) & 255, (b >> 8) & 255, t));
  const bl = Math.round(lerp(a & 255, b & 255, t));
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + bl).toString(16).slice(1);
}
const OT_SRC = "#00D49A", OT_TGT = "#E69F00";
function drawOTbody(tt, p, interp) {
  const d = window.OT_DATA;
  if (!d) { text("loading…", 640, 360, { align: "center", color: MUTED, size: 20 }); return; }
  const x0 = d.x0, x1 = d.x1, n = x0.length, sch = d.sched[interp], F = sch.alpha.length;
  const X = x => 640 + x * 112, Y = y => 350 - y * 108;
  const at = f => sch.alpha[f], st = f => sch.sigma[f];
  // faint interpolant paths (subsampled) — straight for linear, curved for cosine
  ctx.strokeStyle = hexA(MUTED, 0.10); ctx.lineWidth = 1;
  for (let i = 0; i < n; i += 7) {
    ctx.beginPath();
    for (let f = 0; f < F; f += 2) {
      const xx = X(at(f) * x1[i][0] + st(f) * x0[i][0]), yy = Y(at(f) * x1[i][1] + st(f) * x0[i][1]);
      f ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy);
    }
    ctx.stroke();
  }
  // faint source (green) + target (orange) reference clouds
  for (let i = 0; i < n; i += 2) {
    dot(X(x0[i][0]), Y(x0[i][1]), 1.3, hexA(OT_SRC, 0.20), 0);
    dot(X(x1[i][0]), Y(x1[i][1]), 1.3, hexA(OT_TGT, 0.20), 0);
  }
  // moving particles along the interpolant: x_t = alpha x1 + sigma x0
  const fp = lin(p), f = Math.min(F - 1, Math.floor(fp * (F - 1))), a = at(f), s = st(f);
  const col = lerpColor(OT_SRC, OT_TGT, fp);
  for (let i = 0; i < n; i++) dot(X(a * x1[i][0] + s * x0[i][0]), Y(a * x1[i][1] + s * x0[i][1]), 2.0, col, 7);
  text("source  p₀", X(-2.7), 122, { align: "center", color: OT_SRC, size: 15, weight: "700" });
  text("target  p₁", X(2.7), 122, { align: "center", color: OT_TGT, size: 15, weight: "700" });
}
function drawOTlinear(tt, p) { drawOTbody(tt, p, "linear"); }
function drawOTcosine(tt, p) { drawOTbody(tt, p, "cosine"); }

// ===================== Deck: SDE on a line (Ornstein-Uhlenbeck) =============
function drawSdeLine(tt, p) {
  const d = window.SDE_DATA;
  if (!d) { text("loading…", 640, 360, { align: "center", color: MUTED, size: 20 }); return; }
  panelTitle("SDE on a line   ·   Ornstein-Uhlenbeck   dX = −θX dt + σ dW");
  const t = d.t, N = t.length, T = t[N - 1];
  const X0 = 110, X1 = 1170, Y0 = 150, Y1 = 540, ylo = -1.7, yhi = 2.35;
  const X = tv => X0 + (tv / T) * (X1 - X0);
  const Y = xv => Y1 - (clamp(xv, ylo, yhi) - ylo) / (yhi - ylo) * (Y1 - Y0);
  ctx.strokeStyle = hexA(MUTED, 0.10); ctx.lineWidth = 1;
  for (let gx = 0; gx <= T + 0.01; gx += 2) { ctx.beginPath(); ctx.moveTo(X(gx), Y0); ctx.lineTo(X(gx), Y1); ctx.stroke(); }
  ctx.strokeStyle = hexA(MUTED, 0.25); ctx.beginPath(); ctx.moveTo(X0, Y(0)); ctx.lineTo(X1, Y(0)); ctx.stroke();
  text("time  t", X1, Y1 + 26, { align: "right", color: MUTED, size: 13 });
  text("X(t)", X0, Y0 - 12, { align: "left", color: MUTED, size: 13 });

  const k = Math.max(1, Math.floor(lin(p) * (N - 1)));
  glowLine(d.ref.map((v, i) => [X(t[i]), Y(v)]), hexA(TRUTH, 0.07), 7, 0);   // faint full guide
  ctx.save(); ctx.setLineDash([7, 5]); ctx.strokeStyle = hexA(TRUTH, 0.92); ctx.lineWidth = 2; ctx.lineJoin = "round";
  ctx.beginPath(); for (let i = 0; i <= k; i++) { const xx = X(t[i]), yy = Y(d.ref[i]); i ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); } ctx.stroke(); ctx.restore();
  for (const m of d.methods) {
    const pts = []; for (let i = 0; i <= k; i++) pts.push([X(t[i]), Y(m.path[i])]);
    glowLine(pts, m.color, 2.4, 7);
    dot(pts[pts.length - 1][0], pts[pts.length - 1][1], 4.5, m.color, 12);
  }
  let lx = X1 - 250, ly = 172;
  dot(lx + 14, ly - 5, 4, TRUTH, 0); text("reference (fine step)", lx + 28, ly, { size: 14, color: FG }); ly += 24;
  for (const m of d.methods) { dot(lx + 14, ly - 5, 4, m.color, 6); text(`${m.name}   err ${m.err.toFixed(2)}`, lx + 28, ly, { size: 13.5, color: FG }); ly += 24; }
}

// ===================== Deck: Fluid (Taylor-Green / Karman / vorticity) =======
function _frameIdx(p, F) { return Math.min(F - 1, Math.floor(lin(p) * (F - 1))); }

function drawFluidTaylor(tt, p) {
  const d = window.FLUID_DATA && window.FLUID_DATA.taylor;
  if (!d) { text("loading…", 640, 360, { align: "center", color: MUTED, size: 20 }); return; }
  panelTitle("Fluid   ·   Taylor-Green vortices (an exact Navier-Stokes flow)");
  const L = d.L, tr = d.tracers, F = tr.length, n = tr[0].length;
  const S = 414, cx = 640, cy = 312, wrap = v => ((v % L) + L) % L;
  const X = x => cx - S / 2 + wrap(x) / L * S, Y = y => cy + S / 2 - wrap(y) / L * S;
  const k = _frameIdx(p, F), kp = Math.max(0, k - 1);
  for (let i = 0; i < n; i++) {
    const a = tr[k][i], b = tr[kp][i];
    const sp = clamp(Math.hypot(a[0] - b[0], a[1] - b[1]) * 9, 0, 1);
    const base = -Math.cos(a[0]) * Math.cos(a[1]) > 0 ? "#ff9e6b" : "#6cc0ff";  // vorticity sign -> cell
    dot(X(a[0]), Y(a[1]), 1.7, lerpColor("#24405e", base, 0.4 + 0.6 * sp), 0);
  }
  text(d.timing, cx, cy + S / 2 + 26, { align: "center", color: LIME, size: 15, weight: "700" });
  text("TorchEBM batched RK4 advection", cx, cy + S / 2 + 46, { align: "center", color: MUTED, size: 12 });
}

function drawFluidKarman(tt, p) {
  const d = window.FLUID_DATA && window.FLUID_DATA.karman;
  if (!d) { text("loading…", 640, 360, { align: "center", color: MUTED, size: 20 }); return; }
  panelTitle("Fluid   ·   Karman vortex street (point-vortex model)");
  const tr = d.tracers, F = tr.length, n = tr[0].length;
  const X = x => 640 + x * 86, Y = y => 345 - y * 86;
  const k = _frameIdx(p, F), kp = Math.max(0, k - 1);
  for (let i = 0; i < d.vortices.length; i++) {
    const v = d.vortices[i], pos = d.gam[i] < 0 ? "#ff7ea8" : "#7ec8ff";
    dot(X(v[0]), Y(v[1]), 4, pos, 8);
  }
  for (let i = 0; i < n; i++) {
    const a = tr[k][i], b = tr[kp][i];
    const sp = clamp(Math.hypot(a[0] - b[0], a[1] - b[1]) * 11, 0, 1);
    dot(X(a[0]), Y(a[1]), 1.5, lerpColor("#3a7d5a", "#e8ffd0", sp), 0);
  }
}

function drawFluidVorticity(tt, p) {
  const d = window.FLUID_DATA && window.FLUID_DATA.vorticity;
  if (!d) { text("loading…", 640, 360, { align: "center", color: MUTED, size: 20 }); return; }
  panelTitle("Fluid   ·   vorticity as an energy, sampled by Langevin");
  const w = d.walkers, F = w.length, n = w[0].length, R = d.R;
  const X = x => 640 + x * 150, Y = y => 350 - y * 150;
  ctx.save(); ctx.strokeStyle = hexA(TRUTH, 0.18); ctx.setLineDash([5, 5]); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(X(0), Y(0), R * 150, 0, 7); ctx.stroke(); ctx.restore();
  const k = _frameIdx(p, F);
  for (let i = 0; i < n; i++) dot(X(w[k][i][0]), Y(w[k][i][1]), 1.8, "#ff9ec7", 6);
  text("target vorticity ring", X(0), Y(0) + R * 150 + 28, { align: "center", color: MUTED, size: 13 });
}

const TEB = "\\textcolor{#C7FF00}{\\text{TorchEBM}}";   // brand-coloured name for narration
const STORY_SCENES = [
  { dur: 3, draw: drawTitle, cards: [], hideHeader: true },
  {
    dur: 15, draw: drawKepler, title: "\\text{Two-body orbit}", cards: [
      [0, 4, TEB + "\\text{ integrates one Kepler orbit with three schemes.}"],
      [4, 8, "\\text{Forward Euler is not symplectic: energy grows, it spirals out.}"],
      [8, 11.5, "\\text{RK4 is accurate but non-symplectic: a slow secular drift.}"],
      [11.5, 15, "\\text{Symplectic Leapfrog conserves energy; the ellipse holds.}"],
    ]
  },
  {
    dur: 16, draw: drawThreebody, title: "\\text{The three-body problem}", cards: [
      [0, 4, "\\text{Three equal masses on one shared periodic orbit.}"],
      [4, 8, "\\text{A delicate solution: small errors compound quickly.}"],
      [8, 12, "\\text{Forward Euler gains energy and the orbit disintegrates.}"],
      [12, 16, TEB + "\\text{'s RK4 and symplectic Leapfrog preserve it.}"],
    ]
  },
  {
    dur: 14, draw: drawLangevin, title: "\\text{Langevin sampling a double well}", cards: [
      [0, 3.5, "\\text{Overdamped Langevin dynamics, integrated by }" + TEB + "."],
      [3.5, 7, "dX = -\\nabla U(x)\\,dt + \\sqrt{2T}\\,dW \\quad \\text{(noisy gradient flow on }U\\text{)}"],
      [7, 10.5, "\\text{Walkers cross the barrier between the two wells.}"],
      [10.5, 14, TEB + "\\text{'s SDE integrators recover } p \\propto e^{-U/T}."],
    ]
  },
  {
    dur: 14, draw: drawSphere, title: "\\text{Geodesic on a sphere}", cards: [
      [0, 3.5, "\\text{Integration on a curved manifold: the sphere } S^2."],
      [3.5, 7, "\\text{A flat-space step ignores curvature and leaves the manifold.}"],
      [7, 10.5, "\\text{A geodesic (Riemannian) step stays exactly on it.}"],
      [10.5, 14, "\\text{Riemannian HMC, via }" + TEB + "\\text{'s GeneralisedLeapfrog.}"],
    ]
  },
  { dur: 4, draw: drawOutro, cards: [], hideHeader: true },
];

// ---- decks (tabs) — each is its own animation, capturable via ?deck=NAME ----
const DECKS = { story: STORY_SCENES };
const DECK_ORDER = ["story", "sde", "ot", "fluid"];
const DECK_LABELS = { story: "Integrators", sde: "SDE on a line", ot: "Optimal transport", fluid: "Fluid" };
let activeDeck = new URLSearchParams(location.search).get("deck") || "story";
if (!DECK_ORDER.includes(activeDeck)) activeDeck = "story";
const scenes = () => DECKS[activeDeck] || DECKS.story;
const deckTotal = () => scenes().reduce((s, x) => s + x.dur, 0);

DECKS.sde = [{
  dur: 16, draw: drawSdeLine,
  title: "\\text{SDE on a line}\\;\\cdot\\;\\text{Ornstein-Uhlenbeck}\\quad dX = -\\theta X\\,dt + \\sigma\\,dW",
  cards: [
    [0, 4, "\\text{An Ornstein-Uhlenbeck process: mean reversion plus noise.}"],
    [4, 8, TEB + "\\text{ drives every scheme on one shared Brownian path.}"],
    [8, 12, "\\text{At a coarse step, each scheme departs from the reference.}"],
    [12, 16, "\\text{Euler-Maruyama, Heun, Backward-Euler-Maruyama: distinct bias.}"],
  ],
}];

DECKS.ot = [
  {
    dur: 13, draw: drawOTlinear,
    title: "\\text{Optimal transport}\\;\\cdot\\;\\text{Gaussian} \\to \\text{two moons}\\;\\;\\text{(linear interpolant)}",
    cards: [
      [0, 4, "\\text{Generative sampling as transport: a simple source } p_0 \\to \\text{a data distribution } p_1."],
      [4, 8, "\\text{A greedy optimal-transport coupling pairs the points.}"],
      [8, 13, TEB + "\\text{'s LinearInterpolant: straight, minimal-cost geodesics.}"],
    ],
  },
  {
    dur: 13, draw: drawOTcosine,
    title: "\\text{Optimal transport}\\;\\cdot\\;\\text{cosine interpolant}\\;\\;x_t = \\alpha(t)\\,x_1 + \\sigma(t)\\,x_0",
    cards: [
      [0, 4, "\\text{The same source and target, a different interpolant.}"],
      [4, 8, TEB + "\\text{'s CosineInterpolant curves the trajectories.}"],
      [8, 13, "\\text{The interpolant sets the schedule } \\alpha(t),\\,\\sigma(t)."],
    ],
  },
];

DECKS.fluid = [
  {
    dur: 12, draw: drawFluidTaylor,
    title: "\\text{Fluid}\\;\\cdot\\;\\text{Taylor-Green vortices, an exact Navier-Stokes flow}",
    cards: [
      [0, 4, "\\text{Closed streamlines circulate within each vortex cell.}"],
      [4, 8, TEB + "\\text{ advects } 60{,}000 \\text{ tracers in one batched call.}"],
      [8, 12, "\\text{Vectorised RK4: the whole ensemble integrated at once.}"],
    ],
  },
  {
    dur: 11, draw: drawFluidKarman,
    title: "\\text{Fluid}\\;\\cdot\\;\\text{Karman vortex street, a point-vortex model}",
    cards: [
      [0, 4, "\\text{Alternating vortices, staggered into a wake.}"],
      [4, 8, "\\text{Tracers wind through the point-vortex velocity field.}"],
      [8, 11, "\\text{The same }" + TEB + "\\text{ RK4 integrator, a very different field.}"],
    ],
  },
  {
    dur: 11, draw: drawFluidVorticity,
    title: "\\text{Fluid}\\;\\cdot\\;\\text{vorticity as an energy, sampled by Langevin}",
    cards: [
      [0, 4, "\\text{Treat a vorticity field as an energy } U(x)."],
      [4, 8, TEB + "\\text{'s Langevin sampler draws from } p \\propto e^{-U}."],
      [8, 11, "\\text{The walkers condense onto the vortex ring.}"],
    ],
  },
];

function activeCard(scene, t) {
  for (const c of scene.cards) if (t >= c[0] && t < c[1]) {
    const fin = clamp((t - c[0]) / 0.5, 0, 1), fout = clamp((c[1] - t) / 0.5, 0, 1);
    return { text: c[2], a: Math.min(fin, fout) };
  }
  return null;
}

// =====================================================================
// Director
// =====================================================================
function drawChrome(idx, scene, localT) {
  // header logo (skip on title/outro which draw their own big logo)
  if (!scene.hideHeader) {
    drawLogo(64, 22, 34, true);
  }
  // title + narration card, rendered as KaTeX in HTML overlays
  setOverlayTitle(scene.title || "");
  const card = activeCard(scene, localT);
  setOverlayCard(card ? card.text : "", card ? card.a : 0);
  // progress bar
  const total = scenes().slice(0, idx).reduce((s, x) => s + x.dur, 0) + localT;
  const bx = 64, bw = W - 128, by = 696;
  ctx.fillStyle = "rgba(255,255,255,.08)"; ctx.fillRect(bx, by, bw, 3);
  ctx.fillStyle = LIME; ctx.fillRect(bx, by, bw * (total / deckTotal()), 3);
}

function render(elapsed) {
  ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
  const S = scenes();
  let acc = 0, idx = 0, localT = 0;
  for (let i = 0; i < S.length; i++) {
    if (elapsed < acc + S[i].dur || i === S.length - 1) { idx = i; localT = clamp(elapsed - acc, 0, S[i].dur); break; }
    acc += S[i].dur;
  }
  const scene = S[idx];
  ctx.save(); scene.draw(localT, localT / scene.dur); ctx.restore();
  drawChrome(idx, scene, localT);
}

// ---- clock / controls -----------------------------------------------------
let elapsed = 0, playing = true, last = performance.now();
const _pt = new URLSearchParams(location.search).get("t");   // ?t=<seconds> -> freeze one frame (for screenshots)
if (_pt !== null) { elapsed = parseFloat(_pt); playing = false; }
const playBtn = document.getElementById("play"), scrub = document.getElementById("scrub");
const restartBtn = document.getElementById("restart"), recBtn = document.getElementById("rec");
const statusEl = document.getElementById("status");

function loop(now) {
  const dt = Math.min((now - last) / 1000, 0.05); last = now;
  const T = deckTotal();
  if (playing) { elapsed += dt; if (elapsed >= T) { elapsed = T; playing = false; playBtn.textContent = "▶ Play"; if (recording) stopRec(); } }
  render(elapsed);
  scrub.value = Math.round((elapsed / T) * 1000);
  requestAnimationFrame(loop);
}
playBtn.onclick = () => {
  if (elapsed >= deckTotal()) { elapsed = 0; }
  playing = !playing; playBtn.textContent = playing ? "⏸ Pause" : "▶ Play";
};
restartBtn.onclick = () => { elapsed = 0; playing = true; playBtn.textContent = "⏸ Pause"; };
scrub.oninput = () => { elapsed = (scrub.value / 1000) * deckTotal(); if (elapsed >= deckTotal()) elapsed = deckTotal() - 0.001; };

function setDeck(name) {
  if (!DECKS[name]) return;
  activeDeck = name; elapsed = 0; playing = true;
  if (playBtn) playBtn.textContent = "⏸ Pause";
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("on", b.dataset.deck === name));
}
window.setDeck = setDeck;

// ---- recording ------------------------------------------------------------
let recorder = null, chunks = [], recording = false;
function startRec() {
  try {
    const stream = stage.captureStream(30);
    const mt = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find(m => MediaRecorder.isTypeSupported(m));
    recorder = new MediaRecorder(stream, { mimeType: mt, videoBitsPerSecond: 8e6 });
    chunks = []; recorder.ondataavailable = e => e.data.size && chunks.push(e.data);
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
      a.download = "torchebm_integrators.webm"; a.click();
      statusEl.textContent = "saved torchebm_integrators.webm";
    };
    recorder.start(); recording = true; recBtn.classList.add("on"); recBtn.textContent = "■ Stop";
    elapsed = 0; playing = true; playBtn.textContent = "⏸ Pause";
    statusEl.textContent = "recording… (plays once, then auto-saves)";
  } catch (e) { statusEl.textContent = "recording unsupported: " + e.message; }
}
function stopRec() { if (recorder && recording) { recorder.stop(); recording = false; recBtn.classList.remove("on"); recBtn.textContent = "● Record .webm"; } }
recBtn.onclick = () => recording ? stopRec() : startRec();

// ---- capture hooks (used by capture.py via the DevTools protocol) ---------
window.__seek = (tt) => { playing = false; elapsed = tt; render(tt); };
window.__snap = (q) => stage.toDataURL("image/jpeg", q || 0.95);
window.__fontsReady = false;
window.__ready = () => logoReady && window.__fontsReady && (typeof THREE === "undefined" || Sphere.ready());
window.__total = deckTotal();

// ---- capture layout (?capture=1 -> canvas-only, for headless screenshots) --
if (new URLSearchParams(location.search).has("capture")) {
  document.querySelectorAll(".controls, #scrub, .hint, .tabs").forEach(e => e.style.display = "none");
  document.querySelector(".wrap").style.cssText = "padding:0;gap:0";
  document.getElementById("stagewrap").style.cssText =
    "position:fixed;left:0;top:0;width:1280px;height:720px;transform:none";
  stage.style.borderRadius = "0"; stage.style.boxShadow = "none";
} else {
  const fit = () => { document.getElementById("stagewrap").style.zoom = Math.min(1, (window.innerWidth - 28) / 1280); };
  window.addEventListener("resize", fit); fit();
}

// ---- go -------------------------------------------------------------------
Sphere.init();
// Preload KaTeX fonts (so the first captured frame has correct glyphs).
_krender(document.createElement("div"),
  "\\nabla\\,\\sqrt{2T}\\,\\theta\\,\\sigma\\;p\\propto e^{-U/T}\\;S^2\\;" + TEB);
document.fonts.ready.then(() => { window.__fontsReady = true; });
document.querySelectorAll(".tab").forEach(b => b.classList.toggle("on", b.dataset.deck === activeDeck));
requestAnimationFrame(loop);
