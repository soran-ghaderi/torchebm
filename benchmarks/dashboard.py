#!/usr/bin/env python3
r"""Generate an interactive HTML dashboard from pytest-benchmark JSON history.

Usage:
    python benchmarks/dashboard.py
    python benchmarks/dashboard.py -d benchmarks/results -o benchmarks/results/dashboard.html
    python benchmarks/dashboard.py --latest-compare
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def load_benchmark_files(directory):
    pattern = os.path.join(directory, "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    runs = []
    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            if "benchmarks" not in data or "machine_info" not in data:
                continue
            data["_source_file"] = os.path.basename(fpath)
            runs.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    runs.sort(key=lambda r: r.get("datetime", ""))
    return runs


def _run_label(run):
    dt = run.get("datetime", "")[:19].replace("T", " ")
    commit = run.get("commit_info", {}).get("id", "")[:8]
    branch = run.get("commit_info", {}).get("branch", "")
    parts = []
    if commit:
        parts.append(commit)
    if branch:
        parts.append(branch)
    if dt:
        parts.append(dt)
    return " | ".join(parts) if parts else "unknown"


def _short_bench_name(fullname):
    parts = fullname.split("::")
    name = parts[-1] if parts else fullname
    if "[" in name:
        base, params = name.split("[", 1)
        return f"{base}[{params}"
    return name


def extract_all_data(runs):
    benchmarks = []
    run_meta = []
    for ri, run in enumerate(runs):
        label = _run_label(run)
        dt = run.get("datetime", "")
        commit = run.get("commit_info", {}).get("id", "")[:8]
        branch = run.get("commit_info", {}).get("branch", "")
        machine = run.get("machine_info", {})
        run_meta.append({
            "index": ri,
            "label": label,
            "datetime": dt,
            "commit": commit,
            "branch": branch,
            "machine_node": machine.get("node", ""),
            "cpu": machine.get("cpu", {}).get("brand_raw", ""),
            "system": machine.get("system", ""),
            "python": machine.get("python_version", ""),
            "source_file": run.get("_source_file", ""),
            "device": "",
            "gpu_name": "",
            "gpu_vram_gb": "",
            "cuda_version": "",
            "cudnn_version": "",
        })
        for bench in run.get("benchmarks", []):
            stats = bench.get("stats", {})
            extra = bench.get("extra_info", {})
            fullname = bench.get("fullname", bench.get("name", "unknown"))
            benchmarks.append({
                "run_index": ri,
                "run_label": label,
                "run_datetime": dt,
                "fullname": fullname,
                "short_name": _short_bench_name(fullname),
                "module": extra.get("module", "unknown"),
                "scale": extra.get("scale", "unknown"),
                "batch_size": extra.get("batch_size", 0),
                "dim": extra.get("dim", 0),
                "extra_type": extra.get("type", ""),
                "device": extra.get("device", "cpu"),
                "median_ms": round(stats.get("median", 0) * 1000, 4),                "_extra_raw": extra,                "mean_ms": round(stats.get("mean", 0) * 1000, 4),
                "min_ms": round(stats.get("min", 0) * 1000, 4),
                "max_ms": round(stats.get("max", 0) * 1000, 4),
                "stddev_ms": round(stats.get("stddev", 0) * 1000, 4),
                "q1_ms": round(stats.get("q1", 0) * 1000, 4),
                "q3_ms": round(stats.get("q3", 0) * 1000, 4),
                "iqr_ms": round(stats.get("iqr", 0) * 1000, 4),
                "ops": round(stats.get("ops", 0), 2),
                "rounds": stats.get("rounds", 0),
                "outliers": stats.get("outliers", ""),
            })
    return benchmarks, run_meta


def _backfill_gpu_info(benchmarks, run_meta):
    """Extract GPU info from benchmark extra_info and backfill into run_meta."""
    for rm in run_meta:
        run_benches = [b for b in benchmarks if b["run_index"] == rm["index"]]
        for b in run_benches:
            bi = b.get("_extra_raw", {})
            if bi.get("gpu_name"):
                rm["device"] = bi.get("device", "cuda")
                rm["gpu_name"] = bi.get("gpu_name", "")
                rm["gpu_vram_gb"] = bi.get("gpu_vram_gb", "")
                rm["cuda_version"] = bi.get("cuda_version", "")
                rm["cudnn_version"] = bi.get("cudnn_version", "")
                break
            elif bi.get("device"):
                rm["device"] = bi.get("device", "cpu")
                break


def _geomean(values):
    if not values:
        return 1.0
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


def generate_html(benchmarks, run_meta, title="TorchEBM Performance Dashboard",
                   scale_configs=None, bench_params=None):

    modules = sorted(set(b["module"] for b in benchmarks))
    scales = sorted(set(b["scale"] for b in benchmarks),
                    key=lambda s: {"small": 0, "medium": 1, "large": 2}.get(s, 3))
    n_runs = len(run_meta)

    latest_run_idx = run_meta[-1]["index"] if run_meta else -1
    latest_benchmarks = [b for b in benchmarks if b["run_index"] == latest_run_idx]

    # Per-module summary stats for latest run
    mod_summaries = {}
    for mod in modules:
        mb = [b for b in latest_benchmarks if b["module"] == mod]
        if mb:
            mod_summaries[mod] = {
                "count": len(mb),
                "fastest": min(b["median_ms"] for b in mb),
                "slowest": max(b["median_ms"] for b in mb),
                "avg_ops": sum(b["ops"] for b in mb) / len(mb),
            }

    env_info = run_meta[-1] if run_meta else {}

    # Build run option HTML once
    run_options = "".join(
        f'<option value="{r["index"]}"{"selected" if i == n_runs-1 else ""}>{r["label"]}</option>'
        for i, r in enumerate(run_meta)
    )
    run_options_prev = "".join(
        f'<option value="{r["index"]}"{"selected" if i == max(0, n_runs-2) else ""}>{r["label"]}</option>'
        for i, r in enumerate(run_meta)
    )
    module_options = "".join(f'<option value="{m}">{m}</option>' for m in modules)
    scale_options = "".join(f'<option value="{s}">{s}</option>' for s in scales)

    _device_label = env_info.get('gpu_name') or env_info.get('device', 'cpu').upper()
    _device_detail = ""
    if env_info.get('gpu_name'):
        parts = []
        if env_info.get('gpu_vram_gb'):
            parts.append(f"{env_info['gpu_vram_gb']} GB VRAM")
        if env_info.get('cuda_version'):
            parts.append(f"CUDA {env_info['cuda_version']}")
        if env_info.get('cudnn_version'):
            parts.append(f"cuDNN {env_info['cudnn_version']}")
        _device_detail = " &middot; ".join(parts)
    elif env_info.get('device') == 'cpu':
        _device_detail = env_info.get('cpu', 'N/A')[:45]

    # Build methodology info
    _scale_rows = ""
    if scale_configs:
        for sname in ["small", "medium", "large"]:
            cfg = scale_configs.get(sname, {})
            _scale_rows += (
                f'<tr><td><span class="tag">{sname}</span></td>'
                f'<td>{cfg.get("batch_size", "—")}</td>'
                f'<td>{cfg.get("dim", "—")}</td>'
                f'<td>{cfg.get("n_steps", "—")}</td></tr>'
            )
    _param_rows = ""
    if bench_params:
        for k, v in bench_params.items():
            _param_rows += f'<tr><td style="font-family:monospace">{k}</td><td>{v["value"]}</td><td style="color:var(--text-secondary)">{v["desc"]}</td></tr>'

    # Warmup/rounds info from latest run
    _latest_device = env_info.get("device", "cpu")
    if _latest_device == "cuda":
        _warmup_rounds = 10
        _bench_rounds = 30
    else:
        _warmup_rounds = 3
        _bench_rounds = 15

    _cuda_sync_text = ('<b>Yes</b> &mdash; <code>torch.cuda.synchronize()</code> after each timed call to report true GPU execution time, not async kernel launch latency'
                        if _latest_device == "cuda"
                        else '<b>N/A</b> &mdash; CPU timing is synchronous')
    _warmup_desc = ('untimed runs to stabilize GPU clocks, JIT caches, and cuDNN auto-tuner'
                     if _latest_device == "cuda"
                     else 'untimed runs to warm up CPU caches and JIT compilation')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {{
    --bg-primary: #0d1117;
    --bg-card: #161b22;
    --bg-card-alt: #1c2128;
    --border: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #656d76;
    --accent: #58a6ff;
    --green: #3fb950;
    --green-bg: rgba(63,185,80,0.15);
    --red: #f85149;
    --red-bg: rgba(248,81,73,0.15);
    --yellow: #d29922;
    --yellow-bg: rgba(210,153,34,0.15);
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
       background: var(--bg-primary); color: var(--text-primary); line-height: 1.5; }}
.dashboard {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}

.header {{ text-align: center; padding: 32px 0 16px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }}
.header h1 {{ font-size: 1.75em; font-weight: 600; }}
.header .subtitle {{ color: var(--text-secondary); font-size: 0.85em; margin-top: 4px; }}

.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
.summary-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
.summary-card .label {{ font-size: 0.72em; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }}
.summary-card .value {{ font-size: 1.4em; font-weight: 600; color: var(--accent); margin-top: 2px; }}
.summary-card .detail {{ font-size: 0.78em; color: var(--text-muted); margin-top: 2px; }}

.nav {{ display: flex; gap: 2px; border-bottom: 1px solid var(--border); margin-bottom: 20px; overflow-x: auto; }}
.nav-tab {{ padding: 10px 20px; background: none; border: none; color: var(--text-secondary);
            cursor: pointer; font-size: 0.9em; border-bottom: 2px solid transparent;
            white-space: nowrap; transition: all 0.15s; }}
.nav-tab:hover {{ color: var(--text-primary); }}
.nav-tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

.card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px;
         padding: 20px; margin-bottom: 16px; }}
.card h2 {{ font-size: 1.05em; font-weight: 600; margin-bottom: 12px; }}
.card h3 {{ font-size: 0.92em; font-weight: 600; color: var(--text-secondary); margin: 16px 0 8px;
            padding-bottom: 6px; border-bottom: 1px solid var(--border); }}

.controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 16px; }}
.controls label {{ font-size: 0.8em; color: var(--text-secondary); }}
.controls select, .controls input {{
    background: var(--bg-card-alt); color: var(--text-primary); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; font-size: 0.85em; }}
.controls select {{ min-width: 140px; }}

.pill {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75em; font-weight: 500; }}
.pill-green {{ background: var(--green-bg); color: var(--green); }}
.pill-red {{ background: var(--red-bg); color: var(--red); }}
.pill-yellow {{ background: var(--yellow-bg); color: var(--yellow); }}
.pill-neutral {{ background: rgba(139,148,158,0.15); color: var(--text-secondary); }}

table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
th {{ padding: 10px 12px; text-align: left; background: var(--bg-card-alt);
     color: var(--text-secondary); font-weight: 600; font-size: 0.8em;
     text-transform: uppercase; letter-spacing: 0.03em;
     border-bottom: 2px solid var(--border); position: sticky; top: 0; z-index: 1; cursor: pointer; }}
th:hover {{ color: var(--accent); }}
td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); }}
tr:hover td {{ background: var(--bg-card-alt); }}
.table-scroll {{ max-height: 600px; overflow-y: auto; border-radius: 6px; border: 1px solid var(--border); }}

.delta-bar {{ display: inline-block; height: 8px; border-radius: 4px; min-width: 4px; vertical-align: middle; }}
.delta-positive {{ background: var(--green); }}
.delta-negative {{ background: var(--red); }}
.chart-container {{ min-height: 380px; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media (max-width: 1000px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
.tag {{ display: inline-block; padding: 1px 8px; border-radius: 4px; font-size: 0.75em;
       background: rgba(88,166,255,0.12); color: var(--accent); margin-right: 4px; }}

.module-section {{ margin-bottom: 24px; }}
.module-section-header {{ font-size: 1.1em; font-weight: 600; color: var(--accent);
    padding: 12px 0 8px; border-bottom: 2px solid var(--border); margin-bottom: 12px;
    text-transform: capitalize; }}

.info-toggle {{ background: var(--bg-card-alt); border: 1px solid var(--border); border-radius: 6px;
    padding: 8px 16px; font-size: 0.82em; color: var(--accent); cursor: pointer; margin-bottom: 16px;
    display: inline-flex; align-items: center; gap: 6px; }}
.info-toggle:hover {{ background: var(--bg-card); }}
.info-toggle .arrow {{ transition: transform 0.2s; display: inline-block; }}
.info-toggle.open .arrow {{ transform: rotate(90deg); }}
.info-panel {{ display: none; margin-bottom: 20px; }}
.info-panel.open {{ display: block; }}
.info-panel .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
.info-panel table {{ font-size: 0.8em; }}
.info-panel td, .info-panel th {{ padding: 6px 10px; }}
</style>
</head>
<body>
<div class="dashboard">

<div class="header">
    <h1>TorchEBM Performance Dashboard</h1>
    <div class="subtitle">
        Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
        &middot; {len(latest_benchmarks)} benchmarks in latest run
        &middot; {n_runs} run{"s" if n_runs != 1 else ""}
        &middot; {len(modules)} module{"s" if len(modules) != 1 else ""}
    </div>
</div>

<!-- Summary cards row -->
<div class="summary-grid">
    <div class="summary-card">
        <div class="label">Benchmarks (Latest)</div>
        <div class="value">{len(latest_benchmarks)}</div>
        <div class="detail">{len(modules)} modules &middot; {len(scales)} scales</div>
    </div>
    <div class="summary-card">
        <div class="label">Historical Runs</div>
        <div class="value">{n_runs}</div>
        <div class="detail">{len(benchmarks)} total datapoints</div>
    </div>
    <div class="summary-card">
        <div class="label">Device</div>
        <div class="value" style="font-size:0.85em">{_device_label}</div>
        <div class="detail">{_device_detail}</div>
    </div>
    <div class="summary-card">
        <div class="label">System</div>
        <div class="value" style="font-size:0.9em">{env_info.get('system', 'N/A')}</div>
        <div class="detail">{env_info.get('cpu', 'N/A')[:45]}</div>
    </div>
    <div class="summary-card">
        <div class="label">Python / Branch</div>
        <div class="value" style="font-size:0.9em">{env_info.get('python', 'N/A')}</div>
        <div class="detail">{env_info.get('branch', 'N/A')} @ {env_info.get('commit', 'N/A')}</div>
    </div>
</div>

<!-- Methodology Info -->
<button class="info-toggle" onclick="this.classList.toggle('open'); document.getElementById('info-panel').classList.toggle('open')">
    <span class="arrow">&#9654;</span> Benchmark Configuration &amp; Methodology
</button>
<div id="info-panel" class="info-panel">
    <div class="info-panel grid-3">
        <div class="card">
            <h2>Scale Configurations</h2>
            <p style="color:var(--text-secondary);font-size:0.8em;margin-bottom:8px">
                Each benchmark runs at three input scales to measure how performance changes with problem size.
            </p>
            <table>
                <thead><tr><th>Scale</th><th>Batch Size</th><th>Dim</th><th>Steps</th></tr></thead>
                <tbody>{_scale_rows}</tbody>
            </table>
        </div>
        <div class="card">
            <h2>Measurement Methodology</h2>
            <table>
                <tbody>
                    <tr><td style="color:var(--text-secondary)">Warmup Rounds</td><td><b>{_warmup_rounds}</b> &mdash; {_warmup_desc}</td></tr>
                    <tr><td style="color:var(--text-secondary)">Timed Rounds</td><td><b>{_bench_rounds}</b> &mdash; independently timed executions used for statistics</td></tr>
                    <tr><td style="color:var(--text-secondary)">CUDA Sync</td><td>{_cuda_sync_text}</td></tr>
                    <tr><td style="color:var(--text-secondary)">Ops/sec</td><td><b>1 / mean_time</b> &mdash; one &ldquo;op&rdquo; = one complete forward pass of the benchmark function (full batch through all steps)</td></tr>
                </tbody>
            </table>
        </div>
        <div class="card">
            <h2>Benchmark Parameters</h2>
            <p style="color:var(--text-secondary);font-size:0.8em;margin-bottom:8px">
                Shared constants used across all benchmark modules.
            </p>
            <table>
                <thead><tr><th>Parameter</th><th>Value</th><th>Used In</th></tr></thead>
                <tbody>{_param_rows}</tbody>
            </table>
        </div>
    </div>
</div>

<!-- Navigation tabs -->
<div class="nav">
    <button class="nav-tab active" onclick="switchTab('overview')">Overview</button>
    <button class="nav-tab" onclick="switchTab('comparison')">Run Comparison</button>
    <button class="nav-tab" onclick="switchTab('scaling')">Scale Analysis</button>
    <button class="nav-tab" onclick="switchTab('history')">History</button>
    <button class="nav-tab" onclick="switchTab('details')">All Stats</button>
</div>

<!-- ============ TAB: OVERVIEW ============ -->
<div id="tab-overview" class="tab-content active">
    <div class="controls">
        <label>Run:</label>
        <select id="ov-run" onchange="renderOverview()">{run_options}</select>
    </div>
    <div id="overview-container"></div>
</div>

<!-- ============ TAB: COMPARISON ============ -->
<div id="tab-comparison" class="tab-content">
    <div class="card">
        <h2>Run-to-Run Comparison</h2>
        <p style="color:var(--text-secondary);font-size:0.85em;margin-bottom:12px">
            Compares benchmarks <b>within each module separately</b>. Only benchmarks present in both runs are shown.
        </p>
        <div class="controls">
            <label>Baseline (before):</label>
            <select id="cmp-before" onchange="renderComparison()" style="min-width:300px">{run_options_prev}</select>
            <label style="margin-left:12px">Current (after):</label>
            <select id="cmp-after" onchange="renderComparison()" style="min-width:300px">{run_options}</select>
        </div>
    </div>
    <div id="cmp-summary" class="summary-grid" style="margin-top:12px"></div>
    <div id="cmp-modules-container"></div>
</div>

<!-- ============ TAB: SCALING ============ -->
<div id="tab-scaling" class="tab-content">
    <div class="controls">
        <label>Module:</label>
        <select id="sc-module" onchange="renderScaling()">{module_options}</select>
        <label style="margin-left:12px">Run:</label>
        <select id="sc-run" onchange="renderScaling()">{run_options}</select>
    </div>
    <div class="grid-2">
        <div class="card">
            <h2>Absolute Time vs Scale</h2>
            <div id="sc-abs-chart" class="chart-container"></div>
        </div>
        <div class="card">
            <h2>Relative Scaling (1x = small)</h2>
            <div id="sc-rel-chart" class="chart-container"></div>
        </div>
    </div>
    <div class="card">
        <h2>Scale Comparison Table</h2>
        <div class="table-scroll">
            <table id="sc-table">
                <thead><tr>
                    <th>Benchmark</th><th>Small (ms)</th><th>Medium (ms)</th><th>Large (ms)</th>
                    <th>Med/Small</th><th>Large/Small</th>
                </tr></thead>
                <tbody></tbody>
            </table>
        </div>
    </div>
</div>

<!-- ============ TAB: HISTORY ============ -->
<div id="tab-history" class="tab-content">
    <div class="controls">
        <label>Module:</label>
        <select id="hist-module" onchange="renderHistory()">{module_options}</select>
        <label style="margin-left:12px">Scale:</label>
        <select id="hist-scale" onchange="renderHistory()">
            <option value="all">All Scales</option>
            {scale_options}
        </select>
    </div>
    <div class="card">
        <h2 id="hist-title">Performance Over Time</h2>
        <div id="hist-chart" class="chart-container" style="min-height:480px"></div>
    </div>
</div>

<!-- ============ TAB: ALL STATS ============ -->
<div id="tab-details" class="tab-content">
    <div class="controls">
        <label>Run:</label>
        <select id="dt-run" onchange="renderDetails()">{run_options}</select>
        <label style="margin-left:12px">Module:</label>
        <select id="dt-module" onchange="renderDetails()">
            <option value="all">All</option>
            {module_options}
        </select>
        <label style="margin-left:12px">Search:</label>
        <input type="text" id="dt-search" placeholder="Filter..." oninput="renderDetails()">
    </div>
    <div id="details-container"></div>
</div>

<script>
const ALL_BENCHMARKS = {json.dumps(benchmarks)};
const RUN_META = {json.dumps(run_meta)};
const MODULES = {json.dumps(modules)};
const SCALES = {json.dumps(scales)};
const SCALE_ORDER = {{"small": 0, "medium": 1, "large": 2}};

const LAYOUT = {{
    paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
    font: {{ color: '#e6edf3', size: 12 }},
    xaxis: {{ gridcolor: '#21262d', zeroline: false }},
    yaxis: {{ gridcolor: '#21262d', zeroline: false }},
    legend: {{ bgcolor: 'rgba(0,0,0,0)', font: {{ size: 11 }} }},
    margin: {{ t: 36, b: 70, l: 70, r: 20 }},
    hovermode: 'closest', bargap: 0.15, bargroupgap: 0.1,
}};
const SCALE_COLORS = {{ small: '#58a6ff', medium: '#d29922', large: '#f85149' }};
const MOD_COLORS = {{ losses: '#58a6ff', interpolants: '#3fb950', samplers: '#d29922',
                     integrators: '#f85149', models: '#bc8cff', unknown: '#8b949e' }};

function switchTab(name) {{
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById('tab-' + name).classList.add('active');
    ({{ overview: renderOverview, comparison: renderComparison,
       scaling: renderScaling, history: renderHistory, details: renderDetails }})[name]();
}}

function getRunBenches(runIdx) {{
    return ALL_BENCHMARKS.filter(b => b.run_index === parseInt(runIdx));
}}

function benchBase(shortName) {{
    // "test_interpolant[scale=small-interp=linear]" → "interpolant (linear)"
    // "test_leapfrog[scale=small]" → "leapfrog"
    let base = shortName.replace('test_', '');
    if (base.includes('[')) {{
        let [name, paramStr] = base.split('[', 2);
        paramStr = paramStr.replace(']', '');
        // Extract non-scale params
        const parts = paramStr.split('-').filter(p => !p.startsWith('scale='));
        const extras = parts.map(p => p.includes('=') ? p.split('=')[1] : p).filter(Boolean);
        return extras.length ? name + ' (' + extras.join(', ') + ')' : name;
    }}
    return base;
}}

function autoUnit(ms) {{
    if (ms >= 1000) return (ms / 1000).toFixed(2) + ' s';
    if (ms >= 1) return ms.toFixed(3) + ' ms';
    return (ms * 1000).toFixed(1) + ' \u00b5s';
}}

function pillHtml(speedup) {{
    if (speedup >= 1.05) return `<span class="pill pill-green">${{speedup.toFixed(2)}}x</span>`;
    if (speedup < 0.95) return `<span class="pill pill-red">${{speedup.toFixed(2)}}x</span>`;
    if (speedup >= 1.01) return `<span class="pill pill-yellow">${{speedup.toFixed(2)}}x</span>`;
    return `<span class="pill pill-neutral">${{speedup.toFixed(2)}}x</span>`;
}}

function deltaBar(pct, maxPct) {{
    const w = Math.min(Math.abs(pct) / (maxPct || 20) * 80, 80);
    const cls = pct >= 0 ? 'delta-positive' : 'delta-negative';
    return `<span class="delta-bar ${{cls}}" style="width:${{Math.max(w, 4)}}px"></span> ${{pct >= 0 ? '+' : ''}}${{pct.toFixed(1)}}%`;
}}

function geomean(vals) {{
    const v = vals.filter(x => x > 0 && isFinite(x));
    if (!v.length) return 1;
    return Math.exp(v.reduce((s, x) => s + Math.log(x), 0) / v.length);
}}

function ensureDiv(parentId, childId) {{
    let el = document.getElementById(childId);
    if (!el) {{
        el = document.createElement('div');
        el.id = childId;
        document.getElementById(parentId).appendChild(el);
    }}
    return el;
}}

// ================================================================
//  OVERVIEW — one section per module, each with its own chart+table
// ================================================================
function renderOverview() {{
    const runIdx = parseInt(document.getElementById('ov-run').value);
    const data = getRunBenches(runIdx);
    const container = document.getElementById('overview-container');
    container.innerHTML = '';

    for (const mod of MODULES) {{
        const modData = data.filter(b => b.module === mod);
        if (!modData.length) continue;

        const section = document.createElement('div');
        section.className = 'module-section';

        const bases = [...new Set(modData.map(b => benchBase(b.short_name)))];

        // Create chart + table
        const chartId = 'ov-chart-' + mod;
        const opsId = 'ov-ops-' + mod;
        section.innerHTML = `
            <div class="module-section-header">${{mod}} <span style="font-size:0.7em;color:var(--text-muted);font-weight:400">(${{modData.length}} benchmarks)</span></div>
            <div class="grid-2">
                <div class="card"><h2>Median Time by Scale</h2><div id="${{chartId}}" class="chart-container"></div></div>
                <div class="card"><h2>Throughput (ops/sec)</h2><div id="${{opsId}}" class="chart-container"></div></div>
            </div>
            <div class="card"><h2>Stats</h2>
            <div class="table-scroll"><table class="ov-table">
                <thead><tr><th>Benchmark</th><th>Scale</th><th>Median</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Ops/s</th><th>Rounds</th></tr></thead>
                <tbody>${{modData.sort((a,b) => benchBase(a.short_name).localeCompare(benchBase(b.short_name)) || (SCALE_ORDER[a.scale]||0)-(SCALE_ORDER[b.scale]||0))
                    .map(b => `<tr><td>${{benchBase(b.short_name)}}</td><td><span class="tag">${{b.scale}}</span></td>
                    <td><b>${{autoUnit(b.median_ms)}}</b></td><td>${{autoUnit(b.mean_ms)}}</td>
                    <td>${{autoUnit(b.stddev_ms)}}</td><td>${{autoUnit(b.min_ms)}}</td><td>${{autoUnit(b.max_ms)}}</td>
                    <td>${{b.ops.toFixed(1)}}</td><td>${{b.rounds}}</td></tr>`).join('')}}</tbody>
            </table></div></div>
        `;
        container.appendChild(section);

        // Grouped bar chart (time)
        const barTraces = [];
        for (const scale of SCALES) {{
            const sd = modData.filter(b => b.scale === scale);
            if (!sd.length) continue;
            barTraces.push({{
                x: bases.map(base => {{ const m = sd.find(b => benchBase(b.short_name) === base); return m ? base : null; }}).filter(Boolean),
                y: bases.map(base => {{ const m = sd.find(b => benchBase(b.short_name) === base); return m ? m.median_ms : null; }}).filter(x => x !== null),
                error_y: {{ type: 'data', array: bases.map(base => {{ const m = sd.find(b => benchBase(b.short_name) === base); return m ? m.stddev_ms : 0; }}).filter((_, i) => {{ const m = sd.find(b => benchBase(b.short_name) === bases[i]); return !!m; }}), visible: true, thickness: 1.5 }},
                name: scale, type: 'bar',
                marker: {{ color: SCALE_COLORS[scale] }},
                hovertemplate: '%{{x}}<br>%{{y:.4f}} ms<br>Scale: ' + scale + '<extra></extra>',
            }});
        }}
        Plotly.newPlot(chartId, barTraces, {{
            ...LAYOUT, barmode: 'group', height: 380,
            xaxis: {{ ...LAYOUT.xaxis, tickangle: -20 }},
            yaxis: {{ ...LAYOUT.yaxis, title: 'Median Time (ms)' }},
            legend: {{ ...LAYOUT.legend, orientation: 'h', y: 1.12 }},
        }}, {{ responsive: true }});

        // Ops chart
        const opsTraces = [];
        for (const scale of SCALES) {{
            const sd = modData.filter(b => b.scale === scale);
            if (!sd.length) continue;
            opsTraces.push({{
                x: bases.map(base => {{ const m = sd.find(b => benchBase(b.short_name) === base); return m ? base : null; }}).filter(Boolean),
                y: bases.map(base => {{ const m = sd.find(b => benchBase(b.short_name) === base); return m ? m.ops : null; }}).filter(x => x !== null),
                name: scale, type: 'bar',
                marker: {{ color: SCALE_COLORS[scale] }},
                hovertemplate: '%{{x}}<br>%{{y:.1f}} ops/s<extra></extra>',
            }});
        }}
        Plotly.newPlot(opsId, opsTraces, {{
            ...LAYOUT, barmode: 'group', height: 380,
            xaxis: {{ ...LAYOUT.xaxis, tickangle: -20 }},
            yaxis: {{ ...LAYOUT.yaxis, title: 'Operations / sec' }},
            legend: {{ ...LAYOUT.legend, orientation: 'h', y: 1.12 }},
        }}, {{ responsive: true }});
    }}
}}

// ================================================================
//  COMPARISON — per-module sections, no cross-module mixing
// ================================================================
function renderComparison() {{
    const beforeIdx = parseInt(document.getElementById('cmp-before').value);
    const afterIdx = parseInt(document.getElementById('cmp-after').value);
    const beforeData = getRunBenches(beforeIdx);
    const afterData = getRunBenches(afterIdx);
    const beforeMap = {{}};
    beforeData.forEach(b => beforeMap[b.fullname] = b);

    // Build comparisons grouped by module
    const compsByMod = {{}};
    afterData.forEach(b => {{
        if (!beforeMap[b.fullname]) return;
        const bb = beforeMap[b.fullname];
        const speedup = bb.median_ms / b.median_ms;
        const comp = {{
            ...b, before_median_ms: bb.median_ms, after_median_ms: b.median_ms,
            speedup, pct_change: (1 - b.median_ms / bb.median_ms) * 100,
        }};
        if (!compsByMod[b.module]) compsByMod[b.module] = [];
        compsByMod[b.module].push(comp);
    }});

    const summaryEl = document.getElementById('cmp-summary');
    const container = document.getElementById('cmp-modules-container');

    const allComps = Object.values(compsByMod).flat();
    if (!allComps.length) {{
        summaryEl.innerHTML = '<div class="card" style="grid-column:1/-1"><p style="color:var(--text-secondary)">No overlapping benchmarks between selected runs.</p></div>';
        container.innerHTML = '';
        return;
    }}

    // Overall summary
    const allSpeedups = allComps.map(c => c.speedup);
    const improved = allComps.filter(c => c.pct_change > 1).length;
    const regressed = allComps.filter(c => c.pct_change < -1).length;

    // Per-module geomeans
    const modGmHtml = Object.entries(compsByMod).map(([mod, comps]) => {{
        const gm = geomean(comps.map(c => c.speedup));
        const color = gm >= 1 ? 'var(--green)' : 'var(--red)';
        return `<div class="summary-card"><div class="label">${{mod}} geomean</div><div class="value" style="color:${{color}}">${{gm.toFixed(3)}}x</div><div class="detail">${{comps.length}} benchmarks</div></div>`;
    }}).join('');

    summaryEl.innerHTML = `
        <div class="summary-card"><div class="label">Total Compared</div><div class="value">${{allComps.length}}</div></div>
        <div class="summary-card"><div class="label">Improved (&gt;1%)</div><div class="value" style="color:var(--green)">${{improved}}</div></div>
        <div class="summary-card"><div class="label">Regressed (&gt;1%)</div><div class="value" style="color:var(--red)">${{regressed}}</div></div>
        ${{modGmHtml}}
    `;

    // Per-module sections
    container.innerHTML = '';
    for (const mod of MODULES) {{
        const comps = compsByMod[mod];
        if (!comps || !comps.length) continue;

        comps.sort((a, b) => a.pct_change - b.pct_change);
        const maxPct = Math.max(...comps.map(c => Math.abs(c.pct_change)), 1);
        const chartId = 'cmp-chart-' + mod;

        const section = document.createElement('div');
        section.className = 'module-section';
        section.innerHTML = `
            <div class="module-section-header">${{mod}}</div>
            <div class="card"><div id="${{chartId}}" class="chart-container"></div></div>
            <div class="card"><div class="table-scroll"><table>
                <thead><tr><th>Benchmark</th><th>Scale</th><th>Before (ms)</th><th>After (ms)</th><th>Speedup</th><th>Change</th><th>Delta</th></tr></thead>
                <tbody>${{comps.map(c => `<tr>
                    <td>${{benchBase(c.short_name)}}</td><td><span class="tag">${{c.scale}}</span></td>
                    <td>${{c.before_median_ms.toFixed(4)}}</td><td>${{c.after_median_ms.toFixed(4)}}</td>
                    <td>${{pillHtml(c.speedup)}}</td>
                    <td style="color:${{c.pct_change >= 0 ? 'var(--green)' : 'var(--red)'}}">${{c.pct_change >= 0 ? '+' : ''}}${{c.pct_change.toFixed(2)}}%</td>
                    <td>${{deltaBar(c.pct_change, maxPct)}}</td>
                </tr>`).join('')}}</tbody>
            </table></div></div>
        `;
        container.appendChild(section);

        // Horizontal bar chart for this module
        const barColors = comps.map(c => c.pct_change >= 0 ? '#3fb950' : '#f85149');
        Plotly.newPlot(chartId, [{{
            y: comps.map(c => benchBase(c.short_name) + ' [' + c.scale + ']'),
            x: comps.map(c => c.pct_change),
            type: 'bar', orientation: 'h',
            marker: {{ color: barColors }},
            hovertemplate: '%{{y}}<br>%{{x:+.2f}}%<br>Before: ' +
                comps.map(c => c.before_median_ms.toFixed(3)).join(',') + // not visible, just for consistency
                '<extra></extra>',
            customdata: comps.map(c => [c.before_median_ms, c.after_median_ms, c.speedup]),
            hovertemplate: '%{{y}}<br>Change: %{{x:+.2f}}%<br>Before: %{{customdata[0]:.4f}} ms<br>After: %{{customdata[1]:.4f}} ms<br>Speedup: %{{customdata[2]:.3f}}x<extra></extra>',
        }}], {{
            ...LAYOUT, showlegend: false,
            height: Math.max(250, comps.length * 32 + 60),
            xaxis: {{ ...LAYOUT.xaxis, title: 'Change (%) — positive = faster', zeroline: true, zerolinecolor: '#484f58', zerolinewidth: 2 }},
            yaxis: {{ ...LAYOUT.yaxis, automargin: true }},
        }}, {{ responsive: true }});
    }}
}}

// ================================================================
//  SCALING — per-module (already correct, just the module selector)
// ================================================================
function renderScaling() {{
    const mod = document.getElementById('sc-module').value;
    const runIdx = parseInt(document.getElementById('sc-run').value);
    const data = getRunBenches(runIdx).filter(b => b.module === mod);
    const bases = [...new Set(data.map(b => benchBase(b.short_name)))];

    // Absolute time lines
    const absTraces = bases.map(base => {{
        const pts = SCALES.map(s => data.find(b => benchBase(b.short_name) === base && b.scale === s)).filter(Boolean);
        return {{
            x: pts.map(p => p.scale), y: pts.map(p => p.median_ms),
            error_y: {{ type: 'data', array: pts.map(p => p.stddev_ms), visible: true, thickness: 1.5 }},
            name: base, type: 'scatter', mode: 'lines+markers', marker: {{ size: 8 }},
            hovertemplate: base + '<br>%{{x}}: %{{y:.4f}} ms<extra></extra>',
        }};
    }});
    Plotly.react('sc-abs-chart', absTraces, {{
        ...LAYOUT, height: 400,
        xaxis: {{ ...LAYOUT.xaxis, title: 'Scale', categoryorder: 'array', categoryarray: ['small','medium','large'] }},
        yaxis: {{ ...LAYOUT.yaxis, title: 'Median Time (ms)' }},
    }}, {{ responsive: true }});

    // Relative to small
    const relTraces = bases.map(base => {{
        const smallPt = data.find(b => benchBase(b.short_name) === base && b.scale === 'small');
        if (!smallPt || smallPt.median_ms === 0) return null;
        const pts = SCALES.map(s => data.find(b => benchBase(b.short_name) === base && b.scale === s)).filter(Boolean);
        return {{
            x: pts.map(p => p.scale), y: pts.map(p => p.median_ms / smallPt.median_ms),
            name: base, type: 'scatter', mode: 'lines+markers', marker: {{ size: 8 }},
            hovertemplate: base + '<br>%{{x}}: %{{y:.2f}}x vs small<extra></extra>',
        }};
    }}).filter(Boolean);
    Plotly.react('sc-rel-chart', relTraces, {{
        ...LAYOUT, height: 400,
        xaxis: {{ ...LAYOUT.xaxis, title: 'Scale', categoryorder: 'array', categoryarray: ['small','medium','large'] }},
        yaxis: {{ ...LAYOUT.yaxis, title: 'Multiplier (1x = small)' }},
    }}, {{ responsive: true }});

    // Table
    const tbody = document.querySelector('#sc-table tbody');
    tbody.innerHTML = bases.map(base => {{
        const sm = data.find(b => benchBase(b.short_name) === base && b.scale === 'small');
        const md = data.find(b => benchBase(b.short_name) === base && b.scale === 'medium');
        const lg = data.find(b => benchBase(b.short_name) === base && b.scale === 'large');
        const smV = sm ? sm.median_ms : null;
        return `<tr>
            <td>${{base}}</td>
            <td>${{sm ? autoUnit(sm.median_ms) : '—'}}</td>
            <td>${{md ? autoUnit(md.median_ms) : '—'}}</td>
            <td>${{lg ? autoUnit(lg.median_ms) : '—'}}</td>
            <td>${{(sm && md) ? (md.median_ms / sm.median_ms).toFixed(2) + 'x' : '—'}}</td>
            <td>${{(sm && lg) ? (lg.median_ms / sm.median_ms).toFixed(2) + 'x' : '—'}}</td>
        </tr>`;
    }}).join('');
}}

// ================================================================
//  HISTORY — must pick a module (no cross-module)
// ================================================================
function renderHistory() {{
    const mod = document.getElementById('hist-module').value;
    const scaleFilter = document.getElementById('hist-scale').value;

    let data = ALL_BENCHMARKS.filter(b => b.module === mod);
    if (scaleFilter !== 'all') data = data.filter(b => b.scale === scaleFilter);

    document.getElementById('hist-title').textContent = mod + ' — Performance Over Time' + (scaleFilter !== 'all' ? ' (' + scaleFilter + ')' : '');

    const byName = {{}};
    data.forEach(b => {{
        const key = benchBase(b.short_name) + (scaleFilter === 'all' ? ' [' + b.scale + ']' : '');
        if (!byName[key]) byName[key] = [];
        byName[key].push(b);
    }});

    const traces = Object.entries(byName).map(([name, pts]) => {{
        pts.sort((a, b) => a.run_datetime.localeCompare(b.run_datetime));
        return {{
            x: pts.map(p => p.run_label), y: pts.map(p => p.median_ms),
            error_y: {{ type: 'data', array: pts.map(p => p.stddev_ms), visible: true, thickness: 1 }},
            name, type: 'scatter', mode: 'lines+markers', marker: {{ size: 7 }},
            hovertemplate: name + '<br>%{{y:.4f}} ms<br>%{{x}}<extra></extra>',
        }};
    }});

    Plotly.react('hist-chart', traces, {{
        ...LAYOUT, height: 500,
        xaxis: {{ ...LAYOUT.xaxis, title: 'Run', tickangle: -15 }},
        yaxis: {{ ...LAYOUT.yaxis, title: 'Median Time (ms)' }},
    }}, {{ responsive: true }});
}}

// ================================================================
//  DETAILS — grouped by module with section headers
// ================================================================
function renderDetails() {{
    const runIdx = parseInt(document.getElementById('dt-run').value);
    const modFilter = document.getElementById('dt-module').value;
    const search = document.getElementById('dt-search').value.toLowerCase();
    const container = document.getElementById('details-container');

    let data = getRunBenches(runIdx);
    if (modFilter !== 'all') data = data.filter(b => b.module === modFilter);
    if (search) data = data.filter(b => b.short_name.toLowerCase().includes(search) || b.module.toLowerCase().includes(search));

    const mods = modFilter !== 'all' ? [modFilter] : MODULES.filter(m => data.some(b => b.module === m));

    container.innerHTML = mods.map(mod => {{
        const modData = data.filter(b => b.module === mod)
            .sort((a, b) => benchBase(a.short_name).localeCompare(benchBase(b.short_name)) || (SCALE_ORDER[a.scale]||0) - (SCALE_ORDER[b.scale]||0));
        if (!modData.length) return '';
        return `
            <div class="module-section">
                <div class="module-section-header">${{mod}} <span style="font-size:0.7em;color:var(--text-muted);font-weight:400">(${{modData.length}} benchmarks)</span></div>
                <div class="card"><div class="table-scroll"><table>
                    <thead><tr>
                        <th>Benchmark</th><th>Scale</th><th>Median</th><th>Mean</th>
                        <th>Std Dev</th><th>Min</th><th>Max</th><th>IQR</th>
                        <th>Ops/sec</th><th>Rounds</th><th>Outliers</th>
                    </tr></thead>
                    <tbody>${{modData.map(b => `<tr>
                        <td>${{benchBase(b.short_name)}}</td><td><span class="tag">${{b.scale}}</span></td>
                        <td><b>${{autoUnit(b.median_ms)}}</b></td><td>${{autoUnit(b.mean_ms)}}</td>
                        <td>${{autoUnit(b.stddev_ms)}}</td><td>${{autoUnit(b.min_ms)}}</td>
                        <td>${{autoUnit(b.max_ms)}}</td><td>${{autoUnit(b.iqr_ms)}}</td>
                        <td>${{b.ops.toFixed(1)}}</td><td>${{b.rounds}}</td><td>${{b.outliers}}</td>
                    </tr>`).join('')}}</tbody>
                </table></div></div>
            </div>
        `;
    }}).join('');
}}

// Initial render
renderOverview();
</script>
</div>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate TorchEBM benchmark dashboard")
    parser.add_argument(
        "-d", "--directory",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", "dashboard.html"
        ),
    )
    parser.add_argument("--latest-compare", action="store_true")
    parser.add_argument("--title", default="TorchEBM Performance Dashboard")
    args = parser.parse_args()

    runs = load_benchmark_files(args.directory)
    if not runs:
        print(f"No pytest-benchmark JSON files found in {args.directory}")
        print("Run benchmarks first:  pytest benchmarks/ --benchmark-only --benchmark-autosave")
        sys.exit(1)

    print(f"Found {len(runs)} benchmark runs")
    benchmarks, run_meta = extract_all_data(runs)
    _backfill_gpu_info(benchmarks, run_meta)
    for b in benchmarks:
        b.pop("_extra_raw", None)
    print(f"Extracted {len(benchmarks)} total benchmark datapoints across {len(run_meta)} runs")

    # Scale configs and benchmark params for the methodology panel
    _scale_configs = {
        "small": {"batch_size": 64, "dim": 8, "n_steps": 50},
        "medium": {"batch_size": 256, "dim": 32, "n_steps": 100},
        "large": {"batch_size": 1024, "dim": 128, "n_steps": 200},
    }
    _bench_params = {
        "step_size": {"value": "1e-3", "desc": "Integrators, samplers"},
        "diffusion_coeff": {"value": "0.1", "desc": "Euler-Maruyama diffusion coefficient"},
        "noise_scale": {"value": "1.0", "desc": "Langevin dynamics noise scale"},
        "n_leapfrog": {"value": "10", "desc": "HMC leapfrog steps per sample"},
        "cd_k_steps": {"value": "10", "desc": "Contrastive divergence MCMC steps"},
        "cd_step_size": {"value": "1e-2", "desc": "CD internal sampler step size"},
        "n_projections": {"value": "5", "desc": "Sliced score matching random projections"},
        "barrier_height": {"value": "2.0", "desc": "DoubleWell potential barrier height"},
        "mlp_hidden": {"value": "128", "desc": "MLP hidden dimension for energy/velocity models"},
    }

    html = generate_html(
        benchmarks, run_meta, title=args.title,
        scale_configs=_scale_configs, bench_params=_bench_params,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Dashboard written to: {args.output}")


if __name__ == "__main__":
    main()
