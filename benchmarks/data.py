r"""Shared data loading for TorchEBM benchmark results (pytest-benchmark format)."""

import glob
import json
import math
import os


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
                "device": extra.get("device", "cpu"),
                "median_ms": round(stats.get("median", 0) * 1000, 4),
                "_extra_raw": extra,
                "mean_ms": round(stats.get("mean", 0) * 1000, 4),
                "min_ms": round(stats.get("min", 0) * 1000, 4),
                "max_ms": round(stats.get("max", 0) * 1000, 4),
                "stddev_ms": round(stats.get("stddev", 0) * 1000, 4),
                "q1_ms": round(stats.get("q1", 0) * 1000, 4),
                "q3_ms": round(stats.get("q3", 0) * 1000, 4),
                "iqr_ms": round(stats.get("iqr", 0) * 1000, 4),
                "ops": round(stats.get("ops", 0), 2),
                "peak_memory_mb": extra.get("peak_memory_mb"),
                "memory_fragmentation": extra.get("memory_fragmentation"),
                "samples_per_sec": extra.get("samples_per_sec"),
                "total_flops": extra.get("total_flops"),
                "gflops_per_sec": extra.get("gflops_per_sec"),
                "acceptance_rate": extra.get("acceptance_rate"),
                "ess": extra.get("ess"),
                "ess_per_sec": extra.get("ess_per_sec"),
                "loss_value": extra.get("loss_value"),
                "loss_is_finite": extra.get("loss_is_finite"),
                "bench_mode": extra.get("mode", "eager"),
                "rounds": stats.get("rounds", 0),
                "outliers": stats.get("outliers", ""),
                "gradient_evals_per_sec": extra.get("gradient_evals_per_sec"),
                "drift_evals_per_step": extra.get("drift_evals_per_step"),
                "drift_evals_total": extra.get("drift_evals_total"),
                "memory_per_sample_kb": extra.get("memory_per_sample_kb"),
                "n_params": extra.get("n_params"),
                "n_params_m": extra.get("n_params_m"),
                "gradient_evals": extra.get("gradient_evals"),
                "n_steps": extra.get("n_steps"),
                "adaptive": extra.get("adaptive"),
                "includes_backward": extra.get("includes_backward"),
            })
    return benchmarks, run_meta


def backfill_gpu_info(benchmarks, run_meta):
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


def geomean(values):
    if not values:
        return 1.0
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


SCALE_CONFIGS = {
    "small": {"batch_size": 64, "dim": 8, "n_steps": 50},
    "medium": {"batch_size": 256, "dim": 32, "n_steps": 100},
    "large": {"batch_size": 1024, "dim": 128, "n_steps": 200},
}

BENCH_PARAMS = {
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
