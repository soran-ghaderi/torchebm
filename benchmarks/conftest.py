import gc
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.toml")


def _load_config():
    if not os.path.isfile(_CONFIG_PATH) or tomllib is None:
        return {}
    try:
        with open(_CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


CONFIG = _load_config()

SCALES = {
    "small": {"batch_size": 64, "dim": 8, "n_steps": 50},
    "medium": {"batch_size": 256, "dim": 32, "n_steps": 100},
    "large": {"batch_size": 1024, "dim": 128, "n_steps": 200},
}


def is_module_enabled(module_name):
    return CONFIG.get("modules", {}).get(module_name, True)


def is_benchmark_excluded(bench_name):
    excluded = set(CONFIG.get("exclude", {}).get("benchmarks", []))
    return bench_name in excluded


def pytest_addoption(parser):
    parser.addoption(
        "--bench-device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for benchmarks (default: cuda, GPU-first)",
    )
    parser.addoption(
        "--bench-dtype",
        default="float32",
        choices=["float32", "float64"],
    )
    parser.addoption(
        "--bench-scales",
        nargs="+",
        default=["small", "medium", "large"],
        choices=["small", "medium", "large"],
    )
    parser.addoption(
        "--bench-module",
        default=None,
        choices=["losses", "samplers", "integrators", "models", "interpolants"],
    )
    parser.addoption(
        "--bench-compile",
        action="store_true",
        default=False,
        help="Also benchmark torch.compile variants",
    )
    parser.addoption(
        "--bench-amp",
        action="store_true",
        default=False,
        help="Also benchmark mixed precision (float16) variants",
    )


@pytest.fixture(scope="session")
def bench_device(request):
    dev = request.config.getoption("--bench-device")
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for GPU benchmarks but not available")
    return torch.device(dev)


@pytest.fixture(scope="session")
def bench_dtype(request):
    d = request.config.getoption("--bench-dtype")
    return torch.float32 if d == "float32" else torch.float64


def _get_gpu_info(device):
    if device.type != "cuda":
        return {}
    idx = device.index or 0
    props = torch.cuda.get_device_properties(idx)
    return {
        "gpu_name": props.name,
        "gpu_vram_gb": round(props.total_memory / (1024 ** 3), 2),
        "gpu_arch": f"sm_{props.major}{props.minor}",
        "cuda_version": torch.version.cuda or "",
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "",
    }


@pytest.fixture(scope="session")
def _gpu_info(bench_device):
    return _get_gpu_info(bench_device)


@pytest.fixture(scope="session", autouse=True)
def _gpu_benchmark_env(bench_device):
    """Set up CUDA environment for benchmarking."""
    if bench_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.randn(1, device=bench_device)  # warm up CUDA context


@pytest.fixture(autouse=True)
def _cuda_sync_benchmark(benchmark, bench_device, _gpu_info):
    """Patch benchmark.pedantic to add CUDA synchronization and inject device info."""
    benchmark.extra_info["device"] = bench_device.type
    benchmark.extra_info.update(_gpu_info)

    try:
        import torchebm
        benchmark.extra_info["torchebm_version"] = getattr(torchebm, "__version__", "")
    except Exception:
        pass
    benchmark.extra_info["torch_version"] = torch.__version__

    if bench_device.type != "cuda":
        yield
        return
    _orig_pedantic = benchmark.pedantic

    def _synced_pedantic(target, args=(), kwargs=None, **pedantic_kwargs):
        _dev = bench_device

        def _synced_target(*a, **kw):
            result = target(*a, **kw)
            torch.cuda.synchronize(_dev)
            return result

        result = _orig_pedantic(_synced_target, args=args, kwargs=kwargs, **pedantic_kwargs)
        torch.cuda.synchronize(_dev)
        peak = torch.cuda.max_memory_allocated(_dev) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(_dev) / (1024 ** 2)
        benchmark.extra_info["peak_memory_mb"] = round(peak, 2)
        benchmark.extra_info["memory_reserved_mb"] = round(reserved, 2)
        if peak > 0:
            benchmark.extra_info["memory_fragmentation"] = round(reserved / peak, 2)
        return result

    benchmark.pedantic = _synced_pedantic
    yield


def _cuda_setup(device):
    """Pre-benchmark CUDA cleanup."""
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def make_pedantic_setup(device):
    """Return a setup callable for benchmark.pedantic that handles CUDA sync."""
    def setup():
        _cuda_setup(device)
    return setup


def get_bench_params(device):
    """Return (warmup_rounds, rounds, iterations) for benchmark.pedantic."""
    if device.type == "cuda":
        return 10, 30, 1
    return 3, 15, 1
