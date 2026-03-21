#!/usr/bin/env bash
# ======================================================================
# TorchEBM Benchmark Runner (pytest-benchmark)
# ======================================================================
#
# Automation wrapper around pytest-benchmark for consistent,
# reproducible performance tracking.
#
# Usage:
#   bash benchmarks/run.sh                   # Full run (all scales)
#   bash benchmarks/run.sh --quick           # Small scale only
#   bash benchmarks/run.sh --module losses   # Specific module
#   bash benchmarks/run.sh --baseline        # Force save as new baseline
#   bash benchmarks/run.sh --compare         # Compare latest vs baseline
#   bash benchmarks/run.sh --dashboard       # Generate HTML dashboard
#   bash benchmarks/run.sh --ci              # CI mode (regression check)
#
# Results are stored as pytest-benchmark JSON in benchmarks/results/
# ======================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

# ── Defaults ──
SCALES=""
MODULE=""
FILTER=""
DEVICE=""
FORCE_BASELINE=false
CI_MODE=false
QUICK=false
COMPARE=false
DASHBOARD=false
COMPILE=false
AMP=false
SCALING=false

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)        QUICK=true; shift ;;
        --baseline)     FORCE_BASELINE=true; shift ;;
        --ci)           CI_MODE=true; shift ;;
        --compare)      COMPARE=true; shift ;;
        --dashboard)    DASHBOARD=true; shift ;;
        --compile)      COMPILE=true; shift ;;
        --amp)          AMP=true; shift ;;
        --scaling)      SCALING=true; shift ;;
        --module)       MODULE="$2"; shift 2 ;;
        --filter)       FILTER="$2"; shift 2 ;;
        --device)       DEVICE="$2"; shift 2 ;;
        --scales)       SCALES="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash benchmarks/run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick           Run small scale only (fast smoke test)"
            echo "  --baseline        Force save results as new baseline"
            echo "  --compare         Compare latest two runs"
            echo "  --dashboard       Regenerate HTML dashboard"
            echo "  --ci              CI mode (non-interactive, exits non-zero on regression)"
            echo "  --module MODULE   Only benchmark: losses|samplers|integrators|models|interpolants"
            echo "  --filter STRING   Only benchmarks whose name contains STRING (pytest -k)"
            echo "  --compile         Include torch.compile benchmarks"
            echo "  --amp             Include mixed precision (float16) benchmarks"
            echo "  --scaling         Run batch-size scaling sweep"
            echo "  --device DEVICE   Force device: cpu|cuda"
            echo "  --scales SCALES   Space-separated scales: 'small medium large'"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $QUICK; then
    SCALES="small"
fi

# ── Setup ──
mkdir -p "$RESULTS_DIR"
cd "$REPO_ROOT"

if [[ -n "$DEVICE" ]]; then
    EFFECTIVE_DEVICE="$DEVICE"
else
    EFFECTIVE_DEVICE=$(python - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)
fi

TORCHEBM_VERSION=$(python - <<'PY'
try:
    import torchebm
    v = torchebm.__version__.split("+")[0].lstrip("v")
except Exception:
    v = "unknown"
print(v)
PY
)

# ── Dashboard-only mode ──
if $DASHBOARD; then
    echo "Generating dashboard..."
    python benchmarks/dashboard.py -d "$RESULTS_DIR" -o "$RESULTS_DIR/dashboard.html" --latest-compare
    echo "Dashboard: $RESULTS_DIR/dashboard.html"
    exit 0
fi

# ── Compare-only mode ──
if $COMPARE; then
    echo "Generating comparison from latest two runs..."
    python benchmarks/dashboard.py -d "$RESULTS_DIR" -o "$RESULTS_DIR/dashboard.html" --latest-compare
    # Also use pytest-benchmark's built-in compare
    pytest benchmarks/ --benchmark-only --benchmark-compare --benchmark-storage="$RESULTS_DIR" 2>/dev/null || true
    echo "Dashboard: $RESULTS_DIR/dashboard.html"
    exit 0
fi

echo "================================================================"
echo "  TorchEBM Benchmark Suite (pytest-benchmark)"
echo "================================================================"
echo "  Device  : $EFFECTIVE_DEVICE"
echo "  Version : $TORCHEBM_VERSION"

# ── Build pytest command ──
PYTEST_ARGS="benchmarks/ --benchmark-only --benchmark-enable"
PYTEST_ARGS="$PYTEST_ARGS --benchmark-autosave"
PYTEST_ARGS="$PYTEST_ARGS --benchmark-storage=$RESULTS_DIR"
PYTEST_ARGS="$PYTEST_ARGS --benchmark-name=short"
PYTEST_ARGS="$PYTEST_ARGS --benchmark-sort=fullname"
PYTEST_ARGS="$PYTEST_ARGS --benchmark-columns=min,max,mean,stddev,median,iqr,rounds"
PYTEST_ARGS="$PYTEST_ARGS --bench-device=$EFFECTIVE_DEVICE"

if [[ -n "$SCALES" ]]; then
    PYTEST_ARGS="$PYTEST_ARGS --bench-scales $SCALES"
    echo "  Scales  : $SCALES"
fi

# Module filter: pass to --bench-module (registry handles the rest)
if [[ -n "$MODULE" ]]; then
    PYTEST_ARGS="$PYTEST_ARGS --bench-module=$MODULE"
    echo "  Module  : $MODULE"
fi

if [[ -n "$FILTER" ]]; then
    PYTEST_ARGS="$PYTEST_ARGS -k $FILTER"
    echo "  Filter  : $FILTER"
fi

if $COMPILE; then
    PYTEST_ARGS="$PYTEST_ARGS --bench-compile"
    echo "  Compile : enabled"
fi

if $AMP; then
    PYTEST_ARGS="$PYTEST_ARGS --bench-amp"
    echo "  AMP     : enabled"
fi

echo "================================================================"
echo ""

# ── Run benchmarks ──
pytest $PYTEST_ARGS -v

# ── Save versioned copy (replace previous for same version+device) ──
LATEST_FILE=$(ls -t "$RESULTS_DIR"/*.json 2>/dev/null | head -1)
if [[ -n "$LATEST_FILE" ]]; then
    # Remove old versioned copies for this version+device
    for old in "$RESULTS_DIR"/v${TORCHEBM_VERSION}_${EFFECTIVE_DEVICE}_*.json; do
        [[ -f "$old" ]] && rm -f "$old" && echo "  Replaced old result: $(basename "$old")"
    done
    cp "$LATEST_FILE" "$RESULTS_DIR/v${TORCHEBM_VERSION}_${EFFECTIVE_DEVICE}_${TIMESTAMP}.json"
fi

# ── Baseline handling ──
BASELINE="$RESULTS_DIR/baseline_${EFFECTIVE_DEVICE}.json"
if [[ ! -f "$BASELINE" ]] || $FORCE_BASELINE; then
    if [[ -f "$BASELINE" ]]; then
        cp "$BASELINE" "$RESULTS_DIR/baseline_${EFFECTIVE_DEVICE}_${TIMESTAMP}.bak.json"
        echo "  (previous baseline archived)"
    fi
    if [[ -n "$LATEST_FILE" ]]; then
        cp "$LATEST_FILE" "$BASELINE"
        echo ""
        echo "================================================================"
        echo "  Baseline saved: $BASELINE"
        echo "  Next: make changes, then run:  bash benchmarks/run.sh"
        echo "================================================================"
    fi
else
    echo ""
    echo "================================================================"
    echo "  Comparison (use --compare or --dashboard for detailed view)"
    echo "================================================================"
    # Quick inline comparison via pytest-benchmark
    pytest benchmarks/ --benchmark-only --benchmark-compare --benchmark-storage="$RESULTS_DIR" 2>/dev/null || true
fi

# ── Generate dashboard ──
echo ""
echo "Generating dashboard..."
python benchmarks/dashboard.py -d "$RESULTS_DIR" -o "$RESULTS_DIR/dashboard.html" --latest-compare
echo "Dashboard: $RESULTS_DIR/dashboard.html"

# ── CI regression check ──
if $CI_MODE; then
    echo ""
    echo "CI: Checking for regressions..."

    GEOMEAN=$(python3 - "$RESULTS_DIR" <<'PYCHECK'
import json, glob, sys, os

results_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
# Filter to pytest-benchmark format only
runs = []
for f in files:
    try:
        data = json.load(open(f))
        if "benchmarks" in data and "machine_info" in data:
            runs.append(data)
    except Exception:
        pass

runs.sort(key=lambda r: r.get("datetime", ""))
if len(runs) < 2:
    print("1.0")
    sys.exit(0)

baseline, latest = runs[-2], runs[-1]
base_map = {b["fullname"]: b["stats"]["median"] for b in baseline["benchmarks"]}
g = 1.0
n = 0
for b in latest["benchmarks"]:
    name = b["fullname"]
    if name in base_map and base_map[name] > 0:
        g *= base_map[name] / max(b["stats"]["median"], 1e-15)
        n += 1
print(f"{g ** (1/max(n,1)):.4f}")
PYCHECK
)
    echo "  CI: Geometric mean speedup = ${GEOMEAN}x"

    REGRESSED=$(python3 -c "print('yes' if float('$GEOMEAN') < 0.95 else 'no')")
    if [[ "$REGRESSED" == "yes" ]]; then
        echo "  CI: FAILED - Regression detected (${GEOMEAN}x < 0.95x)"
        exit 1
    else
        echo "  CI: PASSED"
    fi
fi
