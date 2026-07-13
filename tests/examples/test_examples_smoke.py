r"""Smoke-run every example in the curriculum.

Discovers each ``examples/**/meta.yaml`` folder and executes its entrypoint as a
subprocess with ``TORCHEBM_SMOKE=1`` (examples shrink their iteration counts under
this flag) on CPU. Only the exit code is asserted; convergence is never checked.
Per-example control lives in ``meta.yaml``::

    ci:
      skip: true        # exclude from the smoke run (heavy showcases)
      timeout: 240      # seconds, default 120

Deselected by default (``-m "not examples"`` in addopts); run explicitly with
``pytest -m examples tests/examples``.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
EXCLUDE_DIRS = {"__pycache__", "vendor", "assets", "configs"}
DEFAULT_TIMEOUT = 120.0


@dataclass
class ExampleCase:
    folder: Path
    entrypoint: str
    ci: dict[str, Any]

    @property
    def slug(self) -> str:
        return self.folder.relative_to(EXAMPLES_DIR).as_posix()


def _discover() -> list[ExampleCase]:
    cases: list[ExampleCase] = []
    for meta_path in sorted(EXAMPLES_DIR.rglob("meta.yaml")):
        parts = meta_path.relative_to(EXAMPLES_DIR).parts[:-1]
        if any(p in EXCLUDE_DIRS or p.startswith((".", "_")) for p in parts):
            continue
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        cases.append(
            ExampleCase(
                folder=meta_path.parent,
                entrypoint=str(meta.get("entrypoint", "main.py")),
                ci=meta.get("ci") or {},
            )
        )
    return cases


@pytest.mark.examples
@pytest.mark.parametrize("case", _discover(), ids=lambda c: c.slug)
def test_example_runs(case: ExampleCase) -> None:
    if case.ci.get("skip"):
        pytest.skip("ci: skip in meta.yaml")
    env = {**os.environ, "TORCHEBM_SMOKE": "1", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, str(case.folder / case.entrypoint)],
        cwd=REPO_ROOT,
        env=env,
        timeout=float(case.ci.get("timeout", DEFAULT_TIMEOUT)),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{case.slug} failed:\n{proc.stderr[-2000:]}"
