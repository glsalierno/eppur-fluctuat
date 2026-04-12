#!/usr/bin/env python3
"""Run the full QSAR pipeline: fetch → descriptors → model → report."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent.parent  # proteina_complexa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=SCRIPTS.parent / "config.yaml")
    args = ap.parse_args()
    cfg = str(args.config)
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    for py in (
        "fetch_toxval_data.py",
        "compute_descriptors.py",
        "build_qsar_model.py",
        "generate_qsar_report.py",
    ):
        cmd = [sys.executable, str(SCRIPTS / py), "--config", cfg]
        print(f"\n>>> {' '.join(cmd)}\n", flush=True)
        subprocess.run(cmd, check=True, cwd=str(SCRIPTS.parent), env=env)
    print("\nDone. See outputs/qsar_report.html and outputs/predictions.csv", flush=True)


if __name__ == "__main__":
    main()
