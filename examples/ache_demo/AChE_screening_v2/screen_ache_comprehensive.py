#!/usr/bin/env python3
"""
Comprehensive AChE (4EY7) virtual screening: 25 ligands, ranked CSV, top-5 interaction analysis,
and AChE_report.html with multiple 3D representations (3Dmol.js).

Requires parent package modules: screen_alkyl_phosphates.py, interaction_analysis.py
and Open Babel (obabel) on PATH.

AutoDock Vina (Python API) is CPU-only — no CUDA. See script docstring in screen_alkyl_phosphates.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_V2 = Path(__file__).resolve().parent
_ROOT = _V2.parent
for p in (_ROOT, _V2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from screen_alkyl_phosphates import (  # noqa: E402
    DEFAULT_BOX,
    DEFAULT_PDB,
    run_screening,
    safe_name,
    write_docking_report_html,
)

try:
    from interaction_analysis import prepare_and_analyze
except ImportError:
    prepare_and_analyze = None

from ache_report_viz import write_ache_report_html  # noqa: E402


def _merge_cas(df: pd.DataFrame, ligands_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(ligands_csv)
    if "cas" not in meta.columns:
        return df
    meta["name"] = meta["name"].astype(str).str.strip()
    df = df.copy()
    df["name_key"] = df["name"].astype(str).str.strip()
    m = meta[["name", "cas"]].rename(columns={"name": "name_key"})
    out = df.merge(m, on="name_key", how="left")
    out = out.drop(columns=["name_key"])
    return out


def pdbqt_to_pdb(pdbqt: Path, pdb: Path) -> None:
    subprocess.run(["obabel", str(pdbqt), "-O", str(pdb)], check=True, capture_output=True, text=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="AChE comprehensive screening (v2)")
    ap.add_argument("--ligands", type=Path, default=here / "ligands.csv")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=here,
        help="Working directory for PDB, CSV, and AChE_report.html (default: this folder)",
    )
    ap.add_argument("--pdb-id", default=DEFAULT_PDB)
    ap.add_argument("--box-size", type=float, default=DEFAULT_BOX)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--n-poses", type=int, default=5)
    ap.add_argument("--vina-cpu", type=int, default=max(1, os.cpu_count() or 4))
    ap.add_argument("--center", choices=["e20", "ser203"], default="e20")
    ap.add_argument("--top-n", type=int, default=5, help="Top N for interactions + report")
    ap.add_argument("--skip-interactions", action="store_true")
    args = ap.parse_args()

    if shutil.which("obabel") is None:
        raise SystemExit("Open Babel 'obabel' not found in PATH.")

    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    df, _receptor_pdbqt, _center = run_screening(
        args.ligands,
        work,
        args.pdb_id,
        args.box_size,
        args.exhaustiveness,
        args.n_poses,
        args.vina_cpu,
        args.center,
    )
    df = _merge_cas(df, args.ligands)

    csv_path = work / "docking_results.csv"
    df.to_csv(csv_path, index=False)
    print("\n=== Docking results (sorted, kcal/mol) ===\n", flush=True)
    cols = [c for c in ["name", "cas", "affinity_kcal_mol", "error"] if c in df.columns]
    print(df[cols].to_string(index=False), flush=True)
    print(f"\nSaved: {csv_path}", flush=True)

    clean_pdb = work / f"{args.pdb_id}_receptor_clean.pdb"
    top = df[df["affinity_kcal_mol"].notna()].head(args.top_n)

    pose_paths: list[Path] = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        p = Path(row["pose_pdbqt"])
        if p.is_file():
            pose_paths.append(p)
            dst = work / f"top{rank}_{safe_name(str(row['name']))}.pdbqt"
            shutil.copy(p, dst)
            pdb_out = work / f"top{rank}_{safe_name(str(row['name']))}.pdb"
            try:
                pdbqt_to_pdb(p, pdb_out)
                print(f"Pose PDB: {pdb_out}", flush=True)
            except subprocess.CalledProcessError:
                print(f"Warning: could not convert {p} to PDB", flush=True)
            print(f"Top pose: {dst}", flush=True)

    simple = work / "docking_report_simple.html"
    if pose_paths:
        write_docking_report_html(clean_pdb, pose_paths, simple, df)
        print(f"Simple multi-ligand view: {simple}", flush=True)

    if (
        args.skip_interactions
        or prepare_and_analyze is None
        or not len(pose_paths)
    ):
        if prepare_and_analyze is None:
            print("Install MDAnalysis + RDKit for interaction report.", file=sys.stderr)
        return

    ia_root = work / "interaction_analysis"
    ia_root.mkdir(parents=True, exist_ok=True)
    results = []
    kept_poses: list[Path] = []
    for _, row in top.iterrows():
        pq = Path(row["pose_pdbqt"])
        if not pq.is_file():
            continue
        name = str(row["name"])
        aff = float(row["affinity_kcal_mol"])
        try:
            results.append(
                prepare_and_analyze(
                    clean_pdb,
                    pq,
                    name,
                    aff,
                    ia_root / safe_name(name),
                    hydrophobic_cutoff=5.0,
                    pocket_radius=5.0,
                    active_site_radius=8.0,
                )
            )
            kept_poses.append(pq)
        except Exception as exc:
            print(f"Interaction analysis failed for {name}: {exc}", file=sys.stderr)

    if not results:
        return

    print(f"\n=== Top {len(results)} — interactions (H-bonds / hydrophobic ≤5 Å / π–π) ===\n", flush=True)
    for r in results:
        print(
            f"{r.name:36s}  {r.affinity_kcal_mol:7.2f}  "
            f"HB {r.n_hbonds:3d}  HP {r.n_hydrophobic_contacts:3d}  ππ {r.n_pipi:2d}",
            flush=True,
        )

    report = work / "AChE_report.html"
    try:
        write_ache_report_html(clean_pdb, results, kept_poses, report)
        print(f"\nInteractive report: {report.resolve()}", flush=True)
    except Exception as exc:
        print(f"Failed to write AChE_report.html: {exc}", file=sys.stderr)
        raise

    if report.is_file() and os.environ.get("DISPLAY"):
        try:
            import webbrowser

            webbrowser.open(report.as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    main()
