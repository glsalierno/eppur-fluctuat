#!/usr/bin/env python3
"""
Dock ligands (SMILES) against a rigid receptor using AutoDock Vina (Python API).

Notes
-----
- Vina is CPU-only in the official Python binding; `docking.gpu` in config is reserved for future engines.
- Requires Open Babel (`obabel`) on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from docking_common import (
    load_config,
    log_path,
    project_root_from_config,
    resolve_center,
    safe_name,
    smiles_to_pdbqt,
    vina_cpu_count,
)

try:
    from vina import Vina
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install vina: pip install vina") from exc


def run_docking(
    vina_engine: Vina,
    ligand_pdbqt: str,
    exhaustiveness: int,
    n_poses: int,
    out_pose: Path,
) -> dict:
    """
    Run AutoDock Vina docking for one ligand PDBQT using an initialized ``Vina`` instance.

    The receptor and maps must already be configured on ``vina_engine``.

    Parameters
    ----------
    vina_engine : vina.Vina
        Configured Vina object (receptor + maps set).
    ligand_pdbqt : str
        Path to ligand PDBQT.
    exhaustiveness : int
        Vina exhaustiveness parameter.
    n_poses : int
        Number of poses to generate (``num_modes``).
    out_pose : pathlib.Path
        Path for the best pose PDBQT output.

    Returns
    -------
    dict
        Keys: ``affinity`` (float, best-mode score, kcal/mol), ``poses`` (list of pose energy
        arrays as returned by ``Vina.energies``), ``pose_path`` (str, resolved path to written pose).

    Raises
    ------
    RuntimeError
        If Vina raises during docking (depends on Vina Python binding).

    Examples
    --------
    Used internally after ``set_receptor`` and ``compute_vina_maps``::

        result = run_docking(v, str(lig_path), exh, n_poses, out_pose_path)
        score = result["affinity"]
    """
    vina_engine.set_ligand_from_file(ligand_pdbqt)
    vina_engine.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    energies = vina_engine.energies(n_poses=n_poses)
    best = (
        float(np.asarray(energies[0]).flatten()[0]) if len(energies) else float("nan")
    )
    vina_engine.write_pose(str(out_pose), overwrite=True)
    return {
        "affinity": best,
        "poses": list(energies),
        "pose_path": str(out_pose.resolve()),
    }


def main() -> None:
    """
    CLI entry: dock all ligands listed in ``config.yaml`` paths with AutoDock Vina (CPU).

    Raises
    ------
    SystemExit
        On missing inputs, missing ``obabel``, or invalid ligand table columns.
    ValueError
        If the ligands CSV lacks required columns.
    """
    ap = argparse.ArgumentParser(
        description="Dock ligands with AutoDock Vina using config.yaml"
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = project_root_from_config(args.config)
    prot = cfg.get("protein", {})
    dock = cfg.get("docking", {})
    paths = cfg.get("paths", {})

    if shutil.which("obabel") is None:
        raise SystemExit("Open Babel 'obabel' not found in PATH.")

    lig_path = base / paths.get("ligands", "workspace/inputs/ligands.csv")
    clean_pdb = base / paths.get(
        "receptor_clean", "workspace/outputs/receptor_clean.pdb"
    )
    rec_pdbqt = base / paths.get("receptor_pdbqt", "workspace/outputs/receptor.pdbqt")
    out_csv = base / paths.get(
        "docking_results", "workspace/outputs/docking_results.csv"
    )

    if not rec_pdbqt.is_file() or not clean_pdb.is_file():
        raise SystemExit(
            f"Missing {rec_pdbqt} or {clean_pdb}. Run: python scripts/prepare_protein.py --config {args.config}"
        )
    if not lig_path.is_file():
        raise SystemExit(f"Ligand table not found: {lig_path}")

    center = resolve_center(cfg, clean_pdb)
    box = prot.get("pocket_size", [20, 20, 20])
    if isinstance(box, (int, float)):
        box = [float(box), float(box), float(box)]
    box = [float(x) for x in box]

    exh = int(dock.get("exhaustiveness", 8))
    n_poses = int(dock.get("num_modes", 9))
    seed = int(dock.get("seed", 42))
    n_cpu = vina_cpu_count(cfg)

    if dock.get("gpu"):
        print(
            "Note: AutoDock Vina uses CPU in this workflow; docking.gpu is ignored until a GPU backend is wired.",
            file=sys.stderr,
        )

    print(
        f"Box center (Å): {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}", flush=True
    )
    print(f"Box size (Å): {box}", flush=True)

    lig_df = pd.read_csv(lig_path)
    if "name" not in lig_df.columns or "smiles" not in lig_df.columns:
        raise ValueError("ligands.csv must have columns: name, smiles")

    dock_dir = base / paths.get("docked_dir", "workspace/outputs/docked")
    dock_dir.mkdir(parents=True, exist_ok=True)
    log_stub = log_path(cfg, base, "dock_ligands_v1")
    with open(log_stub, "w", encoding="utf-8") as lf:
        lf.write(f"seed={seed} center={center} box={box}\n")

    v = Vina(sf_name="vina", cpu=n_cpu, seed=seed)
    v.set_receptor(str(rec_pdbqt))
    v.compute_vina_maps(center=list(center), box_size=box)

    rows = []
    for _, row in lig_df.iterrows():
        name = str(row["name"]).strip()
        smi = str(row["smiles"]).strip()
        tag = safe_name(name)
        lig_pdbqt = dock_dir / f"{tag}_ligand.pdbqt"
        out_pose = dock_dir / f"{tag}_best.pdbqt"
        try:
            smiles_to_pdbqt(smi, lig_pdbqt)
        except subprocess.CalledProcessError as exc:
            rows.append(
                {
                    "name": name,
                    "smiles": smi,
                    "affinity_kcal_mol": np.nan,
                    "pose_pdbqt": "",
                    "error": (exc.stderr or str(exc))[:500],
                }
            )
            continue
        dock_out = run_docking(v, str(lig_pdbqt), exh, n_poses, out_pose)
        rows.append(
            {
                "name": name,
                "smiles": smi,
                "affinity_kcal_mol": dock_out["affinity"],
                "pose_pdbqt": dock_out["pose_path"],
                "error": "",
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        "affinity_kcal_mol", ascending=True, na_position="last"
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df[["name", "affinity_kcal_mol", "error"]].to_string(index=False))
    print(f"\nSaved {out_csv}")
    print(f"Log: {log_stub}")


if __name__ == "__main__":
    main()
