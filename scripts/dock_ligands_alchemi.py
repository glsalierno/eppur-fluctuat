#!/usr/bin/env python3
"""
Docking v2: AutoDock Vina + optional NVIDIA ALCHEMI Toolkit (FIRE2) ligand relaxation
in a merged receptor–ligand complex, then Vina rescoring of the refined pose.

Baseline Vina-only path remains ``scripts/dock_ligands.py``. This script adds the
ALCHEMI refinement stage and optional multi-seed uncertainty (mean/std of scores).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
for p in (_ROOT, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from alchemi_refine import check_alchemi_available, refine_with_alchemi_fire2
from docking_common import (
    load_config,
    log_path,
    project_root_from_config,
    resolve_center,
    safe_name,
    smiles_to_pdbqt,
    vina_cpu_count,
)
from ligand_rescore_prep import ligand_pdb_to_vina_pdbqt

from interaction_analysis import (
    merge_receptor_ligand,
    pdbqt_to_pdb_with_h,
    protonate_receptor_pdb,
)

try:
    import MDAnalysis as mda
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install MDAnalysis: pip install MDAnalysis") from exc

try:
    from vina import Vina
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install vina: pip install vina") from exc


def _extract_ligand_pdb(merged_pdb: Path, out_pdb: Path) -> None:
    u = mda.Universe(str(merged_pdb))
    lig = u.select_atoms("resname LIG")
    if lig.n_atoms == 0:
        raise RuntimeError(f"No resname LIG in {merged_pdb}")
    lig.write(str(out_pdb))


def _vina_affinity_from_score(v: Vina) -> float:
    e = np.asarray(v.score()).flatten()
    return float(e[0]) if e.size else float("nan")


def run_one_seed(
    cfg: dict,
    base: Path,
    lig_df: pd.DataFrame,
    center: Tuple[float, float, float],
    box: List[float],
    dock_dir: Path,
    cache_dir: Path,
    tmp_dir: Path,
    seed: int,
    *,
    use_alchemi: bool,
) -> List[Dict[str, Any]]:
    """
    Dock each ligand with Vina for one random seed, optionally relax with ALCHEMI/FIRE2.

    Parameters
    ----------
    cfg : dict
        Full configuration tree.
    base : pathlib.Path
        Project root directory.
    lig_df : pandas.DataFrame
        Ligand table with ``name`` and ``smiles``.
    center : tuple of float
        Vina map centre (Å).
    box : list of float
        Map box size (Å).
    dock_dir : pathlib.Path
        Output directory for PDBQT poses.
    cache_dir : pathlib.Path
        Receptor protonation cache directory.
    tmp_dir : pathlib.Path
        Scratch directory for merged complexes.
    seed : int
        Vina RNG seed.
    use_alchemi : bool
        Whether to attempt FIRE2 refinement when dependencies exist.

    Returns
    -------
    list of dict
        One row dict per ligand for this seed (affinities, paths, errors).

    Raises
    ------
    SystemExit
        If receptor files are missing before docking starts.
    """
    dock = cfg.get("docking", {})
    paths = cfg.get("paths", {})
    alcfg = cfg.get("alchemi", {})
    exh = int(dock.get("exhaustiveness", 8))
    n_poses = int(dock.get("num_modes", 9))
    n_cpu = vina_cpu_count(cfg)

    clean_pdb = base / paths.get("receptor_clean", "workspace/outputs/receptor_clean.pdb")
    rec_pdbqt = base / paths.get("receptor_pdbqt", "workspace/outputs/receptor.pdbqt")
    if not rec_pdbqt.is_file() or not clean_pdb.is_file():
        raise SystemExit("Missing receptor; run prepare_protein.py first.")

    rec_h = cache_dir / "receptor_protonated_h.pdb"
    protonate_receptor_pdb(clean_pdb, rec_h)

    rows: List[Dict[str, Any]] = []
    use_alchemi = use_alchemi and bool(alcfg.get("enabled", True))
    if use_alchemi and not check_alchemi_available():
        print(
            "ALCHEMI requested but nvalchemi-toolkit/ops not importable; "
            "refine step skipped (install requirements-alchemi.txt).",
            file=sys.stderr,
        )
        use_alchemi = False

    for _, row in lig_df.iterrows():
        name = str(row["name"]).strip()
        smi = str(row["smiles"]).strip()
        tag = safe_name(name)
        lig_pdbqt = dock_dir / f"{tag}_ligand.pdbqt"
        out_pose = dock_dir / f"{tag}_best.pdbqt"
        out_pose_ref = dock_dir / f"{tag}_best_alchemi.pdbqt" if use_alchemi else out_pose

        entry: Dict[str, Any] = {
            "name": name,
            "smiles": smi,
            "seed": seed,
            "affinity_vina_dock_kcal_mol": np.nan,
            "affinity_vina_rescore_kcal_mol": np.nan,
            "affinity_after_alchemi_kcal_mol": np.nan,
            "pose_pdbqt": str(out_pose.resolve()),
            "pose_pdbqt_refined": str(out_pose_ref.resolve()) if use_alchemi else "",
            "alchemi_applied": use_alchemi,
            "alchemi_note": "",
            "error": "",
        }

        v = Vina(sf_name="vina", cpu=n_cpu, seed=seed)
        v.set_receptor(str(rec_pdbqt))
        v.compute_vina_maps(center=list(center), box_size=box)

        try:
            smiles_to_pdbqt(smi, lig_pdbqt)
        except subprocess.CalledProcessError as exc:
            entry["error"] = (exc.stderr or str(exc))[:500]
            rows.append(entry)
            continue

        v.set_ligand_from_file(str(lig_pdbqt))
        v.dock(exhaustiveness=exh, n_poses=n_poses)
        energies = v.energies(n_poses=n_poses)
        best_dock = float(np.asarray(energies[0]).flatten()[0]) if len(energies) else float("nan")
        entry["affinity_vina_dock_kcal_mol"] = best_dock
        v.write_pose(str(out_pose), overwrite=True)

        lig_h = tmp_dir / f"{tag}_lig_h.pdb"
        merged = tmp_dir / f"{tag}_complex.pdb"
        merged_ref = tmp_dir / f"{tag}_complex_ref.pdb"
        lig_ref_pdb = tmp_dir / f"{tag}_lig_ref.pdb"

        pdbqt_to_pdb_with_h(out_pose, lig_h)
        merge_receptor_ligand(rec_h, lig_h, merged)

        v.set_ligand_from_file(str(out_pose))
        entry["affinity_vina_rescore_kcal_mol"] = _vina_affinity_from_score(v)

        if use_alchemi:
            try:
                note, _dev = refine_with_alchemi_fire2(
                    merged,
                    merged_ref,
                    n_steps=int(alcfg.get("fire2_steps", 25)),
                    dt=float(alcfg.get("fire2_dt", 0.02)),
                    device=str(alcfg.get("device", "auto")),
                    use_demo_potential=bool(alcfg.get("use_demo_potential", True)),
                )
                entry["alchemi_note"] = note
                _extract_ligand_pdb(merged_ref, lig_ref_pdb)
                ligand_pdb_to_vina_pdbqt(
                    lig_ref_pdb,
                    out_pose_ref,
                    use_rdkit_largest_fragment=bool(alcfg.get("rescore_rdkit_largest_fragment", False)),
                    fix_multimolecule_pdbqt=bool(alcfg.get("rescore_fix_multimolecule_pdbqt", True)),
                    rigid_pdbqt=bool(alcfg.get("rescore_rigid_pdbqt", False)),
                )
                v.set_ligand_from_file(str(out_pose_ref))
                entry["affinity_after_alchemi_kcal_mol"] = _vina_affinity_from_score(v)
            except Exception as exc:
                entry["alchemi_note"] = f"ALCHEMI failed: {exc}"
                entry["affinity_after_alchemi_kcal_mol"] = float("nan")
        else:
            entry["affinity_after_alchemi_kcal_mol"] = entry["affinity_vina_rescore_kcal_mol"]

        rows.append(entry)

    return rows


def main() -> None:
    """
    CLI entry: multi-seed Vina docking with optional ALCHEMI relaxation and Vina rescoring.

    Raises
    ------
    SystemExit
        On missing tools, missing inputs, or configuration errors.
    ValueError
        If the ligand table is invalid.
    """
    ap = argparse.ArgumentParser(description="Vina + ALCHEMI FIRE2 refinement (docking v2)")
    ap.add_argument("--config", type=Path, default=_ROOT / "config.yaml")
    ap.add_argument("--no-alchemi", action="store_true", help="Vina + merge/rescore only (no FIRE2)")
    ap.add_argument("--seeds", type=int, default=None, help="Override uncertainty.n_seeds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = project_root_from_config(args.config)
    paths = cfg.get("paths", {})
    prot = cfg.get("protein", {})
    dock = cfg.get("docking", {})
    uncfg = cfg.get("uncertainty", {})

    if shutil.which("obabel") is None:
        raise SystemExit("Open Babel 'obabel' not found in PATH.")

    lig_path = base / paths.get("ligands", "workspace/inputs/ligands.csv")
    out_csv = base / paths.get("docking_results", "workspace/outputs/docking_results_alchemi.csv")
    dock_dir = base / paths.get("docked_dir", "workspace/outputs/docked_alchemi")
    cache_dir = base / paths.get("cache_dir", "var/cache")
    tmp_dir = base / paths.get("tmp_dir", "var/tmp")
    for d in (dock_dir, cache_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not lig_path.is_file():
        raise SystemExit(f"Ligand table not found: {lig_path}")

    clean_pdb = base / paths.get("receptor_clean", "workspace/outputs/receptor_clean.pdb")
    center = resolve_center(cfg, clean_pdb)
    box = prot.get("pocket_size", [20, 20, 20])
    if isinstance(box, (int, float)):
        box = [float(box), float(box), float(box)]
    box = [float(x) for x in box]

    if dock.get("gpu"):
        print(
            "Note: Vina uses CPU; docking.gpu is reserved for other engines.",
            file=sys.stderr,
        )

    n_seeds = int(args.seeds) if args.seeds is not None else int(uncfg.get("n_seeds", 1))
    base_seed = int(uncfg.get("base_seed", 42))
    seeds = [base_seed + i for i in range(n_seeds)]

    lig_df = pd.read_csv(lig_path)
    if "name" not in lig_df.columns or "smiles" not in lig_df.columns:
        raise ValueError("ligands table must have columns: name, smiles")

    logf = log_path(cfg, base, "dock_ligands_alchemi")
    with open(logf, "w", encoding="utf-8") as log:
        log.write(f"center={center} box={box} seeds={seeds} no_alchemi={args.no_alchemi}\n")

    all_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        all_rows.extend(
            run_one_seed(
                cfg,
                base,
                lig_df,
                center,
                box,
                dock_dir,
                cache_dir,
                tmp_dir,
                seed,
                use_alchemi=not args.no_alchemi,
            )
        )

    df = pd.DataFrame(all_rows)
    g = df.groupby("name", as_index=False).agg(
        smiles=("smiles", "first"),
        affinity_vina_dock_mean=("affinity_vina_dock_kcal_mol", "mean"),
        affinity_vina_dock_std=("affinity_vina_dock_kcal_mol", "std"),
        affinity_vina_rescore_mean=("affinity_vina_rescore_kcal_mol", "mean"),
        affinity_vina_rescore_std=("affinity_vina_rescore_kcal_mol", "std"),
        affinity_after_alchemi_mean=("affinity_after_alchemi_kcal_mol", "mean"),
        affinity_after_alchemi_std=("affinity_after_alchemi_kcal_mol", "std"),
        alchemi_applied=("alchemi_applied", "first"),
        pose_pdbqt=("pose_pdbqt", "first"),
        pose_pdbqt_refined=("pose_pdbqt_refined", "first"),
        alchemi_note=("alchemi_note", "first"),
    )
    g = g.sort_values("affinity_vina_dock_mean", ascending=True, na_position="last").reset_index(drop=True)
    g.insert(0, "rank", range(1, len(g) + 1))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out_csv, index=False)
    df.to_csv(out_csv.with_name(out_csv.stem + "_per_seed.csv"), index=False)

    print(
        g[
            [
                "name",
                "affinity_vina_dock_mean",
                "affinity_vina_dock_std",
                "affinity_after_alchemi_mean",
                "affinity_after_alchemi_std",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved summary: {out_csv}")
    print(f"Per-seed rows: {out_csv.with_name(out_csv.stem + '_per_seed.csv')}")
    print(f"Log: {logf}")


if __name__ == "__main__":
    main()
