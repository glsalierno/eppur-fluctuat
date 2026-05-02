#!/usr/bin/env python3
"""
Compare AChE Vina docking baseline vs Vina + NVIDIA ALCHEMI FIRE2 refinement.

Uncertainty: multiple Vina seeds → mean ± std of docking score and of post-refinement score.

Writes:
  workspace/outputs/ache_compare/summary.csv
  workspace/outputs/ache_compare/summary.md
  workspace/outputs/ache_compare/per_seed_long.csv
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

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_SCRIPTS = _ROOT / "scripts"
for p in (_ROOT, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import MDAnalysis as mda
from vina import Vina

from ache_vina_screening import (
    DEFAULT_BOX,
    DEFAULT_PDB,
    ensure_receptor,
    reference_ligand_centroid,
    safe_name,
    ser203_ca,
    smiles_to_pdbqt,
)
from alchemi_refine import check_alchemi_available, refine_with_alchemi_fire2
from interaction_analysis import merge_receptor_ligand, pdbqt_to_pdb_with_h, protonate_receptor_pdb
from ligand_rescore_prep import ligand_pdb_to_vina_pdbqt


def _center_tuple(raw_pdb: Path, clean_pdb: Path, mode: str) -> Tuple[float, float, float]:
    # Co-crystallized ligand E20 is HETATM; clean receptor PDB omits it — use raw PDB for E20 centroid.
    if mode == "e20":
        cen = reference_ligand_centroid(raw_pdb, "A", "E20")
    else:
        cen = ser203_ca(clean_pdb, "A")
    if cen is None:
        raise RuntimeError(f"Could not get pocket center ({mode}).")
    return float(cen[0]), float(cen[1]), float(cen[2])


def _extract_ligand_pdb(merged_pdb: Path, out_pdb: Path) -> None:
    u = mda.Universe(str(merged_pdb))
    lig = u.select_atoms("resname LIG")
    if lig.n_atoms == 0:
        raise RuntimeError(f"No LIG in {merged_pdb}")
    lig.write(str(out_pdb))


def _vina_score_total(v: Vina) -> float:
    e = np.asarray(v.score()).flatten()
    return float(e[0]) if e.size else float("nan")


def _ensure_pdb_id_in_work(work_dir: Path, pdb_id: str) -> None:
    dest = work_dir / f"{pdb_id}.pdb"
    if dest.is_file():
        return
    src = _HERE / f"{pdb_id}.pdb"
    if not src.is_file():
        raise FileNotFoundError(
            f"Missing receptor PDB: copy {pdb_id}.pdb into {_HERE} or into {work_dir}."
        )
    shutil.copy2(src, dest)


def run_batch(
    ligands_csv: Path,
    work_dir: Path,
    pdb_id: str,
    box_size: float,
    exhaustiveness: int,
    n_poses: int,
    vina_cpu: int,
    center_mode: str,
    seeds: List[int],
    *,
    use_alchemi: bool,
    alchemi_steps: int,
    alchemi_dt: float,
    alchemi_device: str,
    cache_dir: Path,
    tmp_dir: Path,
    rescore_fix_multimolecule_pdbqt: bool = True,
    rescore_rdkit_largest_fragment: bool = False,
    rescore_rigid_pdbqt: bool = False,
) -> pd.DataFrame:
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _ensure_pdb_id_in_work(work_dir, pdb_id)

    clean_pdb, rec_pdbqt = ensure_receptor(work_dir, pdb_id)
    raw_pdb = work_dir / f"{pdb_id}.pdb"
    center = _center_tuple(raw_pdb, clean_pdb, center_mode)
    box = [float(box_size)] * 3
    lig_df = pd.read_csv(ligands_csv)
    dock_dir = work_dir / "docked"
    dock_dir.mkdir(parents=True, exist_ok=True)

    rec_h = cache_dir / f"{pdb_id}_rec_h.pdb"
    protonate_receptor_pdb(clean_pdb, rec_h)

    rows: List[Dict[str, Any]] = []
    if use_alchemi and not check_alchemi_available():
        raise SystemExit(
            "ALCHEMI path requested but nvalchemi-toolkit / nvalchemi-toolkit-ops not installed. "
            "pip install -r requirements-alchemi.txt"
        )

    for seed in seeds:
        v = Vina(sf_name="vina", cpu=max(1, vina_cpu), seed=int(seed))
        v.set_receptor(str(rec_pdbqt))
        v.compute_vina_maps(center=list(center), box_size=box)

        for _, row in lig_df.iterrows():
            name = str(row["name"]).strip()
            smi = str(row["smiles"]).strip()
            tag = safe_name(name)
            lig_pdbqt = dock_dir / f"{tag}_ligand.pdbqt"
            out_pose = dock_dir / f"{tag}_best_s{seed}.pdbqt"
            rec: Dict[str, Any] = {
                "name": name,
                "smiles": smi,
                "seed": seed,
                "strategy": "vina_alchemi" if use_alchemi else "vina_baseline",
                "affinity_dock_kcal_mol": np.nan,
                "affinity_rescore_kcal_mol": np.nan,
                "affinity_after_refine_kcal_mol": np.nan,
                "error": "",
            }
            try:
                smiles_to_pdbqt(smi, lig_pdbqt)
            except subprocess.CalledProcessError as exc:
                rec["error"] = (exc.stderr or str(exc))[:400]
                rows.append(rec)
                continue

            v.set_ligand_from_file(str(lig_pdbqt))
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            en = v.energies(n_poses=n_poses)
            rec["affinity_dock_kcal_mol"] = (
                float(np.asarray(en[0]).flatten()[0]) if len(en) else float("nan")
            )
            v.write_pose(str(out_pose), overwrite=True)

            v.set_ligand_from_file(str(out_pose))
            rec["affinity_rescore_kcal_mol"] = _vina_score_total(v)

            if use_alchemi:
                lig_h = tmp_dir / f"{tag}_s{seed}_lig_h.pdb"
                merged = tmp_dir / f"{tag}_s{seed}_complex.pdb"
                merged_ref = tmp_dir / f"{tag}_s{seed}_complex_ref.pdb"
                lig_ref = tmp_dir / f"{tag}_s{seed}_lig_ref.pdb"
                out_ref = dock_dir / f"{tag}_best_alchemi_s{seed}.pdbqt"
                try:
                    pdbqt_to_pdb_with_h(out_pose, lig_h)
                    merge_receptor_ligand(rec_h, lig_h, merged)
                    refine_with_alchemi_fire2(
                        merged,
                        merged_ref,
                        n_steps=alchemi_steps,
                        dt=alchemi_dt,
                        device=alchemi_device,
                        use_demo_potential=True,
                    )
                    _extract_ligand_pdb(merged_ref, lig_ref)
                    ligand_pdb_to_vina_pdbqt(
                        lig_ref,
                        out_ref,
                        use_rdkit_largest_fragment=rescore_rdkit_largest_fragment,
                        fix_multimolecule_pdbqt=rescore_fix_multimolecule_pdbqt,
                        rigid_pdbqt=rescore_rigid_pdbqt,
                    )
                    v.set_ligand_from_file(str(out_ref))
                    rec["affinity_after_refine_kcal_mol"] = _vina_score_total(v)
                except Exception as exc:
                    rec["error"] = f"alchemi: {exc}"
                    rec["affinity_after_refine_kcal_mol"] = float("nan")
            else:
                rec["affinity_after_refine_kcal_mol"] = rec["affinity_rescore_kcal_mol"]

            rows.append(rec)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare AChE Vina vs Vina+ALCHEMI")
    ap.add_argument("--ligands", type=Path, default=_HERE / "ligands.csv")
    ap.add_argument("--pdb-id", default=DEFAULT_PDB)
    ap.add_argument("--box-size", type=float, default=DEFAULT_BOX)
    ap.add_argument("--center", choices=["e20", "ser203"], default="e20")
    ap.add_argument("--exhaustiveness", type=int, default=4)
    ap.add_argument("--n-poses", type=int, default=3)
    ap.add_argument("--vina-cpu", type=int, default=max(1, __import__("os").cpu_count() or 4))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--max-ligands", type=int, default=0, help="0 = all ligands")
    ap.add_argument("--alchemi-steps", type=int, default=15)
    ap.add_argument("--alchemi-dt", type=float, default=0.02)
    ap.add_argument("--alchemi-device", default="auto")
    ap.add_argument(
        "--no-rescore-pdbqt-fix",
        action="store_true",
        help="Disable truncation of multi-molecule PDBQT (not recommended for P/S ligands)",
    )
    ap.add_argument(
        "--rescore-rdkit-largest-fragment",
        action="store_true",
        help="Before PDBQT: RDKit largest fragment + strip H (stronger cleanup for thiophosphates)",
    )
    ap.add_argument(
        "--rescore-rigid-pdbqt",
        action="store_true",
        help="Pass obabel -xr (rigid ligand PDBQT) after cleanup",
    )
    args = ap.parse_args()

    out_root = _ROOT / "workspace" / "outputs" / "ache_compare"
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = _ROOT / "var" / "cache" / "ache_compare"
    tmp_dir = _ROOT / "var" / "tmp" / "ache_compare"

    lig = pd.read_csv(args.ligands)
    if args.max_ligands and args.max_ligands > 0:
        lig = lig.head(args.max_ligands).copy()
    tmp_csv = out_root / "_ligands_subset.csv"
    lig.to_csv(tmp_csv, index=False)

    seeds = [args.base_seed + i for i in range(args.seeds)]

    fix_pdbqt = not args.no_rescore_pdbqt_fix

    df_base = run_batch(
        tmp_csv,
        out_root / "baseline_work",
        args.pdb_id,
        args.box_size,
        args.exhaustiveness,
        args.n_poses,
        args.vina_cpu,
        args.center,
        seeds,
        use_alchemi=False,
        alchemi_steps=args.alchemi_steps,
        alchemi_dt=args.alchemi_dt,
        alchemi_device=args.alchemi_device,
        cache_dir=cache_dir,
        tmp_dir=tmp_dir,
        rescore_fix_multimolecule_pdbqt=fix_pdbqt,
        rescore_rdkit_largest_fragment=args.rescore_rdkit_largest_fragment,
        rescore_rigid_pdbqt=args.rescore_rigid_pdbqt,
    )

    df_alc = run_batch(
        tmp_csv,
        out_root / "alchemi_work",
        args.pdb_id,
        args.box_size,
        args.exhaustiveness,
        args.n_poses,
        args.vina_cpu,
        args.center,
        seeds,
        use_alchemi=True,
        alchemi_steps=args.alchemi_steps,
        alchemi_dt=args.alchemi_dt,
        alchemi_device=args.alchemi_device,
        cache_dir=cache_dir,
        tmp_dir=tmp_dir,
        rescore_fix_multimolecule_pdbqt=fix_pdbqt,
        rescore_rdkit_largest_fragment=args.rescore_rdkit_largest_fragment,
        rescore_rigid_pdbqt=args.rescore_rigid_pdbqt,
    )

    long_df = pd.concat([df_base, df_alc], ignore_index=True)
    long_path = out_root / "per_seed_long.csv"
    long_df.to_csv(long_path, index=False)

    summary = lig[["name", "smiles"]].copy() if "smiles" in lig.columns else lig[["name"]].copy()
    for label, df_ in [("baseline", df_base), ("alchemi", df_alc)]:
        g = df_.groupby("name").agg(
            dock_mean=("affinity_dock_kcal_mol", "mean"),
            dock_std=("affinity_dock_kcal_mol", "std"),
            refined_mean=("affinity_after_refine_kcal_mol", "mean"),
            refined_std=("affinity_after_refine_kcal_mol", "std"),
        )
        g.columns = [f"{label}_{c}" for c in g.columns]
        summary = summary.merge(g.reset_index(), on="name", how="left")

    summary_path = out_root / "summary.csv"
    summary.to_csv(summary_path, index=False)

    md_lines = [
        "# AChE docking: Vina baseline vs Vina + ALCHEMI (FIRE2 DemoModel)",
        "",
        f"- Ligands: {len(lig)} | Seeds: {seeds} | exhaustiveness={args.exhaustiveness} | poses={args.n_poses}",
        f"- ALCHEMI: FIRE2 steps={args.alchemi_steps} dt={args.alchemi_dt} (Demo potential — integration test only)",
        "",
        "## Summary (CSV)",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
        f"Full per-seed table: `{long_path.relative_to(_ROOT)}`",
    ]
    (_ROOT / "workspace" / "outputs" / "ache_compare" / "summary.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )

    print(summary.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {long_path}")


if __name__ == "__main__":
    main()
