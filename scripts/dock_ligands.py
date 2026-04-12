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
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    from Bio.PDB import PDBParser
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install biopython: pip install biopython") from exc

try:
    from vina import Vina
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install vina: pip install vina") from exc


def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(name: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")[:80]


def smiles_to_pdbqt(smiles: str, out_pdbqt: Path) -> None:
    subprocess.run(
        ["obabel", f"-:{smiles}", "-O", str(out_pdbqt), "--gen3d", "-p", "7.4"],
        check=True,
        capture_output=True,
        text=True,
    )


def ca_centroid_chain(pdb_path: Path, chain_id: str) -> Optional[np.ndarray]:
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("p", pdb_path)
    coords: List[np.ndarray] = []
    for model in s:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] != " ":
                    continue
                if res.get_resname() in ("HOH", "WAT"):
                    continue
                if "CA" not in res:
                    continue
                a = res["CA"]
                if a.element and str(a.element).upper() != "H":
                    coords.append(a.coord)
    if not coords:
        return None
    return np.mean(np.array(coords), axis=0)


def reference_ligand_centroid(pdb_path: Path, chain_id: str, resname: str) -> Optional[np.ndarray]:
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("p", pdb_path)
    coords: List[np.ndarray] = []
    for model in s:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if res.get_resname().strip() != resname.strip():
                    continue
                for atom in res:
                    if atom.element and str(atom.element).upper() != "H":
                        coords.append(atom.coord)
    if not coords:
        return None
    return np.mean(np.array(coords), axis=0)


def resolve_center(cfg: dict, clean_pdb: Path) -> Tuple[float, float, float]:
    prot = cfg.get("protein", {})
    pc = prot.get("pocket_center", "auto")
    chain = str(prot.get("chain", "A"))
    ref = prot.get("reference_ligand_resname")
    if isinstance(ref, str) and ref.strip():
        cen = reference_ligand_centroid(clean_pdb, chain, ref)
        if cen is None:
            print(f"WARNING: residue '{ref}' not found; falling back to Cα centroid.", file=sys.stderr)
        else:
            return float(cen[0]), float(cen[1]), float(cen[2])
    if isinstance(pc, (list, tuple)) and len(pc) == 3:
        return float(pc[0]), float(pc[1]), float(pc[2])
    if pc == "auto" or pc is None:
        cen = ca_centroid_chain(clean_pdb, chain)
        if cen is None:
            raise RuntimeError(f"Could not compute automatic pocket center for chain {chain}.")
        return float(cen[0]), float(cen[1]), float(cen[2])
    raise ValueError("protein.pocket_center must be 'auto', [x,y,z], or set reference_ligand_resname")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dock ligands with AutoDock Vina using config.yaml")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = args.config.resolve().parent
    prot = cfg.get("protein", {})
    dock = cfg.get("docking", {})
    paths = cfg.get("paths", {})

    if shutil.which("obabel") is None:
        raise SystemExit("Open Babel 'obabel' not found in PATH.")

    lig_path = base / paths.get("ligands", "data/ligands.csv")
    clean_pdb = base / paths.get("receptor_clean", "outputs/receptor_clean.pdb")
    rec_pdbqt = base / paths.get("receptor_pdbqt", "outputs/receptor.pdbqt")
    out_csv = base / paths.get("docking_results", "outputs/docking_results.csv")

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
    cpu = int(dock.get("cpu", 0) or 0)
    n_cpu = max(1, (os.cpu_count() or 4) if cpu <= 0 else cpu)

    if dock.get("gpu"):
        print(
            "Note: AutoDock Vina uses CPU in this workflow; docking.gpu is ignored until a GPU backend is wired.",
            file=sys.stderr,
        )

    print(f"Box center (Å): {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}", flush=True)
    print(f"Box size (Å): {box}", flush=True)

    lig_df = pd.read_csv(lig_path)
    if "name" not in lig_df.columns or "smiles" not in lig_df.columns:
        raise ValueError("ligands.csv must have columns: name, smiles")

    dock_dir = base / "outputs" / "docked"
    dock_dir.mkdir(parents=True, exist_ok=True)

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
        v.set_ligand_from_file(str(lig_pdbqt))
        v.dock(exhaustiveness=exh, n_poses=n_poses)
        energies = v.energies(n_poses=n_poses)
        best = float(np.asarray(energies[0]).flatten()[0]) if len(energies) else float("nan")
        v.write_pose(str(out_pose), overwrite=True)
        rows.append(
            {
                "name": name,
                "smiles": smi,
                "affinity_kcal_mol": best,
                "pose_pdbqt": str(out_pose.resolve()),
                "error": "",
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("affinity_kcal_mol", ascending=True, na_position="last").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df[["name", "affinity_kcal_mol", "error"]].to_string(index=False))
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
