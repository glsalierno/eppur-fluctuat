"""
AChE (4EY7) Vina screening — self-contained replacement for legacy screen_alkyl_phosphates.

Builds receptor PDBQT in the work directory, docks ligands.csv, returns a ranked DataFrame.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from Bio.PDB import PDBIO, PDBParser, Select
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install biopython: pip install biopython") from exc

try:
    from vina import Vina
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install vina: pip install vina") from exc

DEFAULT_PDB = "4EY7"
DEFAULT_BOX = 22.0


class _NotWaterOrHet(Select):
    def accept_atom(self, atom):
        res = atom.get_parent()
        het = res.get_id()[0]
        if het != " ":
            return False
        if res.get_resname() in ("HOH", "WAT"):
            return False
        return True


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")[:80]


def smiles_to_pdbqt(smiles: str, out_pdbqt: Path) -> None:
    subprocess.run(
        ["obabel", f"-:{smiles}", "-O", str(out_pdbqt), "--gen3d", "-p", "7.4"],
        check=True,
        capture_output=True,
        text=True,
    )


def obabel_pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> None:
    subprocess.run(
        ["obabel", str(pdb_path), "-O", str(pdbqt_path), "-xr"],
        check=True,
        capture_output=True,
        text=True,
    )


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


def ser203_ca(pdb_path: Path, chain_id: str = "A") -> Optional[np.ndarray]:
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("p", pdb_path)
    for model in s:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] != " ":
                    continue
                if res.get_resname().strip() != "SER" or res.id[1] != 203:
                    continue
                if "CA" in res:
                    return np.asarray(res["CA"].coord, dtype=float)
    return None


def ensure_receptor(work: Path, pdb_id: str) -> Tuple[Path, Path]:
    """Return (clean_pdb, receptor_pdbqt)."""
    clean = work / f"{pdb_id}_receptor_clean.pdb"
    pdbqt = work / "receptor.pdbqt"
    raw = work / f"{pdb_id}.pdb"
    if not clean.is_file():
        if not raw.is_file():
            raise FileNotFoundError(f"Need {raw} or existing {clean}")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("rec", raw)
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(clean), select=_NotWaterOrHet())
    if not pdbqt.is_file():
        obabel_pdb_to_pdbqt(clean, pdbqt)
    return clean, pdbqt


def run_screening(
    ligands_csv: Path,
    work_dir: Path,
    pdb_id: str,
    box_size: float,
    exhaustiveness: int,
    n_poses: int,
    vina_cpu: int,
    center_mode: str,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Path, Tuple[float, float, float]]:
    """
    Dock all ligands; return (dataframe, receptor_pdbqt_path, center_tuple).
    """
    if shutil.which("obabel") is None:
        raise SystemExit("obabel not found in PATH")

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    clean_pdb, rec_pdbqt = ensure_receptor(work_dir, pdb_id)
    raw_pdb = work_dir / f"{pdb_id}.pdb"

    if center_mode == "e20":
        cen = reference_ligand_centroid(raw_pdb, "A", "E20")
        if cen is None:
            raise RuntimeError("E20 not found in raw PDB (donepezil is HETATM, stripped from clean receptor).")
    elif center_mode == "ser203":
        cen = ser203_ca(clean_pdb, "A")
        if cen is None:
            raise RuntimeError("Ser203 CA not found.")
    else:
        raise ValueError(center_mode)

    center = (float(cen[0]), float(cen[1]), float(cen[2]))
    box = [float(box_size), float(box_size), float(box_size)]

    lig_df = pd.read_csv(ligands_csv)
    if "name" not in lig_df.columns or "smiles" not in lig_df.columns:
        raise ValueError("ligands.csv needs name, smiles")

    dock_dir = work_dir / "docked"
    dock_dir.mkdir(parents=True, exist_ok=True)

    v = Vina(sf_name="vina", cpu=max(1, vina_cpu), seed=int(seed))
    v.set_receptor(str(rec_pdbqt))
    v.compute_vina_maps(center=list(center), box_size=box)

    rows = []
    for _, row in lig_df.iterrows():
        name = str(row["name"]).strip()
        smi = str(row["smiles"]).strip()
        tag = safe_name(name)
        lig_pdbqt = dock_dir / f"{tag}_ligand.pdbqt"
        out_pose = dock_dir / f"{tag}_best.pdbqt"
        rec: dict = {"name": name, "smiles": smi, "affinity_kcal_mol": np.nan, "pose_pdbqt": "", "error": ""}
        if "cas" in lig_df.columns:
            rec["cas"] = row.get("cas", "")
        try:
            smiles_to_pdbqt(smi, lig_pdbqt)
        except subprocess.CalledProcessError as exc:
            rec["error"] = (exc.stderr or str(exc))[:500]
            rows.append(rec)
            continue
        v.set_ligand_from_file(str(lig_pdbqt))
        v.dock(exhaustiveness=int(exhaustiveness), n_poses=int(n_poses))
        energies = v.energies(n_poses=int(n_poses))
        best = float(np.asarray(energies[0]).flatten()[0]) if len(energies) else float("nan")
        v.write_pose(str(out_pose), overwrite=True)
        rec["affinity_kcal_mol"] = best
        rec["pose_pdbqt"] = str(out_pose.resolve())
        rows.append(rec)

    df = pd.DataFrame(rows)
    df = df.sort_values("affinity_kcal_mol", ascending=True, na_position="last").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df, rec_pdbqt, center
