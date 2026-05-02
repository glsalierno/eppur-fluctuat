"""Shared helpers for Vina docking scripts (paths, SMILES→PDBQT, pocket center)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

try:
    from Bio.PDB import PDBParser
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install biopython: pip install biopython") from exc


def load_config(cfg_path: Path) -> dict:
    """
    Load repository YAML configuration.

    Parameters
    ----------
    cfg_path : pathlib.Path
        Path to ``config.yaml``.

    Returns
    -------
    dict
        Parsed configuration tree.

    Raises
    ------
    FileNotFoundError
        If ``cfg_path`` is missing.
    yaml.YAMLError
        On invalid YAML.
    """
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root_from_config(cfg_path: Path) -> Path:
    """
    Directory containing ``config.yaml`` (project root).

    Parameters
    ----------
    cfg_path : pathlib.Path
        Path to the configuration file.

    Returns
    -------
    pathlib.Path
        Resolved parent directory of ``cfg_path``.
    """
    return cfg_path.resolve().parent


def safe_name(name: str) -> str:
    """
    Sanitize a ligand name for filesystem-safe stems.

    Parameters
    ----------
    name : str
        Arbitrary compound label.

    Returns
    -------
    str
        ASCII-safe string truncated to 80 characters.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")[:80]


def smiles_to_pdbqt(smiles: str, out_pdbqt: Path) -> None:
    """
    Convert SMILES to 3D protonated PDBQT via Open Babel CLI.

    Parameters
    ----------
    smiles : str
        Input SMILES.
    out_pdbqt : pathlib.Path
        Output PDBQT path.

    Raises
    ------
    subprocess.CalledProcessError
        If ``obabel`` fails.
    """
    subprocess.run(
        ["obabel", f"-:{smiles}", "-O", str(out_pdbqt), "--gen3d", "-p", "7.4"],
        check=True,
        capture_output=True,
        text=True,
    )


def ca_centroid_chain(pdb_path: Path, chain_id: str) -> Optional[np.ndarray]:
    """
    Centroid of Cα heavy atoms for standard amino-acid residues on a chain.

    Parameters
    ----------
    pdb_path : pathlib.Path
        PDB structure file.
    chain_id : str
        Chain identifier (e.g. ``A``).

    Returns
    -------
    numpy.ndarray or None
        Shape (3,) centroid in Å, or None if no Cα atoms found.
    """
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


def reference_ligand_centroid(
    pdb_path: Path, chain_id: str, resname: str
) -> Optional[np.ndarray]:
    """
    Centroid of heavy atoms for a hetero residue name (co-crystallized ligand).

    Parameters
    ----------
    pdb_path : pathlib.Path
        PDB structure file.
    chain_id : str
        Chain identifier.
    resname : str
        Residue name to match (hetero ligand).

    Returns
    -------
    numpy.ndarray or None
        Shape (3,) centroid in Å, or None if residue not found.
    """
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
    """
    Resolve docking box centre from ``protein`` configuration.

    Parameters
    ----------
    cfg : dict
        Full config (uses ``protein`` section).
    clean_pdb : pathlib.Path
        Clean receptor PDB used for atom coordinates.

    Returns
    -------
    tuple of float
        ``(cx, cy, cz)`` in Å.

    Raises
    ------
    RuntimeError
        If ``pocket_center`` is ``auto`` but no Cα atoms are available.
    ValueError
        If ``pocket_center`` is misconfigured.
    """
    prot = cfg.get("protein", {})
    pc = prot.get("pocket_center", "auto")
    chain = str(prot.get("chain", "A"))
    ref = prot.get("reference_ligand_resname")
    if isinstance(ref, str) and ref.strip():
        cen = reference_ligand_centroid(clean_pdb, chain, ref)
        if cen is None:
            print(
                f"WARNING: residue '{ref}' not found; falling back to Cα centroid.",
                file=sys.stderr,
            )
        else:
            return float(cen[0]), float(cen[1]), float(cen[2])
    if isinstance(pc, (list, tuple)) and len(pc) == 3:
        return float(pc[0]), float(pc[1]), float(pc[2])
    if pc == "auto" or pc is None:
        cen = ca_centroid_chain(clean_pdb, chain)
        if cen is None:
            raise RuntimeError(
                f"Could not compute automatic pocket center for chain {chain}."
            )
        return float(cen[0]), float(cen[1]), float(cen[2])
    raise ValueError(
        "protein.pocket_center must be 'auto', [x,y,z], or set reference_ligand_resname"
    )


def vina_cpu_count(cfg: dict) -> int:
    """
    Effective CPU thread count for Vina from ``docking.cpu`` (0 = auto).

    Parameters
    ----------
    cfg : dict
        Configuration containing ``docking.cpu``.

    Returns
    -------
    int
        At least 1 worker thread.
    """
    dock = cfg.get("docking", {})
    cpu = int(dock.get("cpu", 0) or 0)
    return max(1, (os.cpu_count() or 4) if cpu <= 0 else cpu)


def log_path(cfg: dict, base: Path, stem: str) -> Path:
    """
    Path to a log file under configured ``paths.logs_dir``.

    Parameters
    ----------
    cfg : dict
        Full configuration.
    base : pathlib.Path
        Project root directory.
    stem : str
        Filename stem without extension.

    Returns
    -------
    pathlib.Path
        ``{logs_dir}/{stem}.log`` (directories created as needed).
    """
    paths = cfg.get("paths", {})
    log_dir = base / paths.get("logs_dir", "var/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{stem}.log"
