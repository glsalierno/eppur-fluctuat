#!/usr/bin/env python3
"""Strip waters/hetero, write clean PDB and rigid receptor PDBQT for AutoDock Vina."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from docking_common import load_config, project_root_from_config

try:
    from Bio.PDB import PDBIO, PDBParser, Select
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install biopython: pip install biopython") from exc


class _NotWaterOrHet(Select):
    """Keep standard protein atoms only (drop waters and hetero/ligands)."""

    def accept_atom(self, atom):
        res = atom.get_parent()
        het = res.get_id()[0]
        if het != " ":
            return False
        if res.get_resname() in ("HOH", "WAT"):
            return False
        return True


def obabel_pdb_to_pdbqt(
    pdb_path: Path, pdbqt_path: Path, rigid_receptor: bool = True
) -> None:
    """
    Convert a PDB file to AutoDock PDBQT using Open Babel.

    Parameters
    ----------
    pdb_path : pathlib.Path
        Input PDB path.
    pdbqt_path : pathlib.Path
        Output PDBQT path.
    rigid_receptor : bool, optional
        If True, pass ``-xr`` so the receptor is treated as rigid in PDBQT.

    Raises
    ------
    subprocess.CalledProcessError
        If ``obabel`` exits non-zero.
    FileNotFoundError
        If ``obabel`` is not found (typically raised as CalledProcessError / shell error depending on OS).
    """
    cmd = ["obabel", str(pdb_path), "-O", str(pdbqt_path)]
    if rigid_receptor:
        cmd.extend(["-xr"])
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def write_clean_protein_pdb(raw_pdb: Path, out_clean: Path) -> None:
    """
    Parse a raw PDB and write a cleaned protein-only PDB (no waters / hetero).

    Parameters
    ----------
    raw_pdb : pathlib.Path
        Path to the input PDB file.
    out_clean : pathlib.Path
        Path for the cleaned PDB output.

    Raises
    ------
    FileNotFoundError
        If ``raw_pdb`` does not exist.
    ValueError
        If BioPython fails to parse the structure (propagated from parser).
    """
    out_clean.parent.mkdir(parents=True, exist_ok=True)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", raw_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_clean), select=_NotWaterOrHet())


def prepare_receptor_for_vina(raw_pdb: Path, out_clean: Path, out_pdbqt: Path) -> None:
    """
    Produce ``receptor_clean.pdb`` and rigid ``receptor.pdbqt`` from a raw PDB.

    Parameters
    ----------
    raw_pdb : pathlib.Path
        Input receptor PDB.
    out_clean : pathlib.Path
        Output cleaned PDB path.
    out_pdbqt : pathlib.Path
        Output receptor PDBQT path.

    Raises
    ------
    FileNotFoundError
        If ``raw_pdb`` is missing or ``obabel`` is not installed (depending on environment).
    subprocess.CalledProcessError
        If Open Babel conversion fails.
    """
    write_clean_protein_pdb(raw_pdb, out_clean)
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    obabel_pdb_to_pdbqt(out_clean, out_pdbqt, rigid_receptor=True)


def main() -> None:
    """
    CLI entry: read ``config.yaml``, clean receptor, write PDBQT for Vina.

    Raises
    ------
    SystemExit
        On missing input files, missing ``obabel``, or other fatal setup errors.
    """
    ap = argparse.ArgumentParser(
        description="Prepare receptor PDB and PDBQT from config.yaml"
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
    paths = cfg.get("paths", {})

    raw = base / prot.get("pdb_file", "workspace/inputs/protein.pdb")
    if not raw.is_file():
        raise SystemExit(f"Receptor PDB not found: {raw}")

    out_clean = base / paths.get(
        "receptor_clean", "workspace/outputs/receptor_clean.pdb"
    )
    out_pdbqt = base / paths.get("receptor_pdbqt", "workspace/outputs/receptor.pdbqt")

    if shutil.which("obabel") is None:
        raise SystemExit(
            "Open Babel 'obabel' not found in PATH (conda install -c conda-forge openbabel)."
        )

    prepare_receptor_for_vina(raw, out_clean, out_pdbqt)
    print(f"Wrote {out_clean}")
    print(f"Wrote {out_pdbqt}")


if __name__ == "__main__":
    main()
