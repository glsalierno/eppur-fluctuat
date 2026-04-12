#!/usr/bin/env python3
"""Strip waters/hetero, write clean PDB and rigid receptor PDBQT for AutoDock Vina."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

try:
    from Bio.PDB import PDBIO, PDBParser, Select
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install biopython: pip install biopython") from exc


def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def obabel_pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path, rigid_receptor: bool = True) -> None:
    cmd = ["obabel", str(pdb_path), "-O", str(pdbqt_path)]
    if rigid_receptor:
        cmd.extend(["-xr"])
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare receptor PDB and PDBQT from config.yaml")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = args.config.resolve().parent
    prot = cfg.get("protein", {})
    paths = cfg.get("paths", {})

    raw = base / prot.get("pdb_file", "data/protein.pdb")
    if not raw.is_file():
        raise SystemExit(f"Receptor PDB not found: {raw}")

    out_clean = base / paths.get("receptor_clean", "outputs/receptor_clean.pdb")
    out_pdbqt = base / paths.get("receptor_pdbqt", "outputs/receptor.pdbqt")
    out_clean.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("obabel") is None:
        raise SystemExit("Open Babel 'obabel' not found in PATH (conda install -c conda-forge openbabel).")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", raw)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_clean), select=_NotWaterOrHet())
    obabel_pdb_to_pdbqt(out_clean, out_pdbqt, rigid_receptor=True)
    print(f"Wrote {out_clean}")
    print(f"Wrote {out_pdbqt}")


if __name__ == "__main__":
    main()
