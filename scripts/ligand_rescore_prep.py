"""
Prepare refined ligand coordinates for AutoDock Vina rescoring.

FIRE2 + the demo MLIP can distort organophosphates (P/S centers) so that ligand coordinates
no longer match covalent bond heuristics: a hydrogen can end up far from its heavy atom.
Open Babel then treats it as a **second molecule** and writes a second ``ROOT`` block
(sometimes with garbage coordinates). Vina fails with ``Unknown or inappropriate tag ... > ROOT``.

Note: ``obabel -xr`` (rigid PDBQT) does **not** merge disconnected fragments; use truncation
and/or RDKit largest-fragment cleanup first, then optionally ``-xr`` for a minimal torsion tree.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def truncate_pdbqt_to_first_ligand(pdbqt_path: Path) -> bool:
    """
    Keep only the first ROOT … TORSDOF block (first AutoDock ligand tree).

    Returns True if the file was modified or is already single-ligand; False if no ROOT found.
    """
    text = pdbqt_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    root_idx: int | None = None
    for i, line in enumerate(lines):
        if line.startswith("ROOT"):
            root_idx = i
            break
    if root_idx is None:
        return False
    tors_idx: int | None = None
    for i in range(root_idx, len(lines)):
        if lines[i].startswith("TORSDOF"):
            tors_idx = i
            break
    if tors_idx is None:
        return False
    new_content = "".join(lines[: tors_idx + 1])
    if new_content != text:
        pdbqt_path.write_text(new_content, encoding="utf-8")
    return True


def rdkit_largest_fragment_pdb_to_pdb(in_pdb: Path, out_pdb: Path) -> bool:
    """
    Split disconnected fragments; keep the largest by heavy-atom count; drop explicit H.

    Uses no full sanitization (organophosphates often fail RDKit sanitize). Returns False
    if RDKit cannot parse the PDB.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return False

    mol = Chem.MolFromPDBFile(str(in_pdb), sanitize=False, removeHs=False)
    if mol is None:
        return False
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return False
    best = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(best)
    except Exception:
        pass
    best_noh = Chem.RemoveHs(best, implicitOnly=False)
    Chem.MolToPDBFile(best_noh, str(out_pdb))
    return True


def ligand_pdb_to_vina_pdbqt(
    lig_pdb: Path,
    pdbqt_out: Path,
    *,
    use_rdkit_largest_fragment: bool = False,
    fix_multimolecule_pdbqt: bool = True,
    rigid_pdbqt: bool = False,
) -> None:
    """
    PDB → PDBQT (Open Babel), then optional RDKit pre-pass and/or PDBQT truncation.

    Order when RDKit is on: largest-fragment heavy-only PDB → obabel → truncate.

    Parameters
    ----------
    rigid_pdbqt
        If True, pass Open Babel write option ``-xr`` (rigid ligand, no BRANCH tree).
        Best after geometry is already a single fragment (truncate / RDKit).
    """
    pdb_for_obabel = lig_pdb
    tmp: Path | None = None
    try:
        if use_rdkit_largest_fragment:
            fd, tname = tempfile.mkstemp(suffix="_clean.pdb")
            os.close(fd)
            t = Path(tname)
            tmp = t
            if rdkit_largest_fragment_pdb_to_pdb(lig_pdb, t):
                pdb_for_obabel = t
        obabel_cmd = ["obabel", str(pdb_for_obabel), "-O", str(pdbqt_out)]
        if rigid_pdbqt:
            obabel_cmd.append("-xr")
        subprocess.run(obabel_cmd, check=True, capture_output=True, text=True)
        if fix_multimolecule_pdbqt:
            truncate_pdbqt_to_first_ligand(pdbqt_out)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
