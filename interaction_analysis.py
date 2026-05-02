"""
Protein–ligand interaction analysis for docked poses (MDAnalysis + RDKit).

Criteria (see module manual_hbonds for docking poses):

- H-bonds: geometric (H···A, ∠D–H···A, D···A); MDAnalysis HBA when topology allows,
  else manual scan.
- Hydrophobic contacts: unique hydrophobic residues with any heavy atom within 4.0 Å of a
  ligand heavy atom.
- π–π stacking: ligand aromatic ring vs PHE/TYR/TRP/HIS aromatic ring,
  |n1·n2| > cos(30°), centroid distance < 5.5 Å.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis

HYDROPHOBIC_RESNAMES = {
    "ALA",
    "VAL",
    "LEU",
    "ILE",
    "MET",
    "PHE",
    "TRP",
    "PRO",
    "TYR",
    "CYS",
}

AROMATIC_PROTEIN_RESNAMES = {"PHE", "TYR", "TRP", "HIS", "HID", "HIE", "HIP"}


def _obabel(in_path: Path, out_path: Path, extra: List[str]) -> None:
    cmd = ["obabel", str(in_path), "-O", str(out_path)] + extra
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def pdbqt_to_pdb_with_h(pdbqt_path: Path, out_pdb: Path) -> None:
    """
    Convert ligand PDBQT to a protonated PDB using Open Babel then RDKit ``AddHs``.

    Open Babel alone often omits explicit hydrogens when writing PDB from PDBQT; RDKit adds
    coordinates for polar (and default) hydrogens where parsing succeeds.

    Parameters
    ----------
    pdbqt_path : pathlib.Path
        Docked ligand PDBQT.
    out_pdb : pathlib.Path
        Output PDB with explicit H when RDKit succeeds.

    Raises
    ------
    subprocess.CalledProcessError
        If the initial ``obabel`` conversion fails.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as t:
        tmp = Path(t.name)
    try:
        subprocess.run(
            ["obabel", str(pdbqt_path), "-O", str(tmp)],
            check=True,
            capture_output=True,
            text=True,
        )
        from rdkit import Chem

        mol = Chem.MolFromPDBFile(str(tmp), sanitize=True, removeHs=False)
        if mol is None:
            mol = Chem.MolFromPDBFile(str(tmp), sanitize=False, removeHs=False)
        if mol is None:
            _obabel(pdbqt_path, out_pdb, ["-h", "-p", "7.4"])
            return
        mol = Chem.AddHs(mol, addCoords=True)
        Chem.MolToPDBFile(mol, str(out_pdb))
    finally:
        tmp.unlink(missing_ok=True)


def protonate_receptor_pdb(pdb_path: Path, out_pdb: Path) -> None:
    """
    Protonate a receptor PDB at physiological pH via Open Babel.

    Parameters
    ----------
    pdb_path : pathlib.Path
        Input PDB.
    out_pdb : pathlib.Path
        Protonated PDB output path.

    Raises
    ------
    subprocess.CalledProcessError
        If ``obabel`` fails.
    """
    _obabel(pdb_path, out_pdb, ["-h", "-p", "7.4"])


def merge_receptor_ligand(rec_pdb: Path, lig_pdb: Path, out_pdb: Path) -> None:
    """
    Merge receptor and ligand PDBs into one coordinate file for analysis.

    Ligand residues are relabeled as ``resname LIG``, ``segid L``, chain ``L``.

    Parameters
    ----------
    rec_pdb : pathlib.Path
        Protonated receptor PDB.
    lig_pdb : pathlib.Path
        Protonated ligand PDB.
    out_pdb : pathlib.Path
        Combined PDB output path.

    Raises
    ------
    OSError
        On MDAnalysis write failures (disk / permissions).
    """
    u_rec = mda.Universe(str(rec_pdb))
    u_lig = mda.Universe(str(lig_pdb))
    lig = u_lig.select_atoms("all")
    for r in lig.residues:
        r.resname = "LIG"
    if len(lig.residues) > 0:
        lig.residues[0].resid = 1
    lig.segments[0].segid = "L"
    if hasattr(lig, "chainIDs"):
        lig.chainIDs = np.array(["L"] * len(lig), dtype=object)
    merged = mda.Merge(u_rec.atoms, lig)
    merged.atoms.write(str(out_pdb))


@dataclass
class InteractionResult:
    """
    Summary of non-covalent interactions for one docked pose.

    Attributes
    ----------
    name : str
        Ligand/compound label.
    affinity_kcal_mol : float
        Docking score carried through for reporting (may be NaN).
    n_hbonds : int
        Count of accepted H-bond contacts.
    hbond_residues : list of str
        Unique residue labels participating in H-bonds.
    hbond_pairs : list of tuple
        Donor label, acceptor label, angle (degrees).
    n_hydrophobic_contacts : int
        Unique hydrophobic residues within the hydrophobic cutoff.
    hydrophobic_residues : list of str
        Sorted residue labels for hydrophobic contacts.
    n_pipi : int
        Count of π–π stacking contacts passing geometric criteria.
    pipi_pairs : list of str
        Human-readable π–π pair labels.
    hbond_cylinders : list of tuple
        (H position, acceptor position) for visualization.
    active_site_residue_ids : list of int
        Protein residue IDs within ``active_site_radius`` of the ligand.
    pocket_residue_ids : list of int
        Protein residue IDs within ``pocket_radius``.
    merged_complex_pdb : pathlib.Path
        Path to the merged complex PDB used for analysis.
    """

    name: str
    affinity_kcal_mol: float
    n_hbonds: int
    hbond_residues: List[str]  # unique residue labels
    hbond_pairs: List[Tuple[str, str, float]]  # (donor_res, acceptor_res, angle_deg)
    n_hydrophobic_contacts: int
    hydrophobic_residues: List[str]
    n_pipi: int
    pipi_pairs: List[str]
    # For visualization: H-bond (H pos, A pos) in Å
    hbond_cylinders: List[Tuple[np.ndarray, np.ndarray]]
    active_site_residue_ids: List[
        int
    ]  # protein resid for binding-site shell (around ligand)
    pocket_residue_ids: List[int] = field(
        default_factory=list
    )  # all protein resids within pocket_radius (Å)
    merged_complex_pdb: Path = field(default_factory=lambda: Path("."))


def _res_label(atom: mda.core.groups.Atom) -> str:
    return f"{atom.resname.strip()}{atom.resid}{atom.segid.strip() or atom.chainID}"


def _ring_centroid_normal(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centroid and unit normal of (near) planar ring."""
    c = np.asarray(coords, dtype=float)
    center = c.mean(axis=0)
    if len(c) < 3:
        n = np.array([0.0, 0.0, 1.0])
        return center, n / (np.linalg.norm(n) + 1e-9)
    v1 = c[1] - c[0]
    v2 = c[2] - c[0]
    n = np.cross(v1, v2)
    ln = np.linalg.norm(n)
    if ln < 1e-6:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / ln
    return center, n


def _protein_aromatic_rings(
    u: mda.Universe,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """List of (label, centroid, normal) for protein aromatic rings."""
    out: List[Tuple[str, np.ndarray, np.ndarray]] = []
    prot = u.select_atoms("protein")
    for res in prot.residues:
        rn = res.resname.strip()
        if rn not in AROMATIC_PROTEIN_RESNAMES:
            continue
        ring_atoms = None
        if rn == "PHE" or rn == "TYR":
            names = {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}
            atoms = [a for a in res.atoms if a.name.strip() in names]
            if len(atoms) >= 6:
                ring_atoms = atoms[:6]
        elif rn == "TRP":
            # six-membered ring of indole
            names = {"CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"}
            atoms = [a for a in res.atoms if a.name.strip() in names]
            if len(atoms) >= 6:
                ring_atoms = atoms[:6]
        elif rn in ("HIS", "HID", "HIE", "HIP"):
            names = {"CG", "ND1", "CD2", "CE1", "NE2"}
            atoms = [a for a in res.atoms if a.name.strip() in names]
            if len(atoms) >= 5:
                ring_atoms = atoms
        if ring_atoms is None:
            continue
        coords = np.array([a.position for a in ring_atoms])
        cen, n = _ring_centroid_normal(coords)
        label = f"{rn}{res.resid}{res.segid.strip() or ''}"
        out.append((label, cen, n))
    return out


def _ligand_aromatic_rings_rdkit(
    lig_pdb_path: Path,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """RDKit aromatic rings → centroids/normals matched to PDB coordinates."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise SystemExit("Install RDKit for π–π analysis: pip install rdkit") from exc

    mol = Chem.MolFromPDBFile(str(lig_pdb_path), sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromPDBFile(str(lig_pdb_path), sanitize=False, removeHs=False)
    if mol is None:
        return []
    conf = mol.GetConformer()
    rings: List[Tuple[np.ndarray, np.ndarray]] = []
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if not all(a.GetIsAromatic() for a in atoms):
            continue
        coords = np.array([conf.GetAtomPosition(i) for i in ring], dtype=float)
        cen, n = _ring_centroid_normal(coords)
        rings.append((cen, n))
    return rings


def _count_pipi(
    u: mda.Universe,
    lig_pdb_path: Path,
    angle_max_deg: float = 30.0,
    dist_max: float = 5.5,
) -> Tuple[int, List[str]]:
    lig_rings = _ligand_aromatic_rings_rdkit(lig_pdb_path)
    prot_rings = _protein_aromatic_rings(u)
    cos_min = np.cos(np.radians(angle_max_deg))
    pairs: List[str] = []
    count = 0
    for lc, ln in lig_rings:
        for plabel, pc, pn in prot_rings:
            d = np.linalg.norm(lc - pc)
            if d >= dist_max:
                continue
            cang = abs(float(np.dot(ln, pn)))
            if cang < cos_min:
                continue
            count += 1
            pairs.append(f"LIG↔{plabel}")
    return count, pairs


def _hydrophobic_contacts(
    u: mda.Universe, lig_sel: str, cutoff: float = 4.0
) -> Tuple[int, List[str]]:
    lig = u.select_atoms(lig_sel + " and not name H*")
    shell = u.select_atoms(f"protein and around {cutoff} group ligand", ligand=lig)
    seen: Set[str] = set()
    labels: List[str] = []
    for res in shell.residues:
        rn = res.resname.strip()
        if rn not in HYDROPHOBIC_RESNAMES:
            continue
        lab = f"{rn}{res.resid}{res.segid.strip() or ''}"
        if lab not in seen:
            seen.add(lab)
            labels.append(lab)
    return len(labels), sorted(labels)


def _angle_deg(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle at p2 between p1–p2–p3 in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _donor_heavy_for_h(h: mda.core.groups.Atom) -> Optional[mda.core.groups.Atom]:
    """Nearest non-hydrogen atom in the same residue (approximate bond to H)."""
    res = h.residue
    best = None
    best_d = 99.0
    for a in res.atoms:
        if a.element in (None, "", "H"):
            continue
        d = np.linalg.norm(a.position - h.position)
        if d < best_d and d < 1.15:
            best_d = d
            best = a
    return best


def manual_hbonds(
    u: mda.Universe,
    lig_sel: str,
    h_a_max: float = 3.0,
    d_a_max: float = 4.0,
    angle_min_deg: float = 120.0,
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[np.ndarray, np.ndarray]], Set[str]]:
    """
    Geometric H-bonds for ligand↔protein when CONECT records are missing.

    Includes polar (N/O–H) and weak C–H···O contacts common for organophosphate poses.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Merged complex universe.
    lig_sel : str
        MDAnalysis selection string for the ligand (e.g. ``resname LIG``).
    h_a_max : float, optional
        Maximum H···acceptor distance (Å).
    d_a_max : float, optional
        Maximum heavy-atom donor···acceptor distance (Å).
    angle_min_deg : float, optional
        Minimum D–H···A angle (degrees).

    Returns
    -------
    hb_list : list of tuple
        (donor_label, acceptor_label, angle_deg).
    cylinders : list of tuple
        (H xyz, acceptor xyz) for graphics.
    hb_res : set of str
        Unique residue labels touched by H-bonds.
    """
    lig = u.select_atoms(lig_sel)
    prot = u.select_atoms("protein")
    lig_h = lig.select_atoms("name H* or name H")
    prot_h = prot.select_atoms("name H* or name H")
    lig_acc = lig.select_atoms("name O* N*")
    prot_acc = prot.select_atoms("name O* N*")
    hb_list: List[Tuple[str, str, float]] = []
    cylinders: List[Tuple[np.ndarray, np.ndarray]] = []
    hb_res: Set[str] = set()
    seen: Set[Tuple[int, int]] = set()

    def _scan(h_atoms: mda.AtomGroup, acc_atoms: mda.AtomGroup) -> None:
        for h in h_atoms:
            d_at = _donor_heavy_for_h(h)
            if d_at is None:
                continue
            for acc in acc_atoms:
                ha = np.linalg.norm(h.position - acc.position)
                if ha > h_a_max:
                    continue
                ang = _angle_deg(d_at.position, h.position, acc.position)
                if ang < angle_min_deg:
                    continue
                da = np.linalg.norm(d_at.position - acc.position)
                if da > d_a_max:
                    continue
                key = (int(h.ix), int(acc.ix))
                if key in seen:
                    continue
                seen.add(key)
                d_lab = _res_label(d_at)
                a_lab = _res_label(acc)
                hb_list.append((d_lab, a_lab, ang))
                hb_res.add(d_lab)
                hb_res.add(a_lab)
                cylinders.append((h.position.copy(), acc.position.copy()))

    # ligand H → protein acceptors
    _scan(lig_h, prot_acc)
    # protein H → ligand acceptors
    _scan(prot_h, lig_acc)

    return hb_list, cylinders, hb_res


def analyze_complex(
    merged_pdb: Path,
    lig_pdb_for_rdkit: Path,
    ligand_selection: str = "resname LIG",
    name: str = "",
    affinity: float = float("nan"),
    hydrophobic_cutoff: float = 4.0,
    active_site_radius: float = 8.0,
    pocket_radius: float = 5.0,
) -> InteractionResult:
    """
    Compute hydrogen bonds, hydrophobic contacts, and π–π stacking for a merged complex.

    Attempts MDAnalysis ``HydrogenBondAnalysis`` first; falls back to :func:`manual_hbonds`.

    Parameters
    ----------
    merged_pdb : pathlib.Path
        Receptor + ligand PDB.
    lig_pdb_for_rdkit : pathlib.Path
        Ligand PDB path readable by RDKit for aromatic rings.
    ligand_selection : str, optional
        MDAnalysis selection for ligand atoms.
    name : str, optional
        Compound label stored in the result.
    affinity : float, optional
        Docking affinity echoed into the result object.
    hydrophobic_cutoff : float, optional
        Heavy-atom distance cutoff for hydrophobic contacts (Å).
    active_site_radius : float, optional
        Radius around ligand defining active-site residue IDs (Å).
    pocket_radius : float, optional
        Tighter radius for pocket residue IDs (Å).

    Returns
    -------
    InteractionResult
        Structured counts and labels for reporting / visualization.
    """
    u = mda.Universe(str(merged_pdb))
    lig_sel = ligand_selection

    # H-bonds between ligand and protein
    hb_list: List[Tuple[str, str, float]] = []
    hb_res: Set[str] = set()
    cylinders: List[Tuple[np.ndarray, np.ndarray]] = []

    try:
        hb = HydrogenBondAnalysis(
            universe=u,
            between=[[lig_sel, "protein"], ["protein", lig_sel]],
            d_a_cutoff=3.5,
            d_h_a_angle_cutoff=120,
        )
        hb.hydrogens_sel = hb.guess_hydrogens(f"{lig_sel} or protein")
        hb.acceptors_sel = hb.guess_acceptors(f"{lig_sel} or protein")
        hb.run()
        out = hb.results.hbonds
        if out is not None and len(out):
            for row in out:
                d_ix, h_ix, a_ix = int(row[1]), int(row[2]), int(row[3])
                d_at, h_at, a_at = u.atoms[d_ix], u.atoms[h_ix], u.atoms[a_ix]
                d_lab = _res_label(d_at)
                a_lab = _res_label(a_at)
                ang = float(row[5]) if len(row) > 5 else float("nan")
                hb_list.append((d_lab, a_lab, ang))
                hb_res.add(d_lab)
                hb_res.add(a_lab)
                cylinders.append((h_at.position.copy(), a_at.position.copy()))
    except Exception:
        pass

    if not hb_list:
        hb_list, cylinders, hb_res = manual_hbonds(u, lig_sel)

    n_hb = len(hb_list)
    hb_res_sorted = sorted(hb_res)

    n_hp, hp_labels = _hydrophobic_contacts(u, lig_sel, hydrophobic_cutoff)

    n_pi, pi_pairs = _count_pipi(u, lig_pdb_for_rdkit)

    lig = u.select_atoms(lig_sel)
    site = u.select_atoms(
        f"protein and around {active_site_radius} group ligand", ligand=lig
    )
    active_ids = sorted({int(r.resid) for r in site.residues})
    pocket_shell = u.select_atoms(
        f"protein and around {pocket_radius} group ligand", ligand=lig
    )
    pocket_ids = sorted({int(r.resid) for r in pocket_shell.residues})

    return InteractionResult(
        name=name,
        affinity_kcal_mol=affinity,
        n_hbonds=n_hb,
        hbond_residues=hb_res_sorted,
        hbond_pairs=hb_list,
        n_hydrophobic_contacts=n_hp,
        hydrophobic_residues=hp_labels,
        n_pipi=n_pi,
        pipi_pairs=pi_pairs,
        hbond_cylinders=cylinders,
        active_site_residue_ids=active_ids,
        pocket_residue_ids=pocket_ids,
        merged_complex_pdb=merged_pdb,
    )


def prepare_and_analyze(
    receptor_clean_pdb: Path,
    pose_pdbqt: Path,
    name: str,
    affinity: float,
    work_dir: Path,
    hydrophobic_cutoff: float = 4.0,
    active_site_radius: float = 8.0,
    pocket_radius: float = 5.0,
) -> InteractionResult:
    """
    High-level helper: protonate, merge receptor/ligand from a pose PDBQT, analyze interactions.

    Parameters
    ----------
    receptor_clean_pdb : pathlib.Path
        Clean receptor PDB (without ligand).
    pose_pdbqt : pathlib.Path
        Docked ligand PDBQT.
    name : str
        Compound label.
    affinity : float
        Docking score for bookkeeping.
    work_dir : pathlib.Path
        Scratch/output directory for intermediates.
    hydrophobic_cutoff : float, optional
        Passed to :func:`analyze_complex`.
    active_site_radius : float, optional
        Passed to :func:`analyze_complex`.
    pocket_radius : float, optional
        Passed to :func:`analyze_complex`.

    Returns
    -------
    InteractionResult
        Interaction summary for the pose.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(pose_pdbqt).stem
    lig_h = work_dir / f"{tag}_lig_h.pdb"
    rec_h = work_dir / f"{tag}_rec_h.pdb"
    merged = work_dir / f"{tag}_complex.pdb"

    pdbqt_to_pdb_with_h(pose_pdbqt, lig_h)
    protonate_receptor_pdb(receptor_clean_pdb, rec_h)
    merge_receptor_ligand(rec_h, lig_h, merged)
    return analyze_complex(
        merged,
        lig_h,
        name=name,
        affinity=affinity,
        hydrophobic_cutoff=hydrophobic_cutoff,
        active_site_radius=active_site_radius,
        pocket_radius=pocket_radius,
    )
