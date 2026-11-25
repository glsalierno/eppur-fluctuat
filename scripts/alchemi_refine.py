"""
Post–Vina pose refinement with NVIDIA ALCHEMI Toolkit (FIRE2 + MLIP).

Uses the bundled DemoModel for end-to-end pipeline testing. The Demo potential is
**not** a validated force field for protein–ligand binding; swap in a production
MLIP (MACE, Orbital, …) for scientific studies.

Requires: nvalchemi-toolkit, nvalchemi-toolkit-ops, torch, ase, MDAnalysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import MDAnalysis as mda
import numpy as np
import torch


def _element_atomic_numbers(symbols: np.ndarray) -> np.ndarray:
    import periodictable as pt

    out = []
    for s in symbols:
        sym = str(s).strip()
        if not sym:
            sym = "C"
        try:
            out.append(int(pt.elements.symbol(sym).number))
        except Exception:
            out.append(6)
    return np.asarray(out, dtype=np.int64)


def merged_pdb_to_batch(merged_pdb: Path, device: torch.device):
    """
    Build an ALCHEMI ``Batch`` from a merged receptor–ligand PDB.

    Parameters
    ----------
    merged_pdb : pathlib.Path
        PDB with protein + ``resname LIG`` ligand.
    device : torch.device
        Torch device for tensors.

    Returns
    -------
    Batch
        Single-structure batch for NVIDIA ALCHEMI dynamics (internal type).
    """
    from nvalchemi._typing import AtomCategory
    from nvalchemi.data import AtomicData, Batch

    u = mda.Universe(str(merged_pdb))
    ag = u.atoms
    pos = torch.as_tensor(ag.positions, dtype=torch.float32, device=device)
    sym = np.array([str(x) if x else "X" for x in ag.elements], dtype=object)
    nums = torch.as_tensor(_element_atomic_numbers(sym), dtype=torch.long, device=device)
    res = np.char.strip(ag.resnames.astype(str))
    cat_np = np.where(res == "LIG", AtomCategory.GAS.value, AtomCategory.SPECIAL.value)
    atom_categories = torch.as_tensor(cat_np, dtype=torch.long, device=device)
    n = pos.shape[0]
    data = AtomicData(
        atomic_numbers=nums,
        positions=pos,
        forces=torch.zeros((n, 3), dtype=torch.float32, device=device),
        energy=torch.zeros(1, 1, dtype=torch.float32, device=device),
        velocities=torch.zeros((n, 3), dtype=torch.float32, device=device),
        atom_categories=atom_categories,
    )
    return Batch.from_data_list([data], device=device)


def write_batch_positions_to_pdb(merged_pdb: Path, batch, out_pdb: Path) -> None:
    """
    Replace coordinates in ``merged_pdb`` with relaxed positions from ``batch``.

    Parameters
    ----------
    merged_pdb : pathlib.Path
        Original merged PDB (topology reference).
    batch
        ALCHEMI batch carrying updated positions.
    out_pdb : pathlib.Path
        Output PDB path.

    Raises
    ------
    RuntimeError
        If atom counts disagree between topology and batch coordinates.
    """
    u = mda.Universe(str(merged_pdb))
    coords = batch.positions.detach().cpu().numpy()
    if coords.shape[0] != u.atoms.n_atoms:
        raise RuntimeError(f"Atom count mismatch after refinement: pdb={u.atoms.n_atoms} batch={coords.shape[0]}")
    u.atoms.positions = coords
    u.atoms.write(str(out_pdb))


def refine_with_alchemi_fire2(
    merged_pdb: Path,
    out_pdb: Path,
    *,
    n_steps: int = 25,
    dt: float = 0.02,
    device: str = "auto",
    use_demo_potential: bool = True,
) -> Tuple[str, torch.device]:
    """
    Run FIRE2 with frozen protein (SPECIAL) and mobile ligand (``LIG``) using the demo MLIP.

    Parameters
    ----------
    merged_pdb : pathlib.Path
        Input merged complex.
    out_pdb : pathlib.Path
        Output PDB with updated coordinates.
    n_steps : int, optional
        FIRE2 steps.
    dt : float, optional
        FIRE2 time step.
    device : str, optional
        ``auto``, ``cpu``, or ``cuda``.
    use_demo_potential : bool, optional
        Must be True in this repository wiring.

    Returns
    -------
    tuple of (str, torch.device)
        Human-readable note and device used.

    Raises
    ------
    NotImplementedError
        If ``use_demo_potential`` is False.
    TypeError
        If the optimizer does not return the expected batch type.
    """
    if not use_demo_potential:
        raise NotImplementedError(
            "Only use_demo_potential=True is wired in this repo; plug your MLIP into "
            "nvalchemi.models and replace DemoModelWrapper."
        )

    from nvalchemi.data import Batch
    from nvalchemi.dynamics.hooks.freeze import FreezeAtomsHook
    from nvalchemi.dynamics.optimizers import FIRE2
    from nvalchemi.models.demo import DemoModel, DemoModelWrapper

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    batch = merged_pdb_to_batch(merged_pdb, dev)
    model = DemoModelWrapper(DemoModel())
    model = model.to(dev)
    hooks = [FreezeAtomsHook(freeze_category=-1)]
    opt = FIRE2(model=model, dt=dt, n_steps=n_steps, hooks=hooks)
    opt.compute(batch)
    opt.run(batch, n_steps=n_steps)
    if not isinstance(batch, Batch):
        raise TypeError("expected Batch from FIRE2.run")
    write_batch_positions_to_pdb(merged_pdb, batch, out_pdb)
    note = f"ALCHEMI FIRE2 DemoModel steps={n_steps} device={dev}"
    return note, dev


def check_alchemi_available() -> bool:
    """
    Return True if NVIDIA ALCHEMI toolkit packages import successfully.

    Returns
    -------
    bool
        Import availability flag.
    """
    try:
        import nvalchemi  # noqa: F401
        import nvalchemiops  # noqa: F401
    except ImportError:
        return False
    return True


def main() -> None:
    """
    CLI entry: run FIRE2 on a merged complex PDB and write refined coordinates.

    Raises
    ------
    SystemExit
        If ALCHEMI is not installed or arguments are invalid.
    """
    ap = __import__("argparse").ArgumentParser(description="ALCHEMI FIRE2 refine merged complex PDB")
    ap.add_argument("merged_pdb", type=Path)
    ap.add_argument("out_pdb", type=Path)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if not check_alchemi_available():
        print(
            "Install: pip install nvalchemi-toolkit nvalchemi-toolkit-ops torch ase",
            file=sys.stderr,
        )
        raise SystemExit(1)
    note, dev = refine_with_alchemi_fire2(
        args.merged_pdb,
        args.out_pdb,
        n_steps=args.steps,
        dt=args.dt,
        device=args.device,
    )
    print(note, dev)


if __name__ == "__main__":
    main()
