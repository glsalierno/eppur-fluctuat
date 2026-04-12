#!/usr/bin/env python3
"""RDKit 2D descriptors + covalent / bioactivation flags; merge optional docking CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from mechanism_data import lookup_mechanism
except ImportError:
    def lookup_mechanism(name: str):  # type: ignore
        class M:
            bioactivation_status = "Uncertain"
            bioactivation_factor = 0.2
            metabolic_stability = 0.75

        return M()


def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def bioactivation_code(status: str) -> int:
    s = (status or "").lower()
    if "pro" in s:
        return 2
    if "direct" in s:
        return 1
    return 0


def compute_row(smiles: str, name: str, be: float, p_dist: float | None) -> dict:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, EState, Lipinski

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES for {name}: {smiles}")
    mech = lookup_mechanism(name)
    cov = 0
    if p_dist is not None and pd.notna(p_dist) and float(p_dist) < 3.5:
        cov = 1
    elif p_dist is None or (isinstance(p_dist, float) and pd.isna(p_dist)):
        cov = 0
    row = {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": Lipinski.RingCount(mol),
        "EState_indices_sum": float(sum(EState.EStateIndices(mol))),
        "binding_energy_kcal_mol": float(be),
        "covalent_factor": int(cov),
        "bioactivation_code": bioactivation_code(mech.bioactivation_status),
        "be_x_covalent": float(be) * int(cov),
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = args.config.resolve().parent
    paths = cfg.get("paths", {})
    desc_cfg = cfg.get("descriptors", {})

    lig_path = base / paths.get("ligands", "data/ligands.csv")
    out_path = base / paths.get("computed_features", "data/computed_features.csv")
    dock_csv = cfg.get("data", {}).get("docking_results")
    dock_path = (base / dock_csv).resolve() if dock_csv else None

    dock = None
    if dock_path and dock_path.is_file():
        dock = pd.read_csv(dock_path)
        dock["name"] = dock["name"].astype(str).str.strip()

    lig = pd.read_csv(lig_path)
    rows = []
    for _, r in lig.iterrows():
        name = str(r["name"]).strip()
        smi = str(r["smiles"]).strip()
        be = float(r.get("binding_energy_kcal_mol", float("nan")))
        p_dist = None
        if dock is not None:
            m = dock[dock["name"] == name]
            if len(m):
                p_dist = m.iloc[0].get("p_ser203_dist_angstrom")
        feats = compute_row(smi, name, be, float(p_dist) if p_dist is not None and pd.notna(p_dist) else None)
        feats["name"] = name
        feats["cas"] = r.get("cas", "")
        feats["smiles"] = smi
        feats["type"] = r.get("type", "")
        rows.append(feats)

    df = pd.DataFrame(rows)
    cols = [
        "name",
        "cas",
        "smiles",
        "type",
        "MolWt",
        "MolLogP",
        "TPSA",
        "HBA",
        "HBD",
        "RotatableBonds",
        "RingCount",
        "EState_indices_sum",
        "binding_energy_kcal_mol",
        "covalent_factor",
        "bioactivation_code",
        "be_x_covalent",
    ]
    df = df[[c for c in cols if c in df.columns]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} compounds).")


if __name__ == "__main__":
    main()
