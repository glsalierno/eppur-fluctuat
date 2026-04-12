#!/usr/bin/env python3
"""
Merge manual / literature IC50 values with optional ChEMBL (human AChE, CHEMBL220) activity.

Direct ToxVal DB access is not wired here (EPA bulk export would be offline). Use:
  - data/experimental_manual.csv (required baseline)
  - Optional ChEMBL REST API for compounds not in manual (see config.yaml).

Writes:
  - data/experimental_data.csv
  - data/curated_pIC50.csv  (name, cas, ic50_nM, pIC50, source)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

try:
    import requests
except ImportError:
    requests = None  # type: ignore


def _pic50_from_ic50_nM_np(ic50_nm: float) -> float:
    import numpy as np

    if ic50_nm <= 0:
        return float("nan")
    return 9.0 - float(np.log10(ic50_nm))


def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_chembl_ic50_for_smiles(
    smiles: str,
    target_chembl: str,
    timeout: float = 15.0,
) -> tuple[float | None, str]:
    """Return (IC50 in nM, source note) or (None, reason)."""
    if requests is None:
        return None, "requests not installed"
    try:
        from urllib.parse import quote

        q = quote(smiles, safe="")
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_structures__canonical_smiles__exact={q}&format=json&limit=5"
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None, f"molecule HTTP {r.status_code}"
        js = r.json()
        mols = js.get("molecules") or []
        if not mols:
            return None, "no ChEMBL molecule for SMILES"
        mid = mols[0].get("molecule_chembl_id")
        if not mid:
            return None, "no chembl id"
        time.sleep(0.35)  # be polite
        aurl = (
            f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
            f"?molecule_chembl_id={mid}&target_chembl_id={target_chembl}"
            f"&standard_type=IC50&assay_type=F&limit=20&format=json"
        )
        r2 = requests.get(aurl, timeout=timeout)
        if r2.status_code != 200:
            return None, f"activity HTTP {r2.status_code}"
        acts = r2.json().get("activities") or []
        ic50_uM_values = []
        for a in acts:
            val = a.get("standard_value")
            units = (a.get("standard_units") or "").upper()
            if val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if units in ("NM", "N"):
                ic50_uM_values.append(v / 1000.0)
            elif units in ("UM", "µM", "uM"):
                ic50_uM_values.append(v)
            elif units in ("MM", "M"):
                ic50_uM_values.append(v * 1000.0)
        if not ic50_uM_values:
            return None, "no IC50 with parseable units"
        ic50_nM = min(ic50_uM_values) * 1000.0  # uM -> nM
        return float(ic50_nM), f"ChEMBL min IC50 (approx) for {mid}"
    except Exception as exc:
        return None, str(exc)[:120]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build experimental_data / curated pIC50 tables")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = args.config.resolve().parent
    data_cfg = cfg.get("data", {})
    manual_path = base / data_cfg.get("experimental_manual", "data/experimental_manual.csv")
    out_exp = base / data_cfg.get("experimental_merged", "data/experimental_data.csv")
    out_pic = base / data_cfg.get("curated_pic50", "data/curated_pIC50.csv")
    lig_path = base / cfg.get("paths", {}).get("ligands", "data/ligands.csv")

    manual = pd.read_csv(manual_path)
    manual["name"] = manual["name"].astype(str).str.strip()
    manual["source"] = manual["source"].astype(str) + " (manual table)"

    lig = pd.read_csv(lig_path)
    lig["name"] = lig["name"].astype(str).str.strip()

    source_mode = str(data_cfg.get("experimental_source", "manual")).lower()
    target = cfg.get("chembl_target", "CHEMBL220")

    chembl_rows = []
    if source_mode in ("chembl", "merge") and requests is not None:
        have = set(manual["name"].str.lower())
        for _, row in lig.iterrows():
            nm = str(row["name"]).strip()
            if nm.lower() in have:
                continue
            smi = str(row.get("smiles", "")).strip()
            if not smi:
                continue
            ic50_nm, note = fetch_chembl_ic50_for_smiles(smi, target)
            if ic50_nm is not None:
                chembl_rows.append(
                    {
                        "name": nm,
                        "ic50_nM": ic50_nm,
                        "source": note,
                    }
                )
                have.add(nm.lower())
            time.sleep(0.35)

    chembl_df = pd.DataFrame(chembl_rows) if chembl_rows else pd.DataFrame(columns=["name", "ic50_nM", "source"])

    if len(chembl_df):
        merged_lit = pd.concat([manual, chembl_df], ignore_index=True)
    else:
        merged_lit = manual.copy()

    # Join CAS from ligands
    cas_map = lig.set_index("name")["cas"].to_dict()
    merged_lit["cas"] = merged_lit["name"].map(lambda n: cas_map.get(n, ""))

    import numpy as np

    merged_lit["pIC50"] = merged_lit["ic50_nM"].apply(
        lambda x: _pic50_from_ic50_nM_np(float(x)) if pd.notna(x) and float(x) > 0 else float("nan")
    )

    out_exp.parent.mkdir(parents=True, exist_ok=True)
    merged_lit.to_csv(out_exp, index=False)
    merged_lit[["name", "cas", "ic50_nM", "pIC50", "source"]].to_csv(out_pic, index=False)

    print(f"Wrote {out_exp} ({len(merged_lit)} experimental rows).")
    print(f"Wrote {out_pic}.")
    if source_mode == "chembl" and requests is None:
        print("Install requests + use merge mode for ChEMBL.", file=sys.stderr)


if __name__ == "__main__":
    main()
