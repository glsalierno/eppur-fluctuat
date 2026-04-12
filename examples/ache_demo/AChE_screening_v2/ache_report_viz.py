"""
Build AChE_report.html with 3Dmol.js: representation toggles, catalytic triad highlight, H-bonds, pocket.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from interaction_analysis import InteractionResult


def _pdbqt_to_pdb_text(pdbqt_path: Path) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as t:
        tmp = Path(t.name)
    try:
        subprocess.run(
            ["obabel", str(pdbqt_path), "-O", str(tmp)],
            check=True,
            capture_output=True,
            text=True,
        )
        return tmp.read_text(encoding="utf-8", errors="replace")
    finally:
        tmp.unlink(missing_ok=True)


def catalytic_triad_label_positions(receptor_pdb: Path) -> List[Dict[str, Any]]:
    """CA positions for Ser203, His447, Glu334 (human AChE, 4EY7 chain A)."""
    import MDAnalysis as mda

    u = mda.Universe(str(receptor_pdb))
    triad = [(203, "SER", "Ser203"), (447, "HIS", "His447"), (334, "GLU", "Glu334")]
    out: List[Dict[str, Any]] = []
    for resid, rn, label in triad:
        sel = u.select_atoms(f"protein and resid {resid} and resname {rn} and name CA")
        if len(sel) == 0:
            sel = u.select_atoms(f"protein and resid {resid} and name CA")
        if len(sel) == 0:
            continue
        p = sel[0].position
        out.append({"text": label, "x": float(p[0]), "y": float(p[1]), "z": float(p[2])})
    return out


def _hp_resids_from_labels(labels: List[str]) -> List[int]:
    import re

    rids = []
    for lab in labels:
        m = re.match(r"^[A-Z]{3}(\d+)", lab)
        if m:
            rids.append(int(m.group(1)))
    return sorted(set(rids))[:400]


def write_ache_report_html(
    receptor_clean_pdb: Path,
    interaction_results: List["InteractionResult"],
    pose_pdbqts: List[Path],
    out_html: Path,
) -> None:
    rec_txt = receptor_clean_pdb.read_text(encoding="utf-8", errors="replace")
    rec_js = json.dumps(rec_txt)
    triad = catalytic_triad_label_positions(receptor_clean_pdb)
    triad_js = json.dumps(triad)

    lig_payload: List[Dict[str, Any]] = []
    for res, pq in zip(interaction_results, pose_pdbqts):
        pdb_txt = _pdbqt_to_pdb_text(pq)
        cyls = []
        for s, e in res.hbond_cylinders:
            cyls.append(
                [
                    [float(s[0]), float(s[1]), float(s[2])],
                    [float(e[0]), float(e[1]), float(e[2])],
                ]
            )
        hp_ids = _hp_resids_from_labels(res.hydrophobic_residues)
        lig_payload.append(
            {
                "name": res.name,
                "affinity": float(res.affinity_kcal_mol),
                "pdb": pdb_txt,
                "cylinders": cyls,
                "hydrophobic_resi": hp_ids,
                "pocket_resi": res.pocket_residue_ids[:500],
                "n_hb": res.n_hbonds,
                "n_hp": res.n_hydrophobic_contacts,
                "n_pi": res.n_pipi,
                "hb_res": ", ".join(res.hbond_residues[:16]) or "—",
            }
        )

    lig_js = json.dumps(lig_payload)
    nlig = len(lig_payload)

    # Embedded 3Dmol: one receptor model + one ligand model (swapped by dropdown)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>AChE screening — interactive report (4EY7)</title>
  <script src="https://3dmol.org/build/3Dmol-min.js"></script>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #0f1419; color: #e6edf3; }}
    header {{ padding: 12px 16px; background: #161b22; border-bottom: 1px solid #30363d; }}
    h1 {{ margin: 0; font-size: 1.15rem; }}
    .sub {{ color: #8b949e; font-size: 0.88rem; margin-top: 6px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 10px 16px; background: #161b22; }}
    .toolbar label {{ color: #8b949e; font-size: 0.85rem; }}
    select, button {{
      background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px;
      padding: 6px 10px; font-size: 0.9rem; cursor: pointer;
    }}
    button:hover, select:hover {{ background: #30363d; }}
    #viewport {{ width: 100%; height: 72vh; background: #fff; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1200px; margin: 12px 16px; font-size: 0.88rem; }}
    th, td {{ border: 1px solid #30363d; padding: 8px 10px; text-align: left; }}
    th {{ background: #161b22; }}
    .note {{ padding: 8px 16px; color: #8b949e; font-size: 0.82rem; max-width: 1100px; }}
  </style>
</head>
<body>
  <header>
    <h1>Human AChE (4EY7) — docked poses &amp; interactions</h1>
    <div class="sub">Native pocket center: co-crystallized ligand E20 (donepezil). Catalytic triad: Ser203 · His447 · Glu334.
    Vina uses CPU in the Python API (no CUDA). Representations apply to the protein; ligand stays as sticks.</div>
  </header>
  <div class="toolbar">
    <label>Ligand (top {nlig})</label>
    <select id="ligSelect"></select>
    <label>Protein style</label>
    <select id="reprSelect">
      <option value="cartoon" selected>Cartoon (default)</option>
      <option value="sphere">Space-filling (CPK)</option>
      <option value="ballstick">Ball &amp; stick</option>
      <option value="stick">Stick / skeleton</option>
      <option value="line">Wire (line)</option>
      <option value="surface">Surface (hydrophobicity color)</option>
      <option value="trace">Cα trace</option>
    </select>
    <button type="button" id="btnReset">Reset view</button>
  </div>
  <div id="viewport"></div>
  <p class="note">Yellow cylinders: H-bonds (inferred). Orange sticks: hydrophobic residues (≤5 Å). Triad: green highlight + labels.</p>
  <h2 style="padding:0 16px;font-size:1rem">Top compounds — interactions</h2>
  <table id="rankTable">
    <tr><th>Rank</th><th>Name</th><th>Vina (kcal/mol)</th><th>H-bonds</th><th>H-bond residues</th><th>Hydrophobic (≤5 Å)</th><th>π–π</th></tr>
  </table>
  <script>
    var LIGANDS = {lig_js};
    var REC = {rec_js};
    var TRIAD = {triad_js};
    var viewer = $3Dmol.createViewer(document.getElementById("viewport"), {{backgroundColor: "white"}});
    var triadResi = [203, 447, 334];
    var chainSel = {{chain: "A"}};

    function clearSurface() {{
      try {{ viewer.removeAllSurfaces(); }} catch(e) {{}}
    }}

    function applyProteinStyle(mode) {{
      viewer.setStyle({{model: 0}}, {{}});
      clearSurface();
      if (mode === "cartoon") {{
        viewer.setStyle({{model: 0}}, {{cartoon: {{color: "spectrum"}}}});
      }} else if (mode === "sphere") {{
        viewer.setStyle({{model: 0}}, {{sphere: {{scale: 0.22, colorscheme: "Jmol"}}}});
      }} else if (mode === "ballstick") {{
        viewer.setStyle({{model: 0}}, {{stick: {{radius: 0.12}}, sphere: {{scale: 0.13, colorscheme: "Jmol"}}}});
      }} else if (mode === "stick") {{
        viewer.setStyle({{model: 0}}, {{stick: {{radius: 0.12, colorscheme: "Jmol"}}}});
      }} else if (mode === "line") {{
        viewer.setStyle({{model: 0}}, {{line: {{radius: 0.05, colorscheme: "Jmol"}}}});
      }} else if (mode === "surface") {{
        viewer.setStyle({{model: 0}}, {{cartoon: {{color: "spectrum"}}}});
        viewer.addSurface($3Dmol.VDW, {{opacity: 0.42, colorscheme: "amino"}}, {{model: 0}});
      }} else if (mode === "trace") {{
        viewer.setStyle({{model: 0}}, {{cartoon: {{style: "trace", color: "spectrum"}}}});
      }}
    }}

    function styleTriad() {{
      viewer.setStyle({{model: 0, resi: triadResi, chain: "A"}}, {{stick: {{colorscheme: "greenCarbon", radius: 0.22}}}});
    }}

    function drawHBonds(cyls) {{
      for (var i = 0; i < cyls.length; i++) {{
        var a = cyls[i][0], b = cyls[i][1];
        viewer.addCylinder({{
          start: {{x: a[0], y: a[1], z: a[2]}},
          end: {{x: b[0], y: b[1], z: b[2]}},
          radius: 0.06,
          color: "gold",
          fromCap: 1,
          toCap: 1,
          dashed: true
        }});
      }}
    }}

    function drawTriadLabels() {{
      for (var i = 0; i < TRIAD.length; i++) {{
        var t = TRIAD[i];
        viewer.addLabel(t.text, {{
          position: {{x: t.x, y: t.y, z: t.z}},
          backgroundColor: "rgba(0,0,0,0.65)",
          fontColor: "white",
          fontSize: 12,
          backgroundOpacity: 0.85
        }});
      }}
    }}

    function buildScene(ligIndex) {{
      viewer.removeAllModels();
      viewer.removeAllShapes();
      viewer.removeAllLabels();
      viewer.addModel(REC, "pdb");
      var L = LIGANDS[ligIndex];
      viewer.addModel(L.pdb, "pdb");
      applyProteinStyle(document.getElementById("reprSelect").value);
      // Pocket: light gray sticks for non-hydrophobic contact residues
      var pocket = L.pocket_resi || [];
      var hp = {{}};
      for (var j = 0; j < L.hydrophobic_resi.length; j++) hp[L.hydrophobic_resi[j]] = true;
      for (var k = 0; k < pocket.length; k++) {{
        var r = pocket[k];
        if (hp[r]) {{
          viewer.setStyle({{model: 0, resi: r, chain: "A"}}, {{stick: {{color: "orange", radius: 0.1}}}});
        }} else {{
          viewer.setStyle({{model: 0, resi: r, chain: "A"}}, {{stick: {{color: "#cccccc", radius: 0.06}}}});
        }}
      }}
      styleTriad();
      viewer.setStyle({{model: 1}}, {{stick: {{radius: 0.2, colorscheme: "cyanCarbon"}}}});
      drawHBonds(L.cylinders || []);
      drawTriadLabels();
      viewer.zoomTo({{model: 1}});
      viewer.render();
    }}

    // Init UI
    var sel = document.getElementById("ligSelect");
    for (var i = 0; i < LIGANDS.length; i++) {{
      var o = document.createElement("option");
      o.value = i;
      o.text = (i+1) + ". " + LIGANDS[i].name + " (" + LIGANDS[i].affinity.toFixed(2) + " kcal/mol)";
      sel.appendChild(o);
    }}
    sel.onchange = function() {{ buildScene(parseInt(sel.value, 10)); }};
    document.getElementById("reprSelect").onchange = function() {{
      buildScene(parseInt(sel.value, 10));
    }};
    document.getElementById("btnReset").onclick = function() {{ buildScene(parseInt(sel.value, 10)); }};

    var tbl = document.getElementById("rankTable");
    for (var r = 0; r < LIGANDS.length; r++) {{
      var L = LIGANDS[r];
      var tr = document.createElement("tr");
      tr.innerHTML = "<td>" + (r+1) + "</td><td>" + L.name + "</td><td>" + L.affinity.toFixed(2) + "</td><td>" + L.n_hb + "</td><td>" + L.hb_res + "</td><td>" + L.n_hp + "</td><td>" + L.n_pi + "</td>";
      tbl.appendChild(tr);
    }}

    buildScene(0);
  </script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
