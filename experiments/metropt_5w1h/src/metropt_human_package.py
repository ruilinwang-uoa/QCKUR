"""
metropt_human_package.py -- build the INDEPENDENT human-validation package
for the MetroPT campaign (paper section 6).

Package root: <metropt_5w1h>/human_validation/
  README.md                 design + what the user updates
  RATER_RECRUITMENT.md      recruiting blurb
  instrument/sample_manifest.json   30 windows x 4 systems, per-instance
                                    randomized letter key (seed 42)
  instrument/rating_form.html       self-contained blinded rating form
  ratings_in/               user drops human_validation_ratings-X.json
  analysis/analyze_metropt_human.py runs when ratings exist
  analysis/SELECTION_LOG.md how the 30 windows were chosen

Stratification (prereg section 9): signature-visible positives (<=8),
non-visible positives (7 -- the truthfulness cases), nominal controls
(12), OOC-candidate controls (3). Same rating dimensions as the AI4I
study (comparability) + one MetroPT-specific action-appropriateness.

Usage: python metropt_human_package.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from metropt_handlers import handle, load_spec
from metropt_render import digest_text

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
DEST = ROOT / "human_validation"
SYSTEMS = ["B1", "B2", "B5", "B8"]
SEED = 42


def select_windows(gt: dict) -> tuple[list[dict], list[str]]:
    """Stratified selection of 30 windows; returns (entries, log_lines)."""
    rng = random.Random(SEED)
    allw = list(gt.values())
    visible = [w for w in allw if w["kind"] == "positive"
               and any(w["visible"].values())]
    nonvis = [w for w in allw if w["kind"] == "positive"
              and not any(w["visible"].values())]
    ooc = [w for w in allw if w.get("ooc_candidate")]
    nominal = [w for w in allw if w["kind"] == "control"
               and not w.get("ooc_candidate")]
    rng.shuffle(visible); rng.shuffle(nonvis)
    rng.shuffle(ooc); rng.shuffle(nominal)
    # visible: spread over modes/vehicles/offsets -- prefer distinct
    # (vehicle,event,offset) triples, cap 8
    picked_vis, used = [], set()
    for w in sorted(visible, key=lambda x: x["event"]["lead_h"]):
        k = (w["vehicle"], w["event"]["id"], w["event"]["lead_h"])
        if k not in used:
            picked_vis.append(w); used.add(k)
        if len(picked_vis) == 8:
            break
    sel = (picked_vis[:8] + nonvis[:7] + nominal[:12] + ooc[:3])
    log = [f"visible-signature positives: {len(picked_vis)}",
           f"non-visible positives (truthfulness cases): {min(7,len(nonvis))}",
           f"nominal controls: {min(12,len(nominal))}",
           f"OOC-candidate controls: {min(3,len(ooc))}",
           f"total: {len(sel)} (seed {SEED})"]
    return sel, log


def main() -> None:
    spec = load_spec()
    gt = {f"{g['vehicle']}::{g['window_id']}": g for g in json.load(
        open(RUNS / "metropt_gt.json", encoding="utf-8"))["windows"]}
    sel, log = select_windows(gt)
    rng = random.Random(SEED + 1)

    (DEST / "instrument").mkdir(parents=True, exist_ok=True)
    (DEST / "ratings_in").mkdir(parents=True, exist_ok=True)
    (DEST / "analysis").mkdir(parents=True, exist_ok=True)

    key: dict[str, dict[str, str]] = {}
    blocks: list[str] = []
    for w in sel:
        veh, wid = w["vehicle"], w["window_id"]
        letters = ["A", "B", "C", "D"]
        sys_order = SYSTEMS[:]
        rng.shuffle(sys_order)
        key[wid] = dict(zip(letters, sys_order))
        reports = {}
        for s in SYSTEMS:
            rep = pd.read_parquet(RUNS / f"reports_metropt_{s}.parquet")
            row = rep[(rep.vehicle == veh) & (rep.window_id == wid)].iloc[0]
            reports[s] = str(row.text)
        feat = None
        from metropt_l1 import _feat_row
        feat = _feat_row(veh, wid)
        ctx = digest_text(feat, veh)
        blocks.append(json.dumps({
            "window_id": wid, "vehicle": veh,
            "context": ctx, "reports": reports,
            "letter_key": key[wid]}, ensure_ascii=False))

    manifest = {"seed": SEED, "systems": SYSTEMS, "key": key,
                "n_instances": len(sel), "selection_log": log}
    (DEST / "instrument" / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    (DEST / "analysis" / "SELECTION_LOG.md").write_text(
        "# 30-window stratified selection (seed 42)\n\n"
        + "\n".join(f"- {l}" for l in log)
        + "\n\nWindows:\n"
        + "\n".join(f"- {w['vehicle']} {w['window_id']} "
                    f"(kind={w['kind']}, visible={any(w['visible'].values())})"
                    for w in sel), encoding="utf-8")

    # ---------------- rating form (DOM-built, no innerHTML) --------------
    dims = [("faith", "Faithfulness to the telemetry (1=poor, 5=excellent)"),
            ("compl", "Completeness (1=poor, 5=excellent)"),
            ("coh", "Coherence / readability (1=poor, 5=excellent)"),
            ("act", "Actionability (1=poor, 5=excellent)"),
            ("state", "State claim matches the telemetry (1=poor, 5=excellent)"),
            ("actionok", "Recommended action is appropriate (1=poor, 5=excellent)")]
    form = _render_form(blocks, dims)
    (DEST / "instrument" / "rating_form.html").write_text(
        form, encoding="utf-8")
    print(f"package -> {DEST}")
    print("\n".join(log))
    print("form bytes:", len(form))


def _render_form(blocks_json: list[str], dims: list[tuple[str, str]]) -> str:
    data = "[" + ",".join(blocks_json) + "]"
    dim_rows = "".join(
        f'dimRow("{k}","{t}");' for k, t in dims)
    return """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>MetroPT maintenance-report rating study</title>
<style>
 body{font-family:system-ui;margin:2rem;max-width:60rem}
 .inst{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}
 .rep{background:#f7f7f7;padding:0.7rem;margin:0.5rem 0;white-space:pre-wrap;
      font-family:inherit}
 .score{margin:0.3rem 0}
 label{display:inline-block;width:22rem}
 button{padding:0.6rem 1.4rem;font-size:1rem}
</style></head><body>
<h1>Maintenance-report rating study</h1>
<p>Rater code: <input id="rater"></p>
<p>For each telemetry window you see four reports (A-D, randomized).
Rate each on six 1-5 scales (5 = best, 1 = worst), then rank
readability 1-4 (ties allowed, 1 = most readable, 4 = least readable).
Judge only against the telemetry context shown.</p>
<div id="root"></div>
<button onclick="save()">Download ratings JSON</button>
<script>
const DATA = __DATA__;
const DIMS = __DIMS__;
function el(tag, attrs, txt){const e=document.createElement(tag);
 for(const k in attrs||{}) e.setAttribute(k,attrs[k]);
 if(txt!==undefined) e.appendChild(document.createTextNode(txt));return e;}
function build(){const root=document.getElementById('root');
 DATA.forEach((inst,i)=>{
  const box=el('div',{'class':'inst'});
  box.appendChild(el('h3',{},`Window ${i+1} of ${DATA.length} — ${inst.window_id}`));
  const c=el('div',{'class':'rep'});c.appendChild(el('b',{},'TELEMETRY CONTEXT:'));
  c.appendChild(el('br'));c.appendChild(document.createTextNode(inst.context));
  box.appendChild(c);
  for(const L of ['A','B','C','D']){
   const r=el('div',{'class':'rep'});
   r.appendChild(el('b',{},'REPORT '+L));
   r.appendChild(el('br'));
   r.appendChild(document.createTextNode(inst.reports[inst.letter_key[L]]||''));
   for(const [k,t] of DIMS){
    const row=el('div',{'class':'score'});
    const lab=el('label',{},`${L} — ${t}`);
    const inp=el('input',{type:'number',min:'1',max:'5',
      id:`r_${i}_${L}_${k}`});
    row.appendChild(lab);row.appendChild(inp);r.appendChild(row);}
   const rr=el('div',{'class':'score'});
   rr.appendChild(el('label',{},`${L} — Readability rank (1-4, 1 = best)`));
   rr.appendChild(el('input',{type:'number',min:'1',max:'4',
     id:`r_${i}_${L}_rank`}));
   r.appendChild(rr);box.appendChild(r);}
  root.appendChild(box);});}
function save(){const rater=document.getElementById('rater').value||'X';
 const out={rater:rater,ratings:[]};
 DATA.forEach((inst,i)=>{const rec={window_id:inst.window_id,
  vehicle:inst.vehicle,systems:{}};
  for(const L of ['A','B','C','D']){const o={};
   for(const [k] of DIMS){const e=document.getElementById(`r_${i}_${L}_${k}`);
    o[k]=e?Number(e.value):null;}
   const rk=document.getElementById(`r_${i}_${L}_rank`);
   o.readability_rank=rk?Number(rk.value):null;
   rec.systems[inst.letter_key[L]]=o;}
  out.ratings.push(rec);});
 const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob(
  [JSON.stringify(out,null,1)],{type:'application/json'}));
 a.download=`human_validation_metropt_${rater}.json`;a.click();}
build();
</script></body></html>""".replace("__DATA__", data).replace(
        "__DIMS__", json.dumps(dims))


if __name__ == "__main__":
    main()
