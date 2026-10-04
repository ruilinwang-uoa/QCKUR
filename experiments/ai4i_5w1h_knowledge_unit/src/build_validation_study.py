"""Build the human-validation study instrument (AI4I, three-layer anchoring).

Deliverables (in runs/human_validation/):
  * rating_form.html  -- self-contained blinded rating form (DOM-built, no
    innerHTML). Raters see, per instance, the sensor context + VERIFIED FACTS
    and four anonymized reports (systems B1/B2/B5/B8 shuffled to letters A-D),
    score each on the Layer-3 rubric dimensions (faithfulness / completeness /
    coherence / actionability, 1-5) + 5W1H checkboxes, and rank the four by
    readability. Progress autosaves; an Export button downloads one JSON.
  * sample_manifest.json -- instance selection (30: 11 faulted + 19 healthy,
    the degradation-study anchor subsample) + the letter->system key
    (NOT for the raters).

Companion analysis: analyze_human_validation.py (written alongside).
Rater profile: any literate rater can judge readability/usefulness; the facts
block anchors faithfulness. ~30 instances x 4 reports, about 1-1.5 h.
"""
from __future__ import annotations

import json
import random

import pandas as pd

import config as C
import background as bg

SYSTEMS = ["B1", "B2", "B5", "B8"]
SRC = {"B1": "reports_main_glm_n100_B1B2B8_v3",
       "B2": "reports_main_glm_n100_B1B2B8_v3",
       "B5": "reports_main_glm_n100",
       "B8": "reports_main_glm_n100_B1B2B8_v3"}
N_FAULT, N_HEALTH = 11, 19
SEED = 42
OUT = C.RUNS_DIR / "human_validation"

HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Blinded report-quality study</title>
<style>
 body{font-family:Georgia,serif;max-width:900px;margin:24px auto;padding:0 12px;background:#fafafa;color:#222}
 h1{font-size:1.4em} h2{font-size:1.1em;margin-top:2em;border-bottom:2px solid #4477aa;padding-bottom:4px}
 .facts{background:#eef3f9;border:1px solid #b9cde4;padding:8px 12px;white-space:pre-wrap;font-family:monospace;font-size:.85em}
 .report{border:1px solid #ccc;background:#fff;padding:10px 14px;margin:14px 0;white-space:pre-wrap}
 .report b{color:#4477aa}
 .score{display:inline-block;margin:4px 18px 4px 0}
 input[type=range]{width:150px;vertical-align:middle}
 .cov label{margin-right:10px;font-size:.9em}
 .rank{margin:8px 0 24px 0}
 button{font-size:1.05em;padding:6px 16px;margin:4px}
 #saveNote{color:#666;font-size:.85em}
</style></head><body>
<h1>Blinded report-quality study</h1>
<p>You will see __N__ machine samples. For each: read the sensor context and the
<b>VERIFIED FACTS</b> (ground truth computed from the data), then evaluate four
anonymized reports (<b>A&ndash;D</b>). For each report, rate 1 (very poor) to 5 (excellent):
<i>faithfulness</i> (consistent with the verified facts; penalize invented or wrong
values/modes), <i>completeness</i>, <i>coherence</i> (organized, fluent, internally
consistent), <i>actionability</i> (concrete, correct maintenance action). Tick which of
the six 5W1H aspects it addresses meaningfully. Finally, <b>rank the four reports by
readability</b> (1 = most readable). You may pause at any time; progress is saved in
this browser. When finished, click Export and send the downloaded file back.</p>
<button id="exp1">Export my ratings (JSON)</button><span id="saveNote"></span>
<div id="content"></div>
<button id="exp2">Export my ratings (JSON)</button>
<script type="application/json" id="studydata">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("studydata").textContent);
let state = JSON.parse(localStorage.getItem("hv_state") || "{}");
function save(){
  localStorage.setItem("hv_state", JSON.stringify(state));
  document.getElementById("saveNote").textContent =
    " saved " + new Date().toLocaleTimeString();
}
function el(tag, cls, text){
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function render(){
  const root = document.getElementById("content");
  DATA.forEach(function(d, i){
    const s = state[i] || (state[i] = {rank:{}, reports:{}});
    const sec = el("h2", null, "Sample " + (i+1) + " of " + DATA.length + " \\u2014 " + d.id);
    const facts = el("div", "facts",
      "SENSOR CONTEXT + VERIFIED FACTS:\\n" + d.facts);
    root.appendChild(sec); root.appendChild(facts);
    Object.keys(d.reports).sort().forEach(function(L){
      const r = s.reports[L] || (s.reports[L] = {faith:3,compl:3,coh:3,act:3,cov:{}});
      const card = el("div", "report");
      const head = el("b", null, "Report " + L);
      const body = el("div", null, d.reports[L]);
      card.appendChild(head); card.appendChild(body);
      const row = el("div");
      [["faith","faithfulness"],["compl","completeness"],
       ["coh","coherence"],["act","actionability"]].forEach(function(p){
        const span = el("span", "score");
        span.appendChild(document.createTextNode(p[1] + " "));
        const inp = document.createElement("input");
        inp.type = "range"; inp.min = 1; inp.max = 5; inp.value = r[p[0]];
        const val = el("b", null, String(r[p[0]]));
        inp.addEventListener("input", function(){
          r[p[0]] = +inp.value; val.textContent = inp.value; save(); });
        span.appendChild(inp); span.appendChild(val);
        row.appendChild(span);
      });
      const cov = el("div", "cov", "5W1H addressed: ");
      ["what","why","when","where","who","how"].forEach(function(a){
        const lab = el("label", null, a);
        const box = document.createElement("input");
        box.type = "checkbox"; box.checked = !!r.cov[a];
        box.addEventListener("change", function(){ r.cov[a] = box.checked; save(); });
        lab.insertBefore(box, lab.firstChild);
        cov.appendChild(lab);
      });
      card.appendChild(row); card.appendChild(cov);
      root.appendChild(card);
    });
    const rank = el("div", "rank", "Readability ranking (1 = most readable; each rank once):  ");
    ["A","B","C","D"].forEach(function(L){
      rank.appendChild(document.createTextNode("Report " + L + ": "));
      const sel = document.createElement("select");
      const empty = document.createElement("option");
      empty.textContent = "-"; empty.value = "";
      sel.appendChild(empty);
      [1,2,3,4].forEach(function(n){
        const o = document.createElement("option");
        o.textContent = String(n); o.value = String(n);
        if (s.rank[L] === n) o.selected = true;
        sel.appendChild(o);
      });
      sel.addEventListener("change", function(){ s.rank[L] = +sel.value; save(); });
      rank.appendChild(sel); rank.appendChild(document.createTextNode("  "));
    });
    root.appendChild(rank);
  });
}
function exportJSON(){
  const blob = new Blob([JSON.stringify(state, null, 1)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "human_validation_ratings.json";
  a.click();
}
document.getElementById("exp1").addEventListener("click", exportJSON);
document.getElementById("exp2").addEventListener("click", exportJSON);
render();
</script></body></html>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample = pd.read_parquet(C.DATA_DIR / "sample_315.parquet")
    ctx = bg.build_ctx()
    ids = (list(sample[sample["Machine failure"] == 1]["sample_id"][:N_FAULT])
           + list(sample[sample["Machine failure"] == 0]["sample_id"][:N_HEALTH]))

    rng = random.Random(SEED)
    data, manifest = [], {"seed": SEED, "systems": SYSTEMS, "key": {}, "instances": []}
    reps = {s: pd.read_parquet(C.RUNS_DIR / f"{SRC[s]}.parquet") for s in SYSTEMS}
    for sid in ids:
        row = sample[sample["sample_id"] == sid].iloc[0].to_dict()
        facts = "\n".join(f"{k['category']}: {k['answer']}"
                          for k in bg.canonical_5w1h(row, ctx, use_rules=False))
        letters = ["A", "B", "C", "D"]
        rng.shuffle(letters)
        reports, key_here = {}, {}
        for system, letter in zip(SYSTEMS, letters):
            rep = reps[system][(reps[system]["method"] == system)
                               & (reps[system]["instance_id"] == sid)]["raw_output"].iloc[0]
            reports[letter] = rep
            key_here[letter] = system
        data.append({"id": str(sid), "facts": facts, "reports": reports})
        manifest["key"][str(sid)] = key_here
        manifest["instances"].append(str(sid))

    # "</" inside JSON must be escaped so the inline <script type="application/json">
    # block cannot be broken out of by report text
    payload = json.dumps(data).replace("<", "\\u003c")
    (OUT / "rating_form.html").write_text(
        HTML.replace("__N__", str(len(data))).replace("__DATA__", payload),
        encoding="utf-8")
    (OUT / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"built: {OUT/'rating_form.html'} ({len(data)} instances x {len(SYSTEMS)} reports)")
    print(f"key (keep from raters): {OUT/'sample_manifest.json'}")


if __name__ == "__main__":
    main()
