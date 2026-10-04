"""
engineer_adjudication_package.py -- build the blinded ground-truth
adjudication instrument for a practicing maintenance engineer.

Blocks:
  A. nine documented events -> mode / disposition / reasoning
     + two open contested items (2022 component attribution; 2020
       maintenance-row assignment), presented WITHOUT our adjudications
  B. three calibration-pool exclusions -> genuine anomaly vs variation
  C. 37 pre-failure windows -> actionable / precursor / nothing

Blinding: the engineer sees only raw telemetry digests (frozen feature
extraction), public operator-record excerpts, and descriptive baseline
bands -- never our adjudications, system outputs, or the paper.

Output: adjudication/instrument/adjudication_form.html
(self-contained; DOM-built, no innerHTML; JSON download per rater).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metropt_features import ALL_COLS, window_features

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
PKG = ROOT / "adjudication"
DATA = ROOT / "data" / "metropt3"

RAW = {"metropt3": DATA / "MetroPT3(AirCompressor).csv",
       "metropt2022": DATA / "dataset_train_2022.csv",
       "metropt2022B": DATA / "MetroPT2.csv"}
MANI = {"metropt3": RUNS / "window_manifest.json",
        "metropt2022": RUNS / "window_manifest_metropt2022.json",
        "metropt2022B": RUNS / "window_manifest_metropt2022B.json"}

# public operator-record event spans (dates/times only; modes withheld)
EVENTS = [
    {"id": "E1", "veh": "metropt3", "epoch": "2020 unit (10 s cadence)",
     "span": "2020-04-18 00:00 to 2020-04-18 23:59",
     "log": "Operator maintenance log lists a repair cycle around end of April 2020 (see item R)."},
    {"id": "E2", "veh": "metropt3", "epoch": "2020 unit (10 s cadence)",
     "span": "2020-05-29 23:30 to 2020-05-30 06:00",
     "log": "Operator log lists an event overnight 29-30 May 2020."},
    {"id": "E3", "veh": "metropt3", "epoch": "2020 unit (10 s cadence)",
     "span": "2020-06-05 10:00 to 2020-06-07 14:30",
     "log": "Operator log lists a repair in early June 2020."},
    {"id": "E4", "veh": "metropt3", "epoch": "2020 unit (10 s cadence)",
     "span": "2020-07-15 14:30 to 2020-07-15 19:00",
     "log": "Operator log lists a repair on 15 July 2020."},
    {"id": "A1", "veh": "metropt2022", "epoch": "train A (2022, 1 Hz)",
     "span": "2022-02-28 21:53 to 2022-03-01 02:00",
     "log": "Operator log: failure detected 28 Feb 2022 evening; unit removed."},
    {"id": "A2", "veh": "metropt2022", "epoch": "train A (2022, 1 Hz)",
     "span": "2022-03-23 14:54 to 2022-03-23 15:24",
     "log": "Operator log: brief event 23 Mar 2022 afternoon; unit continued."},
    {"id": "A3", "veh": "metropt2022", "epoch": "train A (2022, 1 Hz)",
     "span": "2022-05-30 12:00 to 2022-06-02 06:18",
     "log": "Operator log: severe failure 30 May 2022; unit removed."},
    {"id": "B1", "veh": "metropt2022B", "epoch": "train B (2022, 1 Hz)",
     "span": "2022-06-04 10:19 to 2022-06-04 14:22",
     "log": "Operator log: failure 4 Jun 2022; unit removed."},
    {"id": "B2", "veh": "metropt2022B", "epoch": "train B (2022, 1 Hz)",
     "span": "2022-07-11 10:10 to 2022-07-14 10:22",
     "log": "Operator log: failure 11 Jul 2022; unit removed 14 Jul."},
]

FEATS = [
    ("tp2_mean", "TP2 compressor pressure, mean (bar)", "{:.2f}"),
    ("tp2_p95", "TP2 p95 (bar)", "{:.2f}"),
    ("tp2_frac_ge9_5", "TP2 fraction >= 9.5 bar (%)", "{:.1%}"),
    ("h1_mean", "H1 reservoir-loop, mean (bar)", "{:.2f}"),
    ("dv_frac_load", "DV_pressure > 0.5 fraction (%)", "{:.1%}"),
    ("mc_mean", "Motor current, mean (A)", "{:.2f}"),
    ("mc_on_frac", "Working-duty fraction (>4.5 A) (%)", "{:.1%}"),
    ("mc_on_maxrun_min", "Longest continuous duty run (min)", "{:.0f}"),
    ("lps_frac", "LPS low-pressure warning, active fraction (%)", "{:.1%}"),
    ("lps_maxrun_min", "LPS longest continuous activation (min)", "{:.0f}"),
    ("oil_mean", "Oil temperature, mean (degC)", "{:.1f}"),
    ("oil_p95", "Oil p95 (degC)", "{:.1f}"),
    ("oil_max", "Oil max (degC)", "{:.1f}"),
    ("oil_trend_c", "Oil warming over window (degC)", "{:+.1f}"),
    ("comp_frac", "COMP active fraction (%)", "{:.0%}"),
    ("towers_frac", "Towers active fraction (%)", "{:.0%}"),
]
BAND_FEATS = ["tp2_p95", "mc_on_frac", "lps_maxrun_min", "oil_max",
              "oil_trend_c", "dv_frac_load"]


def load_gt() -> dict:
    return {f"{g['vehicle']}::{g['window_id']}": g for g in json.load(
        open(RUNS / "metropt_gt.json", encoding="utf-8"))["windows"]}


def load_spec() -> dict:
    return json.load(open(ROOT / "knowledge" / "metropt" / "metropt_spec.json",
                          encoding="utf-8"))


def digest_lines(feat: dict, band: dict | None) -> list:
    lines = []
    for key, label, fmt in FEATS:
        v = feat.get(key)
        if v is None or v != v:
            continue
        line = f"{label}: {fmt.format(v)}"
        if band and key in BAND_FEATS and band.get(key):
            b = band[key]
            line += (f"   [healthy band: p01 {b['p01']:.2f}, "
                     f"median {b['median']:.2f}, p99 {b['p99']:.2f}]")
        lines.append(line)
    cov = feat.get("coverage")
    if cov is not None:
        lines.append(f"Data coverage of the 24 h window: {cov:.0%}")
    return lines


def event_window_feature(gt, veh, event_span_end):
    """feature dict of the event's surviving positive window closest to
    failure end (prefer offset 6)."""
    end = pd.Timestamp(event_span_end)
    best, best_gap = None, None
    for key, g in gt.items():
        if not key.startswith(veh + "::"):
            continue
        te = pd.Timestamp(g["t_end"])
        gap = abs((te - end).total_seconds())
        # windows end BEFORE the failure end; closest = min gap
        if best_gap is None or gap < best_gap:
            best, best_gap = g, gap
    return best["numeric_truth"], best["window_id"]


_raw_cache: dict[str, pd.DataFrame] = {}


def raw(veh: str) -> pd.DataFrame:
    if veh not in _raw_cache:
        header = pd.read_csv(RAW[veh], nrows=0).columns.tolist()
        use = [c for c in ALL_COLS if c in header]
        df = pd.read_csv(RAW[veh], usecols=use)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed",
                                         dayfirst=True)
        _raw_cache[veh] = df.set_index("timestamp").sort_index()
    return _raw_cache[veh]


def ghost_features(veh: str, t_end: str, expected: int) -> dict:
    t1 = pd.Timestamp(t_end)
    t0 = t1 - pd.Timedelta(hours=24)
    w = raw(veh).loc[t0:t1]
    return window_features(w, expected)


def main() -> None:
    gt = load_gt()
    spec = load_spec()

    # ---- Block A
    blockA = []
    for ev in EVENTS:
        feat, wid = event_window_feature(gt, ev["veh"], ev["span"].split(" to ")[1])
        blockA.append({
            "id": ev["id"], "epoch": ev["epoch"], "span": ev["span"],
            "log": ev["log"], "window": wid,
            "digest": digest_lines(feat, None)})
    contested = {
        "attr": {
            "title": "Item X1 - component attribution (open)",
            "text": ("Two failure events occurred on train A in 2022, three "
                     "weeks apart (E-A1 and E-A2 above; telemetry digests "
                     "repeated below). Published documentation disagrees "
                     "about which component failed in each. Judging from "
                     "the telemetry, attribute each event to a component "
                     "(compressor / dryer / client pipes and lines / valves "
                     "/ reservoir / oil circuit / other) and state your "
                     "reasoning."),
            "events": ["A1", "A2"]},
        "row": {
            "title": "Item X2 - maintenance-record row assignment (open)",
            "text": ("The 2020 operator maintenance table contains a "
                     "formatting artefact (a duplicated row number), so one "
                     "maintenance timestamp ('30 Apr 12:00') could belong to "
                     "either the April event (E1) or the May event (E2). "
                     "Telemetry facts: 30 Apr shows a two-hour data gap "
                     "starting exactly 12:00, an outage 15:00-23:00, and LPS "
                     "activation on the 23:00 return; 30 May is a fully "
                     "recorded, ordinary day. Which event does the 30 Apr "
                     "12:00 maintenance stamp belong to?"),
            "options": ["April event (E1)", "May event (E2)",
                        "Cannot determine"]}}

    # ---- Block B (ghost windows)
    blockB = []
    for veh in ("metropt2022", "metropt3"):
        v = spec["per_vehicle"][veh]
        exp = json.load(open(MANI[veh], encoding="utf-8"))["params"]
        expected = int(exp["expected_rows_per_window"])
        for t_end in v["ooc_ledger"]["excluded_window_ends"]:
            f = ghost_features(veh, t_end, expected)
            band = v.get("baseline_stats", {})
            blockB.append({
                "id": f"X-{veh}-{t_end[:10]}", "epoch": veh,
                "t_end": t_end,
                "digest": digest_lines(f, band),
                "note": ("This 24 h window was sampled from the vehicle's "
                         "healthy pool and EXCLUDED by the study team from "
                         "the nominal calibration baseline. Independently "
                         "judge: genuine undocumented operational episode, "
                         "or ordinary variation?")})

    # ---- Block C (37 positives, chronological per vehicle)
    blockC = []
    pos = pd.read_csv(RUNS / "metropt_positive_strata.csv",
                      parse_dates=["t_end"])
    for veh in ("metropt3", "metropt2022", "metropt2022B"):
        band = spec["per_vehicle"][veh].get("baseline_stats", {})
        for _, r in pos[pos.vehicle == veh].sort_values("t_end").iterrows():
            g = gt[f"{veh}::{r.window_id}"]
            blockC.append({
                "id": f"C-{veh[-4:]}-{r.window_id}", "epoch": veh,
                "t_end": str(r.t_end),
                "digest": digest_lines(g["numeric_truth"], band)})

    data = {"blockA": blockA, "contested": contested,
            "blockB": blockB, "blockC": blockC}
    out = PKG / "instrument"
    out.mkdir(parents=True, exist_ok=True)
    (out / "adjudication_data.json").write_text(
        json.dumps(data, indent=1), encoding="utf-8")
    print(f"data -> {out / 'adjudication_data.json'} "
          f"(A={len(blockA)}, B={len(blockB)}, C={len(blockC)})")

    html = FORM_HTML.replace("__DATA__", json.dumps(data))
    (out / "adjudication_form.html").write_text(html, encoding="utf-8")
    print(f"form -> {out / 'adjudication_form.html'}")


FORM_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Independent ground-truth adjudication</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;
      padding:16px;background:#f7f8fa;color:#1F2933}
 h1{font-size:20px} h2{font-size:16px;margin-top:28px;
   border-bottom:2px solid #2E86AB;padding-bottom:4px}
 .card{background:#fff;border:1px solid #d8dce0;border-radius:6px;
   padding:12px;margin:10px 0}
 .ctx{background:#FDF3DC;border:1px solid #EAB23A;border-radius:4px;
   padding:8px;font-size:13px;margin-bottom:8px;white-space:pre-wrap}
 .dig{background:#f2f5f7;border-radius:4px;padding:8px;font-size:12px;
   font-family:ui-monospace,monospace;white-space:pre-wrap;margin-bottom:8px}
 label{display:block;font-size:13px;margin:6px 0 2px;font-weight:600}
 select,textarea{width:100%;padding:6px;font-size:13px;
   border:1px solid #b9c0c6;border-radius:4px;box-sizing:border-box}
 textarea{min-height:48px}
 .q{font-size:13px;margin-top:8px}
 button{padding:10px 18px;font-size:14px;border:0;border-radius:6px;
   background:#2E86AB;color:#fff;cursor:pointer;margin-top:16px}
 .hint{color:#5b6670;font-size:12px}
 #code{padding:6px;font-size:14px}
 .progress{position:sticky;top:0;background:#f7f8fa;padding:6px 0;
   font-size:13px;font-weight:600}
</style></head><body>
<h1>Independent ground-truth adjudication</h1>
<p class="hint">Metro-train air production units (APU). You are judging,
independently and blinded, the ground-truth labels used by a research
study. Work through the three blocks; answers autosave in this browser.
When finished, enter your rater code and download the JSON file.</p>
<p><label>Rater code: <input id="code" placeholder="e.g. ENG1"></label></p>
<div class="progress" id="prog"></div>
<div id="root"></div>
<button onclick="dl()">Download adjudication JSON</button>
<script>
const DATA = __DATA__;
const KEY = 'adj_v1';
let S = {};
try { S = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch(e) {}
function el(t, cls, txt){ const n = document.createElement(t);
  if (cls) n.className = cls; if (txt != null) n.textContent = txt;
  return n; }
function sel(id, opts, q){ const lab = el('label', null, q);
  const s = el('select'); s.dataset.k = id;
  s.appendChild(el('option', null, '-- select --')).value = '';
  opts.forEach(o => { const op = el('option', null, o); op.value = o;
    s.appendChild(op); });
  if (S[id]) s.value = S[id];
  s.onchange = () => { S[id] = s.value; save(); };
  const wrap = el('div', 'q'); wrap.appendChild(lab);
  wrap.appendChild(s); return wrap; }
function txt(id, q, ph){ const lab = el('label', null, q);
  const t = el('textarea'); t.dataset.k = id; t.placeholder = ph || '';
  if (S[id]) t.value = S[id];
  t.oninput = () => { S[id] = t.value; save(); };
  const wrap = el('div', 'q'); wrap.appendChild(lab);
  wrap.appendChild(t); return wrap; }
function save(){ localStorage.setItem(KEY, JSON.stringify(S)); prog(); }
function prog(){ const ks = Object.keys(S).filter(k => S[k]);
  document.getElementById('prog').textContent =
    'Answered fields: ' + ks.length; }
function digest(d){ const box = el('div', 'dig');
  d.forEach(l => box.appendChild(el('div', null, l))); return box; }

const root = document.getElementById('root');
// ---------------- Block A
root.appendChild(el('h2', null,
  'Block A - nine documented failure events'));
root.appendChild(el('p', 'hint',
  'For each event: judge the failure MODE and the DISPOSITION from the ' +
  'telemetry digest and the log excerpt. Two open items follow the list.'));
const byId = {};
DATA.blockA.forEach(ev => { byId[ev.id] = ev; });
DATA.blockA.forEach(ev => {
  const c = el('div', 'card');
  c.appendChild(el('div', 'ctx',
    'Event ' + ev.id + ' | ' + ev.epoch + '\nSpan: ' + ev.span +
    '\n' + ev.log + '\nTelemetry: 24 h window ending at the failure end.'));
  c.appendChild(digest(ev.digest));
  c.appendChild(sel('A_' + ev.id + '_mode',
    ['air leak', 'oil leak', 'both', 'other / cannot determine'],
    'Failure mode:'));
  c.appendChild(sel('A_' + ev.id + '_disp',
    ['removed from service', 'continued in service', 'self-resolved',
     'repaired at scheduled maintenance', 'cannot determine'],
    'Disposition:'));
  c.appendChild(txt('A_' + ev.id + '_why', 'Reasoning (brief):'));
  root.appendChild(c);
});
// contested X1
const cx = DATA.contested.attr;
const c1 = el('div', 'card');
c1.appendChild(el('div', 'ctx', cx.title + '\n\n' + cx.text));
cx.events.forEach(id => {
  const ev = byId[id];
  c1.appendChild(el('label', null, 'Event ' + id + ' telemetry:'));
  c1.appendChild(digest(ev.digest));
  c1.appendChild(sel('X1_' + id + '_comp',
    ['compressor', 'dryer', 'client pipes and lines', 'valves',
     'reservoir', 'oil circuit', 'other', 'cannot determine'],
    'Attributed component for ' + id + ':'));
});
c1.appendChild(txt('X1_why', 'Reasoning:'));
root.appendChild(c1);
// contested X2
const rx = DATA.contested.row;
const c2 = el('div', 'card');
c2.appendChild(el('div', 'ctx', rx.title + '\n\n' + rx.text));
c2.appendChild(sel('X2_assign', rx.options, 'Assignment:'));
c2.appendChild(txt('X2_why', 'Reasoning (brief):'));
root.appendChild(c2);
// ---------------- Block B
root.appendChild(el('h2', null,
  'Block B - three windows excluded from the healthy baseline'));
DATA.blockB.forEach(w => {
  const c = el('div', 'card');
  c.appendChild(el('div', 'ctx',
    w.id + ' | window ending ' + w.t_end + '\n' + w.note));
  c.appendChild(digest(w.digest));
  c.appendChild(sel('B_' + w.id + '_judg',
    ['genuine undocumented operational episode', 'ordinary variation',
     'unsure'], 'Your judgement:'));
  c.appendChild(sel('B_' + w.id + '_conf',
    ['1', '2', '3', '4', '5'], 'Confidence (1 = low, 5 = high):'));
  c.appendChild(txt('B_' + w.id + '_why', 'Reasoning (brief):'));
  root.appendChild(c);
});
// ---------------- Block C
root.appendChild(el('h2', null,
  'Block C - 37 pre-failure windows'));
root.appendChild(el('p', 'hint',
  'Each card is the 24 h window ENDING before a documented failure at ' +
  'the stated time. Judge what the telemetry supports: an actionable ' +
  'fault indication (which?), only a near-threshold precursor, or ' +
  'nothing actionable. Healthy bands in brackets are descriptive ' +
  'statistics of this vehicle\'s nominal pool.'));
DATA.blockC.forEach(w => {
  const c = el('div', 'card');
  c.appendChild(el('div', 'ctx',
    w.id + ' | window ending ' + w.t_end));
  c.appendChild(digest(w.digest));
  c.appendChild(sel('C_' + w.id + '_strat',
    ['actionable fault indication', 'near-threshold precursor only',
     'nothing actionable'], 'Your reading:'));
  c.appendChild(txt('C_' + w.id + '_note',
    'If actionable/precursor: which indication? (brief)', ''));
  root.appendChild(c);
});
prog();
function dl(){
  const out = { rater: document.getElementById('code').value,
                answers: S, ts: new Date().toISOString() };
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob(
    [JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = 'adjudication_' +
    (document.getElementById('code').value || 'XXX') + '.json';
  a.click();
}
</script></body></html>
"""

if __name__ == "__main__":
    main()
