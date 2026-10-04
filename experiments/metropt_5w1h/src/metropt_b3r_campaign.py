"""
metropt_b3r_campaign.py -- rules-in-prompt baseline. B3R is the directly prompted LLM (B3) given, in
addition to the public background and the frozen digest, the operator's rule set WITH this vehicle's
calibrated thresholds and baselines in text form -- exactly the information B1/B2/B8's handlers compute on,
minus the architectural organization. Isolates knowledge access from knowledge embodiment.

Single difference vs gen_b3: a RULES block rendered from the frozen
per-vehicle spec block. Same backbone, seed, budget, and task.

Usage: python metropt_b3r_campaign.py [--smoke 2]
Resume-safe on (vehicle, window_id).
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import llm_call
from metropt_handlers import load_spec
from metropt_l1 import _feat_row
from metropt_render import METROPT_BACKGROUND, REPORT_TASK, digest_text

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
PROVIDER = "openai"          # GLM-4-Flash, same as the campaign
SEED = 42
MAXTOK = 700

TH_NAMES = {
    "A1_mc_on_frac": ("working-duty fraction", lambda x: f">= {x*100:.1f}%"),
    "A1_tp2_frac_ge9_5": ("TP2 fraction >= 9.5 bar",
                          lambda x: f"<= {x*100:.1f}%"),
    "A1b_mc_on_maxrun_min": ("one continuous duty episode (min)",
                             lambda x: f">= {x:.0f}"),
    "A2_lps_maxrun_min": ("LPS continuously active (min)",
                          lambda x: f">= {x:.0f}"),
    "A3_tp2_p95_max": ("TP2 p95 (bar)", lambda x: f"<= {x:.2f}"),
    "A3_mc_on_frac_min": ("duty fraction", lambda x: f">= {x*100:.0f}%"),
    "D1_oil_elev_c": ("oil max over this vehicle's baseline p99 (degC)",
                      lambda x: f">= +{x:.1f}"),
    "D2_oil_trend_c": ("window oil warming (degC)", lambda x: f">= +{x:.1f}"),
}


def rules_text(vehicle: str, spec: dict) -> str:
    v = spec["per_vehicle"][vehicle]
    th = v["thresholds"]
    lines = [
        "OPERATOR RULE SET WITH THIS VEHICLE'S CALIBRATED THRESHOLDS AND "
        "BASELINES (apply these to judge the telemetry):",
        "- Air-leak signature A1 (duty without pressure buildup): "
        f"{TH_NAMES['A1_mc_on_frac'][0]} {TH_NAMES['A1_mc_on_frac'][1](th['A1_mc_on_frac'])} "
        f"while {TH_NAMES['A1_tp2_frac_ge9_5'][0]} "
        f"{TH_NAMES['A1_tp2_frac_ge9_5'][1](th['A1_tp2_frac_ge9_5'])} "
        f"(no-pressure-buildup guard).",
        f"- A1b (episode form): {TH_NAMES['A1b_mc_on_maxrun_min'][0]} "
        f"{TH_NAMES['A1b_mc_on_maxrun_min'][1](th['A1b_mc_on_maxrun_min'])} "
        "while TP2 shows no buildup.",
        f"- A2 sustained low-pressure warning: "
        f"{TH_NAMES['A2_lps_maxrun_min'][0]} "
        f"{TH_NAMES['A2_lps_maxrun_min'][1](th['A2_lps_maxrun_min'])}.",
        f"- A3 deep depression: TP2 p95 <= {th['A3_tp2_p95_max']:.2f} bar "
        f"with duty fraction >= {th['A3_mc_on_frac_min']*100:.0f}%.",
        "- Oil degradation D1: oil max elevation over this vehicle's "
        f"baseline p99 {v['oil_max_p99']:.1f} degC "
        f">= +{th['D1_oil_elev_c']:.1f} degC.",
        f"- Thermal anomaly D2: {TH_NAMES['D2_oil_trend_c'][0]} "
        f"{TH_NAMES['D2_oil_trend_c'][1](th['D2_oil_trend_c'])} "
        "(overload heating; not by itself oil degradation).",
        "- Verdict priority: any air signature => air leak indication; "
        "D1 => oil degradation indication; D2 alone => thermal anomaly; "
        "no signature => healthy (or monitor if a deviation approaches a "
        "threshold). Judge THIS window against these rules and state the "
        "verdict explicitly in your report.",
    ]
    return "\n".join(lines)


def gen_b3r(feat: dict, vehicle: str, spec: dict) -> dict:
    prompt = (METROPT_BACKGROUND + "\n\n" + rules_text(vehicle, spec) +
              "\n\n" + digest_text(feat, vehicle) + "\n\nTASK:\n" + REPORT_TASK)
    res = llm_call.chat(PROVIDER, [
        {"role": "system",
         "content": "You are a predictive-maintenance analyst."},
        {"role": "user", "content": prompt}], seed=SEED, max_tokens=MAXTOK)
    return {"system_id": "B3R", "text": res.get("text") or "",
            "claims": [], "ku_trace": [], "prompt": prompt,
            "ok": bool(res.get("ok"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    spec = load_spec()
    out = RUNS / "reports_metropt_B3R.parquet"
    done = set()
    if out.exists():
        old = pd.read_parquet(out)
        done = {(r.vehicle, r.window_id) for r in old.itertuples()
                if r.ok and (r.text or "").strip()}
    rep0 = pd.read_parquet(RUNS / "reports_metropt_B1.parquet")
    tasks = []
    for _, r in rep0.iterrows():
        if (r.vehicle, r.window_id) in done:
            continue
        if args.smoke:
            continue  # smoke handled below
        tasks.append((r.vehicle, r.window_id, r.kind, r.offset_h,
                      r.event_id))
    if args.smoke:
        # one positive + one control per vehicle
        for veh in rep0.vehicle.unique():
            sub = rep0[rep0.vehicle == veh]
            for kind in ["positive", "control"]:
                row = sub[sub.kind == kind].iloc[0]
                if (row.vehicle, row.window_id) not in done:
                    tasks.append((row.vehicle, row.window_id, row.kind,
                                  row.offset_h, row.event_id))
    print(f"B3R: {len(tasks)} to generate ({len(done)} done)")
    lock, rows = threading.Lock(), []

    def _save(new: pd.DataFrame) -> None:
        full = new if not out.exists() else pd.concat(
            [pd.read_parquet(out), new], ignore_index=True)
        full = full.drop_duplicates(
            subset=["vehicle", "window_id"], keep="last")
        full.to_parquet(out, index=False)

    def work(t):
        veh, wid, kind, off, ev = t
        feat = _feat_row(veh, wid)
        g = gen_b3r(feat, veh, spec)
        return {"vehicle": veh, "window_id": wid, "kind": kind,
                "offset_h": off, "event_id": ev, **g}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        persisted = 0
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                t = futs[fut]
                row = {"vehicle": t[0], "window_id": t[1], "kind": t[2],
                       "offset_h": t[3], "event_id": t[4], "ok": False,
                       "error": str(e)[:200]}
            with lock:
                rows.append(row)
                if i % 10 == 0 or i == len(tasks):
                    new = pd.DataFrame(rows[persisted:])
                    _save(new)
                    persisted = len(rows)
                    print(f"  {i}/{len(tasks)}")
    new = pd.DataFrame(rows[persisted:])
    if len(new):
        _save(new)
    n_ok = pd.read_parquet(out)
    print(f"B3R complete -> {out.name} "
          f"({len(n_ok)} rows, {int(n_ok.ok.sum())} ok)")


if __name__ == "__main__":
    main()
