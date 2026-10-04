"""
metropt_l1.py -- repaired-from-day-one L1 factual checker for the
MetroPT campaign (all AI4I/bearing lessons baked in).

Checks per report:
  NUMERIC    every data-looking number in the text must match a digest
             truth, a unit conversion of one, an explicitly whitelisted
             DERIVED value (oil elevation, coverage, durations), or a
             NON-DATA exemption (ordinals, window/offset conventions,
             documented operator setpoints, word limits)
  CLASSIFICATION  the report's stated state vs the handler verdict for
             the window (the telemetry-implied state) -- claiming a
             fault against a nominal digest (or vice versa) is an error
  RULE       if a rule fired, its evidence class must be mentioned; if
             the verdict is abstain, no definite state may be asserted

Usage: python metropt_l1.py --system B3   (after a campaign run)
       python metropt_l1.py --all
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from metropt_handlers import handle, load_spec

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
SOURCES = [("metropt3", "features_metropt3.parquet"),
           ("metropt2022", "features_metropt2022.parquet"),
           ("metropt2022B", "features_metropt2022B.parquet")]

# documented operator setpoints / public constants (background-level, not
# data claims) -- reports may cite them freely. Includes the diagnostic
# cutoff reference (9.5 bar) and the handler confidence levels (system
# parameters, legitimately printable)
SETPOINTS = {0.0, 4.0, 4.5, 7.0, 8.2, 9.0, 9.5, 10.2, 10.5, 97.9, 80.2,
             0.35, 0.5, 0.85, 0.9, 1.0, 2.0, 3.0, 6.0, 6.8,
             2.8, 5.2}   # 4.5 = working-duty definition; last two:
                         # documented fleet contradiction/judge rates
# non-data numbers: ordinals, aspect counts, window/offset conventions,
# percent caps, word limits
EXEMPT_INTS = set(range(0, 13)) | {15, 24, 30, 48, 60, 72, 100, 150, 180,
                                   360, 1440, 2020, 2022}

_NUM = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?")   # dash after a digit = range, not sign
# strip statistic names (p95/p99/p50/p05) so their digits are not read as
# data values (AI4I lesson class: label artifacts before extraction)
_STATNAME = re.compile(r"[A-Za-z0-9_]*p(?:95|99|50|05)\b")
# echoed datetimes / clock times / percentile ordinals are context
_DATETIME = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?|\d{1,2}:\d{2}(?::\d{2})?")
_PCTORD = re.compile(r"\d{1,2}(?:st|nd|rd|th) percentile")
# legitimate system-parameter contexts: z-scores, confidence levels,
# standard-deviation phrasing, hour-of-window indices
_PARAMCTX = re.compile(r"z[_ =]|confidence|deviations?|hour ")


def _truth_set(feat: dict, spec: dict, vehicle: str) -> set[float]:
    """All legal numeric values for this window (truths + derived)."""
    v = spec["per_vehicle"][vehicle]
    out: set[float] = set()
    for key, x in feat.items():
        if isinstance(x, (int, float)) and x == x:
            out.update({round(float(x), 2), round(float(x), 1),
                        round(float(x), 0)})
            out.add(round(float(x) * 100, 1))   # fraction -> percent form
            out.add(round(float(x) * 100, 0))
    # derived: oil elevation and its percent-free forms (±0.05 variants
    # absorb formatter-rounding differences, e.g. 4.45 -> "4.5" vs 4.4)
    elev = round(float(feat.get("oil_max") or 0) - v["oil_max_p99"], 1)
    out.update({elev, round(abs(elev), 1),
                round(elev + 0.05, 1), round(elev - 0.05, 1)})
    out.add(round(v["oil_max_p99"], 1))
    # baseline statistics the templates legitimately cite (the per-
    # vehicle calibration is public within the report pipeline); only the
    # quantities actually printed: oil_max median/p99, tp2_p95 median,
    # mc_on_frac median -- plus the DERIVED DIFFERENCES a report may
    # legitimately compute ("X below the baseline"): a wrong difference
    # (bad arithmetic) still fails, which is the point
    bs = v.get("baseline_stats", {})
    printed_pairs = []
    for key in ("oil_max", "tp2_p95", "mc_on_frac"):
        st = bs.get(key)
        if not st:
            continue
        for stat in ("median", "p99"):
            b = round(st[stat], 2)
            out.update({b, round(st[stat], 1)})
            out.update({round(st[stat] * 100, 1), round(st[stat] * 100, 0)})
            printed_pairs.append((key, b))
    for key, b in printed_pairs:
        if isinstance(feat.get(key), (int, float)) and feat[key] == feat[key]:
            for nd in (1, 2):
                d = round(abs(float(feat[key]) - b), nd)
                out.update({d, round(d, 1)})
                out.update({round(d * 10, 0) / 10, 1})
    # trend may be phrased by magnitude ("a warming of X degC")
    tv = feat.get("oil_trend_c")
    if tv is not None and tv == tv:
        out.update({round(abs(tv), 1), round(abs(tv), 2)})
    # confidence levels may appear as percents (90% confidence)
    out.update({round(s * 100, 0) for s in (0.35, 0.5, 0.85, 0.9, 1.0)})
    out.update(SETPOINTS)
    return out


def _legal(truth: set[float], val: float) -> bool:
    for t in truth:
        if abs(val - t) <= max(0.06, abs(t) * 0.006):   # rounding tolerance
            return True
    return False


def numeric_check(text: str, feat: dict, spec: dict, vehicle: str):
    truth = _truth_set(feat, spec, vehicle)
    bad, n_seen = [], 0
    text = _STATNAME.sub("pQ", text)          # p95/p99 -> non-numeric
    text = _DATETIME.sub("DT", text)          # echoed timestamps -> neutral
    text = _PCTORD.sub("PCT", text)           # "95th percentile" -> neutral
    for m in _NUM.finditer(text):
        raw = m.group(0)
        pre = text[max(0, m.start() - 12):m.start()]
        if _PARAMCTX.search(pre):
            continue                           # z=..., confidence ...
        val = float(raw)
        n_seen += 1
        if "." not in raw and float(val).is_integer():
            iv = int(val)
            if iv in EXEMPT_INTS:
                continue
            if _legal(truth, float(iv)):
                continue
        if _legal(truth, val):
            continue
        ctx = text[max(0, m.start() - 40):m.end() + 25].replace("\n", " ")
        bad.append({"value": raw, "context": ctx.strip()})
    return {"n_extracted": n_seen, "n_bad": len(bad), "bad": bad[:8]}


# ---- classification ------------------------------------------------------
# NOTE order: monitor must be tested BEFORE healthy (reports legitimately
# open with "nominal operation with deviations that warrant monitoring")
_CLAIM_PATTERNS = [
    ("air_leak_indication", r"air[- ]leak|leak (?:indication|signature)|"
                             r"acute leak"),
    ("oil_degradation_indication", r"oil[- ](?:system )?degradation|"
                                    r"oil leak"),
    ("oil_thermal_anomaly", r"thermal anomaly|overload warming"),
    ("monitor_deviation", r"warrant(?:s|ing)? monitor|worth monitor|"
                          r"deviation worth|monitor closely|keep "
                          r"monitor|require[s]? monitoring"),
    ("abstain", r"not assessable|cannot assess|abstain|no assessment"),
    ("healthy", r"nominal|no (?:actionable |fault )?indication|normal "
                r"operation|no anomaly"),
]
_VERDICT_TO_EXPECTED = {
    "air_leak_indication": "air_leak_indication",
    "air_and_oil_indication": "air_leak_indication",
    "oil_degradation_indication": "oil_degradation_indication",
    "oil_thermal_anomaly": "oil_thermal_anomaly",
    "monitor_deviation": "monitor_deviation",
    "healthy": "healthy",
    "ABSTAIN_telemetry_hole": "abstain",
    "ABSTAIN_out_of_bank": "abstain",
}


def classification_check(text: str, verdict: dict):
    # state claims live in the opening region; differential sentences
    # later in the report legitimately mention competing mechanisms
    # ("not necessarily an oil leak") and must not be read as claims
    t = text[:400].lower()
    hits = [(cls, m.start()) for cls, pat in _CLAIM_PATTERNS
            for m in [re.search(pat, t)] if m]
    claimed = hits[0][0] if hits else "unknown"
    expected = _VERDICT_TO_EXPECTED[verdict["label"]]
    ok = claimed == expected or claimed == "unknown"
    if not ok:
        # a fault claim against nominal telemetry is the salient error
        ok = False
    return {"claimed": claimed, "expected": expected, "ok": ok,
            "ambiguous": claimed == "unknown"}


def rule_check(text: str, verdict: dict, feat: dict):
    label, tier = verdict["label"], verdict["tier"]
    issues = []
    t = text.lower()
    if label == "ABSTAIN_telemetry_hole":
        if re.search(r"shows (?:an? )?(?:fault|leak|degradation)|"
                     r"is (?:faulty|failing)", t):
            issues.append("definite state asserted despite abstention")
    if tier == "rule":
        fired = (verdict["evidence"].get("rule_branches_fired") or {})
        if (fired.get("A2_sustained_lps") and "lps" not in t
                and "low-pressure" not in t and "low pressure" not in t):
            issues.append("sustained-LPS trigger not mentioned")
        if (fired.get("A1a_duty_frac_no_buildup")
                or fired.get("A1b_duty_episode_no_buildup")):
            if not re.search(r"duty|working|current|compressor", t):
                issues.append("duty anomaly not mentioned")
        if (fired.get("D1_oil_elev") and not re.search(r"oil|temperature",
                                                       t)):
            issues.append("oil elevation not mentioned")
    if label == "healthy" and re.search(
            r"air[- ]leak|oil leak|degradation|imminent fail|remove from "
            r"service", t):
        issues.append("fault language against a nominal verdict")
    return {"ok": not issues, "issues": issues}


def l1_check(text: str, feat: dict, verdict: dict, spec: dict,
             vehicle: str) -> dict:
    nc = numeric_check(text, feat, spec, vehicle)
    cc = classification_check(text, verdict)
    rc = rule_check(text, verdict, feat)
    return {"numeric": nc, "classification": cc, "rule": rc,
            "pass": (nc["n_bad"] == 0 and cc["ok"] and rc["ok"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--system", default=None)
    ap.add_argument("--csv", action="store_true",
                    help="dump per-window rows to metropt_l1.csv")
    args = ap.parse_args()
    spec = load_spec()
    systems = ([args.system] if args.system else
               ["B1", "B2", "B3", "B3R", "B5", "B8", "B8R", "B8S7", "B3RS7"])
    csv_rows = []
    for sys_id in systems:
        try:
            rep = pd.read_parquet(f"{RUNS}/reports_metropt_{sys_id}.parquet")
        except FileNotFoundError:
            print(f"{sys_id}: no reports parquet yet -- skipped")
            continue
        n, bad_num, bad_cls, bad_rule, n_txt = 0, 0, 0, 0, 0
        examples = []
        for _, r in rep.iterrows():
            tag = r["vehicle"]
            feat = _feat_row(tag, r["window_id"])
            verdict = handle(feat, tag, spec)
            c = l1_check(str(r["text"]), feat, verdict, spec, tag)
            if r["text"] and isinstance(r["text"], str) and r["text"].strip():
                n_txt += 1
            n += 1
            bad_num += c["numeric"]["n_bad"] > 0
            bad_cls += (not c["classification"]["ok"]) and (
                not c["classification"]["ambiguous"])
            bad_rule += not c["rule"]["ok"]
            if c["numeric"]["bad"]:
                examples.append((r["window_id"], c["numeric"]["bad"][:2]))
            if args.csv:
                csv_rows.append({
                    "system": sys_id, "vehicle": tag,
                    "window_id": r["window_id"], "kind": r["kind"],
                    "event_id": r.get("event_id"),
                    "offset_h": r.get("offset_h"),
                    "verdict": verdict["label"],
                    "numeric": 1.0 if c["numeric"]["n_bad"] == 0 else 0.0,
                    "n_bad_numbers": c["numeric"]["n_bad"],
                    "classification": (1.0 if c["classification"]["ok"] or
                                       c["classification"]["ambiguous"]
                                       else 0.0),
                    "claimed": c["classification"]["claimed"],
                    "rule": 1.0 if c["rule"]["ok"] else 0.0,
                })
        print(f"{sys_id}: n={n} texts={n_txt}  bad-numeric={bad_num}  "
              f"bad-classification={bad_cls}  bad-rule={bad_rule}")
        for wid, ex in examples[:5]:
            print(f"   {wid}: {ex}")
    if args.csv:
        out = pd.DataFrame(csv_rows)
        out.to_csv(f"{RUNS}/metropt_l1.csv", index=False)
        print(f"L1 rows -> {RUNS}/metropt_l1.csv ({len(out)})")


_FEAT_CACHE: dict[str, pd.DataFrame] = {}


def _feat_row(tag: str, window_id: str) -> dict:
    if tag not in _FEAT_CACHE:
        par = dict(SOURCES)[tag]
        _FEAT_CACHE[tag] = pd.read_parquet(f"{RUNS}/{par}")
    return _FEAT_CACHE[tag].loc[window_id].to_dict()


if __name__ == "__main__":
    main()
