"""
analyze_adjudication.py -- compare the engineer's blinded adjudication
(ratings_in/adjudication_*.json) with the study team's ground truth.

Run from anywhere with pandas:
    python analyze_adjudication.py

Outputs per block: agreement rate + disagreement list (as-is, no
persuasion) -> prints and writes adjudication_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
RATINGS = PKG / "ratings_in"
RUNS = PKG.parent / "runs" / "metropt3"

# study-team ground truth (the things being audited)
OUR_MODE = {"E1": "air leak", "E2": "air leak", "E3": "air leak",
            "E4": "air leak", "A1": "air leak", "A2": "air leak",
            "A3": "oil leak", "B1": "air leak", "B2": "oil leak"}
OUR_DISP = {"E1": "repaired (30 Apr, contested row)", "E2": "self-resolved",
            "E3": "repaired", "E4": "repaired",
            "A1": "removed from service", "A2": "continued in service",
            "A3": "removed from service", "B1": "removed from service",
            "B2": "removed from service"}
# our resolution of the two contested items
OUR_X1 = {"A1": "client pipes and lines", "A2": "dryer"}
OUR_X2 = "April event (E1)"
OUR_B = "genuine undocumented operational episode"   # all three
STRAT_MAP = {"actionable fault indication": "signature",
             "near-threshold precursor only": "subthreshold",
             "nothing actionable": "presymptomatic"}


def norm(s: str) -> str:
    return (s or "").strip().lower()


def main() -> None:
    files = sorted(RATINGS.glob("adjudication_*.json"))
    if not files:
        print(f"no adjudication files in {RATINGS} -- waiting")
        return
    strata = pd.read_csv(RUNS / "metropt_positive_strata.csv")
    our_c = {f"C-{r.vehicle[-4:]}-{r.window_id}": r.stratum
             for _, r in strata.iterrows()}

    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        a = d.get("answers", {})
        rater = d.get("rater") or f.stem
        print(f"\n===== {rater} ({f.name}) =====")
        rep = {"rater": rater, "blockA": {}, "disagreements": []}

        # Block A: mode / disposition
        for eid, ours in OUR_MODE.items():
            got = norm(a.get(f"A_{eid}_mode"))
            ok = got.startswith(ours.split()[0])
            rep["blockA"][f"{eid}:mode"] = {"engineer": got, "ours": ours,
                                            "agree": ok}
            if not ok:
                rep["disagreements"].append(
                    {"item": f"{eid} mode", "engineer": got, "ours": ours})
        for eid, ours in OUR_DISP.items():
            got = norm(a.get(f"A_{eid}_disp"))
            key = ours.split()[0]
            ok = (key.split("(")[0].strip() in got)
            rep["blockA"][f"{eid}:disp"] = {"engineer": got, "ours": ours,
                                            "agree": ok}
            if not ok:
                rep["disagreements"].append(
                    {"item": f"{eid} disposition", "engineer": got,
                     "ours": ours})

        # contested items
        for eid, ours in OUR_X1.items():
            got = norm(a.get(f"X1_{eid}_comp"))
            ok = ours.split()[0] in got
            rep["blockA"][f"X1:{eid}"] = {"engineer": got, "ours": ours,
                                          "agree": ok}
            if not ok:
                rep["disagreements"].append(
                    {"item": f"X1 attribution {eid}", "engineer": got,
                     "ours": ours, "why": a.get(f"X1_why", "")})
        got = norm(a.get("X2_assign", ""))
        ok = "april" in got
        rep["blockA"]["X2"] = {"engineer": got, "ours": OUR_X2, "agree": ok}
        if not ok:
            rep["disagreements"].append(
                {"item": "X2 row assignment", "engineer": got,
                 "ours": OUR_X2, "why": a.get("X2_why", "")})

        # Block B
        rep["blockB"] = {}
        for k, v in a.items():
            if k.startswith("B_X-") and k.endswith("_judg"):
                wid = k[2:-5]
                got = norm(v)
                ok = got.startswith("genuine")
                rep["blockB"][wid] = {"engineer": got, "ours": OUR_B,
                                      "agree": ok,
                                      "confidence": a.get(
                                          f"B_{wid}_conf".replace(
                                              "B_X", "B_X"), "")}
                if not ok:
                    rep["disagreements"].append(
                        {"item": f"B {wid}", "engineer": got, "ours": OUR_B})

        # Block C
        rep["blockC"] = {"n": 0, "agree": 0, "detail": {}}
        for cid, ours in our_c.items():
            got = norm(a.get(f"C_{cid}_strat", ""))
            if not got:
                continue
            mapped = STRAT_MAP.get(got, got)
            ok = mapped == ours
            rep["blockC"]["n"] += 1
            rep["blockC"]["agree"] += int(ok)
            rep["blockC"]["detail"][cid] = {"engineer": mapped, "ours": ours,
                                            "agree": ok}
            if not ok:
                rep["disagreements"].append(
                    {"item": f"C {cid}", "engineer": mapped, "ours": ours})

        # ---- Block C cross-tab, kappa, step distribution, and the
        # detection counts the engineer's strata would imply
        cats = ["presymptomatic", "subthreshold", "signature"]
        eng = {cid[2:]: mapped
               for cid, ours in our_c.items()
               if (mapped := STRAT_MAP.get(norm(a.get(f"C_{cid}_strat", "")), ""))}
        n_c = len(eng)
        if n_c:
            po = sum(eng[k] == our_c["C-" + k] for k in eng) / n_c
            pe = sum(sum(1 for k in eng if eng[k] == c) / n_c
                     * sum(1 for k in eng if our_c["C-" + k] == c) / n_c
                     for c in cats)
            lw = sum(1 - abs(cats.index(eng[k])
                             - cats.index(our_c["C-" + k])) / 2
                     for k in eng) / n_c
            pe_lw = sum(sum(1 for k in eng if eng[k] == x) / n_c
                        * sum(1 for k in eng if our_c["C-" + k] == y) / n_c
                        * (1 - abs(cats.index(x) - cats.index(y)) / 2)
                        for x in cats for y in cats)
            steps = [abs(cats.index(eng[k]) - cats.index(our_c["C-" + k]))
                     for k in eng]
            rep["blockC_stats"] = {
                "n": n_c,
                "raw_agreement": round(po, 3),
                "kappa_unweighted": round((po - pe) / (1 - pe), 3),
                "kappa_linear_weighted": round((lw - pe_lw) / (1 - pe_lw), 3),
                "step_distribution": {str(s): steps.count(s)
                                      for s in sorted(set(steps))},
                "cross_tab": {f"eng={e}|ours={o}":
                              sum(1 for k in eng
                                  if eng[k] == e and our_c["C-" + k] == o)
                              for e in cats for o in cats},
                "engineer_actionable": sorted(k for k in eng
                                              if eng[k] == "signature"),
                "downgraded_signature": sorted(
                    k for k in eng
                    if our_c["C-" + k] == "signature" and eng[k] != "signature"),
            }
            dm_path = RUNS / "metropt_detection_matrix.csv"
            if dm_path.exists():
                dm = pd.read_csv(dm_path)

                def k_of(col: str) -> str:
                    v, w = col.split(":")
                    return f"{v[-4:]}-{w}"

                pos6 = [c for c in dm.columns if c != "system"
                        and "ctl_" not in c and eng.get(k_of(c)) == "signature"]
                down = [c for c in dm.columns if c != "system"
                        and "ctl_" not in c
                        and our_c.get("C-" + k_of(c)) == "signature"
                        and eng.get(k_of(c)) != "signature"]
                rep["detection_under_engineer_strata"] = {
                    "consensus_actionable": [k_of(c) for c in pos6],
                    "downgraded": [k_of(c) for c in down],
                    "per_system": {r["system"]: {
                        "consensus_hits": int(sum(r[c] for c in pos6)),
                        "consensus_n": len(pos6),
                        "premature_claims": int(sum(r[c] for c in down)),
                        "premature_n": len(down)}
                        for _, r in dm.iterrows()},
                }
                print("Block C stats:", rep["blockC_stats"]["raw_agreement"],
                      f"(kappa_u={rep['blockC_stats']['kappa_unweighted']},"
                      f" lin={rep['blockC_stats']['kappa_linear_weighted']})",
                      "steps", rep["blockC_stats"]["step_distribution"])

        # summary
        a_items = rep["blockA"]
        a_ok = sum(1 for v in a_items.values() if v["agree"])
        b_ok = sum(1 for v in rep["blockB"].values() if v["agree"])
        print(f"Block A: {a_ok}/{len(a_items)} agree")
        print(f"Block B: {b_ok}/{len(rep['blockB'])} agree")
        if rep["blockC"]["n"]:
            print(f"Block C: {rep['blockC']['agree']}/{rep['blockC']['n']} "
                  f"agree")
        print(f"Disagreements ({len(rep['disagreements'])}):")
        for d_ in rep["disagreements"]:
            print(f"  - {d_['item']}: engineer={d_['engineer']!r} vs "
                  f"ours={d_['ours']!r}")
        out = HERE / "adjudication_summary.json"
        allrep = {}
        if out.exists():
            allrep = json.loads(out.read_text(encoding="utf-8"))
        allrep[rater] = rep
        out.write_text(json.dumps(allrep, indent=1), encoding="utf-8")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
