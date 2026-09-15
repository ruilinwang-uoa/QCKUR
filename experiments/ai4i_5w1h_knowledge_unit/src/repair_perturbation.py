"""Repaired-scorer validity on the perturbation test (no API cost).

The paper states the repaired numeric scorer "retains validity on the
perturbation test (forged values are still caught)" without a number. This
script reproduces the deterministic Layer-1 perturbation (rng seed 0, same
as layer1_eval.py) and asks: under the REPAIRED numeric acceptance rules
(l1_audit.py: extended truth set = row sensors + power + constants +
derived features + dataset statistics; unit conversions accepted), how many
forged numeric values are accidentally accepted? Reports L1 on perturbed
under the original and the repaired numeric rules.
"""
from __future__ import annotations

import json
import random
import re

import numpy as np
import pandas as pd

import config as C
import ku_handlers as kh
from layer1_eval import dimension, perturb, score_layer1, verify

_CONSTS = [200.0, 240.0, 1380.0, 3500.0, 9000.0, 8.6, 11000.0, 12000.0, 13000.0]
_KEY = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
        "Torque [Nm]", "Tool wear [min]", "Power [W]"]
_DERIVED = ["Temp diff [K]", "Overstrain [min*Nm]"]


def close(a: float, b: float, tol: float = 0.05) -> bool:
    return abs(a - b) <= tol * max(abs(b), 1e-9)


def unit_hit(n: float, truth: list[float]) -> bool:
    for v in truth:
        if close(n * 1000.0, v) or close(n, v * 1000.0):
            return True
        if 50.0 <= v <= 4000.0 and close(n * 2 * np.pi / 60.0, v):
            return True
    return False


def main() -> None:
    import layer1_eval  # noqa: F401  (module-level bank loads fine)
    bank = json.loads((C.QUESTION_BANK_DIR / "question_bank_5w1h.json")
                      .read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in bank["questions"]}
    sample = pd.read_parquet(C.DATA_DIR / "sample_315.parquet")
    full = pd.read_parquet(C.DATA_DIR / "ai4i_full_derived.parquet")
    ctx = kh.build_ctx(sample, full)
    gt = pd.read_parquet(C.GROUND_TRUTH_DIR / "gt.parquet")

    stat_vals: list[float] = []
    for c in _KEY + _DERIVED:
        stat_vals += [full[c].mean(), full[c].std(), full[c].min(), full[c].max()]
    stat_vals += [3.39, 10000.0]
    stat_vals = [v for v in stat_vals if v and 3.0 <= v <= 15000.0]

    rng = random.Random(0)
    records, perturbed = [], []
    n_num = n_num_accepted = 0
    acc_examples = []
    for _, gr in gt.iterrows():
        q = questions[gr["question_id"]]
        row = sample[sample["sample_id"] == gr["sample_id"]].iloc[0].to_dict()
        slots = json.loads(gr["slots"])
        dim = dimension(q["handler"])
        records.append({"category": q["category"], "dim": dim,
                        "ok": verify(q, slots, row, ctx)})
        ps = perturb(q, slots, rng)
        if ps is None:
            continue
        ok_p = verify(q, ps, row, ctx)
        perturbed.append({"category": q["category"], "dim": dim, "ok": ok_p})
        if dim == "numeric":
            n_num += 1
            # which numeric slot did the perturbation forge?
            for k in ("value", "probability", "z", "min", "max"):
                if k in ps and isinstance(ps[k], (int, float)):
                    forged = float(ps[k])
                    row_truth = [float(row[c]) for c in _KEY if c in row]
                    row_truth += [float(row[c]) for c in _DERIVED if c in row]
                    # The repaired scorer accepts dataset-level statistics ONLY
                    # when a statistical-context word accompanies the number
                    # (l1_audit._STAT_CTX). A perturbed bare slot value carries
                    # no such context, so stats are NOT accepted here.
                    accept = (any(close(forged, t) for t in row_truth + _CONSTS)
                              or unit_hit(forged, row_truth + _CONSTS))
                    if accept:
                        n_num_accepted += 1
                        if len(acc_examples) < 5:
                            acc_examples.append((q["id"], k, forged))
                    break

    orig = score_layer1(records)
    pert_orig = score_layer1(perturbed)
    # repaired: rule/cls verdicts unchanged; numeric pass rate = fraction NOT
    # accidentally accepted under the extended acceptance rules
    num_pass = (n_num - n_num_accepted) / n_num if n_num else 0.0
    by_dim = {"numeric": {"pass": n_num - n_num_accepted, "n": n_num},
              "rule": {"pass": 0, "n": 0}, "classification": {"pass": 0, "n": 0}}
    for r in perturbed:
        if r["dim"] in ("rule", "classification"):
            by_dim[r["dim"]]["n"] += 1
            by_dim[r["dim"]]["pass"] += int(r["ok"])
    l1_rep = (0.30 * num_pass
              + 0.30 * (by_dim["rule"]["pass"] / by_dim["rule"]["n"] if by_dim["rule"]["n"] else 0)
              + 0.40 * (by_dim["classification"]["pass"] / by_dim["classification"]["n"]
                        if by_dim["classification"]["n"] else 0))
    print(f"self-validation L1 on GT (original): {orig['L1']}")
    print(f"perturbation: n attempted = {len(perturbed)}, "
          f"numeric = {n_num}, rule = {by_dim['rule']['n']}, "
          f"cls = {by_dim['classification']['n']}")
    print(f"L1 on perturbed (original scorer): {pert_orig['L1']}")
    print(f"numeric forged values accidentally accepted under repaired rules: "
          f"{n_num_accepted}/{n_num}")
    if acc_examples:
        print(f"examples: {acc_examples}")
    print(f"L1 on perturbed (repaired numeric rules): {l1_rep:.3f}")
    print("\nverdict: forged values remain caught under the repaired scorer"
          if n_num_accepted / max(n_num, 1) < 0.05 else
          "\nWARNING: non-trivial accidental acceptance - inspect examples")


if __name__ == "__main__":
    main()
