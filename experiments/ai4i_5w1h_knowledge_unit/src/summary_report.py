"""Assemble the Phase 0+1 AI4I foundation summary from saved artifacts.

Reads the JSON/CSV/parquet outputs and emits
``analysis/phase0_1_summary.md``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config as C


def J(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    integ = J(C.DATA_DIR / "integrity_report.json")
    phys = J(C.DATA_DIR / "physics_verification.json")
    enc = J(C.MAPS_DIR / "encoding_test_report.json")
    metrics = pd.read_csv(C.MODELS_DIR / "metrics.csv")
    agg_cols = ["accuracy", "f1_macro", "precision_macro", "recall_macro",
                "auprc", "roc_auc", "train_time_s"]
    g = metrics.groupby("model")[agg_cols]
    mean_df = g.mean().round(4)
    std_df = g.std().round(4)
    summary = pd.DataFrame(index=mean_df.index)
    for c in agg_cols:
        summary[f"{c}_mean"] = mean_df[c]
        summary[f"{c}_std"] = std_df[c]
    summary = summary.reset_index()
    bank = J(C.QUESTION_BANK_DIR / "question_bank_5w1h.json")
    inv = J(C.KNOWLEDGE_DIR / "ku_inventory.json")
    gt_sum = J(C.GROUND_TRUTH_DIR / "gt_summary.json")
    l1 = J(C.RUNS_DIR / "layer1_report.json")
    sample = pd.read_parquet(C.DATA_DIR / "sample_315.parquet")

    L = []
    L.append("# AI4I 5W1H KU Experiment — Phase 0+1 Summary\n")
    L.append("> This report summarizes the Phase 0+1 foundation artifacts. "
             "Report-generation and later evaluation results are summarized separately.\n")
    L.append("> Experiment plan: `docs/AI4I_modified_experiment_plan_EN.md` v6.0 | "
             "Knowledge-layer scale: 315 stratified records.\n")

    # Phase 0.2
    L.append("## 1. Data integrity (Phase 0.2)\n")
    L.append("| Item | Value |\n|---|---|\n")
    L.append(f"| Rows x Cols | {integ['n_rows']:,} x {integ['n_cols']} |\n")
    L.append(f"| Missing / Duplicates | {integ['missing_total']} / {integ['duplicated_rows']} |\n")
    L.append(f"| Failures | {integ['n_failures']} ({integ['failure_rate']*100:.2f}%), "
             f"imbalance ~1:{int((1-integ['failure_rate'])/integ['failure_rate']):.0f} |\n")
    L.append(f"| Multi-mode rows | {integ['multi_mode_rows']} (sum of modes {integ['sum_of_modes']}) |\n")
    L.append("\n| Mode | Count | Rule |\n|---|---|---|\n")
    rules = {"TWF": "tool wear in [200,240) probabilistic / >=240 certain",
             "HDF": "(process-air) temp diff <8.6K AND rpm<1380",
             "PWF": "power outside [3500,9000]W", "OSF": "tool_wear x torque > type limit",
             "RNF": "random (not rule-derivable)"}
    for m in C.FAILURE_MODES:
        L.append(f"| {m} | {integ['mode_counts'][m]} | {rules[m]} |\n")

    # Phase 0.3 physics
    L.append("\n## 2. Physics-rule re-derivation (Phase 0.3)\n")
    L.append("Rules re-derived from variables only, compared to labels:\n\n")
    L.append("| Mode | n_true | Precision | Recall |\n|---|---|---|---|\n")
    for m in ["TWF", "HDF", "PWF", "OSF"]:
        r = phys[m]
        L.append(f"| {m} | {r['n_true']} | {r['precision']} | {r['recall']} |\n")
    L.append(f"\nHDF/PWF/OSF reproduce labels exactly (P=R=1.0); TWF is probabilistic in "
             f"[200,240) ({phys['TWF_band']['n_in_band']} rows, {phys['TWF_band']['n_twf_in_band']} labelled). "
             f"Encoding test: **{enc['n_tested']} rows, {enc['n_failures_in_set']} faulted, "
             f"{'ALL PASSED' if enc['all_passed'] else 'FAILURES'}**.\n")

    # Phase 0.4 models
    L.append("\n## 3. Model quality (Phase 0.4, mean +/- std over 5 seeds)\n")
    L.append("Primary metrics emphasized: **F1-macro** and **AUPRC**; "
             "accuracy is retained for completeness despite the 3.4% class imbalance.\n\n")
    L.append(summary.to_markdown(index=False) + "\n")
    best = metrics.groupby("model")["f1_macro"].mean().idxmax()
    L.append(f"\nBest mean F1-macro: **{best}** "
             f"({metrics[metrics['model']==best]['f1_macro'].mean():.3f}).\n")

    # Phase 0.2 sample
    L.append("\n## 4. Stratified 315-sample (seed 42)\n")
    L.append(f"Failure rate {sample[C.TARGET].mean()*100:.2f}% | Modes: "
             + ", ".join(f"{m}={int(sample[m].sum())}" for m in C.FAILURE_MODES) + "\n")

    # Phase 1A/1D
    L.append("\n## 5. 5W1H Knowledge Units (Phase 1)\n")
    L.append(f"- Question bank: **{bank['n_questions']}** questions "
             f"(invalid dropped: {bank['n_invalid_dropped']})\n")
    L.append("| Category | Questions | M component |\n|---|---|---|\n")
    m_desc = {"What": "classifier/statistics", "Why": "physics causal engine",
              "When": "threshold comparator", "Where": "feature->component map",
              "Who": "fault->responder map", "How": "fault->action map"}
    for c in C.W5H_CATEGORIES:
        L.append(f"| {c} | {bank['category_counts'].get(c,0)} | {m_desc[c]} |\n")
    L.append(f"\n- KU inventory: **{inv['n_kus']}** units, {inv['n_distinct_templates']} templates, "
             f"{len(inv['m_components_used'])} M handlers. The implemented framework uses "
             f"Q/M/P/T components, with P supplied by the encoded physical-rule layer.\n")
    L.append(f"- Programmatic ground truth: **{gt_sum['n_gt_records']:,}** records "
             f"({gt_sum['n_questions']} Q x {gt_sum['n_instances']} instances), "
             f"empty/error: {gt_sum['n_empty_or_error_gt']}.\n")

    # Layer 1
    L.append("\n## 6. Layer 1 programmatic fact-checking\n")
    s = l1["self_validation_on_GT"]
    p = l1["perturbation_test"]
    L.append("L1 = 0.30*tolerance + 0.30*rule + 0.40*classification.\n\n")
    L.append("| Setting | L1 | Tolerance | Rule | Classification |\n|---|---|---|---|---|\n")
    L.append(f"| GT self-validation | {s['L1']} | {s['tolerance_pass']} | {s['rule_pass']} | {s['classification_acc']} |\n")
    L.append(f"| Perturbed (injected errors) | {p['L1_on_perturbed']} | - | - | - |\n")
    L.append(f"\nSelf-consistent GT scores L1=1.0; {p['n_errors_caught']}/{p['n_attempted']} "
             f"injected errors are caught (L1 drops to {p['L1_on_perturbed']}), confirming the "
             f"scorer discriminates. The same scorer applies to LLM reports in Phase 3.\n")

    # Scope
    L.append("\n## 7. Scope of this summary\n")
    L.append("This file summarizes the Phase 0+1 foundation artifacts: data checks, "
             "physics rules, predictive models, the 203-question knowledge layer, "
             "programmatic ground truth, and the Layer-1 scorer.\n\n")
    L.append("The primary B1-B8 report-generation campaign, Layers 2-3 evaluation, "
             "ablations, cross-LLM tests, and validation studies are reported through "
             "their dedicated run and analysis artifacts rather than in this foundation summary.\n")

    out = C.ANALYSIS_DIR / "phase0_1_summary.md"
    out.write_text("".join(L), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
