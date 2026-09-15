"""AI4I data loading, integrity verification, feature derivation, stratified
sampling and EDA report generation (experiment Phase 0.2).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import physics_rules as pr


# --------------------------------------------------------------------------- #
# Load + verify
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(C.RAW_CSV)
    expected_cols = [
        "UDI", "Product ID", "Type", "Air temperature [K]",
        "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
        "Tool wear [min]", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
    ]
    assert list(df.columns) == expected_cols, f"unexpected columns: {list(df.columns)}"
    return df


def integrity_report(df: pd.DataFrame) -> dict:
    rep = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "failure_rate": round(df[C.TARGET].mean(), 4),
        "n_failures": int(df[C.TARGET].sum()),
        "mode_counts": {m: int(df[m].sum()) for m in C.FAILURE_MODES},
        "type_counts": {k: int(v) for k, v in df["Type"].value_counts().items()},
    }
    # multi-mode co-occurrence
    rep["multi_mode_rows"] = int((df[C.FAILURE_MODES].sum(axis=1) > 1).sum())
    rep["sum_of_modes"] = int(df[C.FAILURE_MODES].sum().sum())
    return rep


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns used by models and KUs.

    Vectorized here for speed on the full 10k table; the scalar functions in
    :mod:`physics_rules` remain the single-instance engine used by KUs.
    """
    out = df.copy()
    torque = out["Torque [Nm]"].to_numpy()
    rpm = out["Rotational speed [rpm]"].to_numpy()
    out["Power [W]"] = torque * rpm * 2.0 * np.pi / 60.0
    out["Temp diff [K]"] = out["Process temperature [K]"].to_numpy() - out["Air temperature [K]"].to_numpy()
    out["Overstrain [min*Nm]"] = out["Tool wear [min]"].to_numpy() * torque
    out["Type code"] = out["Type"].map(C.TYPE_MAP)
    out["Failure mode"] = out.apply(_dominant_mode_label, axis=1)
    return out


def _dominant_mode_label(row: pd.Series) -> str:
    """Single readable failure label per row (for stratification & reporting)."""
    active = [m for m in C.FAILURE_MODES if row[m] == 1]
    if not active:
        return "No failure"
    # priority order for rows with >1 mode: deterministic, documented choice
    return "+".join(active)


# --------------------------------------------------------------------------- #
# Physics-rule verification (encoding test backbone)
# --------------------------------------------------------------------------- #
def verify_physics_rules(df: pd.DataFrame) -> dict:
    """Re-derive TWF/HDF/PWF/OSF from variables and compare to dataset labels.

    TWF in [200,240) is probabilistic in the source, so we test the *certain*
    part (wear>=240) for precision/recall and report the band separately. RNF
    is random and is excluded by construction.
    """
    rows = df.to_dict("records")
    pred = [pr.predict_modes_from_physics(r) for r in rows]
    pred_df = pd.DataFrame(pred, index=df.index)

    result = {}
    for mode in ["TWF", "HDF", "PWF", "OSF"]:
        y_true = df[mode].values
        y_pred = pred_df[f"{mode}_pred"].values
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        result[mode] = {"tp": tp, "fp": fp, "fn": fn,
                        "precision": round(prec, 4), "recall": round(rec, 4),
                        "n_true": int(y_true.sum())}
    # TWF probabilistic band analysis
    band = df[df["Tool wear [min]"].between(*C.TOOL_WEAR_REPLACE_BAND)]
    result["TWF_band"] = {
        "n_in_band": int(len(band)),
        "n_twf_in_band": int(band["TWF"].sum()),
        "note": "wear in [200,240) is probabilistic; encoded as candidate, not certain",
    }
    return result


# --------------------------------------------------------------------------- #
# Stratified 315-sample selection
# --------------------------------------------------------------------------- #
def stratified_sample(df: pd.DataFrame, n: int = 315, seed: int = C.PRIMARY_SEED) -> pd.DataFrame:
    """Stratified sample preserving the failure/no-failure and mode proportions.

    With ~3.39% failures, 315 rows yield ~10-11 failure rows -- enough to
    exercise every KU category on real faults while keeping the majority
    distribution representative.
    """
    rng = np.random.RandomState(seed)
    failure_rate = df[C.TARGET].mean()
    n_fail = max(1, int(round(n * failure_rate)))
    n_ok = n - n_fail

    fail_idx = df.index[df[C.TARGET] == 1].to_numpy()
    ok_idx = df.index[df[C.TARGET] == 0].to_numpy()
    chosen_fail = rng.choice(fail_idx, size=min(n_fail, len(fail_idx)), replace=False)
    chosen_ok = rng.choice(ok_idx, size=min(n_ok, len(ok_idx)), replace=False)
    idx = np.concatenate([chosen_fail, chosen_ok])
    sample = df.loc[idx].sort_index().reset_index(drop=True)
    sample["sample_id"] = [f"s{int(i):04d}" for i in range(len(sample))]
    return sample


# --------------------------------------------------------------------------- #
# EDA report
# --------------------------------------------------------------------------- #
def eda_report(df: pd.DataFrame, integrity: dict, physics: dict, sample: pd.DataFrame,
               out_md: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = []
    lines.append("# AI4I 2020 — Exploratory Data Analysis Report\n")
    lines.append(f"- Source: `{C.RAW_CSV.name}`\n")

    lines.append("## 1. Integrity\n")
    lines.append(f"- Rows: **{integrity['n_rows']:,}** | Columns: **{integrity['n_cols']}**\n")
    lines.append(f"- Missing values: **{integrity['missing_total']}** | "
                 f"Duplicated rows: **{integrity['duplicated_rows']}**\n")
    lines.append(f"- Machine failure: **{integrity['n_failures']} / {integrity['n_rows']:,}** "
                 f"(rate **{integrity['failure_rate']*100:.2f}%**, imbalance ≈ "
                 f"1 : {int((1-integrity['failure_rate'])/integrity['failure_rate']):.0f})\n")
    lines.append("### Failure-mode counts\n")
    lines.append("| Mode | Count | Description |\n|---|---|---|\n")
    desc = {"TWF": "Tool wear failure", "HDF": "Heat dissipation failure",
            "PWF": "Power failure", "OSF": "Overstrain failure", "RNF": "Random failure"}
    for m in C.FAILURE_MODES:
        lines.append(f"| {m} | {integrity['mode_counts'][m]} | {desc[m]} |\n")
    lines.append(f"\n- Rows with >1 active mode: **{integrity['multi_mode_rows']}** "
                 f"(sum of modes {integrity['sum_of_modes']} > failure count "
                 f"{integrity['n_failures']})\n")
    lines.append(f"- Product type counts: {integrity['type_counts']}\n")

    lines.append("\n## 2. Numeric feature summary\n")
    desc_df = df[C.FEATURE_NUMERIC + ["Power [W]", "Temp diff [K]", "Overstrain [min*Nm]"]].describe().round(2)
    lines.append(desc_df.to_markdown() + "\n")

    # correlations
    lines.append("\n## 3. Correlation with `Machine failure`\n")
    num = df[C.FEATURE_NUMERIC + ["Power [W]", "Temp diff [K]", "Overstrain [min*Nm]", "Type code"]]
    corr = num.join(df[C.TARGET]).corr()[C.TARGET].drop(C.TARGET).sort_values(key=abs, ascending=False)
    lines.append("| Feature | r(failure) |\n|---|---|\n")
    for feat, r in corr.items():
        lines.append(f"| {feat} | {r:.3f} |\n")

    # figures
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    plot_feats = C.FEATURE_NUMERIC + ["Power [W]"]
    for ax, feat in zip(axes.ravel(), plot_feats):
        for label, sub in df.groupby(C.TARGET):
            ax.hist(sub[feat], bins=40, alpha=0.5,
                    label="failure" if label else "no failure")
        ax.set_title(feat, fontsize=9)
        ax.legend(fontsize=7)
    axes[1, 2].axis("off")
    fig.suptitle("Feature distributions: failure vs no-failure", fontsize=11)
    fig.tight_layout()
    fig_path = C.ANALYSIS_DIR / "eda_feature_distributions.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    lines.append(f"\n![Feature distributions](../analysis/eda_feature_distributions.png)\n")

    lines.append("\n## 4. Physics-rule re-derivation (encoding validation)\n")
    lines.append("Re-derived TWF/HDF/PWF/OSF purely from variables, compared to labels:\n\n")
    lines.append("| Mode | n_true | TP | FP | FN | Precision | Recall |\n"
                 "|---|---|---|---|---|---|---|\n")
    for mode in ["TWF", "HDF", "PWF", "OSF"]:
        r = physics[mode]
        lines.append(f"| {mode} | {r['n_true']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                     f"{r['precision']} | {r['recall']} |\n")
    lines.append(f"\n- TWF replacement band [200,240): {physics['TWF_band']['n_in_band']} rows, "
                 f"{physics['TWF_band']['n_twf_in_band']} labelled TWF "
                 f"(probabilistic in source — encoded as *candidate*).\n")
    lines.append("- RNF is random by construction and excluded from rule derivation.\n")

    lines.append("\n## 5. Stratified 315-sample selection\n")
    s_int = integrity_report(sample)
    lines.append(f"- Seed: **{C.PRIMARY_SEED}** | Size: **{len(sample)}**\n")
    lines.append(f"- Sample failure rate: **{s_int['failure_rate']*100:.2f}%** "
                 f"(population {integrity['failure_rate']*100:.2f}%)\n")
    lines.append(f"- Sample mode counts: {s_int['mode_counts']}\n")
    lines.append(f"- Sample single-readable-label distribution:\n\n")
    vc = sample["Failure mode"].value_counts()
    lines.append("| Label | Count |\n|---|---|\n")
    for lab, cnt in vc.items():
        lines.append(f"| {lab} | {cnt} |\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    df = load_raw()
    integrity = integrity_report(df)
    df = derive_features(df)

    physics = verify_physics_rules(df)

    sample = stratified_sample(df, n=315, seed=C.PRIMARY_SEED)

    df.to_parquet(C.DATA_DIR / "ai4i_full_derived.parquet", index=False)
    sample.to_parquet(C.DATA_DIR / "sample_315.parquet", index=False)
    (C.DATA_DIR / "integrity_report.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8")
    (C.DATA_DIR / "physics_verification.json").write_text(
        json.dumps(physics, indent=2), encoding="utf-8")

    eda_report(df, integrity, physics, sample, C.ANALYSIS_DIR / "eda_report.md")
    print("Integrity:", json.dumps(integrity, indent=2))
    print("\nPhysics verification:", json.dumps(physics, indent=2))
    print("\nSample failure rate:", round(sample[C.TARGET].mean(), 4),
          "| sample size:", len(sample))
    print("Saved: derived parquet, sample_315, integrity_report, physics_verification, eda_report")


if __name__ == "__main__":
    main()
