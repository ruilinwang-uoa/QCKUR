"""Train Logistic Regression / Random Forest / Gradient Boosting classifiers on
AI4I under five seeds with a fixed 70/15/15 stratified split (Phase 0.4).

Quality is measured with F1-macro and AUPRC (not accuracy: the 3.4% failure
rate makes accuracy misleading). Models and test-set predictions are saved for
reuse by the What-class knowledge units.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config as C


def make_estimators(seed: int) -> dict:
    """Instantiate the three model families for one seed."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=400,
            class_weight="balanced", random_state=seed),
    }


def split_data(df: pd.DataFrame, seed: int):
    X = df[C.FEATURES].to_numpy()
    y = df[C.TARGET].to_numpy()
    # 70/15/15, stratified by target, two sequential splits
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "auprc": round(float(average_precision_score(y_true, y_prob)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def main() -> None:
    df = pd.read_parquet(C.DATA_DIR / "ai4i_full_derived.parquet")
    rows, predictions = [], []

    for seed in C.SEEDS:
        X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df, seed)
        # standardize for the linear model (fit on train only)
        scaler = StandardScaler().fit(X_tr)
        for name, est in make_estimators(seed).items():
            Xtr = scaler.transform(X_tr) if name == "logistic_regression" else X_tr
            Xte = scaler.transform(X_te) if name == "logistic_regression" else X_te
            t0 = time.time()
            est.fit(Xtr, y_tr)
            train_time = time.time() - t0
            proba = est.predict_proba(Xte)[:, 1]
            pred = est.predict(Xte)
            m = evaluate(y_te, pred, proba)
            rec = {"seed": seed, "model": name, "n_train": len(y_tr),
                   "n_test": len(y_te), "test_failure_rate": round(float(y_te.mean()), 4),
                   "train_time_s": round(train_time, 3), **m}
            rows.append(rec)
            # persist model + scaler + threshold-0.5 predictions
            joblib.dump({"model": est, "scaler": scaler if name == "logistic_regression" else None,
                         "features": C.FEATURES, "seed": seed},
                        C.MODELS_DIR / f"{name}_{seed}.pkl")
            # stash test predictions for the What-KU classification answers
            predictions.append(pd.DataFrame({
                "seed": seed, "model": name,
                "y_true": y_te, "y_prob": proba, "y_pred": pred,
            }))
            print(f"seed={seed:5d} {name:20s} F1m={m['f1_macro']:.3f} "
                  f"AUPRC={m['auprc']:.3f} AUC={m['roc_auc']:.3f} t={train_time:.2f}s")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(C.MODELS_DIR / "metrics.csv", index=False)

    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.to_parquet(C.MODELS_DIR / "test_predictions.parquet", index=False)

    # mean +/- std summary across seeds
    summary = (metrics.groupby("model")
               [["accuracy", "f1_macro", "precision_macro", "recall_macro",
                 "auprc", "roc_auc", "train_time_s"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(C.MODELS_DIR / "metrics_summary.csv")
    with (C.MODELS_DIR / "metrics_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Model quality (mean +/- std over 5 seeds)\n\n")
        f.write("Primary metrics: **F1-macro** and **AUPRC** (accuracy omitted from "
                "ranking due to 3.4% imbalance).\n\n")
        flat = summary.copy()
        flat.columns = [f"{a}_{b}" for a, b in flat.columns]
        f.write(flat.to_markdown())
        f.write("\n")
    print("\nSaved: metrics.csv, metrics_summary.csv/.md, test_predictions.parquet, 15 model .pkl")


if __name__ == "__main__":
    main()
