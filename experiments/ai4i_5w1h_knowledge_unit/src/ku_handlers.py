"""Knowledge-Unit model handlers (the *M* component) for the 5W1H framework.

Each handler is a deterministic function ``h_name(row, ctx, **args) -> dict`` that
returns a slot dictionary consumed by an answer template (T). Ground truth for an
instance is exactly the handler output evaluated on the true row, so the GT is
fully programmatic (paper constraint: no human reference text).

``ctx`` carries precomputed dataset statistics, the canonical seed-42 model
predictions on the 315-sample, feature importances and reported model quality.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

import config as C
import physics_rules as pr

CANONICAL_SEED = 42                  # seed used for the What-class reference model
MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]
PRIMARY_MODEL = "random_forest"      # strongest model -> used by default What KUs

_PARAM_UNIT = {
    "Air temperature [K]": "K", "Process temperature [K]": "K",
    "Rotational speed [rpm]": "rpm", "Torque [Nm]": "Nm",
    "Tool wear [min]": "min", "Power [W]": "W",
    "Temp diff [K]": "K", "Overstrain [min*Nm]": "min*Nm",
}
_PARAM_SHORT = {
    "Air temperature [K]": "air temperature", "Process temperature [K]": "process temperature",
    "Rotational speed [rpm]": "rotational speed", "Torque [Nm]": "torque",
    "Tool wear [min]": "tool wear", "Power [W]": "mechanical power",
    "Temp diff [K]": "process-air temperature difference",
    "Overstrain [min*Nm]": "overstrain (tool wear x torque)",
}
_MODE_DESC = {"TWF": "tool wear failure", "HDF": "heat dissipation failure",
              "PWF": "power failure", "OSF": "overstrain failure", "RNF": "random failure"}


# --------------------------------------------------------------------------- #
# Context builder
# --------------------------------------------------------------------------- #
def build_ctx(sample_df: pd.DataFrame, full_df: pd.DataFrame) -> dict:
    stats = {}
    for col in C.FEATURES:
        s = full_df[col]
        stats[col] = {"mean": round(float(s.mean()), 3), "std": round(float(s.std()), 3),
                      "min": round(float(s.min()), 3), "max": round(float(s.max()), 3)}
    stats["__failure_rate__"] = round(float(full_df[C.TARGET].mean()), 4)
    stats["__n_failures__"] = int(full_df[C.TARGET].sum())
    stats["__n_total__"] = int(len(full_df))
    stats["__mode_base_rate__"] = {m: round(float(full_df[m].mean()), 4) for m in C.FAILURE_MODES}

    # canonical seed-42 model predictions on the 315-sample
    Xs = sample_df[C.FEATURES].to_numpy()
    pred_lookup = {}
    importance = {}
    for name in MODELS:
        bundle = joblib.load(C.MODELS_DIR / f"{name}_{CANONICAL_SEED}.pkl")
        model, scaler = bundle["model"], bundle["scaler"]
        Xin = scaler.transform(Xs) if scaler is not None else Xs
        proba = model.predict_proba(Xin)[:, 1]
        pred = (proba >= 0.5).astype(int)
        pred_lookup[name] = {"pred": pred.tolist(), "prob": [round(float(p), 4) for p in proba]}
        # feature importance (coerce coefficients to abs for LR)
        if name == "logistic_regression":
            imp = np.abs(model.coef_[0])
        elif hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        else:
            imp = np.zeros(len(C.FEATURES))
        importance[name] = {f: round(float(v), 4) for f, v in zip(C.FEATURES, imp)}

    # reported model quality (seed 42 row of metrics.csv)
    quality = {}
    mpath = C.MODELS_DIR / "metrics.csv"
    if mpath.exists():
        mdf = pd.read_csv(mpath)
        q = mdf[(mdf["model"].isin(MODELS)) & (mdf["seed"] == CANONICAL_SEED)]
        quality = {r.model: {"f1_macro": r.f1_macro, "auprc": r.auprc,
                             "roc_auc": r.roc_auc, "precision_macro": r.precision_macro,
                             "recall_macro": r.recall_macro}
                   for r in q.itertuples()}

    sample_ids = sample_df["sample_id"].tolist()
    return {
        "stats": stats,
        "pred_lookup": pred_lookup,        # model -> {pred:[...], prob:[...]} aligned to sample_df
        "sample_ids": sample_ids,
        "importance": importance,
        "quality": quality,
    }


def _pos(row, ctx) -> int:
    """Position of this row's sample_id within the canonical prediction arrays."""
    return ctx["sample_ids"].index(row["sample_id"])


# --------------------------------------------------------------------------- #
# WHAT handlers (identification / description)
# --------------------------------------------------------------------------- #
def h_machine_failure(row, ctx):
    return {"failed": "yes" if row["Machine failure"] else "no",
            "label": "failure" if row["Machine failure"] else "no failure"}


def h_fault_mode(row, ctx):
    modes = [m for m in C.FAILURE_MODES if row[m] == 1]
    return {"fault": "+".join(modes) if modes else "none",
            "fault_desc": ", ".join(_MODE_DESC[m] for m in modes) if modes else "no fault mode",
            "n_modes": len(modes)}


def h_param_value(row, ctx, param):
    return {"metric": _PARAM_SHORT[param], "value": round(float(row[param]), 2),
            "unit": _PARAM_UNIT[param]}


def h_param_mean(row, ctx, param):
    return {"metric": _PARAM_SHORT[param],
            "value": ctx["stats"][param]["mean"], "unit": _PARAM_UNIT[param]}


def h_param_range(row, ctx, param):
    s = ctx["stats"][param]
    return {"metric": _PARAM_SHORT[param], "min": s["min"], "max": s["max"],
            "unit": _PARAM_UNIT[param]}


def h_product_type(row, ctx):
    quality = {"L": "low", "M": "medium", "H": "high"}[row["Type"]]
    return {"type": row["Type"], "quality": quality}


def h_model_prediction(row, ctx, model=PRIMARY_MODEL):
    p = _pos(row, ctx)
    pred = ctx["pred_lookup"][model]["pred"][p]
    return {"model": model, "predicted_label": "failure" if pred else "no failure"}


def h_model_confidence(row, ctx, model=PRIMARY_MODEL):
    p = _pos(row, ctx)
    prob = ctx["pred_lookup"][model]["prob"][p]
    return {"model": model, "probability": prob}


def h_failure_rate_dataset(row, ctx):
    return {"rate_pct": round(ctx["stats"]["__failure_rate__"] * 100, 2),
            "n_failures": ctx["stats"]["__n_failures__"],
            "n_total": ctx["stats"]["__n_total__"]}


def h_model_quality(row, ctx, metric, model=PRIMARY_MODEL):
    return {"model": model, "metric": metric,
            "value": ctx["quality"][model][metric]}


def h_overall_failure_rate(row, ctx):
    return {"rate_pct": round(ctx["stats"]["__failure_rate__"] * 100, 2)}


# --------------------------------------------------------------------------- #
# WHY handlers (causal reasoning)
# --------------------------------------------------------------------------- #
def h_why_pwf(row, ctx):
    pdiag = pr.power_diagnosis(row)
    return {"mode": "PWF", "reason": pdiag["reason"], "evidence": f"power={pdiag['power_w']}W"}


def h_why_hdf(row, ctx):
    td = pr.thermal_diagnosis(row)
    return {"mode": "HDF",
            "reason": (f"temperature difference {td['temp_diff_k']}K < {C.TEMP_DIFF_LOW}K "
                       f"at low speed {td['rpm']}rpm < {C.RPM_LOW}rpm"),
            "evidence": f"temp_diff={td['temp_diff_k']}K, rpm={td['rpm']}"}


def h_why_osf(row, ctx):
    ov = pr.overstrain(row["Tool wear [min]"], row["Torque [Nm]"])
    limit = C.OSF_LIMIT[row["Type"]]
    return {"mode": "OSF", "reason": f"overstrain {ov:.0f} > {row['Type']}-limit {limit:.0f}",
            "evidence": f"tool_wear x torque = {ov:.0f} min*Nm"}


def h_why_twf(row, ctx):
    w = row["Tool wear [min]"]
    band = C.TOOL_WEAR_REPLACE_BAND
    if w >= C.TOOL_WEAR_FAIL:
        reason = f"tool wear {w}min >= {C.TOOL_WEAR_FAIL}min (end of life)"
    elif band[0] <= w < band[1]:
        reason = f"tool wear {w}min in replacement band {band} (replacement decision point)"
    else:
        reason = f"tool wear {w}min below the failure band"
    return {"mode": "TWF", "reason": reason, "evidence": f"tool_wear={w}min"}


def h_why_failure(row, ctx):
    """Primary causal explanation: pick the first active deterministic mode."""
    for m in ["PWF", "HDF", "OSF", "TWF"]:
        if row[m] == 1:
            return {"reason": f"{m} ({_MODE_DESC[m]}) condition satisfied",
                    "evidence": _why_evidence(row, m)}
    if row["RNF"] == 1:
        return {"reason": "random failure (RNF): no deterministic process cause",
                "evidence": "RNF is stochastic by dataset design"}
    return {"reason": "all operating variables within safe envelopes; no failure triggered",
            "evidence": "no rule-based mode active"}


def _why_evidence(row, mode):
    return {"PWF": h_why_pwf, "HDF": h_why_hdf, "OSF": h_why_osf,
            "TWF": h_why_twf}[mode](row, {})["evidence"]


def h_why_model_predicts(row, ctx, model=PRIMARY_MODEL):
    p = _pos(row, ctx)
    pred = ctx["pred_lookup"][model]["pred"][p]
    imp = sorted(ctx["importance"][model].items(), key=lambda kv: -kv[1])[:3]
    top = ", ".join(f"{_PARAM_SHORT[f]} ({v:.3f})" for f, v in imp)
    return {"model": model,
            "reason": ("high predicted failure probability driven by "
                       f"{top}") if pred else ("low predicted failure probability; "
                       f"top drivers {top}"),
            "evidence": f"top features: {top}"}


def h_why_no_failure(row, ctx):
    modes = [m for m in C.FAILURE_MODES if row[m] == 1]
    if not modes:
        return {"reason": "all physical safety envelopes hold",
                "evidence": "no rule-based failure mode active"}
    return {"reason": "a failure mode is active (contradicts no-failure premise)",
            "evidence": "+".join(modes)}


def h_why_param_deviation(row, ctx, param):
    val = float(row[param])
    mean = ctx["stats"][param]["mean"]
    std = ctx["stats"][param]["std"] or 1e-9
    z = (val - mean) / std
    return {"param": _PARAM_SHORT[param], "value": round(val, 2),
            "mean": mean, "z": round(z, 2),
            "reason": (f"{z:+.2f} std from the mean") + (" (notable)" if abs(z) > 1.5 else "")}


# --------------------------------------------------------------------------- #
# WHEN handlers (stage / threshold judgement)
# --------------------------------------------------------------------------- #
def h_wear_stage(row, ctx):
    ws = pr.wear_stage(row["Tool wear [min]"])
    return {"stage": ws["stage"], "value": ws["tool_wear_min"],
            "threshold": f"[{ws['threshold_band'][0]},{ws['threshold_band'][1]}) / fail>={C.TOOL_WEAR_FAIL}",
            "action": ws["action"]}


def h_thermal_risk(row, ctx):
    td = pr.thermal_diagnosis(row)
    return {"risk": "yes" if td["hdf_risk"] else "no",
            "value": td["temp_diff_k"], "threshold": C.TEMP_DIFF_LOW}


def h_power_threshold(row, ctx):
    pdiag = pr.power_diagnosis(row)
    lo, hi = C.POWER_SAFE_RANGE
    return {"status": pdiag["status"], "value": pdiag["power_w"],
            "threshold": f"[{lo:.0f},{hi:.0f}]W"}


def h_overstrain_threshold(row, ctx):
    ov = pr.overstrain(row["Tool wear [min]"], row["Torque [Nm]"])
    limit = C.OSF_LIMIT[row["Type"]]
    return {"status": "exceeded" if ov > limit else "within limit",
            "value": round(ov, 1), "threshold": limit, "type": row["Type"]}


def h_intervention_needed(row, ctx):
    modes = [m for m in C.FAILURE_MODES if row[m] == 1]
    ws = pr.wear_stage(row["Tool wear [min]"])
    needed = bool(modes) or ws["stage"].startswith("danger") or ws["stage"].startswith("warning")
    urgency = "immediate" if modes and "RNF" not in modes else (
        "soon" if needed else "none")
    return {"needed": "yes" if needed else "no", "urgency": urgency}


def h_tool_replacement_due(row, ctx):
    w = row["Tool wear [min]"]
    due = w >= C.TOOL_WEAR_REPLACE_BAND[0]
    return {"due": "yes" if due else "no", "wear": w,
            "threshold": C.TOOL_WEAR_REPLACE_BAND[0]}


# --------------------------------------------------------------------------- #
# WHERE handlers (spatial localization)
# --------------------------------------------------------------------------- #
def h_locate_anomaly(row, ctx):
    flags = pr.locate_anomalies(row)
    if not flags:
        return {"component": "none", "feature": "none", "deviation": "no anomaly located"}
    primary = flags[0]
    return {"component": primary["component"], "feature": primary["feature"],
            "deviation": primary["value"]}


def h_locate_component(row, ctx, feature):
    comp = C.FEATURE_COMPONENT_MAP.get(feature, "unspecified")
    return {"component": comp, "feature": _PARAM_SHORT[feature],
            "value": round(float(row[feature]), 2), "unit": _PARAM_UNIT[feature]}


def h_fault_component(row, ctx):
    modes = [m for m in C.FAILURE_MODES if row[m] == 1]
    if not modes:
        return {"mode": "none", "component": "none"}
    m = modes[0]
    return {"mode": m, "component": C.FAULT_ACTION_MAP[m]["component"]}


def h_component_status(row, ctx, component):
    """Operational status of a named component derived from its feature(s)."""
    flags = {f["component"]: f for f in pr.locate_anomalies(row)}
    status = "anomalous" if component in flags else "normal"
    feat_map_inv = {v: k for k, v in C.FEATURE_COMPONENT_MAP.items()}
    # pick the first feature whose component matches for the value
    val = None
    for feat, comp in C.FEATURE_COMPONENT_MAP.items():
        if comp == component:
            val = round(float(row[feat]), 2)
            break
    return {"component": component, "status": status, "value": val}


# --------------------------------------------------------------------------- #
# WHO handlers (impact scope / responder)
# --------------------------------------------------------------------------- #
def h_fault_responder(row, ctx):
    modes = [m for m in C.FAILURE_MODES if row[m] == 1]
    if not modes:
        return {"responder": "operator (routine monitoring)", "severity": "none", "mode": "none"}
    # highest-severity active mode dictates responder
    order = ["HDF", "PWF", "OSF", "TWF", "RNF"]
    m = next(x for x in order if x in modes)
    fa = C.FAULT_ACTION_MAP[m]
    return {"responder": fa["responder"], "severity": fa["severity"], "mode": m}


def h_affected_process(row, ctx):
    modes = [m for m in C.FAILURE_MODES if row[m] == 1]
    if "PWF" in modes or "OSF" in modes:
        proc = "machining accuracy and drive load"
    elif "HDF" in modes:
        proc = "thermal stability and cooling"
    elif "TWF" in modes:
        proc = "surface finish and tool life"
    elif "RNF" in modes:
        proc = "unspecified (random)"
    else:
        proc = "no process impact"
    return {"process": proc}


def h_severity_level(row, ctx):
    return h_fault_responder(row, ctx)


# --------------------------------------------------------------------------- #
# HOW handlers (action guidance)
# --------------------------------------------------------------------------- #
def h_recommended_action(row, ctx):
    modes = [m for m in C.FAILURE_MODES if row[m] == 1]
    if not modes:
        return {"action": "no intervention; continue routine monitoring",
                "parameter": "none", "suggested": "maintain current settings", "mode": "none"}
    order = ["HDF", "PWF", "OSF", "TWF", "RNF"]
    m = next(x for x in order if x in modes)
    fa = C.FAULT_ACTION_MAP[m]
    # suggest a concrete parameter tweak where the physics defines one
    if m == "PWF":
        suggested = "reduce torque/feed to bring power within [3500,9000]W"
        parameter = "torque / feed rate"
    elif m == "HDF":
        suggested = "reduce speed drop / improve cooling below 1380rpm threshold"
        parameter = "cooling / rotational speed"
    elif m == "OSF":
        suggested = f"reduce tool_wear x torque below {C.OSF_LIMIT[row['Type']]:.0f}"
        parameter = "feed rate / tool load"
    elif m == "TWF":
        suggested = f"replace tool (wear {row['Tool wear [min]']}min)"
        parameter = "cutting tool"
    else:
        suggested = "inspect machine; log random fault"
        parameter = "general inspection"
    return {"action": fa["action"], "parameter": parameter, "suggested": suggested, "mode": m}


def h_action_for_mode(row, ctx, mode):
    fa = C.FAULT_ACTION_MAP[mode]
    return {"mode": mode, "action": fa["action"], "responder": fa["responder"],
            "component": fa["component"]}


def h_parameter_adjustment(row, ctx, param):
    """Suggested adjustment to keep a parameter within its safe band."""
    val = float(row[param])
    if param == "Power [W]":
        lo, hi = C.POWER_SAFE_RANGE
        if val < lo:
            sug = f"increase to >= {lo:.0f}W (raise torque or rpm)"
        elif val > hi:
            sug = f"decrease to <= {hi:.0f}W (lower torque or rpm)"
        else:
            sug = "within safe band; no change"
    elif param == "Tool wear [min]":
        sug = "replace tool" if val >= C.TOOL_WEAR_REPLACE_BAND[0] else "no change needed"
    else:
        sug = "monitor; no deterministic limit encoded"
    return {"parameter": _PARAM_SHORT[param], "current": round(val, 2),
            "unit": _PARAM_UNIT[param], "suggested": sug}


def h_prevention(row, ctx):
    modes = [m for m in C.FAILURE_MODES if row[m] == 1]
    if not modes:
        return {"measure": "maintain scheduled condition monitoring and tool-life tracking"}
    return {"measure": "address the active failure mode per the recommended action"}


# --------------------------------------------------------------------------- #
# Extra handlers (statistical / model-level, used across categories)
# --------------------------------------------------------------------------- #
def h_param_std(row, ctx, param):
    return {"metric": _PARAM_SHORT[param],
            "value": ctx["stats"][param]["std"], "unit": _PARAM_UNIT[param]}


def h_param_deviation(row, ctx, param):
    """Z-score of this instance's value from the dataset mean (What framing)."""
    val = float(row[param])
    mean = ctx["stats"][param]["mean"]
    std = ctx["stats"][param]["std"] or 1e-9
    z = (val - mean) / std
    return {"metric": _PARAM_SHORT[param], "value": round(val, 2),
            "mean": mean, "std": round(std, 3), "z": round(z, 2),
            "unit": _PARAM_UNIT[param]}


def h_mode_base_rate(row, ctx, mode):
    rate = ctx["stats"]["__mode_base_rate__"][mode]
    return {"mode": mode, "desc": _MODE_DESC[mode], "rate_pct": round(rate * 100, 3)}


def h_feature_importance(row, ctx, feature, model=PRIMARY_MODEL):
    imp = ctx["importance"].get(model, {})
    return {"model": model, "feature": _PARAM_SHORT[feature],
            "importance": imp.get(feature, 0.0)}


def h_threshold_breach(row, ctx, param):
    """Whether a parameter breaches its physics-encoded safe threshold."""
    if param == "Power [W]":
        p = pr.mechanical_power(row["Torque [Nm]"], row["Rotational speed [rpm]"])
        lo, hi = C.POWER_SAFE_RANGE
        breached = p < lo or p > hi
        thresh = f"[{lo:.0f},{hi:.0f}]W"
    elif param == "Tool wear [min]":
        breached = row["Tool wear [min]"] >= C.TOOL_WEAR_REPLACE_BAND[0]
        thresh = f">= {C.TOOL_WEAR_REPLACE_BAND[0]}min"
    elif param == "Overstrain [min*Nm]":
        ov = pr.overstrain(row["Tool wear [min]"], row["Torque [Nm]"])
        breached = ov > C.OSF_LIMIT[row["Type"]]
        thresh = f"{C.OSF_LIMIT[row['Type']]:.0f} ({row['Type']})"
    elif param == "Temp diff [K]":
        td = pr.temp_difference(row["Air temperature [K]"], row["Process temperature [K]"])
        breached = td < C.TEMP_DIFF_LOW and row["Rotational speed [rpm]"] < C.RPM_LOW
        thresh = f"< {C.TEMP_DIFF_LOW}K & rpm<{C.RPM_LOW}"
    else:
        breached = False
        thresh = "no physics threshold encoded"
    return {"metric": _PARAM_SHORT[param], "value": round(float(row[param]), 2),
            "breached": "yes" if breached else "no", "threshold": thresh}


def h_why_not_mode(row, ctx, mode):
    """Why a specific failure mode is *not* triggered for this sample."""
    if row[mode] == 1:
        return {"mode": mode, "reason": f"{mode} IS active (premise false)",
                "evidence": f"{mode}=1"}
    reason_map = {
        "PWF": lambda r: f"power {pr.power_diagnosis(r)['power_w']:.0f}W within [3500,9000]W",
        "HDF": lambda r: (f"temp diff {pr.thermal_diagnosis(r)['temp_diff_k']}K >= {C.TEMP_DIFF_LOW}K "
                          f"or rpm {r['Rotational speed [rpm]']} >= {C.RPM_LOW}"),
        "OSF": lambda r: (f"overstrain {pr.overstrain(r['Tool wear [min]'], r['Torque [Nm]']):.0f} "
                          f"<= limit {C.OSF_LIMIT[r['Type']]:.0f}"),
        "TWF": lambda r: f"tool wear {r['Tool wear [min]']}min < failure band",
        "RNF": lambda r: "random failures are stochastic and cannot be reasoned deterministically",
    }
    fn = reason_map.get(mode)
    reason = fn(row) if fn else f"{mode} not active"
    return {"mode": mode, "reason": reason, "evidence": f"{mode}=0"}


def h_why_rnf(row, ctx):
    if row["RNF"] == 1:
        return {"reason": "random failure (RNF) is stochastic by dataset design",
                "evidence": "RNF has no deterministic process-cause"}
    return {"reason": "no random failure recorded for this sample",
            "evidence": "RNF=0"}


def h_dataset_imbalance(row, ctx):
    rate = ctx["stats"]["__failure_rate__"]
    return {"rate_pct": round(rate * 100, 2),
            "ratio": f"1 : {round((1 - rate) / rate):.0f}"}


# map a machine component to its typical responsible role (Who framing)
_COMPONENT_ROLE = {
    "spindle / drive": "operator + maintenance technician",
    "tool region / drive train": "operator",
    "cutting tool": "operator",
    "cooling / ambient": "maintenance team",
    "process / workpiece": "process engineer",
    "cooling system": "maintenance team",
    "drive / power train": "operator + maintenance",
    "tool holder / spindle": "operator",
}


def h_role_for_component(row, ctx, component):
    role = _COMPONENT_ROLE.get(component, "operator")
    status = "anomalous" if any(f["component"] == component for f in pr.locate_anomalies(row)) else "normal"
    return {"component": component, "role": role, "status": status}


def h_mitigation_for_mode(row, ctx, mode):
    fa = C.FAULT_ACTION_MAP[mode]
    active = row[mode] == 1
    return {"mode": mode, "active": "yes" if active else "no",
            "mitigation": fa["action"], "component": fa["component"]}


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
HANDLERS = {name: fn for name, fn in globals().items() if name.startswith("h_")}


def run_handler(name: str, row, ctx, args: dict | None = None) -> dict:
    args = args or {}
    if name not in HANDLERS:
        raise KeyError(f"unknown handler '{name}'")
    return HANDLERS[name](row, ctx, **args)
