"""Physical failure rules and engineering derivations for AI4I 2020.

This module is the *data-science model M* backbone for the Why / When / Where /
Who / How knowledge units. Every function is deterministic and grounded in the
engineering rules documented with the AI4I 2020 Predictive Maintenance Dataset
(Matzka, 2020; UCI DOI 10.24432/C5HS5C):

  * PWF  -- mechanical power = torque * rpm * 2*pi/60 outside [3500, 9000] W
  * HDF  -- heat dissipation: (process - air) temperature difference < 8.6 K
            while rotational speed < 1380 rpm
  * OSF  -- overstrain: tool_wear * torque exceeds the type-specific limit
            (L:11000, M:12000, H:13000)
  * TWF  -- tool wear failure: wear >= 240 min always fails; wear in [200,240)
            is probabilistic (tool replaced or fails)
  * RNF  -- random failure, not determined by process variables

All thresholds live in :mod:`src.config`; this module only contains the logic.
"""
from __future__ import annotations

import math
from typing import Mapping

from config import (
    OSF_LIMIT,
    POWER_SAFE_RANGE,
    RPM_LOW,
    TEMP_DIFF_LOW,
    TOOL_WEAR_FAIL,
    TOOL_WEAR_REPLACE_BAND,
)

# --------------------------------------------------------------------------- #
# Low-level physics
# --------------------------------------------------------------------------- #
def mechanical_power(torque_nm: float, rpm: float) -> float:
    """Mechanical power in Watts: P = tau * omega, omega = rpm * 2*pi/60."""
    return float(torque_nm) * float(rpm) * 2.0 * math.pi / 60.0


def temp_difference(air_k: float, process_k: float) -> float:
    """Process minus air temperature (the heat that must be dissipated)."""
    return float(process_k) - float(air_k)


def overstrain(tool_wear_min: float, torque_nm: float) -> float:
    """Overstrain load proxy: tool wear times torque (units: min*Nm)."""
    return float(tool_wear_min) * float(torque_nm)


# --------------------------------------------------------------------------- #
# Single-mode detectors (return bool)
# --------------------------------------------------------------------------- #
def is_pwf(torque_nm: float, rpm: float) -> bool:
    lo, hi = POWER_SAFE_RANGE
    p = mechanical_power(torque_nm, rpm)
    return p < lo or p > hi


def is_hdf(air_k: float, process_k: float, rpm: float) -> bool:
    return (temp_difference(air_k, process_k) < TEMP_DIFF_LOW) and (rpm < RPM_LOW)


def is_osf(type_label: str, tool_wear_min: float, torque_nm: float) -> bool:
    limit = OSF_LIMIT.get(type_label)
    if limit is None:
        return False
    return overstrain(tool_wear_min, torque_nm) > limit


def is_twf_certain(tool_wear_min: float) -> bool:
    """Tool-wear failure that is deterministic given the variables."""
    return float(tool_wear_min) >= TOOL_WEAR_FAIL


def is_twf_candidate(tool_wear_min: float) -> bool:
    """Wear inside the replacement band [200, 240) -- *may* fail (TWF)."""
    lo, hi = TOOL_WEAR_REPLACE_BAND
    return lo <= float(tool_wear_min) < hi


def predict_modes_from_physics(row: Mapping) -> dict:
    """Deterministically re-derive the four rule-based failure modes from a row.

    RNF is excluded: by construction it is random and independent of the
    process variables. The returned dict is the physics-based prediction; it is
    compared against the dataset labels in the encoding test.
    """
    type_label = str(row["Type"])
    return {
        "TWF_pred": is_twf_certain(row["Tool wear [min]"]),
        "HDF_pred": is_hdf(
            row["Air temperature [K]"], row["Process temperature [K]"],
            row["Rotational speed [rpm]"]),
        "PWF_pred": is_pwf(row["Torque [Nm]"], row["Rotational speed [rpm]"]),
        "OSF_pred": is_osf(type_label, row["Tool wear [min]"], row["Torque [Nm]"]),
    }


# --------------------------------------------------------------------------- #
# 5W1H derivations (the M outputs that feed Why/When/Where/Who/How KUs)
# --------------------------------------------------------------------------- #
def power_diagnosis(row: Mapping) -> dict:
    """Why: reason about power relative to the safe envelope."""
    p = mechanical_power(row["Torque [Nm]"], row["Rotational speed [rpm]"])
    lo, hi = POWER_SAFE_RANGE
    if p < lo:
        status, reason = "under-power", f"power {p:.0f} W < lower safe limit {lo:.0f} W"
    elif p > hi:
        status, reason = "over-power", f"power {p:.0f} W > upper safe limit {hi:.0f} W"
    else:
        status, reason = "nominal", f"power {p:.0f} W within [{lo:.0f}, {hi:.0f}] W"
    return {"power_w": round(p, 1), "status": status, "reason": reason}


def thermal_diagnosis(row: Mapping) -> dict:
    """Why/When: heat-dissipation risk from the temperature difference & speed."""
    d = temp_difference(row["Air temperature [K]"], row["Process temperature [K]"])
    rpm = row["Rotational speed [rpm]"]
    risk = d < TEMP_DIFF_LOW and rpm < RPM_LOW
    return {"temp_diff_k": round(d, 2), "rpm": rpm, "hdf_risk": bool(risk)}


def wear_stage(tool_wear_min: float) -> dict:
    """When: classify the tool-wear lifecycle stage."""
    w = float(tool_wear_min)
    lo, hi = TOOL_WEAR_REPLACE_BAND
    if w < lo:
        stage, action = "normal", "no intervention needed"
    elif w < hi:
        stage, action = "warning (replacement band)", "schedule tool replacement"
    else:
        stage, action = "danger (wear >= 240 min)", "stop and replace tool immediately"
    return {"tool_wear_min": w, "stage": stage, "threshold_band": [lo, hi], "action": action}


def locate_anomalies(row: Mapping) -> list:
    """Where: flag which feature/component deviates from its safe operating band."""
    flags = []
    p = mechanical_power(row["Torque [Nm]"], row["Rotational speed [rpm]"])
    lo, hi = POWER_SAFE_RANGE
    if p < lo or p > hi:
        flags.append({"feature": "Torque/Rotational speed", "component": "drive / power train",
                      "value": f"power={p:.0f}W outside [{lo:.0f},{hi:.0f}]"})
    if thermal_diagnosis(row)["hdf_risk"]:
        flags.append({"feature": "Air/Process temperature", "component": "cooling system",
                      "value": f"temp diff {thermal_diagnosis(row)['temp_diff_k']:.1f}K < {TEMP_DIFF_LOW}K"})
    if is_twf_certain(row["Tool wear [min]"]) or is_twf_candidate(row["Tool wear [min]"]):
        flags.append({"feature": "Tool wear", "component": "cutting tool",
                      "value": f"wear={row['Tool wear [min]']}min in/at band {TOOL_WEAR_REPLACE_BAND}"})
    return flags
