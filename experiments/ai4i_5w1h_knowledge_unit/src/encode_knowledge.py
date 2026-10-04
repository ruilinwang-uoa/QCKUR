"""Dump the domain-knowledge maps and run the encoding validation test
(experiment Phase 0.3).

Produces:
  * knowledge/maps/feature_component_map.json
  * knowledge/maps/fault_action_map.json
  * knowledge/maps/physics_rules_spec.json
  * knowledge/maps/encoding_test_report.json   (>=20-sample acceptance test)
"""
from __future__ import annotations

import json

import pandas as pd

import config as C
import physics_rules as pr


def dump_maps() -> None:
    (C.MAPS_DIR / "feature_component_map.json").write_text(
        json.dumps(C.FEATURE_COMPONENT_MAP, indent=2, ensure_ascii=False), encoding="utf-8")
    (C.MAPS_DIR / "fault_action_map.json").write_text(
        json.dumps(C.FAULT_ACTION_MAP, indent=2, ensure_ascii=False), encoding="utf-8")
    (C.MAPS_DIR / "physics_rules_spec.json").write_text(json.dumps({
        "mechanical_power": "P[W] = torque[Nm] * rpm * 2*pi/60",
        "power_safe_range_W": list(C.POWER_SAFE_RANGE),
        "hdf_rule": f"(process_temp - air_temp) < {C.TEMP_DIFF_LOW} K AND rpm < {C.RPM_LOW}",
        "osf_limits_minNm_by_type": C.OSF_LIMIT,
        "twf_replace_band_min": list(C.TOOL_WEAR_REPLACE_BAND),
        "twf_fail_min": C.TOOL_WEAR_FAIL,
        "note": "RNF is random by construction and not rule-derivable",
    }, indent=2), encoding="utf-8")


def _check(cond: bool, label: str, fails: list) -> None:
    if not cond:
        fails.append(label)


def encoding_test(df: pd.DataFrame, per_mode: int = 5) -> dict:
    """Validate every KU-relevant derivation on a stratified set of >=20 rows."""
    fails: list[str] = []
    tested = []
    # pick per_mode rows from each deterministic mode + some no-failure rows
    picks = []
    for mode in ["PWF", "HDF", "OSF"]:
        picks += df[df[mode] == 1].head(per_mode).index.tolist()
    picks += df[df["TWF"] == 1].head(per_mode).index.tolist()      # probabilistic
    picks += df[df["RNF"] == 1].head(per_mode).index.tolist()      # random
    picks += df[df[C.TARGET] == 0].head(per_mode).index.tolist()   # healthy
    rows = df.loc[picks].drop_duplicates().to_dict("records")

    for r in rows:
        pdiag = pr.power_diagnosis(r)
        tdiag = pr.thermal_diagnosis(r)
        wstage = pr.wear_stage(r["Tool wear [min]"])
        locs = pr.locate_anomalies(r)
        modes = pr.predict_modes_from_physics(r)
        active = [m for m in C.FAILURE_MODES if r[m] == 1]
        deterministic_active = [m for m in ["TWF", "HDF", "PWF", "OSF"] if r[m] == 1]

        # structural checks (always)
        _check(pdiag["status"] in {"under-power", "over-power", "nominal"},
               f"{r['UDI']}: power status invalid", fails)
        _check(isinstance(wstage["stage"], str) and wstage["stage"],
               f"{r['UDI']}: wear stage empty", fails)
        _check(set(wstage["threshold_band"]) == set(C.TOOL_WEAR_REPLACE_BAND),
               f"{r['UDI']}: wear threshold band wrong", fails)
        _check(all("component" in f and "feature" in f for f in locs) or len(locs) == 0,
               f"{r['UDI']}: anomaly location malformed", fails)

        # physical-consistency checks (the real test)
        if "PWF" in active:
            _check(pdiag["status"] != "nominal",
                   f"{r['UDI']}: PWF but power nominal", fails)
            _check(modes["PWF_pred"] is True,
                   f"{r['UDI']}: PWF not re-derived", fails)
        if "HDF" in active:
            _check(tdiag["hdf_risk"] is True,
                   f"{r['UDI']}: HDF but no thermal risk", fails)
            _check(modes["HDF_pred"] is True,
                   f"{r['UDI']}: HDF not re-derived", fails)
        if "OSF" in active:
            _check(modes["OSF_pred"] is True,
                   f"{r['UDI']}: OSF not re-derived", fails)
        if r["Tool wear [min]"] >= C.TOOL_WEAR_FAIL:
            _check(wstage["stage"].startswith("danger"),
                   f"{r['UDI']}: wear>=240 not 'danger'", fails)
        if any(r[m] == 1 for m in C.FAILURE_MODES):
            # a faulted row should be flagged by >=1 anomaly, EXCEPT a pure
            # random failure (RNF), which by construction is not physics-derivable
            if active != ["RNF"]:
                _check(len(locs) > 0, f"{r['UDI']}: faulted but no anomaly located", fails)

        # fault-action mapping must resolve for every documented mode
        for m in C.FAILURE_MODES:
            fa = C.FAULT_ACTION_MAP[m]
            _check(all(k in fa for k in ("component", "action", "responder", "severity")),
                   f"{r['UDI']}: fault_action_map[{m}] incomplete", fails)

        tested.append({
            "UDI": int(r["UDI"]), "active_modes": active or ["none"],
            "power_status": pdiag["status"], "hdf_risk": tdiag["hdf_risk"],
            "wear_stage": wstage["stage"], "n_anomalies": len(locs),
        })

    return {
        "n_tested": len(tested),
        "n_failures_in_set": sum(1 for t in tested if t["active_modes"] != ["none"]),
        "all_passed": len(fails) == 0,
        "failures": fails,
        "sample_rows": tested[:12],
    }


def main() -> None:
    dump_maps()
    df = pd.read_parquet(C.DATA_DIR / "ai4i_full_derived.parquet")
    report = encoding_test(df, per_mode=5)
    (C.MAPS_DIR / "encoding_test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Maps dumped to knowledge/maps/")
    print(f"Encoding test: {report['n_tested']} rows tested "
          f"({report['n_failures_in_set']} faulted) -> "
          f"{'ALL PASSED' if report['all_passed'] else 'FAILURES: ' + str(report['failures'])}")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
