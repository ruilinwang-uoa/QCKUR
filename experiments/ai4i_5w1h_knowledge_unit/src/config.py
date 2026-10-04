"""Central configuration for the AI4I 5W1H Knowledge-Unit experiment.

Single source of truth for paths, random seeds, train/val/test split ratios,
physical thresholds and fault-mode names. Import from here everywhere so that
no magic numbers are scattered across the pipeline.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]          # experiment root
RAW_CSV = ROOT / "ai4i2020.csv"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
KNOWLEDGE_DIR = ROOT / "knowledge"
QUESTION_BANK_DIR = KNOWLEDGE_DIR / "question_bank"
TEMPLATES_DIR = KNOWLEDGE_DIR / "templates"
GROUND_TRUTH_DIR = KNOWLEDGE_DIR / "ground_truth"
MAPS_DIR = KNOWLEDGE_DIR / "maps"
RUNS_DIR = ROOT / "runs"
LOGS_DIR = ROOT / "logs"
ANALYSIS_DIR = ROOT / "analysis"
CONFIGS_DIR = ROOT / "configs"

for _d in (DATA_DIR, MODELS_DIR, KNOWLEDGE_DIR, QUESTION_BANK_DIR,
           TEMPLATES_DIR, GROUND_TRUTH_DIR, MAPS_DIR, RUNS_DIR, LOGS_DIR,
           ANALYSIS_DIR, CONFIGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEEDS = [42, 1, 7, 2024, 123]          # five seeds, per experimental setup
PRIMARY_SEED = 42                       # used for the fixed 315-sample selection
SPLIT_RATIO = (0.70, 0.15, 0.15)        # train / validation / test

# --------------------------------------------------------------------------- #
# AI4I dataset constants
# --------------------------------------------------------------------------- #
TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FEATURE_NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURE_TYPE = "Type"                   # categorical: L / M / H
TYPE_MAP = {"L": 0, "M": 1, "H": 2}

# Feature set for the predictive-maintenance classifiers (model M for What KUs).
# Raw process variables + product type + three physically meaningful engineered
# features (power, temperature difference, overstrain) that operationalise the
# failure rules and let the linear model capture the rule-based modes.
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type code",
    "Power [W]",
    "Temp diff [K]",
    "Overstrain [min*Nm]",
]

# --------------------------------------------------------------------------- #
# Physics / engineering thresholds (Matzka 2020, AI4I 2020)
# --------------------------------------------------------------------------- #
# Mechanical power: P[W] = torque[Nm] * omega[rad/s]; omega = rpm * 2*pi/60
POWER_SAFE_RANGE = (3500.0, 9000.0)     # PWF when power outside this band
RPM_LOW = 1380.0                        # HDF low-speed threshold
TEMP_DIFF_LOW = 8.6                     # HDF: (process - air) < 8.6 K AND rpm < 1380
TOOL_WEAR_REPLACE_BAND = (200, 240)     # TWF: wear in [200,240) -> replaced or fails
TOOL_WEAR_FAIL = 240                    # TWF: wear >= 240 -> always fail
# OSF overstrain limit (tool_wear * torque) by product quality type
OSF_LIMIT = {"L": 11000.0, "M": 12000.0, "H": 13000.0}

# Feature -> machine component mapping (Where KUs)
FEATURE_COMPONENT_MAP = {
    "Rotational speed [rpm]": "spindle / drive",
    "Torque [Nm]": "tool region / drive train",
    "Tool wear [min]": "cutting tool",
    "Air temperature [K]": "cooling / ambient",
    "Process temperature [K]": "process / workpiece",
}

# Fault mode -> maintenance action mapping (How/Who KUs)
FAULT_ACTION_MAP = {
    "TWF": {
        "component": "cutting tool",
        "action": "replace cutting tool; verify tool-life monitoring threshold",
        "responder": "operator",
        "severity": "medium",
    },
    "HDF": {
        "component": "cooling system",
        "action": "inspect hydraulic/cooling circuit and ventilation; clean filters",
        "responder": "maintenance team",
        "severity": "high",
    },
    "PWF": {
        "component": "drive / power train",
        "action": "reduce feed rate / torque; check motor and drive load",
        "responder": "operator + maintenance",
        "severity": "high",
    },
    "OSF": {
        "component": "tool holder / spindle",
        "action": "reduce feed (tool wear x torque); rebalance tool load",
        "responder": "operator",
        "severity": "medium",
    },
    "RNF": {
        "component": "unspecified",
        "action": "inspect machine; log as random fault; no deterministic rule",
        "responder": "operator",
        "severity": "low",
    },
}

# 5W1H category order (canonical)
W5H_CATEGORIES = ["What", "Why", "When", "Where", "Who", "How"]
def ensure_dirs() -> None:
    """No-op kept for explicitness; dirs are created at import time."""
    return None
