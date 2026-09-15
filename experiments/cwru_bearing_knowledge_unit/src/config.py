"""Path configuration for the CWRU bearing knowledge-unit experiment.

Scientific parameters for the bearing diagnostic handler are defined in the
bearing-specific scripts and knowledge files. This module only centralizes
repository paths and the primary experiment seed.
"""
from __future__ import annotations

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

DATA_DIR = ROOT / "data"
BEARING_DATA_DIR = DATA_DIR / "bearing_cwru"
KNOWLEDGE_DIR = ROOT / "knowledge"
RUNS_DIR = ROOT / "runs"
ANALYSIS_DIR = ROOT / "analysis"
DOCS_DIR = ROOT / "docs"

PRIMARY_SEED = 42

for directory in (DATA_DIR, BEARING_DATA_DIR, KNOWLEDGE_DIR, RUNS_DIR, ANALYSIS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def ensure_dirs() -> None:
    """Ensure the experiment output directories exist."""
    for directory in (DATA_DIR, BEARING_DATA_DIR, KNOWLEDGE_DIR, RUNS_DIR, ANALYSIS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
