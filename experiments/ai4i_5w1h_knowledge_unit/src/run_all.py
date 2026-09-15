"""Run the full Phase 0 + Phase 1 pipeline in order."""
from __future__ import annotations

import subprocess
import sys

PY = sys.executable
STEPS = [
    ("0.2 data", "data.py"),
    ("0.3 encode knowledge", "encode_knowledge.py"),
    ("0.4 train models", "train_models.py"),
    ("1A question bank", "question_bank.py"),
    ("1C ground truth", "ground_truth.py"),
    ("1D KU inventory", "ku_inventory.py"),
    ("1D Layer 1 eval", "layer1_eval.py"),
    ("6  summary report", "summary_report.py"),
]


def main() -> None:
    for label, script in STEPS:
        print(f"\n=== {label}: {script} ===", flush=True)
        r = subprocess.run([PY, script], cwd=".")
        if r.returncode != 0:
            print(f"!! {script} failed (exit {r.returncode})")
            sys.exit(r.returncode)
    print("\nAll Phase 0+1 steps completed.")


if __name__ == "__main__":
    main()
