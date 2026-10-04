"""Assemble the canonical n=100 report set used by the primary table.

Mirrors composite_final.py's honest sources: B3-B6 from the main run, B7 from
B7_v2 (faithful CoVe), B1/B2/B8 from the rule-engine v3 run.  The output
(reports_main_glm_n100_canonical.parquet) is the exact set of reports behind
Table tab:main, so a second judge can re-score precisely these reports.
"""
from __future__ import annotations

import pandas as pd

import config as C

MAIN = "reports_main_glm_n100"
SRC = {s: MAIN for s in ["B3", "B4", "B5", "B6"]}
SRC["B7"] = "reports_main_glm_n100_B7_v2"
for s in ["B1", "B2", "B8"]:
    SRC[s] = "reports_main_glm_n100_B1B2B8_v3"
ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]


def main() -> None:
    frames = []
    for s in ORDER:
        d = pd.read_parquet(C.RUNS_DIR / f"{SRC[s]}.parquet")
        d = d[d["method"] == s].copy()
        assert len(d) == 100, f"{s}: {len(d)} rows"
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    path = C.RUNS_DIR / "reports_main_glm_n100_canonical.parquet"
    out.to_parquet(path, index=False)
    print(f"canonical: {len(out)} rows -> {path.name}")
    print(out.groupby("method").size().to_string())


if __name__ == "__main__":
    main()
