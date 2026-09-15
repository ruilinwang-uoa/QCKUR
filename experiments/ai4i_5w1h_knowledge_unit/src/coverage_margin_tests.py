"""Significance tests for B8's 5W1H coverage margin over the baselines.

The paper states the residual coverage margin is untested for significance
(cross-LLM backend, ~0.7 aspects). Per-instance coverage counts exist for
the PRIMARY backend under two judges (primary DeepSeek + family-disjoint
Kimi K2.6), where the margins are 1.7 (DeepSeek) and 0.2-0.8 (Kimi) aspects;
the Kimi margins are smaller than the cross-LLM 0.7, so testing them is the
more binding case. Cross-LLM per-instance data for B2-B7 was not retained
(system means only), noted in the output.

Paired Wilcoxon on per-instance 5W1H aspect counts (/6), B8 vs each system.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats

import config as C

ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
MAIN = "reports_main_glm_n100"
SRC = {s: MAIN for s in ["B3", "B4", "B5", "B6"]}
SRC["B7"] = "reports_main_glm_n100_B7_v2"
for s in ["B1", "B2", "B8"]:
    SRC[s] = "reports_main_glm_n100_B1B2B8_v3"


def cov6(c) -> int:
    return sum(c.values()) if isinstance(c, dict) else 0


def battery(tag: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wide = pd.DataFrame({m: d.set_index("instance_id")["coverage"].apply(cov6)
                         for m, d in frames.items()})
    rows = []
    for b in ["B5", "B3", "B4", "B6", "B7", "B2", "B1"]:
        d = wide["B8"] - wide[b]
        w = stats.wilcoxon(d, zero_method="wilcox", method="auto")
        rows.append({"judge": tag, "comparison": f"B8 vs {b}",
                     "B8_cov": wide["B8"].mean(), "baseline_cov": wide[b].mean(),
                     "margin": d.mean(), "win_rate": (d > 0).mean(),
                     "p": w.pvalue})
    return pd.DataFrame(rows)


# primary DeepSeek judge (canonical assembly, matches Table 3)
ds_frames = {m: pd.read_parquet(C.RUNS_DIR / f"{SRC[m]}_l3.parquet").query("method == @m")
             for m in ORDER}
ds = battery("DeepSeek (primary)", ds_frames)

# Kimi fourth judge (all 8 systems in one file)
k = pd.read_parquet(C.RUNS_DIR / "reports_main_glm_n100_kimi_l3.parquet")
km_frames = {m: k.query("method == @m") for m in ORDER}
km = battery("Kimi K2.6 (family-disjoint)", km_frames)

out = pd.concat([ds, km], ignore_index=True)
out["margin"] = out["margin"].round(2)
out["B8_cov"] = out["B8_cov"].round(2)
out["baseline_cov"] = out["baseline_cov"].round(2)
out["win_rate"] = (out["win_rate"] * 100).round(0).astype(int)
out["p"] = out["p"].apply(lambda v: f"{v:.1e}" if v >= 1e-5 else "<1e-5")
print(out.to_string(index=False))
out.to_csv(C.ANALYSIS_DIR / "coverage_margin_tests.csv", index=False)
print(f"\nsaved: {C.ANALYSIS_DIR / 'coverage_margin_tests.csv'}")
print("note: cross-LLM per-instance coverage for B2-B7 not retained; "
      "the Kimi margins (0.2-0.8) are smaller than the cross-LLM 0.7.")
