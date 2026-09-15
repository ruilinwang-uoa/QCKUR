"""Round-3 review patches: B8-vs-B2 under both cross-LLM judges, exact Spearman p."""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
from scipy import stats

import config as C

W = {"L1": 0.35, "L2": 0.30, "L3": 0.35}
l1 = pd.read_parquet(C.RUNS_DIR / "reports_crossllm_deepseek_n100_l1.parquet")
l1 = l1[l1["method"].isin(["B1", "B2", "B8"])][["method", "instance_id", "L1"]]


def load(tag_l2, tag_l3):
    l2 = pd.read_parquet(C.RUNS_DIR / tag_l2)[["method", "instance_id", "L2"]]
    l3 = pd.read_parquet(C.RUNS_DIR / tag_l3)[["method", "instance_id", "L3"]]
    d = l1.merge(l2, on=["method", "instance_id"]).merge(l3, on=["method", "instance_id"])
    d["Final"] = W["L1"] * d["L1"] * 100 + W["L2"] * d["L2"] * 100 + W["L3"] * d["L3"] * 20
    return d


def cliffs_delta(a, b):
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


for name, t2, t3 in [
    ("GLM-5.3 (third judge)", "reports_crossllm_deepseek_n100_B1B8_glm53_l2.parquet",
     "reports_crossllm_deepseek_n100_B1B8_glm53_l3.parquet"),
    ("DeepSeek (shared judge)", "reports_crossllm_deepseek_n100_B1B8_l2.parquet",
     "reports_crossllm_deepseek_n100_B1B8_l3.parquet"),
]:
    d = load(t2, t3)
    wide = d.pivot_table(index="instance_id", columns="method", values="Final")
    diff = wide["B8"] - wide["B2"]
    w = stats.wilcoxon(diff, zero_method="wilcox", method="auto")
    cd = cliffs_delta(wide["B8"].values, wide["B2"].values)
    print(f"{name}: B8 {wide['B8'].mean():.1f} vs B2 {wide['B2'].mean():.1f}, "
          f"mean diff {diff.mean():+.2f}, Wilcoxon p = {w.pvalue:.3e}, Cliff's delta = {cd:+.3f}, "
          f"win rate B8>B2: {(diff > 0).mean():.0%}, ties: {(diff == 0).mean():.0%}")

# Exact two-sided permutation p for Spearman rho at n=8
x = np.arange(1, 9)  # shared-judge ranks are a fixed order
obs = 1 - 24 / 504  # rho = 0.9524 from the analysis, exact value
cnt = 0
tot = 0
for perm in permutations(x):
    r = stats.spearmanr(x, perm).statistic
    if abs(r) >= obs - 1e-12:
        cnt += 1
    tot += 1
print(f"\nSpearman exact two-sided permutation p (n=8, |rho|>={obs:.4f}): {cnt}/{tot} = {cnt / tot:.4e}")
print(f"t-approximation p for comparison: {2 * stats.t.sf(7.65, 6):.2e}")
