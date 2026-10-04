"""Objective fluency/readability proxies for the AI4I canonical reports.

Motivates the qualitative fluency (loveliness) claim without human raters:
  * Flesch Reading Ease and Flesch-Kincaid grade (readability)
  * mean sentence length, type-token ratio (within-report lexical diversity)
  * cross-report 4-gram redundancy (rigid boilerplate vs. varied realization):
    1 - distinct_4grams / total_4grams over the 100-report corpus per system.

Systems: B1, B2, B5, B8 from the canonical sources.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
SRC = {"B5": "reports_main_glm_n100"}
for s in ["B1", "B2", "B8"]:
    SRC[s] = "reports_main_glm_n100_B1B2B8_v3"


def syllables(word: str) -> int:
    w = word.lower().strip(".,;:!?()[]")
    groups = re.findall(r"[aeiouy]+", w)
    if w.endswith("e") and len(groups) > 1:
        groups = groups[:-1]
    return max(1, len(groups))


def sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text)
    return [p for p in parts if len(p.split()) >= 2]


def flesch(text: str) -> tuple[float, float]:
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return 0.0, 0.0
    syl = sum(syllables(w) for w in words)
    S = max(1, len(sentences(text)))
    W = len(words)
    fre = 206.835 - 1.015 * (W / S) - 84.6 * (syl / W)
    fk = 0.39 * (W / S) + 11.8 * (syl / W) - 15.59
    return fre, fk


def ngrams(text: str, n: int = 4) -> list[tuple]:
    ws = re.findall(r"[a-z]+", text.lower())
    return list(zip(*[ws[i:] for i in range(n)]))


def main() -> None:
    rows = []
    for s in ["B1", "B2", "B5", "B8"]:
        d = pd.read_parquet(RUNS / f"{SRC[s]}.parquet")
        texts = [t for t in d[d["method"] == s]["raw_output"]
                 if isinstance(t, str) and t.strip()]
        fre = [flesch(t)[0] for t in texts]
        fk = [flesch(t)[1] for t in texts]
        sl = [len(re.findall(r"\S+", t)) / max(1, len(sentences(t)))
              for t in texts]
        ttr = [len(set(re.findall(r"[a-z]+", t.lower())))
               / max(1, len(re.findall(r"[a-z]+", t.lower()))) for t in texts]
        all_4g, per_report = [], []
        for t in texts:
            g = ngrams(t)
            per_report.append(len(set(g)) / max(1, len(g)))
            all_4g += g
        redundancy = 1 - len(set(all_4g)) / max(1, len(all_4g))
        rows.append({
            "system": s,
            "FleschEase": round(pd.Series(fre).mean(), 1),
            "FKgrade": round(pd.Series(fk).mean(), 1),
            "sent_len": round(pd.Series(sl).mean(), 1),
            "TTR": round(pd.Series(ttr).mean(), 3),
            "within_4gram_div": round(pd.Series(per_report).mean(), 3),
            "cross_report_4gram_redundancy": round(redundancy, 3),
        })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    (ROOT / "analysis" / "fluency_summary.md").write_text(
        "# Objective fluency proxies (canonical n=100)\n\n```\n"
        + t.to_string(index=False) + "\n```\n", encoding="utf-8")
    print("Saved: analysis/fluency_summary.md")


if __name__ == "__main__":
    main()
