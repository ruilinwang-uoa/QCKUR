"""Build programmatic ground truth for every (question, instance) pair (Phase 1C).

Because the paper cannot use human reference text, GT is the deterministic
output of each KU's model M evaluated on the true row, rendered through its
template T. This produces 203 questions x 315 instances reference answers plus
their slot dictionaries, ready for Layer-1 fact-checking and later LLM scoring.
"""
from __future__ import annotations

import json

import pandas as pd

import config as C
import ku_handlers as kh
import templates as T


def main() -> None:
    bank = json.loads((C.QUESTION_BANK_DIR / "question_bank_5w1h.json").read_text(encoding="utf-8"))
    questions = bank["questions"]
    sample = pd.read_parquet(C.DATA_DIR / "sample_315.parquet")
    full = pd.read_parquet(C.DATA_DIR / "ai4i_full_derived.parquet")
    ctx = kh.build_ctx(sample, full)

    rows = []
    rows_by_cat = {c: 0 for c in C.W5H_CATEGORIES}
    for _, srow in sample.iterrows():
        row = srow.to_dict()
        for q in questions:
            try:
                slots = kh.run_handler(q["handler"], row, ctx, q["args"])
                text = T.render(q["template_id"], q["category"], slots)
            except Exception as e:  # noqa: BLE001
                slots, text = {"error": str(e)}, ""
            rows.append({
                "question_id": q["id"], "category": q["category"],
                "handler": q["handler"], "sample_id": row["sample_id"],
                "udis": int(row["UDI"]), "slots": json.dumps(slots, ensure_ascii=False),
                "gt_text": text,
            })
            rows_by_cat[q["category"]] += 1

    gt = pd.DataFrame(rows)
    gt.to_parquet(C.GROUND_TRUTH_DIR / "gt.parquet", index=False)

    # per-category GT coverage summary + a few examples
    summary = {
        "n_questions": len(questions),
        "n_instances": len(sample),
        "n_gt_records": len(gt),
        "gt_records_per_category": rows_by_cat,
        "n_empty_or_error_gt": int(((gt["gt_text"] == "") | (gt["slots"].str.contains('"error"'))).sum()),
    }
    (C.GROUND_TRUTH_DIR / "gt_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    examples = (gt.groupby("category", group_keys=False)
                .apply(lambda d: d.head(2)[["question_id", "sample_id", "gt_text"]])
                .reset_index(drop=True))
    examples.to_csv(C.GROUND_TRUTH_DIR / "gt_examples.csv", index=False)

    print(f"GT built: {summary['n_gt_records']} records "
          f"({len(questions)} Q x {len(sample)} instances); "
          f"empty/error: {summary['n_empty_or_error_gt']}")
    print("Per-category:", rows_by_cat)


if __name__ == "__main__":
    main()
