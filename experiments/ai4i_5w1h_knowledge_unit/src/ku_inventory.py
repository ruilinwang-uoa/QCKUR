"""Wire the knowledge units into a single inventory {Q, T, M} (Phase 1D, part 1).

Each KU record binds the question (Q), its template (T) and its model handler
(M). The inventory is the manifest the B8 framework consults at runtime.
"""
from __future__ import annotations

import json
from collections import Counter

import config as C
import templates as Tpl

CATEGORY_M_DESC = {
    "What": "classifier / statistics",
    "Why": "causal-reasoning engine (physics rules)",
    "When": "threshold / lifecycle comparator",
    "Where": "feature->component mapping",
    "Who": "fault->responder mapping",
    "How": "fault->action mapping + parameter adjustment",
}


def main() -> None:
    bank = json.loads((C.QUESTION_BANK_DIR / "question_bank_5w1h.json").read_text(encoding="utf-8"))
    kus = []
    for q in bank["questions"]:
        tpl_text = (Tpl.TEMPLATES.get(q["template_id"])
                    or Tpl.DEFAULT_BY_CATEGORY.get(q["category"], ""))
        kus.append({
            "ku_id": q["id"],
            "category": q["category"],
            "question": q["text"],
            "paraphrases": q["paraphrases"],
            "T_template_id": q["template_id"],
            "T_template": tpl_text,
            "M_handler": q["handler"],
            "M_args": q["args"],
            "M_component": CATEGORY_M_DESC[q["category"]],
        })

    counts = Counter(k["category"] for k in kus)
    inventory = {
        "framework": "5W1H question-centered knowledge unit",
        "n_kus": len(kus),
        "category_counts": dict(counts),
        "m_components_used": sorted({k["M_handler"] for k in kus}),
        "n_distinct_templates": len({k["T_template_id"] for k in kus}),
        "kus": kus,
    }
    (C.KNOWLEDGE_DIR / "ku_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"KU inventory: {len(kus)} units | categories {dict(counts)} | "
          f"{inventory['n_distinct_templates']} templates | "
          f"{len(inventory['m_components_used'])} M handlers")


if __name__ == "__main__":
    main()
